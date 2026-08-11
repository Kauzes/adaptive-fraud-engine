"""The adaptive engine: scores in bits of surprise, decides, and learns."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .population import PopulationPrior
from .profile import CustomerProfile
from .thresholds import ThresholdTuner

VELOCITY_WINDOW = timedelta(minutes=10)
VELOCITY_TRIGGER = 2
TRAVEL_WINDOW = timedelta(minutes=5)


class Decision(str, Enum):
    APPROVE = "approve"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True)
class Transaction:
    customer_id: str
    amount: float
    timestamp: datetime
    device_id: str
    country: str
    merchant: str = "unknown"


@dataclass(frozen=True)
class Signal:
    code: str
    bits: float
    reason: str


@dataclass
class Assessment:
    transaction: Transaction
    score: float
    decision: Decision
    signals: list[Signal]
    confidence: float
    review_at: float
    reject_at: float

    @property
    def reasons(self) -> list[str]:
        return [s.reason for s in self.signals]


class AdaptiveEngine:
    """Scores transactions against a baseline that is learned, never configured."""

    def __init__(self, tuner: ThresholdTuner | None = None) -> None:
        self.prior = PopulationPrior()
        self.profiles: dict[str, CustomerProfile] = {}
        self.tuner = tuner or ThresholdTuner()
        self._pending: dict[int, tuple[Transaction, Assessment]] = {}
        self._next_id = 1


    def profile_for(self, customer_id: str) -> CustomerProfile:
        if customer_id not in self.profiles:
            self.profiles[customer_id] = CustomerProfile(customer_id)
        return self.profiles[customer_id]

    def assess(self, tx: Transaction) -> Assessment:
        profile = self.profile_for(tx.customer_id)
        signals: list[Signal] = []

        signals.extend(self._amount_signal(tx, profile))
        signals.extend(self._hour_signal(tx, profile))
        signals.extend(self._device_signal(tx, profile))
        signals.extend(self._country_signal(tx, profile))
        signals.extend(self._velocity_signals(tx, profile))

        score = sum(s.bits for s in signals)
        review_at, reject_at = self.tuner.thresholds()

        if score >= reject_at:
            decision = Decision.REJECT
        elif score >= review_at:
            decision = Decision.REVIEW
        else:
            decision = Decision.APPROVE

        return Assessment(
            transaction=tx,
            score=round(score, 2),
            decision=decision,
            signals=signals,
            confidence=round(profile.confidence(), 3),
            review_at=review_at,
            reject_at=reject_at,
        )

    def _amount_signal(self, tx: Transaction, profile: CustomerProfile) -> list[Signal]:
        z = profile.amount_z(tx.amount, self.prior)
        if z <= 1.5:
            return []

        bits = (z * z - 1.5 * 1.5) / (2 * math.log(2))
        source = "their own history" if profile.confidence() > 0.5 else "the population baseline"
        return [Signal(
            code="AMOUNT",
            bits=bits,
            reason=f"{tx.amount:,.0f} is {z:.1f} sd above what {source} predicts",
        )]

    def _hour_signal(self, tx: Transaction, profile: CustomerProfile) -> list[Signal]:
        if profile.transactions_learned < 3:
            return []

        bits = profile.hours.surprisal(tx.timestamp.hour) - math.log2(6)
        if bits <= 0:
            return []
        return [Signal(
            code="HOUR",
            bits=bits,
            reason=f"{tx.timestamp:%H:%M} is outside this customer's usual hours",
        )]

    def _device_signal(self, tx: Transaction, profile: CustomerProfile) -> list[Signal]:
        if profile.devices.is_known(tx.device_id):
            return []
        bits = 2.0 * profile.confidence()
        if bits < 0.2:
            return []
        return [Signal(
            code="DEVICE",
            bits=bits,
            reason=f"device {tx.device_id} has never been seen for this customer",
        )]

    def _country_signal(self, tx: Transaction, profile: CustomerProfile) -> list[Signal]:
        if profile.countries.is_known(tx.country):
            return []
        bits = 2.5 * profile.confidence()
        if bits < 0.2:
            return []
        return [Signal(
            code="COUNTRY",
            bits=bits,
            reason=f"first transaction from {tx.country}",
        )]

    def _velocity_signals(self, tx: Transaction, profile: CustomerProfile) -> list[Signal]:
        signals: list[Signal] = []
        recent = [(t, c) for t, c in profile.recent if abs(tx.timestamp - t) <= VELOCITY_WINDOW]

        if len(recent) >= VELOCITY_TRIGGER:
            signals.append(Signal(
                code="VELOCITY",
                bits=1.5 * len(recent),
                reason=f"{len(recent)} attempts in the preceding 10 minutes",
            ))

        jumped = [
            c for t, c in profile.recent
            if c != tx.country and abs(tx.timestamp - t) <= TRAVEL_WINDOW
        ]
        if jumped:
            signals.append(Signal(
                code="TRAVEL",
                bits=5.0,
                reason=f"country changed from {jumped[-1]} to {tx.country} within 5 minutes",
            ))

        return signals


    def submit(self, tx: Transaction) -> tuple[int, Assessment]:
        """Score a transaction and record it. Returns a handle for later resolution."""
        assessment = self.assess(tx)
        profile = self.profile_for(tx.customer_id)

        handle = self._next_id
        self._next_id += 1

        profile.note_attempt(tx.timestamp, tx.country)

        if assessment.decision is Decision.APPROVE:
            self._absorb(tx, profile)
        else:
            self._pending[handle] = (tx, assessment)

        return handle, assessment

    def resolve(self, handle: int, approved: bool) -> None:
        """An analyst's ruling on a flagged transaction."""
        entry = self._pending.pop(handle, None)
        if entry is None:
            return
        tx, assessment = entry

        if approved:
            self.tuner.record_false_positive(assessment.score)
            self._absorb(tx, self.profile_for(tx.customer_id), trusted=True)
        else:
            self.tuner.record_true_positive(assessment.score)

    def report_missed_fraud(self, score: float) -> None:
        """Fraud that was auto-approved and only surfaced later, e.g. by a chargeback."""
        self.tuner.record_false_negative(score)

    def _absorb(self, tx: Transaction, profile: CustomerProfile, trusted: bool = False) -> None:
        profile.learn(tx.amount, tx.timestamp, tx.device_id, tx.country, self.prior, trusted)
        self.prior.observe(tx.customer_id, tx.amount, tx.timestamp.hour)
