"""Per-customer state that starts blank and learns."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from .estimators import CategoryProfile, CircularHourProfile, EwmaStats
from .population import PopulationPrior

# Observations needed before a customer's own history outweighs the population prior.
# w = n/(n+K), so at n=K the estimate is a 50/50 blend.
SHRINKAGE_K = 12.0

# No single transaction may move the baseline by more than this many standard
# deviations. Without it, a patient fraudster can walk the baseline upward until
# large transfers look ordinary.
UPDATE_CLIP_Z = 2.5


@dataclass
class CustomerProfile:
    """Everything the engine has learned about one customer.

    Amounts are modelled in log space: spending is roughly log-normal, so a customer who
    usually pays 50 and occasionally 500 is unremarkable on a log scale but a wild
    outlier on a linear one.
    """

    customer_id: str
    log_amount: EwmaStats = field(default_factory=lambda: EwmaStats(alpha=0.08))
    hours: CircularHourProfile = field(default_factory=CircularHourProfile)
    devices: CategoryProfile = field(default_factory=CategoryProfile)
    countries: CategoryProfile = field(default_factory=CategoryProfile)
    transactions_learned: int = 0
    recent: list[tuple[datetime, str]] = field(default_factory=list)

    # ---------------------------------------------------------------- shrinkage

    def confidence(self) -> float:
        """How far this customer's own history has displaced the population prior (0..1)."""
        n = self.transactions_learned
        return n / (n + SHRINKAGE_K)

    def expected_log_amount(self, prior: PopulationPrior) -> float:
        w = self.confidence()
        personal = self.log_amount.mean if self.transactions_learned else 0.0
        return w * personal + (1 - w) * prior.mean_log_amount

    def expected_log_spread(self, prior: PopulationPrior) -> float:
        w = self.confidence()
        personal = self.log_amount.stdev if self.transactions_learned >= 2 else 0.0
        blended = w * personal + (1 - w) * prior.stdev_log_amount
        # A customer with genuinely identical payments would otherwise get a zero spread,
        # making every rounding difference infinitely surprising.
        return max(blended, 0.15)

    def amount_z(self, amount: float, prior: PopulationPrior) -> float:
        centre = self.expected_log_amount(prior)
        spread = self.expected_log_spread(prior)
        return (math.log1p(max(amount, 0.0)) - centre) / spread

    # ---------------------------------------------------------------- learning

    def learn(self, amount: float, timestamp: datetime, device: str, country: str,
              prior: PopulationPrior, trusted: bool = False) -> None:
        """Fold an approved transaction into the baseline.

        Only ever called for transactions that were approved and not disputed. Learning
        from unresolved traffic is what lets fraud teach the model that fraud is normal.

        `trusted` means a human analyst explicitly cleared this one. The clip exists to
        stop an unattended baseline being walked upward by a patient fraudster; once a
        person has vouched for the transaction that threat model no longer applies, and
        clipping only gets in the way of absorbing a real change like a pay rise.
        """
        clipped = amount if trusted else self._clip(amount, prior)

        self.log_amount.update(math.log1p(max(clipped, 0.0)))
        self.hours.update(timestamp.hour)
        self.devices.update(device)
        self.countries.update(country)
        self.transactions_learned += 1

        self.recent.append((timestamp, country))
        if len(self.recent) > 25:
            self.recent = self.recent[-25:]

    def _clip(self, amount: float, prior: PopulationPrior) -> float:
        """Winsorise an amount before it updates the baseline."""
        if self.transactions_learned < 2:
            return amount

        z = self.amount_z(amount, prior)
        if abs(z) <= UPDATE_CLIP_Z:
            return amount

        bounded = self.expected_log_amount(prior) + math.copysign(
            UPDATE_CLIP_Z * self.expected_log_spread(prior), z
        )
        return math.expm1(bounded)

    def note_attempt(self, timestamp: datetime, country: str) -> None:
        """Record an attempt for velocity purposes, whatever the engine decides.

        Velocity and impossible travel are about attempts, not approvals — a burst of
        refused attempts is itself the signal.
        """
        self.recent.append((timestamp, country))
        if len(self.recent) > 25:
            self.recent = self.recent[-25:]
