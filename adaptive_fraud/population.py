"""The population prior a blank customer borrows from until they have a history."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .estimators import CircularHourProfile, RunningStats


@dataclass
class PopulationPrior:
    """What "normal" looks like across every customer the system has seen.

    A brand-new customer has no baseline of their own, and the window before one exists
    is exactly when account-takeover fraud is easiest. Rather than leaving them
    unprotected or inventing a preset, they start from the population and are pulled
    toward their own behaviour as evidence accumulates.
    """

    log_amount: RunningStats = field(default_factory=RunningStats)
    hours: CircularHourProfile = field(default_factory=CircularHourProfile)
    customers_seen: set[str] = field(default_factory=set)

    def observe(self, customer_id: str, amount: float, hour: int) -> None:
        self.customers_seen.add(customer_id)
        self.log_amount.update(math.log1p(max(amount, 0.0)))
        self.hours.update(hour)

    @property
    def observations(self) -> int:
        return self.log_amount.count

    @property
    def mean_log_amount(self) -> float:
        # Before anything is known, assume a mid-sized payment rather than zero, or the
        # very first transaction in the system looks infinitely surprising.
        return self.log_amount.mean if self.observations else math.log1p(250.0)

    @property
    def stdev_log_amount(self) -> float:
        # A wide default keeps the engine permissive while it is ignorant.
        if self.observations < 2 or self.log_amount.stdev <= 0:
            return 1.2
        return self.log_amount.stdev

    def is_cold(self) -> bool:
        """True while the prior itself is too thin to lean on."""
        return self.observations < 30
