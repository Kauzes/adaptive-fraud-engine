"""Online estimators.

Everything here updates in O(1) time and O(1) memory per observation. A fraud engine
sees an unbounded stream of transactions per customer, so nothing may require keeping
the history around or recomputing from scratch.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

HOURS_IN_DAY = 24


@dataclass
class RunningStats:
    """Mean and variance via Welford's algorithm.

    The naive "sum of squares minus square of sum" formula loses catastrophic precision
    when the variance is small relative to the mean — exactly the case for a customer who
    reliably spends about the same amount. Welford is numerically stable and needs only
    three numbers of state.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float, weight: float = 1.0) -> None:
        """Fold in one observation. `weight` below 1 dampens its influence."""
        if weight <= 0:
            return
        self.count += 1
        delta = value - self.mean
        self.mean += (delta * weight) / self.count
        self.m2 += weight * delta * (value - self.mean)

    @property
    def variance(self) -> float:
        # Sample variance needs 2 points; below that we have no spread information.
        if self.count < 2:
            return 0.0
        return max(self.m2 / (self.count - 1), 0.0)

    @property
    def stdev(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class EwmaStats:
    """Exponentially weighted mean and variance.

    Unlike [RunningStats], this forgets. A customer who gets a raise or moves country
    should not be judged for years against who they used to be, so recent observations
    carry more weight and old ones decay away.
    """

    alpha: float = 0.05
    mean: float = 0.0
    variance: float = 0.0
    count: int = 0

    def update(self, value: float) -> None:
        self.count += 1
        if self.count == 1:
            self.mean = value
            self.variance = 0.0
            return

        delta = value - self.mean
        self.mean += self.alpha * delta
        # Incremental EWMA variance (West, 1979).
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta * delta)

    @property
    def stdev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass
class CircularHourProfile:
    """When this customer is normally active.

    Hours are circular: 23:00 and 00:00 are adjacent, so a plain 24-bucket histogram
    treats midnight-crossing habits as two unrelated peaks. Observations are therefore
    smeared over a small circular kernel, which also means a customer seen at 09:00 is
    not considered wholly unfamiliar at 08:00.
    """

    prior: float = 0.5
    decay: float = 0.999
    kernel: tuple[float, ...] = (0.25, 1.0, 0.25)
    counts: list[float] = field(default_factory=lambda: [0.0] * HOURS_IN_DAY)

    def update(self, hour: int) -> None:
        hour %= HOURS_IN_DAY
        # Decay first so old habits fade rather than accumulate forever.
        self.counts = [c * self.decay for c in self.counts]

        span = len(self.kernel) // 2
        for offset, weight in enumerate(self.kernel, start=-span):
            self.counts[(hour + offset) % HOURS_IN_DAY] += weight

    def probability(self, hour: int) -> float:
        """Smoothed probability of activity in this hour. Never zero."""
        hour %= HOURS_IN_DAY
        total = sum(self.counts) + self.prior * HOURS_IN_DAY
        return (self.counts[hour] + self.prior) / total

    def surprisal(self, hour: int) -> float:
        """Bits of surprise at seeing this hour. Uniform activity gives log2(24) ~ 4.58."""
        return -math.log2(self.probability(hour))

    @property
    def observations(self) -> float:
        return sum(self.counts)


@dataclass
class CategoryProfile:
    """Counts over an open-ended set of labels, e.g. devices or countries.

    New labels appear all the time, so probabilities use add-alpha smoothing over an
    assumed vocabulary slightly larger than what has been seen.
    """

    prior: float = 0.5
    decay: float = 0.999
    reserve: int = 3
    counts: dict[str, float] = field(default_factory=dict)

    def update(self, label: str) -> None:
        for key in self.counts:
            self.counts[key] *= self.decay
        self.counts[label] = self.counts.get(label, 0.0) + 1.0

    def probability(self, label: str) -> float:
        vocabulary = len(self.counts) + self.reserve
        total = sum(self.counts.values()) + self.prior * vocabulary
        return (self.counts.get(label, 0.0) + self.prior) / total

    def surprisal(self, label: str) -> float:
        return -math.log2(self.probability(label))

    def is_known(self, label: str) -> bool:
        return self.counts.get(label, 0.0) >= 1.0

    @property
    def observations(self) -> float:
        return sum(self.counts.values())
