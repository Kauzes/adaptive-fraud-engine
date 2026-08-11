"""Online estimators."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

HOURS_IN_DAY = 24


@dataclass
class RunningStats:
    """Mean and variance via Welford's algorithm."""

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
        if self.count < 2:
            return 0.0
        return max(self.m2 / (self.count - 1), 0.0)

    @property
    def stdev(self) -> float:
        return math.sqrt(self.variance)


@dataclass
class EwmaStats:
    """Exponentially weighted mean and variance."""

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
        self.variance = (1 - self.alpha) * (self.variance + self.alpha * delta * delta)

    @property
    def stdev(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass
class CircularHourProfile:
    """When this customer is normally active."""

    prior: float = 0.5
    decay: float = 0.999
    kernel: tuple[float, ...] = (0.25, 1.0, 0.25)
    counts: list[float] = field(default_factory=lambda: [0.0] * HOURS_IN_DAY)

    def update(self, hour: int) -> None:
        hour %= HOURS_IN_DAY
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
    """Smoothed counts over an open-ended set of labels such as devices or countries."""

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
