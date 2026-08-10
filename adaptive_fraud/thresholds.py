"""Decision thresholds that tune themselves from analyst rulings."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ThresholdTuner:
    """Moves the review and reject boundaries in response to human decisions.

    This is a proportional controller, not a learned model, and that is deliberate: the
    thresholds stay two inspectable numbers with an audit trail of why they moved. The
    signal comes from the analyst queue, which every fraud team already staffs — so the
    labels are free rather than requiring an annotation project.

    - analyst approves a flagged transaction -> false positive -> raise the bar
    - analyst rejects one                    -> true positive  -> hold, nudge down
    - fraud surfaces after auto-approval     -> false negative -> lower the bar sharply
    """

    review_at: float = 6.0
    reject_at: float = 14.0

    rate: float = 0.35
    min_review: float = 2.0
    max_review: float = 20.0
    min_gap: float = 4.0
    max_reject: float = 40.0

    false_positives: int = 0
    true_positives: int = 0
    false_negatives: int = 0
    history: list[tuple[str, float, float]] = field(default_factory=list)

    def thresholds(self) -> tuple[float, float]:
        return round(self.review_at, 2), round(self.reject_at, 2)

    def record_false_positive(self, score: float) -> None:
        """Legitimate traffic got flagged: become less twitchy."""
        self.false_positives += 1
        # Move just past the score that caused the noise, damped by the learning rate.
        target = max(score, self.review_at)
        self.review_at += self.rate * (target - self.review_at) + self.rate
        self._clamp("false_positive")

    def record_true_positive(self, score: float) -> None:
        """A flag was correct: the boundary is roughly right, creep toward the score."""
        self.true_positives += 1
        if score < self.review_at:
            return
        self.review_at -= self.rate * 0.25
        self._clamp("true_positive")

    def record_false_negative(self, score: float) -> None:
        """Fraud slipped through: tighten decisively.

        Weighted harder than a false positive because the costs are not symmetric — an
        annoyed customer is cheaper than a drained account.
        """
        self.false_negatives += 1
        target = min(score, self.review_at)
        self.review_at -= self.rate * 1.5 * max(self.review_at - target, 1.0)
        self._clamp("false_negative")

    def _clamp(self, cause: str) -> None:
        self.review_at = min(max(self.review_at, self.min_review), self.max_review)
        # Reject must stay meaningfully above review or the review band vanishes and
        # every flagged transaction is auto-refused.
        self.reject_at = min(max(self.reject_at, self.review_at + self.min_gap), self.max_reject)
        self.history.append((cause, round(self.review_at, 2), round(self.reject_at, 2)))

    @property
    def decisions_seen(self) -> int:
        return self.false_positives + self.true_positives + self.false_negatives


@dataclass
class StaticThresholds(ThresholdTuner):
    """Control group: identical interface, but never moves.

    Used by the evaluation to answer "does the adaptation actually help, or does the
    learned baseline alone explain the numbers?"
    """

    def record_false_positive(self, score: float) -> None:
        self.false_positives += 1

    def record_true_positive(self, score: float) -> None:
        self.true_positives += 1

    def record_false_negative(self, score: float) -> None:
        self.false_negatives += 1
