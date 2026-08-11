"""Measure the engine against controls on identical labelled streams."""
from __future__ import annotations

from dataclasses import dataclass

from .engine import AdaptiveEngine, Decision, Transaction
from .simulation import LabelledTransaction, generate_stream
from .thresholds import StaticThresholds, ThresholdTuner


@dataclass
class Scorecard:
    name: str
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    reviewed: int = 0

    def record(self, flagged: bool, is_fraud: bool) -> None:
        if flagged and is_fraud:
            self.true_positives += 1
        elif flagged:
            self.false_positives += 1
        elif is_fraud:
            self.false_negatives += 1
        else:
            self.true_negatives += 1

    @property
    def total(self) -> int:
        return self.true_positives + self.false_positives + self.true_negatives + self.false_negatives

    @property
    def precision(self) -> float:
        flagged = self.true_positives + self.false_positives
        return self.true_positives / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        legit = self.false_positives + self.true_negatives
        return self.false_positives / legit if legit else 0.0

    def row(self) -> str:
        return (
            f"{self.name:<26} {self.precision:>9.1%} {self.recall:>9.1%} "
            f"{self.f1:>8.3f} {self.false_positive_rate:>9.2%} {self.false_positives:>8}"
        )

    def merge(self, other: "Scorecard") -> "Scorecard":
        """Pool counts across runs."""
        return Scorecard(
            name=self.name,
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            true_negatives=self.true_negatives + other.true_negatives,
            false_negatives=self.false_negatives + other.false_negatives,
            reviewed=self.reviewed + other.reviewed,
        )


class GlobalRuleBaseline:
    """The naive control: one absolute amount limit for everybody."""

    def __init__(self, limit: float = 3_000.0) -> None:
        self.limit = limit

    def flags(self, tx: Transaction) -> bool:
        return tx.amount >= self.limit


def run_adaptive(stream: list[LabelledTransaction], adaptive_thresholds: bool, name: str) -> Scorecard:
    """Replay the stream, with a simulated analyst closing the feedback loop."""
    tuner = ThresholdTuner() if adaptive_thresholds else StaticThresholds()
    engine = AdaptiveEngine(tuner=tuner)
    card = Scorecard(name)

    for item in stream:
        handle, assessment = engine.submit(item.transaction)
        flagged = assessment.decision is not Decision.APPROVE
        card.record(flagged, item.is_fraud)

        if assessment.decision is Decision.REVIEW:
            card.reviewed += 1
            engine.resolve(handle, approved=not item.is_fraud)
        elif assessment.decision is Decision.REJECT:
            engine.resolve(handle, approved=False if item.is_fraud else True)
        elif item.is_fraud:
            engine.report_missed_fraud(assessment.score)

    return card


def run_global_rule(stream: list[LabelledTransaction], limit: float = 3_000.0) -> Scorecard:
    rule = GlobalRuleBaseline(limit)
    card = Scorecard(f"Global rule (>{limit:,.0f})")
    for item in stream:
        card.record(rule.flags(item.transaction), item.is_fraud)
    return card


DEFAULT_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)


def evaluate(seed: int = 20260810, customers: int = 60, days: int = 120) -> dict:
    stream = generate_stream(seed=seed, customers=customers, days=days)

    return {
        "stream": stream,
        "global_rule": run_global_rule(stream),
        "learned_static": run_adaptive(stream, adaptive_thresholds=False,
                                       name="Learned + fixed cutoffs"),
        "learned_adaptive": run_adaptive(stream, adaptive_thresholds=True,
                                         name="Learned + adaptive cutoffs"),
    }


def evaluate_pooled(seeds=DEFAULT_SEEDS, customers: int = 60, days: int = 120) -> dict:
    keys = ("global_rule", "learned_static", "learned_adaptive")
    pooled: dict[str, Scorecard] = {}
    transactions = 0
    fraud = 0

    for seed in seeds:
        run = evaluate(seed, customers, days)
        transactions += len(run["stream"])
        fraud += sum(1 for i in run["stream"] if i.is_fraud)
        for key in keys:
            pooled[key] = run[key] if key not in pooled else pooled[key].merge(run[key])

    pooled["transactions"] = transactions
    pooled["fraud"] = fraud
    pooled["seeds"] = len(seeds)
    return pooled


def report(seeds=DEFAULT_SEEDS, customers: int = 60, days: int = 120) -> str:
    results = evaluate_pooled(seeds, customers, days)
    total = results["transactions"]
    fraud = results["fraud"]

    lines = [
        "Adaptive fraud engine - evaluation",
        "=" * 74,
        f"{results['seeds']} independent simulations pooled",
        f"{total:,} transactions from {customers} customers over {days} days each",
        f"{fraud:,} fraudulent ({fraud / total:.2%}), across takeover, "
        "card-testing and escalation episodes",
        "",
        f"{'approach':<26} {'precision':>9} {'recall':>9} {'F1':>8} {'FP rate':>9} {'FPs':>8}",
        "-" * 74,
    ]

    for key in ("global_rule", "learned_static", "learned_adaptive"):
        lines.append(results[key].row())

    naive = results["global_rule"]
    best = results["learned_adaptive"]
    static = results["learned_static"]

    lines += [
        "",
        "Against the one-size-fits-all rule:",
        f"  recall      {naive.recall:.1%} -> {best.recall:.1%}",
        f"  precision   {naive.precision:.1%} -> {best.precision:.1%}",
        f"  false alarms {naive.false_positives:,} -> {best.false_positives:,}"
        f"  ({_change(naive.false_positives, best.false_positives)})",
        "",
        "Adaptive vs fixed cutoffs (both on the learned baseline):",
        f"  false alarms {static.false_positives:,} -> {best.false_positives:,}"
        f"  ({_change(static.false_positives, best.false_positives)})",
        f"  recall      {static.recall:.1%} -> {best.recall:.1%}",
        f"  transactions sent to an analyst: {static.reviewed:,} -> {best.reviewed:,}",
    ]
    return "\n".join(lines)


def _change(before: int, after: int) -> str:
    if before == 0:
        return "n/a"
    delta = (after - before) / before
    return f"{delta:+.0%}"


if __name__ == "__main__":
    print(report())
