"""Command line entry point: `python -m adaptive_fraud`."""
from __future__ import annotations

import argparse
import sys
from datetime import timedelta

from .engine import AdaptiveEngine, Decision, Transaction
from .evaluate import DEFAULT_SEEDS, report
from .simulation import START, ARCHETYPES, SimulatedCustomer
import random


def cmd_evaluate(args: argparse.Namespace) -> int:
    print(report(seeds=tuple(args.seeds), customers=args.customers, days=args.days))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Follow one customer from blank state and watch the baseline form."""
    archetype = next(a for a in ARCHETYPES if a.name == args.archetype)
    person = SimulatedCustomer(
        customer_id="watched",
        archetype=archetype,
        device="phone-1",
        rng=random.Random(args.seed),
    )
    engine = AdaptiveEngine()

    print(f"Watching a '{archetype.name}' customer who starts completely blank.")
    print(f"Their true median payment is {archetype.median_amount:,.0f}, "
          "which the engine is never told.\n")
    print(f"{'txn':>4} {'amount':>10} {'expected':>10} {'conf':>6} {'score':>7}  decision")
    print("-" * 62)

    probe_amount = archetype.median_amount * 12

    for i in range(1, args.transactions + 1):
        item = person.legit_transaction(START + timedelta(days=i))
        expected = _expected(engine, "watched")
        _, assessment = engine.submit(item.transaction)

        if i <= 5 or i % args.every == 0:
            print(f"{i:>4} {item.transaction.amount:>10,.0f} {expected:>10,.0f} "
                  f"{assessment.confidence:>6.2f} {assessment.score:>7.2f}  "
                  f"{assessment.decision.value}")

    # The same suspicious payment, judged against three different states of knowledge.
    when = (START + timedelta(days=args.transactions + 1)).replace(hour=3, minute=0)
    probe = Transaction(
        customer_id="watched",
        amount=probe_amount,
        timestamp=when,
        device_id="unknown-device",
        country="RU",
        merchant="Transfer",
    )

    print(f"\nProbe: {probe.amount:,.0f} at 03:00 from an unknown device in RU")
    verdict = engine.assess(probe)
    print(f"\n  against this customer's learned profile -> {verdict.decision.value.upper()} "
          f"(score {verdict.score}, thresholds {verdict.review_at}/{verdict.reject_at})")
    for reason in verdict.reasons:
        print(f"      - {reason}")

    # A blank customer inside a bank that already has other customers: the realistic
    # cold-start case, and the one the population prior exists for.
    populated = AdaptiveEngine()
    for i in range(20):
        other = SimulatedCustomer(f"other-{i}", ARCHETYPES[i % len(ARCHETYPES)],
                                  f"dev-{i}", random.Random(1000 + i))
        for day in range(10):
            populated.submit(other.legit_transaction(START + timedelta(days=day)).transaction)

    cold = populated.assess(
        Transaction("brand-new", probe.amount, when, "unknown-device", "RU", "Transfer")
    )
    print(f"\n  against a brand-new customer, borrowing the population prior -> "
          f"{cold.decision.value.upper()} (score {cold.score})")
    for reason in cold.reasons:
        print(f"      - {reason}")

    # And with nothing known at all.
    empty = AdaptiveEngine().assess(probe)
    print(f"\n  against an empty system that has never seen anyone -> "
          f"{empty.decision.value.upper()} (score {empty.score})")
    if empty.decision is Decision.APPROVE:
        print("      - nothing known, nothing flagged: the genuine cold-start blind spot")
    else:
        print("      - flagged only by the hard-coded fallback (a generic median and a wide")
        print("        spread), not by anything learned. A weak guess, not an informed one.")
    return 0


def _expected(engine: AdaptiveEngine, customer_id: str) -> float:
    import math
    profile = engine.profile_for(customer_id)
    return math.expm1(profile.expected_log_amount(engine.prior))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adaptive_fraud",
        description="A fraud engine with no presets: every customer starts blank.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("evaluate", help="measure against controls on labelled data")
    ev.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    ev.add_argument("--customers", type=int, default=60)
    ev.add_argument("--days", type=int, default=120)
    ev.set_defaults(func=cmd_evaluate)

    w = sub.add_parser("watch", help="watch one customer's baseline form from scratch")
    w.add_argument("--archetype", choices=[a.name for a in ARCHETYPES], default="student")
    w.add_argument("--transactions", type=int, default=60)
    w.add_argument("--every", type=int, default=10)
    w.add_argument("--seed", type=int, default=3)
    w.set_defaults(func=cmd_watch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
