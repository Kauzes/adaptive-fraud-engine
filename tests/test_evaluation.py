import unittest

from adaptive_fraud.evaluate import Scorecard, evaluate, evaluate_pooled, run_global_rule
from adaptive_fraud.simulation import generate_stream

# Small enough to keep the suite fast, large enough for the comparisons to mean something.
SMALL = dict(customers=24, days=60)


class SimulationTests(unittest.TestCase):
    def test_stream_is_deterministic_for_a_seed(self):
        a = generate_stream(seed=7, **SMALL)
        b = generate_stream(seed=7, **SMALL)

        self.assertEqual(len(a), len(b))
        self.assertEqual(
            [i.transaction.amount for i in a[:50]],
            [i.transaction.amount for i in b[:50]],
        )

    def test_different_seeds_produce_different_streams(self):
        a = generate_stream(seed=7, **SMALL)
        b = generate_stream(seed=8, **SMALL)
        self.assertNotEqual(
            [i.transaction.amount for i in a[:50]],
            [i.transaction.amount for i in b[:50]],
        )

    def test_stream_is_time_ordered(self):
        stream = generate_stream(seed=7, **SMALL)
        stamps = [i.transaction.timestamp for i in stream]
        self.assertEqual(stamps, sorted(stamps))

    def test_contains_both_classes_and_all_fraud_episodes(self):
        stream = generate_stream(seed=7, customers=80, days=90)
        episodes = {i.episode for i in stream if i.is_fraud}

        self.assertTrue(any(i.is_fraud for i in stream))
        self.assertTrue(any(not i.is_fraud for i in stream))
        self.assertEqual(episodes, {"takeover", "card_testing", "escalation"})

    def test_fraud_is_rare_as_it_is_in_reality(self):
        stream = generate_stream(seed=7, customers=80, days=90)
        rate = sum(1 for i in stream if i.is_fraud) / len(stream)
        self.assertLess(rate, 0.05)
        self.assertGreater(rate, 0.0)

    def test_fraud_never_lands_on_a_customer_with_no_history(self):
        stream = generate_stream(seed=7, customers=80, days=90)
        first_legit: dict[str, object] = {}
        for item in stream:
            cid = item.transaction.customer_id
            if not item.is_fraud:
                first_legit.setdefault(cid, item.transaction.timestamp)
            else:
                # Scoring an attack against an empty profile would measure the
                # cold-start policy rather than the detector.
                self.assertIn(cid, first_legit)


class ScorecardTests(unittest.TestCase):
    def test_metric_arithmetic(self):
        card = Scorecard("t", true_positives=8, false_positives=2,
                         true_negatives=85, false_negatives=5)

        self.assertAlmostEqual(card.precision, 0.8)
        self.assertAlmostEqual(card.recall, 8 / 13)
        self.assertAlmostEqual(card.false_positive_rate, 2 / 87)
        self.assertAlmostEqual(card.f1, 2 * 0.8 * (8 / 13) / (0.8 + 8 / 13))

    def test_empty_scorecard_does_not_divide_by_zero(self):
        card = Scorecard("empty")
        self.assertEqual(card.precision, 0.0)
        self.assertEqual(card.recall, 0.0)
        self.assertEqual(card.f1, 0.0)

    def test_merge_pools_raw_counts(self):
        a = Scorecard("x", true_positives=3, false_positives=1, true_negatives=10, false_negatives=2)
        b = Scorecard("x", true_positives=5, false_positives=4, true_negatives=20, false_negatives=1)
        merged = a.merge(b)

        self.assertEqual(merged.true_positives, 8)
        self.assertEqual(merged.false_positives, 5)
        self.assertEqual(merged.true_negatives, 30)
        self.assertEqual(merged.false_negatives, 3)


class ComparisonTests(unittest.TestCase):
    """Regression tests on the claims the README makes.

    These pool several seeds, because that is the form the claims take. Fraud is rare
    enough that any single simulation swings by tens of percentage points of recall -
    see `test_single_seeds_vary_enough_to_mislead` below, which exists to stop anyone
    (including me) quoting a one-seed number as though it were stable.
    """

    @classmethod
    def setUpClass(cls):
        cls.pooled = evaluate_pooled(seeds=(11, 12, 13, 14, 15, 16), **SMALL)
        cls.results = evaluate(seed=11, **SMALL)

    def test_learned_baseline_massively_outperforms_a_global_limit(self):
        naive = self.pooled["global_rule"]
        learned = self.pooled["learned_adaptive"]

        self.assertGreater(learned.precision, naive.precision * 10)
        self.assertLess(learned.false_positives, naive.false_positives / 10)
        self.assertGreater(learned.recall, naive.recall)

    def test_adaptive_cutoffs_beat_fixed_ones_on_recall_when_pooled(self):
        static = self.pooled["learned_static"]
        adaptive = self.pooled["learned_adaptive"]
        self.assertGreater(adaptive.recall, static.recall)

    def test_single_seeds_vary_enough_to_mislead(self):
        recalls = [
            evaluate(seed=s, **SMALL)["learned_adaptive"].recall
            for s in (11, 12, 13, 14)
        ]
        # Guards the honesty of the README: if this spread ever collapses the caveat
        # about per-seed variance can go, but while it holds the caveat must stay.
        self.assertGreater(max(recalls) - min(recalls), 0.1)

    def test_a_global_limit_drowns_analysts_in_false_alarms(self):
        naive = run_global_rule(generate_stream(seed=11, **SMALL))
        # Flagging every large payment means flagging every business customer, constantly.
        self.assertGreater(naive.false_positive_rate, 0.05)

    def test_every_transaction_is_accounted_for_exactly_once(self):
        stream_len = len(generate_stream(seed=11, **SMALL))
        for key in ("global_rule", "learned_static", "learned_adaptive"):
            self.assertEqual(self.results[key].total, stream_len)


if __name__ == "__main__":
    unittest.main()
