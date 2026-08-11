import unittest
from datetime import datetime, timedelta

from adaptive_fraud.engine import AdaptiveEngine, Decision, Transaction
from adaptive_fraud.thresholds import ThresholdTuner

BASE = datetime(2026, 3, 2, 12, 0)


def tx(engine_customer="c1", amount=100.0, at=BASE, device="phone-1", country="TR"):
    return Transaction(
        customer_id=engine_customer,
        amount=amount,
        timestamp=at,
        device_id=device,
        country=country,
        merchant="Shop",
    )


def teach(engine, customer="c1", n=40, amount=100.0, hour=12, device="phone-1", country="TR"):
    """Give a customer a consistent history, one transaction per day."""
    for i in range(n):
        engine.submit(tx(customer, amount, BASE + timedelta(days=i, hours=hour - 12), device, country))


class ColdStartTests(unittest.TestCase):
    def test_first_ever_transaction_is_not_rejected(self):
        engine = AdaptiveEngine()
        _, a = engine.submit(tx(amount=120.0))

        self.assertIs(a.decision, Decision.APPROVE)
        self.assertEqual(a.confidence, 0.0)

    def test_blank_customer_is_still_protected_by_the_population_prior(self):
        engine = AdaptiveEngine()
        for c in range(12):
            teach(engine, customer=f"pop{c}", n=8, amount=100.0)

        _, a = engine.submit(tx("newcomer", amount=90_000.0))

        self.assertNotEqual(a.decision, Decision.APPROVE)
        self.assertTrue(any(s.code == "AMOUNT" for s in a.signals))
        self.assertEqual(a.confidence, 0.0)

    def test_confidence_rises_as_history_accumulates(self):
        engine = AdaptiveEngine()
        start = engine.profile_for("c1").confidence()
        teach(engine, n=40)
        self.assertEqual(start, 0.0)
        self.assertGreater(engine.profile_for("c1").confidence(), 0.75)


class LearningTests(unittest.TestCase):
    def test_engine_learns_what_is_normal_for_this_person(self):
        engine = AdaptiveEngine()
        teach(engine, customer="whale", n=40, amount=5_000.0)
        teach(engine, customer="minnow", n=40, amount=40.0)

        _, whale = engine.submit(tx("whale", amount=5_200.0, at=BASE + timedelta(days=60)))
        _, minnow = engine.submit(tx("minnow", amount=5_200.0, at=BASE + timedelta(days=60)))

        self.assertIs(whale.decision, Decision.APPROVE)
        self.assertNotEqual(minnow.decision, Decision.APPROVE)

    def test_unfamiliar_device_matters_more_once_the_customer_is_established(self):
        fresh = AdaptiveEngine()
        fresh.submit(tx(amount=100.0))
        _, early = fresh.submit(tx(amount=100.0, device="other", at=BASE + timedelta(days=1)))

        mature = AdaptiveEngine()
        teach(mature, n=40)
        _, late = mature.submit(tx(amount=100.0, device="other", at=BASE + timedelta(days=60)))

        early_bits = sum(s.bits for s in early.signals if s.code == "DEVICE")
        late_bits = sum(s.bits for s in late.signals if s.code == "DEVICE")
        self.assertGreater(late_bits, early_bits)

    def test_a_step_change_stays_flagged_while_nobody_reviews_it(self):
        """A real limitation, asserted rather than glossed over."""
        engine = AdaptiveEngine()
        teach(engine, n=30, amount=100.0)

        for i in range(60):
            engine.submit(tx(amount=800.0, at=BASE + timedelta(days=40 + i)))

        _, a = engine.submit(tx(amount=820.0, at=BASE + timedelta(days=200)))
        self.assertNotEqual(a.decision, Decision.APPROVE)

    def test_baseline_absorbs_a_pay_rise_once_the_analyst_clears_it(self):
        engine = AdaptiveEngine()
        teach(engine, n=30, amount=100.0)

        for i in range(12):
            handle, a = engine.submit(tx(amount=800.0, at=BASE + timedelta(days=40 + i)))
            if a.decision is not Decision.APPROVE:
                engine.resolve(handle, approved=True)

        _, a = engine.submit(tx(amount=820.0, at=BASE + timedelta(days=200)))
        self.assertIs(a.decision, Decision.APPROVE)


class PoisoningTests(unittest.TestCase):
    def test_a_single_huge_transaction_cannot_drag_the_baseline(self):
        engine = AdaptiveEngine()
        teach(engine, n=40, amount=100.0)
        before = engine.profile_for("c1").log_amount.mean

        profile = engine.profile_for("c1")
        profile.learn(500_000.0, BASE + timedelta(days=50), "phone-1", "TR", engine.prior)
        after = profile.log_amount.mean

        self.assertLess(after - before, 0.5)

    def test_flagged_transactions_are_not_learned_until_an_analyst_clears_them(self):
        engine = AdaptiveEngine()
        teach(engine, n=40, amount=100.0)
        learned_before = engine.profile_for("c1").transactions_learned

        handle, a = engine.submit(tx(amount=40_000.0, at=BASE + timedelta(days=50)))
        self.assertNotEqual(a.decision, Decision.APPROVE)
        self.assertEqual(engine.profile_for("c1").transactions_learned, learned_before)

        engine.resolve(handle, approved=True)
        self.assertEqual(engine.profile_for("c1").transactions_learned, learned_before + 1)


class FeedbackTests(unittest.TestCase):
    def test_repeated_false_positives_make_the_engine_less_twitchy(self):
        engine = AdaptiveEngine()
        teach(engine, n=40, amount=100.0)
        start_review, _ = engine.tuner.thresholds()

        for i in range(6):
            handle, a = engine.submit(tx(amount=900.0, at=BASE + timedelta(days=60 + i)))
            if a.decision is not Decision.APPROVE:
                engine.resolve(handle, approved=True)

        end_review, _ = engine.tuner.thresholds()
        self.assertGreater(end_review, start_review)

    def test_missed_fraud_tightens_the_bar(self):
        engine = AdaptiveEngine()
        teach(engine, n=40)
        start_review, _ = engine.tuner.thresholds()

        engine.report_missed_fraud(score=3.0)
        engine.report_missed_fraud(score=3.0)

        end_review, _ = engine.tuner.thresholds()
        self.assertLess(end_review, start_review)

    def test_review_band_never_collapses(self):
        tuner = ThresholdTuner()
        for _ in range(60):
            tuner.record_false_positive(score=100.0)

        review, reject = tuner.thresholds()
        self.assertGreaterEqual(reject - review, tuner.min_gap)
        self.assertLessEqual(review, tuner.max_review)


class VelocityTests(unittest.TestCase):
    def test_rapid_burst_is_flagged(self):
        engine = AdaptiveEngine()
        teach(engine, n=40, amount=100.0)

        t = BASE + timedelta(days=60)
        for i in range(3):
            _, a = engine.submit(tx(amount=100.0, at=t + timedelta(minutes=i)))

        self.assertTrue(any(s.code == "VELOCITY" for s in a.signals))

    def test_country_jump_within_minutes(self):
        engine = AdaptiveEngine()
        teach(engine, n=40, amount=100.0, country="TR")

        t = BASE + timedelta(days=60)
        engine.submit(tx(amount=100.0, at=t, country="TR"))
        _, a = engine.submit(tx(amount=100.0, at=t + timedelta(minutes=2), country="BR"))

        self.assertTrue(any(s.code == "TRAVEL" for s in a.signals))


class ExplainabilityTests(unittest.TestCase):
    def test_every_flagged_transaction_states_why(self):
        engine = AdaptiveEngine()
        teach(engine, n=40, amount=100.0)
        _, a = engine.submit(
            tx(amount=60_000.0, at=BASE + timedelta(days=60), device="new", country="RU")
        )

        self.assertNotEqual(a.decision, Decision.APPROVE)
        self.assertTrue(a.reasons)
        self.assertTrue(all(r.strip() for r in a.reasons))
        self.assertAlmostEqual(a.score, round(sum(s.bits for s in a.signals), 2), places=2)


if __name__ == "__main__":
    unittest.main()
