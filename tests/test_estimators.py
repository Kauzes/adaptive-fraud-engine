import math
import unittest

from adaptive_fraud.estimators import (
    CategoryProfile,
    CircularHourProfile,
    EwmaStats,
    RunningStats,
)


class RunningStatsTests(unittest.TestCase):
    def test_matches_textbook_mean_and_variance(self):
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        stats = RunningStats()
        for v in values:
            stats.update(v)

        expected_mean = sum(values) / len(values)
        expected_var = sum((v - expected_mean) ** 2 for v in values) / (len(values) - 1)

        self.assertAlmostEqual(stats.mean, expected_mean, places=10)
        self.assertAlmostEqual(stats.variance, expected_var, places=10)

    def test_stays_precise_where_the_naive_formula_collapses(self):
        # Large offset, tiny spread: sum-of-squares cancellation territory.
        values = [1e9 + d for d in (1.0, 2.0, 3.0, 4.0)]
        stats = RunningStats()
        for v in values:
            stats.update(v)

        self.assertAlmostEqual(stats.variance, 1.6666666666, places=4)

    def test_no_spread_information_from_a_single_point(self):
        stats = RunningStats()
        stats.update(42.0)
        self.assertEqual(stats.variance, 0.0)
        self.assertEqual(stats.mean, 42.0)


class EwmaStatsTests(unittest.TestCase):
    def test_tracks_a_level_shift_that_a_plain_mean_would_lag(self):
        ewma = EwmaStats(alpha=0.2)
        running = RunningStats()

        for _ in range(60):
            ewma.update(10.0)
            running.update(10.0)
        for _ in range(30):
            ewma.update(100.0)
            running.update(100.0)

        # The customer now reliably spends 100; EWMA should have followed much closer.
        self.assertGreater(ewma.mean, 95.0)
        self.assertLess(running.mean, 50.0)

    def test_variance_is_non_negative_and_grows_with_spread(self):
        steady = EwmaStats(alpha=0.3)
        jumpy = EwmaStats(alpha=0.3)
        for i in range(50):
            steady.update(10.0)
            jumpy.update(10.0 if i % 2 else 200.0)

        self.assertGreaterEqual(steady.variance, 0.0)
        self.assertGreater(jumpy.stdev, steady.stdev)


class CircularHourProfileTests(unittest.TestCase):
    def test_midnight_neighbours_are_treated_as_adjacent(self):
        profile = CircularHourProfile()
        for _ in range(30):
            profile.update(23)

        # 00:00 borrows from the 23:00 peak; 12:00 does not.
        self.assertGreater(profile.probability(0), profile.probability(12))

    def test_unseen_hour_is_more_surprising_than_a_habitual_one(self):
        profile = CircularHourProfile()
        for _ in range(50):
            profile.update(9)

        self.assertGreater(profile.surprisal(3), profile.surprisal(9))

    def test_uniform_prior_before_any_observation(self):
        profile = CircularHourProfile()
        # No data: every hour equally (un)likely, so surprisal is log2(24).
        self.assertAlmostEqual(profile.surprisal(0), math.log2(24), places=6)
        self.assertAlmostEqual(profile.surprisal(13), math.log2(24), places=6)

    def test_probabilities_form_a_distribution(self):
        profile = CircularHourProfile()
        for hour in (8, 9, 9, 10, 22):
            profile.update(hour)

        total = sum(profile.probability(h) for h in range(24))
        self.assertAlmostEqual(total, 1.0, places=9)


class CategoryProfileTests(unittest.TestCase):
    def test_unseen_label_never_has_zero_probability(self):
        profile = CategoryProfile()
        profile.update("phone-1")
        self.assertGreater(profile.probability("brand-new-device"), 0.0)

    def test_familiar_label_is_less_surprising(self):
        profile = CategoryProfile()
        for _ in range(20):
            profile.update("TR")
        profile.update("DE")

        self.assertLess(profile.surprisal("TR"), profile.surprisal("DE"))
        self.assertLess(profile.surprisal("DE"), profile.surprisal("BR"))

    def test_known_requires_an_actual_sighting(self):
        profile = CategoryProfile()
        profile.update("TR")
        self.assertTrue(profile.is_known("TR"))
        self.assertFalse(profile.is_known("RU"))


if __name__ == "__main__":
    unittest.main()
