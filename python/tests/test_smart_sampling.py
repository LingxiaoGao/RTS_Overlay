import math
import unittest

from aoe4.smart_sampling import ObservationScheduler, SamplingConfig


class SamplingTests(unittest.TestCase):
    def test_default_and_invalid_intervals(self):
        self.assertEqual(SamplingConfig().interval_seconds, 0.5)
        for value in (0.0, -1.0, math.nan, math.inf):
            with self.assertRaises(ValueError):
                SamplingConfig(interval_seconds=value)

    def test_first_sample_and_boundary(self):
        scheduler = ObservationScheduler()
        self.assertTrue(scheduler.due(100.0))
        scheduler.mark_sample(100.0)
        self.assertFalse(scheduler.due(100.4))
        self.assertTrue(scheduler.due(100.5))

    def test_backward_and_reset(self):
        scheduler = ObservationScheduler()
        scheduler.mark_sample(100.0)
        with self.assertRaises(ValueError):
            scheduler.mark_sample(99.0)
        scheduler.reset()
        self.assertTrue(scheduler.due(99.0))


if __name__ == "__main__":
    unittest.main()
