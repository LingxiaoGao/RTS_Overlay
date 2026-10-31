import unittest

from aoe4.smart_calibration import (
    BuildOrderStep,
    CalibrationAnchor,
    CalibrationEngine,
)


class StepValidationTests(unittest.TestCase):
    def test_valid_step(self):
        step = BuildOrderStep('a', 10, 2)
        self.assertEqual((step.id, step.original_time, step.age_dependency),
                         ('a', 10, 2))

    def test_empty_id_rejected(self):
        with self.assertRaises(ValueError):
            BuildOrderStep('', 0)

    def test_negative_time_rejected(self):
        with self.assertRaises(ValueError):
            BuildOrderStep('a', -1)

    def test_non_int_time_rejected(self):
        with self.assertRaises(TypeError):
            BuildOrderStep('a', 1.5)
        with self.assertRaises(TypeError):
            BuildOrderStep('a', True)

    def test_age_out_of_range_rejected(self):
        for age in (0, 1, 5):
            with self.assertRaises(ValueError):
                BuildOrderStep('a', 0, age)

    def test_step_immutable(self):
        step = BuildOrderStep('a', 10)
        with self.assertRaises(Exception):
            step.original_time = 20


class AnchorValidationTests(unittest.TestCase):
    def test_valid_anchor_offset(self):
        self.assertEqual(CalibrationAnchor(2, 300, 250).offset, 50)
        self.assertEqual(CalibrationAnchor(2, 200, 250).offset, -50)

    def test_negative_times_rejected(self):
        with self.assertRaises(ValueError):
            CalibrationAnchor(2, -1, 10)
        with self.assertRaises(ValueError):
            CalibrationAnchor(2, 10, -1)

    def test_bad_age_rejected(self):
        with self.assertRaises(ValueError):
            CalibrationAnchor(1, 10, 10)

    def test_anchor_immutable(self):
        anchor = CalibrationAnchor(2, 10, 10)
        with self.assertRaises(Exception):
            anchor.actual_time = 20


class EngineTests(unittest.TestCase):
    def _steps(self):
        return [
            BuildOrderStep('open', 0),
            BuildOrderStep('feudal', 250, 2),
            BuildOrderStep('castle', 600, 3),
            BuildOrderStep('late', 900),
        ]

    def test_independent_steps_unchanged(self):
        engine = CalibrationEngine(self._steps())
        engine.apply_anchor(CalibrationAnchor(2, 300, 250), 'confirmed')
        times = engine.adjusted_times
        self.assertEqual(times['open'], 0)
        self.assertEqual(times['late'], 900)

    def test_positive_offset(self):
        engine = CalibrationEngine(self._steps())
        engine.apply_anchor(CalibrationAnchor(2, 300, 250), 'confirmed')
        self.assertEqual(engine.adjusted_times['feudal'], 300)
        self.assertEqual(engine.adjusted_times['castle'], 600)

    def test_negative_offset(self):
        engine = CalibrationEngine(self._steps())
        engine.apply_anchor(CalibrationAnchor(2, 200, 250), 'confirmed')
        self.assertEqual(engine.adjusted_times['feudal'], 200)

    def test_negative_result_allowed(self):
        engine = CalibrationEngine([BuildOrderStep('x', 10, 2)])
        engine.apply_anchor(CalibrationAnchor(2, 0, 500), 'confirmed')
        self.assertEqual(engine.adjusted_times['x'], -490)

    def test_separate_anchors(self):
        engine = CalibrationEngine(self._steps())
        self.assertTrue(engine.apply_anchor(
            CalibrationAnchor(2, 300, 250), 'confirmed'))
        self.assertTrue(engine.apply_anchor(
            CalibrationAnchor(3, 500, 600), 'confirmed'))
        times = engine.adjusted_times
        self.assertEqual(times['feudal'], 300)
        self.assertEqual(times['castle'], 500)

    def test_repeated_anchor_rejected_no_mutation(self):
        engine = CalibrationEngine(self._steps())
        engine.apply_anchor(CalibrationAnchor(2, 300, 250), 'confirmed')
        before = engine.adjusted_times
        self.assertFalse(engine.apply_anchor(
            CalibrationAnchor(2, 300, 250), 'confirmed'))
        self.assertEqual(engine.adjusted_times, before)

    def test_conflicting_anchor_rejected_no_mutation(self):
        engine = CalibrationEngine(self._steps())
        engine.apply_anchor(CalibrationAnchor(2, 300, 250), 'confirmed')
        self.assertFalse(engine.apply_anchor(
            CalibrationAnchor(2, 400, 250), 'confirmed'))
        self.assertEqual(engine.adjusted_times['feudal'], 300)

    def test_non_confirmed_statuses_rejected(self):
        for status in ('inferred', 'stale', 'unknown', 'CONFIRMED', '', None):
            engine = CalibrationEngine(self._steps())
            self.assertFalse(engine.apply_anchor(
                CalibrationAnchor(2, 300, 250), status))
            self.assertEqual(engine.anchors, {})
            self.assertEqual(engine.adjusted_times['feudal'], 250)

    def test_rejected_status_does_not_consume_age(self):
        engine = CalibrationEngine(self._steps())
        engine.apply_anchor(CalibrationAnchor(2, 300, 250), 'inferred')
        self.assertTrue(engine.apply_anchor(
            CalibrationAnchor(2, 300, 250), 'confirmed'))

    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            CalibrationEngine([BuildOrderStep('a', 0), BuildOrderStep('a', 1)])

    def test_original_data_unchanged(self):
        steps = self._steps()
        engine = CalibrationEngine(steps)
        engine.apply_anchor(CalibrationAnchor(2, 300, 250), 'confirmed')
        self.assertEqual([s.original_time for s in steps],
                         [0, 250, 600, 900])
        self.assertEqual(engine.steps, tuple(steps))

    def test_order_preserved(self):
        engine = CalibrationEngine(self._steps())
        self.assertEqual([s.id for s in engine.steps],
                         ['open', 'feudal', 'castle', 'late'])

    def test_anchors_copy_is_defensive(self):
        engine = CalibrationEngine(self._steps())
        engine.apply_anchor(CalibrationAnchor(2, 300, 250), 'confirmed')
        snap = engine.anchors
        snap[3] = CalibrationAnchor(3, 1, 1)
        self.assertNotIn(3, engine.anchors)


if __name__ == '__main__':
    unittest.main()
