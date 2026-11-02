"""Tests for anchoring the build-order timer to a confirmed OCR time."""

from __future__ import annotations

import unittest

from aoe4.smart_timer_anchor import anchor_build_order_timer


def _build_timer() -> dict:
    """Create a plain-dict timer mirroring the base-class contract."""
    return {
        'available': True,
        'steps': [],
        'step_starting_flag': False,
        'use_timer': False,
        'run_timer': False,
        'time_sec': 0.0,
        'time_int': 0,
        'time_sec_init': 0.0,
        'absolute_time_init': 0.0,
        'last_time_int': 0,
        'steps_ids': [],
        'last_steps_ids': [],
    }


class TimerAnchorTests(unittest.TestCase):
    def test_initial_anchor_sets_all_fields(self):
        timer = _build_timer()
        anchor_build_order_timer(timer, 125, 1000.5)
        self.assertIs(timer['use_timer'], True)
        self.assertIs(timer['run_timer'], True)
        self.assertEqual(timer['time_sec'], 125.0)
        self.assertEqual(timer['time_int'], 125)
        self.assertEqual(timer['time_sec_init'], 125.0)
        self.assertEqual(timer['absolute_time_init'], 1000.5)
        self.assertEqual(timer['last_time_int'], 125)

    def test_reanchor_updates_all_fields(self):
        timer = _build_timer()
        anchor_build_order_timer(timer, 60, 500.0)
        anchor_build_order_timer(timer, 90, 530.25)
        self.assertIs(timer['use_timer'], True)
        self.assertIs(timer['run_timer'], True)
        self.assertEqual(timer['time_sec'], 90.0)
        self.assertEqual(timer['time_int'], 90)
        self.assertEqual(timer['time_sec_init'], 90.0)
        self.assertEqual(timer['absolute_time_init'], 530.25)
        self.assertEqual(timer['last_time_int'], 90)


if __name__ == '__main__':
    unittest.main()
