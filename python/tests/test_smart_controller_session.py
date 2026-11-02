import unittest

from aoe4.smart_civilization import (
    CivilizationCue,
    CivilizationDecision,
    CivilizationTracker,
)
from aoe4.smart_controller import SmartHudController


class _Resettable:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _GameEnd:
    def __init__(self, detected=False):
        self.detected = detected

    def observe(self, _frame):
        return self.detected


class _Vision:
    def __init__(self, cues):
        self.cues = iter(cues)

    def recognize_loading(self, _frame):
        return next(self.cues, None)


def _controller(cues=(), game_end=False):
    controller = SmartHudController.__new__(SmartHudController)
    controller._civilization_tracker = CivilizationTracker()
    controller._civilization_started = True
    controller._civilization_decision = CivilizationDecision(
        "English", 0.8, "combined"
    )
    controller._loading_observed = True
    controller._match_ended = False
    controller._game_end_detector = _GameEnd(game_end)
    controller._timer_sync = _Resettable()
    controller._pipeline = _Resettable()
    controller._civilization_vision = _Vision(cues)
    controller._next_loading_probe_at = 0.0
    controller._loading_probe_civilization = None
    controller._loading_probe_hits = 0
    return controller


class SmartControllerSessionTests(unittest.TestCase):
    def test_game_end_clears_civilization_timer_and_pipeline(self):
        controller = _controller(game_end=True)

        controller._observe_session_frame(object(), 10.0)

        self.assertTrue(controller._match_ended)
        self.assertIsNone(controller._civilization_decision)
        self.assertFalse(controller._civilization_started)
        self.assertEqual(controller._timer_sync.reset_count, 1)
        self.assertEqual(controller._pipeline.reset_count, 1)

    def test_new_loading_screen_resets_locked_match_after_two_probes(self):
        cue = CivilizationCue("French", 0.75)
        controller = _controller(cues=(cue, cue))

        controller._observe_session_frame(object(), 10.0)
        self.assertEqual(controller._civilization_decision.civilization, "English")
        controller._observe_session_frame(object(), 13.0)

        self.assertIsNone(controller._civilization_decision)
        self.assertFalse(controller._civilization_started)
        self.assertTrue(controller._loading_observed)
        self.assertEqual(
            controller._civilization_tracker.candidate.civilization, "French"
        )
        self.assertEqual(controller._timer_sync.reset_count, 1)
        self.assertEqual(controller._pipeline.reset_count, 1)


if __name__ == "__main__":
    unittest.main()
