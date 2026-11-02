import unittest
from types import SimpleNamespace

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


class _InGameVision:
    def __init__(self, cue):
        self.cue = cue

    def recognize_in_game(self, _flag, _buttons, _provisional):
        return self.cue, False


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
    controller._gameplay_hud_started = False
    controller._pipeline = _Resettable()
    controller._civilization_vision = _Vision(cues)
    controller._next_loading_probe_at = 0.0
    controller._loading_probe_civilization = None
    controller._loading_probe_hits = 0
    return controller


class SmartControllerSessionTests(unittest.TestCase):
    def test_resource_hud_starts_timer_anchor_without_civilization(self):
        controller = _controller()
        controller._civilization_started = False
        controller._civilization_decision = None
        result = SimpleNamespace(
            evidence={"food": object(), "wood": object()},
            state=SimpleNamespace(updated_at=2.0),
        )

        controller._begin_gameplay_timing(result)
        controller._begin_gameplay_timing(result)

        self.assertTrue(controller._gameplay_hud_started)
        self.assertTrue(controller._civilization_started)
        self.assertEqual(controller._timer_sync.reset_count, 1)

    def test_single_or_missing_resource_field_does_not_start_gameplay(self):
        controller = _controller()
        controller._civilization_started = False
        controller._civilization_decision = None

        controller._begin_gameplay_timing(
            SimpleNamespace(evidence={"food": object()}, state=SimpleNamespace(updated_at=1.0))
        )

        self.assertFalse(controller._gameplay_hud_started)
        self.assertFalse(controller._civilization_started)
        self.assertEqual(controller._timer_sync.reset_count, 0)

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

    def test_visible_hud_blocks_false_loading_screen_reset(self):
        cue = CivilizationCue("Holy Roman Empire", 0.9)
        controller = _controller(cues=(cue, cue))
        result = SimpleNamespace(evidence={"food": object()})

        controller._observe_session_frame(object(), 10.0, result)
        controller._observe_session_frame(object(), 13.0, result)

        self.assertEqual(
            controller._civilization_decision.civilization, "English"
        )
        self.assertEqual(controller._timer_sync.reset_count, 0)
        self.assertEqual(controller._pipeline.reset_count, 0)

    def test_in_game_flag_starts_confirmation_without_game_timer(self):
        controller = SmartHudController.__new__(SmartHudController)
        controller._civilization_tracker = CivilizationTracker()
        controller._civilization_started = False
        controller._civilization_decision = None
        controller._civilization_vision = _InGameVision(
            CivilizationCue("Ayyubids", 0.85)
        )
        controller._frame_bgr = lambda frame: frame
        unknown_timer = SimpleNamespace(
            state=SimpleNamespace(
                game_time=SimpleNamespace(
                    value=None,
                    status=None,
                )
            )
        )
        crops = {
            "civilization_flag": object(),
            "civilization_production_buttons": object(),
        }

        for at in (0.0, 5.0, 10.0, 15.0, 30.0):
            controller._observe_civilization(unknown_timer, crops, at)

        self.assertTrue(controller._civilization_started)
        self.assertEqual(
            controller._civilization_decision.civilization, "Ayyubids"
        )
        self.assertEqual(controller._civilization_decision.source, "in_game")

    def test_confirmed_timer_cannot_start_civilization_without_hud_flag(self):
        controller = SmartHudController.__new__(SmartHudController)
        controller._civilization_tracker = CivilizationTracker()
        controller._civilization_started = False
        controller._civilization_decision = None
        controller._civilization_vision = _InGameVision(None)
        controller._frame_bgr = lambda frame: frame
        confirmed_timer = SimpleNamespace(
            state=SimpleNamespace(
                game_time=SimpleNamespace(value=2, status=object())
            )
        )
        crops = {
            "civilization_flag": object(),
            "civilization_production_buttons": object(),
        }

        controller._observe_civilization(confirmed_timer, crops, 10.0)

        self.assertFalse(controller._civilization_started)
        self.assertIsNone(controller._civilization_decision)

    def test_loading_candidate_and_final_lock_are_distinct_states(self):
        controller = SmartHudController.__new__(SmartHudController)
        controller._civilization_tracker = CivilizationTracker()
        controller._civilization_started = False
        controller._civilization_decision = None
        controller._civilization_vision = _InGameVision(
            CivilizationCue("English", 0.90)
        )
        controller._frame_bgr = lambda frame: frame
        controller._civilization_tracker.observe_loading(
            CivilizationCue("English", 0.95)
        )
        unknown_timer = SimpleNamespace(state=SimpleNamespace(game_time=None))
        crops = {
            "civilization_flag": object(),
            "civilization_production_buttons": object(),
        }

        self.assertEqual(
            controller.detected_civilization.source, "loading"
        )
        self.assertFalse(controller.civilization_locked)

        for at in (0.0, 5.0, 10.0, 15.0, 30.0):
            controller._observe_civilization(unknown_timer, crops, at)

        self.assertTrue(controller.civilization_locked)
        self.assertEqual(
            controller.detected_civilization.source, "combined"
        )


if __name__ == "__main__":
    unittest.main()
