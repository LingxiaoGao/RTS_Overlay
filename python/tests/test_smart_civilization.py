import unittest

from aoe4.smart_civilization import CivilizationCue, CivilizationTracker


def cue(name, confidence=0.7):
    return CivilizationCue(name, confidence)


class CivilizationTrackerTests(unittest.TestCase):
    def _lock(self, tracker, name="English", confidence=0.7):
        tracker.start_game(0.0)
        for at in (1.0, 4.0, 9.0, 20.0):
            self.assertIsNone(tracker.observe_in_game(cue(name, confidence), True, at))
        return tracker.finalize(30.0)

    def test_four_nonconsecutive_samples_lock_at_thirty_seconds(self):
        decision = self._lock(CivilizationTracker())
        self.assertEqual(decision.civilization, "English")
        self.assertEqual(decision.source, "in_game")

    def test_missing_frames_preserve_streak_and_buttons_are_not_required(self):
        tracker = CivilizationTracker()
        tracker.start_game(0.0)
        self.assertIsNone(tracker.observe_in_game(cue("English"), True, 1.0))
        self.assertIsNone(tracker.observe_in_game(None, True, 2.0))
        self.assertIsNone(tracker.observe_in_game(cue("English"), False, 3.0))
        self.assertIsNone(tracker.observe_in_game(cue("English"), False, 4.0))
        self.assertIsNone(tracker.observe_in_game(cue("English"), False, 5.0))
        decision = tracker.finalize(30.0)
        self.assertEqual(decision.civilization, "English")

    def test_different_positive_cue_restarts_streak(self):
        tracker = CivilizationTracker()
        tracker.start_game(0.0)
        tracker.observe_in_game(cue("English"), True, 1.0)
        tracker.observe_in_game(cue("French"), True, 2.0)
        tracker.observe_in_game(cue("French"), True, 3.0)
        tracker.observe_in_game(cue("French"), True, 4.0)
        tracker.observe_in_game(cue("French"), True, 5.0)
        self.assertEqual(tracker.finalize(30.0).civilization, "French")

    def test_first_positive_observation_starts_window_without_game_time(self):
        tracker = CivilizationTracker()
        self.assertIsNone(tracker.observe_in_game(cue("English"), True, 10.0))
        self.assertIsNone(tracker.observe_in_game(None, True, 11.0))
        self.assertIsNone(tracker.observe_in_game(cue("English"), True, 12.0))
        tracker.observe_in_game(cue("English"), True, 13.0)
        tracker.observe_in_game(cue("English"), True, 14.0)
        decision = tracker.finalize(40.0)
        self.assertEqual(decision.civilization, "English")

    def test_loading_agreement_and_conflict_arbitration(self):
        combined = CivilizationTracker()
        combined.observe_loading(cue("English", 0.8))
        decision = self._lock(combined, "English", 0.7)
        self.assertEqual((decision.source, decision.confidence), ("combined", 0.8))

        loading_wins = CivilizationTracker()
        loading_wins.observe_loading(cue("French", 0.9))
        decision = self._lock(loading_wins, "English", 0.7)
        self.assertEqual((decision.civilization, decision.source), ("French", "loading"))

        tie = CivilizationTracker()
        tie.observe_loading(cue("French", 0.7))
        decision = self._lock(tie, "English", 0.7)
        self.assertEqual((decision.civilization, decision.source), ("English", "in_game"))

    def test_deadline_lock_immutability_and_new_loading_reset(self):
        late = CivilizationTracker()
        late.start_game(0.0)
        late.observe_in_game(cue("English"), True, 1.0)
        late.observe_in_game(cue("English"), True, 10.0)
        late.observe_in_game(cue("English"), True, 28.0)
        late.observe_in_game(cue("English"), True, 29.0)
        self.assertEqual(late.finalize(31.0).civilization, "English")

        tracker = CivilizationTracker()
        locked = self._lock(tracker)
        self.assertIs(tracker.observe_in_game(cue("French", 1.0), True, 100.0), locked)
        tracker.observe_loading(cue("French", 0.8))
        self.assertIsNone(tracker.decision)

    def test_reset_and_external_cue_validation(self):
        tracker = CivilizationTracker()
        self._lock(tracker)
        tracker.reset()
        self.assertIsNone(tracker.decision)
        for bad in (("", 0.5), ("English", -0.1), ("English", float("nan"))):
            with self.assertRaises(ValueError):
                CivilizationCue(*bad)


if __name__ == "__main__":
    unittest.main()
