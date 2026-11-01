import unittest

from aoe4.smart_build_order_display import build_age_block_layout


class BuildAgeBlockLayoutTests(unittest.TestCase):
    def test_start_and_completion_from_notes(self):
        steps = [
            {"notes": "2:10 age 2", "age": 1},
            {"notes": "", "age": 1},
            {"notes": "", "age": 1},
            {"notes": "", "age": 1},
            {"notes": "", "age": 2},
        ]
        separators, display_row = build_age_block_layout(steps, 1)
        self.assertEqual(separators[0], (("start", 2),))
        self.assertEqual(separators[4], (("completion", 2),))
        self.assertNotIn(1, separators)
        self.assertEqual(display_row, 2)

    def test_ages_three_and_four(self):
        steps = [
            {"notes": "castle", "age": 1},
            {"notes": "", "age": 2},
            {"notes": "帝王", "age": 2},
            {"notes": "", "age": 3},
            {"notes": "", "age": 4},
        ]
        separators, _ = build_age_block_layout(steps, 0)
        self.assertEqual(separators[0], (("start", 3),))
        self.assertEqual(separators[2], (("start", 4),))
        self.assertEqual(separators[3], (("completion", 3),))
        self.assertEqual(separators[4], (("completion", 4),))

    def test_completion_only_fallback(self):
        steps = [
            {"notes": "", "age": 1},
            {"notes": "", "age": 1},
            {"notes": "", "age": 2},
            {"notes": "", "age": 2},
            {"notes": "", "age": 3},
        ]
        separators, display_row = build_age_block_layout(steps, 2)
        self.assertEqual(separators[2], (("completion", 2),))
        self.assertEqual(separators[4], (("completion", 3),))
        self.assertEqual(display_row, 3)

    def test_generic_age_up_assigns_in_order(self):
        steps = [
            {"notes": "age up", "age": 1},
            {"notes": "", "age": 2},
            {"notes": "升本", "age": 2},
            {"notes": "", "age": 3},
        ]
        separators, _ = build_age_block_layout(steps, 0)
        self.assertEqual(separators[0], (("start", 2),))
        self.assertEqual(separators[1], (("completion", 2),))
        self.assertEqual(separators[2], (("start", 3),))
        self.assertEqual(separators[3], (("completion", 3),))

    def test_aoe4_guides_aging_up_phrase_starts_block(self):
        steps = [
            {"notes": ["Start aging up with 4 villagers"], "age": 1},
            {"notes": ["Keep gathering"], "age": 1},
            {"notes": ["Feudal complete"], "age": 2},
        ]
        separators, _ = build_age_block_layout(steps, 1)
        self.assertEqual(separators[0], (("start", 2),))
        self.assertEqual(separators[2], (("completion", 2),))

    def test_no_duplicate_completion(self):
        steps = [
            {"notes": "feudal", "age": 1},
            {"notes": "", "age": 2},
            {"notes": "", "age": 2},
        ]
        separators, _ = build_age_block_layout(steps, 0)
        completions = [kind for kind, _ in separators[1]]
        self.assertEqual(completions, ["completion"])

    def test_notes_may_be_list(self):
        steps = [
            {"notes": ["feudal", "boom"], "age": 1},
            {"notes": None, "age": 2},
        ]
        separators, _ = build_age_block_layout(steps, 0)
        self.assertEqual(separators[0], (("start", 2),))
        self.assertEqual(separators[1], (("completion", 2),))

    def test_current_row_after_separator_counts(self):
        steps = [
            {"notes": "castle", "age": 1},
            {"notes": "imperial", "age": 2},
            {"notes": "", "age": 3},
        ]
        separators, display_row = build_age_block_layout(steps, 1)
        self.assertEqual(separators[0], (("start", 3),))
        self.assertEqual(
            separators[1], (("start", 4), ("completion", 2))
        )
        self.assertEqual(separators[2], (("completion", 3),))
        self.assertEqual(display_row, 4)


if __name__ == "__main__":
    unittest.main()
