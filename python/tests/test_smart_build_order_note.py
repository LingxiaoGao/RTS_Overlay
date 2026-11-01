import tempfile
import unittest
from pathlib import Path

from aoe4.smart_build_order_note import render_note_html


class RenderNoteHtmlTests(unittest.TestCase):
    def test_icons_translation_and_adjacent_markers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "resource").mkdir()
            for name in ("resource_food.webp", "resource_wood.webp"):
                (root / "resource" / name).write_bytes(b"x")
            rendered = render_note_html(
                "Send 3 @resource/resource_food.webp@@resource/resource_wood.webp@ "
                "to enemy base until age up",
                root,
                18,
            )
            self.assertEqual(rendered.count("<img "), 2)
            self.assertIn('width="18"', rendered)
            self.assertIn("派遣 3", rendered)
            self.assertIn("敌方基地", rendered)
            self.assertIn("升级时代", rendered)

    def test_text_is_escaped_and_entity_arrow_is_preserved(self):
        rendered = render_note_html("Build &lt;b&gt;x&lt;/b&gt; -&gt; house", ".", 16)
        self.assertNotIn("<b>", rendered)
        self.assertIn("&lt;b&gt;x&lt;/b&gt;", rendered)
        self.assertIn("-&gt;", rendered)
        self.assertIn("房屋", rendered)

    def test_missing_image_uses_readable_label(self):
        rendered = render_note_html("@unit_worker/royal-villager.webp@", ".", 16)
        self.assertNotIn("<img", rendered)
        self.assertIn("royal villager", rendered)

    def test_invalid_icon_size(self):
        with self.assertRaises(ValueError):
            render_note_html("text", ".", 0)


if __name__ == "__main__":
    unittest.main()
