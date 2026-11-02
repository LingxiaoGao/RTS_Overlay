import os
import unittest
from unittest.mock import patch

from common.keyboard_mouse import KeyboardMouseManagement


class KeyboardMouseHeadlessTest(unittest.TestCase):
    def test_explicit_headless_mode_does_not_start_global_hooks(self):
        with (
            patch.dict(os.environ, {"RTS_OVERLAY_DISABLE_GLOBAL_INPUT": "1"}),
            patch("common.keyboard_mouse.mouse.Listener") as listener,
            patch("common.keyboard_mouse.add_hotkey") as add_hotkey,
        ):
            manager = KeyboardMouseManagement()
            self.assertTrue(manager.update_keyboard_hotkey("next", "ctrl+n"))
            self.assertFalse(manager.is_keyboard_hotkey_pressed("next"))

        listener.assert_not_called()
        add_hotkey.assert_not_called()
        self.assertIsNone(manager.mouse_listener)


if __name__ == "__main__":
    unittest.main()
