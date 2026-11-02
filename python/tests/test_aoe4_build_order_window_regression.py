import os
import shutil
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QPoint, Qt
from PyQt5.QtWidgets import QApplication

import common.rts_overlay as rts_overlay_module
from aoe4.aoe4_game_overlay import AoE4GameOverlay
from common.rts_overlay import PanelID


class _DragEvent:
    def __init__(self, point):
        self._point = point

    def buttons(self):
        return Qt.LeftButton

    def globalPos(self):
        return self._point


class AoE4BuildOrderWindowRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        cls.python_root = os.path.join(cls.repo_root, "python")

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="aoe4-build-window-")
        self.original_user_data_dir = rts_overlay_module.appdirs.user_data_dir
        self.original_popup_message = rts_overlay_module.popup_message
        rts_overlay_module.appdirs.user_data_dir = lambda: self.temp_dir
        rts_overlay_module.popup_message = lambda *args, **kwargs: None

        runtime_root = os.path.join(self.temp_dir, "runtime")
        runtime_python = os.path.join(runtime_root, "python")
        os.makedirs(runtime_python)
        os.symlink(os.path.join(self.repo_root, "docs"), os.path.join(runtime_root, "docs"))

        target = os.path.join(self.temp_dir, "RTS_Overlay", "aoe4", "build_orders")
        os.makedirs(target, exist_ok=True)
        for filename in (
            "aoe4guides-MAL-NMRXBRDXxdlq5lRtI1Nl.json",
            "aoe4guides-MAL-0DSW4iLRlv22BXBjlirM.json",
            "aoe4guides-MAL-IykuH2TryjxXISdqyYAR.json",
            "aoe4guides-FRE-EHbobnEzHzi828z5fsB0.json",
        ):
            shutil.copy(
                os.path.join(self.python_root, "demo_data", "build_orders", filename),
                target,
            )
        self.overlay = AoE4GameOverlay(self.app, runtime_python)
        index = self.overlay.civilization_combo_ids.index("Malians")
        self.overlay.civilization_select.setCurrentIndex(index)
        self.overlay._populate_civilization_build_orders()
        self.app.processEvents()

    def tearDown(self):
        self.overlay.close()
        rts_overlay_module.appdirs.user_data_dir = self.original_user_data_dir
        rts_overlay_module.popup_message = self.original_popup_message
        shutil.rmtree(self.temp_dir)

    def test_highlighted_guide_opens_with_steps_and_full_list_returns(self):
        guide_names = tuple(self.overlay.valid_build_orders)
        self.assertEqual(3, len(guide_names))

        self.overlay.next_panel()
        self.app.processEvents()

        self.assertEqual(PanelID.BUILD_ORDER, self.overlay.selected_panel)
        self.assertIsNotNone(self.overlay.selected_build_order)
        self.assertGreater(len(self.overlay.selected_build_order["build_order"]), 0)
        self.assertGreater(self.overlay.build_order_table.rowCount(), 1)
        empty_message = self.overlay.build_order_table.item(0, 6)
        self.assertFalse(empty_message and empty_message.text() == "未选择建造顺序。")

        self.overlay.next_panel()
        self.app.processEvents()

        self.assertEqual(PanelID.CONFIG, self.overlay.selected_panel)
        self.assertEqual(guide_names, tuple(self.overlay.valid_build_orders))

    def test_detail_window_accepts_drag_and_resize(self):
        self.overlay.next_panel()
        self.app.processEvents()
        self.assertFalse(
            self.overlay.testAttribute(Qt.WA_TransparentForMouseEvents)
        )
        self.assertTrue(self.overlay.build_order_drag_handle.isVisible())
        self.assertTrue(self.overlay.build_order_size_grip.isVisible())

        original_position = self.overlay.pos()
        start = QPoint(original_position.x() + 20, original_position.y() + 20)
        self.overlay._drag_build_order_window(_DragEvent(start))
        self.overlay._drag_build_order_window(_DragEvent(start + QPoint(80, 45)))
        self.assertEqual(original_position + QPoint(80, 45), self.overlay.pos())

        original_size = self.overlay.size()
        self.overlay.resize(original_size.width() + 120, original_size.height() + 90)
        self.app.processEvents()
        self.assertEqual(original_size.width() + 120, self.overlay.width())
        self.assertEqual(original_size.height() + 90, self.overlay.height())
        self.assertEqual(
            self.overlay.width(),
            self.overlay.build_order_table.width()
            + 2 * self.overlay.settings.layout.border_size,
        )

    def test_final_civilization_check_ejects_wrong_open_guide(self):
        french_index = self.overlay.civilization_combo_ids.index("French")
        self.overlay.civilization_select.setCurrentIndex(french_index)
        self.overlay._populate_civilization_build_orders()
        self.assertGreater(len(self.overlay.valid_build_orders), 0)
        self.overlay.next_panel()
        self.assertEqual(
            self.overlay.selected_build_order["civilization"], "French"
        )

        self.overlay.handle_detected_civilization("Malians")
        self.app.processEvents()

        self.assertEqual(PanelID.CONFIG, self.overlay.selected_panel)
        self.assertIsNone(self.overlay.selected_build_order)
        self.assertEqual(
            "Malians",
            self.overlay.civilization_combo_ids[
                self.overlay.civilization_select.currentIndex()
            ],
        )
        self.assertEqual(3, len(self.overlay.valid_build_orders))


if __name__ == "__main__":
    unittest.main()
