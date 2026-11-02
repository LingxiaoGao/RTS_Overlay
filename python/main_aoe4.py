# Game overlay application for Age of Empires IV (AoE4)
import sys
import pathlib
import os
import logging
from logging.handlers import RotatingFileHandler
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from aoe4.aoe4_game_overlay import AoE4GameOverlay
from aoe4.smart_controller import SmartHudController
from aoe4.smart_controller_config import SmartControllerConfig
from aoe4.smart_environment import EnvironmentValidator
from aoe4.smart_status_window import SmartStatusWindow

if __name__ == '__main__':
    directory_main = pathlib.Path(__file__).parent.resolve()
    log_directory = directory_main / "local_config" / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    log_path = log_directory / "aoe4_overlay.log"
    handler = RotatingFileHandler(
        log_path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[handler],
    )
    logger = logging.getLogger("aoe4_overlay")

    def log_uncaught(exc_type, exc_value, exc_traceback):
        logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = log_uncaught
    logger.info("AOE4 Overlay starting; log=%s", log_path)

    runtime_smoke_test = "--runtime-smoke-test" in sys.argv
    smoke_config = None
    if runtime_smoke_test:
        # The release build explicitly includes Qt platform plugins so this
        # one-shot verifier can exercise real widgets on a headless runner.
        # Normal launches do not set this variable and use native Windows UI.
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import cv2  # noqa: F401
        import keyboard  # noqa: F401
        import mss  # noqa: F401
        import numpy  # noqa: F401
        import pynput  # noqa: F401
        from omegaconf import OmegaConf  # noqa: F401
        from aoe4.smart_template_ocr import TemplateHudRecognizer

        logger.info("Runtime smoke dependencies imported")

        smoke_config = SmartControllerConfig.from_file(
            directory_main / "smart_hud_config.json"
        )
        if smoke_config.hud_regions.regions["game_time"].height != 51:
            raise RuntimeError("packaged timer ROI must use the 51px dual-band profile")
        TemplateHudRecognizer(
            smoke_config.hud_regions,
            directory_main / "hud_templates_1280x720",
        ).close()
        logger.info("Runtime smoke packaged OCR data validated")

    app = QApplication(sys.argv)
    if runtime_smoke_test:
        logger.info("Runtime smoke QApplication created")
    window = AoE4GameOverlay(app=app, directory_main=str(directory_main))
    if runtime_smoke_test:
        logger.info("Runtime smoke overlay window created")

    if runtime_smoke_test:
        from PyQt5.QtCore import Qt
        from common.rts_overlay import PanelID

        if window.settings.call_ms != 20:
            raise RuntimeError("local overlay timer cadence must remain 20ms")
        english_index = window.civilization_combo_ids.index("English")
        window.civilization_select.setCurrentIndex(english_index)
        window._populate_civilization_build_orders()
        expected_guides = tuple(window.valid_build_orders)
        if len(expected_guides) < 3:
            raise RuntimeError("packaged English guide list is incomplete")

        window.next_panel()
        app.processEvents()
        if window.selected_panel != PanelID.BUILD_ORDER:
            raise RuntimeError("packaged guide did not open its detail panel")
        if not window.selected_build_order or not window.selected_build_order.get("build_order"):
            raise RuntimeError("packaged guide detail is empty")
        if window.build_order_table.rowCount() <= 1:
            raise RuntimeError("packaged guide table has no build-order rows")
        if window.testAttribute(Qt.WA_TransparentForMouseEvents):
            raise RuntimeError("packaged guide detail remains mouse-transparent")
        old_size = window.size()
        window.resize(old_size.width() + 80, old_size.height() + 60)
        app.processEvents()
        if window.size() == old_size:
            raise RuntimeError("packaged guide detail cannot be resized")

        window.next_panel()
        app.processEvents()
        if tuple(window.valid_build_orders) != expected_guides:
            raise RuntimeError("returning from detail lost packaged guides")

        french_index = window.civilization_combo_ids.index("French")
        window.civilization_select.setCurrentIndex(french_index)
        window._populate_civilization_build_orders()
        window.next_panel()
        window.handle_detected_civilization("English")
        app.processEvents()
        selected_civ = window.civilization_combo_ids[
            window.civilization_select.currentIndex()
        ]
        if window.selected_panel != PanelID.CONFIG or window.selected_build_order is not None:
            raise RuntimeError("wrong-civilization guide was not ejected")
        if selected_civ != "English" or tuple(window.valid_build_orders) != expected_guides:
            raise RuntimeError("final civilization did not restore the full correct guide list")

        window.close()
        logger.info("Packaged runtime and guide-window smoke test passed")
        # Windows keyboard/mouse hook helpers may own non-daemon threads.
        # This code path is a dedicated one-shot verifier, so flush its proof
        # and terminate without waiting for runtime hooks used by normal play.
        logging.shutdown()
        os._exit(0)

    # timer to call the functions related to mouse and keyboard inputs
    timer = QTimer()
    timer.timeout.connect(window.timer_build_order_call)
    timer.timeout.connect(window.timer_mouse_keyboard_call)
    # Keep the original overlay clock/input cadence.  Smart HUD capture has
    # its own 500ms timer below; it only supplies OCR anchors and must not
    # reduce the local build-order timer's refresh rate.
    timer.setInterval(window.settings.call_ms)
    timer.start()

    status_window = SmartStatusWindow()
    status_window.show()

    controller = None
    smart_timer = None

    config_path = directory_main / "smart_hud_config.json"

    report = EnvironmentValidator().validate(config_path)
    if not report.ready:
        logger.error("Smart OCR disabled: %s", " | ".join(report.messages_zh))
        status_window.show_disabled("\n".join(report.messages_zh))
    else:
        try:
            config = SmartControllerConfig.from_file(config_path)
            controller = SmartHudController(
                config,
                template_dir=directory_main / "hud_templates_1280x720",
                game_assets_dir=window.directory_game_pictures,
            )
            last_poll_error = None
            last_civilization_signature = None

            def poll_smart():
                global last_poll_error, last_civilization_signature
                result = controller.poll_safe()
                if result.error != last_poll_error:
                    if result.error:
                        logger.error("Smart OCR poll failed: %s", result.error)
                    elif last_poll_error:
                        logger.info("Smart OCR poll recovered")
                    last_poll_error = result.error
                notifications = ()
                if result.result is not None:
                    notifications = window.apply_smart_pipeline_result(result.result)
                    decision = controller.detected_civilization
                    signature = (
                        None
                        if decision is None
                        else (
                            decision.civilization,
                            decision.source,
                            controller.civilization_locked,
                        )
                    )
                    if decision is not None and signature != last_civilization_signature:
                        window.handle_detected_civilization(decision.civilization)
                        logger.info(
                            "Civilization locked: %s confidence=%.3f source=%s",
                            decision.civilization,
                            decision.confidence,
                            decision.source,
                        )
                        last_civilization_signature = signature
                status_window.show_poll(result, notifications)

            smart_timer = QTimer()
            smart_timer.timeout.connect(poll_smart)
            smart_timer.setInterval(500)
            smart_timer.start()

            app.aboutToQuit.connect(controller.close)
        except Exception as exc:
            logger.exception("Smart OCR initialization failed")
            status_window.show_disabled(f"{type(exc).__name__}: {exc}")

    exit_event = app.exec()
    sys.exit(exit_event)
