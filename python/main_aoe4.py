# Game overlay application for Age of Empires IV (AoE4)
import sys
import pathlib
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

    if "--runtime-smoke-test" in sys.argv:
        import cv2  # noqa: F401
        import keyboard  # noqa: F401
        import mss
        import numpy  # noqa: F401
        import pynput  # noqa: F401
        from omegaconf import OmegaConf  # noqa: F401
        from aoe4.smart_template_ocr import TemplateHudRecognizer

        capture_backend = mss.mss()
        capture_backend.close()

        smoke_config = SmartControllerConfig.from_file(
            directory_main / "smart_hud_config.json"
        )
        TemplateHudRecognizer(
            smoke_config.hud_regions,
            directory_main / "hud_templates_1280x720",
        ).close()
        logger.info("Packaged runtime smoke test passed")
        sys.exit(0)

    app = QApplication(sys.argv)
    window = AoE4GameOverlay(app=app, directory_main=str(directory_main))

    # timer to call the functions related to mouse and keyboard inputs
    timer = QTimer()
    timer.timeout.connect(window.timer_build_order_call)
    timer.timeout.connect(window.timer_mouse_keyboard_call)
    timer.setInterval(max(window.settings.call_ms, 50))
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
            last_civilization = None

            def poll_smart():
                global last_poll_error, last_civilization
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
                    if decision is not None and decision.civilization != last_civilization:
                        window.handle_detected_civilization(decision.civilization)
                        logger.info(
                            "Civilization locked: %s confidence=%.3f source=%s",
                            decision.civilization,
                            decision.confidence,
                            decision.source,
                        )
                        last_civilization = decision.civilization
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
