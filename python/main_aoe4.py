# Game overlay application for Age of Empires IV (AoE4)
import sys
import pathlib
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from aoe4.aoe4_game_overlay import AoE4GameOverlay
from aoe4.smart_controller import SmartHudController
from aoe4.smart_controller_config import SmartControllerConfig
from aoe4.smart_environment import EnvironmentValidator
from aoe4.smart_status_window import SmartStatusWindow

if __name__ == '__main__':
    app = QApplication(sys.argv)
    directory_main = pathlib.Path(__file__).parent.resolve()
    window = AoE4GameOverlay(app=app, directory_main=str(directory_main))

    # timer to call the functions related to mouse and keyboard inputs
    timer = QTimer()
    timer.timeout.connect(window.timer_build_order_call)
    timer.timeout.connect(window.timer_mouse_keyboard_call)
    timer.setInterval(window.settings.call_ms)
    timer.start()

    status_window = SmartStatusWindow()
    status_window.show()

    controller = None
    smart_timer = None

    config_path = directory_main / "smart_hud_config.json"

    report = EnvironmentValidator().validate(config_path)
    if not report.ready:
        status_window.show_disabled("\n".join(report.messages_zh))
    else:
        try:
            config = SmartControllerConfig.from_file(config_path)
            controller = SmartHudController(config)

            def poll_smart():
                result = controller.poll_safe()
                notifications = ()
                if result.result is not None:
                    notifications = window.apply_smart_pipeline_result(result.result)
                status_window.show_poll(result, notifications)

            smart_timer = QTimer()
            smart_timer.timeout.connect(poll_smart)
            smart_timer.setInterval(500)
            smart_timer.start()

            app.aboutToQuit.connect(controller.close)
        except Exception as exc:
            status_window.show_disabled(f"{type(exc).__name__}: {exc}")

    exit_event = app.exec()
    sys.exit(exit_event)
