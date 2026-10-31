from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget

from aoe4.smart_controller import SafePollResult


FIELDS = (
    "game_time",
    "age",
    "food",
    "wood",
    "gold",
    "stone",
    "population_current",
    "population_cap",
)


class SmartStatusWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(flags=Qt.Tool | Qt.WindowStaysOnTopHint)
        self.label = QLabel(self)

    def show_disabled(self, message: str) -> None:
        self.label.setText(f"Smart OCR disabled\n{message}")
        self._fit()

    def show_poll(self, value: SafePollResult) -> None:
        if value.error:
            self.label.setText(f"Error: {value.error}\nFailures: {value.consecutive_failures}")
        else:
            state = value.result.state
            sampled = "Sampled" if value.result.sampled else "Cadence-skipped"
            lines = [sampled]
            for name in FIELDS:
                field = getattr(state, name)
                text = "?" if field.value is None else str(field.value)
                lines.append(f"{name}: {text} [{field.status.value}]")
            self.label.setText("\n".join(lines))
        self._fit()

    def _fit(self) -> None:
        self.label.adjustSize()
        self.resize(self.label.size())

