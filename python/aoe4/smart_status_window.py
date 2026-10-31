from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget

from aoe4.smart_controller import SafePollResult

from aoe4.smart_advisor import AdvicePriority
from aoe4.smart_notifications import Notification

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

FIELD_LABELS = {
    "game_time": "游戏时间",
    "age": "时代",
    "food": "食物",
    "wood": "木材",
    "gold": "黄金",
    "stone": "石料",
    "population_current": "当前人口",
    "population_cap": "人口上限",
}

STATUS_LABELS = {
    "confirmed": "已确认",
    "inferred": "推断",
    "unknown": "未知",
    "stale": "已过期",
}

PRIORITY_LABELS = {
    AdvicePriority.HIGH: "高",
    AdvicePriority.MEDIUM: "中",
    AdvicePriority.LOW: "低",
}


class SmartStatusWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(flags=Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setWindowTitle("智能识别状态")
        self.label = QLabel(self)
        self._last_text = None

    def show_disabled(self, message: str) -> None:
        self._set_text(f"智能 OCR 已停用\n{message}")

    def show_poll(
        self, value: SafePollResult, notifications: tuple[Notification, ...] = ()
    ) -> None:
        if value.error:
            text = f"错误：{value.error}\n连续失败：{value.consecutive_failures}"
        else:
            state = value.result.state
            sampled = "已识别" if value.result.sampled else "等待下一次识别"
            lines = [sampled]
            for name in FIELDS:
                field = getattr(state, name)
                field_text = "?" if field.value is None else str(field.value)
                status = STATUS_LABELS.get(field.status.value, field.status.value)
                lines.append(f"{FIELD_LABELS[name]}：{field_text} [{status}]")
            if notifications:
                lines.append("经济提醒：")
                for item in notifications:
                    lines.append(f"{PRIORITY_LABELS[item.priority]}：{item.message}")
            text = "\n".join(lines)
        self._set_text(text)

    def _set_text(self, text: str) -> None:
        if text == self._last_text:
            return
        self._last_text = text
        self.label.setText(text)
        self._fit()

    def _fit(self) -> None:
        self.label.adjustSize()
        self.resize(self.label.size())

