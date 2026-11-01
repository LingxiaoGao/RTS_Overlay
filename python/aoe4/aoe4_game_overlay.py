# AoE4 game overlay
from __future__ import annotations

import os
import json
import html
import re

from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from PyQt5.QtGui import QBrush, QColor, QIcon, QFont
from PyQt5.QtCore import QSize, Qt

from common.useful_tools import widget_x_end, widget_y_end
from common.rts_overlay import RTSGameOverlay, scale_list_int, PanelID
from common.rts_overlay_images import RTSOverlayImages

from aoe4.aoe4_settings import AoE4OverlaySettings
from aoe4.aoe4_build_order import check_valid_aoe4_build_order
from aoe4.aoe4_civ_icon import aoe4_civilization_icon, get_aoe4_faction_selection
from aoe4.online_build_orders import (
    Aoe4GuidesClient,
    CIVILIZATION_NAMES_ZH,
    OnlineBuildOrderError,
)
from aoe4.smart_build_order_session import SmartBuildOrderSession, apply_schedule_times
from aoe4.smart_advisor import AdvisorEngine, EconomyTarget
from aoe4.smart_notifications import Notification, NotificationManager


class AoE4Images(RTSOverlayImages):
    """AoE4 images"""

    def __init__(self):
        """Constructor"""
        super().__init__()

        self.wood: str = 'resource/resource_wood.webp'  # wood resource
        self.food: str = 'resource/resource_food.webp'  # food resource
        self.gold: str = 'resource/resource_gold.webp'  # gold resource
        self.stone: str = 'resource/resource_stone.webp'  # stone resource
        self.builder: str = 'resource/repair.webp'  # builder icon
        self.population: str = 'building_economy/house.webp'  # population icon
        self.villager: str = 'unit_worker/villager.webp'  # villager icon
        self.age_unknown: str = 'age/age_unknown.webp'  # unknown age image
        self.age_1: str = 'age/age_1.webp'  # first age image (Dark Age)
        self.age_2: str = 'age/age_2.webp'  # second age image (Feudal Age)
        self.age_3: str = 'age/age_3.webp'  # third age image (Castle Age)
        self.age_4: str = 'age/age_4.webp'  # fourth age image (Imperial Age)


class AoE4GameOverlay(RTSGameOverlay):
    """Game overlay application for AoE4."""

    def __init__(self, app: QApplication, directory_main: str):
        """Constructor

        Parameters
        ----------
        app               Main application instance.
        directory_main    Directory where the main file is located.
        """
        super().__init__(
            app=app,
            directory_main=directory_main,
            name_game='aoe4',
            settings_name='aoe4_settings.json',
            images=AoE4Images,
            settings_class=AoE4OverlaySettings,
            check_valid_build_order=check_valid_aoe4_build_order,
            get_faction_selection=get_aoe4_faction_selection,
            build_order_category_name='civilization',
            build_order_timer_step_starting_flag=False,
        )

        # smart calibration session (age-anchor driven)
        self.smart_session = None
        # smart economy advice
        self.advisor = AdvisorEngine()
        self.notifications = NotificationManager()
        self.online_build_client = Aoe4GuidesClient(
            bundled_directory=self.directory_build_orders
        )

        self.build_order_table = QTableWidget(0, 2, self)
        self.build_order_table.setHorizontalHeaderLabels(["时间", "建造顺序"])
        self.build_order_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.build_order_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.build_order_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.build_order_table.verticalHeader().setVisible(False)
        self.build_order_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.build_order_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.build_order_table.setWordWrap(True)
        self._update_build_order_table_style()

        # civilization selection
        layout = self.settings.layout
        color_default = layout.color_default
        style_description = f'color: rgb({color_default[0]}, {color_default[1]}, {color_default[2]})'
        flag_select_size = layout.configuration.flag_select_size

        self.civilization_select = QComboBox(self)
        self.civilization_select.activated.connect(self.update_build_order_display)
        self.civilization_combo_ids = []  # corresponding IDs
        for civ_name, letters_icon in aoe4_civilization_icon.items():
            assert len(letters_icon) == 2
            self.civilization_select.addItem(
                QIcon(os.path.join(self.directory_game_pictures, 'civilization_flag', letters_icon[1])), letters_icon[0]
            )
            self.civilization_combo_ids.append(civ_name)
        self.civilization_select.setIconSize(QSize(flag_select_size[0], flag_select_size[1]))
        self.civilization_select.setStyleSheet(f'QWidget{{ {style_description} }};')
        self.civilization_select.setToolTip('选择文明')
        self.civilization_select.setFont(QFont(layout.font_police, layout.font_size))
        self.civilization_select.adjustSize()

        self._populate_civilization_build_orders()
        self.update_panel_elements()  # update the current panel elements

    def select_build_order(self, key_condition: dict = None):
        """Select a build order and (re)create the smart calibration session."""
        super().select_build_order(key_condition=key_condition)

        self.smart_session = None
        self.notifications.reset()
        if self.selected_build_order is None:
            return

        try:
            self.smart_session = SmartBuildOrderSession.from_build_order(self.selected_build_order)
        except ValueError as exc:
            self.smart_session = None
            print(f"智能校准元数据无效：{exc}")

    def apply_smart_pipeline_result(self, result) -> tuple[Notification, ...]:
        """Apply a pipeline result to the calibration session and produce advice.

        Parameters
        ----------
        result    Pipeline result to consume.

        Returns
        -------
        Notifications emitted for the active target step.
        """
        session = self.smart_session
        if (session is not None) and session.enabled:
            steps = self.build_order_timer['steps']
            if steps:
                update = session.consume(result)
                if update.changed:
                    self.build_order_timer['steps'] = apply_schedule_times(steps, update.times)
                    self.build_order_timer['last_steps_ids'] = []
                    if self.selected_panel == PanelID.BUILD_ORDER:
                        self.update_build_order()

        if self.selected_build_order is None:
            return ()

        steps = self.build_order_timer['steps']
        if not steps:
            return ()

        steps_ids = self.build_order_timer['steps_ids']
        valid_ids = [
            step_id for step_id in steps_ids
            if isinstance(step_id, int) and not isinstance(step_id, bool)
            and 0 <= step_id < len(steps)
        ]
        target_id = max(valid_ids) if valid_ids else 0
        if target_id >= len(steps):
            return ()
        step = steps[target_id]

        try:
            target = EconomyTarget.from_build_order_step(step)
        except (TypeError, ValueError) as exc:
            print(f"智能经济目标无效：{exc}")
            return ()

        advice = self.advisor.evaluate(result.state, target)
        return self.notifications.publish(advice, result.state.updated_at)

    def get_online_build_order_handler(self):
        """Return the AoE4 online build-order handler."""
        return self.download_online_build_orders

    def _show_online_message(self, dialog, icon, title, text):
        """Show a message from the online build-order workflow."""
        msg = QMessageBox(dialog)
        msg.setIcon(icon)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStyleSheet(
            """
            QMessageBox {
                background-color: black;
                color: white;
            }
            QMessageBox QLabel {
                background-color: black;
                color: white;
            }
            QMessageBox QPushButton {
                color: white;
                border: 1px solid white;
                padding: 4px;
                background-color: black;
            }
            """
        )
        msg.setFont(QFont(self.settings.layout.font_police, self.settings.layout.font_size))
        msg.exec_()

    def download_online_build_orders(self, dialog):
        """Fetch, select, and download an AoE4 Guides build order."""
        client = self.online_build_client
        try:
            summaries = client.list_popular_recent()
        except OnlineBuildOrderError as err:
            self._show_online_message(dialog, QMessageBox.Critical, "错误", f"获取在线建造顺序失败：{err}")
            return

        if not summaries:
            self._show_online_message(dialog, QMessageBox.Information, "提示", "没有可用的在线建造顺序。")
            return

        select_dialog = QDialog(dialog)
        select_dialog.setWindowTitle("选择在线建造顺序")
        select_dialog.setModal(True)
        select_dialog.setMinimumSize(640, 400)
        select_dialog.resize(900, 600)
        select_dialog.setSizeGripEnabled(True)
        select_dialog.setWindowFlags(
            select_dialog.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )

        font = QFont(self.settings.layout.font_police, self.settings.layout.font_size)
        button_style = """
            QPushButton {
                color: white;
                border: 1px solid white;
                padding: 4px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 30);
            }
        """

        table_widget = QTableWidget(len(summaries), 3, select_dialog)
        table_widget.setHorizontalHeaderLabels(["文明", "更新日期", "攻略名称"])
        table_widget.setFont(font)
        table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        table_widget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table_widget.verticalHeader().setVisible(False)
        header = table_widget.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        table_widget.setStyleSheet(
            """
            QTableWidget {
                background-color: black;
                color: white;
                border: 1px solid white;
            }
            QHeaderView::section {
                background-color: black;
                color: white;
                border: 1px solid white;
            }
            """
        )
        for row, summary in enumerate(summaries):
            values = (
                CIVILIZATION_NAMES_ZH.get(summary.civilization, summary.civilization),
                summary.updated.strftime('%Y-%m-%d'),
                summary.title,
            )
            for column, value in enumerate(values):
                table_widget.setItem(row, column, QTableWidgetItem(value))
        table_widget.selectRow(0)

        def on_download():
            row = table_widget.currentRow()
            if row < 0 or row >= len(summaries):
                return
            summary = summaries[row]
            try:
                saved_path = client.save_overlay(summary.build_id, self.directory_build_orders)
            except OnlineBuildOrderError as err:
                self._show_online_message(
                    select_dialog, QMessageBox.Critical, "错误", f"下载建造顺序失败：{err}"
                )
                return
            try:
                with open(saved_path, 'r', encoding='utf-8') as handle:
                    saved_build_order = json.load(handle)
                saved_name = saved_build_order['name']
                saved_civilization = saved_build_order['civilization']
            except (OSError, KeyError, TypeError, ValueError) as err:
                self._show_online_message(
                    select_dialog, QMessageBox.Critical, "错误", f"读取已保存建造顺序失败：{err}"
                )
                return

            self.reload(update_settings=False)
            if saved_civilization in self.civilization_combo_ids:
                self.civilization_select.setCurrentIndex(self.civilization_combo_ids.index(saved_civilization))
            self._populate_civilization_build_orders()
            if saved_name not in self.valid_build_orders:
                self._show_online_message(
                    select_dialog, QMessageBox.Critical, "错误", "已保存的建造顺序未出现在当前文明列表中。"
                )
                return

            self.build_order_selection_id = self.valid_build_orders.index(saved_name)
            self.select_build_order(
                key_condition={'civilization': self.civilization_combo_ids[self.civilization_select.currentIndex()]}
            )
            select_dialog.accept()
            dialog.accept()
            self.selected_panel = PanelID.BUILD_ORDER
            self.update_panel_elements()

        download_button = QPushButton("部署攻略", select_dialog)
        download_button.setStyleSheet(button_style)
        download_button.setFont(font)
        download_button.clicked.connect(on_download)

        cancel_button = QPushButton("取消", select_dialog)
        cancel_button.setStyleSheet(button_style)
        cancel_button.setFont(font)
        cancel_button.clicked.connect(select_dialog.reject)

        layout = QVBoxLayout(select_dialog)
        layout.addWidget(table_widget)
        button_layout = QHBoxLayout()
        button_layout.addWidget(download_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        select_dialog.exec_()

    def reload(self, update_settings):
        """Reload the application settings, build orders...

        Parameters
        ----------
        update_settings   True to update (reload) the settings, False to keep the current ones.
        """
        super().reload(update_settings=update_settings)

        # civilization selection
        layout = self.settings.layout
        color_default = layout.color_default
        style_description = f'color: rgb({color_default[0]}, {color_default[1]}, {color_default[2]})'
        flag_select_size = layout.configuration.flag_select_size

        self.civilization_select.setIconSize(QSize(flag_select_size[0], flag_select_size[1]))
        self.civilization_select.setStyleSheet(f'QWidget{{ {style_description} }};')
        self.civilization_select.setFont(QFont(layout.font_police, layout.font_size))
        self.civilization_select.adjustSize()
        self._update_build_order_table_style()

        self._populate_civilization_build_orders()
        self.update_panel_elements()  # update the current panel elements

    def settings_scaling(self):
        """Apply the scaling on the settings."""
        super().settings_scaling()
        assert 0 <= self.scaling_input_selected_id < len(self.scaling_input_combo_ids)
        scaling = self.scaling_input_combo_ids[self.scaling_input_selected_id] / 100.0

        self.settings.layout.configuration.flag_select_size = scale_list_int(
            scaling, self.unscaled_settings.layout.configuration.flag_select_size
        )

    def _update_build_order_table_style(self):
        """Apply the current overlay font and colors to the step table."""
        layout = self.settings.layout
        self.build_order_table.setFont(QFont(layout.font_police, layout.font_size))
        self.build_order_table.setStyleSheet(
            "QTableWidget { background-color: black; color: white; border: 1px solid white; }"
            "QHeaderView::section { background-color: black; color: white; border: 1px solid white; }"
        )

    def select_build_order_id(self, build_order_id: int = -1) -> bool:
        """Select build order ID.

        Parameters
        ----------
        build_order_id    ID of the build order, negative to select next build order.

        Returns
        -------
        True if valid build order selection.
        """
        if self.selected_panel == PanelID.CONFIG:
            if super().select_build_order_id(build_order_id):
                civilization_id = self.civilization_select.currentIndex()
                assert 0 <= civilization_id < len(self.civilization_combo_ids)
                self.obtain_build_order_search(
                    key_condition={'civilization': self.civilization_combo_ids[civilization_id]}
                )
                if build_order_id >= 0:  # directly select in case of clicking
                    self.select_build_order(
                        key_condition={
                            'civilization': self.civilization_combo_ids[self.civilization_select.currentIndex()]
                        }
                    )
                self.config_panel_layout()
                return True
        return False

    def hide_elements(self):
        """Hide elements."""
        super().hide_elements()

        self.civilization_select.hide()
        if hasattr(self, 'build_order_table'):
            self.build_order_table.hide()

    def get_age_image(self, age_id: int) -> str:
        """Get the image for a requested age.

        Parameters
        ----------
        age_id    ID of the age.

        Returns
        -------
        age image with path
        """
        if age_id == 1:
            return self.images.age_1
        elif age_id == 2:
            return self.images.age_2
        elif age_id == 3:
            return self.images.age_3
        elif age_id == 4:
            return self.images.age_4
        else:
            return self.images.age_unknown

    def update_build_order_display(self):
        """Update the build order search matching display."""
        self._populate_civilization_build_orders()
        self.config_panel_layout()

    def _populate_civilization_build_orders(self):
        """Display all guides for the selected civilization without a search field."""
        civilization_id = self.civilization_select.currentIndex()
        assert 0 <= civilization_id < len(self.civilization_combo_ids)
        signals_blocked = self.build_order_search.blockSignals(True)
        self.build_order_search.setText(' ')
        self.obtain_build_order_search(
            key_condition={'civilization': self.civilization_combo_ids[civilization_id]}
        )
        self.build_order_search.setText('')
        self.build_order_search.blockSignals(signals_blocked)
        self.build_order_search.hide()

    def enter_key_actions(self):
        """Actions performed when pressing the Enter key."""
        if self.selected_panel == PanelID.CONFIG:
            self.select_build_order(
                key_condition={'civilization': self.civilization_combo_ids[self.civilization_select.currentIndex()]}
            )

            self.config_panel_layout()  # update layout

    def config_panel_layout(self):
        """Layout of the configuration panel."""
        super().config_panel_layout()
        if self.selected_panel != PanelID.CONFIG:
            return

        # show elements
        self.civilization_select.show()

        layout = self.settings.layout
        border_size = layout.border_size
        horizontal_spacing = layout.horizontal_spacing
        vertical_spacing = layout.vertical_spacing
        action_button_size = layout.action_button_size

        # next Y position
        next_y = (
            border_size
            + max(action_button_size, self.font_size_input.height(), self.scaling_input.height())
            + vertical_spacing
        )

        # build order selection
        self.build_order_title.move(border_size, next_y)
        next_x = border_size + self.build_order_title.width() + horizontal_spacing

        # civilization selection
        self.civilization_select.move(next_x, next_y)

        if self.civilization_select.height() > self.build_order_title.height():
            self.build_order_title.move(
                self.build_order_title.x(), widget_y_end(self.civilization_select) - self.build_order_title.height()
            )
        next_y += max(self.build_order_title.height(), self.civilization_select.height()) + vertical_spacing

        self.build_order_search.hide()
        self.build_order_selection.update_size_position(init_y=next_y)

        max_x = max(
            self.next_panel_button.x_end(),
            widget_x_end(self.civilization_select),
            self.build_order_selection.x() + self.build_order_selection.row_max_width,
        )
        max_y = self.build_order_selection.y() + self.build_order_selection.row_total_height
        self.resize(max_x + border_size, max_y + border_size)
        self.next_panel_button.move(self.width() - border_size - self.next_panel_button.width(), border_size)
        self.hide_panel_button.move(
            self.next_panel_button.x() - horizontal_spacing - self.hide_panel_button.width(),
            self.next_panel_button.y(),
        )
        self.update_position()

    def update_build_order(self):
        """Update the build order panel."""
        super().update_build_order()
        self.build_order_resources.hide()
        self.build_order_notes.hide()
        self.build_order_table.clearContents()

        if self.selected_build_order is None or 'build_order' not in self.selected_build_order:
            self.build_order_table.setRowCount(1)
            self.build_order_table.setItem(0, 0, QTableWidgetItem(''))
            self.build_order_table.setItem(0, 1, QTableWidgetItem('未选择建造顺序。'))
        else:
            timer_mode = self.build_order_timer['use_timer'] and self.build_order_timer['steps']
            steps = self.build_order_timer['steps'] if timer_mode else self.selected_build_order['build_order']
            current_row = (
                max(self.build_order_timer['steps_ids'], default=0)
                if timer_mode
                else self.selected_build_order_step_id
            )
            self.build_order_table.setRowCount(len(steps))
            emphasis = QColor(*self.settings.layout.build_order.color_row_emphasis)
            bright = QColor(*self.settings.layout.color_default)
            muted = QColor(145, 145, 145)

            for row, step in enumerate(steps):
                time_item = QTableWidgetItem(str(step.get('time', '')))
                notes = '  |  '.join(self._readable_build_order_note(note) for note in step['notes'])
                note_item = QTableWidgetItem(notes)
                foreground = bright if row == current_row else muted
                for item in (time_item, note_item):
                    item.setForeground(QBrush(foreground))
                    if row == current_row:
                        item.setBackground(QBrush(emphasis))
                self.build_order_table.setItem(row, 0, time_item)
                self.build_order_table.setItem(row, 1, note_item)

            self.build_order_table.resizeRowsToContents()
            self.build_order_table.selectRow(current_row)
            self.build_order_table.scrollToItem(
                self.build_order_table.item(current_row, 0), QAbstractItemView.PositionAtCenter
            )

        self.build_order_panel_layout()

    @staticmethod
    def _readable_build_order_note(note):
        """Turn inline picture markers into compact readable labels."""
        def replace_picture(match):
            return os.path.splitext(os.path.basename(match.group(1)))[0].replace('-', ' ')

        return html.unescape(re.sub(r'@([^@]+\.(?:webp|png|jpe?g))@', replace_picture, note))

    def build_order_panel_layout(self):
        """Lay out the complete AoE4 build order table."""
        if self.selected_panel != PanelID.BUILD_ORDER:
            return

        layout = self.settings.layout
        border_size = layout.border_size
        vertical_spacing = layout.vertical_spacing
        action_button_size = layout.action_button_size
        action_button_spacing = layout.action_button_spacing
        scale = max(1.0, layout.font_size / 11.0)

        self.next_panel_button.show()
        self.hide_panel_button.show()
        self.build_order_previous_button.hide()
        self.build_order_next_button.hide()
        self.build_order_resources.hide()
        self.build_order_notes.hide()
        self.build_order_table.show()

        timer_controls = []
        if self.selected_build_order is not None:
            self.build_order_step_time.show()
            self.build_order_step_time.adjustSize()
            if self.build_order_timer['available'] and self.build_order_timer['steps']:
                self.build_order_switch_timer_manual.show()
                timer_controls.append(self.build_order_switch_timer_manual)
                if self.build_order_timer['use_timer']:
                    self.build_order_start_stop_timer.show()
                    self.build_order_reset_timer.show()
                    timer_controls.extend([self.build_order_start_stop_timer, self.build_order_reset_timer])

        table_width = int(720 * scale)
        table_height = int(420 * scale)
        next_y = border_size + action_button_size + vertical_spacing
        self.build_order_table.setGeometry(border_size, next_y, table_width, table_height)
        self.resize(table_width + 2 * border_size, next_y + table_height + border_size)

        controls = []
        if self.selected_build_order is not None:
            controls.append(self.build_order_step_time)
            controls.extend(timer_controls)
        controls.extend([self.hide_panel_button, self.next_panel_button])

        if layout.overlay_on_right_side:
            next_x = self.width() - border_size
            for control in reversed(controls):
                next_x -= control.width()
                control.move(next_x, border_size)
                next_x -= action_button_spacing
        else:
            next_x = border_size
            for control in controls:
                control.move(next_x, border_size)
                next_x += control.width() + action_button_spacing

        self.update_position()
