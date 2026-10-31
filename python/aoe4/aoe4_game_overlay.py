# AoE4 game overlay
from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from PyQt5.QtGui import QIcon, QFont
from PyQt5.QtCore import QSize, Qt

from common.useful_tools import widget_x_end, widget_y_end
from common.rts_overlay import RTSGameOverlay, scale_list_int, PanelID
from common.rts_overlay_images import RTSOverlayImages

from aoe4.aoe4_settings import AoE4OverlaySettings
from aoe4.aoe4_build_order import check_valid_aoe4_build_order
from aoe4.aoe4_civ_icon import aoe4_civilization_icon, get_aoe4_faction_selection
from aoe4.online_build_orders import Aoe4GuidesClient, OnlineBuildOrderError
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
        client = Aoe4GuidesClient()
        try:
            summaries = client.list_latest()
        except OnlineBuildOrderError as err:
            self._show_online_message(dialog, QMessageBox.Critical, "错误", f"获取在线建造顺序失败：{err}")
            return

        if not summaries:
            self._show_online_message(dialog, QMessageBox.Information, "提示", "没有可用的在线建造顺序。")
            return

        select_dialog = QDialog(dialog)
        select_dialog.setWindowTitle("选择在线建造顺序")
        select_dialog.setModal(True)
        select_dialog.resize(600, 400)

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

        list_widget = QListWidget(select_dialog)
        list_widget.setFont(font)
        list_widget.setStyleSheet(
            """
            QListWidget {
                background-color: black;
                color: white;
                border: 1px solid white;
            }
            """
        )
        for index, summary in enumerate(summaries):
            item = QListWidgetItem(f"{summary.title} | {summary.civilization} | {summary.author}")
            item.setData(Qt.UserRole, index)
            list_widget.addItem(item)
        list_widget.setCurrentRow(0)

        def on_download():
            item = list_widget.currentItem()
            if item is None:
                return
            summary = summaries[item.data(Qt.UserRole)]
            try:
                saved_path = client.save_overlay(summary.build_id, self.directory_build_orders)
            except OnlineBuildOrderError as err:
                self._show_online_message(
                    select_dialog, QMessageBox.Critical, "错误", f"下载建造顺序失败：{err}"
                )
                return
            self._show_online_message(
                select_dialog,
                QMessageBox.Information,
                "成功",
                f"建造顺序已保存为：{os.path.basename(saved_path)}",
            )
            select_dialog.accept()
            self.reload(update_settings=False)

        download_button = QPushButton("下载", select_dialog)
        download_button.setStyleSheet(button_style)
        download_button.setFont(font)
        download_button.clicked.connect(on_download)

        cancel_button = QPushButton("取消", select_dialog)
        cancel_button.setStyleSheet(button_style)
        cancel_button.setFont(font)
        cancel_button.clicked.connect(select_dialog.reject)

        layout = QVBoxLayout(select_dialog)
        layout.addWidget(list_widget)
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

        self.update_panel_elements()  # update the current panel elements

    def settings_scaling(self):
        """Apply the scaling on the settings."""
        super().settings_scaling()
        assert 0 <= self.scaling_input_selected_id < len(self.scaling_input_combo_ids)
        scaling = self.scaling_input_combo_ids[self.scaling_input_selected_id] / 100.0

        self.settings.layout.configuration.flag_select_size = scale_list_int(
            scaling, self.unscaled_settings.layout.configuration.flag_select_size
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
        civilization_id = self.civilization_select.currentIndex()
        assert 0 <= civilization_id < len(self.civilization_combo_ids)
        self.obtain_build_order_search(key_condition={'civilization': self.civilization_combo_ids[civilization_id]})
        self.config_panel_layout()

    def enter_key_actions(self):
        """Actions performed when pressing the Enter key."""
        if self.selected_panel == PanelID.CONFIG:
            if self.build_order_search.hasFocus():
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

        # build order search
        self.build_order_search.move(border_size, next_y)
        next_y += self.build_order_search.height() + vertical_spacing

        if widget_x_end(self.build_order_search) > widget_x_end(self.civilization_select):
            self.civilization_select.move(
                widget_x_end(self.build_order_search) - self.civilization_select.width(), self.civilization_select.y()
            )
        elif widget_x_end(self.build_order_search) < widget_x_end(self.civilization_select):
            self.build_order_search.resize(
                widget_x_end(self.civilization_select) - self.build_order_search.x(), self.build_order_search.height()
            )

        self.build_order_selection.update_size_position(init_y=next_y)

        self.config_panel_layout_resize_move()  # size and position

    def update_build_order(self):
        """Update the build order panel."""
        super().update_build_order()

        # valid build order selected
        if (self.selected_build_order is not None) and ('build_order' in self.selected_build_order):

            layout = self.settings.layout
            spacing = ' ' * layout.build_order.resource_spacing  # space between the elements

            # get selected steps and corresponding IDs
            selected_steps, selected_steps_ids = self.get_build_order_selected_steps_and_ids()

            # resource line
            images = self.images
            resource_step = selected_steps[selected_steps_ids[-1]]  # ID of the step to use to display the resources

            # target resources
            target_resources = resource_step['resources']
            target_food = target_resources['food']
            target_wood = target_resources['wood']
            target_gold = target_resources['gold']
            target_stone = target_resources['stone']
            target_builder = target_resources['builder'] if ('builder' in target_resources) else -1
            target_villager = resource_step['villager_count']
            target_population = resource_step['population_count']

            # line to display the target resources
            resources_line = images.food + '@ ' + (str(target_food) if (target_food >= 0) else ' ')
            resources_line += spacing + '@' + images.wood + '@ ' + (str(target_wood) if (target_wood >= 0) else ' ')
            resources_line += spacing + '@' + images.gold + '@ ' + (str(target_gold) if (target_gold >= 0) else ' ')
            resources_line += spacing + '@' + images.stone + '@ ' + (str(target_stone) if (target_stone >= 0) else ' ')
            if target_builder > 0:  # add builders count if indicated
                resources_line += spacing + '@' + images.builder + '@ ' + str(target_builder)
            if target_villager >= 0:
                resources_line += spacing + '@' + images.villager + '@ ' + str(target_villager)
            if target_population >= 0:
                resources_line += spacing + '@' + images.population + '@ ' + str(target_population)
            if 1 <= resource_step['age'] <= 4:
                resources_line += spacing + '@' + self.get_age_image(resource_step['age'])
            # add time if indicated
            if layout.show_time_resource and ('time' in resource_step) and (resource_step['time'] != ''):
                resources_line += '@' + spacing + '@' + self.images.time + '@' + resource_step['time']

            self.build_order_resources.add_row_from_picture_line(parent=self, line=str(resources_line))

            # update the notes of the build order
            self.update_build_order_notes(selected_steps, selected_steps_ids)

        self.build_order_panel_layout()  # update layout
