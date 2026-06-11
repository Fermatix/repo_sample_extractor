import pathlib

import mujoco
import numpy as np
import os
import qtawesome as qta
from PySide6.QtCore import QSettings, Qt, Signal, Slot, QSize
from PySide6.QtGui import QGuiApplication, QFont
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QLabel, QLayout, QLineEdit,
    QMainWindow, QMessageBox, QPushButton,
    QSlider, QVBoxLayout, QWidget
)

from so100_mujoco_sim.arm_control import (
    Joint,
    joints_from_model,
    PlaybackRecordState
)
from so100_mujoco_sim.mujoco_viewport import Viewport
from so100_mujoco_sim.update_thread import UpdateThread


class JointWidget(QWidget):
    """
    Wraps up the joint name and slider in a single widget.
    Two position values are displayed, the set position and
    the actual position from the simulation.
    """

    joint_position_changed = Signal(Joint, float)

    def __init__(self, joint: Joint) -> None:
        super().__init__()
        self.joint = joint
        self.actual_position: float = 0.0
        layout = QVBoxLayout()
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        name_label = QLabel(self.joint.name)
        self.actual_value_label = QLabel("0.0")
        self.actual_value_label.setStyleSheet("color: #888")
        self.actual_value_label.setMinimumWidth(40)
        self.value_label = QLabel("0.0")
        self.value_label.setStyleSheet("color: #cccccc")
        self.value_label.setMinimumWidth(40)
        label_layout = QHBoxLayout()
        label_layout.addWidget(name_label)
        label_layout.addStretch()
        
        label_layout.addWidget(self.value_label)
        label_layout.addWidget(self.actual_value_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(self.joint.range[0] * 1000)
        self.slider.setMaximum(self.joint.range[1] * 1000)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self._changed)
        layout.addLayout(label_layout)
        layout.addWidget(self.slider)
        self.setLayout(layout)

    def setValue(self, val: float):
        self.slider.setValue(val * 1000)

    def _changed(self, value: int) -> None:
        self.value_label.setText("{:.2f}".format(value / 1000.0))
        self.joint_position_changed.emit(self.joint, value / 1000.0)

    def set_actual_position(self, position: float) -> None:
        self.actual_value_label.setText("{:.2f}".format(position))


class Window(QMainWindow):
    """
    Main window for the application
    """

    def __init__(self) -> None:
        super().__init__()

        # Initialize QSettings for saving and restoring values
        self.settings = QSettings("lh", "So100MujocoSim")

        self.model = mujoco.MjModel.from_xml_path(str(pathlib.Path(__file__).parent.joinpath('xml/sim_scene.xml')))
        self.joints = joints_from_model(self.model)
        self.data = mujoco.MjData(self.model)
        self.cam = self.create_free_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = True
        self.viewport = Viewport(self.model, self.data, self.cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())

        layout = QHBoxLayout()
        layout.setSpacing(0)
        w = QWidget()
        w.setLayout(layout)
        self.setCentralWidget(w)

        self.th = UpdateThread(self.model, self.data, self)
        self.th.update_ui_joint_values.connect(self._update_ui_joint_values)
        self.th.update_controller_enabled_states.connect(self._update_controllers_enabled)
        self.th.update_primary_controller.connect(self._primary_controller_changed)
        self.th.update_ui_recorded_steps.connect(self._update_playback_recorded_steps)
        self.th.warning.connect(self.show_warning_dialog)

        layout_right_side = QVBoxLayout()
        layout_right_side.setSpacing(8)
        layout_robot_controls = QVBoxLayout()
        self.joint_widgets: list[JointWidget] = []
        layout_robot_controls.addLayout(self.create_right_side_control())
        layout_right_side.addLayout(layout_robot_controls)
        layout_right_side.setContentsMargins(8,8,8,8)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        layout.addLayout(layout_right_side)
        layout.setContentsMargins(0,0,0,0)
        layout.setStretch(0,1)

        self.resize(900, 600)

        self.th.start()

        # Restore saved settings
        self.restore_settings()

        self.statusBar().showMessage(
            f"Ready",
            1000
        )

    @Slot(list)
    def _update_ui_joint_values(self, joint_vals: list):
        for i, jw in enumerate(self.joint_widgets):
            jw.setValue(joint_vals[i])

    def _set_enable_ui_joint_controls(self, enabled: bool):
        for jw in self.joint_widgets:
            jw.slider.setEnabled(enabled)

    @Slot(str)
    def _primary_controller_changed(self, name: str):
        # disable or enable the joint controls depending on if the UI controller
        # is primary
        # should probably get this name from the controller itself
        self._set_enable_ui_joint_controls(name == "User Interface")

        self.statusBar().showMessage(
            f"Now controlling with {name}",
            2000
        )

    def _create_config_group(self) -> QGroupBox:
        # Add the Config group box
        config_layout = QVBoxLayout()
        config_layout.setSpacing(8)

        # Calibration folder selection
        calibration_layout_v = QVBoxLayout()
        calibration_layout_v.setSpacing(0)
        calibration_layout_v.addWidget(QLabel("LeRobot Calibration Folder:"))
        calibration_layout = QHBoxLayout()
        calibration_layout.setSpacing(4)
        self.calibration_folder_edit = QLineEdit()
        self.calibration_folder_edit.setPlaceholderText("Select folder...")
        calibration_folder_open_icon = qta.icon("fa6.folder-open")
        calibration_button = QPushButton(calibration_folder_open_icon, "")
        calibration_button.clicked.connect(self._select_calibration_folder)
        calibration_layout.addWidget(self.calibration_folder_edit)
        calibration_layout.addWidget(calibration_button)
        calibration_layout_v.addLayout(calibration_layout)

        # USB port field
        usb_port_layout = QVBoxLayout()
        usb_port_layout.setSpacing(0)
        usb_port_label = QLabel("USB Port:")
        usb_port_layout.addWidget(usb_port_label)
        self.usb_port_edit = QLineEdit()
        self.usb_port_edit.setPlaceholderText("Enter USB port...")
        usb_port_layout.addWidget(self.usb_port_edit)
        config_layout.addLayout(usb_port_layout)
        config_layout.addLayout(calibration_layout_v)

        config_group = QGroupBox("Config")
        config_group.setLayout(config_layout)
        return config_group

    def _create_sac_model_group(self) -> QGroupBox:
        """Creates a group box for SAC model loading"""
        # Create a group layout
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # SAC model path
        model_layout_v = QVBoxLayout()
        model_layout_v.setSpacing(0)
        model_layout_v.addWidget(QLabel("SAC Model File:"))
        model_layout = QHBoxLayout()
        model_layout.setSpacing(4)
        self.sac_model_path_edit = QLineEdit()
        self.sac_model_path_edit.setPlaceholderText("Select SAC model file (.zip)...")
        model_file_open_icon = qta.icon("fa6.folder-open")
        model_button = QPushButton(model_file_open_icon, "")
        model_button.clicked.connect(self._select_sac_model_file)
        model_layout.addWidget(self.sac_model_path_edit)
        model_layout.addWidget(model_button)
        model_layout_v.addLayout(model_layout)
        layout.addLayout(model_layout_v)
        
        # Normalization stats path (optional)
        norm_layout_v = QVBoxLayout()
        norm_layout_v.setSpacing(0)
        norm_layout_v.addWidget(QLabel("Normalization Stats File (Optional):"))
        norm_layout = QHBoxLayout()
        norm_layout.setSpacing(4)
        self.norm_stats_path_edit = QLineEdit()
        self.norm_stats_path_edit.setPlaceholderText("Select normalization stats file (.pkl)...")
        norm_file_open_icon = qta.icon("fa6.folder-open")
        norm_button = QPushButton(norm_file_open_icon, "")
        norm_button.clicked.connect(self._select_norm_stats_file)
        norm_layout.addWidget(self.norm_stats_path_edit)
        norm_layout.addWidget(norm_button)
        norm_layout_v.addLayout(norm_layout)
        layout.addLayout(norm_layout_v)
        
        # Add target position controls
        target_layout = QVBoxLayout()
        target_layout.setSpacing(0)
        target_layout.addWidget(QLabel("Target Position (X, Y, Z):"))
        target_controls = QHBoxLayout()
        target_controls.setSpacing(4)
        
        # X position
        x_layout = QVBoxLayout()
        x_layout.setSpacing(0)
        x_layout.addWidget(QLabel("X:"))
        self.target_x_slider = QSlider(Qt.Horizontal)
        self.target_x_slider.setMinimum(-300)  # -0.3 meters
        self.target_x_slider.setMaximum(300)   # 0.3 meters
        self.target_x_slider.setValue(0)
        self.target_x_slider.valueChanged.connect(self._update_target_position)
        self.target_x_value = QLabel("0.0")
        self.target_x_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        x_layout.addWidget(self.target_x_slider)
        x_layout.addWidget(self.target_x_value)
        
        # Y position
        y_layout = QVBoxLayout()
        y_layout.setSpacing(0)
        y_layout.addWidget(QLabel("Y:"))
        self.target_y_slider = QSlider(Qt.Horizontal)
        self.target_y_slider.setMinimum(-300)  # -0.3 meters
        self.target_y_slider.setMaximum(300)   # 0.3 meters
        self.target_y_slider.setValue(0)
        self.target_y_slider.valueChanged.connect(self._update_target_position)
        self.target_y_value = QLabel("0.0")
        self.target_y_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        y_layout.addWidget(self.target_y_slider)
        y_layout.addWidget(self.target_y_value)
        
        # Z position
        z_layout = QVBoxLayout()
        z_layout.setSpacing(0)
        z_layout.addWidget(QLabel("Z:"))
        self.target_z_slider = QSlider(Qt.Horizontal)
        self.target_z_slider.setMinimum(0)     # 0 meters (floor)
        self.target_z_slider.setMaximum(300)   # 0.3 meters
        self.target_z_slider.setValue(100)      # Default 0.1m above floor
        self.target_z_slider.valueChanged.connect(self._update_target_position)
        self.target_z_value = QLabel("0.1")
        self.target_z_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        z_layout.addWidget(self.target_z_slider)
        z_layout.addWidget(self.target_z_value)
        
        target_controls.addLayout(x_layout)
        target_controls.addLayout(y_layout)
        target_controls.addLayout(z_layout)
        target_layout.addLayout(target_controls)
        layout.addLayout(target_layout)
        
        # Load model button
        load_button_layout = QHBoxLayout()
        self.load_sac_model_button = QPushButton("Load SAC Model")
        self.load_sac_model_button.clicked.connect(self._load_sac_model)
        load_button_layout.addWidget(self.load_sac_model_button)
        load_button_layout.addStretch()
        layout.addLayout(load_button_layout)
        
        # Create and return the group box
        group = QGroupBox("SAC Model")
        group.setLayout(layout)
        return group

    def _create_robot_control_group(self) -> QGroupBox:
        control_layout = QVBoxLayout()
        # Add the Robot Control group box
        for joint in self.joints:
            widget = JointWidget(joint)
            widget.joint_position_changed.connect(self._joint_position_changed)
            control_layout.addWidget(widget)
            self.joint_widgets.append(widget)
        control_layout.addStretch()

        # reset_button = QPushButton("Reset")
        # reset_button.clicked.connect(self.reset_simulation)
        # control_layout.addWidget(reset_button)

        robot_control_group = QGroupBox("Robot Control")
        robot_control_group.setLayout(control_layout)
        robot_control_group.setMinimumWidth(300)

        return robot_control_group

    def _create_playback_and_record_group(self) -> QGroupBox:
        # Add the Playback and Record group box
        layout = QVBoxLayout()
        layout.setSpacing(8)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(4)
        record_icon = qta.icon('mdi.record', options=[{'color': 'red'}])
        self.record_button = QPushButton(record_icon, "")
        self.record_button.setIconSize(QSize(38, 38))
        self.record_button.clicked.connect(
            lambda: self._playback_record_button_clicked(PlaybackRecordState.RECORDING)
        )

        play_icon = qta.icon('mdi.play', options=[{'color': 'green'}])
        self.play_button = QPushButton(play_icon,"")
        self.play_button.setIconSize(QSize(38, 38))
        self.play_button.clicked.connect(
            lambda: self._playback_record_button_clicked(PlaybackRecordState.PLAYING)
        )

        stop_icon = qta.icon('mdi.stop')
        self.stop_button = QPushButton(stop_icon, "")
        self.stop_button.setIconSize(QSize(38, 38))
        self.stop_button.clicked.connect(
            lambda: self._playback_record_button_clicked(PlaybackRecordState.STOPPED)
        )

        steps_layout = QVBoxLayout()
        steps_layout.setSpacing(0)
        steps_layout.addWidget(QLabel("Step:"))
        self.playback_steps_edit = QLineEdit()
        self.playback_steps_edit.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.playback_steps_edit.setMaximumWidth(100)
        self.playback_steps_edit.setEnabled(False)
        self.playback_steps_edit.setText("0/0")
        text_font = QFont("monospace")
        text_font.setStyleHint(QFont.StyleHint.Monospace) 
        self.playback_steps_edit.setFont(text_font)
        steps_layout.addWidget(self.playback_steps_edit)

        controls_layout.addWidget(self.record_button)
        controls_layout.addWidget(self.play_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addStretch()
        controls_layout.addLayout(steps_layout)
        layout.addLayout(controls_layout)

        playback_file_layout = QVBoxLayout()
        playback_file_layout.setSpacing(0)
        playback_file_layout.addWidget(QLabel("Playback File:"))
        playback_file_edit_layout = QHBoxLayout()
        playback_file_edit_layout.setSpacing(4)

        self.playback_file_load = QPushButton("Load")
        self.playback_file_load.setMaximumWidth(50)
        self.playback_file_load.clicked.connect(self._load_playback_file)
        self.playback_file_save = QPushButton("Save")
        self.playback_file_save.setMaximumWidth(50)
        self.playback_file_save.clicked.connect(self._save_playback_file)

        self.playback_file_edit = QLineEdit()
        self.playback_file_edit.setPlaceholderText("Select file...")
        playback_file_open_icon = qta.icon("fa6.folder-open")
        playback_file_button = QPushButton(playback_file_open_icon, "")
        playback_file_button.clicked.connect(self._select_playback_file)
        
        playback_file_edit_layout.addWidget(self.playback_file_load)
        playback_file_edit_layout.addWidget(self.playback_file_save)
        playback_file_edit_layout.addWidget(self.playback_file_edit)
        playback_file_edit_layout.addWidget(playback_file_button)
        playback_file_edit_layout.setStretch(2,1)
        playback_file_layout.addLayout(playback_file_edit_layout)
        layout.addLayout(playback_file_layout)

        self._update_playback_record_buttons(PlaybackRecordState.STOPPED)

        group = QGroupBox("Playback and Record")
        group.setLayout(layout)
        return group

    def _save_playback_file(self):
        file_path = self.playback_file_edit.text()
        if file_path:
            self.th.save_playback_file(file_path)

            self.statusBar().showMessage(
                f"Playback file '{os.path.basename(file_path)}' saved successfully.",
                2000
            )

    def _load_playback_file(self):
        file_path = self.playback_file_edit.text()
        if file_path:
            if not pathlib.Path(file_path).is_file():
                self.show_warning_dialog(
                    "File Not Found",
                    f"The file '{file_path}' does not exist. Please select a valid file."
                )
                return
            self.th.load_playback_file(file_path)
            self.statusBar().showMessage(
                f"Playback file '{os.path.basename(file_path)}' loaded successfully.",
                2000
            )
        self.th.set_playback_record_state(PlaybackRecordState.STOPPED)

    def _update_playback_record_buttons(self, state: PlaybackRecordState) -> None:
        if state == PlaybackRecordState.RECORDING:
            self.record_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        elif state == PlaybackRecordState.PLAYING:
            self.record_button.setEnabled(False)
            self.play_button.setEnabled(False)
            self.stop_button.setEnabled(True)
        else:
            self.record_button.setEnabled(True)
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(False)

    def _playback_record_button_clicked(self, state: PlaybackRecordState) -> None:
        self._update_playback_record_buttons(state)
        self.th.set_playback_record_state(state)
        if state == PlaybackRecordState.PLAYING:
            playback_record_controller_index = self.controller_dropdown.findText("Playback/Record")
            self.controller_dropdown.setCurrentIndex(playback_record_controller_index)

    def _update_playback_recorded_steps(self, steps: int, current_step: int) -> None:
        self.playback_steps_edit.setText(f"{current_step}/{steps}")

    def _select_playback_file(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select or Create Playback File",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            self.playback_file_edit.setText(file_path)

    def create_right_side_control(self) -> QLayout:
        self.connect_button = QPushButton("Connect")
        self.connect_button.clicked.connect(self._connect_robot)

        # Dropdown for selecting the primary controller
        controller_layout = QHBoxLayout()
        controller_layout.setSpacing(4)
        controller_layout.addWidget(QLabel("Control with:"))
        self.controller_dropdown = QComboBox()
        self.controller_dropdown.addItems(self.th.get_controller_names())
        self.controller_dropdown.setCurrentIndex(self.th.get_primary_controller_index())
        self.controller_dropdown.currentIndexChanged.connect(self._set_primary_controller)
        self._update_controllers_enabled()
        controller_layout.addWidget(self.controller_dropdown)
        controller_layout.setStretch(1,1)

        robot_control_layout = QVBoxLayout()
        robot_control_layout.setContentsMargins(0,0,0,0)
        robot_control_layout.addWidget(self._create_robot_control_group())

        layout = QVBoxLayout()
        layout.setSpacing(12)
        layout.addWidget(self._create_config_group())
        layout.addWidget(self.connect_button)
        layout.addLayout(controller_layout)
        layout.addLayout(robot_control_layout)
        layout.addWidget(self._create_playback_and_record_group())
        layout.addWidget(self._create_sac_model_group())
        return layout

    def _update_controllers_enabled(self):
        controllable_controllers = self.th.get_controllable_controllers()
        for i, c in enumerate(controllable_controllers):
            self.controller_dropdown.model().item(i).setEnabled(c)

        for i, c in enumerate(self.th.get_controller_names()):
            if c == "Robot" and controllable_controllers[i]:
                # the robot controller is enabled, therefore the robot is connected
                # and we should disable the connect button (connecting twice breaks
                # things)
                self.connect_button.setText("Connected")
                self.connect_button.setEnabled(False)

                self.statusBar().showMessage(
                    f"so100 connected",
                    2000
                )
            elif c == "SAC Model" and controllable_controllers[i]:
                # SAC model is loaded, update the button state
                self.load_sac_model_button.setText("Model Loaded")
                self.load_sac_model_button.setEnabled(False)
                
                self.statusBar().showMessage(
                    "SAC model loaded successfully",
                    2000
                )

    def _select_calibration_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Calibration Folder")
        if folder_path:
            self.calibration_folder_edit.setText(folder_path)

    def _set_primary_controller(self, index: int):
        """
        Sets the selected ArmController as the primary controller.
        """
        self.th.set_primary_controller_index(index)

    def restore_settings(self):
        """Restore saved settings for calibration folder and USB port."""
        calibration_folder = self.settings.value("calibration_folder", "")
        usb_port = self.settings.value("usb_port", "")
        playback_file = self.settings.value("playback_file", "")
        sac_model_path = self.settings.value("sac_model_path", "")
        norm_stats_path = self.settings.value("norm_stats_path", "")
        
        # Restore target position settings
        target_x = self.settings.value("target_x", 0, type=int)
        target_y = self.settings.value("target_y", 0, type=int)
        target_z = self.settings.value("target_z", 100, type=int)

        self.calibration_folder_edit.setText(calibration_folder)
        self.usb_port_edit.setText(usb_port)
        self.playback_file_edit.setText(playback_file)
        self.sac_model_path_edit.setText(sac_model_path)
        self.norm_stats_path_edit.setText(norm_stats_path)
        
        # Set the target position sliders and update the display
        self.target_x_slider.setValue(target_x)
        self.target_y_slider.setValue(target_y)
        self.target_z_slider.setValue(target_z)
        
        # Update the target position in the SAC controller
        self._update_target_position()

    def closeEvent(self, event):
        """Save settings when the application is closed."""
        self.settings.setValue("calibration_folder", self.calibration_folder_edit.text())
        self.settings.setValue("usb_port", self.usb_port_edit.text())
        self.settings.setValue("playback_file", self.playback_file_edit.text())
        self.settings.setValue("sac_model_path", self.sac_model_path_edit.text())
        self.settings.setValue("norm_stats_path", self.norm_stats_path_edit.text())
        
        # Save target position values
        self.settings.setValue("target_x", self.target_x_slider.value())
        self.settings.setValue("target_y", self.target_y_slider.value())
        self.settings.setValue("target_z", self.target_z_slider.value())
        
        super().closeEvent(event)

    def _joint_position_changed(self, joint: Joint, position: float) -> None:
        self.th.set_joint_position(joint.name, position)

    def show_warning_dialog(self, title: str, message: str) -> None:
        """
        Displays a warning dialog with the given title and message.

        :param title: The title of the warning dialog.
        :param message: The warning message to display.
        """
        warning_dialog = QMessageBox(self)
        warning_dialog.setIcon(QMessageBox.Icon.Warning)
        warning_dialog.setWindowTitle(title)
        warning_dialog.setText(message)
        warning_dialog.setStandardButtons(QMessageBox.Ok)
        warning_dialog.exec()

    def _connect_robot(self):
        calibration_folder = self.calibration_folder_edit.text()
        usb_port = self.usb_port_edit.text()

        if calibration_folder.strip() == "":
            self.show_warning_dialog(
                "Calibration File Error",
                "Please select a calibration file."
            )
            return
        if usb_port.strip() == "":
            self.show_warning_dialog(
                "USB Port Error",
                "Please enter a USB port."
            )
            return
        # Check if the calibration file exists
        if not pathlib.Path(calibration_folder).exists():
            self.show_warning_dialog(
                "Calibration File Error",
                f"Calibration file '{calibration_folder}' does not exist."
            )
            return

        self.th.connect_real_robot(usb_port, calibration_folder)

    def create_free_camera(self):
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([ 0.0 , 0.0 , 0.0 ])
        cam.distance = self.model.stat.extent * 1.5
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def reset_simulation(self):
        print("Resetting simulation...")
        for jw in self.joint_widgets:
            jw.slider.setValue(0)
            
        # Store current target values from sliders
        target_x_val = self.target_x_slider.value()
        target_y_val = self.target_y_slider.value()
        target_z_val = self.target_z_slider.value()
        print(f"  Pre-reset slider values: ({target_x_val}, {target_y_val}, {target_z_val})")
        
        # Calculate target in meters
        target_x_m = target_x_val / 1000.0
        target_y_m = target_y_val / 1000.0
        target_z_m = target_z_val / 1000.0
        target_pos_meters = np.array([target_x_m, target_y_m, target_z_m])
            
        # Reset MuJoCo state and time, which will reset the target position as well
        mujoco.mj_resetData(self.model, self.data)
        
        # Find the target joint
        target_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
        if target_joint_id != -1:
            # Get the starting index for this joint's position in qpos
            qpos_idx = self.model.jnt_qposadr[target_joint_id]
            # Set the position part of the free joint (first 3 values)
            self.data.qpos[qpos_idx:qpos_idx+3] = target_pos_meters
            print(f"  Immediately set target joint qpos to {target_pos_meters}")
        
        # Reset thread state (but target position in qpos is already set)
        self.th.reset()
        print("  mj_resetData and th.reset() called.")
        
        # Restore target position sliders 
        self.target_x_slider.setValue(target_x_val)
        self.target_y_slider.setValue(target_y_val)
        self.target_z_slider.setValue(target_z_val)
        
        # Update the SAC controller's internal target value
        print("  Calling _update_target_position() post-reset to update SAC controller state.")
        self._update_target_position()
        print("Simulation reset complete.")

    def _select_sac_model_file(self):
        """File dialog to select the SAC model file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAC Model File",
            "",
            "ZIP Files (*.zip);;All Files (*)"
        )
        if file_path:
            self.sac_model_path_edit.setText(file_path)
            
    def _select_norm_stats_file(self):
        """File dialog to select the normalization stats file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Normalization Stats File",
            "",
            "Pickle Files (*.pkl);;All Files (*)"
        )
        if file_path:
            self.norm_stats_path_edit.setText(file_path)
            
    def _load_sac_model(self):
        """Load the SAC model using the update thread"""
        model_path = self.sac_model_path_edit.text()
        if not model_path:
            self.show_warning_dialog(
                "Model Path Error",
                "Please select a SAC model file (.zip)"
            )
            return
            
        # Check if the model file exists
        if not pathlib.Path(model_path).is_file():
            self.show_warning_dialog(
                "Model File Error",
                f"Model file '{model_path}' does not exist."
            )
            return
            
        # Get the norm stats path (optional)
        norm_stats_path = self.norm_stats_path_edit.text() if self.norm_stats_path_edit.text() else None
        if norm_stats_path and not pathlib.Path(norm_stats_path).is_file():
            self.show_warning_dialog(
                "Normalization Stats File Error",
                f"Normalization stats file '{norm_stats_path}' does not exist."
            )
            return
            
        # Load the model
        self.th.load_sac_model(model_path, norm_stats_path)
        self.load_sac_model_button.setEnabled(False)
        self.load_sac_model_button.setText("Loading...")

    def _update_target_position(self) -> None:
        """Update the target position for the SAC controller"""
        # Convert slider values to meters
        x = self.target_x_slider.value() / 1000.0
        y = self.target_y_slider.value() / 1000.0
        z = self.target_z_slider.value() / 1000.0
        
        # Print debug info
        print(f"Setting target position to: ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # Update the labels
        self.target_x_value.setText(f"{x:.3f}")
        self.target_y_value.setText(f"{y:.3f}")
        self.target_z_value.setText(f"{z:.3f}")
        
        # Update the target in the SAC controller
        self.th.set_sac_target([x, y, z])


if __name__ == "__main__":
    app = QApplication()
    app.setStyle('fusion')
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)
    w = Window()
    w.show()
    app.exec()
    w.th.stop()
