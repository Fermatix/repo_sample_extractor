import time
import numpy as np

import mujoco
from PySide6.QtCore import QThread, Signal

from so100_mujoco_sim.arm_control import (
    ArmController,
    MujocoArmController,
    So100ArmController,
    PlaybackRecordController,
    PlaybackRecordState,
    UiArmController,
    SacArmController,
    update_from_controller
)


class UpdateThread(QThread):

    update_ui_joint_values = Signal(list)
    update_ui_recorded_steps = Signal(int, int)
    update_controller_enabled_states = Signal()
    update_primary_controller = Signal(str)
    warning = Signal(str, str)

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, parent=None) -> None:
        super().__init__(parent)
        self.model = model
        self.data = data
        self.mujoco_controller = MujocoArmController(model, data)
        # user interface controller (the UI sliders)
        # reuse the mujoco model joint definition
        self.ui_controller = UiArmController(self.mujoco_controller.joints, self._update_ui)
        self.ui_controller.primary = True
        self._primary_controller_index = 0
        self.real_controller = So100ArmController()
        self.playback_record_controller = PlaybackRecordController(
            self.mujoco_controller.joints,
            self._update_ui_recorded_steps
        )
        # Add SAC controller
        self.sac_controller = SacArmController(self.mujoco_controller.joints, model, data)

        self.arm_controllers: list[ArmController] = []
        self.arm_controllers.append(self.ui_controller)
        self.arm_controllers.append(self.mujoco_controller)
        self.arm_controllers.append(self.real_controller)
        self.arm_controllers.append(self.playback_record_controller)
        self.arm_controllers.append(self.sac_controller)  # Add SAC controller to the list
        
        # Initialize target site visibility
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        if site_id != -1:
            try:
                self.model.site_rgba[site_id, 3] = 0.7  # Make target site clearly visible
            except Exception:
                pass

        # reset the simulation timer
        self.reset()

        self._playback_file = None
        self._playback_file_save = False
        self._playback_file_load = False

        self._do_connection = False
        self._load_sac_model = False
        self._sac_model_path = None
        self._sac_norm_stats_path = None
        
        self.running = True

    @property
    def real_time(self):
        return time.monotonic_ns() - self.real_time_start

    def get_primary_controller(self) -> ArmController | None:
        for ac in self.arm_controllers:
            if ac.primary:
                return ac
        return None

    def run(self) -> None:
        # Find target body and joint indices
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        target_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
        
        if target_joint_id == -1:
            print("Warning: Target joint 'target_joint' not found in model")
        
        # In a free joint, the first 3 values in qpos are for position (x,y,z)
        target_joint_qpos_idx = 0
        if target_joint_id != -1:
            target_joint_qpos_idx = self.model.jnt_qposadr[target_joint_id]
        
        while self.running:
            # don't step the simulation past real time
            # without this the sim usually finishes before it's
            # even visible
            if self.data.time < self.real_time / 1_000_000_000:
                # Update the control loop at a 100hz
                if (time.monotonic_ns() - self.last_robot_update) / 1_000_000_000 >= (1/100):
                    self.last_robot_update = time.monotonic_ns()

                    if self.get_primary_controller_index() != self._primary_controller_index:
                        self._set_primary_controller_index(self._primary_controller_index)
                        self.update_primary_controller.emit(
                            self.arm_controllers[self._primary_controller_index].name
                        )

                    # --- Update the target body position using free joint ---
                    if target_joint_id != -1:
                        try:
                            # Current target position from joint qpos
                            current_target_pos = self.data.qpos[target_joint_qpos_idx:target_joint_qpos_idx+3].copy()
                            desired_target_pos = self.sac_controller.target_pos
                            
                            # Only update if different to avoid unnecessary changes
                            if not np.array_equal(current_target_pos, desired_target_pos):
                                # Set the position part of the free joint
                                self.data.qpos[target_joint_qpos_idx:target_joint_qpos_idx+3] = desired_target_pos
                                
                                # Print debug info (only when position changes)
                                current_thread_obj = QThread.currentThread()
                                print(f"[Thread {current_thread_obj}] Updated target: {current_target_pos} -> {desired_target_pos}")
                        except Exception as e:
                            print(f"Error updating target position: {e}")

                    pc = self.get_primary_controller()
                    pc_name = pc.name

                    # --- Run Update on all controllers first --- 
                    # This reads current state (e.g., real robot pos, sim state) 
                    # and calculates desired outputs (e.g., SAC action)
                    for ac in self.arm_controllers:
                        ac.update()

                    # --- Apply updates based on primary controller --- 
                    if pc_name == "Robot":
                        # Real robot is primary: 
                        # 1. Sim follows Real (already done in mujoco_controller.update implicitly via update_from_controller below? No, need explicit update) 
                        # -> Update sim state from real state first
                        update_from_controller(self.real_controller, self.mujoco_controller)
                        # 2. Real follows SAC (calculated in sac_controller.update() above based on mirrored sim state)
                        update_from_controller(self.sac_controller, self.real_controller)
                        # 3. Send command to real robot
                        self.real_controller.set_positions()
                        # 4. mujoco_controller set_positions (to apply real state to sim actuators)
                        self.mujoco_controller.set_positions() 

                    elif pc_name == "SAC Model":
                        # SAC is primary (controlling simulation):
                        # 1. Sim follows SAC (calculated in sac_controller.update() above)
                        update_from_controller(self.sac_controller, self.mujoco_controller)
                        # 2. Apply control to sim actuators
                        self.mujoco_controller.set_positions()
                        # 3. If real robot connected, make it follow SAC too (optional?)
                        if self.real_controller.is_connected():
                           update_from_controller(self.sac_controller, self.real_controller)
                           self.real_controller.set_positions()
                           
                    elif pc_name == "User Interface":
                         # UI is primary:
                         # 1. Sim follows UI
                         update_from_controller(self.ui_controller, self.mujoco_controller)
                         self.mujoco_controller.set_positions()
                         # 2. Real follows UI (if connected)
                         if self.real_controller.is_connected():
                            update_from_controller(self.ui_controller, self.real_controller)
                            self.real_controller.set_positions()
                            
                    elif pc_name == "Playback/Record":
                         # Playback/Record is primary:
                         # Playback controller calculates its output in its update()
                         # 1. Sim follows Playback
                         update_from_controller(self.playback_record_controller, self.mujoco_controller)
                         self.mujoco_controller.set_positions()
                         # 2. Real follows Playback (if connected)
                         if self.real_controller.is_connected():
                            update_from_controller(self.playback_record_controller, self.real_controller)
                            self.real_controller.set_positions()
                            
                    # Update other secondary controllers as needed (e.g., UI reflects current state)
                    for ac in self.arm_controllers:
                        if not ac.primary:
                            # Make UI reflect the state of the primary controller (or the sim state if Robot is primary)
                            if ac.name == "User Interface":
                                if pc_name == "Robot":
                                    update_from_controller(self.real_controller, self.ui_controller) # UI shows real robot state
                                else:
                                     update_from_controller(pc, self.ui_controller) # UI shows primary controller state
                                self.ui_controller.set_positions() # Triggers UI update signal
                            # Add other non-primary updates if needed
                            pass

                # --- Apply frame skip: Step the simulation multiple times per control action ---
                frame_skip = 5 # Match the ReachEnv value
                # The control inputs (self.data.ctrl) should have been set by ac.set_positions()
                for _ in range(frame_skip):
                    if not self.running: # Check running flag inside loop
                        break
                    # Step the simulation forward
                    mujoco.mj_step(self.model, self.data)
                    # Optional: Small sleep within frame skip if needed for timing,
                    # but usually not necessary as mj_step takes time.
                    # time.sleep(0.0001) 
                    
                # --- Old single step (removed) ---
                # mujoco.mj_step(self.model, self.data)

                # here's where we check if there's something that was requested
                # to be done from the UI thread, and do it
                if self._do_connection:
                    self._connect_real_robot()
                if self._playback_file_save:
                    self._save_playback_file()
                if self._playback_file_load:
                    self._load_playback_file()
                if self._load_sac_model:
                    self._load_sac_model_thread()
            else:
                time.sleep(0.00001)

    def stop(self):
        self.running = False
        self.wait()

    def reset(self):
        """Reset simulation timer and controllers"""
        self.real_time_start = time.monotonic_ns()
        self.last_robot_update = time.monotonic_ns()
        self.mujoco_controller.reset()
        # Target position is managed continuously in the run loop, no reset needed here.
        print("UpdateThread reset called.")

    def set_joint_position(self, joint_name: str, position: float) -> None:
        self.ui_controller.set_joint_set_position(joint_name, position)

    def get_controller_names(self) -> str:
        return [c.name for c in self.arm_controllers]

    def get_controllable_controllers(self) -> list[bool]:
        return [c.controllable for c in self.arm_controllers]

    def get_primary_controller_index(self) -> int:
        for i, c in enumerate(self.arm_controllers):
            if c.primary:
                return i
        return 0

    def set_primary_controller_index(self, index: int) -> int:
        # we need to make sure the primary flag on the controllers
        # is changed in the main update loop as  setting this flag
        # can send commands to the motors (which may be getting sent
        # data from the update thread). So set the desired index
        # here, and do the proper update for thread in the following fn
        self._primary_controller_index = index

    def _set_primary_controller_index(self, index: int) -> int:
        for i, c in enumerate(self.arm_controllers):
            if i == index:
                c.primary = True
                # Always make the target site visible, regardless of controller
                site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
                if site_id != -1:
                    try:
                        # Always visible (full opacity)
                        self.model.site_rgba[site_id, 3] = 0.7
                    except Exception:
                        pass
            else:
                c.primary = False

    def connect_real_robot(self, usb_port: str, calibration_folder: str) -> None:
        self._usb_port = usb_port
        self._calibration_folder = calibration_folder
        self._do_connection = True

    def _connect_real_robot(self) -> None:
        if not self._do_connection:
            return
        self._do_connection = False

        try:
            self.real_controller.connect(self._usb_port, self._calibration_folder)
            self.real_controller.update()
        except Exception as e:
            self.warning.emit("Connection issue", f"Error connecting to real robot: {e}")
            return

        # this updates the ui, but also raises change events that causes the mujoco
        # model to update
        self.update_ui_joint_values.emit(self.real_controller.joint_output_positions)
        # raise event to tell UI that the real robot controller can be enabled
        self.update_controller_enabled_states.emit()

    def _update_ui(self) -> None:
        self.update_ui_joint_values.emit(self.ui_controller.joint_set_positions)

    def set_playback_record_state(self, state: PlaybackRecordState) -> None:
        self.playback_record_controller.set_state(state)

    def _update_ui_recorded_steps(self) -> None:
        self.update_ui_recorded_steps.emit(
            len(self.playback_record_controller.recorded_joint_positions),
            self.playback_record_controller.playback_index
        )

    def save_playback_file(self, file_name: str) -> None:
        self._playback_file = file_name
        self._playback_file_save = True

    def _save_playback_file(self) -> None:
        if self._playback_file_save:
            self.playback_record_controller.save_playback_file(self._playback_file)
            self._playback_file_save = False

    def load_playback_file(self, file_name: str) -> None:
        self._playback_file = file_name
        self._playback_file_load = True

    def _load_playback_file(self) -> None:
        if self._playback_file_load:
            self.playback_record_controller.load_playback_file(self._playback_file)
            self._playback_file_load = False

            self.update_ui_recorded_steps.emit(
                len(self.playback_record_controller.recorded_joint_positions),
                self.playback_record_controller.playback_index
            )

    def load_sac_model(self, model_path: str, norm_stats_path: str = None) -> None:
        self._sac_model_path = model_path
        self._sac_norm_stats_path = norm_stats_path
        self._load_sac_model = True
        
    def _load_sac_model_thread(self) -> None:
        """Load the SAC model in the update thread"""
        if not self._load_sac_model:
            return
        
        self._load_sac_model = False
        
        try:
            success = self.sac_controller.load_model(self._sac_model_path, self._sac_norm_stats_path)
            if success:
                self.update_controller_enabled_states.emit()
            else:
                self.warning.emit("Model Loading Error", f"Failed to load SAC model from {self._sac_model_path}")
        except Exception as e:
            self.warning.emit("Model Loading Error", f"Error loading SAC model: {e}")

    def set_sac_target(self, target_pos: list) -> None:
        """Set the target position for the SAC controller"""
        # Convert to numpy array if not already
        target_pos_np = np.array(target_pos)
        
        # Update the target in the controller
        self.sac_controller.set_target_position(target_pos_np)
        
        # Also set the target position directly in the simulation data
        # This ensures immediate visual feedback without waiting for the next run loop
        target_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_joint")
        if target_joint_id != -1:
            try:
                # Get the starting index for this joint's position in qpos
                qpos_idx = self.model.jnt_qposadr[target_joint_id]
                # Set the position part of the free joint (first 3 values)
                self.data.qpos[qpos_idx:qpos_idx+3] = target_pos_np
                print(f"Direct target update: Set target joint qpos to {target_pos_np}")
            except Exception as e:
                print(f"Error in set_sac_target: {e}")
