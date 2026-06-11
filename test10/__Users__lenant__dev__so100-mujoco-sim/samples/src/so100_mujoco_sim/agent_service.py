import argparse
import json
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QGuiApplication, QImage
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout

from so100_mujoco_sim.arm_control import MUJOCO_SO100_PREFIX, joints_from_model
from so100_mujoco_sim.ik_core import (
    DEFAULT_MAX_JOINT_DELTA,
    PREFIXED_WRIST_CAMERA_NAME,
    calculate_joint_targets,
    clamp,
    interpolate_positions,
    parse_target_position,
    prepare_agent_model_xml,
    solve_ik_position,
)
from so100_mujoco_sim.mujoco_viewport import Viewport


DEFAULT_RENDER_WIDTH = 640
DEFAULT_RENDER_HEIGHT = 480
SCENARIO_CUBE_PICK_PLACE = "cube_pick_place"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

stdout_lock = threading.Lock()


def write_response(response: dict[str, Any]) -> None:
    with stdout_lock:
        print(json.dumps(response), flush=True)


class CommandBridge(QObject):
    command_received = Signal(dict)


class StdinReader(threading.Thread):
    def __init__(self, bridge: CommandBridge) -> None:
        super().__init__(daemon=True)
        self.bridge = bridge

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                write_response({
                    "ok": False,
                    "error": f"invalid JSON: {exc.msg}",
                })
                continue
            if not isinstance(command, dict):
                write_response({
                    "ok": False,
                    "error": "command must be a JSON object",
                })
                continue
            self.bridge.command_received.emit(command)


class AgentWindow(QMainWindow):
    def __init__(
        self,
        memory_path: Path,
        session_dir: Path,
        render_width: int = DEFAULT_RENDER_WIDTH,
        render_height: int = DEFAULT_RENDER_HEIGHT,
    ) -> None:
        super().__init__()
        self.setWindowTitle("SO-100 Agent Service")

        self.memory_path = memory_path
        self.session_dir = session_dir
        self.screenshot_dir = session_dir / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_path.touch(exist_ok=True)

        model_path = prepare_agent_model_xml(Path(__file__).parent / "xml", session_dir / "model_xml")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.joints = joints_from_model(self.model)
        self.joint_ranges = {joint.name: joint.range for joint in self.joints}
        self.joint_targets = self._home_joint_targets()
        self.step_count = 0
        self.render_count = 0

        self._hide_sac_target()

        self.visible_cam = self._create_visible_camera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.viewport = Viewport(self.model, self.data, self.visible_cam, self.opt, self.scn)
        self.viewport.setScreenScale(QGuiApplication.instance().primaryScreen().devicePixelRatio())

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QWidget.createWindowContainer(self.viewport))
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)
        self.resize(1100, 800)

        self.renderer = mujoco.Renderer(
            self.model,
            height=render_height,
            width=render_width,
        )
        self.screenshot_cameras = {
            "top": self._create_fixed_camera(azimuth=90, elevation=-90, distance=0.75),
            "front": self._create_fixed_camera(azimuth=180, elevation=-25, distance=0.75),
            "left": self._create_fixed_camera(azimuth=90, elevation=-25, distance=0.75),
            "gripper": self._create_model_camera(PREFIXED_WRIST_CAMERA_NAME),
        }

        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._idle_step)
        self.timer.start()

        self.reset_environment(SCENARIO_CUBE_PICK_PLACE)

    def _home_joint_targets(self) -> dict[str, float]:
        preferred_home = {
            "Rotation": 0.0,
            "Pitch": -1.57079,
            "Elbow": 1.57079,
            "Wrist_Pitch": 1.57079,
            "Wrist_Roll": -1.57079,
            "Jaw": 0.0,
        }
        targets = {}
        for joint in self.joints:
            low, high = joint.range
            targets[joint.name] = clamp(preferred_home.get(joint.name, 0.0), low, high)
        return targets

    def _create_visible_camera(self) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([0.0, -0.18, 0.08])
        cam.distance = self.model.stat.extent * 1.4
        cam.elevation = -25
        cam.azimuth = 45
        return cam

    def _create_fixed_camera(
        self,
        azimuth: float,
        elevation: float,
        distance: float,
    ) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.lookat = np.array([0.0, -0.22, 0.08])
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
        return cam

    def _create_model_camera(self, name: str) -> mujoco.MjvCamera:
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        if camera_id == -1:
            raise ValueError(f"model camera '{name}' not found")

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        cam.fixedcamid = camera_id
        return cam

    def _hide_sac_target(self) -> None:
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_geom")
        if geom_id != -1:
            self.model.geom_rgba[geom_id, 3] = 0.0

    def _idle_step(self) -> None:
        self._apply_joint_targets()
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1

    def _apply_joint_targets(self) -> None:
        for name, value in self.joint_targets.items():
            actuator_name = MUJOCO_SO100_PREFIX + name
            actuator_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_name,
            )
            if actuator_id != -1:
                self.data.ctrl[actuator_id] = value

    def _set_body_geom_center(self, body_name: str, center: np.ndarray) -> None:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_joint")
        if body_id == -1 or joint_id == -1:
            return

        geom_ids = np.where(self.model.geom_bodyid == body_id)[0]
        geom_pos = self.model.geom_pos[geom_ids[0]] if len(geom_ids) else np.zeros(3)
        qpos_idx = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_idx:qpos_idx + 3] = center - geom_pos
        self.data.qpos[qpos_idx + 3:qpos_idx + 7] = [1.0, 0.0, 0.0, 0.0]

    def _set_free_joint_position(self, joint_name: str, position: np.ndarray) -> None:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            return
        qpos_idx = self.model.jnt_qposadr[joint_id]
        self.data.qpos[qpos_idx:qpos_idx + 3] = position
        self.data.qpos[qpos_idx + 3:qpos_idx + 7] = [1.0, 0.0, 0.0, 0.0]

    def reset_environment(self, scenario: str) -> None:
        if scenario != SCENARIO_CUBE_PICK_PLACE:
            raise ValueError(f"unsupported scenario '{scenario}'")

        mujoco.mj_resetData(self.model, self.data)
        self.joint_targets = self._home_joint_targets()
        for name, value in self.joint_targets.items():
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                MUJOCO_SO100_PREFIX + name,
            )
            if joint_id != -1:
                self.data.qpos[self.model.jnt_qposadr[joint_id]] = value

        self._set_body_geom_center("block_a", np.array([-0.16, -0.28, 0.02]))
        self._set_body_geom_center("block_b", np.array([0.02, -0.30, 0.02]))
        self._set_body_geom_center("block_c", np.array([0.16, -0.28, 0.02]))
        self._set_free_joint_position("target_joint", np.array([0.0, 0.0, -1.0]))
        self._apply_joint_targets()
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

    def handle_command(self, command: dict[str, Any]) -> None:
        request_id = command.get("id")
        try:
            cmd = command.get("cmd")
            if cmd == "reset":
                response = self._handle_reset(command)
            elif cmd == "observe":
                response = self._handle_observe()
            elif cmd == "act":
                response = self._handle_act(command)
            elif cmd == "move_gripper_to":
                response = self._handle_move_gripper_to(command)
            elif cmd == "memory_append":
                response = self._handle_memory_append(command)
            elif cmd == "close":
                response = {"ok": True, "cmd": "close"}
                QTimer.singleShot(0, QApplication.instance().quit)
            else:
                raise ValueError("cmd must be one of reset, observe, act, move_gripper_to, memory_append, close")
            if request_id is not None:
                response["id"] = request_id
            write_response(response)
        except Exception as exc:
            response = {
                "ok": False,
                "error": str(exc),
            }
            if request_id is not None:
                response["id"] = request_id
            write_response(response)

    def _handle_reset(self, command: dict[str, Any]) -> dict[str, Any]:
        scenario = command.get("scenario", SCENARIO_CUBE_PICK_PLACE)
        self.reset_environment(str(scenario))
        return {
            "ok": True,
            "cmd": "reset",
            "observation": self._observe(include_screenshots=True),
        }

    def _handle_observe(self) -> dict[str, Any]:
        return {
            "ok": True,
            "cmd": "observe",
            "observation": self._observe(include_screenshots=True),
        }

    def _parse_motion_steps(self, command: dict[str, Any]) -> int:
        steps = int(command.get("steps", 30))
        if steps < 1 or steps > 1000:
            raise ValueError("steps must be between 1 and 1000")
        return steps

    def _apply_interpolated_joint_targets(self, target: dict[str, float], steps: int) -> None:
        start = dict(self.joint_targets)
        for frame in interpolate_positions(start, target, steps):
            self.joint_targets = frame
            self._apply_joint_targets()
            for _ in range(5):
                mujoco.mj_step(self.model, self.data)
                self.step_count += 1
        self.joint_targets = target
        self._apply_joint_targets()
        mujoco.mj_forward(self.model, self.data)

    def _handle_act(self, command: dict[str, Any]) -> dict[str, Any]:
        mode = str(command.get("mode", "delta"))
        steps = self._parse_motion_steps(command)
        max_delta = float(command.get("max_delta", DEFAULT_MAX_JOINT_DELTA))
        if max_delta <= 0:
            raise ValueError("max_delta must be positive")

        start = dict(self.joint_targets)
        target = calculate_joint_targets(
            start,
            command.get("joints"),
            self.joint_ranges,
            mode,
            max_delta=max_delta,
        )
        self._apply_interpolated_joint_targets(target, steps)
        return {
            "ok": True,
            "cmd": "act",
            "joint_targets": self.joint_targets,
            "observation": self._observe(include_screenshots=False),
        }

    def _handle_move_gripper_to(self, command: dict[str, Any]) -> dict[str, Any]:
        steps = self._parse_motion_steps(command)
        target_position = parse_target_position(command.get("position"))
        result = solve_ik_position(
            self.model,
            self.data,
            self.joint_targets,
            self.joint_ranges,
            target_position,
        )
        self._apply_interpolated_joint_targets(result.joint_targets, steps)
        return {
            "ok": True,
            "cmd": "move_gripper_to",
            "reached": result.reached,
            "position_error": result.position_error,
            "iterations": result.iterations,
            "joint_targets": self.joint_targets,
            "observation": self._observe(include_screenshots=False),
        }

    def _handle_memory_append(self, command: dict[str, Any]) -> dict[str, Any]:
        text = command.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.memory_path.open("a", encoding="utf-8") as file:
            file.write(f"\n## {timestamp}\n\n{text.strip()}\n")
        return {
            "ok": True,
            "cmd": "memory_append",
            "memory_path": str(self.memory_path),
        }

    def _observe(self, include_screenshots: bool) -> dict[str, Any]:
        self._apply_joint_targets()
        mujoco.mj_forward(self.model, self.data)
        observation = {
            "joint_positions": self._joint_positions(),
            "end_effector_pose": self._end_effector_pose(),
            "memory_path": str(self.memory_path),
            "step_count": self.step_count,
        }
        if include_screenshots:
            observation["screenshots"] = self._capture_screenshots()
        return observation

    def _joint_positions(self) -> dict[str, float]:
        positions = {}
        for joint in self.joints:
            joint_name = MUJOCO_SO100_PREFIX + joint.name
            positions[joint.name] = float(self.data.joint(joint_name).qpos[0])
        return positions

    def _end_effector_pose(self) -> dict[str, list[float]]:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "so100_ee_site")
        if site_id != -1:
            position = self.data.site(site_id).xpos.copy()
            matrix = self.data.site(site_id).xmat.copy()
        else:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "so100_Moving_Jaw")
            if body_id == -1:
                raise ValueError("end-effector site/body not found")
            position = self.data.body(body_id).xpos.copy()
            matrix = self.data.body(body_id).xmat.copy()

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, matrix)
        return {
            "position": [float(value) for value in position],
            "quaternion": [float(value) for value in quat],
        }

    def _capture_screenshots(self) -> dict[str, str]:
        self.render_count += 1
        screenshots = {}
        for name, camera in self.screenshot_cameras.items():
            self.renderer.update_scene(self.data, camera=camera)
            pixels = self.renderer.render()
            path = self.screenshot_dir / f"{self.render_count:06d}_{name}.png"
            self._save_rgb_png(pixels, path)
            screenshots[name] = str(path)
        return screenshots

    def _save_rgb_png(self, pixels: np.ndarray, path: Path) -> None:
        image = np.ascontiguousarray(pixels)
        height, width, _ = image.shape
        qimage = QImage(
            image.data,
            width,
            height,
            3 * width,
            QImage.Format.Format_RGB888,
        ).copy()
        if not qimage.save(str(path)):
            raise ValueError(f"failed to save screenshot to {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visible JSON REPL service for SO-100 agent control.")
    parser.add_argument(
        "--memory",
        type=Path,
        default=PROJECT_ROOT / "agent_memory.md",
        help="Path to the persistent agent memory markdown file.",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Directory for screenshots and session artifacts.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_RENDER_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_RENDER_HEIGHT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = args.session_dir
    if session_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = PROJECT_ROOT / "agent_runs" / f"{stamp}-{uuid.uuid4().hex[:8]}"

    app = QApplication()
    app.setStyle("fusion")
    app.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)

    window = AgentWindow(
        memory_path=args.memory,
        session_dir=session_dir,
        render_width=args.width,
        render_height=args.height,
    )
    bridge = CommandBridge()
    bridge.command_received.connect(window.handle_command)
    reader = StdinReader(bridge)
    reader.start()

    window.show()
    write_response({
        "ok": True,
        "cmd": "ready",
        "session_dir": str(session_dir),
        "memory_path": str(args.memory),
    })
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
