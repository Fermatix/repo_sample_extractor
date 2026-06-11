#!/usr/bin/env python
"""Headless JSON control service for the **real** SO-100 arm.

This is the real-robot analogue of ``src/so100_mujoco_sim/agent_service.py``.
It exposes the *same* end-effector interface (`move_gripper_to(x,y,z)`,
`open_gripper`, `close_gripper`, `observe`, ...) over a newline-delimited
JSON stdin/stdout protocol, so the arm can be driven exactly the way the sim
was — send a command, read back the camera image, decide the next move.

It reuses, unchanged:
  * the sim's MuJoCo IK/FK  (`so100_mujoco_sim.ik_core.solve_ik_position`),
    run **headlessly** (no Qt, no renderer) purely for arm kinematics;
  * the real-arm driver     (`so100_mujoco_sim.arm_control.So100ArmController`,
    LeRobot under the hood — handles sign-flips + rad->deg + calibration);
  * the iPhone camera        (`scripts/real_camera.py` -> `RealCamera`).

SAFETY — the whole point of this service is to never snap the arm to a
possibly-wrong IK pose at full servo speed. **No servo registers are written**
(servo Goal_Speed/Acceleration stay at factory defaults). All limiting is in
software, by *how* goal positions are sent:

  Layer 1 (primary)  software trajectory interpolation: every motion is split
                     into many small waypoints so no joint exceeds
                     --max-joint-speed deg/s at --rate-hz.
  Layer 2 (backstop) robot.config.max_relative_target = --max-relative-target
                     (deg) -> LeRobot clamps any single send_action delta.
  Layer 3 (workspace) seed IK from the present pose, enforce a Z-floor + XY box,
                     and *refuse to move* if IK can't reach within
                     --max-ik-error (so we never chase an unreachable target).

Because every commanded waypoint is only a fraction of a degree from the
present pose, the per-step motion is tiny even though the servo is uncapped.

Run (always pass an absolute --session-dir):
    .pixi/envs/default/bin/python scripts/real_agent_service.py \
        --port /dev/tty.usbmodemXXXX \
        --calibration-dir /abs/path/to/calibration_folder \
        --session-dir /abs/path/to/agent_runs/real-test

Smoke-test without hardware (no robot, virtual joint state, optional camera):
    .pixi/envs/default/bin/python scripts/real_agent_service.py \
        --dry-run --no-camera --session-dir /abs/.../real-dry
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

# Make sibling scripts (real_camera.py) importable when launched as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from real_camera import CameraError, RealCamera  # noqa: E402

from so100_mujoco_sim import ik_core  # noqa: E402
from so100_mujoco_sim.arm_control import (  # noqa: E402
    MUJOCO_SO100_PREFIX,
    So100ArmController,
    joints_from_model,
)
from so100_mujoco_sim.ik_core import (  # noqa: E402
    calculate_joint_targets,
    clamp,
    interpolate_positions,
    parse_target_position,
    prepare_agent_model_xml,
    resolve_end_effector,
    solve_ik_position,
)

# --- defaults (all overridable on the CLI) ----------------------------------
DEFAULT_MAX_JOINT_SPEED_DEG = 15.0   # deg/s, per joint (Layer 1)
DEFAULT_RATE_HZ = 30.0               # waypoints/s (Layer 1)
DEFAULT_MAX_RELATIVE_TARGET_DEG = 8.0  # deg, per-command clamp (Layer 2)
DEFAULT_MAX_IK_ERROR = 0.03          # m; refuse to move if IK can't get closer
DEFAULT_Z_FLOOR = 0.02               # m; don't drive the EE below this height
DEFAULT_XY_LIMIT = 0.45              # m; |x|,|y| workspace box (generous)
MAX_WAYPOINTS = 800                  # hard ceiling on a single move

DEFAULT_JAW_OPEN = 1.2               # rad (MuJoCo Jaw convention) — tune on HW
DEFAULT_JAW_CLOSED = 0.15            # rad — tune on HW

# Home pose in MuJoCo-radian convention (matches the sim's home keyframe).
HOME_POSE = {
    "Rotation": 0.0,
    "Pitch": -1.57079,
    "Elbow": 1.57079,
    "Wrist_Pitch": 1.57079,
    "Wrist_Roll": -1.57079,
    "Jaw": 0.0,
}

# The MuJoCo joint name order we expect the model + controller to be aligned to.
# So100ArmController.joints is [shoulder_pan, shoulder_lift, elbow_flex,
# wrist_flex, wrist_roll, gripper] and the sim mirroring (update_from_controller)
# copies positions by *index*, so these correspond one-to-one.
EXPECTED_JOINT_ORDER = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

_stdout_lock = threading.Lock()


def log(message: str) -> None:
    """Human-facing log line -> stderr (stdout stays pure JSON)."""
    print(f"[real-service] {message}", file=sys.stderr, flush=True)


def write_response(response: dict[str, Any]) -> None:
    with _stdout_lock:
        print(json.dumps(response), flush=True)


class RealRobotService:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dry_run = bool(args.dry_run)
        self.max_joint_speed = float(args.max_joint_speed)
        self.rate_hz = float(args.rate_hz)
        self.max_ik_error = float(args.max_ik_error)
        self.z_floor = float(args.z_floor)
        self.xy_limit = float(args.xy_limit)
        self.jaw_open = float(args.jaw_open)
        self.jaw_closed = float(args.jaw_closed)
        self.step_count = 0
        self.render_count = 0

        self.session_dir = Path(args.session_dir).resolve()
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        # --- headless MuJoCo model for arm FK/IK (no renderer, no Qt) ---
        xml_dir = Path(ik_core.__file__).resolve().parent / "xml"
        scene_path = prepare_agent_model_xml(xml_dir, self.session_dir / "model_xml")
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.joints = joints_from_model(self.model)
        self.joint_order = [j.name for j in self.joints]
        self.joint_ranges = {j.name: j.range for j in self.joints}
        self.ee_target_type, self.ee_target_id = resolve_end_effector(self.model)

        if self.joint_order != EXPECTED_JOINT_ORDER:
            raise RuntimeError(
                "unexpected MuJoCo joint order "
                f"{self.joint_order!r}; expected {EXPECTED_JOINT_ORDER!r}. "
                "The index-aligned real<->sim joint mapping assumes this order."
            )

        # --- camera ---
        self.camera: RealCamera | None = None
        if not args.no_camera:
            try:
                self.camera = RealCamera(
                    index=args.camera_index,
                    name_match=args.camera_name,
                    width=(args.camera_width or None),
                )
                log(f"camera ready (device index {self.camera.index})")
            except CameraError as exc:
                log(f"WARNING: camera unavailable ({exc}); continuing without images")
                self.camera = None

        # --- real arm ---
        self.controller = So100ArmController()
        # Virtual joint state used only in --dry-run (radians, MuJoCo convention).
        self._dry_state = dict(HOME_POSE)
        if not self.dry_run:
            log(f"connecting to robot on {args.port} ...")
            self.controller.connect(args.port, args.calibration_dir)
            # Layer 2: clamp any single send_action delta (degrees).
            self.controller.robot.config.max_relative_target = float(args.max_relative_target)
            log(
                "connected; max_relative_target="
                f"{args.max_relative_target} deg, "
                f"speed<={self.max_joint_speed} deg/s @ {self.rate_hz} Hz"
            )
        else:
            log("DRY RUN: not connecting to any robot (virtual joint state)")

    # ------------------------------------------------------------------ I/O
    def read_present(self) -> dict[str, float]:
        """Present joint angles as {MuJoCo_name: radians}."""
        if self.dry_run:
            return dict(self._dry_state)
        self.controller.update()
        values = list(self.controller.joint_actual_positions)
        if len(values) != len(self.joint_order):
            raise RuntimeError(
                f"controller returned {len(values)} joints, "
                f"expected {len(self.joint_order)}"
            )
        return {name: float(values[i]) for i, name in enumerate(self.joint_order)}

    def command_pose(self, target: dict[str, float]) -> None:
        """Send one goal pose (radians, MuJoCo convention) to the arm."""
        ordered = [float(target[name]) for name in self.joint_order]
        if self.dry_run:
            self._dry_state = dict(target)
            return
        self.controller.set_joint_set_positions(ordered)
        self.controller.set_positions()

    def fk_pose(self, joints: dict[str, float]) -> dict[str, list[float]]:
        """Forward kinematics -> end-effector {position, quaternion}."""
        for name, value in joints.items():
            self.data.joint(MUJOCO_SO100_PREFIX + name).qpos[0] = float(value)
        mujoco.mj_forward(self.model, self.data)
        if self.ee_target_type == "site":
            position = self.data.site(self.ee_target_id).xpos.copy()
            matrix = self.data.site(self.ee_target_id).xmat.copy()
        else:
            position = self.data.body(self.ee_target_id).xpos.copy()
            matrix = self.data.body(self.ee_target_id).xmat.copy()
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, matrix)
        return {
            "position": [float(v) for v in position],
            "quaternion": [float(v) for v in quat],
        }

    def capture_image(self, label: str) -> str | None:
        if self.camera is None:
            return None
        self.render_count += 1
        path = self.frames_dir / f"{self.render_count:06d}_{label}.png"
        try:
            self.camera.capture(path)
        except CameraError as exc:
            log(f"WARNING: capture failed ({exc})")
            return None
        return str(path)

    def observe(self, label: str = "observe") -> dict[str, Any]:
        present = self.read_present()
        observation: dict[str, Any] = {
            "joint_positions": present,
            "end_effector_pose": self.fk_pose(present),
            "jaw": present.get("Jaw"),
            "step_count": self.step_count,
            "camera": self.capture_image(label),
        }
        return observation

    # --------------------------------------------------------------- motion
    def move_through(self, target: dict[str, float]) -> int:
        """Layer 1: speed-limited interpolation present -> target.

        Returns the number of waypoints sent.
        """
        start = self.read_present()
        target = {name: float(target.get(name, start[name])) for name in self.joint_order}

        max_delta_rad = max(abs(target[n] - start[n]) for n in self.joint_order)
        max_delta_deg = math.degrees(max_delta_rad)
        if max_delta_deg < 1e-6:
            return 0

        step_deg = self.max_joint_speed / self.rate_hz
        steps = max(1, math.ceil(max_delta_deg / step_deg))
        steps = min(steps, MAX_WAYPOINTS)

        period = 1.0 / self.rate_hz
        for frame in interpolate_positions(start, target, steps):
            self.command_pose(frame)
            self.step_count += 1
            time.sleep(period)
        return steps

    # -------------------------------------------------------------- handlers
    def _clamp_target_position(self, xyz: np.ndarray) -> tuple[np.ndarray, list[str]]:
        notes: list[str] = []
        x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
        if z < self.z_floor:
            notes.append(f"z {z:.3f} clamped up to z_floor {self.z_floor:.3f}")
            z = self.z_floor
        cx = clamp(x, -self.xy_limit, self.xy_limit)
        cy = clamp(y, -self.xy_limit, self.xy_limit)
        if cx != x:
            notes.append(f"x {x:.3f} clamped to {cx:.3f}")
        if cy != y:
            notes.append(f"y {y:.3f} clamped to {cy:.3f}")
        return np.array([cx, cy, z], dtype=float), notes

    def handle_move_gripper_to(self, command: dict[str, Any]) -> dict[str, Any]:
        raw = parse_target_position(command.get("position"))
        target_xyz, notes = self._clamp_target_position(raw)

        present = self.read_present()
        result = solve_ik_position(
            self.model, self.data, present, self.joint_ranges, target_xyz
        )
        response: dict[str, Any] = {
            "ok": True,
            "cmd": "move_gripper_to",
            "requested_position": [float(v) for v in raw],
            "target_position": [float(v) for v in target_xyz],
            "reached": result.reached,
            "position_error": result.position_error,
            "iterations": result.iterations,
            "notes": notes,
        }

        if result.position_error > self.max_ik_error:
            # Layer 3: do NOT move toward an unreachable target.
            response["moved"] = False
            response["waypoints"] = 0
            response["refused"] = (
                f"IK position_error {result.position_error:.4f} m exceeds "
                f"max_ik_error {self.max_ik_error:.4f} m; not moving"
            )
            response["observation"] = self.observe("move_refused")
            return response

        waypoints = self.move_through(result.joint_targets)
        response["moved"] = True
        response["waypoints"] = waypoints
        response["joint_targets"] = {k: float(v) for k, v in result.joint_targets.items()}
        response["observation"] = self.observe("move_gripper_to")
        return response

    def _move_jaw(self, value: float, cmd: str) -> dict[str, Any]:
        present = self.read_present()
        low, high = self.joint_ranges["Jaw"]
        target = dict(present)
        target["Jaw"] = clamp(float(value), low, high)
        waypoints = self.move_through(target)
        return {
            "ok": True,
            "cmd": cmd,
            "jaw_target": target["Jaw"],
            "waypoints": waypoints,
            "observation": self.observe(cmd),
        }

    def handle_open_gripper(self, command: dict[str, Any]) -> dict[str, Any]:
        return self._move_jaw(self.jaw_open, "open_gripper")

    def handle_close_gripper(self, command: dict[str, Any]) -> dict[str, Any]:
        return self._move_jaw(self.jaw_closed, "close_gripper")

    def handle_home(self, command: dict[str, Any]) -> dict[str, Any]:
        present = self.read_present()
        target = dict(HOME_POSE)
        target["Jaw"] = self.jaw_open  # open the gripper at home
        # Do NOT spin the wrist roll during home. Driving it to a fixed -90 deg from
        # an arbitrary current angle can force a ~270 deg twist that strains the wrist
        # cables AND, by dominating the waypoint count, holds the lifting joints under
        # high torque long enough to trip Feetech overload protection. Leave it put.
        target["Wrist_Roll"] = present["Wrist_Roll"]
        waypoints = self.move_through(target)
        return {
            "ok": True,
            "cmd": "home",
            "waypoints": waypoints,
            "observation": self.observe("home"),
        }

    def handle_set_joints(self, command: dict[str, Any]) -> dict[str, Any]:
        mode = command.get("mode", "delta")
        joints = command.get("joints")
        present = self.read_present()
        target = calculate_joint_targets(present, joints, self.joint_ranges, mode)
        waypoints = self.move_through(target)
        return {
            "ok": True,
            "cmd": "set_joints",
            "joint_targets": {k: float(v) for k, v in target.items()},
            "waypoints": waypoints,
            "observation": self.observe("set_joints"),
        }

    def handle_observe(self, command: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True, "cmd": "observe", "observation": self.observe("observe")}

    def dispatch(self, command: dict[str, Any]) -> dict[str, Any]:
        cmd = command.get("cmd")
        handlers = {
            "observe": self.handle_observe,
            "move_gripper_to": self.handle_move_gripper_to,
            "open_gripper": self.handle_open_gripper,
            "close_gripper": self.handle_close_gripper,
            "home": self.handle_home,
            "set_joints": self.handle_set_joints,
        }
        if cmd not in handlers:
            raise ValueError(f"unknown cmd '{cmd}'")
        return handlers[cmd](command)

    def close(self) -> None:
        if not self.dry_run and self.controller.is_connected():
            try:
                self.controller.robot.disconnect()
            except Exception as exc:  # noqa: BLE001
                log(f"WARNING: disconnect failed ({exc})")
        log("closed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless JSON control service for the real SO-100 arm.")
    p.add_argument("--port", help="USB serial port of the follower arm (e.g. /dev/tty.usbmodem...)")
    p.add_argument("--calibration-dir", help="LeRobot calibration folder used by the sim UI")
    p.add_argument("--session-dir", required=True, help="ABSOLUTE dir for frames/model artifacts")
    p.add_argument("--camera-index", type=int, default=None, help="force AVFoundation device index")
    p.add_argument("--camera-name", default="iphone", help="substring to match the camera name")
    p.add_argument("--camera-width", type=int, default=1280, help="capture width px (0 = native)")
    p.add_argument("--no-camera", action="store_true", help="run without any camera")
    p.add_argument("--max-joint-speed", type=float, default=DEFAULT_MAX_JOINT_SPEED_DEG,
                   help="per-joint speed cap, deg/s (Layer 1)")
    p.add_argument("--rate-hz", type=float, default=DEFAULT_RATE_HZ,
                   help="waypoint send rate, Hz (Layer 1)")
    p.add_argument("--max-relative-target", type=float, default=DEFAULT_MAX_RELATIVE_TARGET_DEG,
                   help="per-command joint delta clamp, deg (Layer 2)")
    p.add_argument("--max-ik-error", type=float, default=DEFAULT_MAX_IK_ERROR,
                   help="refuse to move if IK can't reach within this, m (Layer 3)")
    p.add_argument("--z-floor", type=float, default=DEFAULT_Z_FLOOR,
                   help="minimum EE z height, m (Layer 3)")
    p.add_argument("--xy-limit", type=float, default=DEFAULT_XY_LIMIT,
                   help="workspace |x|,|y| box, m (Layer 3)")
    p.add_argument("--jaw-open", type=float, default=DEFAULT_JAW_OPEN, help="open jaw angle, rad")
    p.add_argument("--jaw-closed", type=float, default=DEFAULT_JAW_CLOSED, help="closed jaw angle, rad")
    p.add_argument("--dry-run", action="store_true", help="no robot; virtual joint state for testing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dry_run and (not args.port or not args.calibration_dir):
        print(
            json.dumps({"ok": False, "cmd": "ready",
                        "error": "--port and --calibration-dir are required (or use --dry-run)"}),
            flush=True,
        )
        return 2

    service = RealRobotService(args)
    write_response({
        "ok": True,
        "cmd": "ready",
        "dry_run": service.dry_run,
        "session_dir": str(service.session_dir),
        "camera": service.camera.index if service.camera else None,
        "joint_order": service.joint_order,
        "limits": {
            "max_joint_speed_deg_s": service.max_joint_speed,
            "rate_hz": service.rate_hz,
            "max_relative_target_deg": float(args.max_relative_target),
            "max_ik_error_m": service.max_ik_error,
            "z_floor_m": service.z_floor,
            "xy_limit_m": service.xy_limit,
        },
    })

    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line)
            except json.JSONDecodeError as exc:
                write_response({"ok": False, "error": f"invalid JSON: {exc}"})
                continue

            req_id = command.get("id")
            if command.get("cmd") == "close":
                write_response({"ok": True, "cmd": "close", "id": req_id})
                break
            try:
                response = service.dispatch(command)
            except Exception as exc:  # noqa: BLE001
                response = {"ok": False, "cmd": command.get("cmd"), "error": str(exc)}
            if req_id is not None:
                response["id"] = req_id
            write_response(response)
    finally:
        service.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
