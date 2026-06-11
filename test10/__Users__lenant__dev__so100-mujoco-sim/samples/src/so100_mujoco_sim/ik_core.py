"""Pure (Qt-free) IK / FK helpers for the SO-100 arm.

These functions were originally defined inline in ``agent_service.py``, which
imports PySide6 at module load. They are extracted here so a *headless* service
(e.g. the real-robot bridge in ``scripts/real_agent_service.py``) can reuse the
exact same MuJoCo forward-kinematics and damped-least-squares inverse-kinematics
without dragging in Qt or a renderer.

Nothing here touches Qt, the renderer, or any hardware — it is plain numpy +
MuJoCo math plus a small XML-preparation helper. ``agent_service.py`` imports
these symbols so the simulation keeps behaving identically.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from so100_mujoco_sim.arm_control import MUJOCO_SO100_PREFIX

# Joints the position IK solves over (everything except the Jaw).
IK_JOINT_NAMES = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
DEFAULT_IK_TOLERANCE = 0.005
DEFAULT_IK_MAX_ITERATIONS = 80
DEFAULT_IK_DAMPING = 1e-4
DEFAULT_IK_STEP_SCALE = 0.5
DEFAULT_IK_MAX_JOINT_STEP = 0.1

# Per-command joint delta clamp used by the sim's ``act`` handler.
DEFAULT_MAX_JOINT_DELTA = 0.15

WRIST_CAMERA_NAME = "wrist_camera"
PREFIXED_WRIST_CAMERA_NAME = f"{MUJOCO_SO100_PREFIX}{WRIST_CAMERA_NAME}"


@dataclass(frozen=True)
class IkResult:
    joint_targets: dict[str, float]
    reached: bool
    position_error: float
    iterations: int


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def interpolate_positions(
    start: dict[str, float],
    end: dict[str, float],
    steps: int,
) -> list[dict[str, float]]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    frames = []
    for step in range(1, steps + 1):
        t = step / steps
        frames.append({
            name: start[name] + (end[name] - start[name]) * t
            for name in start
        })
    return frames


def parse_target_position(raw_position: Any) -> np.ndarray:
    if not isinstance(raw_position, (list, tuple, np.ndarray)):
        raise ValueError("position must be a list of 3 numbers")

    try:
        position = np.asarray(raw_position, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("position must be a list of 3 numbers") from exc

    if position.shape != (3,):
        raise ValueError("position must be a list of 3 numbers")
    if not np.all(np.isfinite(position)):
        raise ValueError("position values must be finite")
    return position


def calculate_joint_targets(
    current: dict[str, float],
    command_joints: dict[str, Any],
    joint_ranges: dict[str, tuple[float, float]],
    mode: str,
    max_delta: float = DEFAULT_MAX_JOINT_DELTA,
) -> dict[str, float]:
    if mode not in {"delta", "absolute"}:
        raise ValueError("mode must be 'delta' or 'absolute'")
    if not isinstance(command_joints, dict) or not command_joints:
        raise ValueError("joints must be a non-empty object")

    targets = dict(current)
    for name, raw_value in command_joints.items():
        if name not in joint_ranges:
            raise ValueError(f"unknown joint '{name}'")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"joint '{name}' value must be numeric") from exc

        low, high = joint_ranges[name]
        desired = current[name] + value if mode == "delta" else value
        desired = clamp(desired, current[name] - max_delta, current[name] + max_delta)
        targets[name] = clamp(desired, low, high)

    return targets


def resolve_end_effector(model: mujoco.MjModel) -> tuple[str, int]:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "so100_ee_site")
    if site_id != -1:
        return "site", site_id

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "so100_Moving_Jaw")
    if body_id != -1:
        return "body", body_id

    raise ValueError("end-effector site/body not found")


def end_effector_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_type: str,
    target_id: int,
) -> np.ndarray:
    if target_type == "site":
        return data.site(target_id).xpos.copy()
    if target_type == "body":
        return data.body(target_id).xpos.copy()
    raise ValueError(f"unsupported end-effector target type '{target_type}'")


def end_effector_position_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_type: str,
    target_id: int,
) -> np.ndarray:
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    if target_type == "site":
        mujoco.mj_jacSite(model, data, jacp, jacr, target_id)
    elif target_type == "body":
        mujoco.mj_jacBody(model, data, jacp, jacr, target_id)
    else:
        raise ValueError(f"unsupported end-effector target type '{target_type}'")
    return jacp


def solve_ik_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    current_joint_targets: dict[str, float],
    joint_ranges: dict[str, tuple[float, float]],
    target_position: np.ndarray,
    *,
    joint_names: tuple[str, ...] = IK_JOINT_NAMES,
    tolerance: float = DEFAULT_IK_TOLERANCE,
    max_iterations: int = DEFAULT_IK_MAX_ITERATIONS,
    damping: float = DEFAULT_IK_DAMPING,
    step_scale: float = DEFAULT_IK_STEP_SCALE,
    max_joint_step: float = DEFAULT_IK_MAX_JOINT_STEP,
) -> IkResult:
    if max_iterations < 0:
        raise ValueError("max_iterations must be >= 0")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if damping <= 0:
        raise ValueError("damping must be positive")
    if step_scale <= 0:
        raise ValueError("step_scale must be positive")
    if max_joint_step <= 0:
        raise ValueError("max_joint_step must be positive")

    target_position = parse_target_position(target_position)
    target_type, target_id = resolve_end_effector(model)
    qpos_addrs: list[int] = []
    dof_addrs: list[int] = []
    for name in joint_names:
        if name not in current_joint_targets:
            raise ValueError(f"missing current target for joint '{name}'")
        if name not in joint_ranges:
            raise ValueError(f"missing range for joint '{name}'")
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            MUJOCO_SO100_PREFIX + name,
        )
        if joint_id == -1:
            raise ValueError(f"joint '{name}' not found")
        qpos_addrs.append(int(model.jnt_qposadr[joint_id]))
        dof_addrs.append(int(model.jnt_dofadr[joint_id]))

    saved_qpos = data.qpos.copy()
    saved_qvel = data.qvel.copy()
    saved_ctrl = data.ctrl.copy()
    saved_time = float(data.time)
    best_qpos = np.array([current_joint_targets[name] for name in joint_names], dtype=float)
    best_error = float("inf")
    iterations_done = 0

    try:
        for name, value in current_joint_targets.items():
            joint_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                MUJOCO_SO100_PREFIX + name,
            )
            if joint_id == -1:
                continue
            qpos_addr = model.jnt_qposadr[joint_id]
            low, high = joint_ranges[name]
            data.qpos[qpos_addr] = clamp(float(value), low, high)
        data.qvel[:] = 0.0

        for iteration in range(max_iterations + 1):
            iterations_done = iteration
            mujoco.mj_forward(model, data)
            error_vector = target_position - end_effector_position(model, data, target_type, target_id)
            position_error = float(np.linalg.norm(error_vector))
            if position_error < best_error:
                best_error = position_error
                best_qpos = np.array([data.qpos[qpos_addr] for qpos_addr in qpos_addrs], dtype=float)
            if position_error <= tolerance:
                joint_targets = dict(current_joint_targets)
                for name, value in zip(joint_names, best_qpos):
                    joint_targets[name] = float(value)
                return IkResult(
                    joint_targets=joint_targets,
                    reached=True,
                    position_error=position_error,
                    iterations=iteration,
                )
            if iteration == max_iterations:
                break

            jacobian = end_effector_position_jacobian(model, data, target_type, target_id)[:, dof_addrs]
            damped = jacobian @ jacobian.T + damping * np.eye(3)
            joint_step = jacobian.T @ np.linalg.solve(damped, error_vector)
            joint_step = np.clip(joint_step * step_scale, -max_joint_step, max_joint_step)
            if not np.all(np.isfinite(joint_step)):
                break

            for name, qpos_addr, delta in zip(joint_names, qpos_addrs, joint_step):
                low, high = joint_ranges[name]
                data.qpos[qpos_addr] = clamp(float(data.qpos[qpos_addr] + delta), low, high)

        joint_targets = dict(current_joint_targets)
        for name, value in zip(joint_names, best_qpos):
            joint_targets[name] = float(value)
        return IkResult(
            joint_targets=joint_targets,
            reached=False,
            position_error=best_error,
            iterations=iterations_done,
        )
    finally:
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.ctrl[:] = saved_ctrl
        data.time = saved_time
        mujoco.mj_forward(model, data)


def prepare_agent_model_xml(source_dir: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    arm_tree = ET.parse(source_dir / "so_arm100.xml")
    arm_root = arm_tree.getroot()
    compiler = arm_root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str((source_dir / "assets").resolve()))

    fixed_jaw = arm_root.find(".//body[@name='Fixed_Jaw']")
    if fixed_jaw is None:
        raise ValueError("Fixed_Jaw body not found in SO-100 XML")

    camera = fixed_jaw.find(f"camera[@name='{WRIST_CAMERA_NAME}']")
    if camera is None:
        camera = ET.Element(
            "camera",
            {
                "name": WRIST_CAMERA_NAME,
                "pos": "0 0.045 0.12",
                "zaxis": "0 0.8 0.6",
                "fovy": "75",
            },
        )
        moving_jaw_index = next(
            (
                index
                for index, child in enumerate(list(fixed_jaw))
                if child.tag == "body" and child.get("name") == "Moving_Jaw"
            ),
            len(fixed_jaw),
        )
        fixed_jaw.insert(moving_jaw_index, camera)

    arm_path = output_dir / "so_arm100_agent.xml"
    ET.indent(arm_tree, space="  ")
    arm_tree.write(arm_path, encoding="unicode")

    scene_tree = ET.parse(source_dir / "sim_scene.xml")
    scene_root = scene_tree.getroot()
    model_asset = scene_root.find(".//asset/model[@name='so_arm100']")
    if model_asset is None:
        raise ValueError("so_arm100 model asset not found in scene XML")
    model_asset.set("file", str(arm_path))

    scene_path = output_dir / "sim_scene_agent.xml"
    ET.indent(scene_tree, space="  ")
    scene_tree.write(scene_path, encoding="unicode")
    return scene_path
