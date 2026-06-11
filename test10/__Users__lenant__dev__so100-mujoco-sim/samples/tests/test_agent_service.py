from pathlib import Path

import pytest
import mujoco
import numpy as np

from so100_mujoco_sim.ik_core import (
    DEFAULT_IK_TOLERANCE,
    end_effector_position,
    parse_target_position,
    calculate_joint_targets,
    interpolate_positions,
    prepare_agent_model_xml,
    PREFIXED_WRIST_CAMERA_NAME,
    resolve_end_effector,
    solve_ik_position,
)
from so100_mujoco_sim.arm_control import MUJOCO_SO100_PREFIX, joints_from_model


PREFERRED_HOME = {
    "Rotation": 0.0,
    "Pitch": -1.57079,
    "Elbow": 1.57079,
    "Wrist_Pitch": 1.57079,
    "Wrist_Roll": -1.57079,
    "Jaw": 0.0,
}


def load_agent_model(tmp_path: Path) -> tuple[
    mujoco.MjModel,
    mujoco.MjData,
    dict[str, tuple[float, float]],
    dict[str, float],
]:
    model_path = prepare_agent_model_xml(
        Path("src/so100_mujoco_sim/xml"),
        tmp_path,
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joints = joints_from_model(model)
    joint_ranges = {joint.name: joint.range for joint in joints}
    home_targets = {
        joint.name: max(joint.range[0], min(PREFERRED_HOME.get(joint.name, 0.0), joint.range[1]))
        for joint in joints
    }
    return model, data, joint_ranges, home_targets


def set_model_joint_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_targets: dict[str, float],
) -> None:
    for name, value in joint_targets.items():
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            MUJOCO_SO100_PREFIX + name,
        )
        if joint_id != -1:
            data.qpos[model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(model, data)


def current_end_effector_position(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    target_type, target_id = resolve_end_effector(model)
    return end_effector_position(model, data, target_type, target_id)


def test_calculate_joint_targets_clamps_delta_and_range():
    current = {"Rotation": 0.0, "Jaw": 1.95}
    ranges = {"Rotation": (-2.2, 2.2), "Jaw": (-0.2, 2.0)}

    targets = calculate_joint_targets(
        current,
        {"Rotation": 1.0, "Jaw": 1.0},
        ranges,
        "delta",
        max_delta=0.15,
    )

    assert targets["Rotation"] == pytest.approx(0.15)
    assert targets["Jaw"] == pytest.approx(2.0)


def test_calculate_joint_targets_clamps_absolute_by_command_delta():
    current = {"Rotation": 0.4}
    ranges = {"Rotation": (-2.2, 2.2)}

    targets = calculate_joint_targets(
        current,
        {"Rotation": -2.0},
        ranges,
        "absolute",
        max_delta=0.2,
    )

    assert targets["Rotation"] == pytest.approx(0.2)


def test_calculate_joint_targets_rejects_unknown_joint():
    with pytest.raises(ValueError, match="unknown joint"):
        calculate_joint_targets(
            {"Rotation": 0.0},
            {"Pitch": 0.1},
            {"Rotation": (-2.2, 2.2)},
            "delta",
        )


def test_interpolate_positions_returns_requested_frames():
    frames = interpolate_positions(
        {"Rotation": 0.0, "Jaw": 1.0},
        {"Rotation": 0.3, "Jaw": 0.4},
        3,
    )

    assert len(frames) == 3
    assert frames[0]["Rotation"] == pytest.approx(0.1)
    assert frames[0]["Jaw"] == pytest.approx(0.8)
    assert frames[1]["Rotation"] == pytest.approx(0.2)
    assert frames[1]["Jaw"] == pytest.approx(0.6)
    assert frames[2]["Rotation"] == pytest.approx(0.3)
    assert frames[2]["Jaw"] == pytest.approx(0.4)


def test_parse_target_position_accepts_three_finite_numbers():
    position = parse_target_position([0.1, "-0.2", 0.15])

    np.testing.assert_allclose(position, np.array([0.1, -0.2, 0.15]))


@pytest.mark.parametrize(
    "raw_position",
    [
        None,
        [],
        [0.0, 0.1],
        [0.0, 0.1, 0.2, 0.3],
        [0.0, "x", 0.2],
        [0.0, float("nan"), 0.2],
        [0.0, float("inf"), 0.2],
        "0.0",
    ],
)
def test_parse_target_position_rejects_invalid_values(raw_position):
    with pytest.raises(ValueError, match="position"):
        parse_target_position(raw_position)


def test_solve_ik_position_reaches_fk_generated_target(tmp_path):
    model, data, joint_ranges, home_targets = load_agent_model(tmp_path)
    goal_targets = dict(home_targets)
    goal_targets["Rotation"] = 0.08
    goal_targets["Pitch"] = -1.50
    goal_targets["Elbow"] = 1.50
    goal_targets["Wrist_Pitch"] = 1.45

    set_model_joint_positions(model, data, goal_targets)
    target_position = current_end_effector_position(model, data)
    qpos_before = data.qpos.copy()

    result = solve_ik_position(
        model,
        data,
        home_targets,
        joint_ranges,
        target_position,
        max_iterations=120,
    )

    assert result.reached
    assert result.position_error <= DEFAULT_IK_TOLERANCE
    np.testing.assert_allclose(data.qpos, qpos_before)

    set_model_joint_positions(model, data, result.joint_targets)
    reached_position = current_end_effector_position(model, data)
    assert np.linalg.norm(reached_position - target_position) <= DEFAULT_IK_TOLERANCE


def test_solve_ik_position_preserves_jaw(tmp_path):
    model, data, joint_ranges, home_targets = load_agent_model(tmp_path)
    home_targets["Jaw"] = 1.2
    set_model_joint_positions(model, data, home_targets)
    target_position = current_end_effector_position(model, data)

    result = solve_ik_position(
        model,
        data,
        home_targets,
        joint_ranges,
        target_position,
    )

    assert result.reached
    assert result.joint_targets["Jaw"] == pytest.approx(1.2)


def test_solve_ik_position_returns_best_effort_for_unreachable_target(tmp_path):
    model, data, joint_ranges, home_targets = load_agent_model(tmp_path)

    result = solve_ik_position(
        model,
        data,
        home_targets,
        joint_ranges,
        np.array([10.0, 10.0, 10.0]),
        max_iterations=5,
    )

    assert not result.reached
    assert np.isfinite(result.position_error)
    for name, value in result.joint_targets.items():
        assert np.isfinite(value)
        low, high = joint_ranges[name]
        assert low <= value <= high


def test_prepare_agent_model_xml_adds_prefixed_wrist_camera(tmp_path):
    model_path = prepare_agent_model_xml(
        Path("src/so100_mujoco_sim/xml"),
        tmp_path,
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))

    camera_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        PREFIXED_WRIST_CAMERA_NAME,
    )

    assert camera_id != -1
