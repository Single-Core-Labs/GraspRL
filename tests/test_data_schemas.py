"""Tests for data schemas."""

import pytest
from pydantic import ValidationError
from grasp_rl.data.schemas import TrajectorySchema, TrajectoryStep, StateActionSchema, StateActionStep, Episode


def test_trajectory_step_valid():
    """Test valid trajectory step."""
    step = TrajectoryStep(
        episode_id="ep1",
        timestamp_ms=1000,
        eef_position=[0.1, 0.2, 0.3],
        eef_velocity=[0.0, 0.0, 0.0],
        joint_positions=[0.0, 0.5, 1.0],
    )
    assert step.episode_id == "ep1"
    assert step.timestamp_ms == 1000


def test_trajectory_step_invalid_eef_position_length():
    """Test trajectory step with invalid eef_position length."""
    with pytest.raises(ValidationError):
        TrajectoryStep(
            episode_id="ep1",
            timestamp_ms=1000,
            eef_position=[0.1, 0.2],  # Wrong length
            eef_velocity=[0.0, 0.0, 0.0],
            joint_positions=[0.0, 0.5, 1.0],
        )


def test_trajectory_schema_valid():
    """Test valid trajectory schema."""
    schema = TrajectorySchema(
        robot_id="robot1",
        coordinate_frame="base",
        position_unit="m",
        angle_unit="rad",
        trajectories=[
            TrajectoryStep(
                episode_id="ep1",
                timestamp_ms=1000,
                eef_position=[0.1, 0.2, 0.3],
                eef_velocity=[0.0, 0.0, 0.0],
                joint_positions=[0.0, 0.5, 1.0],
            )
        ],
    )
    assert schema.schema_version == "1.0"
    assert schema.robot_id == "robot1"


def test_trajectory_schema_invalid_schema_version():
    """Test trajectory schema with unsupported schema version."""
    with pytest.raises(ValidationError, match="Unsupported schema version"):
        TrajectorySchema(
            schema_version="2.0",
            robot_id="robot1",
            coordinate_frame="base",
            position_unit="m",
            angle_unit="rad",
            trajectories=[],
        )


def test_trajectory_schema_invalid_position_unit():
    """Test trajectory schema with unknown position unit."""
    with pytest.raises(ValidationError, match="Unknown position unit"):
        TrajectorySchema(
            robot_id="robot1",
            coordinate_frame="base",
            position_unit="inches",  # Invalid
            angle_unit="rad",
            trajectories=[],
        )


def test_trajectory_schema_invalid_angle_unit():
    """Test trajectory schema with unknown angle unit."""
    with pytest.raises(ValidationError, match="Unknown angle unit"):
        TrajectorySchema(
            robot_id="robot1",
            coordinate_frame="base",
            position_unit="m",
            angle_unit="grad",  # Invalid
            trajectories=[],
        )


def test_trajectory_schema_invalid_coordinate_frame():
    """Test trajectory schema with unknown coordinate frame."""
    with pytest.raises(ValidationError, match="Unknown coordinate frame"):
        TrajectorySchema(
            robot_id="robot1",
            coordinate_frame="camera",  # Invalid
            position_unit="m",
            angle_unit="rad",
            trajectories=[],
        )


def test_state_action_step_valid():
    """Test valid state-action step."""
    step = StateActionStep(
        timestamp_ms=1000,
        joint_positions=[0.0, 0.5, 1.0],
        joint_velocities=[0.1, 0.2, 0.3],
        action=[0.0, 0.0, 0.0],
        terminated=False,
    )
    assert step.timestamp_ms == 1000
    assert not step.terminated


def test_state_action_schema_valid():
    """Test valid state-action schema."""
    schema = StateActionSchema(
        robot_id="robot1",
        action_mode="joint_delta",
        episodes=[
            Episode(
                steps=[
                    StateActionStep(
                        timestamp_ms=1000,
                        joint_positions=[0.0, 0.5, 1.0],
                        joint_velocities=[0.1, 0.2, 0.3],
                        action=[0.0, 0.0, 0.0],
                        terminated=False,
                    )
                ]
            )
        ],
    )
    assert schema.schema_version == "1.0"
    assert schema.action_mode == "joint_delta"


def test_state_action_schema_invalid_action_mode():
    """Test state-action schema with unknown action mode."""
    with pytest.raises(ValidationError, match="Unknown action mode"):
        StateActionSchema(
            robot_id="robot1",
            action_mode="torque",  # Invalid
            episodes=[],
        )
