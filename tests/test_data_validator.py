"""Tests for data validator."""

import json
import pytest
from grasp_rl.data.validator import (
    validate_data,
    validate_trajectory_data,
    validate_state_action_data,
    ValidationError,
    ValidationResult,
)


def test_validate_trajectory_data_valid():
    """Test validation of valid trajectory data."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "coordinate_frame": "base",
        "position_unit": "m",
        "angle_unit": "rad",
        "trajectories": [
            {
                "episode_id": "ep1",
                "timestamp_ms": 1000,
                "eef_position": [0.1, 0.2, 0.3],
                "eef_velocity": [0.0, 0.0, 0.0],
                "joint_positions": [0.0, 0.5, 1.0],
            },
            {
                "episode_id": "ep1",
                "timestamp_ms": 2000,
                "eef_position": [0.15, 0.25, 0.35],
                "eef_velocity": [0.01, 0.01, 0.01],
                "joint_positions": [0.1, 0.6, 1.1],
            },
        ],
    }
    result = validate_trajectory_data(data)
    assert result.valid
    assert len(result.errors) == 0


def test_validate_trajectory_data_empty_trajectories():
    """Test validation with empty trajectories list."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "coordinate_frame": "base",
        "position_unit": "m",
        "angle_unit": "rad",
        "trajectories": [],
    }
    result = validate_trajectory_data(data)
    assert not result.valid
    assert any("empty" in err.message.lower() for err in result.errors)


def test_validate_trajectory_data_non_finite_values():
    """Test validation with NaN values."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "coordinate_frame": "base",
        "position_unit": "m",
        "angle_unit": "rad",
        "trajectories": [
            {
                "episode_id": "ep1",
                "timestamp_ms": 1000,
                "eef_position": [float("nan"), 0.2, 0.3],
                "eef_velocity": [0.0, 0.0, 0.0],
                "joint_positions": [0.0, 0.5, 1.0],
            }
        ],
    }
    result = validate_trajectory_data(data)
    assert not result.valid
    assert any("not finite" in err.message for err in result.errors)


def test_validate_trajectory_data_non_increasing_timestamps():
    """Test validation with non-increasing timestamps."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "coordinate_frame": "base",
        "position_unit": "m",
        "angle_unit": "rad",
        "trajectories": [
            {
                "episode_id": "ep1",
                "timestamp_ms": 2000,
                "eef_position": [0.1, 0.2, 0.3],
                "eef_velocity": [0.0, 0.0, 0.0],
                "joint_positions": [0.0, 0.5, 1.0],
            },
            {
                "episode_id": "ep1",
                "timestamp_ms": 1000,  # Decreasing
                "eef_position": [0.15, 0.25, 0.35],
                "eef_velocity": [0.01, 0.01, 0.01],
                "joint_positions": [0.1, 0.6, 1.1],
            },
        ],
    }
    result = validate_trajectory_data(data)
    assert not result.valid
    assert any("strictly increasing" in err.message for err in result.errors)


def test_validate_state_action_data_valid():
    """Test validation of valid state-action data."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "action_mode": "joint_delta",
        "episodes": [
            {
                "steps": [
                    {
                        "timestamp_ms": 1000,
                        "joint_positions": [0.0, 0.5, 1.0],
                        "joint_velocities": [0.1, 0.2, 0.3],
                        "action": [0.0, 0.0, 0.0],
                        "terminated": False,
                    },
                    {
                        "timestamp_ms": 2000,
                        "joint_positions": [0.1, 0.6, 1.1],
                        "joint_velocities": [0.15, 0.25, 0.35],
                        "action": [0.01, 0.01, 0.01],
                        "terminated": False,
                    },
                ]
            }
        ],
    }
    result = validate_state_action_data(data)
    assert result.valid
    assert len(result.errors) == 0


def test_validate_state_action_data_empty_episodes():
    """Test validation with empty episodes list."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "action_mode": "joint_delta",
        "episodes": [],
    }
    result = validate_state_action_data(data)
    assert not result.valid
    assert any("empty" in err.message.lower() for err in result.errors)


def test_validate_state_action_data_infinite_values():
    """Test validation with infinite values."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "action_mode": "joint_delta",
        "episodes": [
            {
                "steps": [
                    {
                        "timestamp_ms": 1000,
                        "joint_positions": [float("inf"), 0.5, 1.0],
                        "joint_velocities": [0.1, 0.2, 0.3],
                        "action": [0.0, 0.0, 0.0],
                        "terminated": False,
                    }
                ]
            }
        ],
    }
    result = validate_state_action_data(data)
    assert not result.valid
    assert any("not finite" in err.message for err in result.errors)


def test_validate_data_auto_detect_trajectory():
    """Test auto-detection of trajectory schema."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "coordinate_frame": "base",
        "position_unit": "m",
        "angle_unit": "rad",
        "trajectories": [
            {
                "episode_id": "ep1",
                "timestamp_ms": 1000,
                "eef_position": [0.1, 0.2, 0.3],
                "eef_velocity": [0.0, 0.0, 0.0],
                "joint_positions": [0.0, 0.5, 1.0],
            }
        ],
    }
    result = validate_data(data)
    assert result.valid


def test_validate_data_auto_detect_state_action():
    """Test auto-detection of state-action schema."""
    data = {
        "schema_version": "1.0",
        "robot_id": "robot1",
        "action_mode": "joint_delta",
        "episodes": [
            {
                "steps": [
                    {
                        "timestamp_ms": 1000,
                        "joint_positions": [0.0, 0.5, 1.0],
                        "joint_velocities": [0.1, 0.2, 0.3],
                        "action": [0.0, 0.0, 0.0],
                        "terminated": False,
                    }
                ]
            }
        ],
    }
    result = validate_data(data)
    assert result.valid


def test_validate_data_unknown_format():
    """Test validation with unknown data format."""
    data = {"robot_id": "robot1", "unknown_key": "value"}
    result = validate_data(data)
    assert not result.valid
    assert any("Unknown data format" in err.message for err in result.errors)


def test_validation_result_to_dict():
    """Test ValidationResult serialization."""
    errors = [
        ValidationError("ep1", 0, "Test error 1"),
        ValidationError("ep2", 5, "Test error 2"),
    ]
    result = ValidationResult(valid=False, errors=errors)
    result_dict = result.to_dict()
    assert result_dict["valid"] is False
    assert result_dict["error_count"] == 2
    assert len(result_dict["errors"]) == 2
