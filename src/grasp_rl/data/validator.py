"""Data validation utilities for robot demonstrations."""

import math
from typing import Any, Dict, List
import numpy as np
from grasp_rl.data.schemas import TrajectorySchema, StateActionSchema


class ValidationError:
    """A single validation error."""

    def __init__(self, episode_id: str, step_index: int, message: str):
        self.episode_id = episode_id
        self.step_index = step_index
        self.message = message

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "episode_id": self.episode_id,
            "step_index": self.step_index,
            "message": self.message,
        }


class ValidationResult:
    """Structured validation result."""

    def __init__(self, valid: bool, errors: List[ValidationError]):
        self.valid = valid
        self.errors = errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "valid": self.valid,
            "error_count": len(self.errors),
            "errors": [e.to_dict() for e in self.errors],
        }


def _check_finite(values: List[float], context: str) -> List[str]:
    """Check that all values are finite (no NaN or inf)."""
    errors = []
    for i, v in enumerate(values):
        if not math.isfinite(v):
            errors.append(f"{context}[{i}] is not finite: {v}")
    return errors


def _check_timestamps_strictly_increasing(timestamps: List[int]) -> List[str]:
    """Check that timestamps are strictly increasing."""
    errors = []
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            errors.append(
                f"Timestamp not strictly increasing at index {i}: "
                f"{timestamps[i]} <= {timestamps[i-1]}"
            )
    return errors


def validate_trajectory_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate trajectory-only demonstration data.

    Args:
        data: Raw data dictionary to validate.

    Returns:
        ValidationResult with errors if any.
    """
    errors = []

    try:
        schema = TrajectorySchema(**data)
    except Exception as e:
        return ValidationResult(valid=False, errors=[ValidationError("", 0, str(e))])

    # Check for non-empty trajectories
    if not schema.trajectories:
        errors.append(ValidationError("", 0, "Trajectories list is empty"))

    # Group by episode_id and check each episode
    episodes: Dict[str, List[TrajectoryStep]] = {}
    for step in schema.trajectories:
        if step.episode_id not in episodes:
            episodes[step.episode_id] = []
        episodes[step.episode_id].append(step)

    for episode_id, steps in episodes.items():
        # Check episode is non-empty
        if not steps:
            errors.append(ValidationError(episode_id, 0, "Episode is empty"))
            continue

        # Sort by timestamp for consistency
        steps_sorted = sorted(steps, key=lambda s: s.timestamp_ms)
        timestamps = [s.timestamp_ms for s in steps_sorted]

        # Check timestamps are strictly increasing
        timestamp_errors = _check_timestamps_strictly_increasing(timestamps)
        for err in timestamp_errors:
            errors.append(ValidationError(episode_id, 0, err))

        # Check all values are finite
        for i, step in enumerate(steps_sorted):
            for field_name, field_value in [
                ("eef_position", step.eef_position),
                ("eef_velocity", step.eef_velocity),
                ("joint_positions", step.joint_positions),
            ]:
                finiteness_errors = _check_finite(
                    field_value, f"{field_name} at step {i}"
                )
                for err in finiteness_errors:
                    errors.append(ValidationError(episode_id, i, err))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_state_action_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate state-action demonstration data.

    Args:
        data: Raw data dictionary to validate.

    Returns:
        ValidationResult with errors if any.
    """
    errors = []

    try:
        schema = StateActionSchema(**data)
    except Exception as e:
        return ValidationResult(valid=False, errors=[ValidationError("", 0, str(e))])

    # Check for non-empty episodes
    if not schema.episodes:
        errors.append(ValidationError("", 0, "Episodes list is empty"))

    for ep_idx, episode in enumerate(schema.episodes):
        episode_id = f"episode_{ep_idx}"

        # Check episode is non-empty
        if not episode.steps:
            errors.append(ValidationError(episode_id, 0, "Episode has no steps"))
            continue

        # Check timestamps are strictly increasing
        timestamps = [s.timestamp_ms for s in episode.steps]
        timestamp_errors = _check_timestamps_strictly_increasing(timestamps)
        for err in timestamp_errors:
            errors.append(ValidationError(episode_id, 0, err))

        # Check all values are finite
        for step_idx, step in enumerate(episode.steps):
            for field_name, field_value in [
                ("joint_positions", step.joint_positions),
                ("joint_velocities", step.joint_velocities),
                ("action", step.action),
            ]:
                finiteness_errors = _check_finite(
                    field_value, f"{field_name} at step {step_idx}"
                )
                for err in finiteness_errors:
                    errors.append(ValidationError(episode_id, step_idx, err))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate demonstration data (auto-detects schema type).

    Args:
        data: Raw data dictionary to validate.

    Returns:
        ValidationResult with errors if any.
    """
    # Auto-detect schema type based on presence of "episodes" key
    if "episodes" in data:
        return validate_state_action_data(data)
    elif "trajectories" in data:
        return validate_trajectory_data(data)
    else:
        return ValidationResult(
            valid=False,
            errors=[ValidationError("", 0, "Unknown data format: missing 'episodes' or 'trajectories' key")],
        )
