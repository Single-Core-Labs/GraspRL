"""Data schemas for robot demonstrations using Pydantic v2."""

from typing import List
from pydantic import BaseModel, Field, field_validator, model_validator


class TrajectoryStep(BaseModel):
    """Single step in a trajectory-only recording."""

    episode_id: str
    timestamp_ms: int = Field(ge=0)
    eef_position: List[float] = Field(min_length=3, max_length=3)
    eef_velocity: List[float] = Field(min_length=3, max_length=3)
    joint_positions: List[float]


class TrajectorySchema(BaseModel):
    """Schema for trajectory-only demonstration data (schema_version 1.0)."""

    schema_version: str = Field(default="1.0")
    robot_id: str
    coordinate_frame: str
    position_unit: str
    angle_unit: str
    trajectories: List[TrajectoryStep]

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        """Ensure schema version is supported."""
        if v != "1.0":
            raise ValueError(f"Unsupported schema version: {v}")
        return v

    @field_validator("position_unit")
    @classmethod
    def check_position_unit(cls, v: str) -> str:
        """Ensure position unit is known."""
        valid_units = {"m", "mm", "cm"}
        if v not in valid_units:
            raise ValueError(f"Unknown position unit: {v}")
        return v

    @field_validator("angle_unit")
    @classmethod
    def check_angle_unit(cls, v: str) -> str:
        """Ensure angle unit is known."""
        valid_units = {"rad", "deg"}
        if v not in valid_units:
            raise ValueError(f"Unknown angle unit: {v}")
        return v

    @field_validator("coordinate_frame")
    @classmethod
    def check_coordinate_frame(cls, v: str) -> str:
        """Ensure coordinate frame is known."""
        valid_frames = {"base", "world", "tcp"}
        if v not in valid_frames:
            raise ValueError(f"Unknown coordinate frame: {v}")
        return v


class StateActionStep(BaseModel):
    """Single step in a state-action demonstration."""

    timestamp_ms: int = Field(ge=0)
    joint_positions: List[float]
    joint_velocities: List[float]
    action: List[float]
    terminated: bool


class Episode(BaseModel):
    """Single episode with state-action steps."""

    steps: List[StateActionStep]


class StateActionSchema(BaseModel):
    """Schema for state-action demonstration data (schema_version 1.0)."""

    schema_version: str = Field(default="1.0")
    robot_id: str
    action_mode: str
    episodes: List[Episode]

    @field_validator("schema_version")
    @classmethod
    def check_schema_version(cls, v: str) -> str:
        """Ensure schema version is supported."""
        if v != "1.0":
            raise ValueError(f"Unsupported schema version: {v}")
        return v

    @field_validator("action_mode")
    @classmethod
    def check_action_mode(cls, v: str) -> str:
        """Ensure action mode is known."""
        valid_modes = {"joint_position_target", "joint_delta", "eef_delta"}
        if v not in valid_modes:
            raise ValueError(f"Unknown action mode: {v}")
        return v
