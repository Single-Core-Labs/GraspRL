"""Data schemas for demonstrations."""

from pydantic import BaseModel


class TrajectorySchema(BaseModel):
    """Schema for trajectory-only data."""
    pass


class StateActionSchema(BaseModel):
    """Schema for state-action demonstrations."""
    pass
