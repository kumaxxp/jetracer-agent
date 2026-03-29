"""Command types sent from PC to Jetson."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class DrivingMode(str, Enum):
    IDLE = "idle"
    MANUAL = "manual"
    AUTO = "auto"


class ControlCommand(BaseModel):
    """Steering and throttle command from PC to Jetson."""

    steering: float = Field(ge=-1.0, le=1.0, description="Steering (-1.0=left, 1.0=right)")
    throttle: float = Field(ge=-1.0, le=1.0, description="Throttle (-1.0=reverse, 1.0=forward)")


class EmergencyStop(BaseModel):
    """Emergency stop command. No parameters needed."""

    reason: str = Field(default="manual", description="Reason for emergency stop")


class ModeChange(BaseModel):
    """Change the driving mode on Jetson."""

    mode: DrivingMode


class SensorConfig(BaseModel):
    """Configure sensor streaming rates."""

    camera_fps: int = Field(default=15, ge=1, le=30)
    lidar_hz: int = Field(default=10, ge=1, le=15)
    yolo_hz: int = Field(default=5, ge=1, le=15)
    status_hz: int = Field(default=2, ge=1, le=10)
