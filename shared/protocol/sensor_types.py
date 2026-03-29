"""Sensor data types shared between PC and Jetson."""

from __future__ import annotations

from pydantic import BaseModel, Field


class CameraFrame(BaseModel):
    """A single camera frame transmitted as JPEG."""

    camera_id: int
    jpeg_b64: str
    width: int
    height: int


class LidarPoint(BaseModel):
    """A single LiDAR measurement point."""

    angle: float = Field(description="Angle in degrees (0-360)")
    distance: float = Field(description="Distance in mm")
    quality: int = Field(default=0, description="Signal quality (0-255)")


class LidarScan(BaseModel):
    """A complete 360-degree LiDAR scan."""

    points: list[LidarPoint]
    rpm: float = Field(description="Rotation speed in RPM")
    scan_count: int = Field(default=0, description="Total points in this scan")


class DetectedObject(BaseModel):
    """A single YOLO-detected object."""

    class_name: str
    confidence: float
    bbox: list[float] = Field(description="[x1, y1, x2, y2] normalized 0-1")


class YoloDetection(BaseModel):
    """YOLO detection results for a camera frame."""

    camera_id: int
    objects: list[DetectedObject]
    frame_seq: int = Field(default=0, description="Corresponding camera frame seq")


class VehicleStatus(BaseModel):
    """Current vehicle state."""

    steering: float = Field(default=0.0, description="Steering angle (-1.0 to 1.0)")
    throttle: float = Field(default=0.0, description="Throttle value (-1.0 to 1.0)")
    connected: bool = Field(default=False, description="Vehicle hardware connected")
    mode: str = Field(default="idle", description="Current driving mode")
