"""WebSocket message envelope shared between PC and Jetson."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class WSEnvelope(BaseModel):
    """Common envelope for all WebSocket messages."""

    type: str
    timestamp: float = Field(default_factory=time.time)
    seq: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)

    def to_json_bytes(self) -> bytes:
        return self.model_dump_json().encode()

    @classmethod
    def from_json_bytes(cls, data: bytes) -> WSEnvelope:
        return cls.model_validate_json(data)


class WSMessage:
    """Factory for building typed WSEnvelope messages."""

    _seq_counter: int = 0

    @classmethod
    def _next_seq(cls) -> int:
        cls._seq_counter += 1
        return cls._seq_counter

    @classmethod
    def camera_frame(cls, camera_id: int, jpeg_b64: str, width: int, height: int) -> WSEnvelope:
        return WSEnvelope(
            type="camera_frame",
            seq=cls._next_seq(),
            payload={"camera_id": camera_id, "jpeg_b64": jpeg_b64, "width": width, "height": height},
        )

    @classmethod
    def lidar_scan(cls, points: list[dict], rpm: float) -> WSEnvelope:
        return WSEnvelope(
            type="lidar_scan",
            seq=cls._next_seq(),
            payload={"points": points, "rpm": rpm},
        )

    @classmethod
    def yolo_detections(cls, camera_id: int, objects: list[dict]) -> WSEnvelope:
        return WSEnvelope(
            type="yolo_detections",
            seq=cls._next_seq(),
            payload={"camera_id": camera_id, "objects": objects},
        )

    @classmethod
    def vehicle_status(cls, steering: float, throttle: float, connected: bool) -> WSEnvelope:
        return WSEnvelope(
            type="vehicle_status",
            seq=cls._next_seq(),
            payload={"steering": steering, "throttle": throttle, "connected": connected},
        )

    @classmethod
    def control_cmd(cls, steering: float, throttle: float) -> WSEnvelope:
        return WSEnvelope(
            type="control_cmd",
            seq=cls._next_seq(),
            payload={"steering": steering, "throttle": throttle},
        )

    @classmethod
    def emergency_stop(cls) -> WSEnvelope:
        return WSEnvelope(
            type="emergency_stop",
            seq=cls._next_seq(),
        )

    @classmethod
    def mode_change(cls, mode: str) -> WSEnvelope:
        return WSEnvelope(
            type="mode_change",
            seq=cls._next_seq(),
            payload={"mode": mode},
        )

    @classmethod
    def heartbeat(cls, uptime_s: float) -> WSEnvelope:
        return WSEnvelope(
            type="heartbeat",
            seq=cls._next_seq(),
            payload={"uptime_s": uptime_s},
        )
