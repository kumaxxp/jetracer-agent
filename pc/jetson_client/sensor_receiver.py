"""Sensor data receiver and buffer.

Decodes incoming WebSocket sensor data and maintains the latest state
for each sensor type. Thread-safe access for the perception pipeline.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from shared.protocol.messages import WSEnvelope
from shared.protocol.sensor_types import LidarPoint, LidarScan, DetectedObject, VehicleStatus

logger = logging.getLogger(__name__)


@dataclass
class SensorState:
    """Buffered state of all sensors from Jetson."""

    # Camera frames (camera_id -> latest frame)
    frames: dict[int, np.ndarray] = field(default_factory=dict)
    frame_timestamps: dict[int, float] = field(default_factory=dict)

    # LiDAR
    lidar_scan: LidarScan | None = None
    lidar_timestamp: float = 0.0

    # YOLO detections (camera_id -> latest detections)
    yolo_detections: dict[int, list[DetectedObject]] = field(default_factory=dict)
    yolo_timestamps: dict[int, float] = field(default_factory=dict)

    # Vehicle
    vehicle_status: VehicleStatus = field(default_factory=VehicleStatus)
    vehicle_timestamp: float = 0.0

    # Connection
    last_heartbeat: float = 0.0
    jetson_uptime: float = 0.0


class SensorReceiver:
    """Receives and buffers sensor data from Jetson.

    Usage:
        receiver = SensorReceiver()
        client.on_message(receiver.handle_message)

        # Later, from perception pipeline:
        frame = receiver.get_frame(camera_id=0)
        scan = receiver.get_lidar_scan()
    """

    def __init__(self):
        self._state = SensorState()
        self._lock = threading.Lock()
        self._frame_count = 0
        self._scan_count = 0

    def handle_message(self, envelope: WSEnvelope):
        """Handle an incoming sensor message from JetsonClient."""
        msg_type = envelope.type
        payload = envelope.payload
        ts = envelope.timestamp

        if msg_type == "camera_frame":
            self._handle_camera_frame(payload, ts)
        elif msg_type == "lidar_scan":
            self._handle_lidar_scan(payload, ts)
        elif msg_type == "yolo_detections":
            self._handle_yolo_detections(payload, ts)
        elif msg_type == "vehicle_status":
            self._handle_vehicle_status(payload, ts)
        elif msg_type == "heartbeat":
            with self._lock:
                self._state.last_heartbeat = time.time()
                self._state.jetson_uptime = payload.get("uptime_s", 0.0)

    def get_frame(self, camera_id: int = 0) -> np.ndarray | None:
        """Get the latest camera frame (decoded numpy array)."""
        with self._lock:
            return self._state.frames.get(camera_id)

    def get_frame_age(self, camera_id: int = 0) -> float:
        """Get age of latest frame in seconds."""
        with self._lock:
            ts = self._state.frame_timestamps.get(camera_id, 0.0)
            return time.time() - ts if ts > 0 else float("inf")

    def get_lidar_scan(self) -> LidarScan | None:
        """Get the latest LiDAR scan."""
        with self._lock:
            return self._state.lidar_scan

    def get_lidar_age(self) -> float:
        """Get age of latest LiDAR scan in seconds."""
        with self._lock:
            return time.time() - self._state.lidar_timestamp if self._state.lidar_timestamp > 0 else float("inf")

    def get_yolo_detections(self, camera_id: int = 0) -> list[DetectedObject]:
        """Get latest YOLO detections for a camera."""
        with self._lock:
            return self._state.yolo_detections.get(camera_id, [])

    def get_vehicle_status(self) -> VehicleStatus:
        """Get latest vehicle status."""
        with self._lock:
            return self._state.vehicle_status

    def is_connected(self) -> bool:
        """Check if we're receiving data from Jetson (heartbeat within 3s)."""
        with self._lock:
            return (time.time() - self._state.last_heartbeat) < 3.0

    def get_stats(self) -> dict:
        """Get receiver statistics."""
        with self._lock:
            return {
                "connected": self.is_connected(),
                "frame_count": self._frame_count,
                "scan_count": self._scan_count,
                "jetson_uptime": self._state.jetson_uptime,
                "active_cameras": list(self._state.frames.keys()),
            }

    def _handle_camera_frame(self, payload: dict, ts: float):
        """Decode JPEG base64 frame to numpy array."""
        camera_id = payload["camera_id"]
        jpeg_b64 = payload["jpeg_b64"]

        jpeg_bytes = base64.b64decode(jpeg_b64)
        np_arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is not None:
            with self._lock:
                self._state.frames[camera_id] = frame
                self._state.frame_timestamps[camera_id] = ts
                self._frame_count += 1

    def _handle_lidar_scan(self, payload: dict, ts: float):
        """Parse LiDAR scan data."""
        points = [
            LidarPoint(angle=p["angle"], distance=p["distance"], quality=p.get("quality", 0))
            for p in payload.get("points", [])
        ]
        scan = LidarScan(points=points, rpm=payload.get("rpm", 0.0), scan_count=len(points))

        with self._lock:
            self._state.lidar_scan = scan
            self._state.lidar_timestamp = ts
            self._scan_count += 1

    def _handle_yolo_detections(self, payload: dict, ts: float):
        """Parse YOLO detection results."""
        camera_id = payload["camera_id"]
        objects = [
            DetectedObject(
                class_name=o["class_name"],
                confidence=o["confidence"],
                bbox=o["bbox"],
            )
            for o in payload.get("objects", [])
        ]

        with self._lock:
            self._state.yolo_detections[camera_id] = objects
            self._state.yolo_timestamps[camera_id] = ts

    def _handle_vehicle_status(self, payload: dict, ts: float):
        """Parse vehicle status."""
        with self._lock:
            self._state.vehicle_status = VehicleStatus(
                steering=payload.get("steering", 0.0),
                throttle=payload.get("throttle", 0.0),
                connected=payload.get("connected", False),
                mode=payload.get("mode", "idle"),
            )
            self._state.vehicle_timestamp = ts
