"""Jetson server configuration."""

from dataclasses import dataclass, field


@dataclass
class JetsonServerConfig:
    # HTTP API
    host: str = "0.0.0.0"
    http_port: int = 8000
    ws_port: int = 8765

    # Camera
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 10
    jpeg_quality: int = 75
    camera_ids: list[int] = field(default_factory=lambda: [0, 1])

    # LiDAR
    lidar_port: str = "/dev/ttyUSB0"
    lidar_baudrate: int = 921600

    # Safety
    max_throttle: float = 0.5
    heartbeat_timeout_s: float = 2.0
    heartbeat_throttle_zero_s: float = 0.5

    # Sensor streaming rates (Hz)
    stream_camera_fps: int = 15
    stream_lidar_hz: int = 10
    stream_yolo_hz: int = 5
    stream_status_hz: int = 2

    # Models
    yolo_model_path: str = "/home/jetson/models/yolov8n.pt"


config = JetsonServerConfig()
