"""HTTP Server 設定"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000

    # カメラ設定
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 15
    jpeg_quality: int = 80

    # モデルパス
    yolo_model_path: str = "/home/jetson/models/yolov8n.pt"
    segmentation_model_path: str = "/home/jetson/models/road_segmentation.onnx"

    # 安全制限
    max_throttle: float = 0.5  # 最大50%に制限


config = ServerConfig()
