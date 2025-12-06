"""POST /analyze - 統合解析"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import cv2
import numpy as np

from ..core.camera_manager import camera_manager
from ..core.yolo_detector import YOLODetector
from ..core.segmentation import SegmentationModel
from ..config import config

router = APIRouter()

# モデル初期化（遅延ロード）
_yolo: YOLODetector = None
_segmentation: SegmentationModel = None


def get_yolo():
    global _yolo
    if _yolo is None:
        _yolo = YOLODetector(config.yolo_model_path)
    return _yolo


def get_segmentation():
    global _segmentation
    if _segmentation is None:
        _segmentation = SegmentationModel(config.segmentation_model_path)
    return _segmentation


@router.post("/analyze")
def analyze_scene():
    """YOLO + セグメンテーション + 画像統計を統合"""
    frame = camera_manager.read()

    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    # 画像統計
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # YOLO検出
    yolo = get_yolo()
    objects = yolo.detect(frame) if yolo.is_ready() else []

    # セグメンテーション
    seg = get_segmentation()
    seg_result = seg.analyze(frame) if seg.is_ready() else {
        "road_ratio": 0.0,
        "road_center_x": 0.5,
        "available": False
    }

    return {
        "timestamp": datetime.now().isoformat(),
        "camera": {
            "brightness": round(brightness, 1),
            "blur_score": round(blur_score, 1),
            "is_usable": blur_score > 100 and 30 < brightness < 220
        },
        "detection": {
            "objects": objects,
            "total_count": len(objects)
        },
        "segmentation": seg_result
    }
