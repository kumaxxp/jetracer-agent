"""YOLO物体検出"""
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path


class YOLODetector:
    def __init__(self, model_path: str):
        self.model = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """モデルロード"""
        try:
            from ultralytics import YOLO
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # デフォルトモデルをダウンロード
                self.model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
            self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """物体検出"""
        if self.model is None or frame is None:
            return []

        results = self.model(frame, verbose=False)
        objects = []

        for r in results:
            for box in r.boxes:
                bbox = box.xyxy[0].tolist()
                obj = {
                    "class": self.model.names[int(box.cls[0])],
                    "confidence": round(float(box.conf[0]), 3),
                    "bbox": [round(x, 1) for x in bbox],
                }
                # 距離推定（bbox下端のY座標から簡易推定）
                bbox_bottom = bbox[3]
                obj["distance_est"] = self._estimate_distance(bbox_bottom, frame.shape[0])
                objects.append(obj)

        return objects

    def _estimate_distance(self, bbox_bottom: float, frame_height: int) -> float:
        """簡易距離推定（bbox下端が画面下に近いほど近い）"""
        # 画面下端 = 近い（0.3m）、画面中央 = 遠い（3m）
        ratio = bbox_bottom / frame_height
        distance = 3.0 - (ratio * 2.7)  # 0.3m ~ 3.0m
        return round(max(0.3, distance), 2)

    def is_ready(self) -> bool:
        return self.model is not None
