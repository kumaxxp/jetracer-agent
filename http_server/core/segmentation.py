"""セグメンテーション"""
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path


class SegmentationModel:
    def __init__(self, model_path: str):
        self.net = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """ONNXモデルロード"""
        try:
            if Path(self.model_path).exists():
                self.net = cv2.dnn.readNetFromONNX(self.model_path)
                # CUDAが利用可能な場合は使用
                try:
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
                except Exception:
                    # CUDAが使えない場合はCPUフォールバック
                    pass
        except Exception as e:
            print(f"[Segmentation] Failed to load model: {e}")
            self.net = None

    def analyze(self, frame: np.ndarray) -> Dict[str, Any]:
        """セグメンテーション解析"""
        if self.net is None or frame is None:
            return {
                "road_ratio": 0.0,
                "road_center_x": 0.5,
                "available": False
            }

        try:
            # 前処理
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (320, 240), swapRB=True
            )

            # 推論
            self.net.setInput(blob)
            output = self.net.forward()

            # 後処理
            mask = np.argmax(output[0], axis=0)
            road_mask = (mask == 1)  # 1 = ROAD

            road_ratio = road_mask.sum() / road_mask.size

            # 走行可能領域の重心X
            if road_mask.any():
                road_center_x = np.where(road_mask)[1].mean() / road_mask.shape[1]
            else:
                road_center_x = 0.5

            return {
                "road_ratio": round(float(road_ratio), 3),
                "road_center_x": round(float(road_center_x), 3),
                "available": True
            }
        except Exception as e:
            print(f"[Segmentation] Error: {e}")
            return {
                "road_ratio": 0.0,
                "road_center_x": 0.5,
                "available": False
            }

    def is_ready(self) -> bool:
        return self.net is not None
