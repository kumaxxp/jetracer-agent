"""カメラ管理（シングルトン）"""
import cv2
import numpy as np
from typing import Optional
import threading


class CameraManager:
    _instance: Optional['CameraManager'] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._cap = None
        self._frame = None
        self._frame_lock = threading.Lock()

    def start(self, width: int = 640, height: int = 480, fps: int = 15):
        """カメラ起動"""
        if self._cap is not None:
            return True

        # GStreamerパイプライン（CSIカメラ用）
        gst_pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )

        self._cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self._cap.isOpened():
            # フォールバック: USB カメラ
            self._cap = cv2.VideoCapture(0)
            if self._cap.isOpened():
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._cap.set(cv2.CAP_PROP_FPS, fps)

        return self._cap.isOpened()

    def read(self) -> Optional[np.ndarray]:
        """フレーム取得"""
        if self._cap is None or not self._cap.isOpened():
            return None

        ret, frame = self._cap.read()
        if ret:
            with self._frame_lock:
                self._frame = frame.copy()
            return frame
        return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """最新フレーム取得（読み取りなし）"""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def is_ready(self) -> bool:
        """カメラ準備完了確認"""
        return self._cap is not None and self._cap.isOpened()

    def stop(self):
        """カメラ停止"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


# シングルトンインスタンス
camera_manager = CameraManager()
