"""カメラ管理（シングルトン）- jetracer_minimal JetCamera使用"""
import sys
import cv2
import numpy as np
from typing import Optional
import threading
import time

# jetracer_minimalのcamera.pyをインポート
sys.path.insert(0, '/home/jetson/projects/jetracer_minimal')
try:
    from camera import JetCamera
    _JETCAM_AVAILABLE = True
    print("[CameraManager] JetCamera from jetracer_minimal loaded")
except ImportError as e:
    _JETCAM_AVAILABLE = False
    print(f"[CameraManager] JetCamera not available: {e}")


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
        self._camera: Optional[JetCamera] = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self.frame_count = 0
        self.width = 640
        self.height = 480
        self.fps = 15

    def start(self, width: int = 640, height: int = 480, fps: int = 15) -> bool:
        """カメラ起動"""
        if self._camera is not None and self._camera.running:
            print("[CameraManager] Camera already running")
            return True

        self.width = width
        self.height = height
        self.fps = fps

        if not _JETCAM_AVAILABLE:
            print("[CameraManager] JetCamera not available, using fallback")
            return self._start_fallback()

        try:
            print(f"[CameraManager] Starting JetCamera: {width}x{height} @ {fps}fps")
            self._camera = JetCamera(
                width=width,
                height=height,
                fps=fps,
                capture_width=1280,
                capture_height=720,
                capture_fps=fps,
            )
            
            if self._camera.start():
                print("[CameraManager] ✓ JetCamera started successfully")
                return True
            else:
                print("[CameraManager] JetCamera.start() failed")
                self._camera = None
                return self._start_fallback()
                
        except Exception as e:
            print(f"[CameraManager] JetCamera error: {e}")
            import traceback
            traceback.print_exc()
            return self._start_fallback()

    def _start_fallback(self) -> bool:
        """USBカメラフォールバック"""
        try:
            print(f"[CameraManager] Trying USB camera fallback")
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[CameraManager] USB camera failed")
                return False
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return False
            
            # ダミーのJetCameraライクなラッパー
            self._camera = _USBCameraWrapper(cap)
            print(f"[CameraManager] ✓ USB fallback started: {frame.shape}")
            return True
            
        except Exception as e:
            print(f"[CameraManager] Fallback error: {e}")
            return False

    def read(self) -> Optional[np.ndarray]:
        """フレーム取得"""
        if self._camera is None:
            return None

        try:
            frame = self._camera.read()
            
            if frame is not None:
                with self._frame_lock:
                    self._frame = frame.copy()
                self.frame_count += 1

                if self.frame_count % 30 == 0:
                    print(f"[CameraManager] Frame #{self.frame_count}: shape={frame.shape}")

            return frame

        except Exception as e:
            print(f"[CameraManager] Read error: {e}")
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """最新フレーム取得（読み取りなし）"""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def is_ready(self) -> bool:
        """カメラ準備完了確認"""
        if self._camera is None:
            return False
        if hasattr(self._camera, 'running'):
            return self._camera.running
        return True

    def stop(self):
        """カメラ停止"""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as e:
                print(f"[CameraManager] Stop error: {e}")

            print(f"[CameraManager] ✓ Stopped (total frames: {self.frame_count})")
            self._camera = None


class _USBCameraWrapper:
    """USBカメラ用の簡易ラッパー（JetCameraインターフェース互換）"""
    def __init__(self, cap):
        self._cap = cap
        self.running = True
    
    def read(self):
        ret, frame = self._cap.read()
        return frame if ret else None
    
    def stop(self):
        self.running = False
        self._cap.release()


# シングルトンインスタンス
camera_manager = CameraManager()