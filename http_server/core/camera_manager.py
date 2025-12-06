"""カメラ管理（シングルトン）- jetracer_minimal JetCamera 使用"""
import sys
import os
import cv2
import numpy as np
from typing import Optional
import threading
import time

# jetracer_minimal の camera.py をインポート
_JETRACER_MINIMAL_PATH = os.path.expanduser('~/jetracer_minimal')
if _JETRACER_MINIMAL_PATH not in sys.path:
    sys.path.insert(0, _JETRACER_MINIMAL_PATH)

try:
    from camera import JetCamera
    _JETCAMERA_AVAILABLE = True
    print("[CameraManager] ✓ JetCamera from jetracer_minimal loaded")
except ImportError as e:
    _JETCAMERA_AVAILABLE = False
    JetCamera = None
    print(f"[CameraManager] ✗ JetCamera not available: {e}")


class CameraManager:
    """シングルトン カメラマネージャー"""
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
        self._camera = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self.frame_count = 0
        self.width = 640
        self.height = 480
        self.fps = 15

    def start(self, width: int = 640, height: int = 480, fps: int = 15) -> bool:
        """カメラ起動"""
        if self._camera is not None:
            if hasattr(self._camera, 'running') and self._camera.running:
                print("[CameraManager] Camera already running")
                return True

        self.width = width
        self.height = height
        self.fps = fps

        # JetCamera を使用
        if _JETCAMERA_AVAILABLE:
            if self._start_jetcamera():
                return True

        # フォールバック: USB カメラ
        return self._start_usb_fallback()

    def _start_jetcamera(self) -> bool:
        """JetCamera (jetracer_minimal) で起動"""
        try:
            print(f"[CameraManager] Starting JetCamera: {self.width}x{self.height} @ {self.fps}fps")

            self._camera = JetCamera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                capture_width=1280,
                capture_height=720,
                capture_fps=self.fps,
            )

            if self._camera.start():
                # テスト読み込み
                time.sleep(0.5)
                test_frame = self._camera.read()
                if test_frame is not None:
                    print(f"[CameraManager] ✓ JetCamera started: shape={test_frame.shape}")
                    return True
                else:
                    print("[CameraManager] ✗ JetCamera test read failed")
            else:
                print("[CameraManager] ✗ JetCamera.start() returned False")

            self._camera.stop()
            self._camera = None
            return False

        except Exception as e:
            print(f"[CameraManager] ✗ JetCamera error: {e}")
            import traceback
            traceback.print_exc()
            self._camera = None
            return False

    def _start_usb_fallback(self) -> bool:
        """V4L2 CSI カメラフォールバック（リサイズ対応）"""
        try:
            print(f"[CameraManager] Trying V4L2 CSI fallback with resize to {self.width}x{self.height}")

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("[CameraManager] ✗ V4L2 camera failed to open")
                return False

            # CSIカメラはネイティブ解像度で取得し、後でリサイズする
            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                print("[CameraManager] ✗ V4L2 camera test read failed")
                return False

            native_shape = frame.shape
            print(f"[CameraManager] Native resolution: {native_shape[1]}x{native_shape[0]}")

            # リサイズテスト
            resized = cv2.resize(frame, (self.width, self.height))

            self._camera = _V4L2CameraWrapper(cap, self.width, self.height)
            print(f"[CameraManager] ✓ V4L2 CSI started: native={native_shape[1]}x{native_shape[0]} -> resize to {self.width}x{self.height}")
            return True

        except Exception as e:
            print(f"[CameraManager] ✗ V4L2 fallback error: {e}")
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
        """最新フレーム取得（新規読み取りなし）"""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def is_ready(self) -> bool:
        """カメラ準備完了確認"""
        if self._camera is None:
            return False
        if hasattr(self._camera, 'running'):
            return self._camera.running
        return True

    def get_resolution(self) -> tuple:
        """現在の解像度を返す"""
        return (self.width, self.height)

    def stop(self):
        """カメラ停止"""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as e:
                print(f"[CameraManager] Stop error: {e}")

            print(f"[CameraManager] ✓ Stopped (total frames: {self.frame_count})")
            self._camera = None


class _V4L2CameraWrapper:
    """V4L2 CSI カメラ用ラッパー（リサイズ対応）"""
    def __init__(self, cap: cv2.VideoCapture, target_width: int, target_height: int):
        self._cap = cap
        self._target_width = target_width
        self._target_height = target_height
        self.running = True

    def read(self) -> Optional[np.ndarray]:
        if not self.running:
            return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        # ネイティブ解像度からターゲット解像度にリサイズ
        if frame.shape[1] != self._target_width or frame.shape[0] != self._target_height:
            frame = cv2.resize(frame, (self._target_width, self._target_height))
        return frame

    def stop(self):
        self.running = False
        if self._cap is not None:
            self._cap.release()


# シングルトンインスタンス
camera_manager = CameraManager()
