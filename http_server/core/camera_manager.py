"""カメラ管理（複数カメラ対応）- jetracer_minimal JetCamera 使用"""
import sys
import os
import cv2
import numpy as np
from typing import Optional, Dict
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


class CameraInstance:
    """個別カメラインスタンス"""
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self._camera = None
        self._frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self.frame_count = 0
        self.width = 320
        self.height = 240
        self.fps = 10

    def start(self, width: int = 320, height: int = 240, fps: int = 10) -> bool:
        """カメラ起動"""
        if self._camera is not None:
            if hasattr(self._camera, 'running') and self._camera.running:
                print(f"[Camera{self.camera_id}] Already running")
                return True

        self.width = width
        self.height = height
        self.fps = fps

        # JetCamera を使用
        if _JETCAMERA_AVAILABLE:
            if self._start_jetcamera():
                return True

        # フォールバック: V4L2
        return self._start_v4l2_fallback()

    def _start_jetcamera(self) -> bool:
        """JetCamera (jetracer_minimal) で起動"""
        try:
            print(f"[Camera{self.camera_id}] Starting JetCamera: {self.width}x{self.height} @ {self.fps}fps")

            self._camera = JetCamera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=self.camera_id,
                capture_width=1280,
                capture_height=720,
                capture_fps=self.fps,
            )

            if self._camera.start():
                time.sleep(0.5)
                test_frame = self._camera.read()
                if test_frame is not None:
                    print(f"[Camera{self.camera_id}] ✓ JetCamera started: shape={test_frame.shape}")
                    return True
                else:
                    print(f"[Camera{self.camera_id}] ✗ JetCamera test read failed")
            else:
                print(f"[Camera{self.camera_id}] ✗ JetCamera.start() returned False")

            self._camera.stop()
            self._camera = None
            return False

        except Exception as e:
            print(f"[Camera{self.camera_id}] ✗ JetCamera error: {e}")
            self._camera = None
            return False

    def _start_v4l2_fallback(self) -> bool:
        """V4L2 CSI カメラフォールバック（リサイズ対応）"""
        try:
            print(f"[Camera{self.camera_id}] Trying V4L2 fallback: {self.width}x{self.height}")

            cap = cv2.VideoCapture(self.camera_id)
            if not cap.isOpened():
                print(f"[Camera{self.camera_id}] ✗ V4L2 failed to open")
                return False

            # CSIカメラの最小モード 1280x720@60fps を使用
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                print(f"[Camera{self.camera_id}] ✗ V4L2 test read failed")
                return False

            capture_shape = frame.shape
            print(f"[Camera{self.camera_id}] Capture mode: {capture_shape[1]}x{capture_shape[0]}")

            self._camera = _V4L2CameraWrapper(cap, self.width, self.height)
            print(f"[Camera{self.camera_id}] ✓ V4L2 started: {capture_shape[1]}x{capture_shape[0]} -> {self.width}x{self.height}")
            return True

        except Exception as e:
            print(f"[Camera{self.camera_id}] ✗ V4L2 fallback error: {e}")
            return False

    def read(self) -> Optional[np.ndarray]:
        """フレーム取得（キャッシュ付き）"""
        if self._camera is None:
            return None

        try:
            frame = self._camera.read()

            if frame is not None:
                with self._frame_lock:
                    self._frame = frame.copy()
                self.frame_count += 1

                if self.frame_count % 100 == 0:
                    print(f"[Camera{self.camera_id}] Frame #{self.frame_count}")

                return frame
            else:
                with self._frame_lock:
                    if self._frame is not None:
                        return self._frame.copy()
                return None

        except Exception as e:
            print(f"[Camera{self.camera_id}] Read error: {e}")
            with self._frame_lock:
                if self._frame is not None:
                    return self._frame.copy()
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

    def stop(self):
        """カメラ停止"""
        if self._camera is not None:
            try:
                self._camera.stop()
            except Exception as e:
                print(f"[Camera{self.camera_id}] Stop error: {e}")

            print(f"[Camera{self.camera_id}] ✓ Stopped (total frames: {self.frame_count})")
            self._camera = None


class MultiCameraManager:
    """複数カメラ管理（シングルトン）"""
    _instance: Optional['MultiCameraManager'] = None
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
        self._cameras: Dict[int, CameraInstance] = {}
        self._default_camera_id = 0

    def start(self, width: int = 320, height: int = 240, fps: int = 10, camera_id: int = 0) -> bool:
        """指定カメラを起動"""
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraInstance(camera_id)
        return self._cameras[camera_id].start(width, height, fps)

    def start_all(self, width: int = 320, height: int = 240, fps: int = 10, camera_ids: list = None) -> Dict[int, bool]:
        """複数カメラを起動"""
        if camera_ids is None:
            camera_ids = [0, 1]
        results = {}
        for cid in camera_ids:
            results[cid] = self.start(width, height, fps, cid)
        return results

    def read(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """指定カメラからフレーム取得"""
        if camera_id not in self._cameras:
            return None
        return self._cameras[camera_id].read()

    def get_latest_frame(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """指定カメラの最新フレーム取得"""
        if camera_id not in self._cameras:
            return None
        return self._cameras[camera_id].get_latest_frame()

    def is_ready(self, camera_id: int = 0) -> bool:
        """指定カメラの準備完了確認"""
        if camera_id not in self._cameras:
            return False
        return self._cameras[camera_id].is_ready()

    def get_resolution(self, camera_id: int = 0) -> tuple:
        """指定カメラの解像度を返す"""
        if camera_id not in self._cameras:
            return (0, 0)
        cam = self._cameras[camera_id]
        return (cam.width, cam.height)

    def stop(self, camera_id: int = None):
        """カメラ停止（camera_id=Noneで全停止）"""
        if camera_id is not None:
            if camera_id in self._cameras:
                self._cameras[camera_id].stop()
                del self._cameras[camera_id]
        else:
            for cid in list(self._cameras.keys()):
                self._cameras[cid].stop()
            self._cameras.clear()

    def get_active_cameras(self) -> list:
        """アクティブなカメラIDリストを返す"""
        return [cid for cid, cam in self._cameras.items() if cam.is_ready()]


# シングルトンインスタンス
camera_manager = MultiCameraManager()
