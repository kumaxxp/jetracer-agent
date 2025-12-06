"""カメラ管理（シングルトン）- jetracer_minimal互換"""
import cv2
import numpy as np
from typing import Optional
import threading
import time


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
        self._camera = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._use_jetcam = False
        self.frame_count = 0
        self.width = 640
        self.height = 480
        self.fps = 15

    def start(self, width: int = 640, height: int = 480, fps: int = 15):
        """カメラ起動"""
        if self._camera is not None:
            return True

        self.width = width
        self.height = height
        self.fps = fps

        # 方法1: jetcam CSICamera を試す
        if self._start_jetcam():
            return True

        # 方法2: GStreamer パイプラインを試す
        if self._start_gstreamer():
            return True

        # 方法3: USB カメラフォールバック
        return self._start_usb_camera()

    def _start_jetcam(self) -> bool:
        """jetcam CSICamera で起動"""
        try:
            from jetcam.csi_camera import CSICamera
            print(f"[Camera] Trying jetcam CSICamera: {self.width}x{self.height} @ {self.fps}fps")

            self._camera = CSICamera(
                width=self.width,
                height=self.height,
                capture_fps=self.fps,
                capture_device=0,
            )

            # 安定化待ち
            time.sleep(1.0)

            # テスト読み込み
            test_frame = self._camera.read()
            if test_frame is None:
                print("[Camera] jetcam test read failed")
                self._camera = None
                return False

            print(f"[Camera] ✓ jetcam started: shape={test_frame.shape}")
            self._use_jetcam = True
            return True

        except Exception as e:
            print(f"[Camera] jetcam failed: {e}")
            self._camera = None
            return False

    def _start_gstreamer(self) -> bool:
        """GStreamer パイプラインで起動"""
        try:
            print(f"[Camera] Trying GStreamer: {self.width}x{self.height} @ {self.fps}fps")

            # nvarguscamerasrc パイプライン
            gst_pipeline = (
                f"nvarguscamerasrc sensor-id=0 ! "
                f"video/x-raw(memory:NVMM), width=1280, height=720, "
                f"format=NV12, framerate={self.fps}/1 ! "
                f"nvvidconv ! "
                f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )

            self._camera = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

            if not self._camera.isOpened():
                print("[Camera] GStreamer pipeline failed to open")
                self._camera = None
                return False

            # テスト読み込み
            ret, test_frame = self._camera.read()
            if not ret or test_frame is None:
                print("[Camera] GStreamer test read failed")
                self._camera.release()
                self._camera = None
                return False

            print(f"[Camera] ✓ GStreamer started: shape={test_frame.shape}")
            self._use_jetcam = False
            return True

        except Exception as e:
            print(f"[Camera] GStreamer failed: {e}")
            self._camera = None
            return False

    def _start_usb_camera(self) -> bool:
        """USB カメラフォールバック"""
        try:
            print(f"[Camera] Trying USB camera: {self.width}x{self.height}")

            self._camera = cv2.VideoCapture(0)

            if not self._camera.isOpened():
                print("[Camera] USB camera failed to open")
                self._camera = None
                return False

            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._camera.set(cv2.CAP_PROP_FPS, self.fps)

            ret, test_frame = self._camera.read()
            if not ret or test_frame is None:
                print("[Camera] USB camera test read failed")
                self._camera.release()
                self._camera = None
                return False

            print(f"[Camera] ✓ USB camera started: shape={test_frame.shape}")
            self._use_jetcam = False
            return True

        except Exception as e:
            print(f"[Camera] USB camera failed: {e}")
            self._camera = None
            return False

    def read(self) -> Optional[np.ndarray]:
        """フレーム取得"""
        if self._camera is None:
            return None

        try:
            if self._use_jetcam:
                # jetcam: RGB形式で返ってくる
                frame = self._camera.read()
                if frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                # OpenCV: BGR形式
                ret, frame = self._camera.read()
                if not ret:
                    frame = None

            if frame is not None:
                with self._frame_lock:
                    self._frame = frame.copy()
                self.frame_count += 1

                if self.frame_count % 30 == 0:
                    print(f"[Camera] Frame #{self.frame_count}: shape={frame.shape}")

            return frame

        except Exception as e:
            print(f"[Camera] Read error: {e}")
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """最新フレーム取得（読み取りなし）"""
        with self._frame_lock:
            return self._frame.copy() if self._frame is not None else None

    def is_ready(self) -> bool:
        """カメラ準備完了確認"""
        return self._camera is not None

    def stop(self):
        """カメラ停止"""
        if self._camera is not None:
            try:
                if self._use_jetcam:
                    self._camera.release()
                else:
                    self._camera.release()
            except Exception as e:
                print(f"[Camera] Stop error: {e}")

            print(f"[Camera] ✓ Stopped (total frames: {self.frame_count})")
            self._camera = None
            self._use_jetcam = False


# シングルトンインスタンス
camera_manager = CameraManager()
