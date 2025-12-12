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
        
        # 設定値
        self.width = 640
        self.height = 480
        self.fps = 30
        self.sensor_mode: Optional[int] = None
        self.capture_width = 1280
        self.capture_height = 720
        self.capture_fps = 60

    def start(self, width: int = 640, height: int = 480, fps: int = 30) -> bool:
        """カメラ起動（デフォルト設定）"""
        return self.start_with_mode(
            sensor_mode=None,  # 自動
            output_width=width,
            output_height=height,
            fps=fps
        )

    def start_with_mode(
        self,
        sensor_mode: Optional[int] = None,
        output_width: int = 640,
        output_height: int = 480,
        fps: int = 30,
        capture_width: Optional[int] = None,
        capture_height: Optional[int] = None
    ) -> bool:
        """センサーモード指定でカメラ起動
        
        Args:
            sensor_mode: センサーモード (0-4)、Noneで自動選択
            output_width: 出力解像度（幅）
            output_height: 出力解像度（高さ）
            fps: フレームレート
            capture_width: キャプチャ解像度（幅）、Noneでモードに応じて自動
            capture_height: キャプチャ解像度（高さ）、Noneでモードに応じて自動
        """
        if self._camera is not None:
            if hasattr(self._camera, 'running') and self._camera.running:
                print(f"[Camera{self.camera_id}] Already running, stopping first...")
                self.stop()

        self.width = output_width
        self.height = output_height
        self.fps = fps
        self.sensor_mode = sensor_mode
        
        # センサーモードに応じたキャプチャ解像度を決定
        if capture_width and capture_height:
            self.capture_width = capture_width
            self.capture_height = capture_height
        else:
            self.capture_width, self.capture_height, self.capture_fps = \
                self._get_capture_params_for_mode(sensor_mode, fps)

        # JetCamera を使用
        if _JETCAMERA_AVAILABLE:
            if self._start_jetcamera_with_mode():
                return True

        # フォールバック: V4L2
        return self._start_v4l2_fallback()

    def _get_capture_params_for_mode(self, sensor_mode: Optional[int], target_fps: int) -> tuple:
        """センサーモードに応じたキャプチャパラメータを返す"""
        # モード別のデフォルト設定 (width, height, max_fps)
        mode_params = {
            0: (3280, 2464, 21),
            1: (3280, 1848, 28),
            2: (1920, 1080, 30),
            3: (1640, 1232, 30),
            4: (1280, 720, 60),
        }
        
        if sensor_mode is not None and sensor_mode in mode_params:
            w, h, max_fps = mode_params[sensor_mode]
            # モードが明示的に指定された場合は、そのモードの最大FPSを使用
            # （例: モード4は60fpsで動作すべき）
            return (w, h, max_fps)
        
        # 自動選択: ターゲットFPSに基づいて最適なモードを選択
        if target_fps > 30:
            return (1280, 720, min(target_fps, 60))  # モード4
        elif target_fps > 21:
            return (1640, 1232, min(target_fps, 30))  # モード3（ビニング、高画質）
        else:
            return (1920, 1080, min(target_fps, 30))  # モード2

    def _start_jetcamera_with_mode(self) -> bool:
        """JetCamera (jetracer_minimal) でセンサーモード指定起動"""
        try:
            print(f"[Camera{self.camera_id}] Starting JetCamera: "
                  f"output={self.width}x{self.height}, "
                  f"capture={self.capture_width}x{self.capture_height}@{self.capture_fps}fps, "
                  f"sensor_mode={self.sensor_mode}")

            self._camera = JetCamera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                device=self.camera_id,
                sensor_mode=self.sensor_mode,
                capture_width=self.capture_width,
                capture_height=self.capture_height,
                capture_fps=self.capture_fps,
            )

            if self._camera.start():
                time.sleep(0.5)
                test_frame = self._camera.read()
                if test_frame is not None:
                    print(f"[Camera{self.camera_id}] ✓ JetCamera started: shape={test_frame.shape}")
                    self.frame_count = 0
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
            import traceback
            traceback.print_exc()
            self._camera = None
            return False

    def _start_jetcamera(self) -> bool:
        """JetCamera (jetracer_minimal) で起動（後方互換）"""
        return self._start_jetcamera_with_mode()

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
            self.frame_count = 0
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

    def get_settings(self) -> dict:
        """現在のカメラ設定を取得"""
        return {
            "camera_id": self.camera_id,
            "output_width": self.width,
            "output_height": self.height,
            "fps": self.fps,
            "sensor_mode": self.sensor_mode,
            "capture_width": self.capture_width,
            "capture_height": self.capture_height,
            "capture_fps": self.capture_fps,
            "frame_count": self.frame_count,
            "is_ready": self.is_ready()
        }

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
        
        # 歪み補正設定
        self._undistort_enabled: Dict[int, bool] = {}  # camera_id -> enabled
        self._calibration_data: Dict[int, Dict] = {}   # camera_id -> {camera_matrix, dist_coeffs, new_camera_matrix}

    def start(self, width: int = 640, height: int = 480, fps: int = 30, camera_id: int = 0) -> bool:
        """指定カメラを起動"""
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraInstance(camera_id)
        return self._cameras[camera_id].start(width, height, fps)

    def start_with_mode(
        self,
        camera_id: int = 0,
        sensor_mode: Optional[int] = None,
        output_width: int = 640,
        output_height: int = 480,
        fps: int = 30
    ) -> bool:
        """センサーモード指定でカメラを起動
        
        Args:
            camera_id: カメラID
            sensor_mode: センサーモード (0-4)、Noneで自動
            output_width: 出力解像度（幅）
            output_height: 出力解像度（高さ）
            fps: フレームレート
        """
        if camera_id not in self._cameras:
            self._cameras[camera_id] = CameraInstance(camera_id)
        
        return self._cameras[camera_id].start_with_mode(
            sensor_mode=sensor_mode,
            output_width=output_width,
            output_height=output_height,
            fps=fps
        )

    def start_all(self, width: int = 640, height: int = 480, fps: int = 30, camera_ids: list = None) -> Dict[int, bool]:
        """複数カメラを起動"""
        if camera_ids is None:
            camera_ids = [0, 1]
        results = {}
        for cid in camera_ids:
            results[cid] = self.start(width, height, fps, cid)
        return results

    def read(self, camera_id: int = 0, apply_undistort: bool = True) -> Optional[np.ndarray]:
        """指定カメラからフレーム取得
        
        Args:
            camera_id: カメラID
            apply_undistort: 歪み補正を適用するか（デフォルトTrue、設定がONの場合のみ適用）
        """
        if camera_id not in self._cameras:
            return None
        
        frame = self._cameras[camera_id].read()
        
        # 歪み補正が有効かつキャリブレーションデータがある場合
        if (apply_undistort and 
            frame is not None and 
            self._undistort_enabled.get(camera_id, False) and 
            camera_id in self._calibration_data):
            
            frame = self._apply_undistort(camera_id, frame)
        
        return frame

    def read_raw(self, camera_id: int = 0) -> Optional[np.ndarray]:
        """歪み補正なしでフレーム取得"""
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

    def get_camera_settings(self, camera_id: int = 0) -> Optional[dict]:
        """指定カメラの現在の設定を取得"""
        if camera_id not in self._cameras:
            return None
        return self._cameras[camera_id].get_settings()

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

    # ====== 歪み補正関連 ======
    
    def set_undistort_enabled(self, camera_id: int, enabled: bool) -> bool:
        """歪み補正の有効/無効を設定
        
        Args:
            camera_id: カメラID
            enabled: 有効にするか
            
        Returns:
            成功したか（キャリブレーションデータがない場合はFalse）
        """
        if enabled and camera_id not in self._calibration_data:
            print(f"[CameraManager] Camera {camera_id}: No calibration data, cannot enable undistort")
            return False
        
        self._undistort_enabled[camera_id] = enabled
        print(f"[CameraManager] Camera {camera_id}: Undistort {'enabled' if enabled else 'disabled'}")
        return True
    
    def is_undistort_enabled(self, camera_id: int) -> bool:
        """歪み補正が有効か確認"""
        return self._undistort_enabled.get(camera_id, False)
    
    def set_calibration_data(self, camera_id: int, camera_matrix: np.ndarray, 
                             dist_coeffs: np.ndarray, image_size: tuple = None):
        """キャリブレーションデータを設定
        
        Args:
            camera_id: カメラID
            camera_matrix: カメラ行列 (3x3)
            dist_coeffs: 歪み係数
            image_size: 画像サイズ (width, height)、Noneの場合はカメラの解像度を使用
        """
        if image_size is None:
            image_size = self.get_resolution(camera_id)
        
        w, h = image_size
        
        # 最適カメラ行列を事前計算
        # alpha=0: 黒い領域なし（クロップ）
        # alpha=1: 全ピクセル保持（黒い領域が大きい）
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix, dist_coeffs, (w, h), 0, (w, h)  # alpha=0 に変更
        )
        
        self._calibration_data[camera_id] = {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "new_camera_matrix": new_camera_matrix,
            "roi": roi,
            "image_size": (w, h)
        }
        
        print(f"[CameraManager] Camera {camera_id}: Calibration data set ({w}x{h}), ROI={roi}")
    
    def load_calibration_from_manager(self):
        """CalibrationManagerからキャリブレーションデータを読み込み"""
        try:
            from .calibration import calibration_manager
            
            for camera_id in [0, 1]:
                if calibration_manager.is_calibrated(camera_id):
                    mtx = calibration_manager.get_camera_matrix(camera_id)
                    dist = calibration_manager.get_dist_coeffs(camera_id)
                    
                    if mtx is not None and dist is not None:
                        result = calibration_manager._results.get(camera_id)
                        image_size = result.image_size if result else None
                        self.set_calibration_data(camera_id, mtx, dist, image_size)
                        print(f"[CameraManager] Camera {camera_id}: Loaded calibration from CalibrationManager")
        except Exception as e:
            print(f"[CameraManager] Failed to load from CalibrationManager: {e}")
            # フォールバック: JSONファイルから直接ロード
            self.load_calibration_from_json()
    
    def load_calibration_from_json(self, json_path: str = None):
        """JSONファイルから直接キャリブレーションデータを読み込み"""
        import json
        from pathlib import Path
        
        if json_path is None:
            # デフォルトパス
            base_dir = Path(__file__).parent.parent.parent
            json_path = base_dir / "calibration_data" / "calibration_results.json"
        else:
            json_path = Path(json_path)
        
        if not json_path.exists():
            print(f"[CameraManager] Calibration JSON not found: {json_path}")
            return
        
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            for cam_id_str, cam_data in data.get("cameras", {}).items():
                camera_id = int(cam_id_str)
                
                camera_matrix = np.array(cam_data["camera_matrix"])
                dist_coeffs = np.array(cam_data["dist_coeffs"])
                image_size = tuple(cam_data["image_size"])
                
                self.set_calibration_data(camera_id, camera_matrix, dist_coeffs, image_size)
                print(f"[CameraManager] Camera {camera_id}: Loaded calibration from JSON")
                
        except Exception as e:
            print(f"[CameraManager] Failed to load calibration from JSON: {e}")
            import traceback
            traceback.print_exc()
    
    def has_calibration(self, camera_id: int) -> bool:
        """キャリブレーションデータがあるか確認"""
        return camera_id in self._calibration_data
    
    def _apply_undistort(self, camera_id: int, frame: np.ndarray) -> np.ndarray:
        """歪み補正を適用"""
        if camera_id not in self._calibration_data:
            return frame
        
        calib = self._calibration_data[camera_id]
        
        try:
            undistorted = cv2.undistort(
                frame,
                calib["camera_matrix"],
                calib["dist_coeffs"],
                None,
                calib["new_camera_matrix"]
            )
            return undistorted
        except Exception as e:
            print(f"[CameraManager] Camera {camera_id}: Undistort error: {e}")
            return frame
    
    def get_undistort_status(self) -> Dict:
        """歪み補正の状態を取得"""
        status = {}
        for camera_id in [0, 1]:
            status[camera_id] = {
                "has_calibration": self.has_calibration(camera_id),
                "enabled": self.is_undistort_enabled(camera_id)
            }
        return status


# シングルトンインスタンス
camera_manager = MultiCameraManager()
