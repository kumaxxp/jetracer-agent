"""カメラキャリブレーション処理"""
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import threading


@dataclass
class CalibrationResult:
    """キャリブレーション結果"""
    camera_id: int
    rms_error: float
    camera_matrix: List[List[float]]
    dist_coeffs: List[float]
    image_size: Tuple[int, int]
    calibrated_at: str
    num_images: int


@dataclass 
class StereoCalibrationResult:
    """ステレオキャリブレーション結果"""
    rms_error: float
    rotation_matrix: List[List[float]]
    translation_vector: List[float]
    essential_matrix: List[List[float]]
    fundamental_matrix: List[List[float]]
    calibrated_at: str
    num_images: int


class CalibrationManager:
    """キャリブレーション管理"""
    
    def __init__(self, 
                 pattern_size: Tuple[int, int] = (10, 7),  # 内側コーナー数 (11×8マス)
                 square_size: float = 23.0,  # mm
                 data_dir: str = "calibration_data"):
        """
        Args:
            pattern_size: チェッカーボードの内側コーナー数 (cols, rows)
            square_size: マス目一辺のサイズ (mm)
            data_dir: データ保存ディレクトリ
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 画像保存ディレクトリ
        self.images_dir = self.data_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # 収集したデータ
        self._captured_data: Dict[int, List[Dict]] = {0: [], 1: []}  # camera_id -> list of data
        self._lock = threading.Lock()
        
        # 最新の検出結果を保持（Capture時に使用）
        self._last_detection: Dict[int, Dict] = {}  # camera_id -> {frame, corners, pattern_size, timestamp}
        
        # キャリブレーション結果
        self._results: Dict[int, CalibrationResult] = {}
        self._stereo_result: Optional[StereoCalibrationResult] = None
        
        # 結果を読み込み（存在する場合）
        self._load_results()
        
        print(f"[Calibration] Initialized: pattern={pattern_size}, square_size={square_size}mm")
    
    def _create_object_points(self, pattern_size: Tuple[int, int]) -> np.ndarray:
        """パターンサイズに基づいてオブジェクトポイントを生成
        
        Args:
            pattern_size: (cols, rows) 内側コーナー数
            
        Returns:
            3Dオブジェクトポイント
        """
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def detect_checkerboard(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int]]]:
        """チェッカーボードを検出
        
        Args:
            image: 入力画像 (BGR)
            
        Returns:
            (検出成功, コーナー座標, プレビュー画像, 検出されたパターンサイズ)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        print(f"[Calibration] Detecting checkerboard: pattern_size={self.pattern_size}, image_shape={gray.shape}")
        
        # 高速検出用フラグ（最初に試す）
        fast_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        
        # 通常検出用フラグ
        normal_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        
        # ステップ1: 設定されたパターンサイズで高速検出
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, fast_flags)
        if ret:
            print(f"[Calibration] Fast detection succeeded with pattern={self.pattern_size}")
            detected_pattern = self.pattern_size
        else:
            # ステップ2: 縦横逆で試す
            reversed_pattern = (self.pattern_size[1], self.pattern_size[0])
            ret, corners = cv2.findChessboardCorners(gray, reversed_pattern, fast_flags)
            if ret:
                print(f"[Calibration] Fast detection succeeded with reversed pattern={reversed_pattern}")
                detected_pattern = reversed_pattern
            else:
                # ステップ3: 設定されたパターンサイズで通常検出（FAST_CHECKなし）
                ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, normal_flags)
                if ret:
                    print(f"[Calibration] Normal detection succeeded with pattern={self.pattern_size}")
                    detected_pattern = self.pattern_size
                else:
                    # ステップ4: 縦横逆で通常検出
                    ret, corners = cv2.findChessboardCorners(gray, reversed_pattern, normal_flags)
                    if ret:
                        print(f"[Calibration] Normal detection succeeded with reversed pattern={reversed_pattern}")
                        detected_pattern = reversed_pattern
        
        if not ret:
            print(f"[Calibration] Primary detection failed, trying alternative patterns...")
            
            # 代替パターンを試す（最小限に）
            alt_patterns = [
                (self.pattern_size[0] - 1, self.pattern_size[1]),
                (self.pattern_size[0], self.pattern_size[1] - 1),
                (self.pattern_size[0] + 1, self.pattern_size[1]),
                (self.pattern_size[0], self.pattern_size[1] + 1),
            ]
            
            for pattern in alt_patterns:
                if pattern[0] < 3 or pattern[1] < 3:
                    continue
                ret, corners = cv2.findChessboardCorners(gray, pattern, fast_flags)
                if ret:
                    print(f"[Calibration] Alternative pattern detected: {pattern}")
                    detected_pattern = pattern
                    break
        
        if not ret:
            print(f"[Calibration] All detection methods failed")
            
            # デバッグ用: 検出失敗画像を保存
            debug_path = self.data_dir / "debug_failed_detection.jpg"
            cv2.imwrite(str(debug_path), image)
            print(f"[Calibration] Saved failed detection image to: {debug_path}")
            
            return False, None, None, None
        
        if detected_pattern != self.pattern_size:
            print(f"[Calibration] WARNING: Detected with different pattern size: {detected_pattern} (expected {self.pattern_size})")
            print(f"[Calibration] Consider updating pattern_size in CalibrationManager")
        
        # サブピクセル精度で補正
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # プレビュー画像作成
        preview = image.copy()
        cv2.drawChessboardCorners(preview, detected_pattern, corners, ret)
        
        return True, corners, preview, detected_pattern
    
    def store_detection(self, camera_id: int, frame: np.ndarray, 
                        corners: np.ndarray, pattern_size: Tuple[int, int]):
        """Capture用に検出結果を保持
        
        Args:
            camera_id: カメラID
            frame: フレーム
            corners: コーナー座標
            pattern_size: 検出されたパターンサイズ
        """
        with self._lock:
            self._last_detection[camera_id] = {
                "frame": frame.copy(),
                "corners": corners.copy(),
                "pattern_size": pattern_size,
                "timestamp": datetime.now()
            }
            print(f"[Calibration] Camera {camera_id}: Detection stored")
    
    def get_stored_detection(self, camera_id: int, max_age_seconds: float = 10.0) -> Optional[Dict]:
        """Capture用に保持された検出結果を取得
        
        Args:
            camera_id: カメラID
            max_age_seconds: 有効な最大経過時間（秒）
            
        Returns:
            検出結果、またはNone
        """
        with self._lock:
            if camera_id not in self._last_detection:
                return None
            
            detection = self._last_detection[camera_id]
            age = (datetime.now() - detection["timestamp"]).total_seconds()
            
            if age > max_age_seconds:
                print(f"[Calibration] Camera {camera_id}: Stored detection expired ({age:.1f}s > {max_age_seconds}s)")
                return None
            
            return detection
    
    def clear_stored_detection(self, camera_id: Optional[int] = None):
        """保持された検出結果をクリア"""
        with self._lock:
            if camera_id is None:
                self._last_detection = {}
            elif camera_id in self._last_detection:
                del self._last_detection[camera_id]
    
    def get_detection_info(self, corners: np.ndarray, image_shape: Tuple[int, int]) -> Dict:
        """検出情報を取得
        
        Args:
            corners: コーナー座標
            image_shape: 画像サイズ (height, width)
            
        Returns:
            検出情報（位置、カバレッジなど）
        """
        h, w = image_shape[:2]
        
        # バウンディングボックス
        x_coords = corners[:, 0, 0]
        y_coords = corners[:, 0, 1]
        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())
        
        # 中心位置（正規化）
        center_x = (x_min + x_max) / 2 / w
        center_y = (y_min + y_max) / 2 / h
        
        # カバレッジ
        coverage = (x_max - x_min) * (y_max - y_min) / (w * h)
        
        # 傾き（最初と最後のコーナーから推定）
        first_corner = corners[0, 0]
        last_corner = corners[-1, 0]
        angle = np.degrees(np.arctan2(
            last_corner[1] - first_corner[1],
            last_corner[0] - first_corner[0]
        ))
        
        return {
            "center_x": center_x,
            "center_y": center_y,
            "coverage": coverage,
            "angle": float(angle),
            "bounds": {
                "x_min": x_min / w,
                "x_max": x_max / w,
                "y_min": y_min / h,
                "y_max": y_max / h
            }
        }
    
    def capture_calibration_image(self, camera_id: int, image: np.ndarray, 
                                   corners: np.ndarray, pattern_size: Tuple[int, int]) -> Dict:
        """キャリブレーション用画像を保存
        
        Args:
            camera_id: カメラID
            image: 画像
            corners: 検出したコーナー座標
            pattern_size: 検出されたパターンサイズ
            
        Returns:
            保存結果
        """
        with self._lock:
            # 画像を保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"cam{camera_id}_{timestamp}.jpg"
            filepath = self.images_dir / filename
            cv2.imwrite(str(filepath), image)
            
            # データを記録（パターンサイズも保存）
            data = {
                "filename": filename,
                "corners": corners.tolist(),
                "pattern_size": pattern_size,  # パターンサイズを保存
                "image_size": (image.shape[1], image.shape[0]),
                "timestamp": timestamp
            }
            self._captured_data[camera_id].append(data)
            
            count = len(self._captured_data[camera_id])
            print(f"[Calibration] Camera {camera_id}: captured image #{count}, pattern={pattern_size}")
            
            return {
                "camera_id": camera_id,
                "filename": filename,
                "count": count,
                "pattern_size": pattern_size
            }
    
    def capture_stereo_pair(self, 
                            image0: np.ndarray, corners0: np.ndarray, pattern0: Tuple[int, int],
                            image1: np.ndarray, corners1: np.ndarray, pattern1: Tuple[int, int]) -> Dict:
        """ステレオペアを同時保存
        
        Args:
            image0, corners0, pattern0: Camera0の画像、コーナー、パターンサイズ
            image1, corners1, pattern1: Camera1の画像、コーナー、パターンサイズ
            
        Returns:
            保存結果
        """
        result0 = self.capture_calibration_image(0, image0, corners0, pattern0)
        result1 = self.capture_calibration_image(1, image1, corners1, pattern1)
        
        return {
            "camera0": result0,
            "camera1": result1,
            "pair_count": min(result0["count"], result1["count"])
        }
    
    def get_status(self) -> Dict:
        """現在の状態を取得"""
        with self._lock:
            return {
                "pattern_size": self.pattern_size,
                "square_size_mm": self.square_size,
                "captured_images": {
                    0: len(self._captured_data[0]),
                    1: len(self._captured_data[1])
                },
                "min_images_required": 10,
                "recommended_images": 20,
                "calibrated": {
                    0: 0 in self._results,
                    1: 1 in self._results,
                    "stereo": self._stereo_result is not None
                },
                "results": {
                    "0": asdict(self._results[0]) if 0 in self._results else None,
                    "1": asdict(self._results[1]) if 1 in self._results else None,
                    "stereo": asdict(self._stereo_result) if self._stereo_result else None
                }
            }
    
    def clear_captured_images(self, camera_id: Optional[int] = None):
        """収集した画像をクリア
        
        Args:
            camera_id: None の場合は両方クリア
        """
        with self._lock:
            if camera_id is None:
                self._captured_data = {0: [], 1: []}
                # ファイルも削除
                for f in self.images_dir.glob("cam*.jpg"):
                    f.unlink()
                print("[Calibration] Cleared all captured images")
            else:
                self._captured_data[camera_id] = []
                for f in self.images_dir.glob(f"cam{camera_id}_*.jpg"):
                    f.unlink()
                print(f"[Calibration] Cleared camera {camera_id} images")
    
    def calibrate_single_camera(self, camera_id: int) -> Optional[CalibrationResult]:
        """単一カメラのキャリブレーション
        
        Args:
            camera_id: カメラID
            
        Returns:
            キャリブレーション結果
        """
        with self._lock:
            data_list = self._captured_data[camera_id]
            
            if len(data_list) < 10:
                print(f"[Calibration] Camera {camera_id}: Not enough images ({len(data_list)} < 10)")
                return None
            
            # オブジェクトポイントと画像ポイントを収集
            obj_points = []
            img_points = []
            image_size = None
            
            for i, data in enumerate(data_list):
                # パターンサイズを取得（保存されていない場合はデフォルト使用）
                pattern_size = data.get("pattern_size", self.pattern_size)
                if isinstance(pattern_size, list):
                    pattern_size = tuple(pattern_size)
                
                # このパターンサイズに対応するオブジェクトポイントを生成
                objp = self._create_object_points(pattern_size)
                
                # cornersの形式を確認して正しく変換
                corners = np.array(data["corners"], dtype=np.float32)
                
                # 形式が(N, 2)の場合は(N, 1, 2)に変換
                if corners.ndim == 2 and corners.shape[1] == 2:
                    corners = corners.reshape(-1, 1, 2)
                elif corners.ndim == 3 and corners.shape[1] == 1 and corners.shape[2] == 2:
                    pass  # 正しい形式
                else:
                    print(f"[Calibration] WARNING: Unexpected corners shape at image {i}: {corners.shape}")
                    continue
                
                # コーナー数とオブジェクトポイント数が一致するか確認
                expected_corners = pattern_size[0] * pattern_size[1]
                if corners.shape[0] != expected_corners:
                    print(f"[Calibration] WARNING: Image {i}: corners count mismatch: {corners.shape[0]} vs {expected_corners}")
                    continue
                
                obj_points.append(objp)
                img_points.append(corners)
                image_size = tuple(data["image_size"])
                
                print(f"[Calibration] Image {i}: pattern={pattern_size}, corners={corners.shape[0]}, objp={objp.shape[0]}")
            
            if len(img_points) < 10:
                print(f"[Calibration] Camera {camera_id}: Not enough valid images after filtering ({len(img_points)} < 10)")
                return None
            
            print(f"[Calibration] Camera {camera_id}: Calibrating with {len(img_points)} images...")
            
            try:
                # キャリブレーション実行
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj_points, img_points, image_size, None, None
                )
                
                result = CalibrationResult(
                    camera_id=camera_id,
                    rms_error=ret,
                    camera_matrix=mtx.tolist(),
                    dist_coeffs=dist.flatten().tolist(),
                    image_size=image_size,
                    calibrated_at=datetime.now().isoformat(),
                    num_images=len(img_points)
                )
                
                self._results[camera_id] = result
                print(f"[Calibration] Camera {camera_id}: RMS error = {ret:.4f}")
                
                # 結果を保存
                self._save_results()
                
                return result
                
            except Exception as e:
                print(f"[Calibration] Camera {camera_id}: Calibration failed with error: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def _parse_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """タイムスタンプ文字列をdatetimeに変換
        
        Args:
            timestamp_str: "YYYYMMDD_HHMMSS_ffffff" 形式
            
        Returns:
            datetime オブジェクト
        """
        try:
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S_%f")
        except ValueError:
            return None
    
    def _find_stereo_pairs(self, max_time_diff_seconds: float = 1.0) -> List[Tuple[Dict, Dict]]:
        """タイムスタンプが近いステレオペアを検索
        
        Args:
            max_time_diff_seconds: ペアと見なす最大時間差（秒）
            
        Returns:
            [(cam0_data, cam1_data), ...] のリスト
        """
        data0 = self._captured_data[0]
        data1 = self._captured_data[1]
        
        pairs = []
        used_indices_1 = set()  # 既にペアになったCamera 1のインデックス
        
        for d0 in data0:
            ts0 = self._parse_timestamp(d0.get("timestamp", ""))
            if ts0 is None:
                continue
            
            best_match = None
            best_diff = float('inf')
            best_idx = -1
            
            for idx, d1 in enumerate(data1):
                if idx in used_indices_1:
                    continue
                    
                ts1 = self._parse_timestamp(d1.get("timestamp", ""))
                if ts1 is None:
                    continue
                
                time_diff = abs((ts0 - ts1).total_seconds())
                
                if time_diff < max_time_diff_seconds and time_diff < best_diff:
                    best_diff = time_diff
                    best_match = d1
                    best_idx = idx
            
            if best_match is not None:
                pairs.append((d0, best_match))
                used_indices_1.add(best_idx)
                print(f"[Calibration] Stereo pair found: {d0['timestamp']} <-> {best_match['timestamp']} (diff: {best_diff:.3f}s)")
        
        return pairs
    
    def calibrate_stereo(self) -> Optional[StereoCalibrationResult]:
        """ステレオキャリブレーション（タイムスタンプベースのペアリング）
        
        Returns:
            ステレオキャリブレーション結果
        """
        with self._lock:
            # 両カメラのキャリブレーション結果が必要
            if 0 not in self._results or 1 not in self._results:
                print("[Calibration] Stereo: Single camera calibration required first")
                return None
            
            # タイムスタンプベースでペアを検索
            pairs = self._find_stereo_pairs(max_time_diff_seconds=1.0)
            
            if len(pairs) < 10:
                print(f"[Calibration] Stereo: Not enough stereo pairs ({len(pairs)} < 10)")
                print(f"[Calibration] Stereo: Use 'Capture Stereo Pair' to capture simultaneous images")
                return None
            
            print(f"[Calibration] Stereo: Found {len(pairs)} valid stereo pairs")
            
            # データを収集
            obj_points = []
            img_points0 = []
            img_points1 = []
            
            for i, (d0, d1) in enumerate(pairs):
                # パターンサイズを取得
                pattern0 = d0.get("pattern_size", self.pattern_size)
                pattern1 = d1.get("pattern_size", self.pattern_size)
                if isinstance(pattern0, list):
                    pattern0 = tuple(pattern0)
                if isinstance(pattern1, list):
                    pattern1 = tuple(pattern1)
                
                # ステレオではパターンサイズが一致する必要がある
                if pattern0 != pattern1:
                    print(f"[Calibration] WARNING: Stereo pair {i} pattern mismatch: {pattern0} vs {pattern1}, skipping")
                    continue
                
                objp = self._create_object_points(pattern0)
                
                # cornersの形式を正しく変換
                corners0 = np.array(d0["corners"], dtype=np.float32)
                corners1 = np.array(d1["corners"], dtype=np.float32)
                
                if corners0.ndim == 2 and corners0.shape[1] == 2:
                    corners0 = corners0.reshape(-1, 1, 2)
                if corners1.ndim == 2 and corners1.shape[1] == 2:
                    corners1 = corners1.reshape(-1, 1, 2)
                
                # コーナー数チェック
                expected_corners = pattern0[0] * pattern0[1]
                if corners0.shape[0] != expected_corners or corners1.shape[0] != expected_corners:
                    print(f"[Calibration] WARNING: Stereo pair {i} corners mismatch, skipping")
                    continue
                
                obj_points.append(objp)
                img_points0.append(corners0)
                img_points1.append(corners1)
            
            if len(obj_points) < 10:
                print(f"[Calibration] Stereo: Not enough valid pairs after filtering ({len(obj_points)} < 10)")
                return None
            
            image_size = tuple(self._results[0].image_size)
            mtx0 = np.array(self._results[0].camera_matrix)
            dist0 = np.array(self._results[0].dist_coeffs)
            mtx1 = np.array(self._results[1].camera_matrix)
            dist1 = np.array(self._results[1].dist_coeffs)
            
            print(f"[Calibration] Stereo: Calibrating with {len(obj_points)} pairs...")
            
            try:
                # ステレオキャリブレーション
                flags = cv2.CALIB_FIX_INTRINSIC
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
                
                ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
                    obj_points, img_points0, img_points1,
                    mtx0, dist0, mtx1, dist1,
                    image_size, criteria=criteria, flags=flags
                )
                
                result = StereoCalibrationResult(
                    rms_error=ret,
                    rotation_matrix=R.tolist(),
                    translation_vector=T.flatten().tolist(),
                    essential_matrix=E.tolist(),
                    fundamental_matrix=F.tolist(),
                    calibrated_at=datetime.now().isoformat(),
                    num_images=len(obj_points)
                )
                
                self._stereo_result = result
                print(f"[Calibration] Stereo: RMS error = {ret:.4f}")
                
                # 結果を保存
                self._save_results()
                
                return result
                
            except Exception as e:
                print(f"[Calibration] Stereo: Calibration failed with error: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    def run_full_calibration(self) -> Dict:
        """フルキャリブレーション実行（単カメラ + ステレオ）"""
        results = {}
        
        # Camera 0
        if len(self._captured_data[0]) >= 10:
            result0 = self.calibrate_single_camera(0)
            results["camera0"] = asdict(result0) if result0 else None
        else:
            results["camera0"] = None
        
        # Camera 1
        if len(self._captured_data[1]) >= 10:
            result1 = self.calibrate_single_camera(1)
            results["camera1"] = asdict(result1) if result1 else None
        else:
            results["camera1"] = None
        
        # Stereo (両方キャリブレーション済みの場合のみ)
        if results.get("camera0") and results.get("camera1"):
            stereo = self.calibrate_stereo()
            results["stereo"] = asdict(stereo) if stereo else None
        else:
            results["stereo"] = None
        
        return results
    
    def undistort_image(self, camera_id: int, image: np.ndarray) -> Optional[np.ndarray]:
        """歪み補正を適用
        
        Args:
            camera_id: カメラID
            image: 入力画像
            
        Returns:
            補正済み画像
        """
        if camera_id not in self._results:
            return None
        
        result = self._results[camera_id]
        mtx = np.array(result.camera_matrix)
        dist = np.array(result.dist_coeffs)
        
        # 最適カメラ行列を計算
        # alpha=0: 黒い領域なし（クロップ）
        # alpha=1: 全ピクセル保持（黒い領域が大きい）
        h, w = image.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))  # alpha=0
        
        # 歪み補正
        undistorted = cv2.undistort(image, mtx, dist, None, new_mtx)
        
        return undistorted
    
    def get_capture_instruction(self) -> Dict:
        """次の撮影指示を生成"""
        with self._lock:
            count0 = len(self._captured_data[0])
            count1 = len(self._captured_data[1])
            max_count = max(count0, count1)
            
            if max_count >= 20:
                return {
                    "status": "ready",
                    "message": "十分な画像が収集されました。キャリブレーションを実行できます。",
                    "instruction": None,
                    "count": max_count
                }
            
            # 撮影位置のガイダンス
            positions = [
                "中央",
                "左上", "右上", "左下", "右下",
                "左", "右", "上", "下",
                "中央（近づける）", "中央（離す）",
                "左（傾ける）", "右（傾ける）",
                "上（傾ける）", "下（傾ける）",
                "中央（時計回りに回転）", "中央（反時計回りに回転）",
                "左上（近づける）", "右下（近づける）",
                "ランダムな位置と角度"
            ]
            
            position = positions[max_count % len(positions)]
            
            return {
                "status": "collecting",
                "message": f"画像 {max_count + 1}/20",
                "instruction": f"チェッカーボードを{position}に配置してください",
                "count": max_count
            }
    
    def _save_results(self):
        """結果をファイルに保存"""
        results = {
            "cameras": {},
            "stereo": None
        }
        
        for camera_id, result in self._results.items():
            results["cameras"][str(camera_id)] = asdict(result)
        
        if self._stereo_result:
            results["stereo"] = asdict(self._stereo_result)
        
        filepath = self.data_dir / "calibration_results.json"
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"[Calibration] Results saved to {filepath}")
    
    def _load_results(self):
        """結果をファイルから読み込み"""
        filepath = self.data_dir / "calibration_results.json"
        if not filepath.exists():
            return
        
        try:
            with open(filepath) as f:
                data = json.load(f)
            
            for camera_id_str, result_data in data.get("cameras", {}).items():
                camera_id = int(camera_id_str)
                self._results[camera_id] = CalibrationResult(**result_data)
            
            if data.get("stereo"):
                self._stereo_result = StereoCalibrationResult(**data["stereo"])
            
            print(f"[Calibration] Loaded results from {filepath}")
        except Exception as e:
            print(f"[Calibration] Failed to load results: {e}")
    
    def is_calibrated(self, camera_id: Optional[int] = None) -> bool:
        """キャリブレーション済みかチェック"""
        if camera_id is not None:
            return camera_id in self._results
        else:
            return 0 in self._results and 1 in self._results
    
    def get_camera_matrix(self, camera_id: int) -> Optional[np.ndarray]:
        """カメラ行列を取得"""
        if camera_id not in self._results:
            return None
        return np.array(self._results[camera_id].camera_matrix)
    
    def get_dist_coeffs(self, camera_id: int) -> Optional[np.ndarray]:
        """歪み係数を取得"""
        if camera_id not in self._results:
            return None
        return np.array(self._results[camera_id].dist_coeffs)


# シングルトンインスタンス
calibration_manager = CalibrationManager()
