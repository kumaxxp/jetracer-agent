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
                 pattern_size: Tuple[int, int] = (5, 3),  # 内側コーナー数
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
        
        # 3Dオブジェクトポイント（チェッカーボードの実座標）
        self.objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        self.objp *= square_size
        
        # 収集したデータ
        self._captured_data: Dict[int, List[Dict]] = {0: [], 1: []}  # camera_id -> list of data
        self._lock = threading.Lock()
        
        # キャリブレーション結果
        self._results: Dict[int, CalibrationResult] = {}
        self._stereo_result: Optional[StereoCalibrationResult] = None
        
        # 結果を読み込み（存在する場合）
        self._load_results()
        
        print(f"[Calibration] Initialized: pattern={pattern_size}, square_size={square_size}mm")
    
    def detect_checkerboard(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """チェッカーボードを検出
        
        Args:
            image: 入力画像 (BGR)
            
        Returns:
            (検出成功, コーナー座標, プレビュー画像)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # チェッカーボード検出
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, flags)
        
        if ret:
            # サブピクセル精度で補正
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # プレビュー画像作成
            preview = image.copy()
            cv2.drawChessboardCorners(preview, self.pattern_size, corners, ret)
            
            return True, corners, preview
        else:
            return False, None, None
    
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
    
    def capture_calibration_image(self, camera_id: int, image: np.ndarray, corners: np.ndarray) -> Dict:
        """キャリブレーション用画像を保存
        
        Args:
            camera_id: カメラID
            image: 画像
            corners: 検出したコーナー座標
            
        Returns:
            保存結果
        """
        with self._lock:
            # 画像を保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"cam{camera_id}_{timestamp}.jpg"
            filepath = self.images_dir / filename
            cv2.imwrite(str(filepath), image)
            
            # データを記録
            data = {
                "filename": filename,
                "corners": corners.tolist(),
                "image_size": (image.shape[1], image.shape[0]),
                "timestamp": timestamp
            }
            self._captured_data[camera_id].append(data)
            
            count = len(self._captured_data[camera_id])
            print(f"[Calibration] Camera {camera_id}: captured image #{count}")
            
            return {
                "camera_id": camera_id,
                "filename": filename,
                "count": count
            }
    
    def capture_stereo_pair(self, 
                            image0: np.ndarray, corners0: np.ndarray,
                            image1: np.ndarray, corners1: np.ndarray) -> Dict:
        """ステレオペアを同時保存
        
        Args:
            image0, corners0: Camera0の画像とコーナー
            image1, corners1: Camera1の画像とコーナー
            
        Returns:
            保存結果
        """
        result0 = self.capture_calibration_image(0, image0, corners0)
        result1 = self.capture_calibration_image(1, image1, corners1)
        
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
                    0: asdict(self._results[0]) if 0 in self._results else None,
                    1: asdict(self._results[1]) if 1 in self._results else None,
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
            
            for data in data_list:
                obj_points.append(self.objp)
                img_points.append(np.array(data["corners"], dtype=np.float32))
                image_size = tuple(data["image_size"])
            
            print(f"[Calibration] Camera {camera_id}: Calibrating with {len(data_list)} images...")
            
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
                num_images=len(data_list)
            )
            
            self._results[camera_id] = result
            print(f"[Calibration] Camera {camera_id}: RMS error = {ret:.4f}")
            
            return result
    
    def calibrate_stereo(self) -> Optional[StereoCalibrationResult]:
        """ステレオキャリブレーション
        
        Returns:
            ステレオキャリブレーション結果
        """
        with self._lock:
            # 両カメラのキャリブレーション結果が必要
            if 0 not in self._results or 1 not in self._results:
                print("[Calibration] Stereo: Single camera calibration required first")
                return None
            
            data0 = self._captured_data[0]
            data1 = self._captured_data[1]
            
            # ペア数を確認
            pair_count = min(len(data0), len(data1))
            if pair_count < 10:
                print(f"[Calibration] Stereo: Not enough pairs ({pair_count} < 10)")
                return None
            
            # データを収集
            obj_points = []
            img_points0 = []
            img_points1 = []
            
            for i in range(pair_count):
                obj_points.append(self.objp)
                img_points0.append(np.array(data0[i]["corners"], dtype=np.float32))
                img_points1.append(np.array(data1[i]["corners"], dtype=np.float32))
            
            image_size = tuple(self._results[0].image_size)
            mtx0 = np.array(self._results[0].camera_matrix)
            dist0 = np.array(self._results[0].dist_coeffs)
            mtx1 = np.array(self._results[1].camera_matrix)
            dist1 = np.array(self._results[1].dist_coeffs)
            
            print(f"[Calibration] Stereo: Calibrating with {pair_count} pairs...")
            
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
                num_images=pair_count
            )
            
            self._stereo_result = result
            print(f"[Calibration] Stereo: RMS error = {ret:.4f}")
            
            return result
    
    def run_full_calibration(self) -> Dict:
        """フルキャリブレーション実行（単カメラ + ステレオ）"""
        results = {}
        
        # Camera 0
        result0 = self.calibrate_single_camera(0)
        results["camera0"] = asdict(result0) if result0 else None
        
        # Camera 1
        result1 = self.calibrate_single_camera(1)
        results["camera1"] = asdict(result1) if result1 else None
        
        # Stereo
        if result0 and result1:
            stereo = self.calibrate_stereo()
            results["stereo"] = asdict(stereo) if stereo else None
        else:
            results["stereo"] = None
        
        # 結果を保存
        self._save_results()
        
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
        h, w = image.shape[:2]
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        
        # 歪み補正
        undistorted = cv2.undistort(image, mtx, dist, None, new_mtx)
        
        # ROIでクロップ（オプション）
        # x, y, w, h = roi
        # undistorted = undistorted[y:y+h, x:x+w]
        
        return undistorted
    
    def get_capture_instruction(self) -> Dict:
        """次の撮影指示を生成"""
        with self._lock:
            count0 = len(self._captured_data[0])
            count1 = len(self._captured_data[1])
            min_count = min(count0, count1)
            
            if min_count >= 20:
                return {
                    "status": "ready",
                    "message": "十分な画像が収集されました。キャリブレーションを実行できます。",
                    "instruction": None,
                    "count": min_count
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
            
            position = positions[min_count % len(positions)]
            
            return {
                "status": "collecting",
                "message": f"画像 {min_count + 1}/20",
                "instruction": f"チェッカーボードを{position}に配置してください",
                "count": min_count
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
