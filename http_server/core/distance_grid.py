"""距離グリッドシステム - 路面平面への距離グリッド投影"""
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, asdict
import threading


@dataclass
class GridCalibrationConfig:
    """グリッドキャリブレーション設定"""
    # カメラ取り付けパラメータ
    camera_height_mm: float = 150.0      # カメラの地面からの高さ (mm)
    camera_pitch_deg: float = 30.0       # カメラの俯角 (degrees, 下向きが正)
    camera_roll_deg: float = 0.0         # カメラのロール角 (degrees)
    
    # グリッド表示範囲
    grid_depth_min_m: float = 0.2        # グリッド最小奥行き (m)
    grid_depth_max_m: float = 3.0        # グリッド最大奥行き (m)
    grid_width_m: float = 1.0            # グリッド全幅 (m)
    
    # グリッド密度
    grid_depth_lines: int = 10           # 奥行き方向の線数
    grid_width_lines: int = 11           # 幅方向の線数
    
    # 微調整パラメータ
    vanishing_point_y: float = 0.3       # 消失点Y位置（0=上端、1=下端）
    perspective_strength: float = 1.0    # 透視効果の強さ


@dataclass
class GridPoint:
    """グリッドポイント（実世界座標とピクセル座標のペア）"""
    world_x: float  # 実世界X座標 (m)
    world_y: float  # 実世界Y座標 (m, 奥行き)
    pixel_x: float  # ピクセルX座標
    pixel_y: float  # ピクセルY座標


class DistanceGridManager:
    """距離グリッド管理"""
    
    def __init__(self, 
                 calibration_results_path: str = "calibration_data/calibration_results.json",
                 grid_config_path: str = "calibration_data/grid_config.json"):
        """
        Args:
            calibration_results_path: カメラキャリブレーション結果のパス
            grid_config_path: グリッド設定の保存パス
        """
        self.calibration_results_path = Path(calibration_results_path)
        self.grid_config_path = Path(grid_config_path)
        
        # カメラキャリブレーション結果
        self._camera_matrices: Dict[int, np.ndarray] = {}
        self._dist_coeffs: Dict[int, np.ndarray] = {}
        self._image_sizes: Dict[int, Tuple[int, int]] = {}
        
        # グリッド設定（カメラごと）
        self._configs: Dict[int, GridCalibrationConfig] = {}
        
        # ロック
        self._lock = threading.Lock()
        
        # キャリブレーション結果を読み込み
        self._load_calibration_results()
        
        # グリッド設定を読み込み
        self._load_grid_config()
        
        print(f"[DistanceGrid] Initialized")
    
    def _load_calibration_results(self):
        """カメラキャリブレーション結果を読み込み"""
        if not self.calibration_results_path.exists():
            print(f"[DistanceGrid] Calibration results not found: {self.calibration_results_path}")
            return
        
        try:
            with open(self.calibration_results_path) as f:
                data = json.load(f)
            
            for cam_id_str, cam_data in data.get("cameras", {}).items():
                cam_id = int(cam_id_str)
                self._camera_matrices[cam_id] = np.array(cam_data["camera_matrix"])
                self._dist_coeffs[cam_id] = np.array(cam_data["dist_coeffs"])
                self._image_sizes[cam_id] = tuple(cam_data["image_size"])
            
            print(f"[DistanceGrid] Loaded calibration for cameras: {list(self._camera_matrices.keys())}")
        except Exception as e:
            print(f"[DistanceGrid] Failed to load calibration: {e}")
    
    def _load_grid_config(self):
        """グリッド設定を読み込み"""
        if not self.grid_config_path.exists():
            print(f"[DistanceGrid] Grid config not found, using defaults")
            return
        
        try:
            with open(self.grid_config_path) as f:
                data = json.load(f)
            
            for cam_id_str, config_data in data.items():
                cam_id = int(cam_id_str)
                self._configs[cam_id] = GridCalibrationConfig(**config_data)
            
            print(f"[DistanceGrid] Loaded grid config for cameras: {list(self._configs.keys())}")
        except Exception as e:
            print(f"[DistanceGrid] Failed to load grid config: {e}")
    
    def _save_grid_config(self):
        """グリッド設定を保存"""
        data = {}
        for cam_id, config in self._configs.items():
            data[str(cam_id)] = asdict(config)
        
        self.grid_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.grid_config_path, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"[DistanceGrid] Saved grid config to {self.grid_config_path}")
    
    def get_config(self, camera_id: int) -> GridCalibrationConfig:
        """カメラのグリッド設定を取得"""
        with self._lock:
            if camera_id not in self._configs:
                self._configs[camera_id] = GridCalibrationConfig()
            return self._configs[camera_id]
    
    def update_config(self, camera_id: int, **kwargs) -> GridCalibrationConfig:
        """カメラのグリッド設定を更新"""
        with self._lock:
            if camera_id not in self._configs:
                self._configs[camera_id] = GridCalibrationConfig()
            
            config = self._configs[camera_id]
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            self._save_grid_config()
            return config
    
    def world_to_pixel(self, camera_id: int, world_x: float, world_y: float) -> Optional[Tuple[float, float]]:
        """実世界座標をピクセル座標に変換
        
        Args:
            camera_id: カメラID
            world_x: 実世界X座標 (m, 左が負、右が正)
            world_y: 実世界Y座標 (m, 奥行き、前方が正)
        
        Returns:
            (pixel_x, pixel_y) または None
        """
        if camera_id not in self._camera_matrices:
            return None
        
        config = self.get_config(camera_id)
        image_size = self._image_sizes.get(camera_id, (640, 480))
        
        # カメラ内部パラメータ
        K = self._camera_matrices[camera_id]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # カメラの取り付けパラメータ（mm → m変換）
        h = config.camera_height_mm / 1000.0
        pitch = np.radians(config.camera_pitch_deg)
        
        # 簡易的な投影計算（ピンホールカメラモデル + 路面平面）
        # 路面は Z=0 平面、カメラは (0, 0, h) にあり、pitch角で傾いている
        
        # 世界座標 (X, Y, 0) をカメラ座標系に変換
        # カメラ座標系: x=右、y=下、z=前
        X_world = world_x
        Y_world = world_y
        Z_world = 0  # 路面
        
        # 回転行列（pitch回転）
        R = np.array([
            [1, 0, 0],
            [0, np.cos(pitch), -np.sin(pitch)],
            [0, np.sin(pitch), np.cos(pitch)]
        ])
        
        # カメラ位置
        t = np.array([0, -h * np.cos(pitch), h * np.sin(pitch)])
        
        # 世界座標 → カメラ座標
        P_world = np.array([X_world, Z_world, Y_world])  # Y_worldが奥行き
        P_cam = R @ P_world + t
        
        x_cam, y_cam, z_cam = P_cam
        
        # カメラ後方は投影しない
        if z_cam <= 0.01:
            return None
        
        # 投影
        u = fx * (x_cam / z_cam) + cx
        v = fy * (y_cam / z_cam) + cy
        
        return (u, v)
    
    def compute_grid_points(self, camera_id: int) -> List[GridPoint]:
        """グリッドポイントを計算
        
        Args:
            camera_id: カメラID
        
        Returns:
            グリッドポイントのリスト
        """
        config = self.get_config(camera_id)
        points = []
        
        # 奥行き方向の距離リスト
        depth_step = (config.grid_depth_max_m - config.grid_depth_min_m) / (config.grid_depth_lines - 1)
        depths = [config.grid_depth_min_m + i * depth_step for i in range(config.grid_depth_lines)]
        
        # 幅方向の位置リスト
        width_step = config.grid_width_m / (config.grid_width_lines - 1)
        half_width = config.grid_width_m / 2
        widths = [-half_width + i * width_step for i in range(config.grid_width_lines)]
        
        for y in depths:
            for x in widths:
                pixel = self.world_to_pixel(camera_id, x, y)
                if pixel is not None:
                    u, v = pixel
                    points.append(GridPoint(
                        world_x=x,
                        world_y=y,
                        pixel_x=u,
                        pixel_y=v
                    ))
        
        return points
    
    def compute_grid_lines(self, camera_id: int) -> Dict[str, List]:
        """グリッド線を計算
        
        Args:
            camera_id: カメラID
        
        Returns:
            {
                "horizontal_lines": [[(x1,y1), (x2,y2), depth_m], ...],
                "vertical_lines": [[(x1,y1), (x2,y2), offset_m], ...],
                "center_line": [(x1,y1), (x2,y2)]
            }
        """
        config = self.get_config(camera_id)
        
        horizontal_lines = []
        vertical_lines = []
        center_line = []
        
        # 奥行き方向の距離リスト
        depth_step = (config.grid_depth_max_m - config.grid_depth_min_m) / (config.grid_depth_lines - 1)
        depths = [config.grid_depth_min_m + i * depth_step for i in range(config.grid_depth_lines)]
        
        # 幅方向の位置リスト
        width_step = config.grid_width_m / (config.grid_width_lines - 1)
        half_width = config.grid_width_m / 2
        widths = [-half_width + i * width_step for i in range(config.grid_width_lines)]
        
        # 水平線（奥行き方向の各距離）
        for y in depths:
            line_points = []
            for x in widths:
                pixel = self.world_to_pixel(camera_id, x, y)
                if pixel is not None:
                    line_points.append(pixel)
            if len(line_points) >= 2:
                horizontal_lines.append({
                    "points": line_points,
                    "depth_m": y
                })
        
        # 垂直線（幅方向の各位置）
        for x in widths:
            line_points = []
            for y in depths:
                pixel = self.world_to_pixel(camera_id, x, y)
                if pixel is not None:
                    line_points.append(pixel)
            if len(line_points) >= 2:
                vertical_lines.append({
                    "points": line_points,
                    "offset_m": x,
                    "is_center": abs(x) < 0.001
                })
        
        # 中央線
        center_points = []
        for y in np.linspace(config.grid_depth_min_m, config.grid_depth_max_m, 50):
            pixel = self.world_to_pixel(camera_id, 0, y)
            if pixel is not None:
                center_points.append(pixel)
        if len(center_points) >= 2:
            center_line = center_points
        
        return {
            "horizontal_lines": horizontal_lines,
            "vertical_lines": vertical_lines,
            "center_line": center_line
        }
    
    def draw_grid_overlay(self, camera_id: int, image: np.ndarray, 
                          color: Tuple[int, int, int] = (0, 255, 0),
                          thickness: int = 1,
                          show_labels: bool = True) -> np.ndarray:
        """画像にグリッドオーバーレイを描画
        
        Args:
            camera_id: カメラID
            image: 入力画像
            color: グリッド色 (BGR)
            thickness: 線の太さ
            show_labels: 距離ラベルを表示するか
        
        Returns:
            グリッドが描画された画像
        """
        result = image.copy()
        
        grid_data = self.compute_grid_lines(camera_id)
        
        # 水平線（奥行き方向）
        for line_data in grid_data["horizontal_lines"]:
            points = line_data["points"]
            depth = line_data["depth_m"]
            
            # 線を描画
            for i in range(len(points) - 1):
                p1 = (int(points[i][0]), int(points[i][1]))
                p2 = (int(points[i+1][0]), int(points[i+1][1]))
                cv2.line(result, p1, p2, color, thickness)
            
            # 距離ラベル
            if show_labels and len(points) > 0:
                label_pt = points[-1]  # 右端にラベル
                label_x = int(label_pt[0]) + 5
                label_y = int(label_pt[1])
                
                # 背景を描画
                label_text = f"{depth:.1f}m"
                (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                cv2.rectangle(result, 
                             (label_x - 2, label_y - th - 2),
                             (label_x + tw + 2, label_y + 2),
                             (0, 0, 0), -1)
                cv2.putText(result, label_text, (label_x, label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 垂直線
        for line_data in grid_data["vertical_lines"]:
            points = line_data["points"]
            is_center = line_data.get("is_center", False)
            
            line_color = (0, 255, 255) if is_center else color  # 中央線は黄色
            line_thickness = thickness + 1 if is_center else thickness
            
            for i in range(len(points) - 1):
                p1 = (int(points[i][0]), int(points[i][1]))
                p2 = (int(points[i+1][0]), int(points[i+1][1]))
                cv2.line(result, p1, p2, line_color, line_thickness)
        
        return result
    
    def get_distance_at_point(self, camera_id: int, pixel_x: int, pixel_y: int) -> Optional[Dict]:
        """ピクセル座標から推定距離を取得
        
        Args:
            camera_id: カメラID
            pixel_x: ピクセルX座標
            pixel_y: ピクセルY座標
        
        Returns:
            {"distance_m": float, "lateral_offset_m": float} または None
        """
        if camera_id not in self._camera_matrices:
            return None
        
        config = self.get_config(camera_id)
        image_size = self._image_sizes.get(camera_id, (640, 480))
        
        K = self._camera_matrices[camera_id]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        h = config.camera_height_mm / 1000.0
        pitch = np.radians(config.camera_pitch_deg)
        
        # ピクセル座標から正規化画像座標
        x_norm = (pixel_x - cx) / fx
        y_norm = (pixel_y - cy) / fy
        
        # カメラ光線の方向（カメラ座標系）
        ray_dir = np.array([x_norm, y_norm, 1.0])
        ray_dir = ray_dir / np.linalg.norm(ray_dir)
        
        # 回転行列（pitch）の逆変換
        R_inv = np.array([
            [1, 0, 0],
            [0, np.cos(-pitch), -np.sin(-pitch)],
            [0, np.sin(-pitch), np.cos(-pitch)]
        ])
        
        # ワールド座標系での光線方向
        ray_world = R_inv @ ray_dir
        
        # カメラ位置（ワールド座標系）
        cam_pos = np.array([0, 0, h])
        
        # 路面（Z=0）との交点を計算
        if abs(ray_world[2]) < 1e-6:
            return None
        
        t = -cam_pos[2] / ray_world[2]
        if t <= 0:
            return None
        
        intersection = cam_pos + t * ray_world
        
        return {
            "distance_m": float(intersection[1]),  # Y方向が奥行き
            "lateral_offset_m": float(intersection[0])  # X方向が横
        }
    
    def get_status(self, camera_id: int) -> Dict:
        """ステータスを取得"""
        config = self.get_config(camera_id)
        has_calibration = camera_id in self._camera_matrices
        
        return {
            "camera_id": camera_id,
            "has_camera_calibration": has_calibration,
            "config": asdict(config),
            "image_size": self._image_sizes.get(camera_id)
        }


# シングルトンインスタンス
distance_grid_manager = DistanceGridManager()
