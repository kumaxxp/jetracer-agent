"""ステアリング計算モジュール

セグメンテーション結果からステアリング/スロットル値を計算する。
2つのモード：
- centroid: 重心ベース（シンプル、高速）
- grid: グリッド経路追従（高度、正確）
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
import time


class SteeringMode(Enum):
    """ステアリング計算モード"""
    CENTROID = "centroid"  # 重心ベース（従来方式）
    GRID = "grid"          # グリッド経路追従


@dataclass
class SteeringParams:
    """ステアリング計算パラメータ"""
    # 計算モード
    mode: SteeringMode = SteeringMode.GRID
    
    # ステアリング
    steering_gain: float = 1.5           # ステアリング感度
    steering_deadzone: float = 0.05      # デッドゾーン（この範囲は0とみなす）
    steering_max: float = 1.0            # 最大ステアリング値
    
    # スロットル
    throttle_base: float = 0.75          # 基本スロットル
    throttle_min: float = 0.10           # 最小スロットル（カーブ時）
    throttle_max: float = 0.75           # 最大スロットル（直進時）
    curve_reduction: float = 0.3         # カーブ時のスロットル減少率
    
    # ROAD判定
    road_stop_threshold: float = 0.10    # これ以下でROAD不足停止
    road_slow_threshold: float = 0.30    # これ以下で減速
    
    # 重み付け（centroidモード用）
    near_weight: float = 1.0             # 下部（足元）の重み
    far_weight: float = 0.3              # 上部（遠方）の重み
    
    # グリッド設定（gridモード用）
    grid_rows: int = 4                   # グリッド行数
    grid_cols: int = 5                   # グリッド列数
    grid_passable_threshold: float = 0.3 # 通行可能とみなす閾値
    exclude_bottom_rows: int = 2         # 車体が映る下部行数（経路計算から除外）


@dataclass
class SteeringCommand:
    """ステアリングコマンド"""
    steering: float          # -1.0（左）〜 +1.0（右）
    throttle: float          # 0.0〜1.0
    stop: bool = False       # 停止フラグ
    reason: str = ""         # 判断理由
    
    # デバッグ情報
    road_ratio: float = 0.0
    centroid_x: float = 0.5
    raw_steering: float = 0.0
    recommended_path: List[int] = None  # グリッドモード時の経路
    
    def __post_init__(self):
        if self.recommended_path is None:
            self.recommended_path = []


@dataclass  
class CameraAnalysis:
    """カメラ分析結果"""
    road_mask: np.ndarray           # ROADマスク (H, W)
    road_ratio: float               # ROAD比率
    centroid_x: float               # ROAD重心X（0-1、0.5が中央）
    left_ratio: float               # 左1/3のROAD比率
    center_ratio: float             # 中央1/3のROAD比率
    right_ratio: float              # 右1/3のROAD比率
    boundary_left: bool             # 左端に壁があるか
    boundary_right: bool            # 右端に壁があるか
    timestamp: float


@dataclass
class GridAnalysis:
    """グリッド分析結果"""
    cell_values: List[List[float]]  # グリッドセルの通行可能性 [row][col]
    road_ratio: float               # 全体のROAD比率
    recommended_path: List[int]     # 各行の推奨列
    recommended_direction: str      # "left", "center", "right"
    confidence: float               # 信頼度
    timestamp: float


class SteeringCalculator:
    """ステアリング計算エンジン"""
    
    def __init__(self, params: SteeringParams = None):
        self.params = params or SteeringParams()
        self._last_steering = 0.0
        self._last_throttle = 0.0
    
    # =========================================================================
    # メイン計算メソッド
    # =========================================================================
    
    def calculate(
        self, 
        road_mask: np.ndarray = None,
        cell_analysis: List[List[float]] = None
    ) -> SteeringCommand:
        """
        ステアリングを計算（モードに応じて自動選択）
        
        Args:
            road_mask: ROADマスク（centroidモード用）
            cell_analysis: グリッドセル分析結果（gridモード用）
        """
        if self.params.mode == SteeringMode.GRID and cell_analysis:
            return self.calculate_steering_grid(cell_analysis)
        elif road_mask is not None:
            analysis = self.analyze_road_mask(road_mask)
            return self.calculate_steering_centroid(analysis)
        else:
            return SteeringCommand(
                steering=0.0, throttle=0.0, stop=True,
                reason="入力データなし"
            )
    
    # =========================================================================
    # グリッドベース経路追従（新方式）
    # =========================================================================
    
    def calculate_steering_grid(self, cell_analysis: List[List[float]]) -> SteeringCommand:
        """
        グリッドセル分析から経路追従ステアリングを計算
        
        Args:
            cell_analysis: グリッドセルの通行可能性 [row][col] = 0.0〜1.0
                          Row 0 = 遠方、Row N = 手前
                          -1.0 = 画面外（無視）
        """
        if not cell_analysis or not cell_analysis[0]:
            return SteeringCommand(
                steering=0.0, throttle=0.0, stop=True,
                reason="グリッドデータなし"
            )
        
        num_rows = len(cell_analysis)
        num_cols = len(cell_analysis[0])
        
        # 全体のROAD比率を計算
        valid_cells = []
        for row in cell_analysis:
            for val in row:
                if val >= 0:  # -1は画面外
                    valid_cells.append(val)
        
        road_ratio = sum(valid_cells) / len(valid_cells) if valid_cells else 0.0
        
        # ROAD不足チェック
        if road_ratio < self.params.road_stop_threshold:
            return SteeringCommand(
                steering=0.0, throttle=0.0, stop=True,
                reason=f"ROAD不足: {road_ratio:.1%}",
                road_ratio=road_ratio
            )
        
        # 連続した推奨経路を計算
        recommended_path = self._compute_continuous_path(cell_analysis)
        
        # 経路からステアリング計算
        raw_steering = self._compute_steering_from_path(recommended_path, num_cols)
        
        # デッドゾーン処理
        if abs(raw_steering) < self.params.steering_deadzone:
            raw_steering = 0.0
        
        # クリップ
        steering = np.clip(raw_steering * self.params.steering_gain, 
                          -self.params.steering_max, self.params.steering_max)
        
        # スロットル計算
        throttle = self._calculate_throttle(steering, road_ratio)
        
        # 方向判定
        if steering < -0.3:
            direction = "左旋回"
        elif steering > 0.3:
            direction = "右旋回"
        elif steering < -0.1:
            direction = "やや左"
        elif steering > 0.1:
            direction = "やや右"
        else:
            direction = "直進"
        
        reason = f"{direction} ROAD:{road_ratio:.0%}"
        
        return SteeringCommand(
            steering=round(steering, 3),
            throttle=round(throttle, 3),
            stop=False,
            reason=reason,
            road_ratio=road_ratio,
            centroid_x=0.5,  # グリッドモードでは使用しない
            raw_steering=raw_steering,
            recommended_path=recommended_path
        )
    
    def _compute_continuous_path(self, grid: List[List[float]]) -> List[int]:
        """
        連続した推奨経路を重心ベースで計算
        
        各行で通行可能領域の重心を計算し、滑らかに繋げる。
        手前の数行（車体が映る部分）は経路計算から除外。
        """
        if not grid or not grid[0]:
            return []
        
        num_rows = len(grid)
        num_cols = len(grid[0])
        center_col = (num_cols - 1) / 2
        
        # 手前の行は車体が映るため除外して計算
        exclude_rows = min(self.params.exclude_bottom_rows, num_rows - 1)
        effective_rows = num_rows - exclude_rows
        
        if effective_rows <= 0:
            return [int(center_col)] * num_rows
        
        # 各行で通行可能領域の重心を計算
        row_centers = []
        for row in range(effective_rows):
            total_weight = 0.0
            weighted_sum = 0.0
            
            for col in range(num_cols):
                value = grid[row][col]
                # -1.0は画面外なのでスキップ
                if value < 0:
                    continue
                if value > self.params.grid_passable_threshold:
                    # 中央に近いほどボーナス
                    center_bonus = 1.0 + 0.2 * (1.0 - abs(col - center_col) / center_col) if center_col > 0 else 1.0
                    weight = value * center_bonus
                    weighted_sum += col * weight
                    total_weight += weight
            
            if total_weight > 0:
                row_center = weighted_sum / total_weight
            else:
                row_center = center_col
            
            row_centers.append(row_center)
        
        # 滑らかな経路を生成（移動平均）
        smoothed_path = []
        window_size = 3
        
        for i in range(len(row_centers)):
            start = max(0, i - window_size // 2)
            end = min(len(row_centers), i + window_size // 2 + 1)
            avg = sum(row_centers[start:end]) / (end - start)
            smoothed_path.append(round(avg))
        
        # 除外した手前の行には、最後の有効な値を使用
        last_col = smoothed_path[-1] if smoothed_path else int(center_col)
        for _ in range(exclude_rows):
            smoothed_path.append(last_col)
        
        # 整数に変換し範囲内に収める
        path = [max(0, min(num_cols - 1, int(c))) for c in smoothed_path]
        
        return path
    
    def _compute_steering_from_path(self, path: List[int], num_cols: int) -> float:
        """
        経路からステアリング値を計算
        
        手前の行（すぐ前）を重視してステアリングを計算。
        """
        if not path or num_cols <= 1:
            return 0.0
        
        center = (num_cols - 1) / 2
        
        # 手前の行を重視する重み付け
        # path[0] = 奥、path[-1] = 手前
        weights = []
        for i in range(len(path)):
            # 手前ほど重みが大きい（指数的）
            weight = 2.0 ** (i / len(path))
            weights.append(weight)
        
        total_weight = sum(weights)
        weighted_col = sum(path[i] * weights[i] for i in range(len(path))) / total_weight
        
        # 中央からのオフセットを-1～1に正規化
        steering = (weighted_col - center) / center if center > 0 else 0.0
        
        return max(-1.0, min(1.0, steering))
    
    def build_grid_from_mask(self, road_mask: np.ndarray) -> List[List[float]]:
        """
        ROADマスクからグリッドセル分析を構築
        
        Args:
            road_mask: 二値マスク (H, W)
            
        Returns:
            cell_analysis: グリッドセルの通行可能性 [row][col]
        """
        h, w = road_mask.shape[:2]
        
        # 二値マスクに変換
        if road_mask.dtype != bool:
            binary_mask = (road_mask == 1)
        else:
            binary_mask = road_mask
        
        num_rows = self.params.grid_rows
        num_cols = self.params.grid_cols
        
        row_height = h // num_rows
        col_width = w // num_cols
        
        cell_analysis = []
        for row in range(num_rows):
            row_values = []
            y_start = row * row_height
            y_end = (row + 1) * row_height if row < num_rows - 1 else h
            
            for col in range(num_cols):
                x_start = col * col_width
                x_end = (col + 1) * col_width if col < num_cols - 1 else w
                
                cell = binary_mask[y_start:y_end, x_start:x_end]
                if cell.size > 0:
                    ratio = cell.sum() / cell.size
                else:
                    ratio = 0.0
                
                row_values.append(float(ratio))
            
            cell_analysis.append(row_values)
        
        return cell_analysis
    
    # =========================================================================
    # 重心ベース計算（従来方式）
    # =========================================================================
    
    def analyze_road_mask(self, road_mask: np.ndarray) -> CameraAnalysis:
        """
        ROADマスクを分析
        
        Args:
            road_mask: 二値マスク (H, W) または セグメンテーションマスク
                       ROADクラス=1を想定
        """
        h, w = road_mask.shape[:2]
        
        # 二値マスクに変換（クラス1がROAD）
        if road_mask.dtype != bool:
            binary_mask = (road_mask == 1)
        else:
            binary_mask = road_mask
        
        # 全体のROAD比率
        total_pixels = binary_mask.size
        road_pixels = binary_mask.sum()
        road_ratio = road_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # 左/中央/右の分割
        w_third = w // 3
        left_mask = binary_mask[:, :w_third]
        center_mask = binary_mask[:, w_third:2*w_third]
        right_mask = binary_mask[:, 2*w_third:]
        
        left_ratio = left_mask.sum() / left_mask.size if left_mask.size > 0 else 0.0
        center_ratio = center_mask.sum() / center_mask.size if center_mask.size > 0 else 0.0
        right_ratio = right_mask.sum() / right_mask.size if right_mask.size > 0 else 0.0
        
        # 境界検出（端10%）
        edge_width = w // 10
        left_edge = binary_mask[:, :edge_width]
        right_edge = binary_mask[:, -edge_width:]
        
        boundary_left = (left_edge.sum() / left_edge.size) < 0.3 if left_edge.size > 0 else False
        boundary_right = (right_edge.sum() / right_edge.size) < 0.3 if right_edge.size > 0 else False
        
        # ROAD重心計算（下部重み付け）
        if road_ratio > 0.01:
            weights = np.linspace(self.params.far_weight, self.params.near_weight, h).reshape(-1, 1)
            weighted_mask = binary_mask.astype(float) * weights
            
            x_coords = np.arange(w)
            weighted_sum = weighted_mask.sum()
            if weighted_sum > 0:
                centroid_x = (weighted_mask.sum(axis=0) * x_coords).sum() / weighted_sum / w
            else:
                centroid_x = 0.5
        else:
            centroid_x = 0.5
        
        return CameraAnalysis(
            road_mask=binary_mask,
            road_ratio=road_ratio,
            centroid_x=centroid_x,
            left_ratio=left_ratio,
            center_ratio=center_ratio,
            right_ratio=right_ratio,
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            timestamp=time.time()
        )
    
    def calculate_steering_centroid(self, analysis: CameraAnalysis) -> SteeringCommand:
        """
        重心ベースのステアリング計算（従来方式）
        
        Args:
            analysis: CameraAnalysis結果
        """
        # ROAD不足チェック
        if analysis.road_ratio < self.params.road_stop_threshold:
            return SteeringCommand(
                steering=0.0,
                throttle=0.0,
                stop=True,
                reason=f"ROAD不足: {analysis.road_ratio:.1%}",
                road_ratio=analysis.road_ratio,
                centroid_x=analysis.centroid_x
            )
        
        # 重心からステアリング計算
        offset = analysis.centroid_x - 0.5
        raw_steering = offset * 2 * self.params.steering_gain
        
        # デッドゾーン処理
        if abs(raw_steering) < self.params.steering_deadzone:
            raw_steering = 0.0
        
        # クリップ
        steering = np.clip(raw_steering, -self.params.steering_max, self.params.steering_max)
        
        # 境界補正
        if analysis.boundary_left and steering < 0.1:
            steering = max(steering, 0.1)
        if analysis.boundary_right and steering > -0.1:
            steering = min(steering, -0.1)
        
        # スロットル計算
        throttle = self._calculate_throttle(steering, analysis.road_ratio)
        
        # 理由生成
        reason = self._generate_reason(steering, analysis)
        
        return SteeringCommand(
            steering=round(steering, 3),
            throttle=round(throttle, 3),
            stop=False,
            reason=reason,
            road_ratio=analysis.road_ratio,
            centroid_x=analysis.centroid_x,
            raw_steering=raw_steering
        )
    
    # =========================================================================
    # 共通ユーティリティ
    # =========================================================================
    
    def _calculate_throttle(self, steering: float, road_ratio: float) -> float:
        """スロットル計算"""
        throttle = self.params.throttle_base
        
        # カーブ時は減速
        curve_factor = abs(steering)
        if curve_factor > 0.3:
            reduction = self.params.curve_reduction * (curve_factor - 0.3) / 0.7
            throttle *= (1 - reduction)
        
        # ROAD比率が低い時は減速
        if road_ratio < self.params.road_slow_threshold:
            slow_factor = road_ratio / self.params.road_slow_threshold
            throttle *= slow_factor
        
        return np.clip(throttle, self.params.throttle_min, self.params.throttle_max)
    
    def _generate_reason(self, steering: float, analysis: CameraAnalysis) -> str:
        """判断理由を生成"""
        parts = []
        
        if steering < -0.3:
            parts.append("左旋回")
        elif steering > 0.3:
            parts.append("右旋回")
        elif steering < -0.1:
            parts.append("やや左")
        elif steering > 0.1:
            parts.append("やや右")
        else:
            parts.append("直進")
        
        parts.append(f"ROAD:{analysis.road_ratio:.0%}")
        
        return " ".join(parts)
    
    def update_params(self, **kwargs):
        """パラメータを更新"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                if key == 'mode' and isinstance(value, str):
                    value = SteeringMode(value)
                setattr(self.params, key, value)
    
    def get_params(self) -> dict:
        """現在のパラメータを取得"""
        return {
            "mode": self.params.mode.value,
            "steering_gain": self.params.steering_gain,
            "steering_deadzone": self.params.steering_deadzone,
            "steering_max": self.params.steering_max,
            "throttle_base": self.params.throttle_base,
            "throttle_min": self.params.throttle_min,
            "throttle_max": self.params.throttle_max,
            "curve_reduction": self.params.curve_reduction,
            "road_stop_threshold": self.params.road_stop_threshold,
            "road_slow_threshold": self.params.road_slow_threshold,
            "near_weight": self.params.near_weight,
            "far_weight": self.params.far_weight,
            "grid_rows": self.params.grid_rows,
            "grid_cols": self.params.grid_cols,
            "grid_passable_threshold": self.params.grid_passable_threshold,
            "exclude_bottom_rows": self.params.exclude_bottom_rows,
        }
    
    def set_mode(self, mode: str):
        """計算モードを設定"""
        self.params.mode = SteeringMode(mode)
    
    def get_mode(self) -> str:
        """現在の計算モードを取得"""
        return self.params.mode.value


# シングルトンインスタンス
steering_calculator = SteeringCalculator()
