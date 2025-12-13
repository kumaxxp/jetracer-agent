"""ステアリング計算モジュール

セグメンテーション結果からステアリング/スロットル値を計算する。
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import time


@dataclass
class SteeringParams:
    """ステアリング計算パラメータ"""
    # ステアリング
    steering_gain: float = 1.5           # ステアリング感度
    steering_deadzone: float = 0.05      # デッドゾーン（この範囲は0とみなす）
    steering_max: float = 1.0            # 最大ステアリング値
    
    # スロットル
    throttle_base: float = 0.15          # 基本スロットル
    throttle_min: float = 0.10           # 最小スロットル（カーブ時）
    throttle_max: float = 0.25           # 最大スロットル（直進時）
    curve_reduction: float = 0.3         # カーブ時のスロットル減少率
    
    # ROAD判定
    road_stop_threshold: float = 0.10    # これ以下でROAD不足停止
    road_slow_threshold: float = 0.30    # これ以下で減速
    
    # 重み付け
    near_weight: float = 1.0             # 下部（足元）の重み
    far_weight: float = 0.3              # 上部（遠方）の重み


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


class SteeringCalculator:
    """ステアリング計算エンジン"""
    
    def __init__(self, params: SteeringParams = None):
        self.params = params or SteeringParams()
        self._last_steering = 0.0
        self._last_throttle = 0.0
    
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
            # 縦方向の重み（下部ほど重い）
            weights = np.linspace(self.params.far_weight, self.params.near_weight, h).reshape(-1, 1)
            weighted_mask = binary_mask.astype(float) * weights
            
            # X方向の重心
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
    
    def calculate_steering(self, analysis: CameraAnalysis) -> SteeringCommand:
        """
        単一カメラの分析からステアリングを計算
        
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
        # centroid_x: 0=左端, 0.5=中央, 1=右端
        # steering: -1=左, 0=中央, +1=右
        offset = analysis.centroid_x - 0.5  # -0.5 〜 +0.5
        raw_steering = offset * 2 * self.params.steering_gain  # -gain 〜 +gain
        
        # デッドゾーン処理
        if abs(raw_steering) < self.params.steering_deadzone:
            raw_steering = 0.0
        
        # クリップ
        steering = np.clip(raw_steering, -self.params.steering_max, self.params.steering_max)
        
        # 境界補正（壁があれば反対方向に補正）
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
    
    def calculate_dual_camera(
        self,
        front_analysis: CameraAnalysis,
        ground_analysis: CameraAnalysis
    ) -> SteeringCommand:
        """
        デュアルカメラ（正面+足元）からステアリングを計算
        
        正面カメラ: 戦略的な方向決定
        足元カメラ: 即時の安全確認
        """
        # 足元でROAD不足なら緊急停止
        if ground_analysis.road_ratio < self.params.road_stop_threshold:
            return SteeringCommand(
                steering=0.0,
                throttle=0.0,
                stop=True,
                reason=f"足元ROAD不足: {ground_analysis.road_ratio:.1%}",
                road_ratio=ground_analysis.road_ratio
            )
        
        # 基本は正面カメラでステアリング計算
        front_cmd = self.calculate_steering(front_analysis)
        
        if front_cmd.stop:
            # 正面もROAD不足なら停止
            return front_cmd
        
        steering = front_cmd.steering
        
        # 足元の状況で補正
        if ground_analysis.boundary_left and steering < 0.15:
            steering = max(steering, 0.15)
        if ground_analysis.boundary_right and steering > -0.15:
            steering = min(steering, -0.15)
        
        # スロットル計算（足元のROAD比率も考慮）
        effective_road_ratio = min(front_analysis.road_ratio, ground_analysis.road_ratio)
        throttle = self._calculate_throttle(steering, effective_road_ratio)
        
        # 理由生成
        reason_parts = [front_cmd.reason]
        if ground_analysis.boundary_left:
            reason_parts.append("左壁")
        if ground_analysis.boundary_right:
            reason_parts.append("右壁")
        
        return SteeringCommand(
            steering=round(steering, 3),
            throttle=round(throttle, 3),
            stop=False,
            reason=" / ".join(reason_parts),
            road_ratio=effective_road_ratio,
            centroid_x=front_analysis.centroid_x,
            raw_steering=front_cmd.raw_steering
        )
    
    def _calculate_throttle(self, steering: float, road_ratio: float) -> float:
        """スロットル計算"""
        # 基本スロットル
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
        
        # クリップ
        throttle = np.clip(throttle, self.params.throttle_min, self.params.throttle_max)
        
        return throttle
    
    def _generate_reason(self, steering: float, analysis: CameraAnalysis) -> str:
        """判断理由を生成"""
        parts = []
        
        # ステアリング方向
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
        
        # ROAD状況
        parts.append(f"ROAD:{analysis.road_ratio:.0%}")
        
        return " ".join(parts)
    
    def update_params(self, **kwargs):
        """パラメータを更新"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
    
    def get_params(self) -> dict:
        """現在のパラメータを取得"""
        return {
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
        }


# シングルトンインスタンス
steering_calculator = SteeringCalculator()
