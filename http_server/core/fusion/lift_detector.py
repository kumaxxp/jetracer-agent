"""持ち上げ検出器

検出ロジック:
- モーター音なし (ZCR < threshold)
- 加速度が不規則に変動
- heading変化が不規則（一方向ではない）
"""
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import fusion_config
from .data_source import get_data_source


@dataclass
class LiftResult:
    """持ち上げ検出結果"""
    lifted: bool = False
    confidence: float = 0.0
    accel_variance: float = 0.0
    heading_variance: float = 0.0
    gravity_z_deviation: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "lifted": self.lifted,
            "confidence": round(self.confidence, 2),
            "accel_variance": round(self.accel_variance, 3),
            "heading_variance": round(self.heading_variance, 2),
            "gravity_z_deviation": round(self.gravity_z_deviation, 3),
            "timestamp": self.timestamp
        }


class LiftDetector:
    """持ち上げ検出器"""

    GRAVITY = 9.81  # m/s²

    def __init__(self):
        self.config = fusion_config

    def check(self) -> LiftResult:
        """持ち上げ状態をチェック"""
        result = LiftResult(timestamp=time.time())

        data_source = get_data_source()
        snapshot = data_source.get_snapshot()

        if not snapshot.acoustic_valid or not snapshot.imu_valid:
            return result

        # モーター停止チェック
        motor_stopped = snapshot.zcr < self.config.motor_stopped_zcr

        if not motor_stopped:
            # モーターが動いている場合は持ち上げではない
            return result

        # 加速度履歴から変動を計算
        accel_history = data_source.get_accel_magnitude_history(seconds=1.0)
        if len(accel_history) < 5:
            return result

        accel_values = [a[1] for a in accel_history]
        result.accel_variance = float(np.var(accel_values))

        # heading履歴から変動を計算
        heading_history = data_source.get_heading_history(seconds=1.0)
        if len(heading_history) >= 5:
            heading_values = [h[1] for h in heading_history]
            # 角度のwrap-aroundを考慮
            heading_diffs = []
            for i in range(1, len(heading_values)):
                diff = heading_values[i] - heading_values[i-1]
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360
                heading_diffs.append(diff)
            result.heading_variance = float(np.var(heading_diffs)) if heading_diffs else 0.0

        # Z軸加速度の重力からの偏差
        result.gravity_z_deviation = abs(snapshot.accel_z - self.GRAVITY)

        # 持ち上げ判定
        high_accel_variance = result.accel_variance > self.config.accel_variance_threshold
        high_heading_variance = result.heading_variance > self.config.heading_variance_threshold
        gravity_abnormal = result.gravity_z_deviation > self.config.gravity_deviation

        # 複数条件のOR（いずれかが満たされれば持ち上げ）
        lift_score = 0
        if high_accel_variance:
            lift_score += 0.4
        if high_heading_variance:
            lift_score += 0.3
        if gravity_abnormal:
            lift_score += 0.3

        result.confidence = lift_score
        result.lifted = lift_score >= 0.5

        return result


# シングルトン
_lift_detector: Optional[LiftDetector] = None


def get_lift_detector() -> LiftDetector:
    global _lift_detector
    if _lift_detector is None:
        _lift_detector = LiftDetector()
    return _lift_detector
