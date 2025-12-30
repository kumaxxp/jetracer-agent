"""スタック（空転）検出器

検出ロジック:
- モーター音あり (ZCR > threshold)
- 加速度 ≈ 0 (車両は動いていない)
- 上記が一定時間継続
"""
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import fusion_config
from .data_source import get_data_source


@dataclass
class StuckResult:
    """スタック検出結果"""
    stuck: bool = False
    duration: float = 0.0
    confidence: float = 0.0
    motor_sound_level: float = 0.0
    accel_magnitude: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stuck": self.stuck,
            "duration": round(self.duration, 2),
            "confidence": round(self.confidence, 2),
            "motor_sound_level": round(self.motor_sound_level, 4),
            "accel_magnitude": round(self.accel_magnitude, 3),
            "timestamp": self.timestamp
        }


class StuckDetector:
    """スタック検出器"""

    def __init__(self):
        self.config = fusion_config
        self._stuck_start_time: Optional[float] = None

    def check(self) -> StuckResult:
        """スタック状態をチェック"""
        result = StuckResult(timestamp=time.time())

        data_source = get_data_source()
        snapshot = data_source.get_snapshot()

        if not snapshot.acoustic_valid or not snapshot.imu_valid:
            return result

        # モーター音レベル（ZCR）
        result.motor_sound_level = snapshot.zcr

        # 加速度magnitude（重力を除いた線形加速度の近似）
        # 簡易的にXY平面の加速度を使用
        accel_xy = np.sqrt(snapshot.accel_x**2 + snapshot.accel_y**2)
        result.accel_magnitude = accel_xy

        # スタック条件チェック
        motor_running = snapshot.zcr > self.config.motor_running_zcr
        vehicle_stopped = accel_xy < self.config.accel_stopped_threshold

        if motor_running and vehicle_stopped:
            # スタック状態開始/継続
            if self._stuck_start_time is None:
                self._stuck_start_time = time.time()

            duration = time.time() - self._stuck_start_time
            result.duration = duration

            # 最小継続時間を超えたらスタック判定
            if duration >= self.config.stuck_min_duration:
                result.stuck = True
                # 信頼度は継続時間に比例
                result.confidence = min(1.0, duration / (self.config.stuck_min_duration * 2))
        else:
            # スタック状態解除
            self._stuck_start_time = None
            result.duration = 0.0

        return result


# シングルトン
_stuck_detector: Optional[StuckDetector] = None


def get_stuck_detector() -> StuckDetector:
    global _stuck_detector
    if _stuck_detector is None:
        _stuck_detector = StuckDetector()
    return _stuck_detector
