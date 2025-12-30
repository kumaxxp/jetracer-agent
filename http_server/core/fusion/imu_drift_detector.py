"""IMUドリフト検出器

検出ロジック:
- モーター音なし（車両停止中）
- heading が一方向に継続的に変化
- 加速度は安定（持ち上げと区別）
"""
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from .config import fusion_config
from .data_source import get_data_source


@dataclass
class DriftResult:
    """IMUドリフト検出結果"""
    drifting: bool = False
    drift_rate: float = 0.0  # deg/s
    drift_direction: str = "none"  # "cw", "ccw", "none"
    duration: float = 0.0
    confidence: float = 0.0
    accel_stable: bool = True
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "drifting": self.drifting,
            "drift_rate_deg_s": round(self.drift_rate, 3),
            "drift_direction": self.drift_direction,
            "duration": round(self.duration, 2),
            "confidence": round(self.confidence, 2),
            "accel_stable": self.accel_stable,
            "timestamp": self.timestamp
        }


class IMUDriftDetector:
    """IMUドリフト検出器"""

    def __init__(self):
        self.config = fusion_config
        self._drift_start_time: Optional[float] = None

    def check(self) -> DriftResult:
        """IMUドリフト状態をチェック"""
        result = DriftResult(timestamp=time.time())

        data_source = get_data_source()
        snapshot = data_source.get_snapshot()

        if not snapshot.acoustic_valid or not snapshot.imu_valid:
            return result

        # モーター停止チェック
        motor_stopped = snapshot.zcr < self.config.motor_stopped_zcr

        if not motor_stopped:
            # モーターが動いている場合はドリフト検出しない
            self._drift_start_time = None
            return result

        # heading履歴を取得
        heading_history = data_source.get_heading_history(seconds=5.0)
        if len(heading_history) < 10:
            return result

        # ドリフト速度を計算（線形回帰）
        times = np.array([h[0] for h in heading_history])
        headings = np.array([h[1] for h in heading_history])

        # 角度のwrap-around処理
        # 累積変化量を計算
        cumulative_change = 0.0
        for i in range(1, len(headings)):
            diff = headings[i] - headings[i-1]
            if diff > 180:
                diff -= 360
            elif diff < -180:
                diff += 360
            cumulative_change += diff

        duration = times[-1] - times[0]
        if duration > 0:
            result.drift_rate = cumulative_change / duration

        # 加速度安定性チェック
        accel_history = data_source.get_accel_magnitude_history(seconds=1.0)
        if accel_history:
            accel_values = [a[1] for a in accel_history]
            accel_variance = np.var(accel_values)
            result.accel_stable = accel_variance < self.config.accel_stable_threshold

        # ドリフト方向
        if abs(result.drift_rate) > self.config.drift_rate_threshold:
            result.drift_direction = "cw" if result.drift_rate > 0 else "ccw"

        # ドリフト判定（モーター停止 AND 継続的なheading変化 AND 加速度安定）
        is_drifting = (
            motor_stopped and
            abs(result.drift_rate) > self.config.drift_rate_threshold and
            result.accel_stable
        )

        if is_drifting:
            if self._drift_start_time is None:
                self._drift_start_time = time.time()

            result.duration = time.time() - self._drift_start_time

            if result.duration >= self.config.drift_min_duration:
                result.drifting = True
                result.confidence = min(1.0, result.duration / (self.config.drift_min_duration * 2))
        else:
            self._drift_start_time = None

        return result


# シングルトン
_drift_detector: Optional[IMUDriftDetector] = None


def get_drift_detector() -> IMUDriftDetector:
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = IMUDriftDetector()
    return _drift_detector
