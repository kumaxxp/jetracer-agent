"""衝突検出器

検出ロジック:
- 衝撃音（RMS急上昇）
- 加速度スパイク
"""
import time
from dataclasses import dataclass
from typing import Optional
from collections import deque
import numpy as np

from .config import fusion_config
from .data_source import get_data_source


@dataclass
class CollisionResult:
    """衝突検出結果"""
    collision: bool = False
    impact_strength: float = 0.0  # 0-1
    rms_spike: float = 0.0
    accel_spike: float = 0.0
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "collision": self.collision,
            "impact_strength": round(self.impact_strength, 2),
            "rms_spike": round(self.rms_spike, 4),
            "accel_spike": round(self.accel_spike, 2),
            "timestamp": self.timestamp
        }


class CollisionDetector:
    """衝突検出器"""

    def __init__(self):
        self.config = fusion_config
        self._rms_history: deque = deque(maxlen=20)
        self._last_collision_time: float = 0
        self._cooldown: float = 1.0  # 連続検出防止

    def check(self) -> CollisionResult:
        """衝突をチェック"""
        result = CollisionResult(timestamp=time.time())

        data_source = get_data_source()
        snapshot = data_source.get_snapshot()

        if not snapshot.acoustic_valid or not snapshot.imu_valid:
            return result

        # RMS履歴更新
        self._rms_history.append(snapshot.rms)

        # クールダウン中はスキップ
        if time.time() - self._last_collision_time < self._cooldown:
            return result

        # RMSスパイク検出
        if len(self._rms_history) >= 5:
            baseline_rms = np.mean(list(self._rms_history)[:-1])
            current_rms = snapshot.rms
            result.rms_spike = current_rms - baseline_rms

        # 加速度スパイク検出
        accel_mag = np.sqrt(
            snapshot.accel_x**2 +
            snapshot.accel_y**2 +
            (snapshot.accel_z - 9.81)**2  # 重力を除去
        )
        result.accel_spike = accel_mag

        # 衝突判定
        rms_impact = result.rms_spike > self.config.impact_rms_threshold
        accel_impact = result.accel_spike > self.config.impact_accel_threshold

        if rms_impact or accel_impact:
            result.collision = True
            self._last_collision_time = time.time()

            # 衝撃強度の計算
            rms_score = min(1.0, result.rms_spike / self.config.impact_rms_threshold) if rms_impact else 0
            accel_score = min(1.0, result.accel_spike / self.config.impact_accel_threshold) if accel_impact else 0
            result.impact_strength = max(rms_score, accel_score)

        return result


# シングルトン
_collision_detector: Optional[CollisionDetector] = None


def get_collision_detector() -> CollisionDetector:
    global _collision_detector
    if _collision_detector is None:
        _collision_detector = CollisionDetector()
    return _collision_detector
