"""安全ガード（最終防壁）

LLMの判断を尊重しつつ、ハードウェア保護のための最低限のチェックを行う。
この層だけがLLM出力を上書きできる。

設計原則:
- LLMが正しく判断していれば、この層は何もしない
- LLMが見逃した致命的な状況のみ介入
- 介入時はログを残し、LLMの改善に活用
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import time

from .sensor_preprocessor import AbstractSensorState, ProximityLevel


@dataclass
class SafetyOverride:
    """安全層による上書き情報"""
    throttle_override: Optional[float] = None
    steering_override: Optional[float] = None
    reason: str = ""
    timestamp: float = 0.0


class SafetyGuard:
    """最終安全層

    LLMの出力を検証し、致命的な状況でのみ介入する。

    介入条件（ハードコード - これらはLLMに委ねない）:
    1. 持ち上げられた状態でのモーター駆動禁止
    2. 接触状態での前進禁止
    3. PWM値の物理的上限
    """

    # 物理的制限値
    MAX_THROTTLE = 0.6          # スロットル上限
    MAX_STEERING = 0.9          # ステアリング上限

    # 介入閾値
    CONTACT_DISTANCE_MM = 80    # これ以下は物理的接触とみなす

    def __init__(self):
        self.last_override: Optional[SafetyOverride] = None
        self.override_count = 0

    def check(
        self,
        throttle: float,
        steering: float,
        sensor_state: AbstractSensorState
    ) -> Tuple[float, float, Optional[SafetyOverride]]:
        """安全チェック（最終防壁）

        Args:
            throttle: LLM→VelocityControllerからのスロットル値
            steering: LLM→VelocityControllerからのステアリング値
            sensor_state: 現在のセンサー状態

        Returns:
            (safe_throttle, safe_steering, override_info)
            override_infoがNoneでなければ介入が発生
        """
        override = None
        safe_throttle = throttle
        safe_steering = steering

        # --- 致命的状況のチェック ---

        # 1. 持ち上げられた状態
        if sensor_state.is_lifted and throttle > 0:
            safe_throttle = 0.0
            override = SafetyOverride(
                throttle_override=0.0,
                reason="LIFTED: Motor disabled for safety",
                timestamp=time.time()
            )

        # 2. 物理的接触状態での前進
        if (sensor_state.lidar_front == ProximityLevel.CONTACT and
            throttle > 0):
            safe_throttle = 0.0
            override = SafetyOverride(
                throttle_override=0.0,
                reason="CONTACT: Forward motion blocked",
                timestamp=time.time()
            )

        # 3. 物理的上限（ハードウェア保護）
        if throttle > self.MAX_THROTTLE:
            safe_throttle = self.MAX_THROTTLE
            override = SafetyOverride(
                throttle_override=self.MAX_THROTTLE,
                reason=f"LIMIT: Throttle capped at {self.MAX_THROTTLE}",
                timestamp=time.time()
            )

        if abs(steering) > self.MAX_STEERING:
            safe_steering = self.MAX_STEERING if steering > 0 else -self.MAX_STEERING
            if override is None:
                override = SafetyOverride(timestamp=time.time())
            override.steering_override = safe_steering
            override.reason += f" LIMIT: Steering capped at {self.MAX_STEERING}"

        # 介入記録
        if override:
            self.last_override = override
            self.override_count += 1
            print(f"[SafetyGuard] OVERRIDE #{self.override_count}: {override.reason}")

        return (safe_throttle, safe_steering, override)

    def get_status(self) -> dict:
        """ステータス取得"""
        return {
            "override_count": self.override_count,
            "last_override": {
                "reason": self.last_override.reason,
                "timestamp": self.last_override.timestamp
            } if self.last_override else None
        }

    def reset_count(self) -> None:
        """介入カウントリセット"""
        self.override_count = 0
