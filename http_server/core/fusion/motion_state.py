"""総合モーション状態推定

状態の優先順位:
1. collision → 緊急停止
2. lifted → 最優先（安全）
3. stuck → 復帰動作
4. imu_drift → 補正
5. running / stopped → 通常状態
"""
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from .stuck_detector import get_stuck_detector, StuckResult
from .lift_detector import get_lift_detector, LiftResult
from .imu_drift_detector import get_drift_detector, DriftResult
from .collision_detector import get_collision_detector, CollisionResult
from .data_source import get_data_source
from .config import fusion_config


class MotionState(str, Enum):
    """モーション状態"""
    UNKNOWN = "unknown"
    STOPPED = "stopped"
    RUNNING = "running"
    STUCK = "stuck"
    LIFTED = "lifted"
    IMU_DRIFT = "imu_drift"
    COLLISION = "collision"


@dataclass
class MotionStateResult:
    """総合状態推定結果"""
    state: str = MotionState.UNKNOWN.value
    confidence: float = 0.0
    recommended_action: str = "none"
    details: dict = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "confidence": round(self.confidence, 2),
            "recommended_action": self.recommended_action,
            "details": self.details,
            "timestamp": self.timestamp
        }


class MotionStateEstimator:
    """総合モーション状態推定器"""

    # 推奨アクション
    ACTIONS = {
        MotionState.UNKNOWN: "wait",
        MotionState.STOPPED: "none",
        MotionState.RUNNING: "none",
        MotionState.STUCK: "reverse_and_retry",
        MotionState.LIFTED: "stop_motor",
        MotionState.IMU_DRIFT: "reset_imu",
        MotionState.COLLISION: "emergency_stop",
    }

    def __init__(self):
        self.config = fusion_config
        self.stuck_detector = get_stuck_detector()
        self.lift_detector = get_lift_detector()
        self.drift_detector = get_drift_detector()
        self.collision_detector = get_collision_detector()

    def estimate(self) -> MotionStateResult:
        """総合状態を推定"""
        result = MotionStateResult(timestamp=time.time())

        # 各検出器の結果を取得
        stuck_result = self.stuck_detector.check()
        lift_result = self.lift_detector.check()
        drift_result = self.drift_detector.check()
        collision_result = self.collision_detector.check()

        # 詳細を保存
        result.details = {
            "stuck": stuck_result.to_dict(),
            "lifted": lift_result.to_dict(),
            "imu_drift": drift_result.to_dict(),
            "collision": collision_result.to_dict()
        }

        # 優先順位に従って状態を決定
        # 1. 衝突（最優先）
        if collision_result.collision:
            result.state = MotionState.COLLISION.value
            result.confidence = collision_result.impact_strength
            result.recommended_action = self.ACTIONS[MotionState.COLLISION]
            return result

        # 2. 持ち上げ
        if lift_result.lifted:
            result.state = MotionState.LIFTED.value
            result.confidence = lift_result.confidence
            result.recommended_action = self.ACTIONS[MotionState.LIFTED]
            return result

        # 3. スタック
        if stuck_result.stuck:
            result.state = MotionState.STUCK.value
            result.confidence = stuck_result.confidence
            result.recommended_action = self.ACTIONS[MotionState.STUCK]
            return result

        # 4. IMUドリフト
        if drift_result.drifting:
            result.state = MotionState.IMU_DRIFT.value
            result.confidence = drift_result.confidence
            result.recommended_action = self.ACTIONS[MotionState.IMU_DRIFT]
            return result

        # 5. 通常状態（走行中 or 停止中）
        data_source = get_data_source()
        snapshot = data_source.get_snapshot()

        if snapshot.acoustic_valid:
            if snapshot.zcr > self.config.motor_running_zcr:
                result.state = MotionState.RUNNING.value
                result.confidence = 0.8
            else:
                result.state = MotionState.STOPPED.value
                result.confidence = 0.8
        else:
            result.state = MotionState.UNKNOWN.value
            result.confidence = 0.0

        result.recommended_action = self.ACTIONS.get(
            MotionState(result.state), "none"
        )

        return result


# シングルトン
_motion_estimator: Optional[MotionStateEstimator] = None


def get_motion_estimator() -> MotionStateEstimator:
    global _motion_estimator
    if _motion_estimator is None:
        _motion_estimator = MotionStateEstimator()
    return _motion_estimator
