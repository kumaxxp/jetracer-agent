"""意図語彙（メタ言語）定義

FunctionGemmaの出力語彙を定義。
数値ではなく意図レベルで制御を記述する。
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class SpeedIntent(str, Enum):
    """速度意図"""
    STOP = "stop"           # 完全停止
    CREEP = "creep"         # 極低速（動き出し確認用）
    SLOW = "slow"           # 低速（安全なデータ収集速度）
    NORMAL = "normal"       # 通常速度
    FAST = "fast"           # 高速


class SpeedChangeIntent(str, Enum):
    """速度変更意図"""
    DECELERATE = "decelerate"       # 減速
    ACCELERATE = "accelerate"       # 加速
    HALVE = "halve"                 # 半分に減速
    EMERGENCY_STOP = "emergency_stop"  # 緊急停止
    MAINTAIN = "maintain"           # 維持


class SteeringIntent(str, Enum):
    """ステアリング意図"""
    STRAIGHT = "straight"       # 直進
    SLIGHT_LEFT = "slight_left"   # 微左
    LEFT = "left"               # 左
    HARD_LEFT = "hard_left"     # 急左
    SLIGHT_RIGHT = "slight_right" # 微右
    RIGHT = "right"             # 右
    HARD_RIGHT = "hard_right"   # 急右
    MORE_LEFT = "more_left"     # さらに左へ
    MORE_RIGHT = "more_right"   # さらに右へ
    CENTER = "center"           # センターへ戻す


class DirectionIntent(str, Enum):
    """方向意図"""
    FORWARD = "forward"   # 前進
    REVERSE = "reverse"   # 後退
    HOLD = "hold"         # 現状維持


class UrgencyLevel(str, Enum):
    """緊急度"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ControlIntent:
    """制御意図（FunctionGemmaの出力）"""
    speed: Optional[SpeedIntent] = None
    speed_change: Optional[SpeedChangeIntent] = None
    steering: Optional[SteeringIntent] = None
    direction: Optional[DirectionIntent] = None
    urgency: UrgencyLevel = UrgencyLevel.NORMAL
    reason: str = ""
    escalate: bool = False  # 脳（汎用LLM）にエスカレートするか

    def to_dict(self) -> dict:
        return {
            "speed": self.speed.value if self.speed else None,
            "speed_change": self.speed_change.value if self.speed_change else None,
            "steering": self.steering.value if self.steering else None,
            "direction": self.direction.value if self.direction else None,
            "urgency": self.urgency.value,
            "reason": self.reason,
            "escalate": self.escalate
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ControlIntent':
        return cls(
            speed=SpeedIntent(data["speed"]) if data.get("speed") else None,
            speed_change=SpeedChangeIntent(data["speed_change"]) if data.get("speed_change") else None,
            steering=SteeringIntent(data["steering"]) if data.get("steering") else None,
            direction=DirectionIntent(data["direction"]) if data.get("direction") else None,
            urgency=UrgencyLevel(data.get("urgency", "normal")),
            reason=data.get("reason", ""),
            escalate=data.get("escalate", False)
        )


# PWM変換マップ（velocity_controllerで使用）
SPEED_TO_THROTTLE = {
    SpeedIntent.STOP: 0.0,
    SpeedIntent.CREEP: 0.10,
    SpeedIntent.SLOW: 0.20,
    SpeedIntent.NORMAL: 0.35,
    SpeedIntent.FAST: 0.50,
}

STEERING_TO_VALUE = {
    SteeringIntent.HARD_LEFT: -0.8,
    SteeringIntent.LEFT: -0.5,
    SteeringIntent.SLIGHT_LEFT: -0.2,
    SteeringIntent.STRAIGHT: 0.0,
    SteeringIntent.CENTER: 0.0,
    SteeringIntent.SLIGHT_RIGHT: 0.2,
    SteeringIntent.RIGHT: 0.5,
    SteeringIntent.HARD_RIGHT: 0.8,
}

# 相対ステアリング調整量
STEERING_ADJUSTMENT = {
    SteeringIntent.MORE_LEFT: -0.15,
    SteeringIntent.MORE_RIGHT: 0.15,
}
