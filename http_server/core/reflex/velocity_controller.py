"""意図→PWM変換コントローラ（純粋マッピング版）

FunctionGemmaの意図出力を実際のPWM値に変換する。
判断ロジックは含まない - 全ての判断はLLMに委ねる。
"""

import time
from dataclasses import dataclass
from typing import Tuple

from .intent_vocabulary import (
    ControlIntent, SpeedIntent, SpeedChangeIntent, SteeringIntent,
    SPEED_TO_THROTTLE, STEERING_TO_VALUE, STEERING_ADJUSTMENT
)


@dataclass
class VelocityState:
    """速度コントローラの内部状態"""
    target_throttle: float = 0.0      # 目標スロットル [0, 1]
    current_throttle: float = 0.0     # 現在のスロットル
    target_steering: float = 0.0      # 目標ステアリング [-1, 1]
    current_steering: float = 0.0     # 現在のステアリング
    last_update: float = 0.0


class VelocityController:
    """意図→PWM変換コントローラ（純粋マッピング）

    このクラスは判断を行わない。
    - 意図（メタ言語）をPWM値に変換するだけ
    - 急変防止のランプ処理のみ実施
    - 判断はすべてFunctionGemma（LLM）が担当
    """

    # ランプパラメータ（急変防止のみ - 判断ではない）
    THROTTLE_RAMP_UP = 0.05      # 加速時の1ステップあたり変化量
    THROTTLE_RAMP_DOWN = 0.10   # 減速時の1ステップあたり変化量（速め）
    STEERING_RAMP = 0.15         # ステアリングの1ステップあたり変化量

    def __init__(self):
        self.state = VelocityState()

    def update(self, intent: ControlIntent) -> Tuple[float, float]:
        """意図を受け取りPWM値を更新（純粋マッピング）

        Args:
            intent: FunctionGemmaからの制御意図

        Returns:
            (throttle, steering) 正規化値
            throttle: [0, 1], steering: [-1, 1]
        """
        self.state.last_update = time.time()

        # === スロットル: 意図→値の純粋マッピング ===

        # 速度意図があれば目標を設定
        if intent.speed is not None:
            self.state.target_throttle = SPEED_TO_THROTTLE.get(intent.speed, 0.0)

        # 速度変更意図があれば目標を調整
        if intent.speed_change is not None:
            self._apply_speed_change(intent.speed_change)

        # ランプ処理（急変防止のための物理的制約、判断ではない）
        self.state.current_throttle = self._ramp(
            current=self.state.current_throttle,
            target=self.state.target_throttle,
            ramp_up=self.THROTTLE_RAMP_UP,
            ramp_down=self.THROTTLE_RAMP_DOWN
        )

        # === ステアリング: 意図→値の純粋マッピング ===

        if intent.steering is not None:
            if intent.steering in STEERING_TO_VALUE:
                # 絶対値マッピング
                self.state.target_steering = STEERING_TO_VALUE[intent.steering]
            elif intent.steering in STEERING_ADJUSTMENT:
                # 相対調整
                self.state.target_steering += STEERING_ADJUSTMENT[intent.steering]
                self.state.target_steering = max(-1.0, min(1.0, self.state.target_steering))

        # ランプ処理
        self.state.current_steering = self._ramp(
            current=self.state.current_steering,
            target=self.state.target_steering,
            ramp_up=self.STEERING_RAMP,
            ramp_down=self.STEERING_RAMP
        )

        return (self.state.current_throttle, self.state.current_steering)

    def _apply_speed_change(self, change: SpeedChangeIntent) -> None:
        """速度変更意図を目標値に適用（純粋マッピング）"""
        if change == SpeedChangeIntent.EMERGENCY_STOP:
            self.state.target_throttle = 0.0
        elif change == SpeedChangeIntent.HALVE:
            self.state.target_throttle *= 0.5
        elif change == SpeedChangeIntent.DECELERATE:
            self.state.target_throttle *= 0.7
        elif change == SpeedChangeIntent.ACCELERATE:
            self.state.target_throttle = min(self.state.target_throttle * 1.3, 0.5)
        # MAINTAIN: 何もしない

    def _ramp(
        self,
        current: float,
        target: float,
        ramp_up: float,
        ramp_down: float
    ) -> float:
        """ランプ処理（急変防止）"""
        if current < target:
            return min(current + ramp_up, target)
        elif current > target:
            return max(current - ramp_down, target)
        return current

    def reset(self) -> None:
        """状態リセット"""
        self.state = VelocityState()

    def get_state(self) -> dict:
        """現在の状態を取得"""
        return {
            "target_throttle": round(self.state.target_throttle, 3),
            "current_throttle": round(self.state.current_throttle, 3),
            "target_steering": round(self.state.target_steering, 3),
            "current_steering": round(self.state.current_steering, 3),
        }
