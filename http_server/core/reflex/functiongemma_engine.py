"""FunctionGemma推論エンジン

270Mパラメータの軽量LLMでセンサー状態から制御意図を生成。
"""

import json
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any

from .intent_vocabulary import (
    ControlIntent, SpeedIntent, SpeedChangeIntent,
    SteeringIntent, DirectionIntent, UrgencyLevel
)
from .sensor_preprocessor import AbstractSensorState


# FunctionGemmaモデルパス
MODEL_PATH = Path.home() / "models" / "functiongemma-2b-it-q4_k_m.gguf"


# システムプロンプト
SYSTEM_PROMPT = """あなたはJetRacer自律走行車の反射神経AIです。
センサー入力を受け取り、意図レベルの制御指示をJSON形式で出力してください。

## 重要な制約
- 数値での速度・角度指定は禁止です
- 必ず意図語彙（メタ言語）で出力してください
- JSONのみを出力し、説明文は不要です

## 意図語彙

【speed】stop, creep, slow, normal, fast
【speed_change】decelerate, accelerate, halve, emergency_stop, maintain
【steering】straight, slight_left, left, hard_left, slight_right, right, hard_right, more_left, more_right, center
【direction】forward, reverse, hold
【urgency】low, normal, high, critical

## センサー状態の読み方

【lidar】front/left/right の値
- clear: 安全（1m以上）
- warning: 注意（0.3-1m）
- danger: 危険（0.3m未満）
- contact: 接触

【imu】
- impact: true なら衝突発生
- lifted: true なら持ち上げられている

【audio】
- motor: stopped/running/spinning
- grip: good/slipping/lost

【road】
- visible: 道路が見えているか
- position: center/left_edge/right_edge/lost

## 出力形式（JSON）

```json
{
  "speed": "slow",
  "steering": "straight",
  "reason": "前方クリア、道路中央"
}
```

## 動作シナリオ

1. 前方danger/contact → emergency_stop
2. grip=lost → halve して様子見
3. road.position=left_edge → slight_right
4. 前方warning → decelerate + 空いている方向へ
5. lifted=true → stop してエスカレート
"""


class FunctionGemmaEngine:
    """FunctionGemma推論エンジン

    llama-cpp-python を使用してローカル推論を実行。
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = Path(model_path) if model_path else MODEL_PATH
        self.llm = None
        self._initialized = False
        self._last_inference_time = 0.0

    def initialize(self) -> bool:
        """モデル初期化"""
        if self._initialized:
            return True

        if not self.model_path.exists():
            print(f"[FunctionGemma] Model not found: {self.model_path}")
            print("[FunctionGemma] Running in fallback mode (rule-based)")
            return False

        try:
            from llama_cpp import Llama

            print(f"[FunctionGemma] Loading model: {self.model_path.name}")
            start = time.time()

            # CPU専用モード（PyTorchとのCUDA競合を回避）
            # 270Mモデルは軽量なのでCPUでも400-500msで推論可能
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=0,   # CPU専用
                n_ctx=2048,       # コンテキスト長
                n_batch=256,
                verbose=False
            )

            load_time = time.time() - start
            print(f"[FunctionGemma] Model loaded ({load_time:.1f}s)")

            # ウォームアップ
            self._warmup()

            self._initialized = True
            return True

        except ImportError:
            print("[FunctionGemma] llama-cpp-python not installed")
            return False
        except Exception as e:
            print(f"[FunctionGemma] Initialization failed: {e}")
            return False

    def _warmup(self, iterations: int = 2):
        """ウォームアップ"""
        print(f"[FunctionGemma] Warming up...")
        dummy_state = AbstractSensorState()
        for _ in range(iterations):
            self._infer_llm(dummy_state)
        print("[FunctionGemma] Warmup complete")

    def infer(self, sensor_state: AbstractSensorState) -> ControlIntent:
        """センサー状態から制御意図を推論

        Args:
            sensor_state: 抽象化されたセンサー状態

        Returns:
            ControlIntent
        """
        start = time.time()

        # LLMが利用可能ならLLM推論
        if self._initialized and self.llm:
            intent = self._infer_llm(sensor_state)
        else:
            # フォールバック: ルールベース
            intent = self._infer_rule_based(sensor_state)

        self._last_inference_time = (time.time() - start) * 1000
        return intent

    def _infer_llm(self, sensor_state: AbstractSensorState) -> ControlIntent:
        """LLMで推論"""
        prompt = f"""センサー状態:
{sensor_state.to_prompt_text()}

上記の状態に対する制御意図をJSONで出力してください。"""

        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=128,
                temperature=0.1,  # 低温で決定的に
            )

            content = response["choices"][0]["message"]["content"]
            return self._parse_response(content)

        except Exception as e:
            print(f"[FunctionGemma] Inference error: {e}")
            return self._infer_rule_based(sensor_state)

    def _parse_response(self, content: str) -> ControlIntent:
        """LLM出力をパース"""
        try:
            # JSON部分を抽出
            json_match = re.search(r'\{[^{}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)

            intent = ControlIntent()

            if "speed" in data:
                try:
                    intent.speed = SpeedIntent(data["speed"])
                except ValueError:
                    pass

            if "speed_change" in data:
                try:
                    intent.speed_change = SpeedChangeIntent(data["speed_change"])
                except ValueError:
                    pass

            if "steering" in data:
                try:
                    intent.steering = SteeringIntent(data["steering"])
                except ValueError:
                    pass

            if "direction" in data:
                try:
                    intent.direction = DirectionIntent(data["direction"])
                except ValueError:
                    pass

            if "urgency" in data:
                try:
                    intent.urgency = UrgencyLevel(data["urgency"])
                except ValueError:
                    pass

            intent.reason = data.get("reason", "")
            intent.escalate = data.get("escalate", False)

            return intent

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[FunctionGemma] Parse error: {e}, content: {content[:100]}")
            return ControlIntent(reason="parse_error")

    def _infer_rule_based(self, state: AbstractSensorState) -> ControlIntent:
        """ルールベースのフォールバック推論"""
        intent = ControlIntent()
        intent.speed = SpeedIntent.SLOW
        intent.steering = SteeringIntent.STRAIGHT
        intent.direction = DirectionIntent.FORWARD

        # 緊急停止条件
        if state.is_lifted:
            intent.speed_change = SpeedChangeIntent.EMERGENCY_STOP
            intent.reason = "lifted"
            intent.escalate = True
            return intent

        if state.impact_detected:
            intent.speed_change = SpeedChangeIntent.EMERGENCY_STOP
            intent.reason = "impact"
            intent.urgency = UrgencyLevel.CRITICAL
            return intent

        from .sensor_preprocessor import ProximityLevel, GripStatus, RoadPosition

        # 前方障害物
        if state.lidar_front == ProximityLevel.CONTACT:
            intent.speed_change = SpeedChangeIntent.EMERGENCY_STOP
            intent.reason = "contact_front"
            return intent

        if state.lidar_front == ProximityLevel.DANGER:
            intent.speed_change = SpeedChangeIntent.EMERGENCY_STOP
            intent.reason = "danger_front"
            return intent

        if state.lidar_front == ProximityLevel.WARNING:
            intent.speed_change = SpeedChangeIntent.DECELERATE
            # 空いている方向へ
            if state.lidar_left == ProximityLevel.CLEAR:
                intent.steering = SteeringIntent.SLIGHT_LEFT
            elif state.lidar_right == ProximityLevel.CLEAR:
                intent.steering = SteeringIntent.SLIGHT_RIGHT
            intent.reason = "warning_front"

        # グリップ喪失
        if state.grip_status == GripStatus.LOST:
            intent.speed_change = SpeedChangeIntent.HALVE
            intent.reason = "grip_lost"
            return intent

        if state.grip_status == GripStatus.SLIPPING:
            intent.speed_change = SpeedChangeIntent.DECELERATE
            intent.reason = "slipping"

        # 道路追従
        if state.road_position == RoadPosition.LEFT_EDGE:
            intent.steering = SteeringIntent.SLIGHT_RIGHT
            intent.reason = "road_left_edge"
        elif state.road_position == RoadPosition.RIGHT_EDGE:
            intent.steering = SteeringIntent.SLIGHT_LEFT
            intent.reason = "road_right_edge"
        elif state.road_position == RoadPosition.LOST:
            intent.speed_change = SpeedChangeIntent.HALVE
            intent.reason = "road_lost"
            intent.escalate = True

        return intent

    def get_status(self) -> Dict[str, Any]:
        """ステータス取得"""
        return {
            "initialized": self._initialized,
            "model_path": str(self.model_path),
            "model_exists": self.model_path.exists(),
            "last_inference_ms": self._last_inference_time,
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def last_inference_time(self) -> float:
        return self._last_inference_time
