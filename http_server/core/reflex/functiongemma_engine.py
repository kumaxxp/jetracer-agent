"""FunctionGemma推論エンジン（HTTP クライアント版）

llama-server（別プロセス）にHTTP経由で推論リクエストを送信。
PyTorchとのCUDA競合を完全に回避。
"""

import json
import time
import re
from typing import Optional, Dict, Any
import httpx

from .intent_vocabulary import (
    ControlIntent, SpeedIntent, SpeedChangeIntent,
    SteeringIntent, DirectionIntent, UrgencyLevel
)
from .sensor_preprocessor import AbstractSensorState


# llama-server 設定
LLAMA_SERVER_URL = "http://127.0.0.1:8081"
COMPLETION_ENDPOINT = f"{LLAMA_SERVER_URL}/completion"
HEALTH_ENDPOINT = f"{LLAMA_SERVER_URL}/health"


# システムプロンプト（Few-shot例付き）
SYSTEM_PROMPT = """JetRacer制御AI。センサー状態からJSON制御指示を出力。

## 語彙
speed: stop/creep/slow/normal/fast
speed_change: emergency_stop/decelerate/halve/accelerate/maintain
steering: straight/slight_left/left/hard_left/slight_right/right/hard_right
urgency: low/normal/high/critical

## センサー値
lidar: clear(安全)/warning(注意)/danger(危険)/contact(接触)
imu.lifted: 持ち上げられている
imu.impact: 衝突
grip: good/slipping/lost
road.position: center/left_edge/right_edge/lost

## 例

入力: lidar.front=clear, lidar.left=clear, lidar.right=clear
出力: {"speed": "normal", "steering": "straight", "reason": "all_clear"}

入力: lidar.front=danger
出力: {"speed_change": "emergency_stop", "urgency": "critical", "reason": "front_danger"}

入力: lidar.front=contact
出力: {"speed_change": "emergency_stop", "urgency": "critical", "reason": "front_contact"}

入力: imu.lifted=true
出力: {"speed_change": "emergency_stop", "escalate": true, "reason": "lifted"}

入力: imu.impact=true
出力: {"speed_change": "emergency_stop", "urgency": "critical", "reason": "impact"}

入力: grip=lost
出力: {"speed_change": "halve", "reason": "grip_lost"}

入力: road.position=left_edge
出力: {"steering": "slight_right", "reason": "road_left_edge"}

入力: road.position=right_edge
出力: {"steering": "slight_left", "reason": "road_right_edge"}

入力: lidar.front=warning, lidar.left=clear
出力: {"speed_change": "decelerate", "steering": "slight_left", "reason": "front_warning"}

JSONのみ出力。説明不要。"""


class FunctionGemmaEngine:
    """FunctionGemma推論エンジン（HTTPクライアント版）

    llama-server（別プロセス）にHTTP経由でリクエストを送信。
    """

    def __init__(self, server_url: str = LLAMA_SERVER_URL):
        self.server_url = server_url
        self.completion_endpoint = f"{server_url}/completion"
        self.health_endpoint = f"{server_url}/health"
        self._initialized = False
        self._server_available = False
        self._last_inference_time = 0.0
        self._http_client: Optional[httpx.Client] = None

    def initialize(self) -> bool:
        """サーバー接続確認"""
        if self._initialized:
            return True

        try:
            self._http_client = httpx.Client(timeout=30.0)

            # ヘルスチェック
            response = self._http_client.get(self.health_endpoint)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    self._server_available = True
                    self._initialized = True
                    print(f"[FunctionGemma] Connected to llama-server at {self.server_url}")
                    return True

            print(f"[FunctionGemma] Server not ready: {response.text}")
            return False

        except httpx.ConnectError:
            print(f"[FunctionGemma] Cannot connect to {self.server_url}")
            print("[FunctionGemma] Please start llama-server first:")
            print("  llama-server --model ~/models/functiongemma-2b-it-q4_k_m.gguf --port 8081 --n-gpu-layers 99")
            return False
        except Exception as e:
            print(f"[FunctionGemma] Initialization failed: {e}")
            return False

    def infer(self, sensor_state: AbstractSensorState) -> ControlIntent:
        """センサー状態から制御意図を推論

        Args:
            sensor_state: 抽象化されたセンサー状態

        Returns:
            ControlIntent
        """
        start = time.time()

        if self._server_available and self._http_client:
            intent = self._infer_http(sensor_state)
        else:
            # フォールバック: ルールベース
            intent = self._infer_rule_based(sensor_state)

        self._last_inference_time = (time.time() - start) * 1000
        return intent

    def _infer_http(self, sensor_state: AbstractSensorState) -> ControlIntent:
        """HTTP経由でllama-serverに推論リクエスト"""
        # センサー状態を簡潔な形式に変換
        sensor_summary = sensor_state.to_compact_text()

        # Gemma 2 チャットフォーマット（Few-shot例と同形式）
        user_content = f"""{SYSTEM_PROMPT}

入力: {sensor_summary}
出力:"""

        prompt = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

        try:
            response = self._http_client.post(
                self.completion_endpoint,
                json={
                    "prompt": prompt,
                    "n_predict": 128,
                    "temperature": 0.1,
                    "top_k": 40,
                    "top_p": 0.9,
                    "stop": ["<end_of_turn>", "\n\n"],
                }
            )

            if response.status_code == 200:
                data = response.json()
                content = data.get("content", "")
                intent = self._parse_response(content)

                # 安全性検証：危険な状態でLLMが適切に応答しなかった場合はルールベースを使用
                if self._needs_safety_override(sensor_state, intent):
                    print(f"[FunctionGemma] Safety override: LLM output invalid for critical state")
                    return self._infer_rule_based(sensor_state)

                return intent
            else:
                print(f"[FunctionGemma] HTTP error: {response.status_code}")
                return self._infer_rule_based(sensor_state)

        except Exception as e:
            print(f"[FunctionGemma] Inference error: {e}")
            return self._infer_rule_based(sensor_state)

    def _needs_safety_override(self, state: AbstractSensorState, intent: ControlIntent) -> bool:
        """LLM出力が安全要件を満たしているかチェック"""
        from .sensor_preprocessor import ProximityLevel, GripStatus, RoadPosition

        # === Critical Safety (必須) ===

        # 前方danger/contactの場合、emergency_stopが必須
        if state.lidar_front in (ProximityLevel.DANGER, ProximityLevel.CONTACT):
            if intent.speed_change != SpeedChangeIntent.EMERGENCY_STOP:
                return True

        # 持ち上げられた場合、emergency_stop + escalateが必須
        if state.is_lifted:
            if intent.speed_change != SpeedChangeIntent.EMERGENCY_STOP or not intent.escalate:
                return True

        # 衝突検知の場合、emergency_stopが必須
        if state.impact_detected:
            if intent.speed_change != SpeedChangeIntent.EMERGENCY_STOP:
                return True

        # === Important Safety (推奨) ===

        # 前方warningの場合、減速が必要
        if state.lidar_front == ProximityLevel.WARNING:
            if intent.speed_change not in (SpeedChangeIntent.DECELERATE, SpeedChangeIntent.HALVE, SpeedChangeIntent.EMERGENCY_STOP):
                return True

        # グリップ喪失時はhalveが必要
        if state.grip_status == GripStatus.LOST:
            if intent.speed_change not in (SpeedChangeIntent.HALVE, SpeedChangeIntent.EMERGENCY_STOP):
                return True

        # 道路端ではステアリング修正が必要
        if state.road_position == RoadPosition.LEFT_EDGE:
            if intent.steering not in (SteeringIntent.SLIGHT_RIGHT, SteeringIntent.RIGHT, SteeringIntent.HARD_RIGHT):
                return True

        if state.road_position == RoadPosition.RIGHT_EDGE:
            if intent.steering not in (SteeringIntent.SLIGHT_LEFT, SteeringIntent.LEFT, SteeringIntent.HARD_LEFT):
                return True

        return False

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
                    intent.urgency = UrgencyLevel(data.get("urgency", "normal"))
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
            "server_url": self.server_url,
            "server_available": self._server_available,
            "last_inference_ms": round(self._last_inference_time, 2),
            "mode": "http" if self._server_available else "rule_based_fallback",
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def last_inference_time(self) -> float:
        return self._last_inference_time

    def close(self):
        """クリーンアップ"""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
