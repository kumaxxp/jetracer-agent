# FunctionGemma 統合実装プロンプト（Jetson側）

このプロンプトをJetsonのClaude Codeに貼り付けて実行してください。

---

## 概要

FunctionGemma（270Mパラメータ）を「反射神経」として jetracer-agent に統合します。

## 設計原則: **全ての判断はLLMが行う**

```
┌─────────────────────────────────────────────────────────────────┐
│                      処理パイプライン                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  センサー生値                                                    │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SensorPreprocessor                                       │   │
│  │ 役割: 値の変換のみ（判断なし）                          │   │
│  │ 例: 距離200mm → "danger"                                │   │
│  └─────────────────────────────────────────────────────────┘   │
│      │ 抽象状態                                                 │
│      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ FunctionGemma (LLM)                    ★全ての判断はここ │   │
│  │ 役割: センサー状態を見て、どう動くか判断                │   │
│  │ 例: "前方danger + 左clear" → "stop + slight_left"       │   │
│  │ ※ if文なし、プロンプトで動作を記述                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│      │ 意図（メタ言語）                                         │
│      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ VelocityController                                       │   │
│  │ 役割: 純粋なマッピングのみ（判断なし）                  │   │
│  │ 例: "slow" → 0.20, "slight_left" → -0.2                 │   │
│  └─────────────────────────────────────────────────────────┘   │
│      │ PWM値                                                    │
│      ▼                                                          │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ SafetyGuard（最終防壁）                                  │   │
│  │ 役割: ハードウェア保護のみ（LLM出力を上書き可能）       │   │
│  │ 例: 持ち上げ検知時 → throttle=0 に強制                  │   │
│  │ ※ LLMが正しく判断していれば何もしない                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│      │                                                          │
│      ▼                                                          │
│  モーター制御                                                    │
└─────────────────────────────────────────────────────────────────┘
```

**従来のif文の山 → LLMへのプロンプトで動作を記述**

## 作業ディレクトリ

```
~/projects/jetracer-agent/
```

## 実装ファイル

| ファイル | 役割 | 判断ロジック |
|----------|------|-------------|
| `intent_vocabulary.py` | 意図定義 | なし |
| `sensor_preprocessor.py` | 値変換 | なし（閾値判定のみ） |
| `functiongemma_engine.py` | **LLM推論** | **全ての判断** |
| `velocity_controller.py` | 意図→PWM | なし（純粋マッピング） |
| `safety_guard.py` | 最終防壁 | ハードウェア保護のみ |

## 実装順序

1. `intent_vocabulary.py` - 意図定義（共通語彙）
2. `sensor_preprocessor.py` - センサー前処理
3. `velocity_controller.py` - 意図→PWM変換（純粋マッピング）
4. `safety_guard.py` - 最終安全層
5. `functiongemma_engine.py` - FunctionGemma推論エンジン
6. HTTP APIルート追加

---

## Step 1: ディレクトリ作成

```bash
mkdir -p ~/projects/jetracer-agent/http_server/core/reflex
touch ~/projects/jetracer-agent/http_server/core/reflex/__init__.py
```

---

## Step 2: intent_vocabulary.py

`http_server/core/reflex/intent_vocabulary.py` を作成：

```python
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
```

---

## Step 3: sensor_preprocessor.py

`http_server/core/reflex/sensor_preprocessor.py` を作成：

```python
"""センサー前処理モジュール

生のセンサー値を抽象状態に変換してFunctionGemmaに渡す。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class ProximityLevel(str, Enum):
    """近接レベル"""
    CLEAR = "clear"         # 安全（1m以上）
    WARNING = "warning"     # 注意（0.3-1m）
    DANGER = "danger"       # 危険（0.3m未満）
    CONTACT = "contact"     # 接触（0.1m未満）


class MotorState(str, Enum):
    """モーター状態（音響から推定）"""
    STOPPED = "stopped"     # 停止
    IDLE = "idle"           # アイドル
    STARTING = "starting"   # 始動中
    RUNNING = "running"     # 走行中
    SPINNING = "spinning"   # 空転中


class GripStatus(str, Enum):
    """グリップ状態（音響から推定）"""
    GOOD = "good"           # 良好
    SLIPPING = "slipping"   # スリップ中
    LOST = "lost"           # グリップ喪失


class RoadPosition(str, Enum):
    """道路位置"""
    CENTER = "center"       # 中央
    LEFT_EDGE = "left_edge" # 左端
    RIGHT_EDGE = "right_edge"  # 右端
    LOST = "lost"           # 道路見失い


@dataclass
class AbstractSensorState:
    """抽象化されたセンサー状態（FunctionGemmaへの入力）"""
    # LiDAR
    lidar_front: ProximityLevel = ProximityLevel.CLEAR
    lidar_left: ProximityLevel = ProximityLevel.CLEAR
    lidar_right: ProximityLevel = ProximityLevel.CLEAR
    lidar_min_distance_mm: int = 9999
    
    # IMU
    impact_detected: bool = False
    tilt_warning: bool = False
    is_lifted: bool = False
    
    # 音響
    motor_state: MotorState = MotorState.STOPPED
    grip_status: GripStatus = GripStatus.GOOD
    
    # 道路（セグメンテーションから）
    road_visible: bool = True
    road_position: RoadPosition = RoadPosition.CENTER
    road_ratio: float = 0.5  # 走行可能領域の割合
    
    def to_prompt_text(self) -> str:
        """FunctionGemma用のプロンプトテキストに変換"""
        lines = [
            f"lidar: front={self.lidar_front.value}, left={self.lidar_left.value}, right={self.lidar_right.value}, min={self.lidar_min_distance_mm}mm",
            f"imu: impact={self.impact_detected}, tilt_warning={self.tilt_warning}, lifted={self.is_lifted}",
            f"audio: motor={self.motor_state.value}, grip={self.grip_status.value}",
            f"road: visible={self.road_visible}, position={self.road_position.value}, ratio={self.road_ratio:.2f}",
        ]
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lidar": {
                "front": self.lidar_front.value,
                "left": self.lidar_left.value,
                "right": self.lidar_right.value,
                "min_distance_mm": self.lidar_min_distance_mm,
            },
            "imu": {
                "impact_detected": self.impact_detected,
                "tilt_warning": self.tilt_warning,
                "is_lifted": self.is_lifted,
            },
            "audio": {
                "motor_state": self.motor_state.value,
                "grip_status": self.grip_status.value,
            },
            "road": {
                "visible": self.road_visible,
                "position": self.road_position.value,
                "ratio": self.road_ratio,
            }
        }


class SensorPreprocessor:
    """センサー前処理クラス"""
    
    # 距離閾値（mm）
    DISTANCE_CONTACT = 100
    DISTANCE_DANGER = 300
    DISTANCE_WARNING = 1000
    
    # 傾斜閾値（度）
    TILT_WARNING_THRESHOLD = 15.0
    LIFT_THRESHOLD = 30.0
    
    def __init__(self):
        self._last_state = AbstractSensorState()
    
    def process(
        self,
        lidar_data: Optional[Dict] = None,
        imu_data: Optional[Dict] = None,
        acoustic_data: Optional[Dict] = None,
        road_data: Optional[Dict] = None
    ) -> AbstractSensorState:
        """生センサーデータを抽象状態に変換
        
        Args:
            lidar_data: LiDARセンサーデータ（8x8グリッド等）
            imu_data: IMUデータ（加速度、ジャイロ、オイラー角）
            acoustic_data: 音響分析データ
            road_data: セグメンテーション結果
        
        Returns:
            AbstractSensorState
        """
        state = AbstractSensorState()
        
        # LiDAR処理
        if lidar_data:
            state = self._process_lidar(state, lidar_data)
        
        # IMU処理
        if imu_data:
            state = self._process_imu(state, imu_data)
        
        # 音響処理
        if acoustic_data:
            state = self._process_acoustic(state, acoustic_data)
        
        # 道路処理
        if road_data:
            state = self._process_road(state, road_data)
        
        self._last_state = state
        return state
    
    def _process_lidar(self, state: AbstractSensorState, data: Dict) -> AbstractSensorState:
        """LiDARデータを処理"""
        # 8x8グリッドから方向別の最小距離を計算
        grid = data.get("grid") or data.get("distances", [])
        
        if not grid or len(grid) < 8:
            return state
        
        # 前方（中央列、上半分）
        front_distances = []
        for row in range(4):  # 上半分
            for col in range(2, 6):  # 中央4列
                if row < len(grid) and col < len(grid[row]):
                    d = grid[row][col]
                    if 0 < d < 4000:  # 有効範囲
                        front_distances.append(d)
        
        # 左側（左2列）
        left_distances = []
        for row in range(8):
            for col in range(2):
                if row < len(grid) and col < len(grid[row]):
                    d = grid[row][col]
                    if 0 < d < 4000:
                        left_distances.append(d)
        
        # 右側（右2列）
        right_distances = []
        for row in range(8):
            for col in range(6, 8):
                if row < len(grid) and col < len(grid[row]):
                    d = grid[row][col]
                    if 0 < d < 4000:
                        right_distances.append(d)
        
        # 最小距離から近接レベルを判定
        state.lidar_front = self._distance_to_level(min(front_distances) if front_distances else 9999)
        state.lidar_left = self._distance_to_level(min(left_distances) if left_distances else 9999)
        state.lidar_right = self._distance_to_level(min(right_distances) if right_distances else 9999)
        
        all_distances = front_distances + left_distances + right_distances
        state.lidar_min_distance_mm = min(all_distances) if all_distances else 9999
        
        return state
    
    def _distance_to_level(self, distance_mm: int) -> ProximityLevel:
        """距離を近接レベルに変換"""
        if distance_mm < self.DISTANCE_CONTACT:
            return ProximityLevel.CONTACT
        elif distance_mm < self.DISTANCE_DANGER:
            return ProximityLevel.DANGER
        elif distance_mm < self.DISTANCE_WARNING:
            return ProximityLevel.WARNING
        else:
            return ProximityLevel.CLEAR
    
    def _process_imu(self, state: AbstractSensorState, data: Dict) -> AbstractSensorState:
        """IMUデータを処理"""
        # 衝突検知（加速度の急変）
        accel = data.get("acceleration", {})
        accel_magnitude = (
            accel.get("x", 0) ** 2 +
            accel.get("y", 0) ** 2 +
            accel.get("z", 0) ** 2
        ) ** 0.5
        
        # 通常は約9.8m/s²、衝突時は急上昇
        state.impact_detected = accel_magnitude > 15.0
        
        # 傾斜警告
        euler = data.get("euler", {})
        roll = abs(euler.get("roll", 0))
        pitch = abs(euler.get("pitch", 0))
        
        state.tilt_warning = roll > self.TILT_WARNING_THRESHOLD or pitch > self.TILT_WARNING_THRESHOLD
        state.is_lifted = roll > self.LIFT_THRESHOLD or pitch > self.LIFT_THRESHOLD
        
        return state
    
    def _process_acoustic(self, state: AbstractSensorState, data: Dict) -> AbstractSensorState:
        """音響データを処理"""
        # モーター状態
        motor = data.get("motor_state", "stopped")
        try:
            state.motor_state = MotorState(motor)
        except ValueError:
            state.motor_state = MotorState.STOPPED
        
        # グリップ状態
        grip = data.get("grip_status", "good")
        try:
            state.grip_status = GripStatus(grip)
        except ValueError:
            state.grip_status = GripStatus.GOOD
        
        return state
    
    def _process_road(self, state: AbstractSensorState, data: Dict) -> AbstractSensorState:
        """道路データを処理"""
        state.road_visible = data.get("visible", True)
        state.road_ratio = data.get("road_ratio", 0.5)
        
        # 道路位置（セグメンテーションの中心から判定）
        center_x = data.get("road_center_x", 0.5)
        if center_x < 0.35:
            state.road_position = RoadPosition.LEFT_EDGE
        elif center_x > 0.65:
            state.road_position = RoadPosition.RIGHT_EDGE
        else:
            state.road_position = RoadPosition.CENTER
        
        if not state.road_visible or state.road_ratio < 0.1:
            state.road_position = RoadPosition.LOST
        
        return state
    
    @property
    def last_state(self) -> AbstractSensorState:
        return self._last_state
```

---

## Step 4: velocity_controller.py（純粋マッピング版）

`http_server/core/reflex/velocity_controller.py` を作成：

**重要: このモジュールは判断ロジックを含まない純粋なマッピング層です。**
**全ての判断はFunctionGemma（LLM）が行い、ここでは意図→PWM変換のみ行います。**

```python
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
```

---

## Step 5: safety_guard.py（最終安全層）

`http_server/core/reflex/safety_guard.py` を作成：

**これはLLMの出力を上書きできる唯一の層です。**
**ハードウェア保護のための最低限のチェックのみ行います。**

```python
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
```

---

## Step 6: functiongemma_engine.py

`http_server/core/reflex/functiongemma_engine.py` を作成：

```python
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
            
            self.llm = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=-1,  # 全レイヤーGPU
                n_ctx=1024,       # コンテキスト長
                n_batch=512,
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
```

---

## Step 7: HTTPルート追加

`http_server/routes/reflex.py` を作成：

```python
"""FunctionGemma Reflex APIルート

アーキテクチャ:
センサー → Preprocessor → FunctionGemma(LLM) → VelocityController → SafetyGuard → PWM出力

全ての判断はFunctionGemma（LLM）が行う。
SafetyGuardはハードウェア保護のための最終防壁のみ。
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from ..core.reflex.functiongemma_engine import FunctionGemmaEngine
from ..core.reflex.sensor_preprocessor import SensorPreprocessor, AbstractSensorState
from ..core.reflex.velocity_controller import VelocityController
from ..core.reflex.safety_guard import SafetyGuard
from ..core.reflex.intent_vocabulary import ControlIntent

router = APIRouter(prefix="/reflex", tags=["FunctionGemma Reflex"])

# シングルトンインスタンス
_engine: Optional[FunctionGemmaEngine] = None
_preprocessor: Optional[SensorPreprocessor] = None
_controller: Optional[VelocityController] = None
_safety_guard: Optional[SafetyGuard] = None


def get_engine() -> FunctionGemmaEngine:
    global _engine
    if _engine is None:
        _engine = FunctionGemmaEngine()
    return _engine


def get_preprocessor() -> SensorPreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = SensorPreprocessor()
    return _preprocessor


def get_controller() -> VelocityController:
    global _controller
    if _controller is None:
        _controller = VelocityController()
    return _controller


def get_safety_guard() -> SafetyGuard:
    global _safety_guard
    if _safety_guard is None:
        _safety_guard = SafetyGuard()
    return _safety_guard


# --- リクエスト/レスポンスモデル ---

class SensorInput(BaseModel):
    lidar: Optional[Dict] = None
    imu: Optional[Dict] = None
    acoustic: Optional[Dict] = None
    road: Optional[Dict] = None


class InferResponse(BaseModel):
    intent: Dict
    sensor_state: Dict
    inference_time_ms: float


class ControlResponse(BaseModel):
    throttle: float
    steering: float
    intent: Dict
    controller_state: Dict
    safety_override: Optional[Dict] = None


# --- エンドポイント ---

@router.get("/status")
async def get_status():
    """FunctionGemmaステータス取得"""
    engine = get_engine()
    controller = get_controller()
    safety = get_safety_guard()
    
    return {
        "engine": engine.get_status(),
        "controller": controller.get_state(),
        "safety": safety.get_status(),
    }


@router.post("/initialize")
async def initialize():
    """FunctionGemmaエンジン初期化"""
    engine = get_engine()
    success = engine.initialize()
    
    if not success:
        return {
            "status": "fallback",
            "message": "Model not found, using rule-based fallback"
        }
    
    return {"status": "initialized"}


@router.post("/infer", response_model=InferResponse)
async def infer(sensor_input: SensorInput):
    """センサー入力から意図を推論（LLMのみ、PWM変換なし）"""
    engine = get_engine()
    preprocessor = get_preprocessor()
    
    if not engine.is_initialized:
        engine.initialize()
    
    # センサー前処理
    sensor_state = preprocessor.process(
        lidar_data=sensor_input.lidar,
        imu_data=sensor_input.imu,
        acoustic_data=sensor_input.acoustic,
        road_data=sensor_input.road
    )
    
    # LLM推論
    intent = engine.infer(sensor_state)
    
    return InferResponse(
        intent=intent.to_dict(),
        sensor_state=sensor_state.to_dict(),
        inference_time_ms=engine.last_inference_time
    )


@router.post("/control", response_model=ControlResponse)
async def control(sensor_input: SensorInput):
    """センサー入力からPWM出力を計算（フルパイプライン）
    
    パイプライン:
    1. SensorPreprocessor: 生値 → 抽象状態
    2. FunctionGemma: 抽象状態 → 意図（LLMが判断）
    3. VelocityController: 意図 → PWM値（純粋マッピング）
    4. SafetyGuard: 最終安全チェック（ハードウェア保護のみ）
    """
    engine = get_engine()
    preprocessor = get_preprocessor()
    controller = get_controller()
    safety = get_safety_guard()
    
    if not engine.is_initialized:
        engine.initialize()
    
    # Step 1: センサー前処理
    sensor_state = preprocessor.process(
        lidar_data=sensor_input.lidar,
        imu_data=sensor_input.imu,
        acoustic_data=sensor_input.acoustic,
        road_data=sensor_input.road
    )
    
    # Step 2: LLM推論（全ての判断はここで行われる）
    intent = engine.infer(sensor_state)
    
    # Step 3: 意図→PWM変換（純粋マッピング、判断なし）
    throttle, steering = controller.update(intent)
    
    # Step 4: 最終安全チェック（ハードウェア保護のみ）
    safe_throttle, safe_steering, override = safety.check(
        throttle, steering, sensor_state
    )
    
    return ControlResponse(
        throttle=safe_throttle,
        steering=safe_steering,
        intent=intent.to_dict(),
        controller_state=controller.get_state(),
        safety_override={
            "reason": override.reason,
            "throttle_override": override.throttle_override,
            "steering_override": override.steering_override,
        } if override else None
    )


@router.post("/emergency_stop")
async def emergency_stop():
    """緊急停止（SafetyGuard経由ではなく即時）"""
    controller = get_controller()
    controller.reset()
    # throttle=0, steering=現状維持で停止
    return {"status": "emergency_stop", "throttle": 0.0}


@router.post("/reset")
async def reset():
    """状態リセット"""
    controller = get_controller()
    safety = get_safety_guard()
    controller.reset()
    safety.reset_count()
    return {"status": "reset"}


@router.get("/architecture")
async def get_architecture():
    """アーキテクチャ情報（デバッグ用）"""
    return {
        "pipeline": [
            "SensorPreprocessor: Raw → Abstract state",
            "FunctionGemma: Abstract state → Intent (ALL decisions here)",
            "VelocityController: Intent → PWM (pure mapping, NO decisions)",
            "SafetyGuard: Final safety check (hardware protection ONLY)",
        ],
        "design_principle": "All decisions are made by LLM, not by if-statements",
        "safety_guard_role": "Override LLM output ONLY for hardware protection",
    }
```

---

## Step 8: ルーター登録

`http_server/main.py` を編集：

```python
# インポート追加
from .routes.reflex import router as reflex_router

# ルーター登録追加
app.include_router(reflex_router)
```

---

## Step 9: テスト

```bash
# サーバー再起動
cd ~/projects/jetracer-agent
python -m http_server.main

# 別ターミナルでテスト

# ステータス確認
curl http://localhost:8000/reflex/status

# 初期化（モデルがなければフォールバック）
curl -X POST http://localhost:8000/reflex/initialize

# 推論テスト
curl -X POST http://localhost:8000/reflex/infer \
  -H "Content-Type: application/json" \
  -d '{
    "lidar": {"grid": [[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[200,200,200,200,200,200,200,200],[150,150,150,150,150,150,150,150]]},
    "imu": {"acceleration": {"x": 0, "y": 0, "z": 9.8}, "euler": {"roll": 0, "pitch": 0}},
    "road": {"visible": true, "road_ratio": 0.4, "road_center_x": 0.5}
  }'

# 制御出力テスト
curl -X POST http://localhost:8000/reflex/control \
  -H "Content-Type: application/json" \
  -d '{
    "lidar": {"grid": [[200,200,200,200,200,200,200,200],[200,200,200,200,200,200,200,200],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[200,200,200,200,200,200,200,200],[150,150,150,150,150,150,150,150]]}
  }'
```

---

## Step 10: FunctionGemmaモデルのダウンロード（オプション）

モデルがない場合はルールベースフォールバックで動作しますが、
LLM推論を使う場合は以下でダウンロード：

```bash
mkdir -p ~/models
cd ~/models

# Gemma 2B Instruct (Q4_K_M) - 約1.5GB
wget https://huggingface.co/lmstudio-community/gemma-2b-it-GGUF/resolve/main/gemma-2b-it-q4_k_m.gguf \
  -O functiongemma-2b-it-q4_k_m.gguf
```

※ FunctionGemma専用モデルは現時点で公開されていないため、
Gemma 2B Instructをベースに使用し、プロンプトで誘導します。

---

## 期待される結果

### ステータス
```json
{
  "engine": {
    "initialized": false,
    "model_exists": false
  },
  "controller": {
    "target_throttle": 0.0,
    "current_throttle": 0.0
  }
}
```

### 推論（フォールバック）
```json
{
  "intent": {
    "speed": "slow",
    "steering": "straight",
    "reason": ""
  },
  "sensor_state": {
    "lidar": {"front": "warning", "left": "clear", "right": "clear"}
  },
  "inference_time_ms": 0.1
}
```

---

## 完了後

Jetson側のAPIが動作したら、次のステップ（PC側UIパネル）に進みます。
