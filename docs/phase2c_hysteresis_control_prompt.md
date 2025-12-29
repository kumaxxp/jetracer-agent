# YANA Phase 2-C: 音響フィードバック・ヒステリシス制御実装

**実行環境**: Jetson Orin Nano Super
**リポジトリ**: `~/jetracer-agent`
**前提**: Phase 2-A, 2-B 完了済み

---

## 目的

Phase 2-Bで特定した始動PWM閾値を使用し、音響フィードバックによるヒステリシス制御を実装する。
これにより、低速走行時の安定した発進・維持制御を実現する。

---

## Phase 2-B 分析結果（パラメータ）

```python
# 分析で得られた値
STARTUP_PWM = 0.147      # 起動用（静止摩擦を確実に超える）
MAINTAIN_PWM = 0.067     # 維持用（動摩擦領域）
THRESHOLD_PWM = 0.117    # 始動検出閾値

# 音響特徴量の変化（閾値前後）
# RMS: +37.0%
# ZCR: +93.8% ← 最も信頼性が高い
```

---

## 制御概念

### ヒステリシス制御フロー

```
状態遷移:
  STOPPED → STARTING → RUNNING → STOPPED
     │         │          │
     │         │          └─ motor_state == STOPPED 検出
     │         └─ 音響で始動検出（ZCR急増）
     └─ start() 呼び出し

PWM制御:
  STOPPED:  PWM = 0.0
  STARTING: PWM = startup_pwm (0.147) で蹴り出し
  RUNNING:  PWM = maintain_pwm (0.067) で維持
```

### タイミング図

```
PWM出力
    ^
0.15│    ┌───┐ startup_pwm
    │    │   │
0.10│····│···│·················· threshold
    │    │   │
0.07│    │   └────────────────── maintain_pwm
    │    │
0.00├────┘
    +────┬───┬──────────────────> time
         │   │
         │   └─ 音響で始動検出
         └─ start() 呼び出し
```

---

## 実装タスク

### 1. AcousticThrottleController クラス

**ファイル**: `http_server/core/acoustic_throttle_controller.py`

```python
"""
音響フィードバック・ヒステリシス制御コントローラー

低速走行時の発進・維持制御を音響状態推定で実現
"""

import time
import asyncio
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Callable
import logging

from .acoustic import AcousticManager
from .acoustic_inference import get_classifier

logger = logging.getLogger(__name__)


class ControlState(Enum):
    """制御状態"""
    STOPPED = "stopped"       # 停止中
    STARTING = "starting"     # 発進中（高PWMで蹴り出し）
    RUNNING = "running"       # 走行中（低PWMで維持）
    BRAKING = "braking"       # 制動中


@dataclass
class ControllerConfig:
    """制御パラメータ"""
    # PWM設定（Phase 2-B分析結果）
    startup_pwm: float = 0.147      # 起動用
    maintain_pwm: float = 0.067     # 維持用
    threshold_pwm: float = 0.117    # 始動検出閾値
    
    # タイミング設定
    startup_timeout: float = 2.0    # 起動タイムアウト（秒）
    startup_max_duration: float = 0.5  # 起動PWM最大継続時間
    
    # 音響検出設定
    zcr_threshold_ratio: float = 1.5  # ZCR変化率閾値（1.5倍で始動と判定）
    rms_threshold_ratio: float = 1.3  # RMS変化率閾値
    
    # 安全設定
    stall_detection_time: float = 2.0  # スタック判定時間
    max_pwm: float = 0.3              # 最大PWM制限


@dataclass
class ControllerState:
    """コントローラー状態"""
    state: ControlState = ControlState.STOPPED
    current_pwm: float = 0.0
    target_speed: float = 0.0
    
    # タイミング
    state_start_time: float = 0.0
    last_update_time: float = 0.0
    
    # 音響ベースライン（STOPPED時の値）
    baseline_zcr: float = 0.0
    baseline_rms: float = 0.0
    
    # 検出フラグ
    motor_started: bool = False
    stall_detected: bool = False


class AcousticThrottleController:
    """
    音響フィードバック・ヒステリシス制御コントローラー
    
    使用例:
        controller = AcousticThrottleController(set_throttle_func)
        controller.start(target_speed=0.2)
        
        # 10Hzで呼び出し
        while running:
            pwm = await controller.update()
            await asyncio.sleep(0.1)
    """
    
    def __init__(
        self,
        set_throttle: Callable[[float], None],
        config: Optional[ControllerConfig] = None
    ):
        """
        Args:
            set_throttle: スロットル設定関数 (pwm: float) -> None
            config: 制御パラメータ
        """
        self.set_throttle = set_throttle
        self.config = config or ControllerConfig()
        self.state = ControllerState()
        
        # 音響マネージャー参照
        self._acoustic_manager: Optional[AcousticManager] = None
        self._classifier = None
        
        # コールバック
        self._on_state_change: Optional[Callable] = None
        
        logger.info(f"AcousticThrottleController initialized: "
                   f"startup={self.config.startup_pwm:.3f}, "
                   f"maintain={self.config.maintain_pwm:.3f}")
    
    def set_state_change_callback(self, callback: Callable):
        """状態変化コールバック設定"""
        self._on_state_change = callback
    
    def _notify_state_change(self, old_state: ControlState, new_state: ControlState):
        """状態変化を通知"""
        logger.info(f"State: {old_state.value} -> {new_state.value}")
        if self._on_state_change:
            self._on_state_change(old_state, new_state)
    
    def _get_acoustic_manager(self) -> Optional[AcousticManager]:
        """音響マネージャー取得（遅延初期化）"""
        if self._acoustic_manager is None:
            self._acoustic_manager = AcousticManager.get_instance()
        return self._acoustic_manager
    
    def _get_classifier(self):
        """分類器取得"""
        if self._classifier is None:
            self._classifier = get_classifier()
        return self._classifier
    
    def start(self, target_speed: float = 0.2):
        """
        走行開始
        
        Args:
            target_speed: 目標速度（0.0-1.0）
        """
        if self.state.state != ControlState.STOPPED:
            logger.warning(f"Cannot start: current state is {self.state.state.value}")
            return
        
        # ベースライン取得
        self._capture_baseline()
        
        # 状態遷移
        old_state = self.state.state
        self.state.state = ControlState.STARTING
        self.state.target_speed = min(target_speed, 1.0)
        self.state.state_start_time = time.time()
        self.state.motor_started = False
        self.state.stall_detected = False
        
        # 起動PWM出力
        self.state.current_pwm = self.config.startup_pwm
        self.set_throttle(self.state.current_pwm)
        
        self._notify_state_change(old_state, self.state.state)
        logger.info(f"Starting with PWM={self.state.current_pwm:.3f}")
    
    def stop(self):
        """走行停止"""
        old_state = self.state.state
        self.state.state = ControlState.STOPPED
        self.state.current_pwm = 0.0
        self.state.target_speed = 0.0
        self.state.motor_started = False
        
        self.set_throttle(0.0)
        
        if old_state != ControlState.STOPPED:
            self._notify_state_change(old_state, self.state.state)
        logger.info("Stopped")
    
    def _capture_baseline(self):
        """音響ベースライン取得"""
        manager = self._get_acoustic_manager()
        if manager and manager.is_capturing():
            features = manager.get_current_features()
            if features:
                self.state.baseline_zcr = features.zcr
                self.state.baseline_rms = features.rms_energy
                logger.debug(f"Baseline captured: ZCR={self.state.baseline_zcr:.4f}, "
                           f"RMS={self.state.baseline_rms:.4f}")
    
    def _detect_motor_start(self) -> bool:
        """
        音響特徴量からモーター始動を検出
        
        Returns:
            始動検出したらTrue
        """
        manager = self._get_acoustic_manager()
        if not manager or not manager.is_capturing():
            return False
        
        features = manager.get_current_features()
        if not features:
            return False
        
        # ベースラインがない場合はスキップ
        if self.state.baseline_zcr == 0:
            return False
        
        # ZCR変化率で判定（最も信頼性が高い）
        zcr_ratio = features.zcr / self.state.baseline_zcr if self.state.baseline_zcr > 0 else 1.0
        rms_ratio = features.rms_energy / self.state.baseline_rms if self.state.baseline_rms > 0 else 1.0
        
        # 閾値判定
        zcr_triggered = zcr_ratio >= self.config.zcr_threshold_ratio
        rms_triggered = rms_ratio >= self.config.rms_threshold_ratio
        
        if zcr_triggered or rms_triggered:
            logger.info(f"Motor start detected: ZCR ratio={zcr_ratio:.2f}, RMS ratio={rms_ratio:.2f}")
            return True
        
        return False
    
    def _detect_motor_stop(self) -> bool:
        """
        音響特徴量からモーター停止を検出
        
        Returns:
            停止検出したらTrue
        """
        classifier = self._get_classifier()
        if not classifier or not classifier.is_loaded:
            return False
        
        manager = self._get_acoustic_manager()
        if not manager or not manager.is_capturing():
            return False
        
        features = manager.get_current_features()
        if not features:
            return False
        
        # 分類器で状態推定
        state, confidence = classifier.predict(features, self.state.current_pwm)
        
        if state == "throttle_stop" and confidence > 0.8:
            logger.info(f"Motor stop detected: state={state}, confidence={confidence:.2f}")
            return True
        
        return False
    
    async def update(self) -> float:
        """
        制御ループ更新（10Hzで呼び出し）
        
        Returns:
            現在のPWM値
        """
        now = time.time()
        self.state.last_update_time = now
        elapsed = now - self.state.state_start_time
        
        if self.state.state == ControlState.STOPPED:
            return 0.0
        
        elif self.state.state == ControlState.STARTING:
            return await self._update_starting(elapsed)
        
        elif self.state.state == ControlState.RUNNING:
            return await self._update_running(elapsed)
        
        elif self.state.state == ControlState.BRAKING:
            return await self._update_braking(elapsed)
        
        return self.state.current_pwm
    
    async def _update_starting(self, elapsed: float) -> float:
        """STARTING状態の更新"""
        
        # 音響で始動検出
        if self._detect_motor_start():
            self.state.motor_started = True
            
            # RUNNING状態へ遷移
            old_state = self.state.state
            self.state.state = ControlState.RUNNING
            self.state.state_start_time = time.time()
            
            # 維持PWMに降下
            self.state.current_pwm = self.config.maintain_pwm
            self.set_throttle(self.state.current_pwm)
            
            self._notify_state_change(old_state, self.state.state)
            logger.info(f"Motor started! Switching to maintain PWM={self.state.current_pwm:.3f}")
            return self.state.current_pwm
        
        # 起動タイムアウトチェック
        if elapsed > self.config.startup_timeout:
            logger.warning("Startup timeout - motor did not start")
            self.stop()
            return 0.0
        
        # 起動PWM最大継続時間を超えたら徐々に増加（オプション）
        if elapsed > self.config.startup_max_duration:
            # PWMを少しずつ増加（最大値まで）
            increment = 0.01 * (elapsed - self.config.startup_max_duration)
            new_pwm = min(
                self.config.startup_pwm + increment,
                self.config.max_pwm
            )
            if new_pwm != self.state.current_pwm:
                self.state.current_pwm = new_pwm
                self.set_throttle(self.state.current_pwm)
                logger.debug(f"Increasing startup PWM to {self.state.current_pwm:.3f}")
        
        return self.state.current_pwm
    
    async def _update_running(self, elapsed: float) -> float:
        """RUNNING状態の更新"""
        
        # 停止検出
        if self._detect_motor_stop():
            logger.warning("Motor stopped unexpectedly")
            self.state.stall_detected = True
            self.stop()
            return 0.0
        
        # 維持PWMで継続
        # 将来的にはPID制御で速度調整
        return self.state.current_pwm
    
    async def _update_braking(self, elapsed: float) -> float:
        """BRAKING状態の更新"""
        # 現状はシンプルに停止
        self.stop()
        return 0.0
    
    def get_status(self) -> dict:
        """現在の状態を取得"""
        return {
            "state": self.state.state.value,
            "current_pwm": self.state.current_pwm,
            "target_speed": self.state.target_speed,
            "motor_started": self.state.motor_started,
            "stall_detected": self.state.stall_detected,
            "config": {
                "startup_pwm": self.config.startup_pwm,
                "maintain_pwm": self.config.maintain_pwm,
                "threshold_pwm": self.config.threshold_pwm,
            }
        }


# === グローバルインスタンス ===

_controller: Optional[AcousticThrottleController] = None


def get_acoustic_throttle_controller() -> Optional[AcousticThrottleController]:
    """コントローラーインスタンス取得"""
    return _controller


def init_acoustic_throttle_controller(
    set_throttle: Callable[[float], None],
    config: Optional[ControllerConfig] = None
) -> AcousticThrottleController:
    """コントローラー初期化"""
    global _controller
    _controller = AcousticThrottleController(set_throttle, config)
    return _controller
```

### 2. APIエンドポイント追加

**ファイル**: `http_server/routes/acoustic.py` に追加

```python
# === ヒステリシス制御エンドポイント ===

from ..core.acoustic_throttle_controller import (
    get_acoustic_throttle_controller,
    ControllerConfig
)


class ThrottleStartRequest(BaseModel):
    target_speed: float = 0.2


@router.post("/throttle/start")
async def start_acoustic_throttle(request: ThrottleStartRequest):
    """音響フィードバック制御で走行開始"""
    controller = get_acoustic_throttle_controller()
    if controller is None:
        raise HTTPException(
            status_code=503,
            detail="Acoustic throttle controller not initialized"
        )
    
    controller.start(target_speed=request.target_speed)
    return {
        "status": "started",
        "target_speed": request.target_speed,
        "startup_pwm": controller.config.startup_pwm
    }


@router.post("/throttle/stop")
async def stop_acoustic_throttle():
    """音響フィードバック制御を停止"""
    controller = get_acoustic_throttle_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    
    controller.stop()
    return {"status": "stopped"}


@router.get("/throttle/status")
async def get_throttle_status():
    """制御状態を取得"""
    controller = get_acoustic_throttle_controller()
    if controller is None:
        return {"initialized": False}
    
    status = controller.get_status()
    status["initialized"] = True
    return status


@router.post("/throttle/config")
async def update_throttle_config(
    startup_pwm: float = None,
    maintain_pwm: float = None,
    threshold_pwm: float = None
):
    """制御パラメータを更新"""
    controller = get_acoustic_throttle_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")
    
    if startup_pwm is not None:
        controller.config.startup_pwm = startup_pwm
    if maintain_pwm is not None:
        controller.config.maintain_pwm = maintain_pwm
    if threshold_pwm is not None:
        controller.config.threshold_pwm = threshold_pwm
    
    return {
        "status": "updated",
        "config": {
            "startup_pwm": controller.config.startup_pwm,
            "maintain_pwm": controller.config.maintain_pwm,
            "threshold_pwm": controller.config.threshold_pwm,
        }
    }
```

### 3. main.py への統合

**ファイル**: `http_server/main.py` に追加

```python
# インポート追加
from .core.acoustic_throttle_controller import init_acoustic_throttle_controller, ControllerConfig

# シナリオエグゼキュータ初期化の後に追加

# === 音響スロットルコントローラー初期化 ===
try:
    # Phase 2-B で特定したパラメータ
    throttle_config = ControllerConfig(
        startup_pwm=0.147,
        maintain_pwm=0.067,
        threshold_pwm=0.117,
    )
    
    _throttle_controller = init_acoustic_throttle_controller(
        set_throttle=_set_throttle,
        config=throttle_config
    )
    print(f"[Server] Acoustic throttle controller initialized: "
          f"startup={throttle_config.startup_pwm:.3f}, "
          f"maintain={throttle_config.maintain_pwm:.3f}")
except Exception as e:
    print(f"[Server] Failed to initialize acoustic throttle controller: {e}")
```

### 4. 制御ループ実装（オプション）

バックグラウンドで10Hz制御ループを実行する場合：

**ファイル**: `http_server/core/acoustic_throttle_controller.py` に追加

```python
class AcousticThrottleControlLoop:
    """バックグラウンド制御ループ"""
    
    def __init__(self, controller: AcousticThrottleController):
        self.controller = controller
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """制御ループ開始"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Control loop started")
    
    async def stop(self):
        """制御ループ停止"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Control loop stopped")
    
    async def _loop(self):
        """10Hz制御ループ"""
        interval = 0.1  # 100ms
        
        while self._running:
            try:
                start_time = time.time()
                
                # 制御更新
                await self.controller.update()
                
                # インターバル調整
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Control loop error: {e}")
                await asyncio.sleep(interval)
```

---

## テスト手順

### 1. サーバー起動

```bash
cd ~/jetracer-agent
source venv/bin/activate
python -m http_server.main
```

起動ログ確認：
```
[Server] Acoustic throttle controller initialized: startup=0.147, maintain=0.067
```

### 2. APIテスト

```bash
# 状態確認
curl http://localhost:8000/acoustic/throttle/status

# 音響キャプチャ開始（必須）
curl -X POST http://localhost:8000/acoustic/capture/start

# 走行開始（車輪を浮かせて！）
curl -X POST http://localhost:8000/acoustic/throttle/start \
  -H "Content-Type: application/json" \
  -d '{"target_speed": 0.2}'

# 状態確認（STARTING → RUNNING に遷移するか）
curl http://localhost:8000/acoustic/throttle/status

# 停止
curl -X POST http://localhost:8000/acoustic/throttle/stop
```

### 3. 期待される動作

1. `throttle/start` 呼び出し
2. PWM = 0.147 で蹴り出し
3. 音響でZCR急増を検出（モーター始動）
4. 自動的に PWM = 0.067 に降下
5. 安定して走行継続

---

## ログ出力例

```
[AcousticThrottleController] Starting with PWM=0.147
[AcousticThrottleController] Baseline captured: ZCR=0.013, RMS=0.0035
[AcousticThrottleController] State: stopped -> starting
... (数百ms後)
[AcousticThrottleController] Motor start detected: ZCR ratio=1.92, RMS ratio=1.37
[AcousticThrottleController] Motor started! Switching to maintain PWM=0.067
[AcousticThrottleController] State: starting -> running
```

---

## 安全に関する注意

⚠️ **必ず車輪を浮かせた状態でテストしてください**

1. JetRacerを台に載せて車輪を浮かせる
2. テスト中は手を車輪から離す
3. 異常があればすぐに電源OFF

---

## 次のステップ（Phase 2-D）

1. **実走行テスト**: 地面に置いて低速走行
2. **パラメータ調整**: 実環境でstartup_pwm, maintain_pwmを微調整
3. **PID制御追加**: maintain_pwmを速度フィードバックで調整
4. **PC側GUI統合**: AutonomousパネルにAcoustic制御を追加
