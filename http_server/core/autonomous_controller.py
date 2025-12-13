"""自律走行コントローラー

セグメンテーションベースの自律走行制御ループを管理する。
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum
import threading


class ControlMode(Enum):
    """制御モード"""
    INIT = "init"               # 初期化中
    MANUAL = "manual"           # 手動（RC）
    AUTO = "auto"               # 自動走行
    EMERGENCY_STOP = "emergency_stop"  # 緊急停止


@dataclass
class ControllerConfig:
    """コントローラー設定"""
    loop_hz: float = 10.0                # 制御ループ周波数
    use_dual_camera: bool = False        # デュアルカメラ使用
    front_camera_id: int = 0             # 正面カメラID
    ground_camera_id: int = 1            # 足元カメラID
    mode_switch_threshold: int = 1500    # RC/AIモード切替閾値（μs）
    
    # セグメンテーション
    use_lightweight_model: bool = True   # 軽量モデル使用
    segmentation_timeout_sec: float = 2.0  # セグメンテーションタイムアウト


@dataclass
class ControllerState:
    """コントローラー状態"""
    mode: ControlMode = ControlMode.INIT
    running: bool = False
    
    # 現在の制御値
    steering: float = 0.0
    throttle: float = 0.0
    
    # センサー状態
    road_ratio: float = 0.0
    lidar_min_mm: int = 9999
    imu_heading: float = 0.0
    imu_roll: float = 0.0
    imu_pitch: float = 0.0
    
    # 統計
    loop_count: int = 0
    last_loop_time_ms: float = 0.0
    avg_loop_time_ms: float = 0.0
    
    # タイムスタンプ
    last_update: float = 0.0
    started_at: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "running": self.running,
            "control": {
                "steering": round(self.steering, 3),
                "throttle": round(self.throttle, 3),
            },
            "sensors": {
                "road_ratio": round(self.road_ratio, 3),
                "lidar_min_mm": self.lidar_min_mm,
                "imu": {
                    "heading": round(self.imu_heading, 1),
                    "roll": round(self.imu_roll, 1),
                    "pitch": round(self.imu_pitch, 1),
                }
            },
            "stats": {
                "loop_count": self.loop_count,
                "last_loop_time_ms": round(self.last_loop_time_ms, 2),
                "avg_loop_time_ms": round(self.avg_loop_time_ms, 2),
                "uptime_sec": round(time.time() - self.started_at, 1) if self.started_at else 0,
            },
            "last_update": self.last_update,
        }


class AutonomousController:
    """自律走行コントローラー"""
    
    def __init__(self, config: ControllerConfig = None):
        self.config = config or ControllerConfig()
        self.state = ControllerState()
        
        self._loop_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._lock = threading.Lock()
        
        # コンポーネント（遅延初期化）
        self._camera_manager = None
        self._sensor_manager = None
        self._segmenter = None
        self._steering_calc = None
        self._safety_guard = None
        self._pwm_controller = None
        
        # コールバック
        self._on_state_change: Optional[Callable] = None
        self._on_emergency: Optional[Callable] = None
    
    def _init_components(self):
        """コンポーネントを初期化"""
        # インポート
        from .camera_manager import camera_manager
        from .i2c_sensors import sensor_manager
        from .steering_calculator import steering_calculator
        from .safety_guard import safety_guard
        from .pwm_control import get_pwm_controller
        
        self._camera_manager = camera_manager
        self._sensor_manager = sensor_manager
        self._steering_calc = steering_calculator
        self._safety_guard = safety_guard
        self._pwm_controller = get_pwm_controller()
        
        # セグメンテーションモデル選択
        if self.config.use_lightweight_model:
            from .lightweight_segmentation import lightweight_segmentation
            self._segmenter = lightweight_segmentation
        else:
            # OneFormerを使う場合（重い）
            self._segmenter = None
    
    async def start(self) -> bool:
        """自律走行を開始"""
        if self.state.running:
            print("[AutoController] Already running")
            return False
        
        print("[AutoController] Starting...")
        
        # コンポーネント初期化
        self._init_components()
        
        # セグメンテーションモデルロード
        if self._segmenter and not self._segmenter.is_loaded():
            if not self._segmenter.load():
                print("[AutoController] Failed to load segmentation model")
                return False
        
        # PWM確認
        if not self._pwm_controller.is_available():
            print("[AutoController] Warning: PWM controller not available")
        
        # 状態初期化
        self.state.running = True
        self.state.mode = ControlMode.INIT
        self.state.started_at = time.time()
        self.state.loop_count = 0
        
        # ループ開始
        self._stop_event.clear()
        self._loop_task = asyncio.create_task(self._control_loop())
        
        print("[AutoController] Started")
        return True
    
    async def stop(self):
        """自律走行を停止"""
        if not self.state.running:
            return
        
        print("[AutoController] Stopping...")
        
        # ループ停止
        self._stop_event.set()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        
        # PWM停止
        if self._pwm_controller:
            self._pwm_controller.stop()
        
        self.state.running = False
        self.state.mode = ControlMode.INIT
        
        print("[AutoController] Stopped")
    
    def emergency_stop(self, reason: str = "Manual trigger"):
        """緊急停止"""
        print(f"[AutoController] EMERGENCY STOP: {reason}")
        
        # 即時停止
        if self._pwm_controller:
            self._pwm_controller.stop()
        
        self.state.mode = ControlMode.EMERGENCY_STOP
        self.state.steering = 0.0
        self.state.throttle = 0.0
        
        if self._safety_guard:
            self._safety_guard.trigger_emergency(reason)
        
        if self._on_emergency:
            self._on_emergency(reason)
    
    def clear_emergency(self) -> bool:
        """緊急停止を解除"""
        if self.state.mode != ControlMode.EMERGENCY_STOP:
            return False
        
        if self._safety_guard:
            self._safety_guard.clear_emergency()
        
        self.state.mode = ControlMode.MANUAL
        print("[AutoController] Emergency cleared, switched to MANUAL mode")
        return True
    
    async def _control_loop(self):
        """メイン制御ループ"""
        loop_interval = 1.0 / self.config.loop_hz
        loop_times = []
        
        print(f"[AutoController] Control loop started ({self.config.loop_hz} Hz)")
        
        while not self._stop_event.is_set():
            loop_start = time.time()
            
            try:
                await self._control_step()
            except Exception as e:
                print(f"[AutoController] Control step error: {e}")
                import traceback
                traceback.print_exc()
            
            # ループ時間計測
            elapsed = time.time() - loop_start
            self.state.last_loop_time_ms = elapsed * 1000
            
            # 平均計算（直近100ループ）
            loop_times.append(elapsed * 1000)
            if len(loop_times) > 100:
                loop_times.pop(0)
            self.state.avg_loop_time_ms = sum(loop_times) / len(loop_times)
            
            self.state.loop_count += 1
            self.state.last_update = time.time()
            
            # 残り時間スリープ
            sleep_time = loop_interval - elapsed
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        print("[AutoController] Control loop ended")
    
    async def _control_step(self):
        """1ステップの制御処理"""
        import concurrent.futures
        
        # 1. モード判定（PWM入力から）
        pwm_data = None
        if self._sensor_manager:
            pwm_data = self._sensor_manager.read_pwm()
            if pwm_data and pwm_data.valid:
                new_mode = self._determine_mode(pwm_data)
                if new_mode != self.state.mode and self.state.mode != ControlMode.EMERGENCY_STOP:
                    print(f"[AutoController] Mode change: {self.state.mode.value} -> {new_mode.value}")
                    self.state.mode = new_mode
        
        # 緊急停止中は何もしない
        if self.state.mode == ControlMode.EMERGENCY_STOP:
            return
        
        # 手動モード中はJetson制御なし
        if self.state.mode == ControlMode.MANUAL:
            return
        
        # 2. センサー読み取り
        imu_data = None
        lidar_data = None
        
        if self._sensor_manager:
            imu_data = self._sensor_manager.read_imu()
            lidar_data = self._sensor_manager.read_distance()
            
            if imu_data and imu_data.valid:
                self.state.imu_heading = imu_data.heading
                self.state.imu_roll = imu_data.roll
                self.state.imu_pitch = imu_data.pitch
            
            if lidar_data and lidar_data.valid:
                self.state.lidar_min_mm = lidar_data.min_distance
        
        # 3. カメラ＆セグメンテーション（非同期実行）
        front_analysis = None
        ground_analysis = None
        
        if self._camera_manager and self._segmenter:
            # 正面カメラ
            front_frame = self._camera_manager.read(self.config.front_camera_id)
            if front_frame is not None:
                # セグメンテーションを別スレッドで実行（イベントループをブロックしない）
                loop = asyncio.get_event_loop()
                seg_result = await loop.run_in_executor(
                    None, self._segmenter.segment, front_frame
                )
                if seg_result:
                    front_analysis = self._steering_calc.analyze_road_mask(seg_result["mask"])
                    self.state.road_ratio = front_analysis.road_ratio
            
            # デュアルカメラの場合は足元も
            if self.config.use_dual_camera:
                ground_frame = self._camera_manager.read(self.config.ground_camera_id)
                if ground_frame is not None:
                    loop = asyncio.get_event_loop()
                    seg_result = await loop.run_in_executor(
                        None, self._segmenter.segment, ground_frame
                    )
                    if seg_result:
                        ground_analysis = self._steering_calc.analyze_road_mask(seg_result["mask"])
        
        # 4. 安全チェック
        if self._safety_guard:
            safety_status = self._safety_guard.check(
                lidar_data=lidar_data,
                road_ratio=self.state.road_ratio,
                imu_data=imu_data
            )
            
            if not safety_status.safe:
                self.emergency_stop("; ".join(safety_status.reasons))
                return
        
        # 5. ステアリング計算
        if front_analysis:
            if ground_analysis and self.config.use_dual_camera:
                command = self._steering_calc.calculate_dual_camera(front_analysis, ground_analysis)
            else:
                command = self._steering_calc.calculate_steering(front_analysis)
            
            if command.stop:
                self.emergency_stop(command.reason)
                return
            
            self.state.steering = command.steering
            self.state.throttle = command.throttle
        
        # 6. PWM出力
        if self._pwm_controller and self._pwm_controller.is_available():
            self._pwm_controller.set_steering(self.state.steering)
            self._pwm_controller.set_throttle(self.state.throttle)
    
    def _determine_mode(self, pwm_data) -> ControlMode:
        """PWM入力からモードを判定"""
        if not pwm_data.valid:
            return self.state.mode
        
        # CH3でRC/AI切替
        if pwm_data.ch3 >= self.config.mode_switch_threshold:
            return ControlMode.AUTO
        else:
            return ControlMode.MANUAL
    
    def get_state(self) -> dict:
        """現在の状態を取得"""
        return self.state.to_dict()
    
    def set_callback(self, on_state_change: Callable = None, on_emergency: Callable = None):
        """コールバックを設定"""
        if on_state_change:
            self._on_state_change = on_state_change
        if on_emergency:
            self._on_emergency = on_emergency
    
    def update_config(self, **kwargs):
        """設定を更新"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def get_config(self) -> dict:
        """現在の設定を取得"""
        return {
            "loop_hz": self.config.loop_hz,
            "use_dual_camera": self.config.use_dual_camera,
            "front_camera_id": self.config.front_camera_id,
            "ground_camera_id": self.config.ground_camera_id,
            "mode_switch_threshold": self.config.mode_switch_threshold,
            "use_lightweight_model": self.config.use_lightweight_model,
        }


# シングルトンインスタンス
autonomous_controller = AutonomousController()
