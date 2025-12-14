"""Step & Check Navigation Controller

安全地点まで移動→停止→確認を繰り返す慎重な自律走行モード。

状態遷移:
  IDLE → PLANNING → MOVING → STOPPING → CONFIRMING → IDLE (繰り返し)
                      ↓
                  EMERGENCY (LiDAR危険検出時)

センサー役割:
- Segmentation: 走行可能領域の検出（PLANNING時）
- Distance Grid: 安全な目標地点の計算（PLANNING時）
- LiDAR 8x8: 障害物監視、非常停止（常時、上半分のみ）
- IMU: 停止確認（CONFIRMING時）
"""
import time
import threading
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

from .i2c_sensors import sensor_manager, IMUData, DistanceData
from .steering_calculator import SteeringCalculator, SteeringCommand
from .lightweight_segmentation import LightweightSegmentation
from .camera_manager import camera_manager


class StepCheckState(Enum):
    """Step & Check 状態"""
    IDLE = "idle"              # 停止中、開始待ち
    PLANNING = "planning"      # セグメンテーション分析、目標決定
    MOVING = "moving"          # 目標に向かって移動中
    STOPPING = "stopping"      # 減速中
    CONFIRMING = "confirming"  # 停止確認中（IMU）
    EMERGENCY = "emergency"    # 非常停止


@dataclass
class StepCheckConfig:
    """Step & Check 設定"""
    # 移動パラメータ
    step_distance_m: float = 1.0        # 1回の移動距離 (m)
    throttle_speed: float = 0.2         # 移動時スロットル (0.0-1.0) - 安全のため低め
    
    # 停止確認
    confirm_time_s: float = 1.0         # 停止確認時間 (s)
    imu_stop_accel_thresh: float = 0.5  # 停止判定加速度閾値 (m/s²)
    imu_stop_gyro_thresh: float = 5.0   # 停止判定ジャイロ閾値 (deg/s)
    
    # LiDAR安全設定
    lidar_safe_distance_mm: int = 300   # 安全距離 (mm)
    lidar_check_rows: int = 4           # チェックする上部行数 (0-3)
    
    # ステアリング
    steering_gain: float = 1.0          # ステアリングゲイン（低めに開始）
    steering_invert: bool = False       # ステアリング反転フラグ
    
    # タイムアウト
    move_timeout_s: float = 10.0        # 移動タイムアウト
    planning_timeout_s: float = 30.0    # 計画タイムアウト（OneFormer用）


@dataclass
class StepCheckStatus:
    """Step & Check 状態情報"""
    state: StepCheckState = StepCheckState.IDLE
    running: bool = False
    
    # 現在の制御値
    steering: float = 0.0
    throttle: float = 0.0
    
    # センサー情報
    lidar_min_mm: int = 9999
    lidar_safe: bool = True
    imu_accel_magnitude: float = 0.0
    imu_gyro_magnitude: float = 0.0
    imu_stopped: bool = False
    
    # 進行状況
    step_count: int = 0
    current_step_start_time: float = 0.0
    confirm_start_time: float = 0.0
    
    # 計画情報
    planned_steering: float = 0.0
    road_ratio: float = 0.0
    
    # 統計
    total_distance_m: float = 0.0
    last_update_time: float = 0.0
    error_message: str = ""


class StepCheckController:
    """Step & Check Navigation Controller"""
    
    def __init__(
        self,
        steering_calc: SteeringCalculator,
        segmentation: LightweightSegmentation,
        pwm_control=None
    ):
        self.steering_calc = steering_calc
        self.segmentation = segmentation
        self.pwm_control = pwm_control
        
        self.config = StepCheckConfig()
        self.status = StepCheckStatus()
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 最新のセンサーデータ
        self._last_imu: Optional[IMUData] = None
        self._last_distance: Optional[DistanceData] = None
        self._last_mask = None
        
    # =========================================================================
    # 公開API
    # =========================================================================
    
    def start(self) -> bool:
        """Step & Check 開始"""
        with self._lock:
            if self._running:
                return False
            
            self._running = True
            self.status.running = True
            self.status.state = StepCheckState.PLANNING
            self.status.step_count = 0
            self.status.error_message = ""
            
            self._thread = threading.Thread(target=self._control_loop, daemon=True)
            self._thread.start()
            
            print("[StepCheck] Started")
            return True
    
    def stop(self) -> bool:
        """Step & Check 停止"""
        with self._lock:
            if not self._running:
                return False
            
            self._running = False
            self.status.running = False
            
            # 車両停止
            self._stop_vehicle()
            
            self.status.state = StepCheckState.IDLE
            
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        
        print("[StepCheck] Stopped")
        return True
    
    def emergency_stop(self):
        """緊急停止"""
        with self._lock:
            self._running = False
            self.status.running = False
            self.status.state = StepCheckState.EMERGENCY
            self._stop_vehicle()
            self.status.error_message = "Emergency stop activated"
        
        print("[StepCheck] EMERGENCY STOP")
    
    def clear_emergency(self):
        """緊急停止解除"""
        with self._lock:
            if self.status.state == StepCheckState.EMERGENCY:
                self.status.state = StepCheckState.IDLE
                self.status.error_message = ""
                print("[StepCheck] Emergency cleared")
    
    def get_status(self) -> dict:
        """現在の状態を取得"""
        with self._lock:
            return {
                "state": self.status.state.value,
                "running": self.status.running,
                "control": {
                    "steering": self.status.steering,
                    "throttle": self.status.throttle,
                },
                "sensors": {
                    "lidar_min_mm": self.status.lidar_min_mm,
                    "lidar_safe": self.status.lidar_safe,
                    "imu_accel": round(self.status.imu_accel_magnitude, 2),
                    "imu_gyro": round(self.status.imu_gyro_magnitude, 2),
                    "imu_stopped": self.status.imu_stopped,
                    "road_ratio": round(self.status.road_ratio, 2),
                },
                "progress": {
                    "step_count": self.status.step_count,
                    "total_distance_m": round(self.status.total_distance_m, 2),
                    "planned_steering": round(self.status.planned_steering, 2),
                },
                "config": {
                    "step_distance_m": self.config.step_distance_m,
                    "lidar_safe_distance_mm": self.config.lidar_safe_distance_mm,
                    "confirm_time_s": self.config.confirm_time_s,
                },
                "error": self.status.error_message,
            }
    
    def update_config(self, **kwargs):
        """設定更新"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    print(f"[StepCheck] Config updated: {key} = {value}")
    
    # =========================================================================
    # 制御ループ
    # =========================================================================
    
    def _control_loop(self):
        """メイン制御ループ"""
        print("[StepCheck] Control loop started")
        
        while self._running:
            try:
                # センサーデータ更新
                self._update_sensors()
                
                # LiDAR安全チェック（常時）
                if not self._check_lidar_safety():
                    self.emergency_stop()
                    break
                
                # 状態に応じた処理
                state = self.status.state
                
                if state == StepCheckState.PLANNING:
                    self._do_planning()
                    
                elif state == StepCheckState.MOVING:
                    self._do_moving()
                    
                elif state == StepCheckState.STOPPING:
                    self._do_stopping()
                    
                elif state == StepCheckState.CONFIRMING:
                    self._do_confirming()
                    
                elif state == StepCheckState.IDLE:
                    time.sleep(0.1)
                    
                elif state == StepCheckState.EMERGENCY:
                    break
                
                self.status.last_update_time = time.time()
                time.sleep(0.05)  # 20Hz
                
            except Exception as e:
                print(f"[StepCheck] Error in control loop: {e}")
                import traceback
                traceback.print_exc()
                self.status.error_message = str(e)
                time.sleep(0.1)
        
        self._stop_vehicle()
        print("[StepCheck] Control loop ended")
    
    # =========================================================================
    # 状態処理
    # =========================================================================
    
    def _do_planning(self):
        """PLANNING: セグメンテーション分析、目標決定"""
        print(f"[StepCheck] PLANNING - Step #{self.status.step_count + 1}")
        
        # セグメンテーション実行
        try:
            # カメラからフレーム取得
            frame = camera_manager.read(0)
            if frame is None:
                self.status.error_message = "Camera capture failed"
                time.sleep(0.5)
                return
            
            # モデルがロードされていなければロード
            if not self.segmentation.is_loaded():
                if not self.segmentation.load():
                    self.status.error_message = "Failed to load segmentation model"
                    time.sleep(0.5)
                    return
            
            # セグメンテーション実行
            result = self.segmentation.segment(frame)
            if result is None:
                self.status.error_message = "Segmentation failed"
                time.sleep(0.5)
                return
            
            mask = result.get('mask')
            inference_time = result.get('inference_time_ms', 0)
            
            if mask is None:
                self.status.error_message = "No mask in segmentation result"
                time.sleep(0.5)
                return
            
            self._last_mask = mask
            
            # ROAD比率計算
            road_pixels = (mask == 1).sum()
            total_pixels = mask.size
            self.status.road_ratio = road_pixels / total_pixels
            
            print(f"[StepCheck] Segmentation: ROAD={self.status.road_ratio*100:.1f}%, time={inference_time:.0f}ms")
            
            # 走行可能領域が少なすぎる場合は停止
            if self.status.road_ratio < 0.05:
                print("[StepCheck] Insufficient road area, staying in IDLE")
                self.status.state = StepCheckState.IDLE
                self.status.error_message = "No road detected"
                return
            
            # ステアリング計算（gridモードを使用）
            cell_analysis = self.steering_calc.build_grid_from_mask(mask)
            command = self.steering_calc.calculate_steering_grid(cell_analysis)
            
            # ステアリングにゲインを適用
            raw_steering = command.steering
            if self.config.steering_invert:
                raw_steering = -raw_steering
            
            self.status.planned_steering = raw_steering * self.config.steering_gain
            self.status.planned_steering = max(-1.0, min(1.0, self.status.planned_steering))
            
            print(f"[StepCheck] Planned steering: {self.status.planned_steering:.2f} (raw: {raw_steering:.2f}, gain: {self.config.steering_gain})")
            
            # 移動開始
            self.status.state = StepCheckState.MOVING
            self.status.current_step_start_time = time.time()
            
        except Exception as e:
            print(f"[StepCheck] Planning error: {e}")
            import traceback
            traceback.print_exc()
            self.status.error_message = str(e)
            time.sleep(0.5)
    
    def _do_moving(self):
        """MOVING: 目標に向かって移動"""
        # タイムアウトチェック
        elapsed = time.time() - self.status.current_step_start_time
        if elapsed > self.config.move_timeout_s:
            print("[StepCheck] Move timeout, stopping")
            self.status.state = StepCheckState.STOPPING
            return
        
        # 距離推定（簡易：時間ベース）
        # 実際の速度は不明だが、おおよそ0.3m/sと仮定
        estimated_speed = 0.3  # m/s
        estimated_distance = elapsed * estimated_speed
        
        if estimated_distance >= self.config.step_distance_m:
            print(f"[StepCheck] Step distance reached ({estimated_distance:.2f}m), stopping")
            self.status.state = StepCheckState.STOPPING
            return
        
        # 移動継続
        self.status.steering = self.status.planned_steering
        self.status.throttle = self.config.throttle_speed
        
        self._apply_control(self.status.steering, self.status.throttle)
    
    def _do_stopping(self):
        """STOPPING: 減速"""
        # 即座にスロットル0
        self.status.throttle = 0.0
        self.status.steering = 0.0
        self._apply_control(0.0, 0.0)
        
        # 確認フェーズへ
        self.status.state = StepCheckState.CONFIRMING
        self.status.confirm_start_time = time.time()
        print("[StepCheck] STOPPING → CONFIRMING")
    
    def _do_confirming(self):
        """CONFIRMING: IMUで停止確認"""
        elapsed = time.time() - self.status.confirm_start_time
        
        # IMUで停止判定
        is_stopped = self._check_imu_stopped()
        self.status.imu_stopped = is_stopped
        
        if elapsed >= self.config.confirm_time_s:
            if is_stopped:
                # 停止確認完了
                self.status.step_count += 1
                self.status.total_distance_m += self.config.step_distance_m
                print(f"[StepCheck] Step #{self.status.step_count} complete. Total: {self.status.total_distance_m:.1f}m")
                
                # 次の計画へ
                self.status.state = StepCheckState.PLANNING
            else:
                # まだ動いている - 待機継続
                print("[StepCheck] Still moving, waiting...")
                self.status.confirm_start_time = time.time()
    
    # =========================================================================
    # センサー処理
    # =========================================================================
    
    def _update_sensors(self):
        """センサーデータ更新"""
        # IMU
        self._last_imu = sensor_manager.read_imu()
        
        # Distance (8x8 LiDAR)
        self._last_distance = sensor_manager.read_distance()
    
    def _check_lidar_safety(self) -> bool:
        """LiDAR安全チェック（上半分のみ）"""
        if self._last_distance is None or not self._last_distance.valid:
            # センサー無効時は安全とみなす（他のセンサーに依存）
            self.status.lidar_min_mm = 9999
            self.status.lidar_safe = True
            return True
        
        # 上半分（Row 0-3）の最小距離を取得
        min_distance = 9999
        check_rows = self.config.lidar_check_rows  # デフォルト4
        
        for row in range(check_rows):
            for col in range(8):
                dist = self._last_distance.distances[row][col]
                # 有効な距離値のみ（20-4000mm）
                if 20 < dist < 4000:
                    min_distance = min(min_distance, dist)
        
        self.status.lidar_min_mm = min_distance
        self.status.lidar_safe = min_distance > self.config.lidar_safe_distance_mm
        
        if not self.status.lidar_safe:
            print(f"[StepCheck] LiDAR DANGER! min={min_distance}mm < {self.config.lidar_safe_distance_mm}mm")
            return False
        
        return True
    
    def _check_imu_stopped(self) -> bool:
        """IMUで停止判定"""
        if self._last_imu is None or not self._last_imu.valid:
            # IMU無効時は時間ベースで判定（フォールバック）
            return True
        
        # 加速度の大きさ（重力を除く）
        # 静止時はおおよそ (0, 0, 9.8) m/s²
        accel_x = self._last_imu.accel_x
        accel_y = self._last_imu.accel_y
        accel_z = self._last_imu.accel_z - 9.8  # 重力補正
        
        accel_magnitude = math.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
        self.status.imu_accel_magnitude = accel_magnitude
        
        # ジャイロの大きさ
        gyro_magnitude = math.sqrt(
            self._last_imu.gyro_x**2 +
            self._last_imu.gyro_y**2 +
            self._last_imu.gyro_z**2
        )
        self.status.imu_gyro_magnitude = gyro_magnitude
        
        # 閾値以下なら停止
        is_stopped = (
            accel_magnitude < self.config.imu_stop_accel_thresh and
            gyro_magnitude < self.config.imu_stop_gyro_thresh
        )
        
        return is_stopped
    
    # =========================================================================
    # 車両制御
    # =========================================================================
    
    def _apply_control(self, steering: float, throttle: float):
        """制御値を適用"""
        if self.pwm_control is None:
            return
        
        try:
            self.pwm_control.set_steering(steering)
            self.pwm_control.set_throttle(throttle)
        except Exception as e:
            print(f"[StepCheck] PWM control error: {e}")
    
    def _stop_vehicle(self):
        """車両停止"""
        self.status.steering = 0.0
        self.status.throttle = 0.0
        self._apply_control(0.0, 0.0)
