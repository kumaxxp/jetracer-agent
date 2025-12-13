"""安全監視モジュール

自動走行時の安全条件をチェックし、緊急停止を判断する。
"""
import time
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class SafetyLevel(Enum):
    """安全レベル"""
    SAFE = "safe"               # 正常
    WARNING = "warning"         # 警告（減速推奨）
    DANGER = "danger"           # 危険（停止推奨）
    EMERGENCY = "emergency"     # 緊急停止


@dataclass
class SafetyParams:
    """安全監視パラメータ"""
    # LiDAR
    lidar_stop_distance_mm: int = 150      # 緊急停止距離
    lidar_slow_distance_mm: int = 300      # 減速距離
    lidar_center_cols: tuple = (2, 6)      # 中央領域（列）
    lidar_center_rows: tuple = (2, 6)      # 中央領域（行）
    
    # セグメンテーション
    road_stop_threshold: float = 0.10      # 緊急停止ROAD比率
    road_slow_threshold: float = 0.25      # 減速ROAD比率
    
    # IMU
    tilt_stop_threshold_deg: float = 30.0  # 緊急停止傾斜
    tilt_slow_threshold_deg: float = 20.0  # 警告傾斜
    
    # 通信
    heartbeat_timeout_sec: float = 3.0     # 通信断タイムアウト
    
    # 有効/無効
    lidar_enabled: bool = True
    imu_enabled: bool = True
    road_check_enabled: bool = True


@dataclass
class SafetyStatus:
    """安全状態"""
    level: SafetyLevel
    safe: bool
    reasons: List[str] = field(default_factory=list)
    
    # センサー値
    lidar_min_mm: int = 9999
    road_ratio: float = 1.0
    tilt_deg: float = 0.0
    
    # タイムスタンプ
    timestamp: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "level": self.level.value,
            "safe": self.safe,
            "reasons": self.reasons,
            "sensors": {
                "lidar_min_mm": self.lidar_min_mm,
                "road_ratio": round(self.road_ratio, 3),
                "tilt_deg": round(self.tilt_deg, 1),
            },
            "timestamp": self.timestamp
        }


class SafetyGuard:
    """安全監視エンジン"""
    
    def __init__(self, params: SafetyParams = None):
        self.params = params or SafetyParams()
        self._last_heartbeat = time.time()
        self._emergency_active = False
        self._emergency_reason = ""
    
    def check(
        self,
        lidar_data = None,
        road_ratio: float = None,
        imu_data = None
    ) -> SafetyStatus:
        """
        安全状態をチェック
        
        Args:
            lidar_data: DistanceData (8x8グリッド)
            road_ratio: ROAD比率 (0.0-1.0)
            imu_data: IMUData
        
        Returns:
            SafetyStatus
        """
        reasons = []
        level = SafetyLevel.SAFE
        
        lidar_min = 9999
        tilt = 0.0
        
        # 1. LiDARチェック
        if self.params.lidar_enabled and lidar_data is not None:
            lidar_result = self._check_lidar(lidar_data)
            lidar_min = lidar_result["min_mm"]
            
            if lidar_result["level"] == SafetyLevel.EMERGENCY:
                level = SafetyLevel.EMERGENCY
                reasons.append(f"障害物: {lidar_min}mm")
            elif lidar_result["level"] == SafetyLevel.WARNING and level != SafetyLevel.EMERGENCY:
                level = SafetyLevel.WARNING
                reasons.append(f"接近: {lidar_min}mm")
        
        # 2. ROAD比率チェック
        if self.params.road_check_enabled and road_ratio is not None:
            road_result = self._check_road(road_ratio)
            
            if road_result["level"] == SafetyLevel.EMERGENCY and level != SafetyLevel.EMERGENCY:
                level = SafetyLevel.EMERGENCY
                reasons.append(f"ROAD不足: {road_ratio:.0%}")
            elif road_result["level"] == SafetyLevel.WARNING and level == SafetyLevel.SAFE:
                level = SafetyLevel.WARNING
                reasons.append(f"ROAD少: {road_ratio:.0%}")
        
        # 3. IMU傾斜チェック
        if self.params.imu_enabled and imu_data is not None:
            imu_result = self._check_imu(imu_data)
            tilt = imu_result["tilt_deg"]
            
            if imu_result["level"] == SafetyLevel.EMERGENCY:
                level = SafetyLevel.EMERGENCY
                reasons.append(f"傾斜: {tilt:.1f}°")
            elif imu_result["level"] == SafetyLevel.WARNING and level == SafetyLevel.SAFE:
                level = SafetyLevel.WARNING
                reasons.append(f"傾斜注意: {tilt:.1f}°")
        
        # 安全判定
        safe = level in (SafetyLevel.SAFE, SafetyLevel.WARNING)
        
        return SafetyStatus(
            level=level,
            safe=safe,
            reasons=reasons,
            lidar_min_mm=lidar_min,
            road_ratio=road_ratio if road_ratio is not None else 1.0,
            tilt_deg=tilt,
            timestamp=time.time()
        )
    
    def _check_lidar(self, lidar_data) -> dict:
        """LiDARデータをチェック"""
        # 中央領域の最小距離を取得
        distances = lidar_data.distances  # 8x8 list
        
        center_distances = []
        for r in range(self.params.lidar_center_rows[0], self.params.lidar_center_rows[1]):
            for c in range(self.params.lidar_center_cols[0], self.params.lidar_center_cols[1]):
                d = distances[r][c]
                if d > 0:  # 有効な距離のみ
                    center_distances.append(d)
        
        if not center_distances:
            return {"level": SafetyLevel.SAFE, "min_mm": 9999}
        
        min_distance = min(center_distances)
        
        if min_distance < self.params.lidar_stop_distance_mm:
            return {"level": SafetyLevel.EMERGENCY, "min_mm": min_distance}
        elif min_distance < self.params.lidar_slow_distance_mm:
            return {"level": SafetyLevel.WARNING, "min_mm": min_distance}
        else:
            return {"level": SafetyLevel.SAFE, "min_mm": min_distance}
    
    def _check_road(self, road_ratio: float) -> dict:
        """ROAD比率をチェック"""
        if road_ratio < self.params.road_stop_threshold:
            return {"level": SafetyLevel.EMERGENCY}
        elif road_ratio < self.params.road_slow_threshold:
            return {"level": SafetyLevel.WARNING}
        else:
            return {"level": SafetyLevel.SAFE}
    
    def _check_imu(self, imu_data) -> dict:
        """IMU傾斜をチェック"""
        # roll/pitchの絶対値の最大を取得
        tilt = max(abs(imu_data.roll), abs(imu_data.pitch))
        
        if tilt > self.params.tilt_stop_threshold_deg:
            return {"level": SafetyLevel.EMERGENCY, "tilt_deg": tilt}
        elif tilt > self.params.tilt_slow_threshold_deg:
            return {"level": SafetyLevel.WARNING, "tilt_deg": tilt}
        else:
            return {"level": SafetyLevel.SAFE, "tilt_deg": tilt}
    
    def trigger_emergency(self, reason: str):
        """緊急停止をトリガー"""
        self._emergency_active = True
        self._emergency_reason = reason
        print(f"[SafetyGuard] EMERGENCY STOP: {reason}")
    
    def clear_emergency(self):
        """緊急停止を解除"""
        self._emergency_active = False
        self._emergency_reason = ""
        print("[SafetyGuard] Emergency cleared")
    
    def is_emergency(self) -> bool:
        """緊急停止状態か確認"""
        return self._emergency_active
    
    def get_emergency_reason(self) -> str:
        """緊急停止の理由を取得"""
        return self._emergency_reason
    
    def heartbeat(self):
        """通信ハートビートを更新"""
        self._last_heartbeat = time.time()
    
    def check_heartbeat(self) -> bool:
        """通信タイムアウトをチェック"""
        elapsed = time.time() - self._last_heartbeat
        return elapsed < self.params.heartbeat_timeout_sec
    
    def update_params(self, **kwargs):
        """パラメータを更新"""
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
    
    def get_params(self) -> dict:
        """現在のパラメータを取得"""
        return {
            "lidar_stop_distance_mm": self.params.lidar_stop_distance_mm,
            "lidar_slow_distance_mm": self.params.lidar_slow_distance_mm,
            "road_stop_threshold": self.params.road_stop_threshold,
            "road_slow_threshold": self.params.road_slow_threshold,
            "tilt_stop_threshold_deg": self.params.tilt_stop_threshold_deg,
            "tilt_slow_threshold_deg": self.params.tilt_slow_threshold_deg,
            "heartbeat_timeout_sec": self.params.heartbeat_timeout_sec,
            "lidar_enabled": self.params.lidar_enabled,
            "imu_enabled": self.params.imu_enabled,
            "road_check_enabled": self.params.road_check_enabled,
        }


# シングルトンインスタンス
safety_guard = SafetyGuard()
