"""センサーデータ取得のラッパー"""
import time
from typing import Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

from ..acoustic import AcousticManager
from ..i2c_sensors import sensor_manager


@dataclass
class SensorSnapshot:
    """センサーデータのスナップショット"""
    # 音響
    zcr: float = 0.0
    rms: float = 0.0

    # IMU
    heading: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0

    # メタ
    timestamp: float = 0.0
    acoustic_valid: bool = False
    imu_valid: bool = False


class SensorDataSource:
    """センサーデータの統合取得"""

    def __init__(self, history_size: int = 50):
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)

    def get_snapshot(self) -> SensorSnapshot:
        """現在のセンサーデータを取得"""
        snapshot = SensorSnapshot(timestamp=time.time())

        # 音響データ取得
        try:
            manager = AcousticManager.get_instance()
            if manager.is_capturing():
                features = manager.get_current_features()
                if features:
                    snapshot.zcr = features.zcr
                    snapshot.rms = features.rms_energy
                    snapshot.acoustic_valid = True
        except Exception:
            pass

        # IMUデータ取得
        try:
            imu_data = sensor_manager.read_imu()
            if imu_data and imu_data.valid:
                snapshot.heading = imu_data.heading
                snapshot.roll = imu_data.roll
                snapshot.pitch = imu_data.pitch
                snapshot.accel_x = imu_data.accel_x
                snapshot.accel_y = imu_data.accel_y
                snapshot.accel_z = imu_data.accel_z
                snapshot.gyro_x = imu_data.gyro_x
                snapshot.gyro_y = imu_data.gyro_y
                snapshot.gyro_z = imu_data.gyro_z
                snapshot.imu_valid = True
        except Exception:
            pass

        # 履歴に追加
        self.history.append(snapshot)

        return snapshot

    def get_history(self, seconds: float = 5.0) -> list:
        """指定秒数分の履歴を取得"""
        if not self.history:
            return []

        cutoff = time.time() - seconds
        return [s for s in self.history if s.timestamp >= cutoff]

    def get_heading_history(self, seconds: float = 5.0) -> list:
        """heading履歴を取得"""
        history = self.get_history(seconds)
        return [(s.timestamp, s.heading) for s in history if s.imu_valid]

    def get_accel_magnitude_history(self, seconds: float = 1.0) -> list:
        """加速度magnitude履歴を取得"""
        history = self.get_history(seconds)
        result = []
        for s in history:
            if s.imu_valid:
                mag = np.sqrt(s.accel_x**2 + s.accel_y**2 + s.accel_z**2)
                result.append((s.timestamp, mag))
        return result


# シングルトン
_data_source: Optional[SensorDataSource] = None


def get_data_source() -> SensorDataSource:
    global _data_source
    if _data_source is None:
        _data_source = SensorDataSource()
    return _data_source
