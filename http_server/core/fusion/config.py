"""センサーフュージョン設定"""
from dataclasses import dataclass


@dataclass
class FusionConfig:
    """フュージョン検出の閾値設定"""

    # スタック検知
    motor_running_zcr: float = 0.003      # モーター走行中のZCR下限
    accel_stopped_threshold: float = 0.1   # m/s² - 停止とみなす加速度
    stuck_min_duration: float = 1.0        # 秒 - スタック判定の最小継続時間

    # 持ち上げ検出
    motor_stopped_zcr: float = 0.001       # モーター停止のZCR上限
    accel_variance_threshold: float = 2.0  # m/s² - 持ち上げ時の加速度変動
    gravity_deviation: float = 2.0         # m/s² - Z軸の重力偏差
    heading_variance_threshold: float = 5.0  # deg² - heading変動閾値

    # IMUドリフト検出
    drift_rate_threshold: float = 0.5      # deg/s - ドリフト速度閾値
    accel_stable_threshold: float = 0.3    # m/s² - 安定判定の加速度
    drift_min_duration: float = 3.0        # 秒 - ドリフト判定の最小継続時間

    # 衝突検知
    impact_rms_threshold: float = 0.5      # 衝撃音のRMS閾値
    impact_accel_threshold: float = 15.0   # m/s² - 衝撃加速度


# グローバル設定インスタンス
fusion_config = FusionConfig()
