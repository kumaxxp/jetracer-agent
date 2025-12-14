"""車両制御 - pwm_controlモジュールを使用

注意: NvidiaRacecar (jetracerパッケージ) は使用しない。
NvidiaRacecarは独自にPCA9685を初期化し、我々のpwm_controlと競合するため。
"""
from typing import Optional


class VehicleController:
    def __init__(self):
        self._steering = 0.0
        self._throttle = 0.0
        self._pwm = None
        self._init_pwm()

    def _init_pwm(self):
        """PWMコントローラー取得（既存インスタンスを使用）"""
        try:
            from .pwm_control import get_pwm_controller
            self._pwm = get_pwm_controller()
            if self._pwm.is_available():
                print("[Vehicle] ✓ Using pwm_control (not NvidiaRacecar)")
            else:
                print("[Vehicle] ✗ PWM controller not available")
                self._pwm = None
        except Exception as e:
            print(f"[Vehicle] ✗ PWM init error: {e}")
            self._pwm = None

    def set_steering(self, value: float):
        """ステアリング設定 (-1.0 ~ 1.0)"""
        self._steering = max(-1.0, min(1.0, value))
        if self._pwm and self._pwm.is_available():
            self._pwm.set_steering(self._steering)

    def set_throttle(self, value: float, max_limit: float = 0.5):
        """スロットル設定 (0.0 ~ max_limit)"""
        self._throttle = max(0.0, min(max_limit, value))
        if self._pwm and self._pwm.is_available():
            self._pwm.set_throttle(self._throttle)

    def stop(self):
        """緊急停止"""
        self._throttle = 0.0
        self._steering = 0.0
        if self._pwm and self._pwm.is_available():
            self._pwm.stop()

    def get_status(self) -> dict:
        """現在の状態取得"""
        return {
            "steering": self._steering,
            "throttle": self._throttle,
            "connected": self._pwm is not None and self._pwm.is_available()
        }


# シングルトン
vehicle_controller = VehicleController()
