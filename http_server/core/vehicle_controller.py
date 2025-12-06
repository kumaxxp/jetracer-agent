"""車両制御"""
from typing import Optional


class VehicleController:
    def __init__(self):
        self._steering = 0.0
        self._throttle = 0.0
        self._car = None
        self._init_jetracer()

    def _init_jetracer(self):
        """JetRacer初期化"""
        try:
            from jetracer.nvidia_racecar import NvidiaRacecar
            self._car = NvidiaRacecar()
            self._car.steering = 0.0
            self._car.throttle = 0.0
        except Exception as e:
            print(f"[Vehicle] JetRacer not available: {e}")
            self._car = None

    def set_steering(self, value: float):
        """ステアリング設定 (-1.0 ~ 1.0)"""
        self._steering = max(-1.0, min(1.0, value))
        if self._car:
            self._car.steering = self._steering

    def set_throttle(self, value: float, max_limit: float = 0.5):
        """スロットル設定 (0.0 ~ max_limit)"""
        self._throttle = max(0.0, min(max_limit, value))
        if self._car:
            self._car.throttle = self._throttle

    def stop(self):
        """緊急停止"""
        self._throttle = 0.0
        self._steering = 0.0
        if self._car:
            self._car.throttle = 0.0
            self._car.steering = 0.0

    def get_status(self) -> dict:
        """現在の状態取得"""
        return {
            "steering": self._steering,
            "throttle": self._throttle,
            "connected": self._car is not None
        }


# シングルトン
vehicle_controller = VehicleController()
