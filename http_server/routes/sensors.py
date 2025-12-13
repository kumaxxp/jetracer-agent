"""センサー関連APIルート - FaBo JetRacer対応版

I2Cセンサー（BNO055 IMU、FaBo PWM入力、VL53L7CX距離計）のエンドポイント
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from ..core.i2c_sensors import sensor_manager, IMUData, PWMInputData

router = APIRouter(prefix="/sensors", tags=["sensors"])


class DeviceInfo(BaseModel):
    """検出デバイス情報"""
    address: str
    address_int: int
    name: str
    type: str


class ScanResponse(BaseModel):
    """I2Cスキャン結果"""
    timestamp: str
    bus: int
    devices: List[DeviceInfo]
    count: int


class InitRequest(BaseModel):
    """センサー初期化リクエスト"""
    sensor_type: str  # "imu", "pwm_input", "distance"
    address: Optional[int] = None


class LEDRequest(BaseModel):
    """LED制御リクエスト"""
    color: str  # red, blue, yellow, green, white, orange, magenta, lime, pink, off, normal


@router.get("/scan", response_model=ScanResponse)
def scan_i2c_bus():
    """I2Cバスをスキャンしてデバイスを検出
    
    FaBo JetRacer標準デバイス:
    - 0x08: ESP32S3 (PWM入力)
    - 0x28/0x29: BNO055 (IMU)
    - 0x33: VL53L7CX (距離計)
    - 0x40: PCA9685 (サーボドライバ)
    """
    devices = sensor_manager.scan_devices()
    
    return ScanResponse(
        timestamp=datetime.now().isoformat(),
        bus=sensor_manager.bus_num,
        devices=[DeviceInfo(**d) for d in devices],
        count=len(devices)
    )


@router.post("/init")
def initialize_sensor(req: InitRequest):
    """センサー初期化
    
    sensor_type:
    - "imu": BNO055 (default: 0x29 for FaBo JetRacer)
    - "pwm_input": FaBo JetRacer PWM (default: 0x08)
    - "distance": VL53L7CX (default: 0x33)
    """
    result = {"timestamp": datetime.now().isoformat(), "success": False}
    
    if req.sensor_type == "imu":
        address = req.address or 0x29  # FaBo JetRacerはAD0=HIGHなので0x29
        success = sensor_manager.initialize_imu(address)
        result["success"] = success
        result["sensor"] = "BNO055"
        result["address"] = f"0x{address:02X}"
        
    elif req.sensor_type == "pwm_input":
        address = req.address or 0x08
        success = sensor_manager.initialize_pwm_input(address)
        result["success"] = success
        result["sensor"] = "FaBo JetRacer PWM"
        result["address"] = f"0x{address:02X}"
        if success and sensor_manager.pwm:
            result["board_revision"] = sensor_manager.pwm._board_revision
            result["firmware_version"] = sensor_manager.pwm._firmware_version
        
    elif req.sensor_type == "distance":
        address = req.address or 0x33
        success = sensor_manager.initialize_distance(address)
        result["success"] = success
        result["sensor"] = "VL53L7CX"
        result["address"] = f"0x{address:02X}"
        if not success:
            result["note"] = "VL53L7CX requires ST library for full implementation"
        
    else:
        result["error"] = f"Unknown sensor type: {req.sensor_type}"
        
    return result


@router.get("/imu")
def read_imu():
    """BNO055 IMUデータ読み取り
    
    Returns:
    - euler: オイラー角（heading/roll/pitch）
    - accel: 加速度（x/y/z, m/s²）
    - gyro: ジャイロ（x/y/z, deg/s）
    - mag: 磁気（x/y/z, μT）
    - temperature: 温度（°C）
    - calibration: キャリブレーション状態（0-3）
    """
    data = sensor_manager.read_imu()
    
    if data is None:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": "IMU not initialized",
            "data": None
        }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "data": data.to_dict()
    }


@router.get("/pwm_input")
def read_pwm_input():
    """FaBo JetRacer PWM入力データ読み取り
    
    Returns:
    - channels: 各チャンネルのPWM値
      - ch1_steering: ステアリング
      - ch2_throttle: スロットル  
      - ch3_mode: モード切替
    - mode: "rc", "ai", "transition"
    - board_info: ボードリビジョン、ファームウェアバージョン
    """
    data = sensor_manager.read_pwm()
    
    if data is None:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": "PWM input not initialized",
            "data": None
        }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "data": data.to_dict()
    }


@router.get("/distance")
def read_distance():
    """VL53L7CX距離計データ読み取り（将来実装）
    
    Returns:
    - grid_8x8: 8x8の距離データ（mm）
    - statistics: 最小/最大/平均距離
    """
    data = sensor_manager.read_distance()
    
    if data is None:
        return {
            "timestamp": datetime.now().isoformat(),
            "error": "Distance sensor not initialized (requires ST library)",
            "data": None
        }
    
    return {
        "timestamp": datetime.now().isoformat(),
        "data": data.to_dict()
    }


@router.get("/all")
def read_all_sensors():
    """全センサーデータ読み取り"""
    return sensor_manager.get_all_data()


@router.get("/status")
def get_sensor_status():
    """センサー状態取得"""
    status = sensor_manager.get_status()
    status["timestamp"] = datetime.now().isoformat()
    return status


@router.post("/led")
def set_led_color(req: LEDRequest):
    """FaBo JetRacer LEDカラー設定
    
    Colors: red, blue, yellow, green, white, orange, magenta, lime, pink, off, normal
    """
    success = sensor_manager.set_led_color(req.color)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "color": req.color
    }


@router.post("/led/{color}")
def set_led_color_path(color: str):
    """FaBo JetRacer LEDカラー設定（パス指定版）
    
    Example: POST /sensors/led/green
    """
    success = sensor_manager.set_led_color(color)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "success": success,
        "color": color
    }
