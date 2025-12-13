"""I2Cセンサー管理モジュール - FaBo JetRacer対応版

サポートセンサー:
- FaBo JetRacer PWM入力 (ESP32S3 @ 0x08)
- BNO055 IMU (@ 0x28/0x29)
- VL53L7CX 距離計 (@ 0x33) - 将来実装

Jetson Orin Nano: I2C Bus 7
Jetson Nano: I2C Bus 1
"""
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum
import threading

# SMBus を使用（Jetsonで利用可能）
try:
    import smbus
    HAS_SMBUS = True
except ImportError:
    try:
        import smbus2 as smbus
        HAS_SMBUS = True
    except ImportError:
        HAS_SMBUS = False
        print("[I2C] smbus/smbus2 not available - using mock mode")


# =============================================================================
# データクラス定義
# =============================================================================

@dataclass
class IMUData:
    """BNO055 IMUセンサーデータ"""
    # オイラー角 (degrees)
    heading: float = 0.0    # Yaw (0-360)
    roll: float = 0.0
    pitch: float = 0.0
    # 加速度 (m/s²)
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    # ジャイロ (deg/s)
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    # 磁気 (μT)
    mag_x: float = 0.0
    mag_y: float = 0.0
    mag_z: float = 0.0
    # 温度 (°C)
    temperature: float = 0.0
    # キャリブレーション状態 (0-3, 3=完全)
    calib_sys: int = 0
    calib_gyro: int = 0
    calib_accel: int = 0
    calib_mag: int = 0
    # タイムスタンプ
    timestamp: float = 0.0
    # 有効フラグ
    valid: bool = False
    
    def to_dict(self) -> dict:
        return {
            "euler": {
                "heading": round(self.heading, 1),
                "roll": round(self.roll, 1),
                "pitch": round(self.pitch, 1),
                "unit": "deg"
            },
            "accel": {
                "x": round(self.accel_x, 3),
                "y": round(self.accel_y, 3),
                "z": round(self.accel_z, 3),
                "unit": "m/s²"
            },
            "gyro": {
                "x": round(self.gyro_x, 2),
                "y": round(self.gyro_y, 2),
                "z": round(self.gyro_z, 2),
                "unit": "deg/s"
            },
            "mag": {
                "x": round(self.mag_x, 1),
                "y": round(self.mag_y, 1),
                "z": round(self.mag_z, 1),
                "unit": "μT"
            },
            "temperature": round(self.temperature, 1),
            "calibration": {
                "sys": self.calib_sys,
                "gyro": self.calib_gyro,
                "accel": self.calib_accel,
                "mag": self.calib_mag,
                "status": "OK" if all(c >= 2 for c in [self.calib_sys, self.calib_gyro, self.calib_accel, self.calib_mag]) else "Calibrating"
            },
            "timestamp": self.timestamp,
            "valid": self.valid
        }


@dataclass
class PWMInputData:
    """FaBo JetRacer PWM入力データ"""
    # 各チャンネルのパルス幅 (μs) - 通常 1000-2000μs
    ch1: int = 1500  # ステアリング
    ch2: int = 1500  # スロットル
    ch3: int = 1500  # モード切替 (約1000=RC, 約2000=AI)
    # 正規化値 (-1.0 ~ 1.0)
    ch1_normalized: float = 0.0
    ch2_normalized: float = 0.0
    ch3_normalized: float = 0.0
    # モード判定
    mode: str = "unknown"  # "rc", "ai", "unknown"
    # ボード情報
    board_revision: str = ""
    firmware_version: int = 0
    # タイムスタンプ
    timestamp: float = 0.0
    # 有効フラグ
    valid: bool = False
    
    def to_dict(self) -> dict:
        return {
            "channels": {
                "ch1_steering": {"raw_us": self.ch1, "normalized": round(self.ch1_normalized, 3)},
                "ch2_throttle": {"raw_us": self.ch2, "normalized": round(self.ch2_normalized, 3)},
                "ch3_mode": {"raw_us": self.ch3, "normalized": round(self.ch3_normalized, 3)},
            },
            "mode": self.mode,
            "board_info": {
                "revision": self.board_revision,
                "firmware": self.firmware_version
            },
            "timestamp": self.timestamp,
            "valid": self.valid
        }


@dataclass 
class DistanceData:
    """VL53L7CX 距離計データ（将来実装）"""
    # 8x8 の距離データ (mm)
    distances: List[List[int]] = field(default_factory=lambda: [[0]*8 for _ in range(8)])
    # 最小/最大/平均距離
    min_distance: int = 0
    max_distance: int = 0
    avg_distance: float = 0.0
    # タイムスタンプ
    timestamp: float = 0.0
    valid: bool = False
    
    def to_dict(self) -> dict:
        return {
            "grid_8x8": self.distances,
            "statistics": {
                "min_mm": self.min_distance,
                "max_mm": self.max_distance,
                "avg_mm": round(self.avg_distance, 1)
            },
            "timestamp": self.timestamp,
            "valid": self.valid
        }


# =============================================================================
# BNO055 IMU センサー
# =============================================================================

class BNO055Registers(IntEnum):
    """BNO055 レジスタアドレス"""
    # チップ情報
    CHIP_ID = 0x00
    # ページ選択
    PAGE_ID = 0x07
    # 動作モード
    OPR_MODE = 0x3D
    PWR_MODE = 0x3E
    SYS_TRIGGER = 0x3F
    # データ出力
    EULER_H_LSB = 0x1A
    EULER_H_MSB = 0x1B
    EULER_R_LSB = 0x1C
    EULER_R_MSB = 0x1D
    EULER_P_LSB = 0x1E
    EULER_P_MSB = 0x1F
    ACCEL_DATA_X_LSB = 0x08
    GYRO_DATA_X_LSB = 0x14
    MAG_DATA_X_LSB = 0x0E
    TEMP = 0x34
    # キャリブレーション
    CALIB_STAT = 0x35


class BNO055OperationMode(IntEnum):
    """BNO055 動作モード"""
    CONFIG = 0x00
    ACCONLY = 0x01
    MAGONLY = 0x02
    GYROONLY = 0x03
    ACCMAG = 0x04
    ACCGYRO = 0x05
    MAGGYRO = 0x06
    AMG = 0x07
    IMU = 0x08
    COMPASS = 0x09
    M4G = 0x0A
    NDOF_FMC_OFF = 0x0B
    NDOF = 0x0C  # 推奨: 9軸フュージョン


class BNO055Sensor:
    """BNO055 9軸IMUセンサー"""
    
    DEFAULT_ADDRESS = 0x29  # AD0=HIGH の場合 (FaBo JetRacer標準)
    CHIP_ID_VALUE = 0xA0
    
    def __init__(self, bus: int = 7, address: int = DEFAULT_ADDRESS):
        self.bus_num = bus
        self.address = address
        self._bus = None
        self._initialized = False
        self._lock = threading.Lock()
        
    def initialize(self) -> tuple[bool, str]:
        """BNO055 初期化
        
        Returns:
            (success, message)
        """
        if not HAS_SMBUS:
            return False, "smbus not available"
            
        try:
            self._bus = smbus.SMBus(self.bus_num)
            
            # チップID確認
            try:
                chip_id = self._bus.read_byte_data(self.address, BNO055Registers.CHIP_ID)
            except Exception as e:
                return False, f"Failed to read chip ID: {e}"
                
            print(f"[BNO055] Chip ID: 0x{chip_id:02X}")
            
            if chip_id != self.CHIP_ID_VALUE:
                return False, f"Invalid chip ID: 0x{chip_id:02X} (expected 0xA0)"
            
            # CONFIG モードに設定
            self._bus.write_byte_data(self.address, BNO055Registers.OPR_MODE, BNO055OperationMode.CONFIG)
            time.sleep(0.025)
            
            # リセット
            self._bus.write_byte_data(self.address, BNO055Registers.SYS_TRIGGER, 0x20)
            time.sleep(0.65)  # リセット待ち
            
            # チップID再確認
            chip_id = self._bus.read_byte_data(self.address, BNO055Registers.CHIP_ID)
            if chip_id != self.CHIP_ID_VALUE:
                return False, f"Reset failed, chip ID: 0x{chip_id:02X}"
            
            # 通常電源モード
            self._bus.write_byte_data(self.address, BNO055Registers.PWR_MODE, 0x00)
            time.sleep(0.01)
            
            # ページ0選択
            self._bus.write_byte_data(self.address, BNO055Registers.PAGE_ID, 0x00)
            
            # NDOF モード（9軸フュージョン）
            self._bus.write_byte_data(self.address, BNO055Registers.OPR_MODE, BNO055OperationMode.NDOF)
            time.sleep(0.02)
            
            self._initialized = True
            msg = f"Initialized at address 0x{self.address:02X} (NDOF mode)"
            print(f"[BNO055] {msg}")
            return True, msg
            
        except Exception as e:
            return False, f"Initialization error: {e}"
    
    def read(self) -> IMUData:
        """IMUデータ読み取り"""
        data = IMUData(timestamp=time.time())
        
        if not self._initialized or not self._bus:
            return data
            
        try:
            with self._lock:
                # オイラー角 (6バイト)
                euler_raw = self._bus.read_i2c_block_data(
                    self.address, BNO055Registers.EULER_H_LSB, 6
                )
                # 加速度 (6バイト)
                accel_raw = self._bus.read_i2c_block_data(
                    self.address, BNO055Registers.ACCEL_DATA_X_LSB, 6
                )
                # ジャイロ (6バイト)
                gyro_raw = self._bus.read_i2c_block_data(
                    self.address, BNO055Registers.GYRO_DATA_X_LSB, 6
                )
                # 磁気 (6バイト)
                mag_raw = self._bus.read_i2c_block_data(
                    self.address, BNO055Registers.MAG_DATA_X_LSB, 6
                )
                # 温度 (1バイト)
                temp_raw = self._bus.read_byte_data(self.address, BNO055Registers.TEMP)
                # キャリブレーション (1バイト)
                calib_raw = self._bus.read_byte_data(self.address, BNO055Registers.CALIB_STAT)
            
            # オイラー角変換 (1/16 deg)
            data.heading = self._to_signed16(euler_raw[0] | (euler_raw[1] << 8)) / 16.0
            data.roll = self._to_signed16(euler_raw[2] | (euler_raw[3] << 8)) / 16.0
            data.pitch = self._to_signed16(euler_raw[4] | (euler_raw[5] << 8)) / 16.0
            
            # 加速度変換 (1/100 m/s²)
            data.accel_x = self._to_signed16(accel_raw[0] | (accel_raw[1] << 8)) / 100.0
            data.accel_y = self._to_signed16(accel_raw[2] | (accel_raw[3] << 8)) / 100.0
            data.accel_z = self._to_signed16(accel_raw[4] | (accel_raw[5] << 8)) / 100.0
            
            # ジャイロ変換 (1/16 deg/s)
            data.gyro_x = self._to_signed16(gyro_raw[0] | (gyro_raw[1] << 8)) / 16.0
            data.gyro_y = self._to_signed16(gyro_raw[2] | (gyro_raw[3] << 8)) / 16.0
            data.gyro_z = self._to_signed16(gyro_raw[4] | (gyro_raw[5] << 8)) / 16.0
            
            # 磁気変換 (1/16 μT)
            data.mag_x = self._to_signed16(mag_raw[0] | (mag_raw[1] << 8)) / 16.0
            data.mag_y = self._to_signed16(mag_raw[2] | (mag_raw[3] << 8)) / 16.0
            data.mag_z = self._to_signed16(mag_raw[4] | (mag_raw[5] << 8)) / 16.0
            
            # 温度（符号付き8bit）
            data.temperature = self._to_signed8(temp_raw)
            
            # キャリブレーション状態
            data.calib_sys = (calib_raw >> 6) & 0x03
            data.calib_gyro = (calib_raw >> 4) & 0x03
            data.calib_accel = (calib_raw >> 2) & 0x03
            data.calib_mag = calib_raw & 0x03
            
            data.valid = True
            
        except Exception as e:
            print(f"[BNO055] Read error: {e}")
            
        return data
    
    @staticmethod
    def _to_signed16(val: int) -> int:
        """16bit符号なし→符号あり変換"""
        if val >= 0x8000:
            return val - 0x10000
        return val
    
    @staticmethod
    def _to_signed8(val: int) -> int:
        """8bit符号なし→符号あり変換"""
        if val >= 0x80:
            return val - 0x100
        return val
    
    def close(self):
        """クリーンアップ"""
        if self._bus:
            self._bus.close()
            self._bus = None


# =============================================================================
# FaBo JetRacer PWM入力リーダー
# =============================================================================

class FaBoJetRacerPWMReader:
    """FaBo JetRacer PWM入力リーダー (ESP32S3 DevKit)
    
    - I2Cアドレス: 0x08
    - レジスタ 0x00: ボード情報 (12バイト)
    - レジスタ 0x01: PWMデータ (32バイト, 各4バイトBig-Endian)
    """
    
    DEFAULT_ADDRESS = 0x08
    REG_BOARD_INFO = 0x00
    REG_PWM_DATA = 0x01
    
    # PWM正規化パラメータ
    PWM_MIN = 1000  # μs
    PWM_MAX = 2000  # μs
    PWM_CENTER = 1500  # μs
    
    # モード判定閾値
    MODE_THRESHOLD = 1500  # これ以上はauto、未満はmanual
    
    def __init__(self, bus: int = 7, address: int = DEFAULT_ADDRESS):
        self.bus_num = bus
        self.address = address
        self._bus = None
        self._initialized = False
        self._lock = threading.Lock()
        self._board_revision = ""
        self._firmware_version = 0
        
    def initialize(self) -> bool:
        """初期化"""
        if not HAS_SMBUS:
            print(f"[FaBoPWM] Mock mode - no actual hardware")
            return False
            
        try:
            self._bus = smbus.SMBus(self.bus_num)
            
            # ボード情報読み取り（最初の読み取りは捨てる - ESP32S3の特性）
            try:
                self._bus.read_i2c_block_data(self.address, self.REG_BOARD_INFO, 12)
                time.sleep(0.01)
            except:
                pass
            
            # 2回目の読み取りで正しいデータを取得
            data = self._bus.read_i2c_block_data(self.address, self.REG_BOARD_INFO, 12)
            self._board_revision = f"{data[0]}.{data[1]}.{data[2]}"
            self._firmware_version = data[3]
            
            self._initialized = True
            print(f"[FaBoPWM] Initialized at address 0x{self.address:02X}")
            print(f"[FaBoPWM] Board Revision: {self._board_revision}")
            print(f"[FaBoPWM] Firmware Version: {self._firmware_version}")
            return True
            
        except Exception as e:
            print(f"[FaBoPWM] Initialization failed: {e}")
            return False
    
    def read(self) -> PWMInputData:
        """PWM入力データ読み取り"""
        data = PWMInputData(timestamp=time.time())
        data.board_revision = self._board_revision
        data.firmware_version = self._firmware_version
        
        if not self._initialized or not self._bus:
            return data
            
        try:
            with self._lock:
                # 32バイト読み取り
                raw = self._bus.read_i2c_block_data(self.address, self.REG_PWM_DATA, 32)
            
            # 各チャンネル: 4バイト Big-Endian
            data.ch1 = (raw[0] << 24) | (raw[1] << 16) | (raw[2] << 8) | raw[3]
            data.ch2 = (raw[4] << 24) | (raw[5] << 16) | (raw[6] << 8) | raw[7]
            data.ch3 = (raw[8] << 24) | (raw[9] << 16) | (raw[10] << 8) | raw[11]
            
            # 異常値チェック（プロポ未接続時は異常な値が返る）
            if not self._is_valid_pwm(data.ch1) or not self._is_valid_pwm(data.ch2):
                data.mode = "no_signal"
                data.valid = False
                return data
            
            # 正規化 (-1.0 ~ 1.0)
            data.ch1_normalized = self._normalize_pwm(data.ch1)
            data.ch2_normalized = self._normalize_pwm(data.ch2)
            data.ch3_normalized = self._normalize_pwm(data.ch3)
            
            # モード判定 (1500以上=auto, 1500未満=manual)
            if not self._is_valid_pwm(data.ch3):
                data.mode = "no_signal"  # CH3が無効な場合
            elif data.ch3 >= self.MODE_THRESHOLD:
                data.mode = "auto"
            else:
                data.mode = "manual"
            
            data.valid = True
            
        except Exception as e:
            print(f"[FaBoPWM] Read error: {e}")
            
        return data
    
    def _is_valid_pwm(self, pulse_us: int) -> bool:
        """PWM値が有効範囲内かチェック（500-2500μsを許容）"""
        return 500 <= pulse_us <= 2500
    
    def _normalize_pwm(self, pulse_us: int) -> float:
        """PWMパルス幅を正規化
        
        1000μs = -1.0 (-100%)
        1500μs = 0.0 (0%)
        2000μs = +1.0 (+100%)
        """
        # 範囲外チェック
        if pulse_us < self.PWM_MIN:
            return -1.0
        if pulse_us > self.PWM_MAX:
            return 1.0
            
        # -1.0 ~ 1.0 に正規化 (中心=1500, レンジ=500)
        return (pulse_us - self.PWM_CENTER) / 500.0
    
    def set_led_color(self, color: str) -> bool:
        """LEDカラー設定
        
        Args:
            color: red, blue, yellow, green, white, orange, magenta, lime, pink, off, normal
        """
        if not self._initialized or not self._bus:
            return False
            
        color_map = {
            "red": 0x1a,
            "blue": 0x1b,
            "yellow": 0x1c,
            "green": 0x1d,
            "white": 0x1e,
            "orange": 0x1f,
            "magenta": 0x20,
            "lime": 0x21,
            "pink": 0x22,
            "off": 0x30,
            "normal": 0x10,
        }
        
        if color not in color_map:
            print(f"[FaBoPWM] Unknown color: {color}")
            return False
            
        try:
            with self._lock:
                self._bus.read_i2c_block_data(self.address, color_map[color], 12)
            return True
        except Exception as e:
            print(f"[FaBoPWM] LED control error: {e}")
            return False
    
    def close(self):
        """クリーンアップ"""
        if self._bus:
            self._bus.close()
            self._bus = None


# =============================================================================
# VL53L7CX 距離計（プレースホルダー）
# =============================================================================

class VL53L7CXSensor:
    """VL53L7CX 8x8マルチゾーンToFセンサー
    
    注意: VL53L7CXは複雑な初期化シーケンスが必要なため、
    ST公式ライブラリ（vl53l7cx_python）の使用を推奨
    """
    
    DEFAULT_ADDRESS = 0x33
    
    def __init__(self, bus: int = 7, address: int = DEFAULT_ADDRESS):
        self.bus_num = bus
        self.address = address
        self._initialized = False
        print(f"[VL53L7CX] Note: Full implementation requires ST library")
        
    def initialize(self) -> bool:
        """初期化（未実装）"""
        print(f"[VL53L7CX] Initialization not implemented")
        print(f"[VL53L7CX] Consider using: pip install vl53l7cx")
        return False
    
    def read(self) -> DistanceData:
        """距離データ読み取り（未実装）"""
        return DistanceData(timestamp=time.time())
    
    def close(self):
        pass


# =============================================================================
# センサー統合管理
# =============================================================================

class I2CSensorManager:
    """I2Cセンサー統合管理 - FaBo JetRacer対応"""
    
    # Jetson Orin Nano: バス7, Jetson Nano: バス1
    DEFAULT_BUS_ORIN = 7
    DEFAULT_BUS_NANO = 1
    
    def __init__(self, bus: int = None):
        # バス番号自動検出
        if bus is None:
            bus = self._detect_bus()
        self.bus_num = bus
        
        self.imu: Optional[BNO055Sensor] = None
        self.pwm: Optional[FaBoJetRacerPWMReader] = None
        self.distance: Optional[VL53L7CXSensor] = None
        self._devices: Dict[str, Tuple[int, str]] = {}
        
        print(f"[I2CSensorManager] Using I2C bus {self.bus_num}")
    
    def _detect_bus(self) -> int:
        """I2Cバスを自動検出"""
        import os
        # Orin Nanoのバス7を優先チェック
        if os.path.exists("/dev/i2c-7"):
            return 7
        elif os.path.exists("/dev/i2c-1"):
            return 1
        else:
            print("[I2CSensorManager] Warning: No I2C bus detected, defaulting to 7")
            return 7
        
    def scan_devices(self) -> List[Dict]:
        """I2Cバスをスキャンしてデバイスを検出"""
        devices = []
        
        if not HAS_SMBUS:
            print("[I2C] Cannot scan - smbus not available")
            return devices
            
        try:
            bus = smbus.SMBus(self.bus_num)
            
            for addr in range(0x03, 0x78):  # 有効なI2Cアドレス範囲
                try:
                    bus.read_byte(addr)
                    device_info = self._identify_device(bus, addr)
                    devices.append({
                        "address": f"0x{addr:02X}",
                        "address_int": addr,
                        **device_info
                    })
                except:
                    pass
                    
            bus.close()
            
        except Exception as e:
            print(f"[I2C] Scan error: {e}")
            
        return devices
    
    def _identify_device(self, bus, addr: int) -> dict:
        """デバイスを識別"""
        # FaBo JetRacer PWMボード
        if addr == 0x08:
            return {"name": "FaBo JetRacer (ESP32S3)", "type": "pwm_input"}
        
        # BNO055
        if addr in [0x28, 0x29]:
            try:
                chip_id = bus.read_byte_data(addr, 0x00)
                if chip_id == 0xA0:
                    return {"name": "BNO055 9-axis IMU", "type": "imu"}
            except:
                pass
            return {"name": "Unknown (BNO055 address)", "type": "unknown"}
        
        # VL53L7CX
        if addr == 0x33:
            return {"name": "VL53L7CX ToF Sensor", "type": "distance"}
        
        # OLEDディスプレイ (SSD1306/SSD1309)
        if addr == 0x3C or addr == 0x3D:
            return {"name": "OLED Display (SSD1306)", "type": "display"}
            
        # PCA9685（サーボドライバ）
        if addr == 0x40:
            return {"name": "PCA9685 Servo Driver", "type": "servo"}
            
        # MPU6050/MPU9250
        if addr in [0x68, 0x69]:
            try:
                who_am_i = bus.read_byte_data(addr, 0x75)
                if who_am_i == 0x68:
                    return {"name": "MPU6050", "type": "imu"}
                elif who_am_i == 0x71:
                    return {"name": "MPU9250", "type": "imu"}
            except:
                pass
            return {"name": "Unknown IMU", "type": "unknown"}
                
        return {"name": "Unknown", "type": "unknown"}
    
    def initialize_imu(self, address: int = 0x29) -> tuple[bool, str]:
        """BNO055 IMU初期化"""
        self.imu = BNO055Sensor(self.bus_num, address)
        return self.imu.initialize()
    
    def initialize_pwm_input(self, address: int = 0x08) -> bool:
        """FaBo PWM入力初期化"""
        self.pwm = FaBoJetRacerPWMReader(self.bus_num, address)
        return self.pwm.initialize()
    
    def initialize_distance(self, address: int = 0x33) -> bool:
        """VL53L7CX距離計初期化"""
        self.distance = VL53L7CXSensor(self.bus_num, address)
        return self.distance.initialize()
    
    def read_imu(self) -> Optional[IMUData]:
        """IMUデータ読み取り"""
        if self.imu:
            return self.imu.read()
        return None
    
    def read_pwm(self) -> Optional[PWMInputData]:
        """PWMデータ読み取り"""
        if self.pwm:
            return self.pwm.read()
        return None
    
    def read_distance(self) -> Optional[DistanceData]:
        """距離データ読み取り"""
        if self.distance:
            return self.distance.read()
        return None
    
    def set_led_color(self, color: str) -> bool:
        """LED色設定"""
        if self.pwm:
            return self.pwm.set_led_color(color)
        return False
    
    def get_all_data(self) -> dict:
        """全センサーデータ取得"""
        result = {
            "timestamp": time.time(),
            "bus": self.bus_num,
            "imu": None,
            "pwm_input": None,
            "distance": None
        }
        
        if self.imu and self.imu._initialized:
            imu_data = self.imu.read()
            result["imu"] = imu_data.to_dict() if imu_data else None
            
        if self.pwm and self.pwm._initialized:
            pwm_data = self.pwm.read()
            result["pwm_input"] = pwm_data.to_dict() if pwm_data else None
        
        if self.distance and self.distance._initialized:
            dist_data = self.distance.read()
            result["distance"] = dist_data.to_dict() if dist_data else None
            
        return result
    
    def get_status(self) -> dict:
        """センサー状態取得"""
        return {
            "bus": self.bus_num,
            "imu": {
                "type": "BNO055",
                "initialized": self.imu is not None and self.imu._initialized,
                "address": f"0x{self.imu.address:02X}" if self.imu else None
            },
            "pwm_input": {
                "type": "FaBo JetRacer",
                "initialized": self.pwm is not None and self.pwm._initialized,
                "address": f"0x{self.pwm.address:02X}" if self.pwm else None,
                "board_revision": self.pwm._board_revision if self.pwm else None,
                "firmware_version": self.pwm._firmware_version if self.pwm else None
            },
            "distance": {
                "type": "VL53L7CX",
                "initialized": self.distance is not None and self.distance._initialized,
                "address": f"0x{self.distance.address:02X}" if self.distance else None
            }
        }
    
    def close(self):
        """全センサークリーンアップ"""
        if self.imu:
            self.imu.close()
        if self.pwm:
            self.pwm.close()
        if self.distance:
            self.distance.close()


# シングルトンインスタンス
sensor_manager = I2CSensorManager()
