"""
PWM Control Module for JetRacer
Handles PCA9685 PWM controller for steering and throttle control.
Based on FaBo JetRacer implementation.
"""
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any


class PWMControllerMock:
    """Mock PWM controller for development/testing without hardware."""
    
    def __init__(self, bus_num: int = 7, address: int = 0x40, pwm_freq: int = 60):
        self.bus_num = bus_num
        self.address = address
        self.pwm_freq = pwm_freq
        self.channels = {i: 0 for i in range(16)}
        print(f"[PWM Mock] Initialized (bus={bus_num}, addr=0x{address:02x}, freq={pwm_freq}Hz)")
    
    def set_pwm(self, channel: int, value: int):
        """Set PWM value for a channel (mock)."""
        self.channels[channel] = value
        print(f"[PWM Mock] CH{channel} = {value}")
    
    def get_pwm(self, channel: int) -> int:
        """Get current PWM value for a channel."""
        return self.channels.get(channel, 0)
    
    def close(self):
        """Close connection (mock)."""
        print("[PWM Mock] Closed")


class PWMController:
    """
    PWM Controller using PCA9685 via SMBus.
    Compatible with Jetson Orin Nano.
    """
    
    # PCA9685 Registers
    PCA9685_MODE1 = 0x00
    PCA9685_PRESCALE = 0xFE
    LED0_ON_L = 0x06
    LED0_ON_H = 0x07
    LED0_OFF_L = 0x08
    LED0_OFF_H = 0x09
    
    def __init__(self, bus_num: int = 7, address: int = 0x40, pwm_freq: int = 60):
        """
        Initialize PWM controller.
        
        Args:
            bus_num: I2C bus number (7 for Jetson Orin Nano)
            address: PCA9685 I2C address (default 0x40)
            pwm_freq: PWM frequency in Hz (60 for RC servos)
        """
        self.bus_num = bus_num
        self.address = address
        self.pwm_freq = pwm_freq
        self.bus = None
        
        try:
            import smbus
            self.bus = smbus.SMBus(bus_num)
            self._initialize_pca9685()
            print(f"[PWM] Initialized (bus={bus_num}, addr=0x{address:02x}, freq={pwm_freq}Hz)")
        except ImportError:
            raise ImportError("smbus module not found. Install with: pip install smbus")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PWM controller: {e}")
    
    def _initialize_pca9685(self):
        """Initialize PCA9685 chip."""
        # Reset PCA9685
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, 0x00)
        time.sleep(0.005)
        
        # Set PWM frequency
        prescale = int(25000000.0 / (4096.0 * self.pwm_freq) - 1)
        old_mode = self.bus.read_byte_data(self.address, self.PCA9685_MODE1)
        new_mode = (old_mode & 0x7F) | 0x10  # Sleep mode
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, new_mode)
        self.bus.write_byte_data(self.address, self.PCA9685_PRESCALE, prescale)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, self.PCA9685_MODE1, old_mode | 0xA1)
    
    def set_pwm(self, channel: int, value: int):
        """
        Set PWM value for a channel.
        
        Args:
            channel: PWM channel (0-15)
            value: PWM value (0-4095, typically 200-600 for servos)
        """
        if not 0 <= channel <= 15:
            raise ValueError(f"Channel must be 0-15, got {channel}")
        
        value = max(0, min(4095, value))
        
        reg_offset = 4 * channel
        self.bus.write_byte_data(self.address, self.LED0_ON_L + reg_offset, 0 & 0xFF)
        self.bus.write_byte_data(self.address, self.LED0_ON_H + reg_offset, 0 >> 8)
        self.bus.write_byte_data(self.address, self.LED0_OFF_L + reg_offset, value & 0xFF)
        self.bus.write_byte_data(self.address, self.LED0_OFF_H + reg_offset, value >> 8)
    
    def get_pwm(self, channel: int) -> int:
        """Get current PWM value for a channel."""
        reg_offset = 4 * channel
        low = self.bus.read_byte_data(self.address, self.LED0_OFF_L + reg_offset)
        high = self.bus.read_byte_data(self.address, self.LED0_OFF_H + reg_offset)
        return (high << 8) | low
    
    def close(self):
        """Close I2C connection."""
        if self.bus:
            self.bus.close()
            print("[PWM] Connection closed")


class JetRacerPWM:
    """
    High-level JetRacer PWM control with parameter management.
    Handles steering and throttle with calibration values.
    """
    
    STEERING_CHANNEL = 0
    THROTTLE_CHANNEL = 1
    
    DEFAULT_PARAMS = {
        "pwm_steering": {
            "left": 310,
            "center": 410,
            "right": 510
        },
        "pwm_speed": {
            "front": 430,
            "stop": 410,
            "back": 390
        }
    }
    
    def __init__(
        self,
        config_path: str = "configs/pwm_params.json",
        bus_num: int = 7,
        address: int = 0x40,
        mock: bool = False
    ):
        """
        Initialize JetRacer PWM controller.
        
        Args:
            config_path: Path to PWM parameters JSON file
            bus_num: I2C bus number
            address: PCA9685 I2C address
            mock: Use mock controller (for testing without hardware)
        """
        self.config_path = Path(config_path)
        self.mock = mock
        self._initialized = False
        
        # Initialize PWM controller
        try:
            if mock:
                self.pwm = PWMControllerMock(bus_num, address)
            else:
                self.pwm = PWMController(bus_num, address)
            self._initialized = True
        except Exception as e:
            print(f"[JetRacerPWM] Failed to initialize: {e}")
            self.pwm = None
        
        # Load or create default parameters
        self.params = self._load_params()
    
    def is_available(self) -> bool:
        """Check if PWM controller is available."""
        return self._initialized and self.pwm is not None
    
    def _load_params(self) -> Dict[str, Any]:
        """Load PWM parameters from file or use defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path) as f:
                    params = json.load(f)
                print(f"[JetRacerPWM] Loaded params from {self.config_path}")
                return params
            except Exception as e:
                print(f"[JetRacerPWM] Failed to load params: {e}")
        
        print("[JetRacerPWM] Using default parameters")
        return self.DEFAULT_PARAMS.copy()
    
    def save_params(self, params: Optional[Dict] = None):
        """Save PWM parameters to file."""
        if params is not None:
            self.params = params
        
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.params, f, indent=2, ensure_ascii=False)
        print(f"[JetRacerPWM] Saved params to {self.config_path}")
    
    def get_params(self) -> Dict[str, Any]:
        """Get current PWM parameters."""
        return self.params.copy()
    
    def get_board_info(self) -> Dict[str, Any]:
        """Get board and I2C information."""
        board_name, bus_num = detect_jetson_board()
        return {
            "board": board_name,
            "i2c_bus": bus_num,
            "pca9685_address": "0x40",
            "mock_mode": self.mock,
            "available": self.is_available()
        }
    
    # =========================================================================
    # Low-level PWM Control
    # =========================================================================
    
    def set_steering_pwm(self, value: int) -> bool:
        """Set raw PWM value for steering."""
        if not self.is_available():
            return False
        self.pwm.set_pwm(self.STEERING_CHANNEL, value)
        return True
    
    def set_throttle_pwm(self, value: int) -> bool:
        """Set raw PWM value for throttle."""
        if not self.is_available():
            return False
        self.pwm.set_pwm(self.THROTTLE_CHANNEL, value)
        return True
    
    # =========================================================================
    # Calibration Methods
    # =========================================================================
    
    def test_steering_center(self) -> Dict[str, Any]:
        """Set steering to center position."""
        center = self.params["pwm_steering"]["center"]
        success = self.set_steering_pwm(center)
        return {"success": success, "value": center, "position": "center"}
    
    def test_steering_left(self) -> Dict[str, Any]:
        """Set steering to full left position."""
        left = self.params["pwm_steering"]["left"]
        success = self.set_steering_pwm(left)
        return {"success": success, "value": left, "position": "left"}
    
    def test_steering_right(self) -> Dict[str, Any]:
        """Set steering to full right position."""
        right = self.params["pwm_steering"]["right"]
        success = self.set_steering_pwm(right)
        return {"success": success, "value": right, "position": "right"}
    
    def test_steering_value(self, value: int) -> Dict[str, Any]:
        """Test arbitrary steering PWM value."""
        success = self.set_steering_pwm(value)
        return {"success": success, "value": value, "channel": "steering"}
    
    def test_throttle_stop(self) -> Dict[str, Any]:
        """Set throttle to stop position."""
        stop = self.params["pwm_speed"]["stop"]
        success = self.set_throttle_pwm(stop)
        return {"success": success, "value": stop, "position": "stop"}
    
    def test_throttle_forward(self) -> Dict[str, Any]:
        """Set throttle to forward position."""
        front = self.params["pwm_speed"]["front"]
        success = self.set_throttle_pwm(front)
        return {"success": success, "value": front, "position": "forward"}
    
    def test_throttle_backward(self) -> Dict[str, Any]:
        """Set throttle to backward position (with ESC sequence)."""
        if not self.is_available():
            return {"success": False, "error": "PWM not available"}
        
        stop = self.params["pwm_speed"]["stop"]
        back = self.params["pwm_speed"]["back"]
        
        # ESC reverse sequence
        self.set_throttle_pwm(stop)
        time.sleep(0.1)
        self.set_throttle_pwm(stop)
        time.sleep(0.1)
        self.set_throttle_pwm(back)
        
        return {"success": True, "value": back, "position": "backward"}
    
    def test_throttle_value(self, value: int) -> Dict[str, Any]:
        """Test arbitrary throttle PWM value."""
        success = self.set_throttle_pwm(value)
        return {"success": success, "value": value, "channel": "throttle"}
    
    def test_steering_range(self) -> Dict[str, Any]:
        """Test full steering range (left -> right -> center)."""
        if not self.is_available():
            return {"success": False, "error": "PWM not available"}
        
        left = self.params["pwm_steering"]["left"]
        right = self.params["pwm_steering"]["right"]
        center = self.params["pwm_steering"]["center"]
        
        self.set_steering_pwm(left)
        time.sleep(1.0)
        self.set_steering_pwm(right)
        time.sleep(1.0)
        self.set_steering_pwm(center)
        
        return {
            "success": True,
            "sequence": ["left", "right", "center"],
            "values": [left, right, center]
        }
    
    # =========================================================================
    # High-level Control (-1.0 to 1.0)
    # =========================================================================
    
    def set_steering(self, value: float) -> bool:
        """Set steering position (-1.0 = left, 0.0 = center, 1.0 = right)."""
        if not self.is_available():
            return False
        
        value = max(-1.0, min(1.0, value))
        
        center = self.params["pwm_steering"]["center"]
        left = self.params["pwm_steering"]["left"]
        right = self.params["pwm_steering"]["right"]
        
        if value < 0:
            pwm_value = int(center + value * (center - left))
        else:
            pwm_value = int(center + value * (right - center))
        
        return self.set_steering_pwm(pwm_value)
    
    def set_throttle(self, value: float) -> bool:
        """Set throttle position (-1.0 = reverse, 0.0 = stop, 1.0 = forward)."""
        if not self.is_available():
            return False
        
        value = max(-1.0, min(1.0, value))
        
        stop = self.params["pwm_speed"]["stop"]
        front = self.params["pwm_speed"]["front"]
        back = self.params["pwm_speed"]["back"]
        
        if abs(value) < 0.05:
            pwm_value = stop
        elif value > 0:
            pwm_value = int(stop + value * (front - stop))
        else:
            pwm_value = int(stop + value * (stop - back))
        
        return self.set_throttle_pwm(pwm_value)
    
    def stop(self) -> bool:
        """Stop all movement (emergency stop)."""
        if not self.is_available():
            return False
        
        self.set_steering(0.0)
        self.set_throttle(0.0)
        print("[JetRacerPWM] Emergency stop")
        return True
    
    def close(self):
        """Clean shutdown."""
        if self.is_available():
            self.stop()
            self.pwm.close()


def detect_jetson_board() -> tuple:
    """
    Detect Jetson board and return appropriate I2C bus number.
    
    Returns:
        Tuple of (board_name, bus_number)
    """
    try:
        import Jetson.GPIO as GPIO
        board_name = GPIO.gpio_pin_data.get_data()[0]
    except Exception:
        board_name = "UNKNOWN"
    
    bus_map = {
        "JETSON_NX": 8,
        "JETSON_XAVIER": 8,
        "JETSON_NANO": 1,
        "JETSON_ORIN": 7,
        "JETSON_ORIN_NANO": 7,
    }
    
    bus_num = bus_map.get(board_name, 7)  # Default to 7 for Orin
    
    return board_name, bus_num


# Singleton instance
_pwm_controller: Optional[JetRacerPWM] = None


def get_pwm_controller(mock: bool = None) -> JetRacerPWM:
    """
    Get singleton PWM controller instance.
    
    Args:
        mock: Force mock mode (None = auto-detect based on hardware)
    """
    global _pwm_controller
    if _pwm_controller is None:
        # Determine config path relative to this file
        config_path = Path(__file__).parent.parent.parent / "configs" / "pwm_params.json"
        
        # Auto-detect mock mode if not specified
        if mock is None:
            try:
                import smbus
                mock = False
            except ImportError:
                print("[PWM] smbus not available, using mock mode")
                mock = True
        
        _pwm_controller = JetRacerPWM(
            config_path=str(config_path),
            mock=mock
        )
    return _pwm_controller
