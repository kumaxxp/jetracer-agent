"""センサー情報取得モジュール - GStreamer経由でカメラ capabilities を取得"""
import subprocess
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import time


@dataclass
class SensorMode:
    """センサーモード情報"""
    mode_id: int
    width: int
    height: int
    fps: float
    analog_gain_min: float
    analog_gain_max: float
    exposure_min_ns: int
    exposure_max_ns: int
    
    @property
    def exposure_min_us(self) -> float:
        """露出時間（マイクロ秒）"""
        return self.exposure_min_ns / 1000
    
    @property
    def exposure_max_ms(self) -> float:
        """露出時間（ミリ秒）"""
        return self.exposure_max_ns / 1_000_000
    
    def to_dict(self) -> dict:
        """辞書に変換"""
        return {
            "mode_id": self.mode_id,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "analog_gain": {
                "min": self.analog_gain_min,
                "max": self.analog_gain_max
            },
            "exposure": {
                "min_us": round(self.exposure_min_us, 1),
                "max_ms": round(self.exposure_max_ms, 1)
            },
            "label": f"{self.width}×{self.height} @ {self.fps:.0f}fps"
        }


@dataclass
class OutputResolution:
    """出力解像度オプション"""
    width: int
    height: int
    label: str
    
    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "label": self.label
        }


# デフォルト出力解像度オプション
DEFAULT_OUTPUT_RESOLUTIONS = [
    OutputResolution(640, 480, "640×480 (VGA)"),
    OutputResolution(480, 360, "480×360 (Preview)"),
    OutputResolution(320, 240, "320×240 (QVGA)"),
    OutputResolution(224, 224, "224×224 (ResNet)"),
]


class SensorCapabilities:
    """センサー capabilities 管理クラス"""
    
    def __init__(self):
        self._capabilities: Dict[int, List[SensorMode]] = {}
        self._sensor_names: Dict[int, str] = {}
        self._lock = threading.Lock()
        self._initialized = False
    
    def probe_sensor(self, camera_id: int = 0, timeout: float = 5.0) -> List[SensorMode]:
        """指定カメラのセンサーモードを取得
        
        GStreamerパイプラインを一時的に起動してログからモード情報をパース
        """
        print(f"[SensorCapabilities] Probing camera {camera_id}...")
        
        # 短時間だけパイプラインを起動してモード情報を取得
        pipeline_cmd = [
            "gst-launch-1.0",
            f"nvarguscamerasrc sensor-id={camera_id} num-buffers=1",
            "!", "fakesink"
        ]
        
        try:
            # stderrにセンサー情報が出力される
            result = subprocess.run(
                " ".join(pipeline_cmd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            # stdout と stderr 両方をチェック
            output = result.stdout + result.stderr
            
            modes = self._parse_sensor_modes(output)
            
            if modes:
                with self._lock:
                    self._capabilities[camera_id] = modes
                    self._sensor_names[camera_id] = self._detect_sensor_name(output)
                print(f"[SensorCapabilities] Camera {camera_id}: Found {len(modes)} modes")
            else:
                print(f"[SensorCapabilities] Camera {camera_id}: No modes found, using defaults")
                modes = self._get_default_modes()
                with self._lock:
                    self._capabilities[camera_id] = modes
                    self._sensor_names[camera_id] = "Unknown"
            
            return modes
            
        except subprocess.TimeoutExpired:
            print(f"[SensorCapabilities] Camera {camera_id}: Probe timeout")
            return self._get_default_modes()
        except Exception as e:
            print(f"[SensorCapabilities] Camera {camera_id}: Probe error: {e}")
            return self._get_default_modes()
    
    def _parse_sensor_modes(self, output: str) -> List[SensorMode]:
        """GStreamerログからセンサーモード情報をパース
        
        例:
        GST_ARGUS: 3280 x 2464 FR = 21.000000 fps Duration = 47619048 ; Analog Gain range min 1.000000, max 10.625000; Exposure Range min 13000, max 683709000;
        """
        modes = []
        
        # パターン: width x height FR = fps fps ... Analog Gain range min X, max Y; Exposure Range min A, max B;
        pattern = r'(\d+)\s*x\s*(\d+)\s+FR\s*=\s*([\d.]+)\s*fps.*?Analog Gain range min\s*([\d.]+),\s*max\s*([\d.]+).*?Exposure Range min\s*(\d+),\s*max\s*(\d+)'
        
        matches = re.findall(pattern, output, re.IGNORECASE)
        
        for i, match in enumerate(matches):
            try:
                mode = SensorMode(
                    mode_id=i,
                    width=int(match[0]),
                    height=int(match[1]),
                    fps=float(match[2]),
                    analog_gain_min=float(match[3]),
                    analog_gain_max=float(match[4]),
                    exposure_min_ns=int(match[5]),
                    exposure_max_ns=int(match[6])
                )
                modes.append(mode)
                print(f"[SensorCapabilities] Mode {i}: {mode.width}x{mode.height}@{mode.fps}fps")
            except (ValueError, IndexError) as e:
                print(f"[SensorCapabilities] Parse error for match {i}: {e}")
                continue
        
        return modes
    
    def _detect_sensor_name(self, output: str) -> str:
        """センサー名を検出（可能であれば）"""
        # IMX219, IMX477 などの検出を試みる
        if "imx219" in output.lower():
            return "IMX219"
        elif "imx477" in output.lower():
            return "IMX477"
        elif "imx708" in output.lower():
            return "IMX708"
        
        # 解像度からの推測
        if "3280" in output and "2464" in output:
            return "IMX219 (8MP)"
        elif "4032" in output and "3040" in output:
            return "IMX477 (12MP)"
        
        return "CSI Camera"
    
    def _get_default_modes(self) -> List[SensorMode]:
        """デフォルトのIMX219モード（プローブ失敗時用）"""
        return [
            SensorMode(0, 3280, 2464, 21.0, 1.0, 10.625, 13000, 683709000),
            SensorMode(1, 3280, 1848, 28.0, 1.0, 10.625, 13000, 683709000),
            SensorMode(2, 1920, 1080, 30.0, 1.0, 10.625, 13000, 683709000),
            SensorMode(3, 1640, 1232, 30.0, 1.0, 10.625, 13000, 683709000),
            SensorMode(4, 1280, 720, 60.0, 1.0, 10.625, 13000, 683709000),
        ]
    
    def initialize(self, camera_ids: List[int] = None) -> Dict[int, bool]:
        """複数カメラの capabilities を初期化
        
        Args:
            camera_ids: プローブするカメラIDリスト（デフォルト: [0, 1]）
        
        Returns:
            各カメラの成功/失敗
        """
        if camera_ids is None:
            camera_ids = [0, 1]
        
        results = {}
        for cid in camera_ids:
            modes = self.probe_sensor(cid)
            results[cid] = len(modes) > 0
        
        self._initialized = True
        return results
    
    def get_modes(self, camera_id: int = 0) -> List[SensorMode]:
        """指定カメラのセンサーモード一覧を取得"""
        with self._lock:
            if camera_id not in self._capabilities:
                # 未取得の場合はプローブ
                return self.probe_sensor(camera_id)
            return self._capabilities[camera_id]
    
    def get_mode(self, camera_id: int, mode_id: int) -> Optional[SensorMode]:
        """特定のモードを取得"""
        modes = self.get_modes(camera_id)
        for mode in modes:
            if mode.mode_id == mode_id:
                return mode
        return None
    
    def get_sensor_name(self, camera_id: int = 0) -> str:
        """センサー名を取得"""
        with self._lock:
            return self._sensor_names.get(camera_id, "Unknown")
    
    def get_capabilities(self, camera_id: int = 0) -> dict:
        """APIレスポンス用の capabilities 辞書を取得"""
        modes = self.get_modes(camera_id)
        
        return {
            "camera_id": camera_id,
            "sensor_name": self.get_sensor_name(camera_id),
            "modes": [m.to_dict() for m in modes],
            "output_resolutions": [r.to_dict() for r in DEFAULT_OUTPUT_RESOLUTIONS],
            "current_mode": None  # カメラマネージャーから取得する必要あり
        }
    
    def get_all_capabilities(self) -> dict:
        """全カメラの capabilities を取得"""
        with self._lock:
            return {
                "cameras": {
                    cid: self.get_capabilities(cid) 
                    for cid in self._capabilities.keys()
                },
                "initialized": self._initialized
            }


# シングルトンインスタンス
sensor_capabilities = SensorCapabilities()
