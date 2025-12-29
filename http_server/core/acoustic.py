"""
音響状態推定のコアモジュール（Jetson側）

機能:
- USBマイクからのリアルタイム音声取得
- MFCC特徴量抽出
- モーター/サーボ状態の推定（将来）
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List
from enum import Enum
import threading
import time
import logging

logger = logging.getLogger(__name__)

# PyAudio/librosaの遅延インポート
_pyaudio = None
_librosa = None


def _get_pyaudio():
    """PyAudioの遅延インポート"""
    global _pyaudio
    if _pyaudio is None:
        import pyaudio
        _pyaudio = pyaudio
    return _pyaudio


def _get_librosa():
    """librosaの遅延インポート"""
    global _librosa
    if _librosa is None:
        import librosa
        _librosa = librosa
    return _librosa


class MotorState(str, Enum):
    STOPPED = "stopped"
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    SPINNING = "spinning"
    STALLED = "stalled"


class ServoState(str, Enum):
    IDLE = "idle"
    MOVING = "moving"
    END_STOP = "end_stop"


@dataclass
class AudioFeatures:
    """音響特徴量"""
    mfcc_mean: List[float]       # 13次元
    mfcc_std: List[float]        # 13次元
    delta_mfcc_mean: List[float] # 13次元
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    rms_energy: float
    zcr: float
    timestamp: float

    def to_vector(self) -> np.ndarray:
        """全特徴量を1次元ベクトルに結合（約50次元）"""
        return np.concatenate([
            self.mfcc_mean,
            self.mfcc_std,
            self.delta_mfcc_mean,
            [self.spectral_centroid, self.spectral_rolloff,
             self.spectral_bandwidth, self.rms_energy, self.zcr]
        ])


@dataclass
class AcousticState:
    """音響状態推定結果"""
    motor_state: str
    servo_state: str
    speed_estimate: float
    confidence: float
    features: Optional[AudioFeatures]
    timestamp: float


class AcousticCapture:
    """USBマイクからの音声キャプチャ"""

    def __init__(
        self,
        sample_rate: int = 48000,  # USB mics typically support 48kHz
        chunk_size: int = 4800,  # 100ms @ 48kHz
        buffer_seconds: float = 1.0,
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.buffer_size = int(buffer_seconds * sample_rate / chunk_size)
        self.device_index = device_index

        self.buffer: deque = deque(maxlen=self.buffer_size)
        self.pa = None
        self.stream = None
        self.is_running = False
        self._lock = threading.Lock()

    def start(self):
        """録音開始"""
        if self.is_running:
            return

        pyaudio = _get_pyaudio()
        self.pa = pyaudio.PyAudio()

        # デバイス自動検出
        if self.device_index is None:
            self.device_index = self._find_usb_microphone()

        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )

        self.is_running = True
        logger.info(f"Audio capture started (device: {self.device_index})")

    def stop(self):
        """録音停止"""
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        if self.pa:
            self.pa.terminate()
            self.pa = None
        logger.info("Audio capture stopped")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudioコールバック"""
        pyaudio = _get_pyaudio()
        audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self.buffer.append(audio)
        return (None, pyaudio.paContinue)

    def get_latest_chunk(self) -> Optional[np.ndarray]:
        """最新の100msチャンクを取得"""
        with self._lock:
            if len(self.buffer) > 0:
                return self.buffer[-1].copy()
        return None

    def get_buffer(self) -> np.ndarray:
        """バッファ全体（直近1秒）を取得"""
        with self._lock:
            if len(self.buffer) > 0:
                return np.concatenate(list(self.buffer))
        return np.array([])

    def _find_usb_microphone(self) -> Optional[int]:
        """USBマイクのデバイスインデックスを自動検出"""
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                name = info['name'].lower()
                if 'usb' in name or 'microphone' in name:
                    logger.info(f"Found USB microphone: {info['name']} (index: {i})")
                    return i
        # 見つからない場合はデフォルト入力デバイス
        default_info = self.pa.get_default_input_device_info()
        return default_info['index'] if default_info else None

    @staticmethod
    def list_devices() -> List[dict]:
        """利用可能なオーディオデバイス一覧"""
        pyaudio = _get_pyaudio()
        pa = pyaudio.PyAudio()
        devices = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                devices.append({
                    'index': i,
                    'name': info['name'],
                    'channels': info['maxInputChannels'],
                    'sample_rate': int(info['defaultSampleRate'])
                })
        pa.terminate()
        return devices


class AcousticFeatureExtractor:
    """MFCC等の特徴量抽出"""

    def __init__(
        self,
        sample_rate: int = 48000,  # Match capture sample rate
        n_mfcc: int = 13,
        n_fft: int = 2048,  # Larger FFT for 48kHz
        hop_length: int = 512
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> AudioFeatures:
        """音声チャンクから特徴量を抽出"""
        librosa = _get_librosa()

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio, sr=self.sample_rate,
            n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        delta_mfcc = librosa.feature.delta(mfcc)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length
        )

        # Energy features
        rms = librosa.feature.rms(y=audio, frame_length=self.n_fft, hop_length=self.hop_length)
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=self.n_fft, hop_length=self.hop_length)

        return AudioFeatures(
            mfcc_mean=mfcc.mean(axis=1).tolist(),
            mfcc_std=mfcc.std(axis=1).tolist(),
            delta_mfcc_mean=delta_mfcc.mean(axis=1).tolist(),
            spectral_centroid=float(centroid.mean()),
            spectral_rolloff=float(rolloff.mean()),
            spectral_bandwidth=float(bandwidth.mean()),
            rms_energy=float(rms.mean()),
            zcr=float(zcr.mean()),
            timestamp=time.time()
        )


class AcousticManager:
    """音響処理の統合管理クラス"""

    _instance: Optional['AcousticManager'] = None

    def __init__(self):
        self.capture = AcousticCapture()
        self.extractor = AcousticFeatureExtractor()
        self.is_recording = False
        self.recorded_data: List[dict] = []
        self.recording_session: str = ""

    @classmethod
    def get_instance(cls) -> 'AcousticManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self):
        """音響キャプチャ開始"""
        self.capture.start()

    def stop(self):
        """音響キャプチャ停止"""
        self.capture.stop()

    def is_capturing(self) -> bool:
        """キャプチャ中かどうか"""
        return self.capture.is_running

    def get_current_features(self) -> Optional[AudioFeatures]:
        """現在の音響特徴量を取得"""
        chunk = self.capture.get_latest_chunk()
        if chunk is not None and len(chunk) > 0:
            return self.extractor.extract(chunk)
        return None

    def get_current_state(self) -> AcousticState:
        """現在の音響状態を取得（将来的に推論を追加）"""
        features = self.get_current_features()

        # TODO: 学習済みモデルで状態推定
        # 現在はダミー値
        return AcousticState(
            motor_state=MotorState.STOPPED.value,
            servo_state=ServoState.IDLE.value,
            speed_estimate=0.0,
            confidence=0.0,
            features=features,
            timestamp=time.time()
        )

    def start_recording(self, session_name: str) -> dict:
        """録音セッション開始"""
        self.is_recording = True
        self.recorded_data = []
        self.recording_session = session_name
        return {"status": "recording_started", "session": session_name}

    def stop_recording(self) -> dict:
        """録音セッション停止"""
        self.is_recording = False
        return {
            "status": "recording_stopped",
            "session": self.recording_session,
            "frames": len(self.recorded_data)
        }

    def record_frame(self, motor_state: str, servo_state: str,
                     pwm_throttle: float, pwm_steering: float) -> dict:
        """現在のフレームを記録"""
        if not self.is_recording:
            return {"error": "Not recording"}

        features = self.get_current_features()
        if features is None:
            return {"error": "No audio data"}

        frame = {
            "features": asdict(features),
            "labels": {
                "motor_state": motor_state,
                "servo_state": servo_state,
                "pwm_throttle": pwm_throttle,
                "pwm_steering": pwm_steering
            }
        }
        self.recorded_data.append(frame)
        return {"status": "recorded", "frame_count": len(self.recorded_data)}

    def get_recorded_data(self) -> List[dict]:
        """録音データ取得"""
        return self.recorded_data

    def list_devices(self) -> List[dict]:
        """オーディオデバイス一覧"""
        return AcousticCapture.list_devices()
