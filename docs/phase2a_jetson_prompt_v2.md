# YANA Phase 2-A 音響実験プロンプト（Jetson側）v2

**実行環境**: Jetson Orin Nano Super
**リポジトリ**: `~/jetracer-agent`

---

## 概要

車輪を浮かせた状態で、モーター音を3つのシナリオで録音する。
PWM値は自動制御され、音響特徴量とPWM値がペアで記録される。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│ Jetson (jetracer-agent)                                     │
│                                                             │
│  http_server/                                               │
│  ├── core/                                                  │
│  │   ├── acoustic.py         # 音響キャプチャ・特徴抽出     │
│  │   └── scenario.py         # シナリオ実行エンジン         │
│  └── routes/                                                │
│      └── acoustic.py         # APIエンドポイント            │
│                                                             │
│  既存モジュール（利用）:                                     │
│  ├── core/motor.py           # PWM制御                      │
│  └── core/servo.py           # サーボ制御                   │
└─────────────────────────────────────────────────────────────┘
```

---

## シナリオ定義

### 1. throttle_stop（停止）
- **目的**: 環境音のベースライン取得
- **PWM**: throttle=0, steering=0（固定）
- **時間**: 5秒
- **サンプリング**: 10Hz（50サンプル）

### 2. throttle_starting（始動）
- **目的**: 静止→回転開始の遷移を捉える
- **PWM**: throttle=0→0.5（徐々に上げる）, steering=0（固定）
- **時間**: 10秒
- **サンプリング**: 10Hz（100サンプル）
- **PWM変化**: 0.05/秒 で増加

### 3. throttle_slow（低速走行）
- **目的**: 安定回転時の音を取得
- **PWM**: throttle=0.2（車輪が回る最低限）, steering=0（固定）
- **時間**: 5秒
- **サンプリング**: 10Hz（50サンプル）

---

## 実装タスク

### 1. ディレクトリ構造

```
jetracer-agent/
└── http_server/
    ├── core/
    │   ├── acoustic.py      # 新規: 音響処理
    │   ├── scenario.py      # 新規: シナリオ実行
    │   └── ... (既存)
    └── routes/
        ├── acoustic.py      # 新規: APIエンドポイント
        └── ... (既存)
```

### 2. http_server/core/acoustic.py

```python
"""
音響キャプチャと特徴抽出（Jetson側）
"""

import pyaudio
import numpy as np
import librosa
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional, List
import threading
import time
import logging

logger = logging.getLogger(__name__)


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
    
    def to_dict(self) -> dict:
        return asdict(self)


class AudioCapture:
    """USBマイクからの音声キャプチャ"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        device_index: Optional[int] = None
    ):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.device_index = device_index
        
        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None
        self.is_running = False
        
        self.latest_chunk: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        
    def start(self):
        """録音開始"""
        if self.is_running:
            return
            
        self.pa = pyaudio.PyAudio()
        
        if self.device_index is None:
            self.device_index = self._find_usb_microphone()
            
        self.stream = self.pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._callback
        )
        
        self.is_running = True
        logger.info(f"AudioCapture started (device: {self.device_index})")
        
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
        logger.info("AudioCapture stopped")
        
    def _callback(self, in_data, frame_count, time_info, status):
        """PyAudioコールバック"""
        audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self.latest_chunk = audio
        return (None, pyaudio.paContinue)
    
    def get_chunk(self) -> Optional[np.ndarray]:
        """最新の音声チャンクを取得"""
        with self._lock:
            if self.latest_chunk is not None:
                return self.latest_chunk.copy()
        return None
    
    def _find_usb_microphone(self) -> int:
        """USBマイクを自動検出"""
        for i in range(self.pa.get_device_count()):
            info = self.pa.get_device_info_by_index(i)
            if info['maxInputChannels'] > 0:
                name = info['name'].lower()
                if 'usb' in name or 'microphone' in name:
                    logger.info(f"Found USB mic: {info['name']} (index: {i})")
                    return i
        # デフォルト
        default = self.pa.get_default_input_device_info()
        return default['index']
    
    @staticmethod
    def list_devices() -> List[dict]:
        """利用可能なオーディオデバイス一覧"""
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


class AudioFeatureExtractor:
    """MFCC等の特徴量抽出"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 256
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract(self, audio: np.ndarray) -> AudioFeatures:
        """音声チャンクから特徴量を抽出"""
        if len(audio) < self.n_fft:
            # パディング
            audio = np.pad(audio, (0, self.n_fft - len(audio)))
        
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
            zcr=float(zcr.mean())
        )


class AcousticManager:
    """音響処理の統合管理（シングルトン）"""
    
    _instance: Optional['AcousticManager'] = None
    
    def __init__(self):
        self.capture = AudioCapture()
        self.extractor = AudioFeatureExtractor()
        
    @classmethod
    def get_instance(cls) -> 'AcousticManager':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def start(self):
        """キャプチャ開始"""
        self.capture.start()
        
    def stop(self):
        """キャプチャ停止"""
        self.capture.stop()
        
    def is_running(self) -> bool:
        """キャプチャ中か"""
        return self.capture.is_running
        
    def get_features(self) -> Optional[AudioFeatures]:
        """現在の音響特徴量を取得"""
        chunk = self.capture.get_chunk()
        if chunk is not None and len(chunk) > 0:
            return self.extractor.extract(chunk)
        return None
    
    def list_devices(self) -> List[dict]:
        """オーディオデバイス一覧"""
        return AudioCapture.list_devices()
```

### 3. http_server/core/scenario.py

```python
"""
シナリオ実行エンジン（Jetson側）

3つのシナリオを自動実行し、PWM値と音響特徴量を記録する。
"""

import asyncio
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable
from enum import Enum
import logging

from .acoustic import AcousticManager, AudioFeatures

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    THROTTLE_STOP = "throttle_stop"
    THROTTLE_STARTING = "throttle_starting"
    THROTTLE_SLOW = "throttle_slow"


@dataclass
class ScenarioConfig:
    """シナリオ設定"""
    name: ScenarioType
    duration: float          # 秒
    sample_rate: int         # Hz
    throttle_start: float    # 開始時のスロットル
    throttle_end: float      # 終了時のスロットル
    steering: float          # ステアリング（固定）
    
    @property
    def total_samples(self) -> int:
        return int(self.duration * self.sample_rate)


# シナリオ定義
SCENARIOS = {
    ScenarioType.THROTTLE_STOP: ScenarioConfig(
        name=ScenarioType.THROTTLE_STOP,
        duration=5.0,
        sample_rate=10,
        throttle_start=0.0,
        throttle_end=0.0,
        steering=0.0
    ),
    ScenarioType.THROTTLE_STARTING: ScenarioConfig(
        name=ScenarioType.THROTTLE_STARTING,
        duration=10.0,
        sample_rate=10,
        throttle_start=0.0,
        throttle_end=0.5,
        steering=0.0
    ),
    ScenarioType.THROTTLE_SLOW: ScenarioConfig(
        name=ScenarioType.THROTTLE_SLOW,
        duration=5.0,
        sample_rate=10,
        throttle_start=0.2,
        throttle_end=0.2,
        steering=0.0
    ),
}


@dataclass
class SampleData:
    """1サンプルのデータ"""
    timestamp: float
    elapsed: float           # シナリオ開始からの経過時間
    pwm_throttle: float
    pwm_steering: float
    audio_features: Optional[dict]
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioResult:
    """シナリオ実行結果"""
    scenario: str
    config: dict
    start_time: float
    end_time: float
    samples: List[dict]
    
    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sample_count": len(self.samples),
            "samples": self.samples
        }


class ScenarioExecutor:
    """シナリオ実行エンジン"""
    
    def __init__(
        self,
        set_throttle_func: Callable[[float], None],
        set_steering_func: Callable[[float], None]
    ):
        """
        Args:
            set_throttle_func: スロットル設定関数 (値: -1.0 ~ 1.0)
            set_steering_func: ステアリング設定関数 (値: -1.0 ~ 1.0)
        """
        self.set_throttle = set_throttle_func
        self.set_steering = set_steering_func
        self.acoustic = AcousticManager.get_instance()
        
        self.is_running = False
        self.current_scenario: Optional[ScenarioType] = None
        self.progress = 0.0
        self.current_throttle = 0.0
        self.current_steering = 0.0
        self.samples: List[SampleData] = []
        self.result: Optional[ScenarioResult] = None
        
    def get_status(self) -> dict:
        """現在の状態を取得"""
        return {
            "running": self.is_running,
            "scenario": self.current_scenario.value if self.current_scenario else None,
            "progress": self.progress,
            "current_pwm": {
                "throttle": self.current_throttle,
                "steering": self.current_steering
            },
            "samples_collected": len(self.samples)
        }
    
    async def execute(self, scenario_type: ScenarioType) -> ScenarioResult:
        """シナリオを実行"""
        if self.is_running:
            raise RuntimeError("Another scenario is running")
            
        config = SCENARIOS[scenario_type]
        logger.info(f"Starting scenario: {scenario_type.value}")
        
        self.is_running = True
        self.current_scenario = scenario_type
        self.progress = 0.0
        self.samples = []
        
        # 音響キャプチャ開始
        if not self.acoustic.is_running():
            self.acoustic.start()
            await asyncio.sleep(0.5)  # 安定待ち
        
        start_time = time.time()
        sample_interval = 1.0 / config.sample_rate
        
        try:
            # 初期PWM設定
            self.current_throttle = config.throttle_start
            self.current_steering = config.steering
            self.set_throttle(self.current_throttle)
            self.set_steering(self.current_steering)
            
            for i in range(config.total_samples):
                if not self.is_running:
                    break
                    
                sample_start = time.time()
                elapsed = sample_start - start_time
                
                # PWM値を計算（線形補間）
                if config.throttle_start != config.throttle_end:
                    t = elapsed / config.duration
                    self.current_throttle = (
                        config.throttle_start + 
                        (config.throttle_end - config.throttle_start) * t
                    )
                    self.set_throttle(self.current_throttle)
                
                # 音響特徴量を取得
                features = self.acoustic.get_features()
                
                # サンプル記録
                sample = SampleData(
                    timestamp=sample_start,
                    elapsed=elapsed,
                    pwm_throttle=self.current_throttle,
                    pwm_steering=self.current_steering,
                    audio_features=features.to_dict() if features else None
                )
                self.samples.append(sample)
                
                # 進捗更新
                self.progress = (i + 1) / config.total_samples
                
                # 次のサンプルまで待機
                processing_time = time.time() - sample_start
                sleep_time = sample_interval - processing_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    
        finally:
            # PWMをゼロに戻す
            self.set_throttle(0.0)
            self.set_steering(0.0)
            self.current_throttle = 0.0
            self.current_steering = 0.0
            
        end_time = time.time()
        
        # 結果を保存
        self.result = ScenarioResult(
            scenario=scenario_type.value,
            config=asdict(config),
            start_time=start_time,
            end_time=end_time,
            samples=[s.to_dict() for s in self.samples]
        )
        
        self.is_running = False
        self.progress = 1.0
        logger.info(f"Scenario completed: {len(self.samples)} samples")
        
        return self.result
    
    def stop(self):
        """シナリオを中断"""
        self.is_running = False
        self.set_throttle(0.0)
        self.set_steering(0.0)
        
    def get_result(self) -> Optional[ScenarioResult]:
        """最後の実行結果を取得"""
        return self.result


# グローバルインスタンス（main.pyで初期化）
_executor: Optional[ScenarioExecutor] = None


def init_executor(set_throttle: Callable, set_steering: Callable):
    """エグゼキュータを初期化"""
    global _executor
    _executor = ScenarioExecutor(set_throttle, set_steering)
    return _executor


def get_executor() -> ScenarioExecutor:
    """エグゼキュータを取得"""
    if _executor is None:
        raise RuntimeError("ScenarioExecutor not initialized. Call init_executor first.")
    return _executor
```

### 4. http_server/routes/acoustic.py

```python
"""
音響APIエンドポイント（Jetson側）

エンドポイント:
- GET  /acoustic/devices           - オーディオデバイス一覧
- POST /acoustic/capture/start     - キャプチャ開始
- POST /acoustic/capture/stop      - キャプチャ停止
- GET  /acoustic/features          - 現在の特徴量

- GET  /acoustic/scenarios         - 利用可能なシナリオ一覧
- POST /acoustic/scenario/start    - シナリオ実行開始
- POST /acoustic/scenario/stop     - シナリオ中断
- GET  /acoustic/scenario/status   - シナリオ実行状態
- GET  /acoustic/scenario/result   - シナリオ実行結果
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional

from ..core.acoustic import AcousticManager
from ..core.scenario import (
    ScenarioType, SCENARIOS, 
    get_executor, ScenarioResult
)

router = APIRouter(prefix="/acoustic", tags=["acoustic"])


# === リクエストモデル ===

class ScenarioStartRequest(BaseModel):
    scenario: str  # "throttle_stop", "throttle_starting", "throttle_slow"


# === デバイス管理 ===

@router.get("/devices")
async def list_devices():
    """オーディオデバイス一覧"""
    manager = AcousticManager.get_instance()
    return {"devices": manager.list_devices()}


# === キャプチャ制御 ===

@router.post("/capture/start")
async def start_capture():
    """音響キャプチャ開始"""
    try:
        manager = AcousticManager.get_instance()
        manager.start()
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/capture/stop")
async def stop_capture():
    """音響キャプチャ停止"""
    manager = AcousticManager.get_instance()
    manager.stop()
    return {"status": "stopped"}


@router.get("/features")
async def get_features():
    """現在の音響特徴量を取得"""
    manager = AcousticManager.get_instance()
    if not manager.is_running():
        raise HTTPException(status_code=400, detail="Capture not running")
    
    features = manager.get_features()
    if features is None:
        raise HTTPException(status_code=503, detail="No audio data")
    
    return features.to_dict()


# === シナリオ管理 ===

@router.get("/scenarios")
async def list_scenarios():
    """利用可能なシナリオ一覧"""
    scenarios = []
    for name, config in SCENARIOS.items():
        scenarios.append({
            "name": name.value,
            "duration": config.duration,
            "sample_rate": config.sample_rate,
            "total_samples": config.total_samples,
            "throttle_range": [config.throttle_start, config.throttle_end],
            "steering": config.steering
        })
    return {"scenarios": scenarios}


@router.post("/scenario/start")
async def start_scenario(request: ScenarioStartRequest, background_tasks: BackgroundTasks):
    """シナリオ実行開始（バックグラウンド）"""
    try:
        scenario_type = ScenarioType(request.scenario)
    except ValueError:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown scenario: {request.scenario}. "
                   f"Available: {[s.value for s in ScenarioType]}"
        )
    
    executor = get_executor()
    
    if executor.is_running:
        raise HTTPException(status_code=409, detail="Another scenario is running")
    
    # バックグラウンドで実行
    background_tasks.add_task(executor.execute, scenario_type)
    
    return {
        "status": "started",
        "scenario": scenario_type.value,
        "config": SCENARIOS[scenario_type].__dict__
    }


@router.post("/scenario/stop")
async def stop_scenario():
    """シナリオ中断"""
    executor = get_executor()
    executor.stop()
    return {"status": "stopped"}


@router.get("/scenario/status")
async def get_scenario_status():
    """シナリオ実行状態"""
    executor = get_executor()
    return executor.get_status()


@router.get("/scenario/result")
async def get_scenario_result():
    """シナリオ実行結果（最後の実行）"""
    executor = get_executor()
    result = executor.get_result()
    
    if result is None:
        raise HTTPException(status_code=404, detail="No result available")
    
    return result.to_dict()
```

### 5. main.py への統合

`http_server/main.py` に以下を追加:

```python
# 既存のimportに追加
from .routes import acoustic
from .core.scenario import init_executor

# 既存のモーター/サーボ制御モジュールをimport（実際のパスに合わせる）
# from .core.motor import set_throttle
# from .core.servo import set_steering

# === シナリオエグゼキュータ初期化 ===
# 実際のモーター/サーボ制御関数に置き換える
def set_throttle(value: float):
    """スロットル設定（-1.0 ~ 1.0）"""
    # TODO: 実際のPWM制御を呼び出す
    # motor.set_throttle(value)
    print(f"[PWM] Throttle: {value:.3f}")

def set_steering(value: float):
    """ステアリング設定（-1.0 ~ 1.0）"""
    # TODO: 実際のPWM制御を呼び出す
    # servo.set_steering(value)
    print(f"[PWM] Steering: {value:.3f}")

# エグゼキュータを初期化
init_executor(set_throttle, set_steering)

# ルーター登録
app.include_router(acoustic.router)
```

---

## 依存関係

`requirements.txt` に追加:

```
pyaudio>=0.2.13
librosa>=0.10.0
soundfile>=0.12.0
```

インストール:

```bash
# Jetson
sudo apt install python3-pyaudio portaudio19-dev
pip install librosa soundfile --break-system-packages
```

---

## テスト手順

### 1. オーディオデバイス確認

```bash
# マイク一覧
arecord -l

# API経由
curl http://localhost:8000/acoustic/devices
```

### 2. 単体テスト

```bash
# キャプチャテスト
curl -X POST http://localhost:8000/acoustic/capture/start
curl http://localhost:8000/acoustic/features
curl -X POST http://localhost:8000/acoustic/capture/stop
```

### 3. シナリオ実行テスト

```bash
# シナリオ一覧
curl http://localhost:8000/acoustic/scenarios

# throttle_stop 実行
curl -X POST http://localhost:8000/acoustic/scenario/start \
  -H "Content-Type: application/json" \
  -d '{"scenario": "throttle_stop"}'

# 状態確認（実行中に繰り返し）
curl http://localhost:8000/acoustic/scenario/status

# 結果取得
curl http://localhost:8000/acoustic/scenario/result
```

---

## 注意事項

1. **車輪を浮かせる**: 台に載せるなどして車輪が自由に回転できる状態にする
2. **PWM制御の接続**: `main.py` の `set_throttle`, `set_steering` を実際のモーター/サーボ制御に接続する
3. **throttle_slow の値調整**: 車輪が回り始める最低限のPWM値は車両によって異なる。必要に応じて `SCENARIOS` の値を調整する
