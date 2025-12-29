# YANA Phase 2-A 実験開始プロンプト

以下の指示に従って、jetracer-agentリポジトリでPhase 2-Aの実験を開始してください。

---

## プロジェクト概要

**リポジトリ**: `/home/kuma/projects/jetracer-agent`
**目的**: YANA (Your Autonomous Navigation Assistant) Phase 2-A 要素実験
**プラットフォーム**: 開発はWSL2 Ubuntu 22.04、最終デプロイはJetson Orin Nano Super

---

## 実験1: 音響特徴抽出実験（最優先）

### 目的
USBマイクでモーター・サーボ音を取得し、MFCC特徴量を抽出して状態との相関を確認する。

### タスク

1. **ディレクトリ構造作成**
```
jetracer-agent/
├── experiments/
│   └── phase2a/
│       ├── acoustic/
│       │   ├── __init__.py
│       │   ├── audio_capture.py      # マイク入力
│       │   ├── feature_extractor.py  # MFCC抽出
│       │   ├── visualizer.py         # スペクトログラム可視化
│       │   └── recorder.py           # 状態別録音ツール
│       └── README.md
```

2. **audio_capture.py の実装**
```python
"""
USBマイクからリアルタイムで音声を取得するモジュール

要件:
- PyAudioでUSBマイクから16kHzモノラルで取得
- リングバッファで直近1秒分を保持
- 100ms単位でチャンクを返す
- Jetson/PC両対応（デバイス自動検出）
"""

import pyaudio
import numpy as np
from collections import deque
from typing import Optional
import threading

class AudioCapture:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size: int = 1600,  # 100ms @ 16kHz
        buffer_seconds: float = 1.0,
        device_index: Optional[int] = None
    ):
        """
        Args:
            sample_rate: サンプリングレート
            chunk_size: 1回の読み取りサンプル数
            buffer_seconds: バッファに保持する秒数
            device_index: マイクデバイスインデックス（Noneで自動検出）
        """
        pass
    
    def start(self):
        """録音開始"""
        pass
    
    def stop(self):
        """録音停止"""
        pass
    
    def read_chunk(self) -> np.ndarray:
        """最新の100msチャンクを取得"""
        pass
    
    def get_buffer(self) -> np.ndarray:
        """バッファ全体（直近1秒）を取得"""
        pass
    
    @staticmethod
    def list_devices() -> list[dict]:
        """利用可能なオーディオデバイス一覧"""
        pass
    
    @staticmethod
    def find_usb_microphone() -> Optional[int]:
        """USBマイクのデバイスインデックスを自動検出"""
        pass
```

3. **feature_extractor.py の実装**
```python
"""
音声からMFCC等の特徴量を抽出するモジュール

要件:
- librosaでMFCC (13係数) + Δ-MFCC
- Spectral Centroid, Rolloff, Bandwidth
- RMS Energy, Zero Crossing Rate
- 約50次元の特徴ベクトルを返す
"""

import librosa
import numpy as np
from dataclasses import dataclass

@dataclass
class AudioFeatures:
    mfcc_mean: np.ndarray      # (13,)
    mfcc_std: np.ndarray       # (13,)
    delta_mfcc_mean: np.ndarray # (13,)
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    rms_energy: float
    zcr: float
    
    def to_vector(self) -> np.ndarray:
        """全特徴量を1次元ベクトルに結合"""
        pass

class AudioFeatureExtractor:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 256
    ):
        pass
    
    def extract(self, audio: np.ndarray) -> AudioFeatures:
        """音声チャンクから特徴量を抽出"""
        pass
    
    def extract_batch(self, audio_chunks: list[np.ndarray]) -> list[AudioFeatures]:
        """複数チャンクをバッチ処理"""
        pass
```

4. **visualizer.py の実装**
```python
"""
音響特徴量の可視化ツール

要件:
- リアルタイムスペクトログラム表示
- MFCC係数の時系列プロット
- 状態別の特徴量比較グラフ
- matplotlibで実装
"""

import matplotlib.pyplot as plt
import numpy as np

class AcousticVisualizer:
    def plot_spectrogram(self, audio: np.ndarray, sample_rate: int, save_path: str = None):
        """スペクトログラムを表示/保存"""
        pass
    
    def plot_mfcc(self, mfcc: np.ndarray, sample_rate: int, save_path: str = None):
        """MFCC係数を表示/保存"""
        pass
    
    def compare_states(self, state_features: dict[str, list[AudioFeatures]], save_path: str = None):
        """状態別の特徴量を比較表示"""
        pass
```

5. **recorder.py の実装**
```python
"""
状態別の音声録音ツール

要件:
- 各モーター/サーボ状態での録音
- WAVファイル + JSONメタデータ保存
- キーボード操作でラベル付け
- セッション管理
"""

import wave
import json
from pathlib import Path
from datetime import datetime
from enum import Enum

class MotorState(Enum):
    STOPPED = "stopped"
    IDLE = "idle"
    STARTING = "starting"
    RUNNING_SLOW = "running_slow"
    RUNNING_FAST = "running_fast"
    SPINNING = "spinning"
    STALLED = "stalled"

class ServoState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    END_STOP = "end_stop"

class AcousticRecorder:
    def __init__(self, output_dir: str = "data/acoustic"):
        pass
    
    def start_session(self, session_name: str = None) -> str:
        """録音セッション開始"""
        pass
    
    def record_state(
        self,
        motor_state: MotorState,
        servo_state: ServoState,
        duration_sec: float = 5.0,
        pwm_throttle: float = 0.0,
        pwm_steering: float = 0.0,
        notes: str = ""
    ) -> dict:
        """指定状態での録音"""
        pass
    
    def end_session(self) -> dict:
        """セッション終了、サマリー返却"""
        pass
```

6. **テストスクリプト作成**

`experiments/phase2a/test_acoustic.py`:
```python
"""
音響実験のテストスクリプト

実行方法:
    python -m experiments.phase2a.test_acoustic

テスト内容:
1. マイクデバイス検出
2. リアルタイム音声取得
3. MFCC特徴量抽出
4. スペクトログラム表示
"""

def main():
    # 1. デバイス一覧表示
    # 2. マイク接続テスト
    # 3. 5秒間録音
    # 4. 特徴量抽出
    # 5. 可視化
    pass

if __name__ == "__main__":
    main()
```

### 依存関係

`requirements.txt` に追加:
```
pyaudio>=0.2.13
librosa>=0.10.0
soundfile>=0.12.0
matplotlib>=3.7.0
```

### 注意事項

- PyAudioはJetsonでは `sudo apt install python3-pyaudio` でインストール
- PC (WSL2) では `pip install pyaudio` の前に `sudo apt install portaudio19-dev`
- サンプリングレートは16kHzに統一（Jetsonでの処理負荷考慮）

---

## 実験2: NanoSAM動作確認（次のステップ）

※音響実験完了後に実施

### 概要
- NanoSAMリポジトリのクローンとセットアップ
- TensorRTエンジン生成
- 推論速度ベンチマーク

---

## 実験3: FunctionGemma実験（最後のステップ）

※NanoSAM実験完了後に実施

### 概要
- FunctionGemma モデルのダウンロード
- MCPツール選択精度テスト
- レスポンス時間測定

---

## 成果物

Phase 2-A完了時に以下を確認:
- [ ] 各モーター状態のMFCCスペクトログラム画像
- [ ] 状態間の特徴量差異の分析レポート
- [ ] NanoSAM推論時間ベンチマーク結果
- [ ] FunctionGemmaツール選択精度レポート

---

## 開始コマンド

```bash
cd /home/kuma/projects/jetracer-agent
git checkout -b feature/phase2a-experiments

# 依存関係インストール (PC)
sudo apt install portaudio19-dev
pip install pyaudio librosa soundfile matplotlib

# ディレクトリ作成
mkdir -p experiments/phase2a/acoustic
mkdir -p data/acoustic

# 実装開始
```
