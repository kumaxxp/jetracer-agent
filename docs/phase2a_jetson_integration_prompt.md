# YANA Phase 2-A 音響分類モデル統合（Jetson側）

**実行環境**: Jetson Orin Nano Super
**リポジトリ**: `~/jetracer-agent`
**前提**: Phase 2-A 音響キャプチャ・シナリオ実行が実装済み

---

## 概要

PC側で訓練した音響状態分類モデル（Random Forest）をJetson側に統合し、リアルタイム推論APIを提供する。

## 事前準備

### 1. モデルファイル転送

PC側で実行：
```bash
cd /home/kuma/projects/yana-brain
scp models/acoustic_classifier.pkl jetson@192.168.1.65:~/jetracer-agent/models/
```

Jetson側でディレクトリ作成（必要な場合）：
```bash
mkdir -p ~/jetracer-agent/models
```

---

## 実装タスク

### 1. ディレクトリ構造

```
jetracer-agent/
├── models/
│   └── acoustic_classifier.pkl    # 転送済みモデル
└── http_server/
    ├── core/
    │   ├── acoustic.py            # 既存
    │   ├── scenario.py            # 既存
    │   └── acoustic_inference.py  # 新規: 推論モジュール
    └── routes/
        └── acoustic.py            # 既存（エンドポイント追加）
```

### 2. http_server/core/acoustic_inference.py

```python
"""
音響状態推論モジュール（Jetson側）

訓練済みモデルを使用してリアルタイムで音響状態を分類
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Tuple, List
import logging

from .acoustic import AudioFeatures

logger = logging.getLogger(__name__)


class AcousticStateClassifier:
    """音響状態分類器"""
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: モデルファイルパス（Noneの場合はデフォルトパス）
        """
        if model_path is None:
            # デフォルトパス
            model_path = Path(__file__).parent.parent.parent / "models" / "acoustic_classifier.pkl"
        
        self.model_path = Path(model_path)
        self.model = None
        self.label_names: List[str] = []
        self.is_loaded = False
        
        self._load_model()
    
    def _load_model(self):
        """モデルをロード"""
        if not self.model_path.exists():
            logger.warning(f"Model file not found: {self.model_path}")
            return
        
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.label_names = model_data['label_names']
            self.is_loaded = True
            logger.info(f"Acoustic classifier loaded: {self.label_names}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    
    def extract_features(
        self, 
        audio_features: AudioFeatures, 
        pwm_throttle: float = 0.0
    ) -> Optional[np.ndarray]:
        """
        AudioFeaturesから特徴量ベクトルを抽出
        
        Args:
            audio_features: 音響特徴量
            pwm_throttle: 現在のスロットルPWM値
        
        Returns:
            45次元の特徴量ベクトル（shape: (1, 45)）
        """
        if audio_features is None:
            return None
        
        # 特徴量を連結（訓練時と同じ順序）
        feature_vector = (
            audio_features.mfcc_mean +           # 13次元
            audio_features.mfcc_std +            # 13次元
            audio_features.delta_mfcc_mean +     # 13次元
            [
                audio_features.spectral_centroid,
                audio_features.spectral_rolloff,
                audio_features.spectral_bandwidth,
                audio_features.rms_energy,
                audio_features.zcr,
                pwm_throttle,                    # 1次元
            ]
        )
        
        return np.array(feature_vector).reshape(1, -1)
    
    def predict(
        self, 
        audio_features: AudioFeatures, 
        pwm_throttle: float = 0.0
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        音響状態を推論
        
        Args:
            audio_features: 音響特徴量
            pwm_throttle: 現在のスロットルPWM値
        
        Returns:
            (状態名, 信頼度) のタプル。推論失敗時は (None, None)
        """
        if not self.is_loaded:
            logger.warning("Model not loaded")
            return None, None
        
        features = self.extract_features(audio_features, pwm_throttle)
        
        if features is None:
            return None, None
        
        try:
            # 推論
            label_idx = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            state = self.label_names[label_idx]
            confidence = float(probabilities[label_idx])
            
            return state, confidence
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, None
    
    def predict_with_all_probs(
        self, 
        audio_features: AudioFeatures, 
        pwm_throttle: float = 0.0
    ) -> Tuple[Optional[str], Optional[dict]]:
        """
        全クラスの確率付きで推論
        
        Returns:
            (状態名, {状態: 確率} の辞書) のタプル
        """
        if not self.is_loaded:
            return None, None
        
        features = self.extract_features(audio_features, pwm_throttle)
        
        if features is None:
            return None, None
        
        try:
            label_idx = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            state = self.label_names[label_idx]
            prob_dict = {
                name: float(prob) 
                for name, prob in zip(self.label_names, probabilities)
            }
            
            return state, prob_dict
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return None, None


# グローバルインスタンス（シングルトン）
_classifier: Optional[AcousticStateClassifier] = None


def get_classifier() -> Optional[AcousticStateClassifier]:
    """分類器インスタンスを取得"""
    global _classifier
    if _classifier is None:
        _classifier = AcousticStateClassifier()
    return _classifier


def init_classifier(model_path: str = None) -> AcousticStateClassifier:
    """分類器を初期化"""
    global _classifier
    _classifier = AcousticStateClassifier(model_path)
    return _classifier
```

### 3. http_server/routes/acoustic.py への追加

既存の `acoustic.py` に以下のエンドポイントを追加：

```python
# === 追加のimport ===
from ..core.acoustic_inference import get_classifier
import time


# === 推論エンドポイント ===

@router.get("/state/predict")
async def predict_state(pwm_throttle: float = 0.0):
    """
    現在の音響状態を推論
    
    Args:
        pwm_throttle: 現在のスロットルPWM値（クエリパラメータ）
    
    Returns:
        推論結果（状態、信頼度、全クラス確率）
    """
    # 分類器取得
    classifier = get_classifier()
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Acoustic classifier not available"
        )
    
    # 音響マネージャー取得
    manager = AcousticManager.get_instance()
    if not manager.is_running():
        raise HTTPException(
            status_code=400, 
            detail="Audio capture not running. Call /acoustic/capture/start first"
        )
    
    # 最新の特徴量取得
    features = manager.get_features()
    if features is None:
        raise HTTPException(
            status_code=503, 
            detail="No audio features available"
        )
    
    # 推論（全クラス確率付き）
    state, probabilities = classifier.predict_with_all_probs(features, pwm_throttle)
    
    if state is None:
        raise HTTPException(
            status_code=500, 
            detail="Prediction failed"
        )
    
    return {
        "state": state,
        "confidence": probabilities[state],
        "probabilities": probabilities,
        "pwm_throttle": pwm_throttle,
        "timestamp": time.time()
    }


@router.get("/state/continuous")
async def get_continuous_state(
    pwm_throttle: float = 0.0,
    include_features: bool = False
):
    """
    継続的な状態監視用エンドポイント
    
    特徴量も含めた詳細情報を返す（デバッグ用）
    """
    classifier = get_classifier()
    manager = AcousticManager.get_instance()
    
    if not manager.is_running():
        return {
            "status": "not_running",
            "state": None,
            "timestamp": time.time()
        }
    
    features = manager.get_features()
    
    result = {
        "status": "running",
        "timestamp": time.time(),
        "pwm_throttle": pwm_throttle
    }
    
    if features is None:
        result["state"] = None
        result["error"] = "No features"
    elif classifier is None or not classifier.is_loaded:
        result["state"] = None
        result["error"] = "Classifier not loaded"
    else:
        state, probs = classifier.predict_with_all_probs(features, pwm_throttle)
        result["state"] = state
        result["probabilities"] = probs
        result["confidence"] = probs[state] if probs else None
    
    if include_features and features:
        result["features"] = {
            "rms_energy": features.rms_energy,
            "spectral_centroid": features.spectral_centroid,
            "zcr": features.zcr,
            "mfcc_mean_0": features.mfcc_mean[0] if features.mfcc_mean else None
        }
    
    return result


@router.get("/classifier/status")
async def get_classifier_status():
    """分類器の状態を取得"""
    classifier = get_classifier()
    
    if classifier is None:
        return {
            "loaded": False,
            "error": "Classifier not initialized"
        }
    
    return {
        "loaded": classifier.is_loaded,
        "model_path": str(classifier.model_path),
        "labels": classifier.label_names if classifier.is_loaded else []
    }
```

### 4. main.py への統合

`http_server/main.py` に以下を追加：

```python
# === 追加のimport ===
from .core.acoustic_inference import init_classifier

# === 起動時の初期化（lifespan または startup イベント内）===

# 分類器を初期化
try:
    classifier = init_classifier()
    if classifier.is_loaded:
        print(f"✓ Acoustic classifier loaded: {classifier.label_names}")
    else:
        print("⚠ Acoustic classifier not loaded (model file missing?)")
except Exception as e:
    print(f"⚠ Failed to initialize acoustic classifier: {e}")
```

---

## 依存関係

`requirements.txt` に追加（まだない場合）：

```
joblib>=1.3.0
scikit-learn>=1.3.0
```

インストール：
```bash
pip install joblib scikit-learn --break-system-packages
```

---

## テスト手順

### 1. モデルファイル確認

```bash
ls -la ~/jetracer-agent/models/
# acoustic_classifier.pkl が存在することを確認
```

### 2. HTTPサーバー起動

```bash
cd ~/jetracer-agent
python -m http_server.main
```

起動ログで以下を確認：
```
✓ Acoustic classifier loaded: ['throttle_slow', 'throttle_starting', 'throttle_stop']
```

### 3. API テスト

```bash
# 分類器状態確認
curl http://localhost:8000/acoustic/classifier/status

# キャプチャ開始
curl -X POST http://localhost:8000/acoustic/capture/start

# 推論テスト（PWM=0 で停止状態を期待）
curl "http://localhost:8000/acoustic/state/predict?pwm_throttle=0.0"

# 詳細情報付き
curl "http://localhost:8000/acoustic/state/continuous?pwm_throttle=0.0&include_features=true"

# キャプチャ停止
curl -X POST http://localhost:8000/acoustic/capture/stop
```

### 4. 期待される応答

```json
{
  "state": "throttle_stop",
  "confidence": 0.95,
  "probabilities": {
    "throttle_stop": 0.95,
    "throttle_starting": 0.03,
    "throttle_slow": 0.02
  },
  "pwm_throttle": 0.0,
  "timestamp": 1703845200.123
}
```

---

## PC側からの統合テスト

PC側で以下を実行：

```python
from src.acoustic import AcousticClient

client = AcousticClient()

# キャプチャ開始
client.start_capture()

# 推論
import time
for _ in range(10):
    response = client._get("/acoustic/state/predict?pwm_throttle=0.0")
    print(f"State: {response['state']}, Confidence: {response['confidence']:.1%}")
    time.sleep(0.5)

# キャプチャ停止
client.stop_capture()
```

---

## トラブルシューティング

### モデルがロードされない

```bash
# ファイル存在確認
ls -la ~/jetracer-agent/models/acoustic_classifier.pkl

# パーミッション確認
chmod 644 ~/jetracer-agent/models/acoustic_classifier.pkl

# Pythonから直接確認
python3 -c "import joblib; m = joblib.load('models/acoustic_classifier.pkl'); print(m)"
```

### scikit-learnバージョン不一致

PC側とJetson側で異なるバージョンの場合、警告が出ることがあります：

```bash
# Jetson側のバージョン確認
python3 -c "import sklearn; print(sklearn.__version__)"

# PC側と合わせる場合
pip install scikit-learn==1.3.0 --break-system-packages
```

### 推論が遅い

```python
# acoustic_inference.py 内で計測
import time

def predict(self, audio_features, pwm_throttle=0.0):
    start = time.perf_counter()
    # ... 推論処理 ...
    elapsed = (time.perf_counter() - start) * 1000
    logger.debug(f"Inference time: {elapsed:.1f}ms")
```

目標: < 10ms/推論

---

## 次のステップ

1. **実車テスト**: 実際に走行しながら推論精度を確認
2. **PWM除外モデル**: PWMなしで純粋な音響のみの分類を試す
3. **制御への統合**: 推論結果を走行制御にフィードバック
