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
