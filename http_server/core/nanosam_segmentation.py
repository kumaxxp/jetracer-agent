"""NanoSAMセグメンテーションモジュール

機能:
- TensorRTエンジンによる高速セグメンテーション
- 走行可能領域の検出
- ステアリング推奨値の計算
"""
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import threading
import logging

logger = logging.getLogger(__name__)

# NanoSAMの遅延インポート
_predictor_class = None
_cv2 = None


def _get_cv2():
    global _cv2
    if _cv2 is None:
        import cv2
        _cv2 = cv2
    return _cv2


def _get_predictor_class():
    """最適化版Predictorをインポート"""
    global _predictor_class
    if _predictor_class is None:
        try:
            # 最適化版を優先
            import sys
            sys.path.insert(0, str(Path.home() / "projects/nanosam"))
            from nanosam.utils.predictor_optimized import PredictorOptimized
            _predictor_class = PredictorOptimized
            logger.info("Using optimized NanoSAM predictor")
        except ImportError:
            # フォールバック
            from nanosam.utils.predictor import Predictor
            _predictor_class = Predictor
            logger.info("Using standard NanoSAM predictor")
    return _predictor_class


@dataclass
class SegmentationResult:
    """セグメンテーション結果"""
    mask: np.ndarray                # バイナリマスク (H, W)
    road_ratio: float               # 走行可能領域の割合
    inference_time_ms: float        # 推論時間
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "road_ratio": round(self.road_ratio, 3),
            "inference_time_ms": round(self.inference_time_ms, 2),
            "mask_shape": list(self.mask.shape),
            "timestamp": self.timestamp
        }


@dataclass
class SteeringResult:
    """ステアリング計算結果"""
    steering: float                 # -1.0 (左) ~ 1.0 (右)
    confidence: float               # 信頼度
    road_center_x: float            # 走行可能領域の中心X (正規化)
    road_width_ratio: float         # 走行可能領域の幅比率
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "steering": round(self.steering, 3),
            "confidence": round(self.confidence, 2),
            "road_center_x": round(self.road_center_x, 3),
            "road_width_ratio": round(self.road_width_ratio, 3),
            "timestamp": self.timestamp
        }


class NanoSAMSegmentation:
    """NanoSAMセグメンテーションエンジン"""

    DEFAULT_ENCODER = str(Path.home() / "projects/nanosam/data/resnet18_image_encoder.engine")
    DEFAULT_DECODER = str(Path.home() / "projects/nanosam/data/mobile_sam_mask_decoder.engine")

    def __init__(
        self,
        encoder_path: Optional[str] = None,
        decoder_path: Optional[str] = None,
        auto_init: bool = False
    ):
        self.encoder_path = encoder_path or self.DEFAULT_ENCODER
        self.decoder_path = decoder_path or self.DEFAULT_DECODER

        self._predictor = None
        self._initialized = False
        self._lock = threading.Lock()
        self._last_mask: Optional[np.ndarray] = None
        self._last_image_shape: Optional[Tuple[int, int]] = None

        if auto_init:
            self.initialize()

    def initialize(self) -> Tuple[bool, str]:
        """モデル初期化"""
        if self._initialized:
            return True, "Already initialized"

        # エンジンファイル存在確認
        if not Path(self.encoder_path).exists():
            return False, f"Encoder not found: {self.encoder_path}"
        if not Path(self.decoder_path).exists():
            return False, f"Decoder not found: {self.decoder_path}"

        try:
            Predictor = _get_predictor_class()
            self._predictor = Predictor(
                image_encoder_engine=self.encoder_path,
                mask_decoder_engine=self.decoder_path
            )
            self._initialized = True
            logger.info(f"NanoSAM initialized: encoder={self.encoder_path}")
            return True, "Initialized successfully"
        except Exception as e:
            logger.error(f"NanoSAM initialization failed: {e}")
            return False, str(e)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def segment_image(
        self,
        image: np.ndarray,
        point: Optional[Tuple[int, int]] = None
    ) -> SegmentationResult:
        """画像をセグメンテーション

        Args:
            image: BGR画像 (H, W, 3)
            point: クリックポイント (x, y)、Noneの場合は画像下部中央

        Returns:
            SegmentationResult
        """
        result = SegmentationResult(
            mask=np.zeros((1, 1), dtype=np.uint8),
            road_ratio=0.0,
            inference_time_ms=0.0,
            timestamp=time.time()
        )

        if not self._initialized:
            logger.warning("NanoSAM not initialized")
            return result

        h, w = image.shape[:2]

        # デフォルトポイント: 画像下部中央（床面を指定）
        if point is None:
            point = (w // 2, int(h * 0.85))

        try:
            with self._lock:
                start = time.perf_counter()

                # RGB変換
                cv2 = _get_cv2()
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # セグメンテーション実行
                self._predictor.set_image(image_rgb)
                mask, scores, logits = self._predictor.predict(
                    np.array([[point[0], point[1]]]),
                    np.array([1])  # foreground
                )

                elapsed = (time.perf_counter() - start) * 1000

            # マスク処理 (GPU tensor -> CPU numpy)
            if hasattr(mask, 'cpu'):
                mask_np = mask[0].cpu().numpy()
            else:
                mask_np = mask[0]

            mask_binary = (mask_np > 0).astype(np.uint8)

            # 4チャンネルの場合は最初のチャンネルを使用
            if mask_binary.ndim == 3:
                mask_binary = mask_binary[0]

            result.mask = mask_binary
            result.road_ratio = mask_binary.sum() / mask_binary.size
            result.inference_time_ms = elapsed

            self._last_mask = mask_binary
            self._last_image_shape = (h, w)

        except Exception as e:
            logger.error(f"Segmentation failed: {e}")

        return result

    def segment_from_camera(self) -> SegmentationResult:
        """カメラからの現在フレームをセグメンテーション"""
        from .camera_manager import camera_manager

        frame = camera_manager.get_latest_frame(camera_id=0)

        if frame is None:
            return SegmentationResult(
                mask=np.zeros((1, 1), dtype=np.uint8),
                road_ratio=0.0,
                inference_time_ms=0.0,
                timestamp=time.time()
            )

        return self.segment_image(frame)

    def calculate_steering(
        self,
        mask: Optional[np.ndarray] = None,
        roi_top_ratio: float = 0.5
    ) -> SteeringResult:
        """走行可能領域からステアリング値を計算

        Args:
            mask: セグメンテーションマスク（Noneの場合は最後の結果を使用）
            roi_top_ratio: ROIの上端位置（画像上端からの比率）

        Returns:
            SteeringResult
        """
        result = SteeringResult(
            steering=0.0,
            confidence=0.0,
            road_center_x=0.5,
            road_width_ratio=0.0,
            timestamp=time.time()
        )

        if mask is None:
            mask = self._last_mask

        if mask is None or mask.size == 0:
            return result

        h, w = mask.shape[:2]

        # ROI: 画像下半分（走行判断に重要な領域）
        roi_top = int(h * roi_top_ratio)
        roi_mask = mask[roi_top:, :]

        if roi_mask.sum() == 0:
            # 走行可能領域なし
            result.confidence = 0.0
            return result

        # 走行可能領域の重心を計算
        y_indices, x_indices = np.where(roi_mask > 0)

        if len(x_indices) == 0:
            return result

        # 重心X座標（正規化: 0-1）
        center_x = x_indices.mean() / w
        result.road_center_x = center_x

        # 走行可能領域の幅
        road_width = x_indices.max() - x_indices.min()
        result.road_width_ratio = road_width / w

        # ステアリング計算: 中心からのオフセット
        # center_x = 0.5 → steering = 0
        # center_x = 0.0 → steering = -1 (左へ)
        # center_x = 1.0 → steering = +1 (右へ)
        result.steering = (center_x - 0.5) * 2.0

        # 信頼度: 走行可能領域の大きさに比例
        roi_ratio = roi_mask.sum() / roi_mask.size
        result.confidence = min(1.0, roi_ratio * 3)  # 30%以上で信頼度1.0

        return result

    def get_road_mask_encoded(self) -> Optional[str]:
        """最後のマスクをBase64エンコードで取得"""
        if self._last_mask is None:
            return None

        import base64
        cv2 = _get_cv2()

        # PNG圧縮
        _, buffer = cv2.imencode('.png', self._last_mask * 255)
        return base64.b64encode(buffer).decode('utf-8')

    def get_colored_mask(
        self,
        mask: Optional[np.ndarray] = None,
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> Optional[np.ndarray]:
        """カラーマスク画像を生成"""
        if mask is None:
            mask = self._last_mask

        if mask is None:
            return None

        cv2 = _get_cv2()
        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        colored[mask > 0] = color
        return colored

    def get_status(self) -> dict:
        """状態を取得"""
        return {
            "initialized": self._initialized,
            "encoder_path": self.encoder_path,
            "decoder_path": self.decoder_path,
            "encoder_exists": Path(self.encoder_path).exists(),
            "decoder_exists": Path(self.decoder_path).exists(),
            "last_mask_shape": list(self._last_mask.shape) if self._last_mask is not None else None
        }

    def close(self):
        """リソース解放"""
        self._predictor = None
        self._initialized = False


# シングルトン
_nanosam_instance: Optional[NanoSAMSegmentation] = None


def get_nanosam() -> NanoSAMSegmentation:
    """NanoSAMインスタンスを取得"""
    global _nanosam_instance
    if _nanosam_instance is None:
        _nanosam_instance = NanoSAMSegmentation()
    return _nanosam_instance
