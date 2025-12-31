"""NanoSAM TensorRT推論モジュール

既存のnanosam最適化版Predictorをラップし、HTTP API用のインターフェースを提供。
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

# nanosam をPythonパスに追加
NANOSAM_PATH = Path.home() / "projects" / "nanosam"
if str(NANOSAM_PATH) not in sys.path:
    sys.path.insert(0, str(NANOSAM_PATH))

# TensorRTエンジンのパス
ENCODER_ENGINE = NANOSAM_PATH / "data" / "resnet18_image_encoder.engine"
DECODER_ENGINE = NANOSAM_PATH / "data" / "mobile_sam_mask_decoder.engine"


class PromptType(Enum):
    """プロンプトタイプ"""
    POINT = "point"
    BOX = "box"
    AUTO = "auto"


@dataclass
class SegmentationResult:
    """セグメンテーション結果"""
    mask: np.ndarray              # バイナリマスク (H, W) uint8
    score: float                  # 信頼度スコア
    inference_time_ms: float      # 推論時間
    prompt_type: PromptType       # 使用したプロンプトタイプ


class NanoSAMInference:
    """NanoSAM TensorRT推論クラス

    シングルトンパターンで実装（メモリ節約のため）

    Usage:
        sam = NanoSAMInference.get_instance()
        result = sam.segment_point(image, [(320, 240, 1)])
    """

    _instance = None

    @classmethod
    def get_instance(cls) -> 'NanoSAMInference':
        """シングルトンインスタンス取得"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def release_instance(cls):
        """インスタンス解放"""
        if cls._instance is not None:
            cls._instance.cleanup()
            cls._instance = None

    def __init__(self):
        """初期化（直接呼び出さず、get_instance()を使用）"""
        self.predictor = None
        self._initialized = False
        self._last_image_features = None
        self._last_image_shape = None

    def initialize(self) -> bool:
        """モデル初期化"""
        if self._initialized:
            return True

        if not ENCODER_ENGINE.exists():
            print(f"[NanoSAM] Encoder engine not found: {ENCODER_ENGINE}")
            return False

        if not DECODER_ENGINE.exists():
            print(f"[NanoSAM] Decoder engine not found: {DECODER_ENGINE}")
            return False

        try:
            # 最適化版Predictorを使用
            try:
                from nanosam.utils.predictor_optimized import PredictorOptimized as Predictor
                print("[NanoSAM] Using optimized predictor")
            except ImportError:
                from nanosam.utils.predictor import Predictor
                print("[NanoSAM] Using standard predictor")

            print(f"[NanoSAM] Loading engines...")
            start = time.time()

            self.predictor = Predictor(
                image_encoder_engine=str(ENCODER_ENGINE),
                mask_decoder_engine=str(DECODER_ENGINE)
            )

            load_time = time.time() - start
            print(f"[NanoSAM] Engines loaded ({load_time:.1f}s)")

            # ウォームアップ
            self._warmup()

            self._initialized = True
            return True

        except Exception as e:
            print(f"[NanoSAM] Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _warmup(self, iterations: int = 3):
        """ウォームアップ"""
        print(f"[NanoSAM] Warming up ({iterations} iterations)...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)

        for i in range(iterations):
            self.predictor.set_image(dummy)
            _ = self.predictor.predict(np.array([[320, 240]]), np.array([1]))

        print("[NanoSAM] Warmup complete")

    def _set_image(self, image: np.ndarray) -> float:
        """画像をセット（特徴量抽出）

        同じ画像に対する複数のプロンプトを効率化するため、
        画像特徴量をキャッシュする。

        Args:
            image: BGR画像 (H, W, 3)

        Returns:
            エンコード時間 (ms)
        """
        # 画像が変わっていなければスキップ
        if (self._last_image_shape == image.shape and
            self._last_image_features is not None):
            return 0.0

        start = time.perf_counter()
        self.predictor.set_image(image)
        encode_time = (time.perf_counter() - start) * 1000

        self._last_image_shape = image.shape
        self._last_image_features = True  # フラグとして使用

        return encode_time

    def segment_point(
        self,
        image: np.ndarray,
        points: List[Tuple[int, int, int]]
    ) -> SegmentationResult:
        """ポイントプロンプトでセグメンテーション

        Args:
            image: BGR画像 (H, W, 3)
            points: [(x, y, label), ...]
                    label: 1=前景, 0=背景

        Returns:
            SegmentationResult
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("NanoSAM not initialized")

        start = time.perf_counter()

        # 画像特徴量抽出
        encode_time = self._set_image(image)

        # ポイント配列作成
        point_coords = np.array([[p[0], p[1]] for p in points])
        point_labels = np.array([p[2] for p in points])

        # マスク生成
        mask, score, _ = self.predictor.predict(point_coords, point_labels)

        total_time = (time.perf_counter() - start) * 1000

        # マスクを uint8 に変換 (GPU tensor対応)
        if hasattr(mask, 'cpu'):
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask[0]

        # 4チャンネルの場合は最初のチャンネルを使用
        if mask_np.ndim == 3:
            mask_np = mask_np[0]

        mask_uint8 = (mask_np > 0).astype(np.uint8) * 255

        # スコア処理（score shape: (1, N) where N is number of masks）
        if hasattr(score, 'cpu'):
            score_np = score.cpu().numpy().flatten()
        else:
            score_np = np.array(score).flatten()
        score_val = float(score_np[0]) if len(score_np) > 0 else 0.0

        return SegmentationResult(
            mask=mask_uint8,
            score=score_val,
            inference_time_ms=total_time,
            prompt_type=PromptType.POINT
        )

    def segment_box(
        self,
        image: np.ndarray,
        box: Tuple[int, int, int, int]
    ) -> SegmentationResult:
        """ボックスプロンプトでセグメンテーション

        Args:
            image: BGR画像 (H, W, 3)
            box: (x1, y1, x2, y2) 矩形座標

        Returns:
            SegmentationResult
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("NanoSAM not initialized")

        start = time.perf_counter()

        # 画像特徴量抽出
        encode_time = self._set_image(image)

        # ボックスの中心をポイントとして使用
        x1, y1, x2, y2 = box
        point_coords = np.array([
            [(x1 + x2) // 2, (y1 + y2) // 2],  # 中心
        ])
        point_labels = np.array([1])  # 前景

        mask, score, _ = self.predictor.predict(point_coords, point_labels)

        total_time = (time.perf_counter() - start) * 1000

        # マスクを uint8 に変換
        if hasattr(mask, 'cpu'):
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask[0]

        if mask_np.ndim == 3:
            mask_np = mask_np[0]

        mask_uint8 = (mask_np > 0).astype(np.uint8) * 255

        # ボックス外をマスクアウト
        box_mask = np.zeros_like(mask_uint8)
        box_mask[y1:y2, x1:x2] = 255
        mask_uint8 = cv2.bitwise_and(mask_uint8, box_mask)

        # スコア処理
        if hasattr(score, 'cpu'):
            score_np = score.cpu().numpy().flatten()
        else:
            score_np = np.array(score).flatten()
        score_val = float(score_np[0]) if len(score_np) > 0 else 0.0

        return SegmentationResult(
            mask=mask_uint8,
            score=score_val,
            inference_time_ms=total_time,
            prompt_type=PromptType.BOX
        )

    def segment_auto(
        self,
        image: np.ndarray,
        grid_size: int = 8
    ) -> List[SegmentationResult]:
        """自動セグメンテーション（グリッドポイント）

        画像全体をグリッド状のポイントでセグメント

        Args:
            image: BGR画像 (H, W, 3)
            grid_size: グリッドサイズ（デフォルト8x8=64ポイント）

        Returns:
            List[SegmentationResult]
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError("NanoSAM not initialized")

        h, w = image.shape[:2]
        results = []

        start = time.perf_counter()

        # 画像特徴量抽出（1回のみ）
        encode_time = self._set_image(image)

        # グリッドポイント生成
        for row in range(grid_size):
            for col in range(grid_size):
                x = int((col + 0.5) * w / grid_size)
                y = int((row + 0.5) * h / grid_size)

                point_coords = np.array([[x, y]])
                point_labels = np.array([1])

                mask, score, _ = self.predictor.predict(point_coords, point_labels)

                # マスク処理
                if hasattr(mask, 'cpu'):
                    mask_np = mask[0].cpu().numpy()
                else:
                    mask_np = mask[0]

                if mask_np.ndim == 3:
                    mask_np = mask_np[0]

                mask_uint8 = (mask_np > 0).astype(np.uint8) * 255

                # スコア処理
                if hasattr(score, 'cpu'):
                    score_np = score.cpu().numpy().flatten()
                else:
                    score_np = np.array(score).flatten()
                score_val = float(score_np[0]) if len(score_np) > 0 else 0.0

                results.append(SegmentationResult(
                    mask=mask_uint8,
                    score=score_val,
                    inference_time_ms=0,  # 後で計算
                    prompt_type=PromptType.AUTO
                ))

        total_time = (time.perf_counter() - start) * 1000

        # 各結果の時間を更新
        per_mask_time = total_time / len(results)
        for result in results:
            result.inference_time_ms = per_mask_time

        return results

    def segment_road(self, image: np.ndarray) -> SegmentationResult:
        """道路領域をセグメント（プリセット）

        画像下部中央にポイントを配置して道路を抽出

        Args:
            image: BGR画像 (H, W, 3)

        Returns:
            SegmentationResult
        """
        h, w = image.shape[:2]

        # 道路は通常、画像下部中央にある
        points = [
            (w // 2, int(h * 0.85), 1),      # 下部中央（前景）
            (w // 2, int(h * 0.70), 1),      # やや上（前景）
            (w // 4, int(h * 0.30), 0),      # 左上（背景）
            (3 * w // 4, int(h * 0.30), 0),  # 右上（背景）
        ]

        return self.segment_point(image, points)

    def segment_obstacle(self, image: np.ndarray) -> SegmentationResult:
        """中央の障害物をセグメント（プリセット）

        画像中央にポイントを配置して障害物を抽出

        Args:
            image: BGR画像 (H, W, 3)

        Returns:
            SegmentationResult
        """
        h, w = image.shape[:2]

        points = [
            (w // 2, h // 2, 1),  # 中央（前景）
        ]

        return self.segment_point(image, points)

    def get_status(self) -> Dict[str, Any]:
        """ステータス取得"""
        return {
            "initialized": self._initialized,
            "encoder_engine": str(ENCODER_ENGINE),
            "decoder_engine": str(DECODER_ENGINE),
            "encoder_exists": ENCODER_ENGINE.exists(),
            "decoder_exists": DECODER_ENGINE.exists(),
        }

    def cleanup(self):
        """リソース解放"""
        self.predictor = None
        self._initialized = False
        self._last_image_features = None
        self._last_image_shape = None
        print("[NanoSAM] Resources released")

    @property
    def is_initialized(self) -> bool:
        return self._initialized
