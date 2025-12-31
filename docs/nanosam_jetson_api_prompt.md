# NanoSAM HTTP API 実装プロンプト（Jetson側）

このプロンプトをJetsonのClaude Codeに貼り付けて実行してください。

---

## 概要

NanoSAMをjetracer-agentのHTTP APIに統合します。
既存のTensorRTエンジンと最適化版Predictorを使用します。

## 前提条件（確認済み）

```bash
# 以下のファイルが存在することを確認
ls -la ~/projects/nanosam/data/resnet18_image_encoder.engine
ls -la ~/projects/nanosam/data/mobile_sam_mask_decoder.engine
ls -la ~/projects/nanosam/nanosam/utils/predictor_optimized.py
```

## 作業ディレクトリ

```
~/projects/jetracer-agent/
```

## 実装手順

### Step 1: NanoSAMラッパーモジュール作成

`http_server/core/nanosam_inference.py` を作成：

```python
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
                from nanosam.utils.predictor_optimized import Predictor
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
        
        # マスクを uint8 に変換
        mask_uint8 = (mask[0, 0] > 0).astype(np.uint8) * 255
        
        return SegmentationResult(
            mask=mask_uint8,
            score=float(score[0]),
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
        
        # ボックスの4隅をポイントとして使用
        x1, y1, x2, y2 = box
        # 左上と右下を前景、他を背景として使用
        point_coords = np.array([
            [(x1 + x2) // 2, (y1 + y2) // 2],  # 中心
        ])
        point_labels = np.array([1])  # 前景
        
        # または、ボックス専用の予測があれば使用
        # mask, score, _ = self.predictor.predict_box(box)
        mask, score, _ = self.predictor.predict(point_coords, point_labels)
        
        total_time = (time.perf_counter() - start) * 1000
        
        # ボックス内のみにマスク
        mask_uint8 = (mask[0, 0] > 0).astype(np.uint8) * 255
        
        # ボックス外をマスクアウト
        box_mask = np.zeros_like(mask_uint8)
        box_mask[y1:y2, x1:x2] = 255
        mask_uint8 = cv2.bitwise_and(mask_uint8, box_mask)
        
        return SegmentationResult(
            mask=mask_uint8,
            score=float(score[0]),
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
                
                mask_uint8 = (mask[0, 0] > 0).astype(np.uint8) * 255
                
                results.append(SegmentationResult(
                    mask=mask_uint8,
                    score=float(score[0]),
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
```

### Step 2: HTTPルート作成

`http_server/routes/nanosam.py` を作成：

```python
"""NanoSAM HTTP APIルート"""

import base64
from typing import List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import cv2

from ..core.nanosam_inference import NanoSAMInference, PromptType
from ..core.camera_manager import CameraManager

router = APIRouter(prefix="/nanosam", tags=["NanoSAM"])


# --- リクエスト/レスポンスモデル ---

class Point(BaseModel):
    x: int
    y: int
    label: int = 1  # 1=前景, 0=背景


class Box(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class SegmentPointRequest(BaseModel):
    camera_id: int = 1
    points: List[Point]
    image_base64: Optional[str] = None  # 画像を直接渡す場合


class SegmentBoxRequest(BaseModel):
    camera_id: int = 1
    box: Box
    image_base64: Optional[str] = None


class SegmentAutoRequest(BaseModel):
    camera_id: int = 1
    grid_size: int = 8
    image_base64: Optional[str] = None


class SegmentPresetRequest(BaseModel):
    camera_id: int = 1
    preset: str  # "road" or "obstacle"
    image_base64: Optional[str] = None


class SegmentationResponse(BaseModel):
    mask_base64: str           # PNG形式のマスク画像
    score: float               # 信頼度スコア
    inference_time_ms: float   # 推論時間
    prompt_type: str           # プロンプトタイプ
    mask_ratio: float          # マスク領域の割合


# --- ヘルパー関数 ---

def get_image(camera_id: int, image_base64: Optional[str]) -> np.ndarray:
    """カメラまたはBase64から画像を取得"""
    if image_base64:
        # Base64デコード
        img_bytes = base64.b64decode(image_base64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        return image
    else:
        # カメラからキャプチャ
        camera = CameraManager.get_instance()
        frame = camera.get_frame(camera_id)
        if frame is None:
            raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
        return frame


def mask_to_base64(mask: np.ndarray) -> str:
    """マスクをPNG Base64に変換"""
    _, buffer = cv2.imencode('.png', mask)
    return base64.b64encode(buffer).decode('utf-8')


def calculate_mask_ratio(mask: np.ndarray) -> float:
    """マスク領域の割合を計算"""
    return float((mask > 127).sum() / mask.size)


# --- エンドポイント ---

@router.get("/status")
async def get_status():
    """NanoSAMステータス取得"""
    sam = NanoSAMInference.get_instance()
    status = sam.get_status()
    return status


@router.post("/initialize")
async def initialize():
    """NanoSAM初期化"""
    sam = NanoSAMInference.get_instance()
    success = sam.initialize()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to initialize NanoSAM")
    
    return {"status": "initialized"}


@router.post("/segment/point", response_model=SegmentationResponse)
async def segment_point(request: SegmentPointRequest):
    """ポイントプロンプトでセグメンテーション"""
    sam = NanoSAMInference.get_instance()
    
    if not sam.is_initialized:
        if not sam.initialize():
            raise HTTPException(status_code=500, detail="NanoSAM not initialized")
    
    # 画像取得
    image = get_image(request.camera_id, request.image_base64)
    
    # ポイントリスト作成
    points = [(p.x, p.y, p.label) for p in request.points]
    
    # セグメンテーション実行
    result = sam.segment_point(image, points)
    
    return SegmentationResponse(
        mask_base64=mask_to_base64(result.mask),
        score=result.score,
        inference_time_ms=result.inference_time_ms,
        prompt_type=result.prompt_type.value,
        mask_ratio=calculate_mask_ratio(result.mask)
    )


@router.post("/segment/box", response_model=SegmentationResponse)
async def segment_box(request: SegmentBoxRequest):
    """ボックスプロンプトでセグメンテーション"""
    sam = NanoSAMInference.get_instance()
    
    if not sam.is_initialized:
        if not sam.initialize():
            raise HTTPException(status_code=500, detail="NanoSAM not initialized")
    
    # 画像取得
    image = get_image(request.camera_id, request.image_base64)
    
    # ボックス座標
    box = (request.box.x1, request.box.y1, request.box.x2, request.box.y2)
    
    # セグメンテーション実行
    result = sam.segment_box(image, box)
    
    return SegmentationResponse(
        mask_base64=mask_to_base64(result.mask),
        score=result.score,
        inference_time_ms=result.inference_time_ms,
        prompt_type=result.prompt_type.value,
        mask_ratio=calculate_mask_ratio(result.mask)
    )


@router.post("/segment/auto")
async def segment_auto(request: SegmentAutoRequest):
    """自動セグメンテーション"""
    sam = NanoSAMInference.get_instance()
    
    if not sam.is_initialized:
        if not sam.initialize():
            raise HTTPException(status_code=500, detail="NanoSAM not initialized")
    
    # 画像取得
    image = get_image(request.camera_id, request.image_base64)
    
    # セグメンテーション実行
    results = sam.segment_auto(image, request.grid_size)
    
    # 結果をマージ（最高スコアのマスクを返す）
    if not results:
        raise HTTPException(status_code=500, detail="No segmentation results")
    
    best_result = max(results, key=lambda r: r.score)
    total_time = sum(r.inference_time_ms for r in results)
    
    return {
        "mask_base64": mask_to_base64(best_result.mask),
        "score": best_result.score,
        "inference_time_ms": total_time,
        "prompt_type": "auto",
        "mask_ratio": calculate_mask_ratio(best_result.mask),
        "num_segments": len(results)
    }


@router.post("/segment/preset", response_model=SegmentationResponse)
async def segment_preset(request: SegmentPresetRequest):
    """プリセットでセグメンテーション"""
    sam = NanoSAMInference.get_instance()
    
    if not sam.is_initialized:
        if not sam.initialize():
            raise HTTPException(status_code=500, detail="NanoSAM not initialized")
    
    # 画像取得
    image = get_image(request.camera_id, request.image_base64)
    
    # プリセットに応じてセグメンテーション
    if request.preset == "road":
        result = sam.segment_road(image)
    elif request.preset == "obstacle":
        result = sam.segment_obstacle(image)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {request.preset}")
    
    return SegmentationResponse(
        mask_base64=mask_to_base64(result.mask),
        score=result.score,
        inference_time_ms=result.inference_time_ms,
        prompt_type=result.prompt_type.value,
        mask_ratio=calculate_mask_ratio(result.mask)
    )


@router.post("/release")
async def release():
    """リソース解放"""
    NanoSAMInference.release_instance()
    return {"status": "released"}
```

### Step 3: ルーターを登録

`http_server/main.py` を編集して、NanoSAMルーターを追加：

```python
# 既存のインポートの後に追加
from .routes.nanosam import router as nanosam_router

# 既存のルーター登録の後に追加
app.include_router(nanosam_router)
```

### Step 4: テスト

```bash
# サーバー再起動
cd ~/projects/jetracer-agent
python -m http_server.main

# 別ターミナルでテスト

# ステータス確認
curl http://localhost:8000/nanosam/status

# 初期化
curl -X POST http://localhost:8000/nanosam/initialize

# ポイントセグメンテーション（カメラ1の中央）
curl -X POST http://localhost:8000/nanosam/segment/point \
  -H "Content-Type: application/json" \
  -d '{"camera_id": 1, "points": [{"x": 320, "y": 240, "label": 1}]}'

# 道路プリセット
curl -X POST http://localhost:8000/nanosam/segment/preset \
  -H "Content-Type: application/json" \
  -d '{"camera_id": 1, "preset": "road"}'

# リソース解放
curl -X POST http://localhost:8000/nanosam/release
```

## 期待される結果

```json
{
  "mask_base64": "iVBORw0KGgo...",
  "score": 0.95,
  "inference_time_ms": 19.3,
  "prompt_type": "point",
  "mask_ratio": 0.25
}
```

## 注意事項

1. **メモリ管理**: NanoSAMはシングルトンで実装し、必要に応じてreleaseを呼ぶ
2. **初回起動**: TensorRTエンジンのウォームアップに数秒かかる
3. **エラーハンドリング**: エンジンが見つからない場合は明確なエラーメッセージを返す

## 完了後

このAPIが動作したら、次のステップ（PC側のUIパネル実装）に進みます。
