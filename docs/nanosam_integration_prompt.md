# NanoSAM jetracer-agent統合プロンプト

このプロンプトをJetsonのClaude Codeに貼り付けて実行してください。

---

## タスク概要

NanoSAMをjetracer-agentのHTTP APIに統合します。
既に生成済みのTensorRTエンジンと最適化版Predictorを使用します。

**前提条件（完了済み）:**
- `~/projects/nanosam/data/resnet18_image_encoder.engine` 存在
- `~/projects/nanosam/data/mobile_sam_mask_decoder.engine` 存在
- `~/projects/nanosam/nanosam/utils/predictor_optimized.py` 存在

## 実装内容

### 1. NanoSAMコアモジュール

**ファイル: `http_server/core/nanosam_segmentation.py`**

```python
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
            from nanosam.utils.predictor_optimized import Predictor
            _predictor_class = Predictor
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
            
            # マスク処理
            mask_binary = (mask[0] > 0).astype(np.uint8)
            
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
        from .camera_manager import get_camera_manager
        
        camera = get_camera_manager()
        frame = camera.get_frame()
        
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
```

### 2. APIルート

**ファイル: `http_server/routes/nanosam.py`**

```python
"""NanoSAM APIルート

エンドポイント:
- POST /nanosam/init           - モデル初期化
- GET  /nanosam/status         - 状態取得
- POST /nanosam/segment        - 画像セグメンテーション
- GET  /nanosam/segment/camera - カメラ画像をセグメンテーション
- GET  /nanosam/road_mask      - 走行可能領域マスク取得
- GET  /nanosam/steering       - ステアリング推奨値取得
- POST /nanosam/segment_and_steer - セグメント＋ステアリング一括
"""
import base64
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from ..core.nanosam_segmentation import get_nanosam

router = APIRouter(prefix="/nanosam", tags=["nanosam"])


class InitRequest(BaseModel):
    """初期化リクエスト"""
    encoder_path: Optional[str] = None
    decoder_path: Optional[str] = None


class SegmentRequest(BaseModel):
    """セグメンテーションリクエスト"""
    image_base64: str              # Base64エンコードされた画像
    point_x: Optional[int] = None  # クリックポイントX
    point_y: Optional[int] = None  # クリックポイントY


class SteeringRequest(BaseModel):
    """ステアリング計算リクエスト"""
    roi_top_ratio: float = 0.5     # ROI上端位置


@router.post("/init")
async def initialize_nanosam(req: InitRequest = None):
    """NanoSAMモデルを初期化
    
    TensorRTエンジンをロードします。初回は数秒かかります。
    """
    nanosam = get_nanosam()
    
    if req and req.encoder_path:
        nanosam.encoder_path = req.encoder_path
    if req and req.decoder_path:
        nanosam.decoder_path = req.decoder_path
    
    success, message = nanosam.initialize()
    
    return {
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status")
async def get_status():
    """NanoSAMの状態を取得"""
    nanosam = get_nanosam()
    status = nanosam.get_status()
    status["timestamp"] = datetime.now().isoformat()
    return status


@router.post("/segment")
async def segment_image(req: SegmentRequest):
    """Base64画像をセグメンテーション
    
    Args:
        image_base64: Base64エンコードされたJPEG/PNG画像
        point_x, point_y: クリックポイント（省略時は画像下部中央）
    
    Returns:
        road_ratio: 走行可能領域の割合
        inference_time_ms: 推論時間
        mask_base64: マスク画像（PNG、Base64）
    """
    nanosam = get_nanosam()
    
    if not nanosam.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="NanoSAM not initialized. Call POST /nanosam/init first."
        )
    
    # Base64デコード
    try:
        import cv2
        image_data = base64.b64decode(req.image_base64)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
    
    # ポイント設定
    point = None
    if req.point_x is not None and req.point_y is not None:
        point = (req.point_x, req.point_y)
    
    # セグメンテーション実行
    result = nanosam.segment_image(image, point)
    
    # マスクをBase64エンコード
    mask_base64 = nanosam.get_road_mask_encoded()
    
    return {
        **result.to_dict(),
        "mask_base64": mask_base64
    }


@router.get("/segment/camera")
async def segment_camera():
    """カメラの現在フレームをセグメンテーション
    
    Returns:
        road_ratio: 走行可能領域の割合
        inference_time_ms: 推論時間
        mask_base64: マスク画像（PNG、Base64）
    """
    nanosam = get_nanosam()
    
    if not nanosam.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="NanoSAM not initialized. Call POST /nanosam/init first."
        )
    
    result = nanosam.segment_from_camera()
    mask_base64 = nanosam.get_road_mask_encoded()
    
    return {
        **result.to_dict(),
        "mask_base64": mask_base64
    }


@router.get("/road_mask")
async def get_road_mask(format: str = "json"):
    """走行可能領域マスクを取得
    
    Args:
        format: "json" (Base64) or "png" (バイナリ)
    """
    nanosam = get_nanosam()
    
    mask_base64 = nanosam.get_road_mask_encoded()
    
    if mask_base64 is None:
        raise HTTPException(
            status_code=404,
            detail="No mask available. Run segmentation first."
        )
    
    if format == "png":
        mask_bytes = base64.b64decode(mask_base64)
        return Response(content=mask_bytes, media_type="image/png")
    
    return {
        "mask_base64": mask_base64,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/steering")
async def get_steering(roi_top_ratio: float = 0.5):
    """最後のセグメンテーション結果からステアリング値を計算
    
    Args:
        roi_top_ratio: ROI上端位置（0.0-1.0、デフォルト0.5=下半分）
    
    Returns:
        steering: ステアリング値（-1.0=左, 0=中央, 1.0=右）
        confidence: 信頼度
        road_center_x: 走行可能領域の中心X
        road_width_ratio: 走行可能領域の幅比率
    """
    nanosam = get_nanosam()
    
    result = nanosam.calculate_steering(roi_top_ratio=roi_top_ratio)
    
    if result.confidence == 0:
        raise HTTPException(
            status_code=404,
            detail="No valid road mask. Run segmentation first."
        )
    
    return result.to_dict()


@router.post("/segment_and_steer")
async def segment_and_calculate_steering(roi_top_ratio: float = 0.5):
    """カメラ画像をセグメントしてステアリング値を一括計算
    
    リアルタイム走行用の統合API。
    
    Returns:
        segmentation: セグメンテーション結果
        steering: ステアリング計算結果
    """
    nanosam = get_nanosam()
    
    if not nanosam.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="NanoSAM not initialized. Call POST /nanosam/init first."
        )
    
    # セグメンテーション
    seg_result = nanosam.segment_from_camera()
    
    # ステアリング計算
    steer_result = nanosam.calculate_steering(roi_top_ratio=roi_top_ratio)
    
    return {
        "segmentation": seg_result.to_dict(),
        "steering": steer_result.to_dict(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/benchmark")
async def run_benchmark(iterations: int = 20):
    """NanoSAMベンチマーク実行
    
    Args:
        iterations: 繰り返し回数
    """
    nanosam = get_nanosam()
    
    if not nanosam.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="NanoSAM not initialized."
        )
    
    import time
    from ..core.camera_manager import get_camera_manager
    
    camera = get_camera_manager()
    frame = camera.get_frame()
    
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")
    
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        nanosam.segment_image(frame)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    
    return {
        "iterations": iterations,
        "mean_ms": round(np.mean(times), 2),
        "std_ms": round(np.std(times), 2),
        "min_ms": round(np.min(times), 2),
        "max_ms": round(np.max(times), 2),
        "fps": round(1000 / np.mean(times), 1),
        "timestamp": datetime.now().isoformat()
    }
```

### 3. main.pyへの統合

`http_server/main.py` に以下を追加:

```python
# 既存のimport群の後に追加
from .routes.nanosam import router as nanosam_router

# 既存のrouter登録の後に追加
app.include_router(nanosam_router)
```

### 4. 依存関係の確認

NanoSAMモジュールへのパスが通っていることを確認:

```bash
# nanosam がインポートできることを確認
cd ~/projects/jetracer-agent
source venv/bin/activate
python3 -c "
import sys
sys.path.insert(0, '/home/jetson/projects/nanosam')
from nanosam.utils.predictor_optimized import Predictor
print('OK: NanoSAM predictor_optimized available')
"
```

## テストコマンド

実装後、以下のコマンドでテスト:

```bash
# サーバー起動
cd ~/projects/jetracer-agent
source venv/bin/activate
python -m http_server.main

# 別ターミナルで

# 1. NanoSAM初期化（数秒かかる）
curl -X POST http://localhost:8000/nanosam/init
# → {"success": true, "message": "Initialized successfully", ...}

# 2. 状態確認
curl http://localhost:8000/nanosam/status
# → {"initialized": true, "encoder_path": "...", ...}

# 3. カメラ画像をセグメンテーション
curl http://localhost:8000/nanosam/segment/camera
# → {"road_ratio": 0.35, "inference_time_ms": 19.5, ...}

# 4. ステアリング値取得
curl http://localhost:8000/nanosam/steering
# → {"steering": 0.12, "confidence": 0.85, ...}

# 5. セグメント＋ステアリング一括
curl -X POST http://localhost:8000/nanosam/segment_and_steer
# → {"segmentation": {...}, "steering": {...}}

# 6. ベンチマーク
curl "http://localhost:8000/nanosam/benchmark?iterations=20"
# → {"mean_ms": 19.3, "fps": 51.8, ...}
```

## 完了チェックリスト

- [ ] `http_server/core/nanosam_segmentation.py` 作成
- [ ] `http_server/routes/nanosam.py` 作成
- [ ] `main.py` に router 追加
- [ ] NanoSAMモジュールのパス確認
- [ ] サーバー起動確認
- [ ] `/nanosam/init` 動作確認
- [ ] `/nanosam/segment/camera` 動作確認
- [ ] `/nanosam/steering` 動作確認
- [ ] `/nanosam/segment_and_steer` 動作確認

## 期待される結果

| API | 期待値 |
|-----|-------|
| 初期化時間 | 2-5秒 |
| セグメンテーション時間 | ~20ms |
| ステアリング計算 | ~1ms |
| segment_and_steer 合計 | ~25ms |
