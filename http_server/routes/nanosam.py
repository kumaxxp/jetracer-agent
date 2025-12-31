"""NanoSAM HTTP APIルート

エンドポイント:
- 基本操作:
  - GET  /nanosam/status         - 状態取得
  - POST /nanosam/initialize     - モデル初期化
  - POST /nanosam/release        - リソース解放

- セグメンテーション（新API）:
  - POST /nanosam/segment/point  - ポイントプロンプト
  - POST /nanosam/segment/box    - ボックスプロンプト
  - POST /nanosam/segment/auto   - 自動セグメンテーション
  - POST /nanosam/segment/preset - プリセット（road/obstacle）

- レガシー互換:
  - POST /nanosam/init           - 初期化（互換性）
  - GET  /nanosam/segment/camera - カメラ画像セグメンテーション
  - POST /nanosam/segment_and_steer - セグメント＋ステアリング
  - GET  /nanosam/steering       - ステアリング推奨値
  - GET  /nanosam/benchmark      - ベンチマーク
"""

import base64
import numpy as np
import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from ..core.nanosam_inference import NanoSAMInference, PromptType
from ..core.nanosam_segmentation import get_nanosam
from ..core.camera_manager import camera_manager

router = APIRouter(prefix="/nanosam", tags=["nanosam"])


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
    camera_id: int = 0
    points: List[Point]
    image_base64: Optional[str] = None  # 画像を直接渡す場合


class SegmentBoxRequest(BaseModel):
    camera_id: int = 0
    box: Box
    image_base64: Optional[str] = None


class SegmentAutoRequest(BaseModel):
    camera_id: int = 0
    grid_size: int = 8
    image_base64: Optional[str] = None


class SegmentPresetRequest(BaseModel):
    camera_id: int = 0
    preset: str  # "road" or "obstacle"
    image_base64: Optional[str] = None


class SegmentationResponse(BaseModel):
    mask_base64: str           # PNG形式のマスク画像
    score: float               # 信頼度スコア
    inference_time_ms: float   # 推論時間
    prompt_type: str           # プロンプトタイプ
    mask_ratio: float          # マスク領域の割合


class InitRequest(BaseModel):
    """初期化リクエスト（レガシー互換）"""
    encoder_path: Optional[str] = None
    decoder_path: Optional[str] = None


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
        frame = camera_manager.get_latest_frame(camera_id)
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


# --- 基本エンドポイント ---

@router.get("/status")
async def get_status():
    """NanoSAMステータス取得"""
    sam = NanoSAMInference.get_instance()
    status = sam.get_status()
    status["timestamp"] = datetime.now().isoformat()
    return status


@router.post("/initialize")
async def initialize():
    """NanoSAM初期化"""
    sam = NanoSAMInference.get_instance()
    success = sam.initialize()

    if not success:
        raise HTTPException(status_code=500, detail="Failed to initialize NanoSAM")

    return {"status": "initialized", "timestamp": datetime.now().isoformat()}


@router.post("/release")
async def release():
    """リソース解放"""
    NanoSAMInference.release_instance()
    return {"status": "released", "timestamp": datetime.now().isoformat()}


# --- 新セグメンテーションAPI ---

@router.post("/segment/point", response_model=SegmentationResponse)
async def segment_point(request: SegmentPointRequest):
    """ポイントプロンプトでセグメンテーション

    ポイント座標と前景/背景ラベルを指定してセグメント。
    """
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
    """ボックスプロンプトでセグメンテーション

    矩形領域を指定してセグメント。
    """
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
    """自動セグメンテーション

    グリッド状のポイントで画像全体をセグメント。
    """
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
        "num_segments": len(results),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/segment/preset", response_model=SegmentationResponse)
async def segment_preset(request: SegmentPresetRequest):
    """プリセットでセグメンテーション

    preset: "road" - 道路領域を抽出
    preset: "obstacle" - 中央の障害物を抽出
    """
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


# --- レガシー互換エンドポイント ---

@router.post("/init")
async def initialize_nanosam_legacy(req: InitRequest = None):
    """NanoSAMモデルを初期化（レガシー互換）"""
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


@router.get("/segment/camera")
async def segment_camera():
    """カメラの現在フレームをセグメンテーション（レガシー互換）"""
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


@router.get("/steering")
async def get_steering(roi_top_ratio: float = 0.5):
    """ステアリング推奨値取得（レガシー互換）"""
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
    """セグメント＋ステアリング一括計算（レガシー互換）"""
    nanosam = get_nanosam()

    if not nanosam.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="NanoSAM not initialized. Call POST /nanosam/init first."
        )

    seg_result = nanosam.segment_from_camera()
    steer_result = nanosam.calculate_steering(roi_top_ratio=roi_top_ratio)

    return {
        "segmentation": seg_result.to_dict(),
        "steering": steer_result.to_dict(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/benchmark")
async def run_benchmark(iterations: int = 20):
    """NanoSAMベンチマーク実行"""
    sam = NanoSAMInference.get_instance()

    if not sam.is_initialized:
        if not sam.initialize():
            raise HTTPException(status_code=503, detail="NanoSAM not initialized")

    import time

    frame = camera_manager.get_latest_frame(camera_id=0)

    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    # ウォームアップ
    for _ in range(3):
        sam.segment_point(frame, [(320, 240, 1)])

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        sam.segment_point(frame, [(320, 240, 1)])
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
