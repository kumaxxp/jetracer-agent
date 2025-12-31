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
    from ..core.camera_manager import camera_manager

    frame = camera_manager.get_latest_frame(camera_id=0)

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
