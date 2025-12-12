"""ベンチマーク用 API エンドポイント"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import time
import cv2
import base64
import numpy as np

from ..core.camera_manager import camera_manager
from ..config import config

router = APIRouter()


class BenchmarkRequest(BaseModel):
    """ベンチマークリクエスト"""
    camera_id: int = 0
    num_frames: int = 10
    include_segmentation: bool = False
    segmentation_model: str = "lightweight"  # "lightweight" or "oneformer"


class SegmentationBenchmarkRequest(BaseModel):
    """セグメンテーションベンチマークリクエスト"""
    camera_id: int = 0
    num_frames: int = 10
    model_type: str = "lightweight"  # "lightweight" or "oneformer"


@router.post("/camera")
def benchmark_camera(request: BenchmarkRequest):
    """カメラキャプチャのベンチマーク
    
    指定フレーム数をキャプチャして、FPSと処理時間を計測
    """
    camera_id = request.camera_id
    num_frames = request.num_frames
    
    if not camera_manager.is_ready(camera_id):
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not ready")
    
    if num_frames < 1 or num_frames > 100:
        raise HTTPException(status_code=400, detail="num_frames must be 1-100")
    
    # ウォームアップ
    for _ in range(3):
        camera_manager.read(camera_id)
    
    # ベンチマーク実行
    frame_times = []
    frames_captured = 0
    
    start_time = time.perf_counter()
    
    for i in range(num_frames):
        t0 = time.perf_counter()
        frame = camera_manager.read(camera_id)
        t1 = time.perf_counter()
        
        if frame is not None:
            frames_captured += 1
            frame_times.append((t1 - t0) * 1000)  # ms
    
    total_time = time.perf_counter() - start_time
    
    # 統計計算
    if frame_times:
        avg_frame_time = sum(frame_times) / len(frame_times)
        min_frame_time = min(frame_times)
        max_frame_time = max(frame_times)
        fps = frames_captured / total_time if total_time > 0 else 0
    else:
        avg_frame_time = min_frame_time = max_frame_time = 0
        fps = 0
    
    # 最後のフレームをプレビュー用に返す
    last_frame_base64 = None
    if frame is not None:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
        last_frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "camera_id": camera_id,
        "num_frames": num_frames,
        "frames_captured": frames_captured,
        "total_time_ms": round(total_time * 1000, 2),
        "fps": round(fps, 2),
        "frame_time_ms": {
            "avg": round(avg_frame_time, 2),
            "min": round(min_frame_time, 2),
            "max": round(max_frame_time, 2)
        },
        "camera_settings": camera_manager.get_camera_settings(camera_id),
        "preview_base64": last_frame_base64,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/segmentation")
def benchmark_segmentation(request: SegmentationBenchmarkRequest):
    """セグメンテーションのベンチマーク
    
    指定フレーム数でセグメンテーション処理を実行してFPSを計測
    """
    camera_id = request.camera_id
    num_frames = request.num_frames
    model_type = request.model_type
    
    if not camera_manager.is_ready(camera_id):
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not ready")
    
    if num_frames < 1 or num_frames > 50:
        raise HTTPException(status_code=400, detail="num_frames must be 1-50")
    
    # モデルタイプに応じて処理を分岐
    if model_type == "lightweight":
        return _benchmark_lightweight(camera_id, num_frames)
    elif model_type == "oneformer":
        return _benchmark_oneformer(camera_id, num_frames)
    else:
        raise HTTPException(status_code=400, detail="model_type must be 'lightweight' or 'oneformer'")


def _benchmark_lightweight(camera_id: int, num_frames: int) -> dict:
    """Lightweightモデルのベンチマーク"""
    try:
        from ..core.lightweight_segmentation import lightweight_segmentation
    except ImportError:
        raise HTTPException(status_code=503, detail="Lightweight segmentation not available")
    
    if not lightweight_segmentation.is_loaded():
        raise HTTPException(status_code=503, detail="Lightweight model not loaded")
    
    # ウォームアップ
    frame = camera_manager.read(camera_id)
    if frame is not None:
        lightweight_segmentation.segment(frame)
    
    # ベンチマーク実行
    inference_times = []
    total_times = []
    results = []
    
    start_time = time.perf_counter()
    
    for i in range(num_frames):
        t0 = time.perf_counter()
        
        frame = camera_manager.read(camera_id)
        if frame is None:
            continue
        
        t1 = time.perf_counter()
        
        result = lightweight_segmentation.segment(frame)
        
        t2 = time.perf_counter()
        
        capture_time = (t1 - t0) * 1000
        inference_time = (t2 - t1) * 1000
        total_time = (t2 - t0) * 1000
        
        inference_times.append(inference_time)
        total_times.append(total_time)
        
        results.append({
            "frame": i,
            "capture_ms": round(capture_time, 2),
            "inference_ms": round(inference_time, 2),
            "total_ms": round(total_time, 2),
            "road_percentage": round(result.get("road_percentage", 0), 2) if result else 0
        })
    
    total_elapsed = time.perf_counter() - start_time
    
    # 統計計算
    if inference_times:
        avg_inference = sum(inference_times) / len(inference_times)
        min_inference = min(inference_times)
        max_inference = max(inference_times)
        avg_total = sum(total_times) / len(total_times)
        seg_fps = len(inference_times) / total_elapsed if total_elapsed > 0 else 0
    else:
        avg_inference = min_inference = max_inference = avg_total = 0
        seg_fps = 0
    
    # 最後の結果をプレビュー用に取得
    preview_base64 = None
    if results and frame is not None:
        result = lightweight_segmentation.segment(frame)
        if result and "overlay_base64" in result:
            preview_base64 = result["overlay_base64"]
    
    return {
        "camera_id": camera_id,
        "model_type": "lightweight",
        "num_frames": num_frames,
        "frames_processed": len(inference_times),
        "total_elapsed_ms": round(total_elapsed * 1000, 2),
        "segmentation_fps": round(seg_fps, 2),
        "inference_time_ms": {
            "avg": round(avg_inference, 2),
            "min": round(min_inference, 2),
            "max": round(max_inference, 2)
        },
        "total_time_ms": {
            "avg": round(avg_total, 2)
        },
        "frame_results": results,
        "preview_base64": preview_base64,
        "timestamp": datetime.now().isoformat()
    }


def _benchmark_oneformer(camera_id: int, num_frames: int) -> dict:
    """OneFormerモデルのベンチマーク（重いので少ないフレーム数推奨）"""
    try:
        from ..core.oneformer_segmentation import oneformer_segmentation
    except ImportError:
        raise HTTPException(status_code=503, detail="OneFormer not available")
    
    # OneFormerは重いのでフレーム数を制限
    num_frames = min(num_frames, 5)
    
    # ベンチマーク実行
    inference_times = []
    results = []
    
    start_time = time.perf_counter()
    
    for i in range(num_frames):
        frame = camera_manager.read(camera_id)
        if frame is None:
            continue
        
        t0 = time.perf_counter()
        result = oneformer_segmentation.segment(frame)
        t1 = time.perf_counter()
        
        inference_time = (t1 - t0) * 1000
        inference_times.append(inference_time)
        
        results.append({
            "frame": i,
            "inference_ms": round(inference_time, 2),
            "num_classes": result.get("num_classes", 0) if result else 0
        })
    
    total_elapsed = time.perf_counter() - start_time
    
    # 統計計算
    if inference_times:
        avg_inference = sum(inference_times) / len(inference_times)
        min_inference = min(inference_times)
        max_inference = max(inference_times)
        seg_fps = len(inference_times) / total_elapsed if total_elapsed > 0 else 0
    else:
        avg_inference = min_inference = max_inference = 0
        seg_fps = 0
    
    return {
        "camera_id": camera_id,
        "model_type": "oneformer",
        "num_frames": num_frames,
        "frames_processed": len(inference_times),
        "total_elapsed_ms": round(total_elapsed * 1000, 2),
        "segmentation_fps": round(seg_fps, 2),
        "inference_time_ms": {
            "avg": round(avg_inference, 2),
            "min": round(min_inference, 2),
            "max": round(max_inference, 2)
        },
        "frame_results": results,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/fps/{camera_id}")
def get_realtime_fps(camera_id: int, sample_frames: int = 30):
    """リアルタイムFPS計測
    
    指定フレーム数をサンプリングしてFPSを返す（軽量版）
    """
    if camera_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid camera_id")
    
    if not camera_manager.is_ready(camera_id):
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not ready")
    
    sample_frames = min(max(sample_frames, 5), 60)
    
    start_time = time.perf_counter()
    frames_captured = 0
    
    for _ in range(sample_frames):
        frame = camera_manager.read(camera_id)
        if frame is not None:
            frames_captured += 1
    
    elapsed = time.perf_counter() - start_time
    fps = frames_captured / elapsed if elapsed > 0 else 0
    
    return {
        "camera_id": camera_id,
        "fps": round(fps, 2),
        "sample_frames": sample_frames,
        "frames_captured": frames_captured,
        "elapsed_ms": round(elapsed * 1000, 2)
    }
