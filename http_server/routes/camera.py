"""POST /capture - カメラ画像取得 + センサー capabilities API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import cv2
import base64

from ..core.camera_manager import camera_manager
from ..core.sensor_capabilities import sensor_capabilities
from ..config import config

router = APIRouter()


class CaptureRequest(BaseModel):
    """キャプチャリクエスト"""
    camera_id: Optional[int] = 0


class CameraRestartRequest(BaseModel):
    """カメラ再起動リクエスト"""
    camera_id: int = 0
    sensor_mode: Optional[int] = None
    output_width: int = 640
    output_height: int = 480
    fps: int = 30


@router.post("/capture")
def capture_frame(request: CaptureRequest = CaptureRequest()):
    """カメラ画像をBase64で返す
    
    Args:
        request: camera_idを指定（デフォルト: 0）
    """
    camera_id = request.camera_id
    frame = camera_manager.read(camera_id)

    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")

    # JPEG エンコード
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)

    return {
        "timestamp": datetime.now().isoformat(),
        "camera_id": camera_id,
        "image_base64": base64.b64encode(buffer).decode('utf-8'),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "format": "jpeg"
    }


@router.post("/capture/{camera_id}")
def capture_frame_by_id(camera_id: int):
    """指定カメラの画像をBase64で返す（パス指定版）"""
    frame = camera_manager.read(camera_id)

    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")

    # JPEG エンコード
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)

    return {
        "timestamp": datetime.now().isoformat(),
        "camera_id": camera_id,
        "image_base64": base64.b64encode(buffer).decode('utf-8'),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "format": "jpeg"
    }


# ==================== Sensor Capabilities API ====================

@router.get("/capabilities")
def get_all_capabilities():
    """全カメラのセンサー capabilities を取得"""
    return sensor_capabilities.get_all_capabilities()


@router.get("/capabilities/{camera_id}")
def get_camera_capabilities(camera_id: int):
    """指定カメラのセンサー capabilities を取得"""
    if camera_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid camera_id. Use 0 or 1.")
    
    caps = sensor_capabilities.get_capabilities(camera_id)
    
    # 現在の設定を追加
    caps["current_settings"] = camera_manager.get_camera_settings(camera_id)
    caps["is_ready"] = camera_manager.is_ready(camera_id)
    
    return caps


@router.post("/capabilities/probe")
def probe_capabilities(camera_ids: List[int] = None):
    """センサー capabilities を再取得（プローブ）
    
    Args:
        camera_ids: プローブするカメラID（デフォルト: [0, 1]）
    """
    if camera_ids is None:
        camera_ids = [0, 1]
    
    # 不正なIDをフィルタ
    camera_ids = [cid for cid in camera_ids if cid in [0, 1]]
    
    results = sensor_capabilities.initialize(camera_ids)
    
    return {
        "probed": camera_ids,
        "results": results,
        "capabilities": sensor_capabilities.get_all_capabilities()
    }


# ==================== Camera Restart API ====================

@router.post("/restart")
def restart_camera(request: CameraRestartRequest):
    """カメラを指定設定で再起動
    
    センサーモードを変更する場合はカメラの再起動が必要
    """
    camera_id = request.camera_id
    
    if camera_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid camera_id. Use 0 or 1.")
    
    print(f"[Camera API] Restart request: camera={camera_id}, mode={request.sensor_mode}, "
          f"output={request.output_width}x{request.output_height}@{request.fps}fps")
    
    try:
        # カメラ停止
        camera_manager.stop(camera_id)
        
        # 新しい設定で起動
        success = camera_manager.start_with_mode(
            camera_id=camera_id,
            sensor_mode=request.sensor_mode,
            output_width=request.output_width,
            output_height=request.output_height,
            fps=request.fps
        )
        
        if not success:
            raise HTTPException(
                status_code=503, 
                detail=f"Failed to restart camera {camera_id} with new settings"
            )
        
        return {
            "success": True,
            "camera_id": camera_id,
            "settings": {
                "sensor_mode": request.sensor_mode,
                "output_width": request.output_width,
                "output_height": request.output_height,
                "fps": request.fps
            },
            "message": f"Camera {camera_id} restarted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Camera restart error: {e}")


@router.get("/settings/{camera_id}")
def get_camera_settings(camera_id: int):
    """カメラの現在の設定を取得"""
    if camera_id not in [0, 1]:
        raise HTTPException(status_code=400, detail="Invalid camera_id. Use 0 or 1.")
    
    settings = camera_manager.get_camera_settings(camera_id)
    
    if settings is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not initialized")
    
    return settings
