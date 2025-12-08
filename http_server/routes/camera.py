"""POST /capture - カメラ画像取得"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import cv2
import base64

from ..core.camera_manager import camera_manager
from ..config import config

router = APIRouter()


class CaptureRequest(BaseModel):
    """キャプチャリクエスト"""
    camera_id: Optional[int] = 0


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
