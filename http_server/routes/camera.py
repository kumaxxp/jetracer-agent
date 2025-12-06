"""POST /capture - カメラ画像取得"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import cv2
import base64

from ..core.camera_manager import camera_manager
from ..config import config

router = APIRouter()


@router.post("/capture")
def capture_frame():
    """カメラ画像をBase64で返す"""
    frame = camera_manager.read()

    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not available")

    # JPEG エンコード
    encode_param = [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)

    return {
        "timestamp": datetime.now().isoformat(),
        "image_base64": base64.b64encode(buffer).decode('utf-8'),
        "width": frame.shape[1],
        "height": frame.shape[0],
        "format": "jpeg"
    }
