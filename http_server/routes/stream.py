"""MJPEG ストリーミング エンドポイント"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
import cv2
import time
from typing import Generator

from ..core.camera_manager import camera_manager
from ..config import config

router = APIRouter()


def generate_mjpeg() -> Generator[bytes, None, None]:
    """MJPEG フレーム生成ジェネレーター"""
    frame_interval = 1.0 / 15  # 15 FPS
    last_frame_time = 0

    while True:
        try:
            # FPS制御
            current_time = time.time()
            elapsed = current_time - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            last_frame_time = time.time()

            # フレーム取得
            frame = camera_manager.read()
            if frame is None:
                # カメラが準備できていない場合は待機
                time.sleep(0.1)
                continue

            # JPEG エンコード
            _, jpeg = cv2.imencode(
                '.jpg',
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
            )

            # MJPEG フレーム形式で yield
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                jpeg.tobytes() +
                b'\r\n'
            )

        except GeneratorExit:
            print("[MJPEG] Stream closed by client")
            break
        except Exception as e:
            print(f"[MJPEG] Error: {e}")
            time.sleep(0.5)


@router.get("/stream")
async def mjpeg_stream():
    """
    MJPEG ストリーミングエンドポイント

    ブラウザで直接表示可能:
    <img src="http://jetson:8000/stream" />
    """
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
        }
    )


@router.get("/snapshot")
async def snapshot():
    """
    単一フレーム取得（JPEG画像として返す）
    """
    frame = camera_manager.read()
    if frame is None:
        raise HTTPException(status_code=503, detail="Camera not ready")

    _, jpeg = cv2.imencode(
        '.jpg',
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality]
    )

    return Response(
        content=jpeg.tobytes(),
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache"}
    )
