"""キャリブレーション API ルーター"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import cv2
import base64
import numpy as np

from ..core.calibration import calibration_manager
from ..core.camera_manager import camera_manager


router = APIRouter(prefix="/calibration", tags=["calibration"])


class PositionRequest(BaseModel):
    x: float
    y: float


@router.get("/status")
def get_calibration_status():
    """キャリブレーション状態を取得"""
    return calibration_manager.get_status()


@router.post("/detect/{camera_id}")
def detect_checkerboard(camera_id: int):
    """チェッカーボードを検出
    
    Args:
        camera_id: カメラID (0 or 1)
        
    Returns:
        detected: 検出成功
        preview_base64: コーナー描画済み画像（Base64）
        info: 検出情報（位置、カバレッジなど）
    """
    # フレーム取得
    frame = camera_manager.read(camera_id)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Camera {camera_id} not available")
    
    # チェッカーボード検出
    detected, corners, preview = calibration_manager.detect_checkerboard(frame)
    
    if detected:
        # プレビュー画像をBase64に変換
        _, buffer = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
        preview_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 検出情報
        info = calibration_manager.get_detection_info(corners, frame.shape)
        
        return {
            "detected": True,
            "camera_id": camera_id,
            "preview_base64": preview_base64,
            "info": info,
            "pattern_size": calibration_manager.pattern_size
        }
    else:
        # 元画像を返す
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        preview_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {
            "detected": False,
            "camera_id": camera_id,
            "preview_base64": preview_base64,
            "info": None,
            "pattern_size": calibration_manager.pattern_size
        }


@router.post("/capture/{camera_id}")
def capture_single(camera_id: int):
    """単一カメラでキャリブレーション画像を撮影"""
    # フレーム取得
    frame = camera_manager.read(camera_id)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Camera {camera_id} not available")
    
    # チェッカーボード検出
    detected, corners, _ = calibration_manager.detect_checkerboard(frame)
    
    if not detected:
        raise HTTPException(status_code=400, detail="Checkerboard not detected")
    
    # 保存
    result = calibration_manager.capture_calibration_image(camera_id, frame, corners)
    
    # 次の指示を取得
    instruction = calibration_manager.get_capture_instruction()
    
    return {
        **result,
        "instruction": instruction
    }


@router.post("/capture-stereo")
def capture_stereo():
    """両カメラで同時にキャリブレーション画像を撮影"""
    # 両カメラからフレーム取得
    frame0 = camera_manager.read(0)
    frame1 = camera_manager.read(1)
    
    if frame0 is None:
        raise HTTPException(status_code=500, detail="Camera 0 not available")
    if frame1 is None:
        raise HTTPException(status_code=500, detail="Camera 1 not available")
    
    # チェッカーボード検出
    detected0, corners0, preview0 = calibration_manager.detect_checkerboard(frame0)
    detected1, corners1, preview1 = calibration_manager.detect_checkerboard(frame1)
    
    if not detected0 or not detected1:
        # どちらかで検出失敗した場合、プレビューを返す
        def encode_preview(frame, preview, detected):
            img = preview if detected and preview is not None else frame
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
            return base64.b64encode(buffer).decode('utf-8')
        
        return {
            "success": False,
            "camera0": {
                "detected": detected0,
                "preview_base64": encode_preview(frame0, preview0, detected0)
            },
            "camera1": {
                "detected": detected1,
                "preview_base64": encode_preview(frame1, preview1, detected1)
            },
            "message": "両カメラでチェッカーボードを検出できませんでした"
        }
    
    # 両方で検出成功 → 保存
    result = calibration_manager.capture_stereo_pair(
        frame0, corners0, frame1, corners1
    )
    
    # プレビュー画像を追加
    _, buffer0 = cv2.imencode('.jpg', preview0, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, buffer1 = cv2.imencode('.jpg', preview1, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    result["camera0"]["preview_base64"] = base64.b64encode(buffer0).decode('utf-8')
    result["camera1"]["preview_base64"] = base64.b64encode(buffer1).decode('utf-8')
    result["success"] = True
    
    # 次の指示
    result["instruction"] = calibration_manager.get_capture_instruction()
    
    return result


@router.post("/run")
def run_calibration():
    """キャリブレーションを実行"""
    try:
        status = calibration_manager.get_status()
        
        # 画像数チェック
        cam0_count = status["captured_images"].get(0, 0)
        cam1_count = status["captured_images"].get(1, 0)
        
        if cam0_count < 10 and cam1_count < 10:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough images: camera0={cam0_count}, camera1={cam1_count} (need 10+)"
            )
        
        # キャリブレーション実行
        results = calibration_manager.run_full_calibration()
        
        return {
            "success": True,
            "results": results
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run/{camera_id}")
def run_single_calibration(camera_id: int):
    """単一カメラのキャリブレーションを実行"""
    try:
        status = calibration_manager.get_status()
        
        if status["captured_images"][camera_id] < 10:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough images for camera {camera_id}: {status['captured_images'][camera_id]} < 10"
            )
        
        result = calibration_manager.calibrate_single_camera(camera_id)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Calibration failed - check server logs")
        
        return {
            "success": True,
            "camera_id": camera_id,
            "rms_error": result.rms_error,
            "num_images": result.num_images
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/result")
def get_calibration_result():
    """キャリブレーション結果を取得"""
    return calibration_manager.get_status()["results"]


@router.post("/undistort/{camera_id}")
def undistort_image(camera_id: int):
    """歪み補正済み画像を取得"""
    if not calibration_manager.is_calibrated(camera_id):
        raise HTTPException(
            status_code=400,
            detail=f"Camera {camera_id} not calibrated"
        )
    
    # フレーム取得
    frame = camera_manager.read(camera_id)
    if frame is None:
        raise HTTPException(status_code=500, detail=f"Camera {camera_id} not available")
    
    # 歪み補正
    undistorted = calibration_manager.undistort_image(camera_id, frame)
    
    if undistorted is None:
        raise HTTPException(status_code=500, detail="Undistortion failed")
    
    # 比較画像を作成（元画像と補正画像を横に並べる）
    comparison = np.hstack([frame, undistorted])
    
    # Base64エンコード
    _, buffer_orig = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, buffer_undist = cv2.imencode('.jpg', undistorted, [cv2.IMWRITE_JPEG_QUALITY, 85])
    _, buffer_comp = cv2.imencode('.jpg', comparison, [cv2.IMWRITE_JPEG_QUALITY, 85])
    
    return {
        "camera_id": camera_id,
        "original_base64": base64.b64encode(buffer_orig).decode('utf-8'),
        "undistorted_base64": base64.b64encode(buffer_undist).decode('utf-8'),
        "comparison_base64": base64.b64encode(buffer_comp).decode('utf-8')
    }


@router.delete("/clear")
def clear_captured_images(camera_id: Optional[int] = None):
    """収集した画像をクリア"""
    calibration_manager.clear_captured_images(camera_id)
    return {
        "success": True,
        "cleared": "all" if camera_id is None else f"camera_{camera_id}"
    }


@router.get("/instruction")
def get_capture_instruction():
    """次の撮影指示を取得"""
    return calibration_manager.get_capture_instruction()


@router.post("/save")
def save_calibration():
    """キャリブレーション結果を保存"""
    calibration_manager._save_results()
    return {"success": True, "message": "Calibration results saved"}


@router.post("/load")
def load_calibration():
    """キャリブレーション結果を読み込み"""
    calibration_manager._load_results()
    return {
        "success": True,
        "status": calibration_manager.get_status()
    }
