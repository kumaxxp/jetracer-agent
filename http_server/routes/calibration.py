"""キャリブレーション API ルーター"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dataclasses import asdict
import cv2
import base64
import numpy as np

from ..core.calibration import calibration_manager
from ..core.camera_manager import camera_manager


router = APIRouter(prefix="/calibration", tags=["calibration"])


# キャリブレーション用高解像度設定
CALIB_WIDTH = 640
CALIB_HEIGHT = 480


def get_high_res_frame(camera_id: int) -> Optional[np.ndarray]:
    """キャリブレーション用に高解像度フレームを取得
    
    方法1: 通常のカメラから取得し、1280x720 -> 640x480 にリサイズ
    方法2: カメラが使用中の場合は既存フレームを拡大
    """
    try:
        # まず既存のカメラマネージャーからフレームを取得
        frame = camera_manager.read_raw(camera_id)
        if frame is not None:
            # 元のサイズが640x480未満なら拡大
            h, w = frame.shape[:2]
            if w < CALIB_WIDTH or h < CALIB_HEIGHT:
                frame = cv2.resize(frame, (CALIB_WIDTH, CALIB_HEIGHT), interpolation=cv2.INTER_LINEAR)
                print(f"[Calibration] Upscaled frame: {w}x{h} -> {CALIB_WIDTH}x{CALIB_HEIGHT}")
            elif w > CALIB_WIDTH or h > CALIB_HEIGHT:
                frame = cv2.resize(frame, (CALIB_WIDTH, CALIB_HEIGHT))
                print(f"[Calibration] Downscaled frame: {w}x{h} -> {CALIB_WIDTH}x{CALIB_HEIGHT}")
            else:
                print(f"[Calibration] Frame size OK: {w}x{h}")
            return frame
        
        print(f"[Calibration] camera_manager.read_raw returned None, trying direct capture")
        
        # フォールバック: 直接カメラを開く（既存のカメラが停止している場合）
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[Calibration] Failed to open camera {camera_id}")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        for _ in range(3):
            ret, frame = cap.read()
        
        cap.release()
        
        if not ret or frame is None:
            return None
        
        frame = cv2.resize(frame, (CALIB_WIDTH, CALIB_HEIGHT))
        print(f"[Calibration] Got direct capture frame: {frame.shape}")
        return frame
        
    except Exception as e:
        print(f"[Calibration] High-res capture error: {e}")
        return None


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
    
    複数フレームを試して検出率を向上させる
    
    Args:
        camera_id: カメラID (0 or 1)
        
    Returns:
        detected: 検出成功
        preview_base64: コーナー描画済み画像（Base64）
        info: 検出情報（位置、カバレッジなど）
    """
    import time
    
    MAX_ATTEMPTS = 10  # 最大試行回数
    FRAME_INTERVAL = 0.1  # フレーム間隔（秒）
    
    last_frame = None
    
    for attempt in range(MAX_ATTEMPTS):
        # フレーム取得
        frame = camera_manager.read_raw(camera_id)  # 歪み補正なしで取得
        if frame is None:
            if attempt == MAX_ATTEMPTS - 1:
                raise HTTPException(status_code=500, detail=f"Camera {camera_id} not available")
            time.sleep(FRAME_INTERVAL)
            continue
        
        last_frame = frame.copy()
        
        # チェッカーボード検出
        detected, corners, preview, detected_pattern = calibration_manager.detect_checkerboard(frame)
        
        if detected:
            print(f"[Calibration] Camera {camera_id}: Detected on attempt {attempt + 1}/{MAX_ATTEMPTS}")
            
            # 検出結果を保持（Capture用）
            calibration_manager.store_detection(camera_id, frame, corners, detected_pattern)
            
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
                "pattern_size": detected_pattern,
                "expected_pattern_size": calibration_manager.pattern_size,
                "attempts": attempt + 1
            }
        
        # 次のフレームまで待機
        time.sleep(FRAME_INTERVAL)
    
    # 全ての試行で失敗
    print(f"[Calibration] Camera {camera_id}: Detection failed after {MAX_ATTEMPTS} attempts")
    
    # 最後のフレームを返す
    if last_frame is not None:
        _, buffer = cv2.imencode('.jpg', last_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        preview_base64 = base64.b64encode(buffer).decode('utf-8')
    else:
        preview_base64 = None
    
    return {
        "detected": False,
        "camera_id": camera_id,
        "preview_base64": preview_base64,
        "info": None,
        "pattern_size": None,
        "expected_pattern_size": calibration_manager.pattern_size,
        "attempts": MAX_ATTEMPTS
    }


@router.post("/capture/{camera_id}")
def capture_single(camera_id: int):
    """単一カメラでキャリブレーション画像を撮影
    
    Detectで保持された結果を使用。期限切れの場合は再検出。
    """
    # まず保持された検出結果を確認（10秒以内）
    stored = calibration_manager.get_stored_detection(camera_id, max_age_seconds=10.0)
    
    if stored:
        # 保持された結果を使用
        frame = stored["frame"]
        corners = stored["corners"]
        detected_pattern = stored["pattern_size"]
        print(f"[Calibration] Camera {camera_id}: Using stored detection")
    else:
        # 保持された結果がない/期限切れの場合は新たに検出
        frame = camera_manager.read(camera_id)
        if frame is None:
            raise HTTPException(status_code=500, detail=f"Camera {camera_id} not available")
        
        detected, corners, _, detected_pattern = calibration_manager.detect_checkerboard(frame)
        
        if not detected:
            raise HTTPException(
                status_code=400, 
                detail="Checkerboard not detected. Please run Detect first."
            )
    
    # 保存（パターンサイズも渡す）
    result = calibration_manager.capture_calibration_image(camera_id, frame, corners, detected_pattern)
    
    # 使用済みの検出結果をクリア
    calibration_manager.clear_stored_detection(camera_id)
    
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
    
    # チェッカーボード検出（4つの戻り値）
    detected0, corners0, preview0, pattern0 = calibration_manager.detect_checkerboard(frame0)
    detected1, corners1, preview1, pattern1 = calibration_manager.detect_checkerboard(frame1)
    
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
    
    # パターンサイズが一致しているか確認
    if pattern0 != pattern1:
        return {
            "success": False,
            "camera0": {"detected": True, "pattern_size": pattern0},
            "camera1": {"detected": True, "pattern_size": pattern1},
            "message": f"Pattern size mismatch: camera0={pattern0}, camera1={pattern1}"
        }
    
    # 両方で検出成功 → 保存（パターンサイズも渡す）
    result = calibration_manager.capture_stereo_pair(
        frame0, corners0, pattern0,
        frame1, corners1, pattern1
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
    """キャリブレーションを実行（単体のみ、ステレオは含まない）"""
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
        
        # 単体キャリブレーションのみ実行（ステレオは別エンドポイント）
        results = {}
        
        if cam0_count >= 10:
            result0 = calibration_manager.calibrate_single_camera(0)
            results["camera0"] = asdict(result0) if result0 else None
        else:
            results["camera0"] = None
        
        if cam1_count >= 10:
            result1 = calibration_manager.calibrate_single_camera(1)
            results["camera1"] = asdict(result1) if result1 else None
        else:
            results["camera1"] = None
        
        results["stereo"] = None  # ステレオは別途実行
        
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


@router.post("/run-stereo")
def run_stereo_calibration():
    """ステレオキャリブレーションのみ実行（既存の単体キャリブレーション結果を使用）"""
    try:
        # 両カメラのキャリブレーション結果が必要
        if not calibration_manager.is_calibrated(0) or not calibration_manager.is_calibrated(1):
            raise HTTPException(
                status_code=400,
                detail="Both cameras must be calibrated first. Run single camera calibration."
            )
        
        # タイムスタンプベースでステレオペア数を取得
        pair_count = calibration_manager.get_stereo_pair_count(max_time_diff_seconds=1.0)
        
        if pair_count < 10:
            status = calibration_manager.get_status()
            cam0_count = status["captured_images"].get(0, 0)
            cam1_count = status["captured_images"].get(1, 0)
            raise HTTPException(
                status_code=400,
                detail=f"Not enough stereo pairs: {pair_count} valid pairs found (need 10+). "
                       f"Images: cam0={cam0_count}, cam1={cam1_count}. "
                       f"Use 'Capture Stereo Pair' to capture simultaneous images."
            )
        
        # ステレオキャリブレーションのみ実行
        stereo_result = calibration_manager.calibrate_stereo()
        
        if stereo_result is None:
            raise HTTPException(
                status_code=500,
                detail="Stereo calibration failed. Check server logs."
            )
        
        return {
            "success": True,
            "rms_error": stereo_result.rms_error,
            "translation_vector": stereo_result.translation_vector,
            "num_images": stereo_result.num_images
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
    
    # カメラマネージャーにも読み込み
    camera_manager.load_calibration_from_manager()
    
    return {
        "success": True,
        "status": calibration_manager.get_status()
    }


@router.post("/undistort-mode/{camera_id}")
def set_undistort_mode(camera_id: int, enabled: bool = True):
    """歪み補正のON/OFFを設定
    
    Args:
        camera_id: カメラID (0 or 1)
        enabled: 有効にするか
    """
    # キャリブレーションデータがない場合は読み込みを試みる
    if enabled and not camera_manager.has_calibration(camera_id):
        camera_manager.load_calibration_from_manager()
    
    success = camera_manager.set_undistort_enabled(camera_id, enabled)
    
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"Camera {camera_id} not calibrated. Run calibration first."
        )
    
    return {
        "success": True,
        "camera_id": camera_id,
        "undistort_enabled": enabled
    }


@router.get("/undistort-status")
def get_undistort_status():
    """歪み補正の状態を取得"""
    return camera_manager.get_undistort_status()
