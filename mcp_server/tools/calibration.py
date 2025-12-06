"""キャリブレーション MCP ツール"""
import httpx
from typing import Optional

# HTTP APIのベースURL（同一マシン上で動作）
API_BASE_URL = "http://localhost:8000"


async def get_calibration_status() -> dict:
    """キャリブレーション状態を取得
    
    Returns:
        pattern_size: チェッカーボードパターンサイズ
        square_size_mm: マス目サイズ
        captured_images: 撮影済み画像数
        calibrated: キャリブレーション済みフラグ
        results: キャリブレーション結果
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{API_BASE_URL}/calibration/status")
        resp.raise_for_status()
        return resp.json()


async def detect_checkerboard(camera_id: int) -> dict:
    """チェッカーボードを検出
    
    Args:
        camera_id: カメラID (0 or 1)
        
    Returns:
        detected: 検出成功
        info: 検出情報（位置、カバレッジ）
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{API_BASE_URL}/calibration/detect/{camera_id}")
        resp.raise_for_status()
        result = resp.json()
        
        # Base64画像は除外して返す（LLMには不要）
        return {
            "detected": result.get("detected"),
            "camera_id": result.get("camera_id"),
            "info": result.get("info"),
            "pattern_size": result.get("pattern_size")
        }


async def capture_calibration_pair() -> dict:
    """両カメラでキャリブレーション用画像をペアで撮影
    
    Returns:
        success: 撮影成功
        pair_count: 現在のペア数
        instruction: 次の撮影指示
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{API_BASE_URL}/calibration/capture-stereo")
        resp.raise_for_status()
        result = resp.json()
        
        return {
            "success": result.get("success"),
            "camera0_detected": result.get("camera0", {}).get("detected"),
            "camera1_detected": result.get("camera1", {}).get("detected"),
            "pair_count": result.get("pair_count"),
            "instruction": result.get("instruction")
        }


async def run_calibration() -> dict:
    """キャリブレーションを実行
    
    Returns:
        success: 成功
        camera0_rms: Camera0のRMSエラー
        camera1_rms: Camera1のRMSエラー
        stereo_rms: ステレオRMSエラー
    """
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(f"{API_BASE_URL}/calibration/run")
        resp.raise_for_status()
        result = resp.json()
        
        results = result.get("results", {})
        
        return {
            "success": result.get("success"),
            "camera0_rms": results.get("camera0", {}).get("rms_error"),
            "camera1_rms": results.get("camera1", {}).get("rms_error"),
            "stereo_rms": results.get("stereo", {}).get("rms_error") if results.get("stereo") else None,
            "camera0_images": results.get("camera0", {}).get("num_images"),
            "camera1_images": results.get("camera1", {}).get("num_images")
        }


async def get_calibration_instruction() -> dict:
    """次の撮影指示を取得
    
    Returns:
        status: collecting/ready
        message: 状態メッセージ
        instruction: 撮影指示
        count: 現在の枚数
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{API_BASE_URL}/calibration/instruction")
        resp.raise_for_status()
        return resp.json()


async def clear_calibration_images() -> dict:
    """収集した画像をすべてクリア
    
    Returns:
        success: 成功
        cleared: クリアした対象
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.delete(f"{API_BASE_URL}/calibration/clear")
        resp.raise_for_status()
        return resp.json()
