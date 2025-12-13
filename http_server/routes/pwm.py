"""PWM Control API - PWMキャリブレーション用エンドポイント"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from ..core.pwm_control import get_pwm_controller

router = APIRouter(prefix="/pwm", tags=["PWM Control"])


# =========================================================================
# Request/Response Models
# =========================================================================

class PWMParams(BaseModel):
    """PWMパラメータ"""
    pwm_steering: Dict[str, int] = Field(
        description="ステアリングPWM値 (left, center, right)"
    )
    pwm_speed: Dict[str, int] = Field(
        description="スロットルPWM値 (front, stop, back)"
    )


class PWMTestRequest(BaseModel):
    """PWMテストリクエスト"""
    channel: str = Field(description="チャンネル: steering or throttle")
    value: int = Field(ge=0, le=4095, description="PWM値 (0-4095)")


class PWMTestPositionRequest(BaseModel):
    """PWMポジションテストリクエスト"""
    position: str = Field(description="ポジション: center, left, right, stop, forward, backward")


# =========================================================================
# Endpoints
# =========================================================================

@router.get("/status")
def get_pwm_status() -> Dict[str, Any]:
    """PWMコントローラーの状態を取得"""
    pwm = get_pwm_controller()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "available": pwm.is_available(),
        "board_info": pwm.get_board_info(),
        "params": pwm.get_params()
    }


@router.get("/params")
def get_pwm_params() -> Dict[str, Any]:
    """現在のPWMパラメータを取得"""
    pwm = get_pwm_controller()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "params": pwm.get_params()
    }


@router.post("/params")
def save_pwm_params(params: PWMParams) -> Dict[str, Any]:
    """PWMパラメータを保存"""
    pwm = get_pwm_controller()
    
    new_params = {
        "pwm_steering": params.pwm_steering,
        "pwm_speed": params.pwm_speed
    }
    
    pwm.save_params(new_params)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "saved",
        "params": pwm.get_params()
    }


@router.post("/test/steering/center")
def test_steering_center() -> Dict[str, Any]:
    """ステアリング中央をテスト"""
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_steering_center()
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/test/steering/left")
def test_steering_left() -> Dict[str, Any]:
    """ステアリング左をテスト"""
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_steering_left()
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/test/steering/right")
def test_steering_right() -> Dict[str, Any]:
    """ステアリング右をテスト"""
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_steering_right()
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/test/steering/range")
def test_steering_range() -> Dict[str, Any]:
    """ステアリング可動範囲をテスト（左→右→中央）"""
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_steering_range()
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/test/steering/value/{value}")
def test_steering_value(value: int) -> Dict[str, Any]:
    """ステアリングに任意のPWM値を設定"""
    if not 0 <= value <= 4095:
        raise HTTPException(status_code=400, detail="Value must be 0-4095")
    
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_steering_value(value)
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/test/throttle/stop")
def test_throttle_stop() -> Dict[str, Any]:
    """スロットル停止をテスト"""
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_throttle_stop()
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/test/throttle/forward")
def test_throttle_forward() -> Dict[str, Any]:
    """スロットル前進をテスト"""
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_throttle_forward()
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/test/throttle/backward")
def test_throttle_backward() -> Dict[str, Any]:
    """スロットル後退をテスト（ESCシーケンス付き）"""
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_throttle_backward()
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/test/throttle/value/{value}")
def test_throttle_value(value: int) -> Dict[str, Any]:
    """スロットルに任意のPWM値を設定"""
    if not 0 <= value <= 4095:
        raise HTTPException(status_code=400, detail="Value must be 0-4095")
    
    pwm = get_pwm_controller()
    
    if not pwm.is_available():
        raise HTTPException(status_code=503, detail="PWM controller not available")
    
    result = pwm.test_throttle_value(value)
    result["timestamp"] = datetime.now().isoformat()
    return result


@router.post("/stop")
def emergency_stop() -> Dict[str, Any]:
    """緊急停止"""
    pwm = get_pwm_controller()
    
    success = pwm.stop()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "status": "stopped" if success else "failed",
        "success": success
    }
