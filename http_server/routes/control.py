"""POST /control - 車両制御"""
from fastapi import APIRouter
from pydantic import BaseModel, Field
from datetime import datetime

from ..core.vehicle_controller import vehicle_controller
from ..config import config

router = APIRouter()


class ControlRequest(BaseModel):
    steering: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="ステアリング (-1:左, 1:右)"
    )
    throttle: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="スロットル (0~1)"
    )


@router.post("/control")
def set_control(req: ControlRequest):
    """車両制御"""
    vehicle_controller.set_steering(req.steering)
    vehicle_controller.set_throttle(req.throttle, max_limit=config.max_throttle)

    return {
        "timestamp": datetime.now().isoformat(),
        "status": "ok",
        "applied": {
            "steering": req.steering,
            "throttle": min(req.throttle, config.max_throttle)
        }
    }


@router.post("/stop")
def emergency_stop():
    """緊急停止"""
    vehicle_controller.stop()

    return {
        "timestamp": datetime.now().isoformat(),
        "status": "stopped"
    }
