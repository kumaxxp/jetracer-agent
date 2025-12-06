"""GET /status - 車両状態"""
from fastapi import APIRouter
from datetime import datetime

from ..core.camera_manager import camera_manager
from ..core.vehicle_controller import vehicle_controller

router = APIRouter()


@router.get("/status")
def get_status():
    """システム状態を返す"""
    return {
        "timestamp": datetime.now().isoformat(),
        "camera": {
            "connected": camera_manager.is_ready()
        },
        "vehicle": vehicle_controller.get_status(),
        "system": {
            "api_version": "1.0"
        }
    }
