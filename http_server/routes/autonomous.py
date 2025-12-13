"""自律走行API"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from ..core.autonomous_controller import autonomous_controller, ControlMode
from ..core.steering_calculator import steering_calculator
from ..core.safety_guard import safety_guard
from ..core.data_collector import data_collector

router = APIRouter(prefix="/auto", tags=["autonomous"])


# =============================================================================
# リクエスト/レスポンスモデル
# =============================================================================

class StartRequest(BaseModel):
    """自律走行開始リクエスト"""
    use_dual_camera: bool = False
    use_lightweight_model: bool = True


class SteeringParamsUpdate(BaseModel):
    """ステアリングパラメータ更新"""
    steering_gain: Optional[float] = None
    steering_deadzone: Optional[float] = None
    throttle_base: Optional[float] = None
    throttle_min: Optional[float] = None
    throttle_max: Optional[float] = None
    curve_reduction: Optional[float] = None
    road_stop_threshold: Optional[float] = None
    road_slow_threshold: Optional[float] = None


class SafetyParamsUpdate(BaseModel):
    """安全パラメータ更新"""
    lidar_stop_distance_mm: Optional[int] = None
    lidar_slow_distance_mm: Optional[int] = None
    road_stop_threshold: Optional[float] = None
    road_slow_threshold: Optional[float] = None
    tilt_stop_threshold_deg: Optional[float] = None
    lidar_enabled: Optional[bool] = None
    imu_enabled: Optional[bool] = None
    road_check_enabled: Optional[bool] = None


class CollectStartRequest(BaseModel):
    """データ収集開始リクエスト"""
    session_id: Optional[str] = None
    fps: float = 1.0
    save_ground_camera: bool = False


# =============================================================================
# 自律走行制御API
# =============================================================================

@router.post("/start")
async def start_autonomous(req: StartRequest = None):
    """自律走行を開始"""
    req = req or StartRequest()
    
    # 設定更新
    autonomous_controller.update_config(
        use_dual_camera=req.use_dual_camera,
        use_lightweight_model=req.use_lightweight_model
    )
    
    success = await autonomous_controller.start()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start autonomous mode")
    
    return {
        "success": True,
        "message": "Autonomous mode started",
        "config": autonomous_controller.get_config(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/stop")
async def stop_autonomous():
    """自律走行を停止"""
    await autonomous_controller.stop()
    
    return {
        "success": True,
        "message": "Autonomous mode stopped",
        "timestamp": datetime.now().isoformat()
    }


@router.post("/emergency")
async def emergency_stop(reason: str = "API trigger"):
    """緊急停止"""
    autonomous_controller.emergency_stop(reason)
    
    return {
        "success": True,
        "message": "Emergency stop triggered",
        "reason": reason,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/clear-emergency")
async def clear_emergency():
    """緊急停止を解除"""
    success = autonomous_controller.clear_emergency()
    
    if not success:
        return {
            "success": False,
            "message": "Not in emergency state",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "success": True,
        "message": "Emergency cleared",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status")
async def get_status():
    """現在の状態を取得"""
    return {
        "controller": autonomous_controller.get_state(),
        "safety": safety_guard.get_params(),
        "steering": steering_calculator.get_params(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/state")
async def get_state():
    """制御状態のみを取得（軽量）"""
    return autonomous_controller.get_state()


# =============================================================================
# パラメータ設定API
# =============================================================================

@router.get("/params/steering")
async def get_steering_params():
    """ステアリングパラメータを取得"""
    return {
        "params": steering_calculator.get_params(),
        "timestamp": datetime.now().isoformat()
    }


@router.put("/params/steering")
async def update_steering_params(req: SteeringParamsUpdate):
    """ステアリングパラメータを更新"""
    updates = {k: v for k, v in req.dict().items() if v is not None}
    
    if not updates:
        raise HTTPException(status_code=400, detail="No parameters to update")
    
    steering_calculator.update_params(**updates)
    
    return {
        "success": True,
        "updated": list(updates.keys()),
        "params": steering_calculator.get_params(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/params/safety")
async def get_safety_params():
    """安全パラメータを取得"""
    return {
        "params": safety_guard.get_params(),
        "timestamp": datetime.now().isoformat()
    }


@router.put("/params/safety")
async def update_safety_params(req: SafetyParamsUpdate):
    """安全パラメータを更新"""
    updates = {k: v for k, v in req.dict().items() if v is not None}
    
    if not updates:
        raise HTTPException(status_code=400, detail="No parameters to update")
    
    safety_guard.update_params(**updates)
    
    return {
        "success": True,
        "updated": list(updates.keys()),
        "params": safety_guard.get_params(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/config")
async def get_config():
    """コントローラー設定を取得"""
    return {
        "config": autonomous_controller.get_config(),
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# データ収集API
# =============================================================================

@router.post("/collect/start")
async def start_collecting(req: CollectStartRequest = None):
    """データ収集を開始"""
    req = req or CollectStartRequest()
    
    # 設定更新
    data_collector.update_config(
        fps=req.fps,
        save_ground_camera=req.save_ground_camera
    )
    
    # セッション開始
    session = data_collector.start_session(req.session_id)
    
    # 収集開始
    success = await data_collector.start_collecting()
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to start data collection")
    
    return {
        "success": True,
        "session": session.to_dict(),
        "timestamp": datetime.now().isoformat()
    }


@router.post("/collect/stop")
async def stop_collecting():
    """データ収集を停止"""
    await data_collector.stop_collecting()
    session = data_collector.end_session()
    
    return {
        "success": True,
        "session": session.to_dict() if session else None,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/collect/status")
async def get_collect_status():
    """データ収集状態を取得"""
    return {
        "collecting": data_collector.is_collecting(),
        "session": data_collector.get_session_info(),
        "config": data_collector.get_config(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/collect/sessions")
async def list_sessions():
    """セッション一覧を取得"""
    sessions = data_collector.list_sessions()
    
    return {
        "sessions": sessions,
        "count": len(sessions),
        "timestamp": datetime.now().isoformat()
    }


# =============================================================================
# デバッグAPI
# =============================================================================

@router.get("/debug/steering")
async def debug_steering():
    """ステアリング計算のデバッグ情報"""
    from ..core.camera_manager import camera_manager
    from ..core.lightweight_segmentation import lightweight_segmentation
    
    result = {
        "timestamp": datetime.now().isoformat()
    }
    
    # カメラキャプチャ
    frame = camera_manager.capture(0)
    if frame is None:
        result["error"] = "Failed to capture frame"
        return result
    
    # セグメンテーション
    if not lightweight_segmentation.is_loaded():
        lightweight_segmentation.load()
    
    seg_result = lightweight_segmentation.segment(frame)
    if seg_result is None:
        result["error"] = "Failed to segment"
        return result
    
    # 分析
    analysis = steering_calculator.analyze_road_mask(seg_result["mask"])
    command = steering_calculator.calculate_steering(analysis)
    
    result["segmentation"] = {
        "road_percentage": seg_result["road_percentage"],
        "inference_time_ms": seg_result["inference_time_ms"],
    }
    result["analysis"] = {
        "road_ratio": round(analysis.road_ratio, 3),
        "centroid_x": round(analysis.centroid_x, 3),
        "left_ratio": round(analysis.left_ratio, 3),
        "center_ratio": round(analysis.center_ratio, 3),
        "right_ratio": round(analysis.right_ratio, 3),
        "boundary_left": analysis.boundary_left,
        "boundary_right": analysis.boundary_right,
    }
    result["command"] = {
        "steering": command.steering,
        "throttle": command.throttle,
        "stop": command.stop,
        "reason": command.reason,
        "raw_steering": round(command.raw_steering, 3),
    }
    
    return result


@router.get("/debug/safety")
async def debug_safety():
    """安全チェックのデバッグ情報"""
    from ..core.i2c_sensors import sensor_manager
    
    result = {
        "timestamp": datetime.now().isoformat()
    }
    
    # センサー読み取り
    imu_data = sensor_manager.read_imu()
    lidar_data = sensor_manager.read_distance()
    
    # 安全チェック
    status = safety_guard.check(
        lidar_data=lidar_data,
        road_ratio=0.5,  # ダミー
        imu_data=imu_data
    )
    
    result["sensors"] = {
        "imu_available": imu_data is not None and imu_data.valid if imu_data else False,
        "lidar_available": lidar_data is not None and lidar_data.valid if lidar_data else False,
    }
    
    if imu_data and imu_data.valid:
        result["imu"] = {
            "heading": round(imu_data.heading, 1),
            "roll": round(imu_data.roll, 1),
            "pitch": round(imu_data.pitch, 1),
        }
    
    if lidar_data and lidar_data.valid:
        result["lidar"] = {
            "min_distance": lidar_data.min_distance,
            "max_distance": lidar_data.max_distance,
            "avg_distance": round(lidar_data.avg_distance, 1),
        }
    
    result["safety"] = status.to_dict()
    
    return result
