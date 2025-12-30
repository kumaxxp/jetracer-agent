"""センサーフュージョンAPIルート

エンドポイント:
- GET /fusion/motion_state   - 総合状態
- GET /fusion/stuck          - スタック検出
- GET /fusion/lifted         - 持ち上げ検出
- GET /fusion/collision      - 衝突検知
- GET /fusion/imu_drift      - IMUドリフト検出
- POST /fusion/config        - 設定更新
- GET /fusion/config         - 設定取得
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from ..core.fusion import (
    get_motion_estimator,
    get_stuck_detector,
    get_lift_detector,
    get_drift_detector,
    get_collision_detector,
    fusion_config
)

router = APIRouter(prefix="/fusion", tags=["fusion"])


class ConfigUpdateRequest(BaseModel):
    """設定更新リクエスト"""
    motor_running_zcr: Optional[float] = None
    accel_stopped_threshold: Optional[float] = None
    stuck_min_duration: Optional[float] = None
    motor_stopped_zcr: Optional[float] = None
    accel_variance_threshold: Optional[float] = None
    drift_rate_threshold: Optional[float] = None
    impact_rms_threshold: Optional[float] = None
    impact_accel_threshold: Optional[float] = None


@router.get("/motion_state")
async def get_motion_state():
    """総合モーション状態を取得

    Returns:
        state: 状態 (stopped/running/stuck/lifted/imu_drift/collision)
        confidence: 信頼度 (0-1)
        recommended_action: 推奨アクション
        details: 各検出器の詳細結果
    """
    estimator = get_motion_estimator()
    result = estimator.estimate()
    return result.to_dict()


@router.get("/stuck")
async def get_stuck_status():
    """スタック検出状態を取得

    スタック = モーター音あり + 車両移動なし
    """
    detector = get_stuck_detector()
    result = detector.check()
    return result.to_dict()


@router.get("/lifted")
async def get_lift_status():
    """持ち上げ検出状態を取得

    持ち上げ = モーター音なし + 不規則な動き
    """
    detector = get_lift_detector()
    result = detector.check()
    return result.to_dict()


@router.get("/collision")
async def get_collision_status():
    """衝突検知状態を取得

    衝突 = 衝撃音 + 加速度スパイク
    """
    detector = get_collision_detector()
    result = detector.check()
    return result.to_dict()


@router.get("/imu_drift")
async def get_imu_drift_status():
    """IMUドリフト検出状態を取得

    ドリフト = モーター音なし + heading継続変化 + 加速度安定
    """
    detector = get_drift_detector()
    result = detector.check()
    return result.to_dict()


@router.post("/config")
async def update_config(req: ConfigUpdateRequest):
    """フュージョン設定を更新"""
    updates = {}

    if req.motor_running_zcr is not None:
        fusion_config.motor_running_zcr = req.motor_running_zcr
        updates["motor_running_zcr"] = req.motor_running_zcr

    if req.accel_stopped_threshold is not None:
        fusion_config.accel_stopped_threshold = req.accel_stopped_threshold
        updates["accel_stopped_threshold"] = req.accel_stopped_threshold

    if req.stuck_min_duration is not None:
        fusion_config.stuck_min_duration = req.stuck_min_duration
        updates["stuck_min_duration"] = req.stuck_min_duration

    if req.motor_stopped_zcr is not None:
        fusion_config.motor_stopped_zcr = req.motor_stopped_zcr
        updates["motor_stopped_zcr"] = req.motor_stopped_zcr

    if req.accel_variance_threshold is not None:
        fusion_config.accel_variance_threshold = req.accel_variance_threshold
        updates["accel_variance_threshold"] = req.accel_variance_threshold

    if req.drift_rate_threshold is not None:
        fusion_config.drift_rate_threshold = req.drift_rate_threshold
        updates["drift_rate_threshold"] = req.drift_rate_threshold

    if req.impact_rms_threshold is not None:
        fusion_config.impact_rms_threshold = req.impact_rms_threshold
        updates["impact_rms_threshold"] = req.impact_rms_threshold

    if req.impact_accel_threshold is not None:
        fusion_config.impact_accel_threshold = req.impact_accel_threshold
        updates["impact_accel_threshold"] = req.impact_accel_threshold

    return {
        "status": "updated",
        "updates": updates,
        "timestamp": datetime.now().isoformat()
    }


@router.get("/config")
async def get_config():
    """現在のフュージョン設定を取得"""
    return {
        "motor_running_zcr": fusion_config.motor_running_zcr,
        "accel_stopped_threshold": fusion_config.accel_stopped_threshold,
        "stuck_min_duration": fusion_config.stuck_min_duration,
        "motor_stopped_zcr": fusion_config.motor_stopped_zcr,
        "accel_variance_threshold": fusion_config.accel_variance_threshold,
        "gravity_deviation": fusion_config.gravity_deviation,
        "heading_variance_threshold": fusion_config.heading_variance_threshold,
        "drift_rate_threshold": fusion_config.drift_rate_threshold,
        "accel_stable_threshold": fusion_config.accel_stable_threshold,
        "drift_min_duration": fusion_config.drift_min_duration,
        "impact_rms_threshold": fusion_config.impact_rms_threshold,
        "impact_accel_threshold": fusion_config.impact_accel_threshold,
    }
