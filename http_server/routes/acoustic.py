"""
音響APIエンドポイント（Jetson側）

エンドポイント:
- GET  /acoustic/devices     - オーディオデバイス一覧
- POST /acoustic/start       - キャプチャ開始
- POST /acoustic/stop        - キャプチャ停止
- GET  /acoustic/status      - キャプチャ状態
- GET  /acoustic/features    - 現在の特徴量取得
- GET  /acoustic/state       - 現在の状態取得
- POST /acoustic/record/start - 録音開始
- POST /acoustic/record/stop  - 録音停止
- POST /acoustic/record/frame - フレーム記録
- GET  /acoustic/record/data  - 録音データ取得

シナリオ:
- GET  /acoustic/scenarios         - 利用可能なシナリオ一覧
- POST /acoustic/scenario/start    - シナリオ実行開始
- POST /acoustic/scenario/stop     - シナリオ中断
- GET  /acoustic/scenario/status   - シナリオ実行状態
- GET  /acoustic/scenario/result   - シナリオ実行結果
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dataclasses import asdict

from ..core.acoustic import AcousticManager
from ..core.scenario import ScenarioType, SCENARIOS, get_executor

router = APIRouter(prefix="/acoustic", tags=["acoustic"])


class RecordFrameRequest(BaseModel):
    motor_state: str
    servo_state: str
    pwm_throttle: float = 0.0
    pwm_steering: float = 0.0


class RecordStartRequest(BaseModel):
    session_name: str


class ScenarioStartRequest(BaseModel):
    scenario: str  # "throttle_stop", "throttle_starting", "throttle_slow"


@router.get("/devices")
async def list_devices():
    """オーディオデバイス一覧"""
    try:
        manager = AcousticManager.get_instance()
        return {"devices": manager.list_devices()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def start_capture():
    """音響キャプチャ開始"""
    try:
        manager = AcousticManager.get_instance()
        manager.start()
        return {"status": "started"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_capture():
    """音響キャプチャ停止"""
    try:
        manager = AcousticManager.get_instance()
        manager.stop()
        return {"status": "stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_capture_status():
    """キャプチャ状態を取得"""
    manager = AcousticManager.get_instance()
    return {
        "is_capturing": manager.is_capturing(),
        "is_recording": manager.is_recording,
        "recording_session": manager.recording_session if manager.is_recording else None,
        "recorded_frames": len(manager.recorded_data) if manager.is_recording else 0
    }


@router.get("/features")
async def get_features():
    """現在の音響特徴量を取得"""
    manager = AcousticManager.get_instance()
    if not manager.is_capturing():
        raise HTTPException(status_code=503, detail="Audio capture not running")

    features = manager.get_current_features()
    if features is None:
        raise HTTPException(status_code=503, detail="No audio data available")
    return asdict(features)


@router.get("/state")
async def get_state():
    """現在の音響状態を取得"""
    manager = AcousticManager.get_instance()
    if not manager.is_capturing():
        raise HTTPException(status_code=503, detail="Audio capture not running")

    state = manager.get_current_state()
    return {
        "motor_state": state.motor_state,
        "servo_state": state.servo_state,
        "speed_estimate": state.speed_estimate,
        "confidence": state.confidence,
        "features": asdict(state.features) if state.features else None,
        "timestamp": state.timestamp
    }


@router.post("/record/start")
async def start_recording(request: RecordStartRequest):
    """録音セッション開始"""
    manager = AcousticManager.get_instance()
    if not manager.is_capturing():
        raise HTTPException(status_code=400, detail="Audio capture not running. Start capture first.")
    return manager.start_recording(request.session_name)


@router.post("/record/stop")
async def stop_recording():
    """録音セッション停止"""
    manager = AcousticManager.get_instance()
    return manager.stop_recording()


@router.post("/record/frame")
async def record_frame(request: RecordFrameRequest):
    """現在のフレームを記録"""
    manager = AcousticManager.get_instance()
    result = manager.record_frame(
        motor_state=request.motor_state,
        servo_state=request.servo_state,
        pwm_throttle=request.pwm_throttle,
        pwm_steering=request.pwm_steering
    )
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/record/data")
async def get_recorded_data():
    """録音データ取得"""
    manager = AcousticManager.get_instance()
    return {"data": manager.get_recorded_data()}


# === シナリオ管理 ===

@router.get("/scenarios")
async def list_scenarios():
    """利用可能なシナリオ一覧"""
    scenarios = []
    for name, config in SCENARIOS.items():
        scenarios.append({
            "name": name.value,
            "duration": config.duration,
            "sample_rate": config.sample_rate,
            "total_samples": config.total_samples,
            "throttle_range": [config.throttle_start, config.throttle_end],
            "steering": config.steering
        })
    return {"scenarios": scenarios}


@router.post("/scenario/start")
async def start_scenario(request: ScenarioStartRequest, background_tasks: BackgroundTasks):
    """シナリオ実行開始（バックグラウンド）"""
    try:
        scenario_type = ScenarioType(request.scenario)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown scenario: {request.scenario}. "
                   f"Available: {[s.value for s in ScenarioType]}"
        )

    try:
        executor = get_executor()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if executor.is_running:
        raise HTTPException(status_code=409, detail="Another scenario is running")

    # バックグラウンドで実行
    background_tasks.add_task(executor.execute, scenario_type)

    config = SCENARIOS[scenario_type]
    return {
        "status": "started",
        "scenario": scenario_type.value,
        "config": {
            "duration": config.duration,
            "sample_rate": config.sample_rate,
            "total_samples": config.total_samples,
            "throttle_start": config.throttle_start,
            "throttle_end": config.throttle_end,
            "steering": config.steering
        }
    }


@router.post("/scenario/stop")
async def stop_scenario():
    """シナリオ中断"""
    try:
        executor = get_executor()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    executor.stop()
    return {"status": "stopped"}


@router.get("/scenario/status")
async def get_scenario_status():
    """シナリオ実行状態"""
    try:
        executor = get_executor()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return executor.get_status()


@router.get("/scenario/result")
async def get_scenario_result():
    """シナリオ実行結果（最後の実行）"""
    try:
        executor = get_executor()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = executor.get_result()

    if result is None:
        raise HTTPException(status_code=404, detail="No result available")

    return result.to_dict()
