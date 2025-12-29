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

推論:
- GET  /acoustic/state/predict     - 音響状態を推論
- GET  /acoustic/state/continuous  - 継続的な状態監視
- GET  /acoustic/classifier/status - 分類器の状態
"""

import time
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dataclasses import asdict

from ..core.acoustic import AcousticManager
from ..core.scenario import ScenarioType, SCENARIOS, get_executor
from ..core.acoustic_inference import get_classifier

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


# === PC互換エイリアス ===
# PC側クライアントが /acoustic/capture/start を呼び出すため

@router.post("/capture/start")
async def start_capture_alias():
    """音響キャプチャ開始（/capture/start エイリアス）"""
    return await start_capture()


@router.post("/capture/stop")
async def stop_capture_alias():
    """音響キャプチャ停止（/capture/stop エイリアス）"""
    return await stop_capture()


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


# === 推論エンドポイント ===

@router.get("/state/predict")
async def predict_state(pwm_throttle: float = 0.0):
    """
    現在の音響状態を推論

    Args:
        pwm_throttle: 現在のスロットルPWM値（クエリパラメータ）

    Returns:
        推論結果（状態、信頼度、全クラス確率）
    """
    # 分類器取得
    classifier = get_classifier()
    if classifier is None or not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Acoustic classifier not available"
        )

    # 音響マネージャー取得
    manager = AcousticManager.get_instance()
    if not manager.is_capturing():
        raise HTTPException(
            status_code=400,
            detail="Audio capture not running. Call /acoustic/start first"
        )

    # 最新の特徴量取得
    features = manager.get_current_features()
    if features is None:
        raise HTTPException(
            status_code=503,
            detail="No audio features available"
        )

    # 推論（全クラス確率付き）
    state, probabilities = classifier.predict_with_all_probs(features, pwm_throttle)

    if state is None:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )

    return {
        "state": state,
        "confidence": probabilities[state],
        "probabilities": probabilities,
        "pwm_throttle": pwm_throttle,
        "timestamp": time.time()
    }


@router.get("/state/continuous")
async def get_continuous_state(
    pwm_throttle: float = 0.0,
    include_features: bool = False
):
    """
    継続的な状態監視用エンドポイント

    特徴量も含めた詳細情報を返す（デバッグ用）
    """
    classifier = get_classifier()
    manager = AcousticManager.get_instance()

    if not manager.is_capturing():
        return {
            "status": "not_running",
            "state": None,
            "timestamp": time.time()
        }

    features = manager.get_current_features()

    result = {
        "status": "running",
        "timestamp": time.time(),
        "pwm_throttle": pwm_throttle
    }

    if features is None:
        result["state"] = None
        result["error"] = "No features"
    elif classifier is None or not classifier.is_loaded:
        result["state"] = None
        result["error"] = "Classifier not loaded"
    else:
        state, probs = classifier.predict_with_all_probs(features, pwm_throttle)
        result["state"] = state
        result["probabilities"] = probs
        result["confidence"] = probs[state] if probs else None

    if include_features and features:
        result["features"] = {
            "rms_energy": features.rms_energy,
            "spectral_centroid": features.spectral_centroid,
            "zcr": features.zcr,
            "mfcc_mean_0": features.mfcc_mean[0] if features.mfcc_mean else None
        }

    return result


@router.get("/classifier/status")
async def get_classifier_status():
    """分類器の状態を取得"""
    classifier = get_classifier()

    if classifier is None:
        return {
            "loaded": False,
            "error": "Classifier not initialized"
        }

    return {
        "loaded": classifier.is_loaded,
        "model_path": str(classifier.model_path),
        "labels": classifier.label_names if classifier.is_loaded else []
    }


# === ヒステリシス制御エンドポイント ===

from ..core.acoustic_throttle_controller import (
    get_acoustic_throttle_controller,
    get_control_loop,
    ControllerConfig
)


class ThrottleStartRequest(BaseModel):
    target_speed: float = 0.2


@router.post("/throttle/start")
async def start_acoustic_throttle(request: ThrottleStartRequest):
    """音響フィードバック制御で走行開始"""
    controller = get_acoustic_throttle_controller()
    if controller is None:
        raise HTTPException(
            status_code=503,
            detail="Acoustic throttle controller not initialized"
        )

    # 音響キャプチャが動作中か確認
    manager = AcousticManager.get_instance()
    if not manager.is_capturing():
        raise HTTPException(
            status_code=400,
            detail="Audio capture not running. Call /acoustic/capture/start first"
        )

    # 制御ループ開始
    control_loop = get_control_loop()
    if control_loop:
        import asyncio
        asyncio.create_task(control_loop.start())

    controller.start(target_speed=request.target_speed)
    return {
        "status": "started",
        "target_speed": request.target_speed,
        "startup_pwm": controller.config.startup_pwm
    }


@router.post("/throttle/stop")
async def stop_acoustic_throttle():
    """音響フィードバック制御を停止"""
    controller = get_acoustic_throttle_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    # 制御ループ停止
    control_loop = get_control_loop()
    if control_loop:
        import asyncio
        asyncio.create_task(control_loop.stop())

    controller.stop()
    return {"status": "stopped"}


@router.get("/throttle/status")
async def get_throttle_status():
    """制御状態を取得"""
    controller = get_acoustic_throttle_controller()
    if controller is None:
        return {"initialized": False}

    status = controller.get_status()
    status["initialized"] = True
    return status


@router.post("/throttle/config")
async def update_throttle_config(
    startup_pwm: float = None,
    maintain_pwm: float = None,
    threshold_pwm: float = None
):
    """制御パラメータを更新"""
    controller = get_acoustic_throttle_controller()
    if controller is None:
        raise HTTPException(status_code=503, detail="Controller not initialized")

    if startup_pwm is not None:
        controller.config.startup_pwm = startup_pwm
    if maintain_pwm is not None:
        controller.config.maintain_pwm = maintain_pwm
    if threshold_pwm is not None:
        controller.config.threshold_pwm = threshold_pwm

    return {
        "status": "updated",
        "config": {
            "startup_pwm": controller.config.startup_pwm,
            "maintain_pwm": controller.config.maintain_pwm,
            "threshold_pwm": controller.config.threshold_pwm,
        }
    }
