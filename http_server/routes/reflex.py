"""FunctionGemma Reflex APIルート

アーキテクチャ:
センサー → Preprocessor → FunctionGemma(LLM) → VelocityController → SafetyGuard → PWM出力

全ての判断はFunctionGemma（LLM）が行う。
SafetyGuardはハードウェア保護のための最終防壁のみ。
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any

from ..core.reflex.functiongemma_engine import FunctionGemmaEngine
from ..core.reflex.sensor_preprocessor import SensorPreprocessor, AbstractSensorState
from ..core.reflex.velocity_controller import VelocityController
from ..core.reflex.safety_guard import SafetyGuard
from ..core.reflex.intent_vocabulary import ControlIntent

router = APIRouter(prefix="/reflex", tags=["FunctionGemma Reflex"])

# シングルトンインスタンス
_engine: Optional[FunctionGemmaEngine] = None
_preprocessor: Optional[SensorPreprocessor] = None
_controller: Optional[VelocityController] = None
_safety_guard: Optional[SafetyGuard] = None


def set_engine(engine: FunctionGemmaEngine) -> None:
    """外部からエンジンをセット（main.pyから呼び出し）"""
    global _engine
    _engine = engine


def get_engine() -> FunctionGemmaEngine:
    global _engine
    if _engine is None:
        _engine = FunctionGemmaEngine()
    return _engine


def get_preprocessor() -> SensorPreprocessor:
    global _preprocessor
    if _preprocessor is None:
        _preprocessor = SensorPreprocessor()
    return _preprocessor


def get_controller() -> VelocityController:
    global _controller
    if _controller is None:
        _controller = VelocityController()
    return _controller


def get_safety_guard() -> SafetyGuard:
    global _safety_guard
    if _safety_guard is None:
        _safety_guard = SafetyGuard()
    return _safety_guard


# --- リクエスト/レスポンスモデル ---

class SensorInput(BaseModel):
    lidar: Optional[Dict] = None
    imu: Optional[Dict] = None
    acoustic: Optional[Dict] = None
    road: Optional[Dict] = None


class InferResponse(BaseModel):
    intent: Dict
    sensor_state: Dict
    inference_time_ms: float


class ControlResponse(BaseModel):
    throttle: float
    steering: float
    intent: Dict
    controller_state: Dict
    safety_override: Optional[Dict] = None
    sensor_state: Optional[Dict] = None


# --- エンドポイント ---

@router.get("/status")
async def get_status():
    """FunctionGemmaステータス取得"""
    engine = get_engine()
    controller = get_controller()
    safety = get_safety_guard()

    return {
        "engine": engine.get_status(),
        "controller": controller.get_state(),
        "safety": safety.get_status(),
    }


@router.post("/initialize")
async def initialize():
    """FunctionGemmaエンジン初期化"""
    engine = get_engine()
    success = engine.initialize()

    if not success:
        return {
            "status": "fallback",
            "message": "Model not found, using rule-based fallback"
        }

    return {"status": "initialized"}


@router.post("/infer", response_model=InferResponse)
async def infer(sensor_input: SensorInput):
    """センサー入力から意図を推論（LLMのみ、PWM変換なし）"""
    engine = get_engine()
    preprocessor = get_preprocessor()

    if not engine.is_initialized:
        engine.initialize()

    # センサー前処理
    sensor_state = preprocessor.process(
        lidar_data=sensor_input.lidar,
        imu_data=sensor_input.imu,
        acoustic_data=sensor_input.acoustic,
        road_data=sensor_input.road
    )

    # LLM推論
    intent = engine.infer(sensor_state)

    return InferResponse(
        intent=intent.to_dict(),
        sensor_state=sensor_state.to_dict(),
        inference_time_ms=engine.last_inference_time
    )


@router.post("/control", response_model=ControlResponse)
async def control(sensor_input: SensorInput):
    """センサー入力からPWM出力を計算（フルパイプライン）

    パイプライン:
    1. SensorPreprocessor: 生値 → 抽象状態
    2. FunctionGemma: 抽象状態 → 意図（LLMが判断）
    3. VelocityController: 意図 → PWM値（純粋マッピング）
    4. SafetyGuard: 最終安全チェック（ハードウェア保護のみ）
    """
    engine = get_engine()
    preprocessor = get_preprocessor()
    controller = get_controller()
    safety = get_safety_guard()

    if not engine.is_initialized:
        engine.initialize()

    # Step 1: センサー前処理
    sensor_state = preprocessor.process(
        lidar_data=sensor_input.lidar,
        imu_data=sensor_input.imu,
        acoustic_data=sensor_input.acoustic,
        road_data=sensor_input.road
    )

    # Step 2: LLM推論（全ての判断はここで行われる）
    intent = engine.infer(sensor_state)

    # Step 3: 意図→PWM変換（純粋マッピング、判断なし）
    throttle, steering = controller.update(intent)

    # Step 4: 最終安全チェック（ハードウェア保護のみ）
    safe_throttle, safe_steering, override = safety.check(
        throttle, steering, sensor_state
    )

    return ControlResponse(
        throttle=safe_throttle,
        steering=safe_steering,
        intent=intent.to_dict(),
        controller_state=controller.get_state(),
        safety_override={
            "reason": override.reason,
            "throttle_override": override.throttle_override,
            "steering_override": override.steering_override,
        } if override else None,
        sensor_state=sensor_state.to_dict()
    )


@router.post("/emergency_stop")
async def emergency_stop():
    """緊急停止（SafetyGuard経由ではなく即時）"""
    controller = get_controller()
    controller.reset()
    # throttle=0, steering=現状維持で停止
    return {"status": "emergency_stop", "throttle": 0.0}


@router.post("/reset")
async def reset():
    """状態リセット"""
    controller = get_controller()
    safety = get_safety_guard()
    controller.reset()
    safety.reset_count()
    return {"status": "reset"}


@router.get("/architecture")
async def get_architecture():
    """アーキテクチャ情報（デバッグ用）"""
    return {
        "pipeline": [
            "SensorPreprocessor: Raw → Abstract state",
            "FunctionGemma: Abstract state → Intent (ALL decisions here)",
            "VelocityController: Intent → PWM (pure mapping, NO decisions)",
            "SafetyGuard: Final safety check (hardware protection ONLY)",
        ],
        "design_principle": "All decisions are made by LLM, not by if-statements",
        "safety_guard_role": "Override LLM output ONLY for hardware protection",
    }
