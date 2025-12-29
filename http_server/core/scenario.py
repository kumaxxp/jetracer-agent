"""
シナリオ実行エンジン（Jetson側）

3つのシナリオを自動実行し、PWM値と音響特徴量を記録する。
"""

import asyncio
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable
from enum import Enum
import logging

from .acoustic import AcousticManager, AudioFeatures

logger = logging.getLogger(__name__)


class ScenarioType(str, Enum):
    THROTTLE_STOP = "throttle_stop"
    THROTTLE_STARTING = "throttle_starting"
    THROTTLE_SLOW = "throttle_slow"


@dataclass
class ScenarioConfig:
    """シナリオ設定"""
    name: ScenarioType
    duration: float          # 秒
    sample_rate: int         # Hz
    throttle_start: float    # 開始時のスロットル
    throttle_end: float      # 終了時のスロットル
    steering: float          # ステアリング（固定）

    @property
    def total_samples(self) -> int:
        return int(self.duration * self.sample_rate)


# シナリオ定義
SCENARIOS = {
    ScenarioType.THROTTLE_STOP: ScenarioConfig(
        name=ScenarioType.THROTTLE_STOP,
        duration=5.0,
        sample_rate=10,
        throttle_start=0.0,
        throttle_end=0.0,
        steering=0.0
    ),
    ScenarioType.THROTTLE_STARTING: ScenarioConfig(
        name=ScenarioType.THROTTLE_STARTING,
        duration=10.0,
        sample_rate=10,
        throttle_start=0.0,
        throttle_end=0.5,
        steering=0.0
    ),
    ScenarioType.THROTTLE_SLOW: ScenarioConfig(
        name=ScenarioType.THROTTLE_SLOW,
        duration=5.0,
        sample_rate=10,
        throttle_start=0.2,
        throttle_end=0.2,
        steering=0.0
    ),
}


@dataclass
class SampleData:
    """1サンプルのデータ"""
    timestamp: float
    elapsed: float           # シナリオ開始からの経過時間
    pwm_throttle: float
    pwm_steering: float
    audio_features: Optional[dict]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ScenarioResult:
    """シナリオ実行結果"""
    scenario: str
    config: dict
    start_time: float
    end_time: float
    samples: List[dict]

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario,
            "config": self.config,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "sample_count": len(self.samples),
            "samples": self.samples
        }


class ScenarioExecutor:
    """シナリオ実行エンジン"""

    def __init__(
        self,
        set_throttle_func: Callable[[float], None],
        set_steering_func: Callable[[float], None]
    ):
        """
        Args:
            set_throttle_func: スロットル設定関数 (値: -1.0 ~ 1.0)
            set_steering_func: ステアリング設定関数 (値: -1.0 ~ 1.0)
        """
        self.set_throttle = set_throttle_func
        self.set_steering = set_steering_func
        self.acoustic = AcousticManager.get_instance()

        self.is_running = False
        self.current_scenario: Optional[ScenarioType] = None
        self.progress = 0.0
        self.current_throttle = 0.0
        self.current_steering = 0.0
        self.samples: List[SampleData] = []
        self.result: Optional[ScenarioResult] = None

    def get_status(self) -> dict:
        """現在の状態を取得"""
        return {
            "running": self.is_running,
            "scenario": self.current_scenario.value if self.current_scenario else None,
            "progress": self.progress,
            "current_pwm": {
                "throttle": self.current_throttle,
                "steering": self.current_steering
            },
            "samples_collected": len(self.samples)
        }

    async def execute(self, scenario_type: ScenarioType) -> ScenarioResult:
        """シナリオを実行"""
        if self.is_running:
            raise RuntimeError("Another scenario is running")

        config = SCENARIOS[scenario_type]
        logger.info(f"Starting scenario: {scenario_type.value}")

        self.is_running = True
        self.current_scenario = scenario_type
        self.progress = 0.0
        self.samples = []

        # 音響キャプチャ開始
        if not self.acoustic.is_capturing():
            self.acoustic.start()
            await asyncio.sleep(0.5)  # 安定待ち

        start_time = time.time()
        sample_interval = 1.0 / config.sample_rate

        try:
            # 初期PWM設定
            self.current_throttle = config.throttle_start
            self.current_steering = config.steering
            self.set_throttle(self.current_throttle)
            self.set_steering(self.current_steering)

            for i in range(config.total_samples):
                if not self.is_running:
                    break

                sample_start = time.time()
                elapsed = sample_start - start_time

                # PWM値を計算（線形補間）
                if config.throttle_start != config.throttle_end:
                    t = elapsed / config.duration
                    self.current_throttle = (
                        config.throttle_start +
                        (config.throttle_end - config.throttle_start) * t
                    )
                    self.set_throttle(self.current_throttle)

                # 音響特徴量を取得
                features = self.acoustic.get_current_features()

                # サンプル記録
                sample = SampleData(
                    timestamp=sample_start,
                    elapsed=elapsed,
                    pwm_throttle=self.current_throttle,
                    pwm_steering=self.current_steering,
                    audio_features=asdict(features) if features else None
                )
                self.samples.append(sample)

                # 進捗更新
                self.progress = (i + 1) / config.total_samples

                # 次のサンプルまで待機
                processing_time = time.time() - sample_start
                sleep_time = sample_interval - processing_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        finally:
            # PWMをゼロに戻す
            self.set_throttle(0.0)
            self.set_steering(0.0)
            self.current_throttle = 0.0
            self.current_steering = 0.0

        end_time = time.time()

        # config を dict に変換
        config_dict = {
            "name": config.name.value,
            "duration": config.duration,
            "sample_rate": config.sample_rate,
            "throttle_start": config.throttle_start,
            "throttle_end": config.throttle_end,
            "steering": config.steering,
            "total_samples": config.total_samples
        }

        # 結果を保存
        self.result = ScenarioResult(
            scenario=scenario_type.value,
            config=config_dict,
            start_time=start_time,
            end_time=end_time,
            samples=[s.to_dict() for s in self.samples]
        )

        self.is_running = False
        self.progress = 1.0
        logger.info(f"Scenario completed: {len(self.samples)} samples")

        return self.result

    def stop(self):
        """シナリオを中断"""
        self.is_running = False
        self.set_throttle(0.0)
        self.set_steering(0.0)

    def get_result(self) -> Optional[ScenarioResult]:
        """最後の実行結果を取得"""
        return self.result


# グローバルインスタンス（main.pyで初期化）
_executor: Optional[ScenarioExecutor] = None


def init_executor(set_throttle: Callable, set_steering: Callable) -> ScenarioExecutor:
    """エグゼキュータを初期化"""
    global _executor
    _executor = ScenarioExecutor(set_throttle, set_steering)
    return _executor


def get_executor() -> ScenarioExecutor:
    """エグゼキュータを取得"""
    if _executor is None:
        raise RuntimeError("ScenarioExecutor not initialized. Call init_executor first.")
    return _executor
