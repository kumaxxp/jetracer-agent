"""センサー前処理モジュール

生のセンサー値を抽象状態に変換してFunctionGemmaに渡す。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any


class ProximityLevel(str, Enum):
    """近接レベル"""
    CLEAR = "clear"         # 安全（1m以上）
    WARNING = "warning"     # 注意（0.3-1m）
    DANGER = "danger"       # 危険（0.3m未満）
    CONTACT = "contact"     # 接触（0.1m未満）


class MotorState(str, Enum):
    """モーター状態（音響から推定）"""
    STOPPED = "stopped"     # 停止
    IDLE = "idle"           # アイドル
    STARTING = "starting"   # 始動中
    RUNNING = "running"     # 走行中
    SPINNING = "spinning"   # 空転中


class GripStatus(str, Enum):
    """グリップ状態（音響から推定）"""
    GOOD = "good"           # 良好
    SLIPPING = "slipping"   # スリップ中
    LOST = "lost"           # グリップ喪失


class RoadPosition(str, Enum):
    """道路位置"""
    CENTER = "center"       # 中央
    LEFT_EDGE = "left_edge" # 左端
    RIGHT_EDGE = "right_edge"  # 右端
    LOST = "lost"           # 道路見失い


@dataclass
class AbstractSensorState:
    """抽象化されたセンサー状態（FunctionGemmaへの入力）"""
    # LiDAR
    lidar_front: ProximityLevel = ProximityLevel.CLEAR
    lidar_left: ProximityLevel = ProximityLevel.CLEAR
    lidar_right: ProximityLevel = ProximityLevel.CLEAR
    lidar_min_distance_mm: int = 9999

    # IMU
    impact_detected: bool = False
    tilt_warning: bool = False
    is_lifted: bool = False

    # 音響
    motor_state: MotorState = MotorState.STOPPED
    grip_status: GripStatus = GripStatus.GOOD

    # 道路（セグメンテーションから）
    road_visible: bool = True
    road_position: RoadPosition = RoadPosition.CENTER
    road_ratio: float = 0.5  # 走行可能領域の割合

    def to_prompt_text(self) -> str:
        """FunctionGemma用のプロンプトテキストに変換"""
        lines = [
            f"lidar: front={self.lidar_front.value}, left={self.lidar_left.value}, right={self.lidar_right.value}, min={self.lidar_min_distance_mm}mm",
            f"imu: impact={self.impact_detected}, tilt_warning={self.tilt_warning}, lifted={self.is_lifted}",
            f"audio: motor={self.motor_state.value}, grip={self.grip_status.value}",
            f"road: visible={self.road_visible}, position={self.road_position.value}, ratio={self.road_ratio:.2f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lidar": {
                "front": self.lidar_front.value,
                "left": self.lidar_left.value,
                "right": self.lidar_right.value,
                "min_distance_mm": self.lidar_min_distance_mm,
            },
            "imu": {
                "impact_detected": self.impact_detected,
                "tilt_warning": self.tilt_warning,
                "is_lifted": self.is_lifted,
            },
            "audio": {
                "motor_state": self.motor_state.value,
                "grip_status": self.grip_status.value,
            },
            "road": {
                "visible": self.road_visible,
                "position": self.road_position.value,
                "ratio": self.road_ratio,
            }
        }


class SensorPreprocessor:
    """センサー前処理クラス"""

    # 距離閾値（mm）
    DISTANCE_CONTACT = 100
    DISTANCE_DANGER = 300
    DISTANCE_WARNING = 1000

    # 傾斜閾値（度）
    TILT_WARNING_THRESHOLD = 15.0
    LIFT_THRESHOLD = 30.0

    def __init__(self):
        self._last_state = AbstractSensorState()

    def process(
        self,
        lidar_data: Optional[Dict] = None,
        imu_data: Optional[Dict] = None,
        acoustic_data: Optional[Dict] = None,
        road_data: Optional[Dict] = None
    ) -> AbstractSensorState:
        """生センサーデータを抽象状態に変換

        Args:
            lidar_data: LiDARセンサーデータ（8x8グリッド等）
            imu_data: IMUデータ（加速度、ジャイロ、オイラー角）
            acoustic_data: 音響分析データ
            road_data: セグメンテーション結果

        Returns:
            AbstractSensorState
        """
        state = AbstractSensorState()

        # LiDAR処理
        if lidar_data:
            state = self._process_lidar(state, lidar_data)

        # IMU処理
        if imu_data:
            state = self._process_imu(state, imu_data)

        # 音響処理
        if acoustic_data:
            state = self._process_acoustic(state, acoustic_data)

        # 道路処理
        if road_data:
            state = self._process_road(state, road_data)

        self._last_state = state
        return state

    def _process_lidar(self, state: AbstractSensorState, data: Dict) -> AbstractSensorState:
        """LiDARデータを処理"""
        # 8x8グリッドから方向別の最小距離を計算
        grid = data.get("grid") or data.get("distances", [])

        if not grid or len(grid) < 8:
            return state

        # 前方（中央列、上半分）
        front_distances = []
        for row in range(4):  # 上半分
            for col in range(2, 6):  # 中央4列
                if row < len(grid) and col < len(grid[row]):
                    d = grid[row][col]
                    if 0 < d < 4000:  # 有効範囲
                        front_distances.append(d)

        # 左側（左2列）
        left_distances = []
        for row in range(8):
            for col in range(2):
                if row < len(grid) and col < len(grid[row]):
                    d = grid[row][col]
                    if 0 < d < 4000:
                        left_distances.append(d)

        # 右側（右2列）
        right_distances = []
        for row in range(8):
            for col in range(6, 8):
                if row < len(grid) and col < len(grid[row]):
                    d = grid[row][col]
                    if 0 < d < 4000:
                        right_distances.append(d)

        # 最小距離から近接レベルを判定
        state.lidar_front = self._distance_to_level(min(front_distances) if front_distances else 9999)
        state.lidar_left = self._distance_to_level(min(left_distances) if left_distances else 9999)
        state.lidar_right = self._distance_to_level(min(right_distances) if right_distances else 9999)

        all_distances = front_distances + left_distances + right_distances
        state.lidar_min_distance_mm = min(all_distances) if all_distances else 9999

        return state

    def _distance_to_level(self, distance_mm: int) -> ProximityLevel:
        """距離を近接レベルに変換"""
        if distance_mm < self.DISTANCE_CONTACT:
            return ProximityLevel.CONTACT
        elif distance_mm < self.DISTANCE_DANGER:
            return ProximityLevel.DANGER
        elif distance_mm < self.DISTANCE_WARNING:
            return ProximityLevel.WARNING
        else:
            return ProximityLevel.CLEAR

    def _process_imu(self, state: AbstractSensorState, data: Dict) -> AbstractSensorState:
        """IMUデータを処理"""
        # 衝突検知（加速度の急変）
        accel = data.get("acceleration", {})
        accel_magnitude = (
            accel.get("x", 0) ** 2 +
            accel.get("y", 0) ** 2 +
            accel.get("z", 0) ** 2
        ) ** 0.5

        # 通常は約9.8m/s²、衝突時は急上昇
        state.impact_detected = accel_magnitude > 15.0

        # 傾斜警告
        euler = data.get("euler", {})
        roll = abs(euler.get("roll", 0))
        pitch = abs(euler.get("pitch", 0))

        state.tilt_warning = roll > self.TILT_WARNING_THRESHOLD or pitch > self.TILT_WARNING_THRESHOLD
        state.is_lifted = roll > self.LIFT_THRESHOLD or pitch > self.LIFT_THRESHOLD

        return state

    def _process_acoustic(self, state: AbstractSensorState, data: Dict) -> AbstractSensorState:
        """音響データを処理"""
        # モーター状態
        motor = data.get("motor_state", "stopped")
        try:
            state.motor_state = MotorState(motor)
        except ValueError:
            state.motor_state = MotorState.STOPPED

        # グリップ状態
        grip = data.get("grip_status", "good")
        try:
            state.grip_status = GripStatus(grip)
        except ValueError:
            state.grip_status = GripStatus.GOOD

        return state

    def _process_road(self, state: AbstractSensorState, data: Dict) -> AbstractSensorState:
        """道路データを処理"""
        state.road_visible = data.get("visible", True)
        state.road_ratio = data.get("road_ratio", 0.5)

        # 道路位置（セグメンテーションの中心から判定）
        center_x = data.get("road_center_x", 0.5)
        if center_x < 0.35:
            state.road_position = RoadPosition.LEFT_EDGE
        elif center_x > 0.65:
            state.road_position = RoadPosition.RIGHT_EDGE
        else:
            state.road_position = RoadPosition.CENTER

        if not state.road_visible or state.road_ratio < 0.1:
            state.road_position = RoadPosition.LOST

        return state

    @property
    def last_state(self) -> AbstractSensorState:
        return self._last_state
