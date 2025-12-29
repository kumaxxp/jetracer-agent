# YANA Phase 2-C: PWMパラメータ修正

**実行環境**: Jetson Orin Nano Super
**リポジトリ**: `~/jetracer-agent`

---

## 背景

実機テストで以下が判明：
- stop PWM: 390
- 動き出しPWM: 405（ギリギリ）
- 低速走行PWM: 405-410
- 現在の front: 400（範囲が狭すぎて制御不能）

---

## タスク

### 1. pwm_params.json を修正

**ファイル**: `configs/pwm_params.json`

```json
{
  "pwm_steering": {
    "left": 465,
    "center": 400,
    "right": 335
  },
  "pwm_speed": {
    "front": 420,
    "stop": 390,
    "back": 360
  }
}
```

変更点：
- `front`: 400 → **420**（制御範囲を30に拡大）

### 2. acoustic_throttle_controller.py のパラメータ修正

**ファイル**: `http_server/core/acoustic_throttle_controller.py`

`ControllerConfig` クラスのデフォルト値を修正：

```python
@dataclass
class ControllerConfig:
    """制御パラメータ"""
    # PWM設定（front=420, stop=390 に基づく）
    # throttle 0.5 = stop + 0.5 * (front - stop) = 390 + 0.5 * 30 = 405
    # throttle 0.6 = stop + 0.6 * (front - stop) = 390 + 0.6 * 30 = 408
    startup_pwm: float = 0.6       # PWM 408 → 確実に動く
    maintain_pwm: float = 0.5      # PWM 405 → ギリギリ動く（低速）
    threshold_pwm: float = 0.5     # 始動検出閾値
```

### 3. main.py の初期化パラメータも修正（もしハードコードされていれば）

**ファイル**: `http_server/main.py`

`ControllerConfig` の初期化部分を確認し、必要なら修正：

```python
# Phase 2-B で特定したパラメータ → 新しい値に更新
throttle_config = ControllerConfig(
    startup_pwm=0.6,      # 確実に動く
    maintain_pwm=0.5,     # 低速維持
    threshold_pwm=0.5,
)
```

---

## 検証

修正後、以下をテスト：

```bash
# サーバー再起動
python -m http_server.main

# 別ターミナルで
# 1. 直接PWM制御テスト
curl -X POST http://localhost:8000/control \
  -H "Content-Type: application/json" \
  -d '{"throttle": 0.5, "steering": 0.0}'
# → PWM 405 で低速回転するはず

curl -X POST http://localhost:8000/stop

# 2. 音響制御テスト
curl -X POST http://localhost:8000/acoustic/capture/start
curl -X POST http://localhost:8000/acoustic/throttle/start \
  -H "Content-Type: application/json" \
  -d '{"target_speed": 0.2}'

# 状態確認
curl http://localhost:8000/acoustic/throttle/status
# → state: "running", current_pwm: 0.5

curl -X POST http://localhost:8000/acoustic/throttle/stop
```

---

## 期待される動作

1. `throttle/start` → PWM 0.6 (408) で蹴り出し
2. 音響で始動検出 → PWM 0.5 (405) に降下
3. 低速で安定走行
