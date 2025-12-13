# YANA 自律走行システム設計書

**作成日**: 2024年12月13日  
**対象**: Jetson Orin Nano Super ローカル自律走行  
**走行環境**: 室内廊下（壁沿い）

---

## 1. システム概要

### 1.1 設計方針

| 項目 | 方針 |
|------|------|
| 制御主体 | Jetson単体（PC/LLMはオプション） |
| 走行方式 | セグメンテーション結果からステアリング計算 |
| 安全機構 | 自動モード時のみ機能（手動時は無効） |
| データ収集 | 1fps、画像+steering/throttleペア |
| VLM | 不使用（負荷・信頼性の問題） |

### 1.2 システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Jetson Orin Nano Super                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Autonomous Controller (100ms loop)              │   │
│  │                                                              │   │
│  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │   │
│  │  │   Sensor     │   │  Steering    │   │   Safety     │    │   │
│  │  │   Fusion     │──►│  Calculator  │──►│   Guard      │    │   │
│  │  └──────────────┘   └──────────────┘   └──────────────┘    │   │
│  │         ▲                                      │            │   │
│  │         │                                      ▼            │   │
│  │  ┌──────┴───────────────────────────────────────────┐      │   │
│  │  │              PWM Output Controller               │      │   │
│  │  └──────────────────────────────────────────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                      │
│         ┌────────────────────┼────────────────────┐                │
│         │                    │                    │                │
│  ┌──────┴──────┐  ┌─────────┴─────────┐  ┌──────┴──────┐         │
│  │   Camera    │  │   Segmentation    │  │   Sensors   │         │
│  │   (Dual)    │  │   (OneFormer/     │  │  IMU/LiDAR  │         │
│  │             │  │    Lightweight)   │  │  PWM Input  │         │
│  └─────────────┘  └───────────────────┘  └─────────────┘         │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              Data Collector (1fps)                           │   │
│  │   • 画像保存                                                  │   │
│  │   • steering/throttle記録                                    │   │
│  │   • センサーログ（オプション）                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              HTTP API Server (FastAPI)                       │   │
│  │   • 状態監視 / パラメータ調整 / モード切替                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                              ▲
                              │ WiFi (監視・設定用)
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          PC (オプション)                            │
│   • NiceGUI ダッシュボード（監視・パラメータ調整）                  │
│   • LLM介入（将来：異常時の判断支援）                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 動作モード

### 2.1 モード定義

| モード | CH3値 | 制御主体 | 安全機構 | 説明 |
|--------|-------|---------|---------|------|
| `manual` | < 1500μs | RC送信機 | ❌ 無効 | 完全手動操作 |
| `auto` | >= 1500μs | Jetson | ✅ 有効 | セグメンテーション追従 |
| `emergency_stop` | - | - | - | 緊急停止状態 |

### 2.2 モード遷移

```
                    ┌──────────────┐
                    │    INIT      │
                    └──────┬───────┘
                           │ 起動完了
                           ▼
              ┌────────────────────────┐
              │                        │
              ▼                        ▼
       ┌──────────┐  CH3切替   ┌──────────┐
       │  MANUAL  │◄──────────►│   AUTO   │
       └──────────┘            └────┬─────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │ 安全条件違反  │               │
                    ▼               │               │
           ┌────────────────┐      │               │
           │ EMERGENCY_STOP │◄─────┘               │
           └───────┬────────┘                      │
                   │ 手動リセット                   │
                   └───────────────────────────────┘
```

---

## 3. セグメンテーション → ステアリング変換

### 3.1 基本アルゴリズム

```python
def calculate_steering(road_mask: np.ndarray) -> float:
    """
    ROADマスクからステアリング値を計算
    
    Args:
        road_mask: 二値マスク (H, W), True=走行可能
    
    Returns:
        steering: -1.0（左）〜 +1.0（右）
    """
    h, w = road_mask.shape
    
    # 下半分（足元）を重視した重み付け
    weights = np.linspace(0.3, 1.0, h).reshape(-1, 1)
    weighted_mask = road_mask.astype(float) * weights
    
    # ROAD領域の重心X座標を計算
    if weighted_mask.sum() < 0.01:
        return 0.0  # ROAD領域がない場合は直進
    
    x_coords = np.arange(w)
    centroid_x = (weighted_mask.sum(axis=0) * x_coords).sum() / weighted_mask.sum()
    
    # 中心からのオフセットを -1.0 〜 +1.0 に正規化
    center_x = w / 2
    offset = (centroid_x - center_x) / center_x
    
    return np.clip(offset * STEERING_GAIN, -1.0, 1.0)
```

### 3.2 パラメータ

| パラメータ | 初期値 | 説明 |
|-----------|--------|------|
| `STEERING_GAIN` | 1.5 | ステアリング感度 |
| `THROTTLE_BASE` | 0.15 | 基本スロットル |
| `THROTTLE_CURVE_REDUCTION` | 0.3 | カーブ時の減速率 |
| `ROAD_THRESHOLD` | 0.1 | 最低ROAD比率（停止判定） |

### 3.3 デュアルカメラ統合

```python
def integrate_dual_camera(front_analysis, ground_analysis) -> SteeringCommand:
    """
    正面カメラと足元カメラの情報を統合
    
    - 正面カメラ: 遠方の道の方向（戦略的）
    - 足元カメラ: 即座の安全確認（戦術的）
    """
    # 足元カメラでROADがない場合は緊急停止
    if ground_analysis.total_road_ratio < 0.1:
        return SteeringCommand(steering=0, throttle=0, stop=True)
    
    # 基本は正面カメラのステアリングを採用
    steering = front_analysis.calculated_steering
    
    # 足元の状況で補正
    if ground_analysis.boundary.left:
        steering = max(steering, 0.1)  # 左壁があれば右寄りに
    if ground_analysis.boundary.right:
        steering = min(steering, -0.1)  # 右壁があれば左寄りに
    
    # スロットル計算
    throttle = THROTTLE_BASE
    if abs(steering) > 0.3:
        throttle *= (1 - THROTTLE_CURVE_REDUCTION)
    
    return SteeringCommand(steering=steering, throttle=throttle)
```

---

## 4. 安全システム

### 4.1 緊急停止条件

| 条件 | 閾値 | センサー |
|------|------|---------|
| 前方障害物 | < 150mm | LiDAR (8x8中央) |
| ROAD領域不足 | < 10% | セグメンテーション |
| 車体傾斜 | > 30° | IMU (roll/pitch) |
| 通信断 | > 3秒 | HTTPハートビート |

### 4.2 安全チェック実装

```python
@dataclass
class SafetyStatus:
    safe: bool
    reason: Optional[str] = None
    lidar_min_mm: int = 9999
    road_ratio: float = 1.0
    tilt_deg: float = 0.0

def check_safety(
    lidar_data: DistanceData,
    road_ratio: float,
    imu_data: IMUData
) -> SafetyStatus:
    """安全状態をチェック"""
    
    # LiDAR: 中央4x4の最小距離
    center_distances = [
        lidar_data.distances[r][c]
        for r in range(2, 6) for c in range(2, 6)
        if lidar_data.distances[r][c] > 0
    ]
    lidar_min = min(center_distances) if center_distances else 9999
    
    if lidar_min < 150:
        return SafetyStatus(False, f"障害物検知: {lidar_min}mm", lidar_min)
    
    # ROAD比率
    if road_ratio < 0.1:
        return SafetyStatus(False, f"ROAD不足: {road_ratio:.1%}", road_ratio=road_ratio)
    
    # IMU傾斜
    tilt = max(abs(imu_data.roll), abs(imu_data.pitch))
    if tilt > 30:
        return SafetyStatus(False, f"傾斜過大: {tilt:.1f}°", tilt_deg=tilt)
    
    return SafetyStatus(True, lidar_min_mm=lidar_min, road_ratio=road_ratio, tilt_deg=tilt)
```

---

## 5. データ収集システム

### 5.1 収集データ形式

```
data/
└── sessions/
    └── {session_id}/
        ├── metadata.json      # セッション情報
        ├── frames/
        │   ├── 000001.jpg     # 正面カメラ画像
        │   ├── 000002.jpg
        │   └── ...
        └── log.csv            # 制御ログ
```

### 5.2 log.csv 形式

```csv
timestamp,frame_id,steering,throttle,mode,road_ratio,lidar_min_mm,heading,roll,pitch
1702450000.123,000001,0.15,0.20,auto,0.65,450,45.2,1.3,-0.5
1702450001.123,000002,0.22,0.18,auto,0.58,380,46.1,1.5,-0.8
...
```

### 5.3 収集トリガー

| トリガー | 条件 | 説明 |
|---------|------|------|
| 時間ベース | 1秒間隔 | 基本収集 |
| イベントベース | ステアリング変化 > 0.1 | 重要シーン強調 |
| 手動 | API呼び出し | デバッグ用 |

---

## 6. 制御ループ

### 6.1 メインループ（100ms）

```python
async def control_loop():
    """メイン制御ループ (10Hz)"""
    
    while running:
        loop_start = time.time()
        
        # 1. モード確認
        pwm_data = sensor_manager.read_pwm()
        mode = "auto" if pwm_data.ch3 >= 1500 else "manual"
        
        if mode == "manual":
            # 手動モード: Jetson制御なし
            await asyncio.sleep(0.1)
            continue
        
        # 2. センサー読み取り
        imu_data = sensor_manager.read_imu()
        lidar_data = sensor_manager.read_distance()
        
        # 3. カメラ＆セグメンテーション
        front_frame = camera_manager.capture(0)
        ground_frame = camera_manager.capture(1)
        
        front_seg = segmenter.infer(front_frame)
        ground_seg = segmenter.infer(ground_frame)
        
        # 4. 安全チェック
        safety = check_safety(lidar_data, front_seg.road_ratio, imu_data)
        
        if not safety.safe:
            execute_emergency_stop(safety.reason)
            continue
        
        # 5. ステアリング計算
        command = calculate_steering_command(front_seg, ground_seg)
        
        # 6. 制御出力
        vehicle_controller.set_steering(command.steering)
        vehicle_controller.set_throttle(command.throttle)
        
        # 7. ループ時間調整
        elapsed = time.time() - loop_start
        if elapsed < 0.1:
            await asyncio.sleep(0.1 - elapsed)
```

### 6.2 データ収集ループ（1秒）

```python
async def data_collection_loop():
    """データ収集ループ (1Hz)"""
    
    while running and collecting:
        # 現在のフレームと制御値を保存
        frame = camera_manager.capture(0)  # 正面カメラ
        
        record = {
            "timestamp": time.time(),
            "frame_id": frame_counter,
            "steering": current_steering,
            "throttle": current_throttle,
            "mode": current_mode,
            "road_ratio": current_road_ratio,
            "lidar_min_mm": current_lidar_min,
            "heading": imu_data.heading,
            "roll": imu_data.roll,
            "pitch": imu_data.pitch
        }
        
        save_frame(frame, frame_counter)
        append_log(record)
        
        frame_counter += 1
        await asyncio.sleep(1.0)
```

---

## 7. API エンドポイント

### 7.1 制御API

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/auto/start` | POST | 自動走行開始 |
| `/auto/stop` | POST | 自動走行停止 |
| `/auto/status` | GET | 現在の状態取得 |
| `/auto/params` | GET/PUT | パラメータ取得/設定 |

### 7.2 データ収集API

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/collect/start` | POST | 収集開始 |
| `/collect/stop` | POST | 収集停止 |
| `/collect/sessions` | GET | セッション一覧 |
| `/collect/download/{id}` | GET | データダウンロード |

### 7.3 デバッグAPI

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/debug/steering` | GET | ステアリング計算詳細 |
| `/debug/safety` | GET | 安全状態詳細 |
| `/debug/snapshot` | GET | 全状態スナップショット |

---

## 8. 実装順序

### Phase 1: 基盤（優先）

1. **ステアリング計算モジュール** (`steering_calculator.py`)
   - セグメンテーション→ステアリング変換
   - パラメータ調整可能

2. **安全監視モジュール** (`safety_guard.py`)
   - 緊急停止条件チェック
   - 状態ログ

3. **制御ループ** (`autonomous_controller.py`)
   - メインループ実装
   - モード切替

### Phase 2: データ収集

4. **データ収集モジュール** (`data_collector.py`)
   - セッション管理
   - 画像/ログ保存

5. **収集API** (`routes/collect.py`)
   - 開始/停止
   - セッション管理

### Phase 3: 統合・チューニング

6. **パラメータチューニングUI**
   - リアルタイム調整
   - 可視化

7. **統合テスト**
   - 廊下走行テスト
   - パラメータ最適化

---

## 9. 設定ファイル

### autonomous_config.yaml

```yaml
# 自律走行設定

control:
  loop_hz: 10                    # 制御ループ周波数
  steering_gain: 1.5             # ステアリング感度
  throttle_base: 0.15            # 基本スロットル
  throttle_curve_reduction: 0.3  # カーブ減速率

safety:
  enabled: true
  lidar_min_mm: 150              # 障害物停止距離
  road_threshold: 0.1            # 最低ROAD比率
  tilt_threshold_deg: 30         # 傾斜停止角度
  
segmentation:
  model: "lightweight"           # lightweight / oneformer
  camera_id: 0                   # 使用カメラ (0=正面)
  use_dual_camera: true          # デュアルカメラ統合

data_collection:
  enabled: false
  fps: 1
  save_ground_camera: false      # 足元カメラも保存
  save_segmentation: false       # セグメンテーション結果も保存
```

---

## 10. 将来の拡張

### 10.1 学習ベース走行（Phase E以降）

```
収集データ → 学習 → ResNetモデル → 高速推論
                         ↓
        セグメンテーションベースと切替可能
```

### 10.2 LLM介入（オプション）

- 異常検知時のみLLMに状況を送信
- 人間への説明生成
- 走行ログの自動分析

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2024-12-13 | 初版作成 |
