# PWM Control Integration Guide

## 概要

JetRacerのステアリングとスロットルを制御するPWMキャリブレーション機能をjetracer-agentとyana-brainに統合しました。

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────────┐
│                          yana-brain (PC)                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    NiceGUI Dashboard                      │   │
│  │  ┌─────────┬─────────┬─────────┬──────────┬───────────┐  │   │
│  │  │ Cameras │ Training │ Sensors │   PWM   │ Navigation│  │   │
│  │  └─────────┴─────────┴─────────┴──────────┴───────────┘  │   │
│  │                              │                            │   │
│  │                    PWMCalibrationPanel                    │   │
│  │                              │                            │   │
│  └──────────────────────────────┼────────────────────────────┘   │
│                                 │ HTTP API                       │
└─────────────────────────────────┼────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   jetracer-agent (Jetson)                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                     FastAPI Server                        │   │
│  │                                                           │   │
│  │  /pwm/status      - PWMコントローラー状態                 │   │
│  │  /pwm/params      - パラメータ取得/保存                   │   │
│  │  /pwm/test/*      - 各種テストエンドポイント              │   │
│  │  /pwm/stop        - 緊急停止                              │   │
│  └──────────────────────────────┼────────────────────────────┘   │
│                                 │                                │
│                                 ▼                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    JetRacerPWM                            │   │
│  │                                                           │   │
│  │  - PWMController (SMBus/I2C)                              │   │
│  │  - PCA9685 @ 0x40, 60Hz                                   │   │
│  │  - Channel 0: Steering                                    │   │
│  │  - Channel 1: Throttle                                    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## ファイル構成

### jetracer-agent (Jetson側)

```
jetracer-agent/
├── http_server/
│   ├── core/
│   │   └── pwm_control.py      # PWMコントローラーコア
│   ├── routes/
│   │   └── pwm.py              # REST APIエンドポイント
│   └── main.py                 # サーバーエントリーポイント（更新）
└── configs/
    └── pwm_params.json         # PWMパラメータ設定
```

### yana-brain (PC側)

```
yana-brain/
└── src/
    ├── jetson_client.py        # APIクライアント（更新）
    └── ui/
        ├── dashboard.py        # ダッシュボード（更新）
        └── pwm_calibration_panel.py  # PWMキャリブレーションUI
```

## APIエンドポイント

| Method | Endpoint | 説明 |
|--------|----------|------|
| GET | `/pwm/status` | PWMコントローラー状態取得 |
| GET | `/pwm/params` | 現在のPWMパラメータ取得 |
| POST | `/pwm/params` | PWMパラメータ保存 |
| POST | `/pwm/test/steering/center` | ステアリング中央テスト |
| POST | `/pwm/test/steering/left` | ステアリング左テスト |
| POST | `/pwm/test/steering/right` | ステアリング右テスト |
| POST | `/pwm/test/steering/range` | ステアリング可動範囲テスト |
| POST | `/pwm/test/steering/value/{value}` | 任意のステアリングPWM値設定 |
| POST | `/pwm/test/throttle/stop` | スロットル停止テスト |
| POST | `/pwm/test/throttle/forward` | スロットル前進テスト |
| POST | `/pwm/test/throttle/backward` | スロットル後退テスト (ESCシーケンス付き) |
| POST | `/pwm/test/throttle/value/{value}` | 任意のスロットルPWM値設定 |
| POST | `/pwm/stop` | 緊急停止 |

## PWMパラメータ形式

```json
{
  "pwm_steering": {
    "left": 310,
    "center": 410,
    "right": 510
  },
  "pwm_speed": {
    "front": 430,
    "stop": 410,
    "back": 390
  }
}
```

## ハードウェア仕様

| 項目 | 値 |
|------|-----|
| PWMコントローラー | PCA9685 |
| I2Cアドレス | 0x40 |
| PWM周波数 | 60Hz (RCサーボ標準) |
| PWM範囲 | 0-4095 (12-bit) |
| サーボ有効範囲 | 200-600 (典型値) |
| ステアリングチャンネル | 0 |
| スロットルチャンネル | 1 |

### Jetson I2Cバス番号

| ボード | バス番号 |
|--------|---------|
| Jetson Nano | 1 |
| Jetson NX / Xavier | 8 |
| Jetson Orin / Orin Nano | 7 |

## 使用方法

### 1. yana-brainダッシュボードから

1. yana-brainを起動: `python -m src.main`
2. ブラウザでダッシュボードを開く
3. **PWM** タブを選択
4. **Refresh Status** で接続確認
5. 各パラメータを調整してテスト
6. **Save Parameters** で保存

### 2. REST API直接呼び出し

```bash
# 状態確認
curl http://jetson:8000/pwm/status

# ステアリング中央テスト
curl -X POST http://jetson:8000/pwm/test/steering/center

# 任意のPWM値設定
curl -X POST http://jetson:8000/pwm/test/steering/value/400

# 緊急停止
curl -X POST http://jetson:8000/pwm/stop
```

### 3. Pythonから

```python
from jetson_client import JetsonClient

client = JetsonClient("jetson-host", 8000)

# 状態確認
status = client.get_pwm_status()

# ステアリングテスト
result = client.test_pwm_steering("center")

# パラメータ保存
params = {
    "pwm_steering": {"left": 310, "center": 410, "right": 510},
    "pwm_speed": {"front": 430, "stop": 410, "back": 390}
}
client.save_pwm_params(params)

# 緊急停止
client.pwm_stop()
```

## キャリブレーション手順

### ステアリング調整

1. **中央値 (Center)** を設定
   - 車輪がまっすぐになるPWM値を探す
   - Check ボタンで確認

2. **振幅 (Amplitude)** を調整
   - 左右の曲がり角度を決定
   - スライダーで調整

3. **左右テスト**
   - Check L / Check R で個別確認
   - L↔R Test で全範囲テスト

4. **逆転 (Reverse)** 
   - 左右が逆の場合はチェック

### スロットル調整

1. **停止 (Stop)** を設定
   - モーターが停止するPWM値
   - ESCのニュートラル位置

2. **前進 (Forward)** を調整
   - 前進時のPWM値オフセット
   - 正の値で前進

3. **後退 (Backward)** を調整
   - 後退時のPWM値オフセット
   - 負の値で後退
   - ESCリバースシーケンス自動実行

## 安全上の注意

⚠️ **重要**: キャリブレーション中は必ず車体を浮かせてください！

- 車輪が接地した状態でテストすると車体が急に動く可能性があります
- 緊急停止ボタン (🛑 STOP) を常に確認できる状態で作業してください
- スロットルテストは低い値から始めてください

## トラブルシューティング

### PWM Not Available

```
[PWM] smbus not available, using mock mode
```

**原因**: smbusモジュールがインストールされていない

**対処**: 
```bash
pip install smbus --break-system-packages
```

### I2C Permission Denied

**原因**: I2Cアクセス権限がない

**対処**:
```bash
sudo usermod -aG i2c $USER
# ログアウト・再ログイン
```

### デバイスが見つからない

**確認方法**:
```bash
# I2Cデバイススキャン
i2cdetect -y 7  # Orin Nanoの場合

# PCA9685が0x40に表示されるはず
     0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
...
40: 40 -- -- -- -- -- -- -- -- -- -- -- -- -- -- --
```

### 開発用モックモード

smbusがない環境(PC等)では自動的にモックモードになります。
モックモードではPWM値がコンソールに出力されるだけで実際のハードウェア制御は行いません。
