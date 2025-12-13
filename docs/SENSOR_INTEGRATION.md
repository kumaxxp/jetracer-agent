# センサー統合ガイド - FaBo JetRacer対応版

Jetson Orin Nano上のI2Cセンサーの統合ガイドです。

## 1. ハードウェア構成

### FaBo JetRacer標準デバイス

| センサー | I2Cアドレス | 説明 |
|---------|------------|------|
| ESP32S3 DevKit | 0x08 | RC受信機のPWM信号をI2C経由で送信 |
| BNO055 | 0x28/0x29 | 9軸IMU（加速度/ジャイロ/磁気） |
| VL53L7CX | 0x33 | 8x8マルチゾーンToFセンサー |
| PCA9685 | 0x40 | サーボ/モータードライバ |

### I2Cバス番号

| ボード | バス番号 |
|--------|---------|
| Jetson Orin Nano | **7** |
| Jetson Nano | 1 |

## 2. I2Cバスの確認

```bash
# Jetson Orin Nano の場合
sudo i2cdetect -y -r 7

# 期待される出力:
#      0  1  2  3  4  5  6  7  8  9  a  b  c  d  e  f
# 00:          -- -- -- -- -- 08 -- -- -- -- -- -- -- 
# 20: -- -- -- -- -- -- -- -- 28 -- -- -- -- -- -- -- 
# 30: -- -- -- 33 -- -- -- -- -- -- -- -- -- -- -- -- 
# 40: 40 -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- 
```

## 3. FaBo JetRacer PWM入力 (ESP32S3)

### 概要

ESP32S3 DevKitがI2Cスレーブとして動作し、RC受信機からのPWM信号を読み取ってJetsonに送信します。

### プロトコル

- **アドレス**: 0x08
- **レジスタ 0x00**: ボード情報 (12バイト)
  - bytes 0-2: ボードリビジョン (e.g., "2.0.3")
  - byte 3: ファームウェアバージョン
- **レジスタ 0x01**: PWMデータ (32バイト)
  - bytes 0-3: CH1 ステアリング (32bit Big-Endian, μs)
  - bytes 4-7: CH2 スロットル (32bit Big-Endian, μs)
  - bytes 8-11: CH3 モード切替 (32bit Big-Endian, μs)

### PWM値の解釈

| チャンネル | 用途 | 典型値 |
|-----------|------|--------|
| CH1 | ステアリング | 1000-2000μs (中央: ~1520) |
| CH2 | スロットル | 1000-2000μs (中央: ~1520) |
| CH3 | モード切替 | ~1000: RCモード, ~2000: AIモード |

### LED制御

ESP32S3ボードの外部LEDを制御できます:

```python
# レジスタコマンド
0x1a: 赤
0x1b: 青
0x1c: 黄
0x1d: 緑
0x1e: 白
0x1f: 橙
0x20: マゼンタ
0x21: 黄緑
0x22: ピンク
0x30: 消灯
0x10: 通常モード
```

### API

```bash
# 初期化
curl -X POST "http://jetson:8000/sensors/init" \
  -H "Content-Type: application/json" \
  -d '{"sensor_type": "pwm_input"}'

# データ読み取り
curl "http://jetson:8000/sensors/pwm_input"

# LED制御
curl -X POST "http://jetson:8000/sensors/led/green"
```

## 4. BNO055 9軸IMU

### 概要

Bosch BNO055は、加速度計、ジャイロスコープ、磁力計を統合した9軸IMUで、
オンボードのセンサーフュージョンにより安定したオイラー角を出力します。

### データ出力

| データ | 単位 | 説明 |
|--------|------|------|
| Heading | deg | 方位角 (0-360°) |
| Roll | deg | ロール角 |
| Pitch | deg | ピッチ角 |
| Accel X/Y/Z | m/s² | 加速度 |
| Gyro X/Y/Z | deg/s | 角速度 |
| Mag X/Y/Z | μT | 磁気 |
| Temperature | °C | 温度 |

### キャリブレーション

BNO055は自動キャリブレーションを行いますが、最良の結果を得るには:

1. **ジャイロ**: 静止状態で数秒待つ
2. **加速度計**: ゆっくりと様々な角度に傾ける
3. **磁力計**: 8の字を描くように回転させる

キャリブレーション状態は0-3で表示（3=完全）。

### API

```bash
# 初期化
curl -X POST "http://jetson:8000/sensors/init" \
  -H "Content-Type: application/json" \
  -d '{"sensor_type": "imu"}'

# データ読み取り
curl "http://jetson:8000/sensors/imu"
```

## 5. VL53L7CX 距離計（将来実装）

### 概要

STマイクロエレクトロニクス製の8x8マルチゾーンToFセンサー。
64点の距離データを同時に取得可能。

**注意**: VL53L7CXは複雑な初期化シーケンスが必要なため、
ST公式Pythonライブラリの使用を推奨します。

```bash
pip install vl53l7cx --break-system-packages
```

## 6. API エンドポイント一覧

| エンドポイント | メソッド | 説明 |
|---------------|---------|------|
| `/sensors/scan` | GET | I2Cデバイススキャン |
| `/sensors/init` | POST | センサー初期化 |
| `/sensors/imu` | GET | BNO055データ読み取り |
| `/sensors/pwm_input` | GET | PWM入力データ読み取り |
| `/sensors/distance` | GET | 距離データ読み取り |
| `/sensors/all` | GET | 全センサーデータ |
| `/sensors/status` | GET | センサー状態 |
| `/sensors/led` | POST | LED色設定（JSON） |
| `/sensors/led/{color}` | POST | LED色設定（パス） |

## 7. UIでの使用

yana-brainダッシュボードの「Sensors」タブで:

1. **Devices**: I2Cデバイスをスキャンして検出
2. **IMU (BNO055)**: 
   - オイラー角（方位/ロール/ピッチ）
   - 加速度/ジャイロ/磁気
   - キャリブレーション状態
3. **PWM Input**: 
   - RC/AIモード表示
   - 各チャンネルの値
4. **LED Control**: ボードLEDの色を変更
5. **Graphs**: リアルタイムグラフ

**Auto**スイッチをONにすると、100ms間隔で自動更新されます。

## 8. トラブルシューティング

### I2Cデバイスが検出されない

```bash
# 権限確認
ls -la /dev/i2c-*

# グループ追加
sudo usermod -aG i2c $USER
# 再ログイン必要

# smbus インストール
pip install smbus --break-system-packages
# または
pip install smbus2 --break-system-packages
```

### ESP32S3の初回読み取りがおかしい

ESP32S3の特性として、I2C送信より受信が先に呼ばれることがあります。
コード側で最初の読み取りを捨てる処理を入れています。

### BNO055のキャリブレーションが完了しない

- 磁気干渉のある場所を避ける
- 電源ノイズをチェック（デカップリングコンデンサ）
- I2Cプルアップ抵抗の確認（4.7kΩ推奨）

## 9. サンプルコード

### 基本的な読み取り

```python
import smbus
import time

# Jetson Orin Nano: bus 7
bus = smbus.SMBus(7)

# FaBo PWM読み取り
addr_pwm = 0x08
data = bus.read_i2c_block_data(addr_pwm, 0x01, 32)
ch1 = (data[0] << 24) | (data[1] << 16) | (data[2] << 8) | data[3]
ch2 = (data[4] << 24) | (data[5] << 16) | (data[6] << 8) | data[7]
ch3 = (data[8] << 24) | (data[9] << 16) | (data[10] << 8) | data[11]
print(f"Steering: {ch1}μs, Throttle: {ch2}μs, Mode: {ch3}μs")

# BNO055読み取り（オイラー角）
addr_imu = 0x28
euler = bus.read_i2c_block_data(addr_imu, 0x1A, 6)
heading = ((euler[1] << 8) | euler[0]) / 16.0
roll = ((euler[3] << 8) | euler[2]) / 16.0
pitch = ((euler[5] << 8) | euler[4]) / 16.0
print(f"Heading: {heading}°, Roll: {roll}°, Pitch: {pitch}°")
```

### RC/AIモード判定

```python
def get_control_mode(ch3_us: int) -> str:
    """PWM CH3からコントロールモードを判定"""
    if ch3_us < 1300:
        return "rc"      # マニュアル操作
    elif ch3_us > 1700:
        return "ai"      # 自律走行
    else:
        return "transition"  # 切り替え中
```

## 10. 参考リンク

- [FaBo JetRacer](https://github.com/FaBoPlatform/JetRacer)
- [BNO055 データシート](https://www.bosch-sensortec.com/products/smart-sensors/bno055/)
- [VL53L7CX ドライバ](https://www.st.com/en/imaging-and-photonics-solutions/vl53l7cx.html)
