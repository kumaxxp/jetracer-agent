# 距離グリッドキャリブレーションシステム

## 概要

カメラ画像に距離グリッドを投影し、セグメンテーション結果と組み合わせてAIに正確な距離情報を提供するシステムです。

```
┌─────────────────────────────────────────────────────────────┐
│  カメラ画像 + 距離グリッドオーバーレイ                        │
│  ┌─────────────────────────────────┐                        │
│  │      ● ● ● ● ● ● ● ● ● ●       │  ← 3.0m              │
│  │     ● ● ● ● ● ● ● ● ● ●        │     2.5m              │
│  │    ● ● ● ● ● ● ● ● ● ●         │     2.0m              │
│  │   ● ● ● ● ● ● ● ● ● ●          │     1.5m              │
│  │  ● ● ● ● ● ● ● ● ● ●           │     1.0m              │
│  │ ─────────────────────          │     0.5m              │
│  │──────────────────────          │  ← 0.2m              │
│  └─────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

## 機能

1. **グリッドキャリブレーション**
   - カメラの取り付け高さ（mm）
   - カメラの俯角（度）
   - グリッドの奥行き範囲（m）
   - グリッドの幅（m）

2. **リアルタイムプレビュー**
   - カメラ映像にグリッドオーバーレイ
   - パラメータ調整時に即座に反映

3. **セグメンテーション統合**
   - ROAD領域とグリッドの合成表示
   - 各距離でのROAD比率分析
   - ナビゲーションヒントの自動生成

4. **クリック距離計測**
   - 画像上をクリックすると推定距離を表示

## 使い方

### 1. yana-brainを起動

```bash
cd ~/projects/yana-brain
source venv/bin/activate
python -m src.main
```

### 2. Distance Gridタブを開く

ブラウザで `http://localhost:8080` にアクセスし、"Distance Grid" タブをクリック

### 3. グリッドキャリブレーション

**Camera 0（正面カメラ）の場合：**

1. **カメラ高さを設定**
   - JetRacerに取り付けたカメラの地面からの高さ（mm）
   - 例: 150mm

2. **俯角を設定**
   - カメラの下向き角度（度）
   - 0° = 水平、90° = 真下
   - 例: 30°

3. **グリッド範囲を設定**
   - Min Depth: 最も近い距離（例: 0.2m）
   - Max Depth: 最も遠い距離（例: 3.0m）
   - Width: グリッドの左右幅（例: 1.0m）

4. **"Apply Config"をクリック**

5. **"Update Preview"をクリック**してグリッドを確認

### 4. セグメンテーション + グリッド分析

1. **"Analyze"ボタンをクリック**

2. 分析結果が表示されます：
   - **Navigation Hint**: 推奨進行方向（FORWARD/LEFT/RIGHT/STOP）
   - **Depth Analysis**: 各距離でのROAD比率
   - **Lateral Analysis**: 左右のROAD分布

3. **画像をクリック**すると、その位置の推定距離が表示されます

## API エンドポイント

### グリッド設定

```
GET  /distance-grid/{camera_id}/status     - ステータス取得
GET  /distance-grid/{camera_id}/config     - 設定取得
PUT  /distance-grid/{camera_id}/config     - 設定更新
```

### グリッドデータ

```
GET  /distance-grid/{camera_id}/grid-lines - グリッド線データ
GET  /distance-grid/{camera_id}/preview    - プレビュー画像
```

### セグメンテーション統合

```
GET  /distance-grid/{camera_id}/overlay-on-segmentation - オーバーレイ画像
GET  /distance-grid/{camera_id}/analyze-segmentation    - フル分析
POST /distance-grid/{camera_id}/distance-at-point       - 距離計測
```

## 設定パラメータ

| パラメータ | 説明 | デフォルト | 範囲 |
|-----------|------|-----------|------|
| camera_height_mm | カメラ高さ | 150 | 50-500 |
| camera_pitch_deg | カメラ俯角 | 30 | 0-90 |
| grid_depth_min_m | 最小奥行き | 0.2 | 0.1-1.0 |
| grid_depth_max_m | 最大奥行き | 3.0 | 1.0-5.0 |
| grid_width_m | グリッド幅 | 1.0 | 0.5-3.0 |
| grid_depth_lines | 奥行き線数 | 10 | - |
| grid_width_lines | 幅方向線数 | 11 | - |

## AIへの情報提供

分析結果はJSONで取得でき、AIに以下の情報を提供できます：

```json
{
  "depth_analysis": [
    {"depth_m": 0.5, "road_ratio": 0.85},
    {"depth_m": 1.0, "road_ratio": 0.72},
    {"depth_m": 1.5, "road_ratio": 0.45},
    {"depth_m": 2.0, "road_ratio": 0.12}
  ],
  "lateral_analysis": [
    {"offset_m": -0.5, "road_ratio": 0.30},
    {"offset_m": 0.0, "road_ratio": 0.85},
    {"offset_m": 0.5, "road_ratio": 0.40}
  ],
  "navigation_hint": {
    "forward_clear": true,
    "max_clear_distance_m": 1.5,
    "recommended_direction": "forward",
    "confidence": 0.75
  }
}
```

これにより、AIは以下のような判断が可能になります：

- 「1.5mまでは道が開いている」
- 「それより先は障害物がある」
- 「左右に比べて中央の道が最も開いている」
- 「信頼度75%で前進を推奨」

## トラブルシューティング

### グリッドが正しく表示されない

1. カメラキャリブレーションが完了していることを確認
   - `/calibration/status` でチェック

2. カメラ高さと俯角が現実に近い値か確認
   - 実際にメジャーで測定してみる

### 距離が正確でない

1. チェッカーボードキャリブレーションを再実行
2. カメラ取り付け角度を確認
3. より多くの位置で手動キャリブレーション

### セグメンテーションとの合成がおかしい

1. ROADマッピングが正しく設定されているか確認
2. 歪み補正（undistort）を有効にしてみる

## ファイル構成

```
jetracer-agent/
├── http_server/
│   ├── core/
│   │   └── distance_grid.py      # グリッド計算ロジック
│   └── routes/
│       └── distance_grid.py      # API エンドポイント
└── calibration_data/
    └── grid_config.json          # グリッド設定保存ファイル

yana-brain/
└── src/
    └── ui/
        └── distance_grid_panel.py # GUIコンポーネント
```
