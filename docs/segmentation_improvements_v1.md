# セグメンテーション処理改良 技術資料

**作成日**: 2025年12月12日  
**対象システム**: JetRacer自律走行システム (jetracer-agent / yana-brain)  
**プラットフォーム**: Jetson Orin Nano Super (8GB) + PC (RTX 2080 Super)

---

## 1. 概要

本資料は、JetRacerプロジェクトにおけるセグメンテーション処理の改良内容をまとめたものである。主な改良点は以下の通り：

1. ADE20K完全ラベルマッピングの実装（学習データ品質問題の解決）
2. 学習データ検証機能の追加
3. 軽量モデル（DeepLabV3+）によるリアルタイム推論の実装
4. モデルキャッシュによる推論高速化

---

## 2. 問題の発見と解決

### 2.1 学習データ品質問題

**発見された問題:**
- 学習済み軽量モデルが、床以外のオブジェクト（布団、クッション、壁など）をROAD（走行可能領域）として誤分類
- 学習マスクの検証で、ROAD比率が86.3%と異常に高い値を示した（床面積は約33.6%のはず）

**根本原因:**
`training_manager.py`内のADE20Kラベル変換テーブルが不完全だった。

```python
# 旧実装（11クラスのみ）
ADE20K_LABEL_IDS = {
    "wall": 1,
    "floor": 4,
    "ceiling": 6,
    ...  # 139クラスが欠落
}
```

150クラス中11クラスしか定義されておらず、未定義クラス（139クラス）がデフォルトでROADとして扱われていた。

### 2.2 解決策：完全ADE20Kマッピング

**新規ファイル:** `http_server/core/ade20k_full_labels.py`

```python
# ADE20K 150クラス完全定義（0-indexed）
ADE20K_LABELS = {
    0: "wall",
    1: "building",
    2: "sky",
    3: "floor",      # ← ROAD候補
    4: "tree",
    5: "ceiling",
    6: "road",       # ← ROAD候補
    ...
    149: "crt screen"
}

# 逆引き辞書
ADE20K_ID_TO_NAME = ADE20K_LABELS
ADE20K_NAME_TO_ID = {v: k for k, v in ADE20K_LABELS.items()}

def get_road_label_ids(road_label_names: list) -> set:
    """ROADラベル名からIDセットを取得"""
    return {ADE20K_NAME_TO_ID[name] for name in road_label_names 
            if name in ADE20K_NAME_TO_ID}
```

**重要な修正点:**
- ADE20Kは**0-indexed**（0〜149）
- 旧コードは1-indexedと誤認していた
- `floor`はID=3、`road`はID=6

---

## 3. 学習データ検証機能

### 3.1 デバッグAPI

**エンドポイント:** `GET /training/debug/{dataset_name}/masks`

学習データの品質を視覚的に確認するためのAPI。

**レスポンス:**
```json
{
  "dataset_name": "1213",
  "road_labels": ["floor", "road", "earth"],
  "images": [
    {
      "name": "frame_000001.jpg",
      "original_base64": "...",
      "oneformer_mask_base64": "...",  // カラー表示
      "training_mask_base64": "...",   // 緑=ROAD, 赤=MYCAR
      "classes": [
        {"id": 0, "name": "wall", "percentage": 15.2, "is_road": false},
        {"id": 3, "name": "floor", "percentage": 33.6, "is_road": true},
        ...
      ],
      "stats": {
        "other_pct": 64.2,
        "road_pct": 33.6,
        "mycar_pct": 2.2
      }
    }
  ]
}
```

### 3.2 PC側UI（training_panel.py）

Training タブに「Debug: Training Data Verification」セクションを追加。

- 最大5枚の画像を横並びで表示
- 各画像について：Original | OneFormer Mask | Training Mask
- クラス統計とROADフラグを表示

---

## 4. 軽量モデルによるリアルタイム推論

### 4.1 アーキテクチャ比較

| 項目 | OneFormer | DeepLabV3+ (Lightweight) |
|------|-----------|--------------------------|
| クラス数 | 150 (ADE20K) | 3 (Other/ROAD/MYCAR) |
| 推論時間 | ~15秒 | ~50ms |
| 用途 | アノテーション、高精度分析 | リアルタイム走行判断 |
| バックボーン | Swin Transformer | MobileNetV2 |

### 4.2 API実装

**エンドポイント:** `GET /distance-grid/{camera_id}/analyze-segmentation-lightweight`

```python
@router.get("/{camera_id}/analyze-segmentation-lightweight")
async def analyze_segmentation_lightweight(camera_id: int, undistort: bool = False):
    """軽量モデル（DeepLabV3+）でセグメンテーション結果をグリッド分析"""
    
    # PyTorchモデルを優先（CUDA対応）
    if pth_path.exists():
        segmentation, inference_time, model_type = _run_lightweight_pth(frame, pth_path)
    elif onnx_path.exists():
        segmentation, inference_time, model_type = _run_lightweight_onnx(frame, onnx_path)
```

**レスポンス:**
```json
{
  "camera_id": 0,
  "model_type": "PyTorch (cuda)",
  "inference_time_ms": 50.3,
  "road_percentage": 33.6,
  "cell_analysis": [[0.95, 0.92, ...], ...],
  "navigation_hint": {
    "recommended_steering": 0.05,
    "recommended_throttle": 0.6,
    "confidence": 0.85
  },
  "overlay_base64": "..."
}
```

### 4.3 OpenCV DNN CUDA問題

**発生した問題:**
JetsonのOpenCVがCUDA DNNサポートなしでビルドされていた。

```bash
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i cuda
# → 出力なし（CUDAサポートなし）
```

**エラーメッセージ:**
```
cv2.error: (-215:Assertion failed) preferableBackend != DNN_BACKEND_CUDA || 
IS_DNN_CUDA_TARGET(preferableTarget) in function 'validateBackendAndTarget'
```

**解決策:**
ONNXモデルではなくPyTorchモデルを優先使用するように変更。

```python
# モデル優先順位
# 1. PyTorch (.pth) - CUDA対応、高速
# 2. ONNX (.onnx) - CPUフォールバック、低速
```

---

## 5. モデルキャッシュによる高速化

### 5.1 問題

毎回のリクエストでモデルをロードしていたため、推論時間が800ms以上かかっていた。

### 5.2 解決策

グローバルキャッシュを実装し、モデルをメモリに保持。

```python
# 軽量モデルのキャッシュ
_lightweight_model_cache = {
    "model": None,
    "device": None,
    "path": None
}

def _run_lightweight_pth(frame: np.ndarray, model_path) -> tuple:
    global _lightweight_model_cache
    
    # キャッシュヒット判定
    if (_lightweight_model_cache["model"] is None or 
        _lightweight_model_cache["path"] != model_path_str):
        
        # 新規ロード
        model = smp.DeepLabV3Plus(...)
        model.load_state_dict(torch.load(model_path_str))
        model.to(device)
        model.eval()
        
        # キャッシュ保存
        _lightweight_model_cache["model"] = model
        _lightweight_model_cache["path"] = model_path_str
    else:
        # キャッシュ利用
        model = _lightweight_model_cache["model"]
    
    # 推論実行
    with torch.no_grad():
        output = model(img)
```

### 5.3 性能改善結果

| 状態 | 推論時間 |
|------|---------|
| キャッシュなし（毎回ロード） | ~858ms |
| キャッシュあり（2回目以降） | ~50ms |

**17倍の高速化**を達成。

---

## 6. ファイル構成

### 6.1 jetracer-agent（Jetson側）

```
http_server/
├── core/
│   ├── ade20k_full_labels.py   # 新規：ADE20K 150クラス完全定義
│   └── training_manager.py      # 修正：完全マッピング使用
├── routes/
│   ├── distance_grid.py         # 修正：軽量モデル推論、キャッシュ
│   └── training.py              # 修正：デバッグAPI追加
```

### 6.2 yana-brain（PC側）

```
src/
├── jetson_client.py             # 修正：デバッグAPI呼び出し追加
└── ui/
    ├── training_panel.py        # 修正：デバッグセクション追加
    └── ai_decision_panel.py     # 修正：Lightweightボタン追加
```

---

## 7. 使用方法

### 7.1 学習データ検証

1. PC側UIのTrainingタブを開く
2. 「Debug: Training Data Verification」セクションでデータセットを選択
3. 「🔍 Load Debug Data」をクリック
4. ROAD比率が期待通りか確認（床面積と一致すべき）

**異常検出時の対処:**
```bash
# 古い学習データを削除
rm -rf ~/jetracer_data/datasets/{dataset_name}/training_data

# 再度学習を実行（正しいマッピングで再生成）
```

### 7.2 モデル比較

1. PC側UIのCamerasタブ → AI Decision Visualization
2. 「🔍 OneFormer」ボタン：高精度分析（~15秒）
3. 「⚡ Lightweight」ボタン：高速分析（~50ms）

---

## 8. 今後の課題

1. **TensorRT最適化**: PyTorchモデルをTensorRTに変換し、さらに高速化（目標: <20ms）
2. **CUDA対応OpenCVのビルド**: ONNXモデルもGPUで実行可能に
3. **学習データ量の増加**: 現在の少量データでは汎化性能が不足
4. **リアルタイム自律走行への統合**: 軽量モデルを走行制御ループに組み込み

---

## 9. 参考情報

### ADE20Kラベル一覧（主要クラス）

| ID | ラベル名 | ROAD候補 |
|----|---------|----------|
| 0 | wall | No |
| 3 | floor | **Yes** |
| 6 | road | **Yes** |
| 9 | grass | Optional |
| 11 | sidewalk | Optional |
| 13 | earth | Optional |
| 29 | rug | No |
| 52 | path | Optional |

### モデルパス

- PyTorch: `~/models/best_model.pth`
- ONNX: `~/models/road_segmentation.onnx`

### 関連ドキュメント

- [jetracer_project_plan_v2.md](../../../jetracer_project_plan_v2.md)
- [jetracer_technical_reference_v2.md](../../../jetracer_technical_reference_v2.md)
