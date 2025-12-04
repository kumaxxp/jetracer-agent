# YANA - Your Autonomous Navigation Assistant

JetRacer自律走行プロジェクトのAIアシスタント。Jetson Orin Nano上でローカルLLM（Qwen2.5）とMCPサーバーを組み合わせ、自然言語でデータ収集・アノテーション・訓練をサポートします。

## 特徴

- **ローカルLLM**: Qwen2.5-1.5B (GGUF) をGPUアクセラレーションで実行
- **MCP統合**: 画像処理ツールをLLMが自律的に呼び出し
- **GUI/CLI両対応**: NiceGUIによるWebインターフェースとCLIモード
- **セッション管理**: 作業状態を自動保存、中断からの再開に対応
- **セルフチェック**: 起動時にカメラ・モデル・リソースを自動確認

## 動作環境

- NVIDIA Jetson Orin Nano (8GB)
- JetPack 6.x
- Python 3.10

## プロジェクト構成

```
jetracer-agent/
├── main.py               # エントリーポイント（GUI/CLI）
├── cli.py                # CLIモード直接起動
├── agent/
│   └── agent.py          # YANAエージェント
├── yana/
│   ├── config.py         # 設定・パス定義
│   ├── prompts.py        # システムプロンプト
│   └── session.py        # セッション状態管理
├── mcp_server/
│   ├── server.py         # MCPサーバー
│   └── tools/
│       ├── image_list.py     # 画像一覧
│       ├── image_info.py     # 画像情報
│       ├── quality.py        # 品質評価
│       ├── segment.py        # セグメンテーション
│       ├── system.py         # システムチェック
│       └── detection.py      # 物体検出（YOLO）
├── gui/
│   ├── app.py            # メインGUI
│   ├── chat_panel.py     # YANAチャットパネル
│   └── event_bridge.py   # GUIイベント通知
├── models/               # モデルファイル（.gitignore対象）
└── static/               # 静的ファイル（アバター等）
```

## セットアップ

```bash
# 仮想環境の作成（JetPackパッケージを継承）
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt

# LLMモデルのダウンロード
mkdir -p ~/models
cd ~/models
wget https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf

# YOLOv8モデル（オプション - 自動ダウンロードも可能）
pip install ultralytics
```

## 使い方

### GUIモード（推奨）

```bash
python main.py
# または
python main.py --port 8081
```

ブラウザで http://192.168.1.65:8080 にアクセス

### CLIモード

```bash
python main.py --cli
# または
python cli.py
```

## YANA の機能

### 起動シーケンス

YANAは起動時に自動でセルフチェックを行います：

```
Your Autonomous Navigation Assistant..YANA..起動しました

システムチェック結果:
  カメラ: OK (640x480)
  → 机と椅子が見えます。室内のようですね。
  LLMモデル: OK
  セグメンテーション: OK
  メモリ: OK (3.2GB空き)

全システム正常です。何かお手伝いできることはありますか？
```

### 対話例

```
You: 画像フォルダの一覧を見せて
  [Tool] list_images({"folder": "/home/jetson/data"})
YANA: フォルダ内に45枚の画像があります。

You: ブレている画像を除外して
  [Tool] evaluate_quality({"image_path": "..."})
YANA: 45枚中、5枚にブレを検出しました。除外しますか？
```

### セッション管理

- 作業状態は `~/.yana/session.json` に自動保存
- 中断後の再起動時、前回の続きから再開可能
- `reset` コマンドでセッションをリセット

## MCPツール一覧

| ツール | 説明 |
|--------|------|
| `list_images` | フォルダ内の画像一覧を取得 |
| `get_image_info` | 画像のメタ情報を取得 |
| `evaluate_quality` | 画像の品質（ブレ・露出）を評価 |
| `segment_image` | セグメンテーション処理 |
| `check_camera` | カメラ接続確認・フレーム取得 |
| `analyze_frame` | 画像の明るさ・コントラスト分析 |
| `detect_objects` | YOLOv8による物体検出 |
| `check_system_resources` | メモリ・GPU状況確認 |
| `check_model_files` | モデルファイル存在確認 |

## 設定

`yana/config.py` で各種パスや閾値を変更可能：

```python
# モデルパス
LLM_MODEL_PATH = Path.home() / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

# 画像品質閾値
DARK_THRESHOLD = 20       # 真っ暗判定
BLUR_THRESHOLD = 100      # ブレ判定

# LLM設定
LLM_N_CTX = 4096
LLM_MAX_TOKENS = 1024
```

## トラブルシューティング

### カメラが認識されない

```bash
# CSIカメラ確認
ls /dev/video*
nvgstcapture-1.0
```

### LLMの推論が遅い

```python
# verbose=Trueで確認
llm = Llama(model_path=path, verbose=True)
# "offloading X layers to GPU" を確認
```

### メモリ不足

```bash
# 状況確認
tegrastats

# コンテキスト長を減らす（yana/config.py）
LLM_N_CTX = 2048  # 4096→2048
```

## ライセンス

MIT

## 関連プロジェクト

- [jetracer_minimal](https://github.com/kumaxxp/jetracer_minimal) - データ収集・走行制御
- [jetracer_annotation_tool](https://github.com/kumaxxp/jetracer_annotation_tool) - アノテーション・学習
