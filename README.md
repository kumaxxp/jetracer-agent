# JetRacer Agent

NVIDIA Jetson上で動作するJetRacer用AIエージェント。ローカルLLM（Qwen2.5）とMCPサーバーを組み合わせ、自然言語で画像処理タスクを実行できます。

## 機能

- **ローカルLLM**: Qwen2.5-1.5B (GGUF形式) をGPUアクセラレーションで実行
- **MCPサーバー**: 画像処理ツールを提供
  - `list_images` - フォルダ内の画像一覧取得
  - `get_image_info` - 画像メタ情報の取得
  - `segment_image` - 画像セグメンテーション
  - `evaluate_quality` - 画像品質評価（ブレ・露出）

## 動作環境

- NVIDIA Jetson Orin Nano
- JetPack 6.x
- Python 3.10

## セットアップ

```bash
# 仮想環境の作成
python3 -m venv venv
source venv/bin/activate

# 依存パッケージのインストール
pip install -r requirements.txt

# モデルのダウンロード
mkdir -p models
# Qwen2.5-1.5B-Instruct-GGUF をダウンロードして models/ に配置
```

## 使い方

```bash
python main.py
```

```
JetRacer Agent Ready
Type 'quit' to exit

You: 画像フォルダの一覧を見せて
  [Tool] list_images({"folder": "/path/to/images"})
  [Result] {"images": ["image1.jpg", "image2.jpg"]}...
Agent: フォルダ内に2枚の画像があります...
```

## プロジェクト構成

```
jetracer-agent/
├── main.py           # エントリーポイント
├── agent/            # LLMエージェント
├── mcp_server/       # MCPサーバー・ツール
├── models/           # LLMモデル（.gitignore対象）
├── scripts/          # ユーティリティスクリプト
└── tests/            # テスト
```

## ライセンス

MIT
