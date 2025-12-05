#!/bin/bash
# Gemma 3 4B セットアップスクリプト (Jetson対応)

set -e

echo "=== Gemma 3 4B セットアップ ==="

# Ollamaがインストールされているか確認
if ! command -v ollama &> /dev/null; then
    echo "Ollamaをインストールします..."
    curl -fsSL https://ollama.com/install.sh | sh

    # インストール確認
    if ! command -v ollama &> /dev/null; then
        echo "ERROR: Ollamaのインストールに失敗しました"
        echo "手動でインストールしてください: https://ollama.com/download"
        exit 1
    fi
    echo "Ollamaインストール完了"
fi

# Ollamaが起動しているか確認
if ! pgrep -x "ollama" > /dev/null; then
    echo "Ollamaを起動します..."
    ollama serve &
    sleep 5
fi

# Gemma 3 4Bをダウンロード
echo ""
echo "Gemma 3 4Bをダウンロード中..."
echo "（約2.5GB、数分かかります）"
ollama pull gemma3:4b

# メモリ状況確認
echo ""
echo "=== メモリ状況 ==="
free -h

echo ""
echo "=== セットアップ完了 ==="
echo ""
echo "テストコマンド:"
echo "  ollama run gemma3:4b"
echo ""
echo "YANAテスト:"
echo "  cd ~/projects/jetracer-agent"
echo "  source venv/bin/activate"
echo "  python -m yana.llm.tool_calling_test"
