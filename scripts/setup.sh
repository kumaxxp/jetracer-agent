#!/bin/bash
# scripts/setup.sh

set -e

PROJECT_DIR=~/projects/jetracer-agent
MODEL_DIR=$PROJECT_DIR/models
MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf"

# 1. ディレクトリ作成
mkdir -p $PROJECT_DIR/{models,mcp_server/tools,agent,tests,scripts}

# 2. 仮想環境作成
cd $PROJECT_DIR
python3 -m venv venv --system-site-packages

# 3. activate スクリプトにCUDA設定追加
cat << 'EOF' >> venv/bin/activate

# CUDA settings for JetPack 6.2.1
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
EOF

# 4. 仮想環境有効化
source venv/bin/activate

# 5. pip更新
pip install --upgrade pip setuptools wheel

# 6. llama-cpp-python インストール（CUDA有効）
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir

# 7. その他パッケージ
pip install mcp httpx pydantic numpy opencv-python pillow

# 8. モデルダウンロード（未存在時のみ）
if [ ! -f "$MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
    echo "Downloading model..."
    wget -O $MODEL_DIR/qwen2.5-1.5b-instruct-q4_k_m.gguf $MODEL_URL
fi

# 9. requirements.txt 生成
pip freeze > requirements.txt

echo "Setup complete!"
echo "Activate with: source venv/bin/activate"
