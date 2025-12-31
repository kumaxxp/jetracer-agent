#!/bin/bash

# llama-server + jetracer-agent 起動スクリプト

LLAMA_SERVER_BIN="${HOME}/projects/llama.cpp/build/bin/llama-server"
MODEL_PATH="${HOME}/models/functiongemma-270m-it-Q4_K_M.gguf"
LLAMA_PORT=8081
JETRACER_PORT=8000

# 色付きログ
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# クリーンアップ関数
cleanup() {
    log_info "Shutting down..."
    if [ ! -z "$LLAMA_PID" ]; then
        kill $LLAMA_PID 2>/dev/null
        log_info "llama-server stopped"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# 前提条件チェック
if [ ! -f "$LLAMA_SERVER_BIN" ]; then
    log_error "llama-server not found at $LLAMA_SERVER_BIN"
    log_error "Please build llama.cpp first:"
    echo "  cd ~/projects/llama.cpp && mkdir build && cd build"
    echo "  cmake .. -DGGML_CUDA=ON && cmake --build . -j\$(nproc)"
    exit 1
fi

if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model not found at $MODEL_PATH"
    exit 1
fi

# 既存プロセス確認
if lsof -i :$LLAMA_PORT > /dev/null 2>&1; then
    log_warn "Port $LLAMA_PORT already in use, killing existing process..."
    fuser -k $LLAMA_PORT/tcp 2>/dev/null
    sleep 1
fi

if lsof -i :$JETRACER_PORT > /dev/null 2>&1; then
    log_warn "Port $JETRACER_PORT already in use, killing existing process..."
    fuser -k $JETRACER_PORT/tcp 2>/dev/null
    sleep 1
fi

# llama-server 起動
log_info "Starting llama-server on port $LLAMA_PORT..."
$LLAMA_SERVER_BIN \
    --model "$MODEL_PATH" \
    --port $LLAMA_PORT \
    --host 127.0.0.1 \
    --n-gpu-layers 99 \
    --ctx-size 1024 \
    --threads 4 \
    --log-disable \
    &
LLAMA_PID=$!

# サーバー起動待機
log_info "Waiting for llama-server to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:$LLAMA_PORT/health | grep -q "ok"; then
        log_info "llama-server is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "llama-server failed to start"
        cleanup
    fi
    sleep 1
done

# jetracer-agent 起動
log_info "Starting jetracer-agent on port $JETRACER_PORT..."
cd ~/projects/jetracer-agent
source venv/bin/activate
python -m http_server.main

# メインプロセス終了時にクリーンアップ
cleanup
