#!/bin/bash
# JetRacer HTTP Server 起動スクリプト

cd ~/projects/jetracer-agent
source venv/bin/activate

echo "Starting JetRacer HTTP API Server..."
echo "Access: http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "Endpoints:"
echo "  GET  /status  - システム状態"
echo "  POST /capture - カメラ画像取得"
echo "  POST /analyze - 統合解析"
echo "  POST /control - 車両制御"
echo "  POST /stop    - 緊急停止"
echo ""

python -m http_server.main
