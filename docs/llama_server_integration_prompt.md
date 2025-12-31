# llama-server プロセス分離実装プロンプト

llama-cpp-python の CUDA 競合問題を回避するため、llama.cpp の llama-server を別プロセスで起動し、HTTP API 経由で通信する構成に変更します。

---

## アーキテクチャ

```
┌─────────────────────────────────────────────────────────────┐
│                    Jetson Orin Nano                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────┐                   │
│  │ jetracer-agent (メインプロセス)      │                   │
│  │ - PyTorch セグメンテーション         │                   │
│  │ - センサー処理                       │                   │
│  │ - HTTP API (ポート 8000)             │                   │
│  └──────────────────┬──────────────────┘                   │
│                     │ HTTP (localhost:8081)                 │
│                     ▼                                       │
│  ┌─────────────────────────────────────┐                   │
│  │ llama-server (別プロセス)            │                   │
│  │ - FunctionGemma 270M                │                   │
│  │ - CUDA 推論                         │                   │
│  │ - OpenAI互換API (ポート 8081)        │                   │
│  └─────────────────────────────────────┘                   │
│                                                             │
│  ※ CUDAコンテキストが完全に分離され、競合なし              │
└─────────────────────────────────────────────────────────────┘
```

---

## Step 1: llama.cpp ビルド（CUDA有効）

```bash
# 作業ディレクトリ
cd ~/projects
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# CUDA有効でビルド
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)

# ビルド確認
ls -la bin/llama-server
# → bin/llama-server が存在すればOK

# シンボリックリンク作成（オプション）
sudo ln -sf $(pwd)/bin/llama-server /usr/local/bin/llama-server
```

---

## Step 2: llama-server 動作確認

```bash
# FunctionGemma 270M モデルのパス確認
ls -la ~/models/functiongemma-270m-it-Q4_K_M.gguf

# サーバー起動テスト
~/projects/llama.cpp/build/bin/llama-server \
  --model ~/models/functiongemma-270m-it-Q4_K_M.gguf \
  --port 8081 \
  --host 127.0.0.1 \
  --n-gpu-layers 99 \
  --ctx-size 1024 \
  --threads 4

# 別ターミナルでAPI確認
curl http://localhost:8081/health
# → {"status":"ok"} が返ればOK

# 推論テスト
curl -X POST http://localhost:8081/completion \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "センサー状態:\nlidar: front=danger, left=clear, right=clear\n\n制御意図をJSONで出力:",
    "n_predict": 64,
    "temperature": 0.1
  }'
```

---

## Step 3: functiongemma_engine.py を HTTP クライアントに変更

`http_server/core/reflex/functiongemma_engine.py` を以下に置き換え：

```python
"""FunctionGemma推論エンジン（HTTP クライアント版）

llama-server（別プロセス）にHTTP経由で推論リクエストを送信。
PyTorchとのCUDA競合を完全に回避。
"""

import json
import time
import re
import asyncio
from typing import Optional, Dict, Any
import httpx

from .intent_vocabulary import (
    ControlIntent, SpeedIntent, SpeedChangeIntent, 
    SteeringIntent, DirectionIntent, UrgencyLevel
)
from .sensor_preprocessor import AbstractSensorState


# llama-server 設定
LLAMA_SERVER_URL = "http://127.0.0.1:8081"
COMPLETION_ENDPOINT = f"{LLAMA_SERVER_URL}/completion"
HEALTH_ENDPOINT = f"{LLAMA_SERVER_URL}/health"


# システムプロンプト
SYSTEM_PROMPT = """あなたはJetRacer自律走行車の反射神経AIです。
センサー入力を受け取り、意図レベルの制御指示をJSON形式で出力してください。

## 重要な制約
- 数値での速度・角度指定は禁止です
- 必ず意図語彙（メタ言語）で出力してください
- JSONのみを出力し、説明文は不要です

## 意図語彙

【speed】stop, creep, slow, normal, fast
【speed_change】decelerate, accelerate, halve, emergency_stop, maintain
【steering】straight, slight_left, left, hard_left, slight_right, right, hard_right, more_left, more_right, center
【direction】forward, reverse, hold
【urgency】low, normal, high, critical

## センサー状態の読み方

【lidar】front/left/right の値
- clear: 安全（1m以上）
- warning: 注意（0.3-1m）
- danger: 危険（0.3m未満）
- contact: 接触

【imu】
- impact: true なら衝突発生
- lifted: true なら持ち上げられている

【audio】
- motor: stopped/running/spinning
- grip: good/slipping/lost

【road】
- visible: 道路が見えているか
- position: center/left_edge/right_edge/lost

## 出力形式（JSON）

{"speed": "slow", "steering": "straight", "reason": "前方クリア"}

## 動作シナリオ

1. 前方danger/contact → emergency_stop
2. grip=lost → halve して様子見
3. road.position=left_edge → slight_right
4. 前方warning → decelerate + 空いている方向へ
5. lifted=true → stop してエスカレート
"""


class FunctionGemmaEngine:
    """FunctionGemma推論エンジン（HTTPクライアント版）
    
    llama-server（別プロセス）にHTTP経由でリクエストを送信。
    """
    
    def __init__(self, server_url: str = LLAMA_SERVER_URL):
        self.server_url = server_url
        self.completion_endpoint = f"{server_url}/completion"
        self.health_endpoint = f"{server_url}/health"
        self._initialized = False
        self._server_available = False
        self._last_inference_time = 0.0
        self._http_client: Optional[httpx.Client] = None
    
    def initialize(self) -> bool:
        """サーバー接続確認"""
        if self._initialized:
            return True
        
        try:
            self._http_client = httpx.Client(timeout=30.0)
            
            # ヘルスチェック
            response = self._http_client.get(self.health_endpoint)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    self._server_available = True
                    self._initialized = True
                    print(f"[FunctionGemma] Connected to llama-server at {self.server_url}")
                    return True
            
            print(f"[FunctionGemma] Server not ready: {response.text}")
            return False
            
        except httpx.ConnectError:
            print(f"[FunctionGemma] Cannot connect to {self.server_url}")
            print("[FunctionGemma] Please start llama-server first:")
            print(f"  llama-server --model ~/models/functiongemma-270m-it-Q4_K_M.gguf --port 8081 --n-gpu-layers 99")
            return False
        except Exception as e:
            print(f"[FunctionGemma] Initialization failed: {e}")
            return False
    
    def infer(self, sensor_state: AbstractSensorState) -> ControlIntent:
        """センサー状態から制御意図を推論
        
        Args:
            sensor_state: 抽象化されたセンサー状態
        
        Returns:
            ControlIntent
        """
        start = time.time()
        
        if self._server_available and self._http_client:
            intent = self._infer_http(sensor_state)
        else:
            # フォールバック: ルールベース
            intent = self._infer_rule_based(sensor_state)
        
        self._last_inference_time = (time.time() - start) * 1000
        return intent
    
    def _infer_http(self, sensor_state: AbstractSensorState) -> ControlIntent:
        """HTTP経由でllama-serverに推論リクエスト"""
        prompt = f"""{SYSTEM_PROMPT}

センサー状態:
{sensor_state.to_prompt_text()}

上記の状態に対する制御意図をJSONで出力してください。"""
        
        try:
            response = self._http_client.post(
                self.completion_endpoint,
                json={
                    "prompt": prompt,
                    "n_predict": 128,
                    "temperature": 0.1,
                    "top_k": 40,
                    "top_p": 0.9,
                    "stop": ["\n\n", "```"],
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                content = data.get("content", "")
                return self._parse_response(content)
            else:
                print(f"[FunctionGemma] HTTP error: {response.status_code}")
                return self._infer_rule_based(sensor_state)
                
        except Exception as e:
            print(f"[FunctionGemma] Inference error: {e}")
            return self._infer_rule_based(sensor_state)
    
    def _parse_response(self, content: str) -> ControlIntent:
        """LLM出力をパース"""
        try:
            # JSON部分を抽出
            json_match = re.search(r'\{[^{}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(content)
            
            intent = ControlIntent()
            
            if "speed" in data:
                try:
                    intent.speed = SpeedIntent(data["speed"])
                except ValueError:
                    pass
            
            if "speed_change" in data:
                try:
                    intent.speed_change = SpeedChangeIntent(data["speed_change"])
                except ValueError:
                    pass
            
            if "steering" in data:
                try:
                    intent.steering = SteeringIntent(data["steering"])
                except ValueError:
                    pass
            
            if "direction" in data:
                try:
                    intent.direction = DirectionIntent(data["direction"])
                except ValueError:
                    pass
            
            if "urgency" in data:
                try:
                    intent.urgency = UrgencyLevel(data.get("urgency", "normal"))
                except ValueError:
                    pass
            
            intent.reason = data.get("reason", "")
            intent.escalate = data.get("escalate", False)
            
            return intent
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[FunctionGemma] Parse error: {e}, content: {content[:100]}")
            return ControlIntent(reason="parse_error")
    
    def _infer_rule_based(self, state: AbstractSensorState) -> ControlIntent:
        """ルールベースのフォールバック推論"""
        intent = ControlIntent()
        intent.speed = SpeedIntent.SLOW
        intent.steering = SteeringIntent.STRAIGHT
        intent.direction = DirectionIntent.FORWARD
        
        # 緊急停止条件
        if state.is_lifted:
            intent.speed_change = SpeedChangeIntent.EMERGENCY_STOP
            intent.reason = "lifted"
            intent.escalate = True
            return intent
        
        if state.impact_detected:
            intent.speed_change = SpeedChangeIntent.EMERGENCY_STOP
            intent.reason = "impact"
            intent.urgency = UrgencyLevel.CRITICAL
            return intent
        
        from .sensor_preprocessor import ProximityLevel, GripStatus, RoadPosition
        
        # 前方障害物
        if state.lidar_front == ProximityLevel.CONTACT:
            intent.speed_change = SpeedChangeIntent.EMERGENCY_STOP
            intent.reason = "contact_front"
            return intent
        
        if state.lidar_front == ProximityLevel.DANGER:
            intent.speed_change = SpeedChangeIntent.EMERGENCY_STOP
            intent.reason = "danger_front"
            return intent
        
        if state.lidar_front == ProximityLevel.WARNING:
            intent.speed_change = SpeedChangeIntent.DECELERATE
            if state.lidar_left == ProximityLevel.CLEAR:
                intent.steering = SteeringIntent.SLIGHT_LEFT
            elif state.lidar_right == ProximityLevel.CLEAR:
                intent.steering = SteeringIntent.SLIGHT_RIGHT
            intent.reason = "warning_front"
        
        # グリップ喪失
        if state.grip_status == GripStatus.LOST:
            intent.speed_change = SpeedChangeIntent.HALVE
            intent.reason = "grip_lost"
            return intent
        
        if state.grip_status == GripStatus.SLIPPING:
            intent.speed_change = SpeedChangeIntent.DECELERATE
            intent.reason = "slipping"
        
        # 道路追従
        if state.road_position == RoadPosition.LEFT_EDGE:
            intent.steering = SteeringIntent.SLIGHT_RIGHT
            intent.reason = "road_left_edge"
        elif state.road_position == RoadPosition.RIGHT_EDGE:
            intent.steering = SteeringIntent.SLIGHT_LEFT
            intent.reason = "road_right_edge"
        elif state.road_position == RoadPosition.LOST:
            intent.speed_change = SpeedChangeIntent.HALVE
            intent.reason = "road_lost"
            intent.escalate = True
        
        return intent
    
    def get_status(self) -> Dict[str, Any]:
        """ステータス取得"""
        return {
            "initialized": self._initialized,
            "server_url": self.server_url,
            "server_available": self._server_available,
            "last_inference_ms": round(self._last_inference_time, 2),
            "mode": "http" if self._server_available else "rule_based_fallback",
        }
    
    @property
    def is_initialized(self) -> bool:
        return self._initialized
    
    @property
    def last_inference_time(self) -> float:
        return self._last_inference_time
    
    def close(self):
        """クリーンアップ"""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
```

---

## Step 4: 起動スクリプト作成

`~/projects/jetracer-agent/scripts/start_with_llm.sh` を作成：

```bash
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
    fuser -k $LLAMA_PORT/tcp
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
python -m http_server.main

# メインプロセス終了時にクリーンアップ
cleanup
```

```bash
# 実行権限付与
chmod +x ~/projects/jetracer-agent/scripts/start_with_llm.sh
```

---

## Step 5: 動作確認

### 5.1 起動

```bash
# 統合起動スクリプトで起動
cd ~/projects/jetracer-agent
./scripts/start_with_llm.sh
```

### 5.2 API テスト

```bash
# llama-server ヘルスチェック
curl http://localhost:8081/health

# FunctionGemma 初期化
curl -X POST http://localhost:8000/reflex/initialize

# ステータス確認
curl http://localhost:8000/reflex/status

# 推論テスト（LLM）
curl -X POST http://localhost:8000/reflex/infer \
  -H "Content-Type: application/json" \
  -d '{
    "lidar": {"grid": [[200,200,200,200,200,200,200,200],[200,200,200,200,200,200,200,200],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500],[500,500,500,500,500,500,500,500]]}
  }'
```

---

## 期待される結果

### ステータス
```json
{
  "engine": {
    "initialized": true,
    "server_url": "http://127.0.0.1:8081",
    "server_available": true,
    "mode": "http",
    "last_inference_ms": 25.5
  }
}
```

### 推論（前方danger）
```json
{
  "intent": {
    "speed": null,
    "speed_change": "emergency_stop",
    "steering": "straight",
    "reason": "前方danger検出"
  },
  "inference_time_ms": 30.2
}
```

---

## トラブルシューティング

### llama-server が起動しない

```bash
# CUDAが認識されているか確認
nvidia-smi

# 手動で起動してエラーメッセージ確認
~/projects/llama.cpp/build/bin/llama-server \
  --model ~/models/functiongemma-270m-it-Q4_K_M.gguf \
  --port 8081 \
  --n-gpu-layers 99
```

### HTTPタイムアウト

```python
# functiongemma_engine.py のタイムアウトを増やす
self._http_client = httpx.Client(timeout=60.0)  # 30→60秒
```

### GPU使用されない

```bash
# llama-server 起動時のログで確認
# "offloading N layers to GPU" が表示されるはず

# n-gpu-layers を明示的に指定
--n-gpu-layers 99
```

---

## 完了後

1. llama-server + jetracer-agent が正常動作
2. CUDA推論が有効（推論時間 20-50ms 期待）
3. PyTorchセグメンテーションとの競合なし

次のステップ：PC側UIパネルの追加、または実走行テスト
