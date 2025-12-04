"""YANA設定"""

from pathlib import Path

# プロジェクトルート
PROJECT_ROOT = Path(__file__).parent.parent

# モデルパス
MODEL_DIR = Path.home() / "models"
LLM_MODEL_PATH = MODEL_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
SEGMENTATION_MODEL_PATH = PROJECT_ROOT / "models" / "road_segmentation.onnx"
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolov8n.pt"

# MCPサーバー
MCP_SERVER_PATH = PROJECT_ROOT / "mcp_server" / "server.py"

# セッション保存先
YANA_DATA_DIR = Path.home() / ".yana"
SESSION_FILE = YANA_DATA_DIR / "session.json"
HISTORY_FILE = YANA_DATA_DIR / "history.jsonl"

# カメラ設定
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# 画像品質閾値
DARK_THRESHOLD = 20       # これ以下は真っ暗
DIM_THRESHOLD = 50        # 暗め
BRIGHT_THRESHOLD = 200    # 明るすぎ
BLUR_THRESHOLD = 100      # ブレ判定

# メモリ閾値
MEMORY_WARNING_MB = 1500  # 警告を出す空き容量
MEMORY_OK_MB = 2000       # 正常とみなす空き容量

# LLM設定
LLM_N_CTX = 4096
LLM_MAX_TOKENS = 1024
