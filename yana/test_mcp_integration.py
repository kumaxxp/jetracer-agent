#!/usr/bin/env python3
"""MCP統合テスト"""

import asyncio
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from yana.llm.agent import YANAAgent

# ダミーツール（実際のMCPツールの代わり）
def list_images(folder: str = "data") -> dict:
    """画像一覧を取得（ダミー）"""
    return {
        "folder": folder,
        "images": ["img_001.jpg", "img_002.jpg", "img_003.jpg"],
        "count": 3
    }

def check_camera() -> dict:
    """カメラ状態確認（ダミー）"""
    return {
        "connected": True,
        "resolution": "640x480",
        "fps": 15,
        "frame_path": "/tmp/yana_startup_frame.jpg"
    }

def analyze_frame(image_path: str = "") -> dict:
    """画像明るさ分析（ダミー）"""
    return {
        "path": image_path,
        "brightness": 128,
        "contrast": 45.2,
        "edge_density": 0.35,
        "is_very_dark": False,
        "is_dark": False,
        "is_bright": False
    }

def detect_objects(image_path: str = "", confidence: float = 0.5) -> dict:
    """物体検出（ダミー）"""
    return {
        "path": image_path,
        "detected_objects": {
            "chair": 2,
            "laptop": 1,
            "person": 0
        },
        "total_count": 3
    }

def check_system_resources() -> dict:
    """システムリソース確認（ダミー）"""
    return {
        "memory_total_mb": 8192,
        "memory_used_mb": 4500,
        "memory_free_mb": 3692,
        "gpu_memory_used_mb": 2100,
        "cpu_percent": 25.5
    }

def check_model_files() -> dict:
    """モデルファイル確認（ダミー）"""
    return {
        "llm": {"exists": True, "path": "~/models/gemma3-4b.gguf"},
        "yolo": {"exists": True, "path": "models/yolov8n.pt"},
        "segmentation": {"exists": False, "path": "models/road_segmentation.onnx"}
    }

def evaluate_quality(image_path: str = "") -> dict:
    """画像品質評価（ダミー）"""
    return {
        "path": image_path,
        "blur_score": 125.5,
        "brightness": 128,
        "is_blurry": False,
        "is_usable": True
    }

async def main():
    # エージェント初期化
    print("="*60)
    print("YANA Agent (Gemma 3 4B) - MCP統合テスト")
    print("="*60)

    # アプローチを選択（テスト結果に応じて変更）
    approach = "hybrid"
    print(f"アプローチ: {approach}")
    print()

    agent = YANAAgent(approach=approach)

    # ツール登録
    agent.register_tool("list_images", list_images, "フォルダ内の画像一覧を取得")
    agent.register_tool("check_camera", check_camera, "カメラの状態を確認")
    agent.register_tool("analyze_frame", analyze_frame, "画像の明るさ・コントラスト分析")
    agent.register_tool("detect_objects", detect_objects, "YOLOv8による物体検出")
    agent.register_tool("check_system_resources", check_system_resources, "メモリ・GPU使用状況確認")
    agent.register_tool("check_model_files", check_model_files, "モデルファイル存在確認")
    agent.register_tool("evaluate_quality", evaluate_quality, "画像品質評価")

    print(f"登録ツール数: {len(agent.tools)}")
    print("-" * 40)
    print("終了するには 'quit' と入力")
    print()

    while True:
        try:
            user_input = input("あなた: ")
            if user_input.lower() in ['quit', 'q', '終了']:
                break
            if not user_input.strip():
                continue

            response = await agent.run(user_input)
            print(f"\nYANA: {response}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"エラー: {e}\n")

    print("\nGoodbye!")

if __name__ == "__main__":
    asyncio.run(main())
