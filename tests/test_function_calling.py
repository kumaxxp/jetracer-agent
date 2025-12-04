#!/usr/bin/env python3
"""Function Calling テスト"""

import json
from llama_cpp import Llama
from pathlib import Path

MODEL_PATH = Path.home() / "projects/jetracer-agent/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_images",
            "description": "指定フォルダ内の画像ファイル一覧を取得する",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "画像フォルダのパス"
                    }
                },
                "required": ["folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "segment_image",
            "description": "画像をセグメンテーション処理する",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {
                        "type": "string",
                        "description": "画像ファイルのパス"
                    }
                },
                "required": ["image_path"]
            }
        }
    }
]

def test_function_calling():
    """ツール呼び出しテスト"""
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )

    response = llm.create_chat_completion(
        messages=[
            {"role": "user", "content": "/home/jetson/imagesフォルダの画像一覧を見せて"}
        ],
        tools=TOOLS,
        tool_choice="auto",
    )

    message = response["choices"][0]["message"]
    print(f"Response: {json.dumps(message, indent=2, ensure_ascii=False)}")

    if "tool_calls" in message:
        print("✓ Function calling test passed")
        for tool_call in message["tool_calls"]:
            print(f"  Tool: {tool_call['function']['name']}")
            print(f"  Args: {tool_call['function']['arguments']}")
    else:
        print("⚠ No tool calls in response (may need prompt adjustment)")

if __name__ == "__main__":
    test_function_calling()
