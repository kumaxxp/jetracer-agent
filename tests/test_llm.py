#!/usr/bin/env python3
"""LLM基本動作テスト"""

from llama_cpp import Llama
from pathlib import Path

MODEL_PATH = Path.home() / "projects/jetracer-agent/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"

def test_basic_generation():
    """基本的な応答生成テスト"""
    llm = Llama(
        model_path=str(MODEL_PATH),
        n_gpu_layers=-1,
        n_ctx=2048,
        verbose=False,
    )

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": "こんにちは"}],
        max_tokens=64,
    )

    content = response["choices"][0]["message"]["content"]
    print(f"Response: {content}")
    assert len(content) > 0, "Empty response"
    print("✓ Basic generation test passed")

if __name__ == "__main__":
    test_basic_generation()
