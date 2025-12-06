#!/usr/bin/env python3
"""
軽量テストスクリプト - LLMロードなしでYANAの機能をテスト
VSCodeリモート開発時のメモリ負荷を軽減
"""

import asyncio
import json
import sys
from pathlib import Path

# プロジェクトパスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# =============================================================================
# キーワード検出テスト（LLM不要）
# =============================================================================

# Vision機能用キーワード
VISION_KEYWORDS = [
    "何が見える", "何が映って", "説明して", "見て", "画像を見て",
    "写真を見て", "これは何", "何ですか", "教えて"
]

TOOL_KEYWORDS = {
    "check_camera": [
        "カメラ", "撮影", "状態確認", "動いて", "接続", "起動", "カメラ確認"
    ],
    "list_images": [
        "画像一覧", "画像を見", "リスト", "ファイル一覧", "中身を確認",
        "フォルダの中", "ディレクトリ", "何がある", "画像ファイル"
    ],
    "analyze_frame": [
        "明るさ", "コントラスト", "分析", "フレーム分析"
    ],
    "detect_objects": [
        "物体検出", "何が映", "検出", "認識", "YOLO", "オブジェクト"
    ],
    "check_system_resources": [
        "メモリ", "リソース", "GPU", "システム", "使用状況"
    ],
    "check_model_files": [
        "モデルファイル", "モデル確認", "ファイル存在"
    ],
    "evaluate_quality": [
        "品質", "ブレ", "露出", "クオリティ", "使える"
    ],
    "segment_image": [
        "セグメント", "領域", "道路検出", "セグメンテーション"
    ],
    "get_image_info": [
        "画像情報", "メタ情報", "サイズ", "解像度"
    ],
}


def detect_tool(text: str, available_tools: set) -> tuple:
    """キーワードマッチングでツール検出（スコアベース）"""
    best_tool = None
    best_score = 0

    for tool_name, keywords in TOOL_KEYWORDS.items():
        if tool_name not in available_tools:
            continue

        score = 0
        matched_keywords = []
        for kw in keywords:
            if kw in text:
                score += len(kw)
                matched_keywords.append(kw)

        if score > best_score:
            best_score = score
            best_tool = tool_name

    return best_tool, best_score


def detect_vision_request(text: str) -> bool:
    """Vision（画像説明）リクエストを検出"""
    return any(kw in text for kw in VISION_KEYWORDS)


def test_keyword_detection():
    """キーワード検出テスト"""
    print("=" * 50)
    print("1. キーワード検出テスト（LLM不要）")
    print("=" * 50)

    # Vision キーワードテスト
    print("\n  [Vision キーワード]")
    vision_tests = [
        ("カメラに何が映っていますか？", True),
        ("この画像を見て説明して", True),
        ("カメラの状態を確認して", False),
    ]
    for text, expected in vision_tests:
        result = detect_vision_request(text)
        status = "✓" if result == expected else "✗"
        print(f"  {status} \"{text}\" → Vision={result}")

    print("\n  [ツール キーワード]")

    available_tools = set(TOOL_KEYWORDS.keys())

    test_cases = [
        ("カメラの状態を確認して", "check_camera"),
        ("カメラを確認", "check_camera"),
        ("撮影して", "check_camera"),
        ("画像一覧を見せて", "list_images"),
        ("フォルダの中を確認して", "list_images"),
        ("dataフォルダの画像を見せて", "list_images"),
        ("明るさを分析して", "analyze_frame"),
        ("メモリ確認", "check_system_resources"),
        ("物体検出して", "detect_objects"),
        ("画像の品質をチェック", "evaluate_quality"),
    ]

    passed = 0
    failed = 0

    for text, expected in test_cases:
        result, score = detect_tool(text, available_tools)
        if result == expected:
            print(f"  ✓ \"{text}\" → {result} (score:{score})")
            passed += 1
        else:
            print(f"  ✗ \"{text}\" → {result} (期待:{expected})")
            failed += 1

    print(f"\n結果: {passed}/{len(test_cases)} 成功")
    return failed == 0


# =============================================================================
# MCPツールテスト（LLM不要）
# =============================================================================

async def test_mcp_tools():
    """MCPツール実行テスト"""
    print("\n" + "=" * 50)
    print("2. MCPツール実行テスト（LLM不要）")
    print("=" * 50)

    venv_python = Path(__file__).parent.parent / "venv" / "bin" / "python3"
    mcp_server = Path(__file__).parent.parent / "mcp_server" / "server.py"

    python_cmd = str(venv_python) if venv_python.exists() else "python3"

    server_params = StdioServerParameters(
        command=python_cmd,
        args=[str(mcp_server)],
    )

    issues = []

    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_result = await session.list_tools()
                tools = {t.name: t for t in tools_result.tools}
                print(f"  接続OK: {len(tools)}ツール利用可能")

                # check_camera
                print("\n  [check_camera]")
                try:
                    result = await session.call_tool("check_camera", {})
                    data = json.loads(result.content[0].text)
                    if data.get("connected"):
                        print(f"    ✓ カメラ接続: {data.get('resolution')}")
                    else:
                        print(f"    △ カメラ未接続: {data.get('error', 'N/A')}")
                except Exception as e:
                    issues.append(f"check_camera: {e}")
                    print(f"    ✗ エラー: {e}")

                # check_system_resources
                print("\n  [check_system_resources]")
                try:
                    result = await session.call_tool("check_system_resources", {})
                    data = json.loads(result.content[0].text)
                    free_mb = data.get("memory_free_mb", 0)
                    print(f"    ✓ メモリ空き: {free_mb}MB")
                    if free_mb < 1000:
                        issues.append(f"メモリ不足: {free_mb}MB")
                        print(f"    ⚠ 警告: メモリ不足")
                except Exception as e:
                    issues.append(f"check_system_resources: {e}")
                    print(f"    ✗ エラー: {e}")

                # check_model_files
                print("\n  [check_model_files]")
                try:
                    result = await session.call_tool("check_model_files", {})
                    data = json.loads(result.content[0].text)
                    llm = data.get("llm", {})
                    if llm.get("exists"):
                        print(f"    ✓ LLM: {llm.get('path')}")
                    else:
                        issues.append("LLMモデル未検出")
                        print(f"    ✗ LLM未検出")
                except Exception as e:
                    issues.append(f"check_model_files: {e}")
                    print(f"    ✗ エラー: {e}")

                # list_images
                print("\n  [list_images]")
                try:
                    result = await session.call_tool("list_images", {"folder": "data"})
                    data = json.loads(result.content[0].text)
                    print(f"    ✓ 画像数: {data.get('count', 0)}枚")
                except Exception as e:
                    print(f"    △ {e}")

    except Exception as e:
        issues.append(f"MCP接続エラー: {e}")
        print(f"  ✗ MCP接続失敗: {e}")

    return issues


# =============================================================================
# メイン
# =============================================================================

async def main():
    print("\n" + "=" * 50)
    print("YANA 軽量テスト（LLMロードなし）")
    print("=" * 50)
    print("※ このテストはLLMをロードしないため、メモリ負荷が低いです\n")

    # キーワード検出テスト
    keyword_ok = test_keyword_detection()

    # MCPツールテスト
    mcp_issues = await test_mcp_tools()

    # サマリ
    print("\n" + "=" * 50)
    print("テスト結果サマリ")
    print("=" * 50)

    if keyword_ok and not mcp_issues:
        print("✓ 全テスト成功（LLM応答は未テスト）")
    else:
        if not keyword_ok:
            print("✗ キーワード検出に問題あり")
        if mcp_issues:
            print(f"△ MCP問題: {len(mcp_issues)}件")
            for issue in mcp_issues:
                print(f"  - {issue}")

    print("\n※ LLM応答テストは 'python3 cli.py' で手動実行してください")


if __name__ == "__main__":
    asyncio.run(main())
