#!/usr/bin/env python3
"""
Gemma 3 4Bでのツール呼び出しテスト
4つのアプローチを比較評価する
"""

import requests
import json
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# =============================================================================
# 設定
# =============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:4b"

# テストケース: (入力, 期待するツール名, 期待する引数)
TEST_CASES = [
    ("dataフォルダの画像一覧を見せて", "list_images", {"folder": "data"}),
    ("imagesフォルダの中身を確認", "list_images", {"folder": "images"}),
    ("カメラの状態を確認して", "check_camera", {}),
    ("カメラは動いてる？", "check_camera", {}),
    ("img_001.jpgを分析して", "analyze_image", {"path": "img_001.jpg"}),
    ("data/frame_010.jpgの品質をチェック", "analyze_image", {"path": "data/frame_010.jpg"}),
    ("こんにちは", None, None),
    ("今日の天気は？", None, None),
]

# =============================================================================
# データ構造
# =============================================================================

@dataclass
class ToolCallResult:
    """ツール呼び出し結果"""
    success: bool
    tool_name: Optional[str]
    args: Dict[str, Any]
    raw_output: str
    error: Optional[str] = None

# =============================================================================
# Ollama呼び出し
# =============================================================================

def call_ollama(prompt: str, json_mode: bool = False) -> str:
    """Ollamaにプロンプトを送信"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 300,
        }
    }
    if json_mode:
        payload["format"] = "json"

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"ERROR: {e}"

# =============================================================================
# アプローチ1: ReActパターン
# =============================================================================

class ReActApproach:
    """ReActパターンでツール呼び出し"""

    SYSTEM_PROMPT = """あなたはJetRacerロボットのアシスタントYANAです。

利用可能なツール:
- list_images(folder): 指定フォルダ内の画像一覧を取得
- check_camera(): カメラの状態を確認
- analyze_image(path): 指定パスの画像品質を評価

ユーザーの要求に応えるため、以下の形式で回答してください：

Thought: [何をすべきか考える]
Action: [ツール名]
Action Input: {"引数名": "値"}

ツールが不要な場合：
Thought: [考え]
Answer: [最終回答]

重要: ActionとAction Inputは必ず上記の形式で出力してください。"""

    def process(self, user_input: str) -> ToolCallResult:
        prompt = f"{self.SYSTEM_PROMPT}\n\nユーザー: {user_input}"
        output = call_ollama(prompt)
        return self._parse_output(output)

    def _parse_output(self, output: str) -> ToolCallResult:
        # Action パターン検出
        action_match = re.search(r'Action:\s*(\w+)', output, re.IGNORECASE)
        input_match = re.search(r'Action Input:\s*(\{.*?\})', output, re.DOTALL)

        if action_match:
            tool_name = action_match.group(1).lower()
            args = {}
            if input_match:
                try:
                    args = json.loads(input_match.group(1))
                except json.JSONDecodeError:
                    pass
            return ToolCallResult(
                success=True,
                tool_name=tool_name,
                args=args,
                raw_output=output
            )

        # Answer パターン（ツール不要）
        if re.search(r'Answer:', output, re.IGNORECASE):
            return ToolCallResult(
                success=True,
                tool_name=None,
                args={},
                raw_output=output
            )

        return ToolCallResult(
            success=False,
            tool_name=None,
            args={},
            raw_output=output,
            error="パース失敗"
        )

# =============================================================================
# アプローチ2: シンプルJSON
# =============================================================================

class SimpleJSONApproach:
    """JSON形式でツール呼び出し"""

    SYSTEM_PROMPT = """あなたはJetRacerロボットのアシスタントです。

利用可能なツール:
1. list_images - 画像一覧取得。引数: folder(文字列)
2. check_camera - カメラ状態確認。引数: なし
3. analyze_image - 画像品質評価。引数: path(文字列)

ユーザーの要求に対して、必ず以下のJSON形式のみで回答してください。

ツールを使う場合:
{"tool": "ツール名", "args": {"引数名": "値"}}

ツールを使わない場合（雑談など）:
{"tool": null, "reply": "回答テキスト"}"""

    def process(self, user_input: str) -> ToolCallResult:
        prompt = f"{self.SYSTEM_PROMPT}\n\nユーザー: {user_input}\n\nJSON出力:"
        output = call_ollama(prompt, json_mode=True)
        return self._parse_output(output)

    def _parse_output(self, output: str) -> ToolCallResult:
        try:
            # JSON部分を抽出
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return ToolCallResult(
                    success=True,
                    tool_name=data.get("tool"),
                    args=data.get("args", {}),
                    raw_output=output
                )
        except json.JSONDecodeError:
            pass

        return ToolCallResult(
            success=False,
            tool_name=None,
            args={},
            raw_output=output,
            error="JSONパース失敗"
        )

# =============================================================================
# アプローチ3: 2段階処理（意図分類→引数抽出）
# =============================================================================

class TwoStageApproach:
    """2段階: まず意図を分類、次に引数を抽出"""

    INTENT_PROMPT = """ユーザーの要求を以下のカテゴリに分類してください。

カテゴリ:
- LIST_IMAGES: 画像一覧を見たい
- CHECK_CAMERA: カメラ状態を確認したい
- ANALYZE_IMAGE: 画像を分析したい
- CHAT: 雑談・その他

カテゴリ名のみを出力してください。"""

    def process(self, user_input: str) -> ToolCallResult:
        # Stage 1: 意図分類
        intent_prompt = f"{self.INTENT_PROMPT}\n\nユーザー: {user_input}\n\nカテゴリ:"
        intent_output = call_ollama(intent_prompt)
        intent = intent_output.strip().upper()

        # Stage 2: 意図に応じて処理
        if "LIST_IMAGES" in intent:
            folder = self._extract_folder(user_input)
            return ToolCallResult(
                success=True,
                tool_name="list_images",
                args={"folder": folder},
                raw_output=f"Intent: {intent}, Folder: {folder}"
            )

        elif "CHECK_CAMERA" in intent:
            return ToolCallResult(
                success=True,
                tool_name="check_camera",
                args={},
                raw_output=f"Intent: {intent}"
            )

        elif "ANALYZE_IMAGE" in intent:
            path = self._extract_path(user_input)
            return ToolCallResult(
                success=True,
                tool_name="analyze_image",
                args={"path": path},
                raw_output=f"Intent: {intent}, Path: {path}"
            )

        else:  # CHAT
            return ToolCallResult(
                success=True,
                tool_name=None,
                args={},
                raw_output=f"Intent: {intent}"
            )

    def _extract_folder(self, text: str) -> str:
        """テキストからフォルダ名を抽出"""
        # LLMで抽出
        prompt = f"次の文からフォルダ名を抽出してください。なければdataと答えてください。\n文: {text}\nフォルダ名:"
        result = call_ollama(prompt).strip()
        # 余計な文字を除去
        result = re.sub(r'[「」\'\"\s]', '', result)
        return result if result and len(result) < 50 else "data"

    def _extract_path(self, text: str) -> str:
        """テキストからファイルパスを抽出"""
        # 正規表現で直接抽出を試みる
        path_match = re.search(r'[\w/]+\.(?:jpg|jpeg|png)', text, re.IGNORECASE)
        if path_match:
            return path_match.group()

        # LLMで抽出
        prompt = f"次の文から画像ファイルのパスを抽出してください。\n文: {text}\nパス:"
        result = call_ollama(prompt).strip()
        return result if result else "unknown.jpg"

# =============================================================================
# アプローチ4: ハイブリッド（キーワード + LLM補助）
# =============================================================================

class HybridApproach:
    """キーワードマッチング + LLM補助"""

    TOOL_KEYWORDS = {
        "list_images": ["画像一覧", "画像を見", "リスト", "ファイル一覧", "中身を確認", "フォルダ"],
        "check_camera": ["カメラ", "撮影", "状態確認", "動いて"],
        "analyze_image": ["分析", "品質", "チェック", "評価", ".jpg", ".png"],
    }

    def process(self, user_input: str) -> ToolCallResult:
        # キーワードでツール検出
        tool = self._detect_tool(user_input)

        if tool:
            args = self._extract_args(tool, user_input)
            return ToolCallResult(
                success=True,
                tool_name=tool,
                args=args,
                raw_output=f"Keyword match: {tool}"
            )
        else:
            return ToolCallResult(
                success=True,
                tool_name=None,
                args={},
                raw_output="No tool needed"
            )

    def _detect_tool(self, text: str) -> Optional[str]:
        """キーワードマッチングでツール検出"""
        text_lower = text.lower()
        for tool, keywords in self.TOOL_KEYWORDS.items():
            if any(kw in text_lower or kw in text for kw in keywords):
                return tool
        return None

    def _extract_args(self, tool: str, text: str) -> Dict[str, Any]:
        """ツールに応じた引数抽出"""
        if tool == "list_images":
            # フォルダ名パターン
            folder_match = re.search(r'(\w+)フォルダ|フォルダ[「：:\s]*(\w+)', text)
            if folder_match:
                folder = folder_match.group(1) or folder_match.group(2)
                return {"folder": folder}
            # 一般的なフォルダ名
            for common in ["data", "images", "output", "raw"]:
                if common in text.lower():
                    return {"folder": common}
            return {"folder": "data"}

        elif tool == "analyze_image":
            path_match = re.search(r'[\w/.-]+\.(?:jpg|jpeg|png)', text, re.IGNORECASE)
            if path_match:
                return {"path": path_match.group()}
            return {"path": "unknown.jpg"}

        return {}

# =============================================================================
# 評価・比較
# =============================================================================

def evaluate_approach(name: str, approach, test_cases: List[tuple]) -> Dict[str, Any]:
    """アプローチを評価"""
    print(f"\n{'='*60}")
    print(f"テスト: {name}")
    print('='*60)

    results = []
    correct_tool = 0
    correct_args = 0

    for user_input, expected_tool, expected_args in test_cases:
        result = approach.process(user_input)

        # ツール名の正誤判定
        tool_correct = (result.tool_name == expected_tool)
        if tool_correct:
            correct_tool += 1

        # 引数の正誤判定（ツールがある場合のみ）
        args_correct = False
        if expected_tool and expected_args:
            # 引数の主要キーが一致するか
            args_correct = all(
                result.args.get(k) == v or (k in result.args)
                for k, v in expected_args.items()
            )
            if args_correct:
                correct_args += 1
        elif expected_tool is None:
            args_correct = True
            correct_args += 1

        status = "✅" if tool_correct else "❌"
        print(f"{status} 入力: {user_input}")
        print(f"   期待: tool={expected_tool}, args={expected_args}")
        print(f"   結果: tool={result.tool_name}, args={result.args}")
        if not result.success:
            print(f"   エラー: {result.error}")
        print()

        results.append({
            "input": user_input,
            "expected_tool": expected_tool,
            "detected_tool": result.tool_name,
            "tool_correct": tool_correct,
            "args_correct": args_correct,
        })

    total = len(test_cases)

    return {
        "name": name,
        "tool_accuracy": correct_tool / total,
        "args_accuracy": correct_args / total,
        "results": results,
    }

def run_comparison():
    """全アプローチを比較"""
    approaches = [
        ("ReAct パターン", ReActApproach()),
        ("シンプルJSON", SimpleJSONApproach()),
        ("2段階処理", TwoStageApproach()),
        ("ハイブリッド", HybridApproach()),
    ]

    all_results = []

    for name, approach in approaches:
        result = evaluate_approach(name, approach, TEST_CASES)
        all_results.append(result)

    # サマリ表示
    print("\n" + "="*60)
    print("結果サマリ")
    print("="*60)
    print(f"{'アプローチ':<20} {'ツール検出':<15} {'引数抽出':<15}")
    print("-"*50)

    for r in sorted(all_results, key=lambda x: -x["tool_accuracy"]):
        tool_bar = "█" * int(r["tool_accuracy"] * 10)
        args_bar = "█" * int(r["args_accuracy"] * 10)
        print(f"{r['name']:<20} {tool_bar:<10} {r['tool_accuracy']*100:>4.0f}%  {args_bar:<10} {r['args_accuracy']*100:>4.0f}%")

    # 推奨を出力
    best = max(all_results, key=lambda x: x["tool_accuracy"])
    print(f"\n推奨アプローチ: {best['name']} (ツール検出精度: {best['tool_accuracy']*100:.0f}%)")

    return all_results

# =============================================================================
# メイン
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        # 単一アプローチテスト
        approach_name = sys.argv[2] if len(sys.argv) > 2 else "react"

        approaches = {
            "react": ReActApproach(),
            "json": SimpleJSONApproach(),
            "twostage": TwoStageApproach(),
            "hybrid": HybridApproach(),
        }

        if approach_name in approaches:
            approach = approaches[approach_name]
            print(f"テスト: {approach_name}")
            while True:
                user_input = input("\n入力 (qで終了): ")
                if user_input.lower() == 'q':
                    break
                result = approach.process(user_input)
                print(f"結果: tool={result.tool_name}, args={result.args}")
                print(f"生出力: {result.raw_output[:200]}...")
        else:
            print(f"不明なアプローチ: {approach_name}")
            print(f"利用可能: {list(approaches.keys())}")
    else:
        # 全アプローチ比較
        run_comparison()
