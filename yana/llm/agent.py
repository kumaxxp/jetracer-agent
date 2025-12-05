#!/usr/bin/env python3
"""
YANA Agent - Gemma 3 4B対応版
テスト結果に基づいて最適なアプローチを使用
"""

import asyncio
import json
import re
import logging
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """ツール呼び出し情報"""
    name: str
    args: Dict[str, Any]

@dataclass
class AgentResponse:
    """エージェント応答"""
    tool_call: Optional[ToolCall]
    reply: Optional[str]
    raw_output: str

class YANAAgent:
    """YANA Agent - Gemma 3 4B版"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434/api/generate",
        model: str = "gemma3:4b",
        approach: str = "hybrid",  # react, json, twostage, hybrid
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.approach = approach
        self.tools: Dict[str, dict] = {}

        logger.info(f"YANAAgent初期化: model={model}, approach={approach}")

    def register_tool(self, name: str, func: Callable, description: str = ""):
        """MCPツールを登録"""
        self.tools[name] = {
            "func": func,
            "description": description,
        }
        logger.info(f"ツール登録: {name}")

    def _call_ollama(self, prompt: str, json_mode: bool = False) -> str:
        """Ollama API呼び出し"""
        payload = {
            "model": self.model,
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
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama呼び出しエラー: {e}")
            return ""

    def process(self, user_input: str) -> AgentResponse:
        """ユーザー入力を処理"""
        if self.approach == "react":
            return self._process_react(user_input)
        elif self.approach == "json":
            return self._process_json(user_input)
        elif self.approach == "twostage":
            return self._process_twostage(user_input)
        else:  # hybrid
            return self._process_hybrid(user_input)

    # -------------------------------------------------------------------------
    # アプローチ実装
    # -------------------------------------------------------------------------

    def _process_react(self, user_input: str) -> AgentResponse:
        """ReActパターン"""
        tool_list = "\n".join([
            f"- {name}: {info['description']}"
            for name, info in self.tools.items()
        ])

        prompt = f"""あなたはJetRacerロボットのアシスタントYANAです。

利用可能なツール:
{tool_list}

ユーザーの要求に応えるため、以下の形式で回答してください：

Thought: [何をすべきか考える]
Action: [ツール名]
Action Input: {{"引数名": "値"}}

ツールが不要な場合：
Thought: [考え]
Answer: [最終回答]

ユーザー: {user_input}"""

        output = self._call_ollama(prompt)

        # パース
        action_match = re.search(r'Action:\s*(\w+)', output, re.IGNORECASE)
        input_match = re.search(r'Action Input:\s*(\{.*?\})', output, re.DOTALL)

        if action_match:
            tool_name = action_match.group(1).lower()
            args = {}
            if input_match:
                try:
                    args = json.loads(input_match.group(1))
                except:
                    pass
            return AgentResponse(
                tool_call=ToolCall(name=tool_name, args=args),
                reply=None,
                raw_output=output
            )

        # 回答のみ
        answer_match = re.search(r'Answer:\s*(.+)', output, re.DOTALL)
        reply = answer_match.group(1).strip() if answer_match else output

        return AgentResponse(
            tool_call=None,
            reply=reply,
            raw_output=output
        )

    def _process_json(self, user_input: str) -> AgentResponse:
        """シンプルJSON"""
        tool_list = "\n".join([
            f"{i+1}. {name} - {info['description']}"
            for i, (name, info) in enumerate(self.tools.items())
        ])

        prompt = f"""あなたはJetRacerロボットのアシスタントです。

利用可能なツール:
{tool_list}

ユーザーの要求に対して、必ず以下のJSON形式のみで回答してください。

ツールを使う場合:
{{"tool": "ツール名", "args": {{"引数名": "値"}}}}

ツールを使わない場合:
{{"tool": null, "reply": "回答テキスト"}}

ユーザー: {user_input}

JSON出力:"""

        output = self._call_ollama(prompt, json_mode=True)

        try:
            json_match = re.search(r'\{.*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                if data.get("tool"):
                    return AgentResponse(
                        tool_call=ToolCall(name=data["tool"], args=data.get("args", {})),
                        reply=None,
                        raw_output=output
                    )
                else:
                    return AgentResponse(
                        tool_call=None,
                        reply=data.get("reply", ""),
                        raw_output=output
                    )
        except:
            pass

        return AgentResponse(
            tool_call=None,
            reply="すみません、理解できませんでした。",
            raw_output=output
        )

    def _process_twostage(self, user_input: str) -> AgentResponse:
        """2段階処理"""
        # Stage 1: 意図分類
        intent_prompt = f"""ユーザーの要求を分類してください。
カテゴリ: {', '.join(self.tools.keys())}, CHAT
カテゴリ名のみを出力。

ユーザー: {user_input}
カテゴリ:"""

        intent = self._call_ollama(intent_prompt).strip().lower()

        # ツール名を検出
        detected_tool = None
        for tool_name in self.tools.keys():
            if tool_name in intent:
                detected_tool = tool_name
                break

        if detected_tool:
            args = self._extract_args_for_tool(detected_tool, user_input)
            return AgentResponse(
                tool_call=ToolCall(name=detected_tool, args=args),
                reply=None,
                raw_output=f"Intent: {intent}"
            )

        return AgentResponse(
            tool_call=None,
            reply=self._generate_chat_reply(user_input),
            raw_output=f"Intent: {intent}"
        )

    def _process_hybrid(self, user_input: str) -> AgentResponse:
        """ハイブリッド（キーワード + 引数抽出）"""
        # キーワードベースのツール検出
        tool_keywords = {
            "list_images": ["画像一覧", "画像を見", "リスト", "ファイル", "フォルダの中"],
            "check_camera": ["カメラ", "撮影", "状態"],
            "analyze_frame": ["分析", "明るさ", "コントラスト"],
            "analyze_image": ["分析", "品質", "チェック", ".jpg", ".png"],
            "detect_objects": ["物体検出", "何が映", "検出"],
            "capture_frame": ["撮って", "キャプチャ", "写真"],
            "segment_image": ["セグメント", "領域", "道路検出"],
            "check_system_resources": ["メモリ", "リソース", "GPU"],
            "check_model_files": ["モデル", "ファイル確認"],
            "evaluate_quality": ["品質", "ブレ", "露出"],
        }

        detected_tool = None
        for tool_name, keywords in tool_keywords.items():
            if tool_name in self.tools:
                if any(kw in user_input for kw in keywords):
                    detected_tool = tool_name
                    break

        if detected_tool:
            args = self._extract_args_for_tool(detected_tool, user_input)
            return AgentResponse(
                tool_call=ToolCall(name=detected_tool, args=args),
                reply=None,
                raw_output=f"Keyword match: {detected_tool}"
            )

        return AgentResponse(
            tool_call=None,
            reply=self._generate_chat_reply(user_input),
            raw_output="No tool matched"
        )

    # -------------------------------------------------------------------------
    # ヘルパー
    # -------------------------------------------------------------------------

    def _extract_args_for_tool(self, tool: str, text: str) -> Dict[str, Any]:
        """ツールに応じた引数抽出"""
        if tool == "list_images":
            folder_match = re.search(r'(\w+)フォルダ', text)
            if folder_match:
                return {"folder": folder_match.group(1)}
            for common in ["data", "images", "output"]:
                if common in text.lower():
                    return {"folder": common}
            return {"folder": "data"}

        elif tool in ["analyze_image", "analyze_frame", "evaluate_quality", "segment_image"]:
            path_match = re.search(r'[\w/.-]+\.(?:jpg|jpeg|png)', text, re.IGNORECASE)
            if path_match:
                return {"image_path": path_match.group()}
            return {"image_path": ""}

        elif tool == "detect_objects":
            path_match = re.search(r'[\w/.-]+\.(?:jpg|jpeg|png)', text, re.IGNORECASE)
            if path_match:
                return {"image_path": path_match.group()}
            return {"image_path": ""}

        return {}

    def _generate_chat_reply(self, user_input: str) -> str:
        """通常の会話応答を生成"""
        prompt = f"""あなたはJetRacerロボットのアシスタントYANAです。
親切に短く回答してください。

ユーザー: {user_input}
YANA:"""
        return self._call_ollama(prompt)

    # -------------------------------------------------------------------------
    # ツール実行
    # -------------------------------------------------------------------------

    async def run(self, user_input: str) -> str:
        """ユーザー入力を処理し、必要ならツールを実行"""
        response = self.process(user_input)

        if response.tool_call:
            tool_name = response.tool_call.name
            args = response.tool_call.args

            if tool_name in self.tools:
                logger.info(f"ツール実行: {tool_name}({args})")
                print(f"  [Tool] {tool_name}({args})")
                try:
                    tool_func = self.tools[tool_name]["func"]
                    if asyncio.iscoroutinefunction(tool_func):
                        result = await tool_func(**args)
                    else:
                        result = tool_func(**args)

                    print(f"  [Result] {str(result)[:100]}...")
                    # 結果を自然言語で説明
                    return self._explain_result(tool_name, result)
                except Exception as e:
                    logger.error(f"ツール実行エラー: {e}")
                    return f"ツール実行中にエラーが発生しました: {e}"
            else:
                return f"不明なツール: {tool_name}"

        return response.reply or "すみません、よく分かりませんでした。"

    def _explain_result(self, tool_name: str, result: Any) -> str:
        """ツール実行結果を自然言語で説明"""
        result_str = json.dumps(result, ensure_ascii=False, indent=2) if isinstance(result, (dict, list)) else str(result)
        prompt = f"""以下のツール実行結果をユーザーに分かりやすく説明してください。

ツール: {tool_name}
結果: {result_str}

簡潔に説明:"""
        return self._call_ollama(prompt)
