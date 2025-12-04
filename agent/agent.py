#!/usr/bin/env python3
"""YANA - Your Autonomous Navigation Assistant"""

import json
import asyncio
from pathlib import Path
from typing import Any, Optional, AsyncGenerator
from dataclasses import dataclass, field

from llama_cpp import Llama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# YANA設定
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from yana.config import LLM_MODEL_PATH, MCP_SERVER_PATH, LLM_N_CTX, LLM_MAX_TOKENS
from yana.prompts import SYSTEM_PROMPT, build_startup_prompt
from yana.session import SessionManager, Event

# 代替パス（環境によって変わる場合）
MODEL_PATH = LLM_MODEL_PATH
if not MODEL_PATH.exists():
    MODEL_PATH = Path.home() / "projects/jetracer-agent/models/qwen2.5-3b-instruct-q4_k_m.gguf"

MCP_PATH = MCP_SERVER_PATH
if not MCP_PATH.exists():
    MCP_PATH = Path.home() / "projects/jetracer-agent/mcp_server/server.py"


@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None


@dataclass
class YANAAgent:
    """YANA エージェント"""
    llm: Llama = field(init=False)
    mcp_session: Optional[ClientSession] = None
    messages: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)
    session_manager: SessionManager = field(default_factory=SessionManager)
    
    # コールバック
    on_message: Optional[callable] = None  # メッセージ出力時のコールバック
    on_tool_call: Optional[callable] = None  # ツール呼び出し時のコールバック

    def __post_init__(self):
        print("Loading LLM...")
        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=-1,
            n_ctx=LLM_N_CTX,
            verbose=False,
            chat_format="chatml",
        )
        print("LLM loaded.")

        # YANAシステムプロンプト
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    async def connect_mcp(self):
        """MCPサーバーに接続"""
        server_params = StdioServerParameters(
            command="python3",
            args=[str(MCP_PATH)],
        )

        self._stdio_client = stdio_client(server_params)
        self._read, self._write = await self._stdio_client.__aenter__()
        self.mcp_session = ClientSession(self._read, self._write)
        await self.mcp_session.__aenter__()
        await self.mcp_session.initialize()

        # ツール定義を取得
        tools_result = await self.mcp_session.list_tools()
        self.tools = self._convert_tools(tools_result.tools)
        print(f"Connected to MCP server. {len(self.tools)} tools available.")

    async def disconnect_mcp(self):
        """MCPサーバーから切断"""
        if self.mcp_session:
            await self.mcp_session.__aexit__(None, None, None)
        if hasattr(self, '_stdio_client'):
            await self._stdio_client.__aexit__(None, None, None)

    def _convert_tools(self, mcp_tools) -> list[dict]:
        """MCPツール定義をOpenAI形式に変換"""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                }
            }
            for tool in mcp_tools
        ]

    async def _execute_tool(self, name: str, arguments: dict) -> str:
        """MCPツールを実行"""
        result = await self.mcp_session.call_tool(name, arguments)
        return result.content[0].text

    async def startup(self) -> AsyncGenerator[str, None]:
        """起動シーケンスを実行"""
        # 前回のセッション確認
        if self.session_manager.is_resumable():
            context = self.session_manager.get_context_for_yana()
            recent_events = self.session_manager.get_recent_events(5)
            startup_prompt = build_startup_prompt(context, recent_events)
        else:
            startup_prompt = build_startup_prompt()
        
        # 起動プロンプトでエージェントループ実行
        async for chunk in self.run_with_tools(startup_prompt, is_startup=True):
            yield chunk

    async def run_with_tools(self, user_input: str, is_startup: bool = False) -> AsyncGenerator[str, None]:
        """ツール呼び出しを含むエージェントループ（ストリーミング）"""
        
        if not is_startup:
            # 通常のユーザー入力はセッションコンテキストを付加
            context = self.session_manager.get_context_for_yana()
            if context and self.session_manager.state.phase.value != "idle":
                full_input = f"現在の状態:\n{context}\n\nユーザーの指示:\n{user_input}"
            else:
                full_input = user_input
            
            self.messages.append({"role": "user", "content": full_input})
            
            # イベント記録
            self.session_manager.record_event("yana", "user_input", {"input": user_input})
        else:
            self.messages.append({"role": "user", "content": user_input})

        while True:
            response = self.llm.create_chat_completion(
                messages=self.messages,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
                max_tokens=LLM_MAX_TOKENS,
            )

            message = response["choices"][0]["message"]

            # ツール呼び出しがある場合
            if "tool_calls" in message and message["tool_calls"]:
                self.messages.append(message)

                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])

                    # コールバック通知
                    if self.on_tool_call:
                        self.on_tool_call(func_name, func_args)
                    
                    print(f"  [Tool] {func_name}({func_args})")
                    result = await self._execute_tool(func_name, func_args)
                    print(f"  [Result] {result[:100]}...")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
                    
                    # イベント記録
                    self.session_manager.record_event("yana", "tool_call", {
                        "tool": func_name,
                        "args": func_args,
                        "result_preview": result[:200]
                    })
                
                # 次のイテレーションへ
                continue
            
            else:
                # 通常の応答
                content = message.get("content", "")
                self.messages.append({"role": "assistant", "content": content})
                
                # コールバック通知
                if self.on_message:
                    self.on_message(content)
                
                yield content
                break

    async def chat(self, user_input: str) -> str:
        """ユーザー入力を処理して応答を返す（非ストリーミング）"""
        result = ""
        async for chunk in self.run_with_tools(user_input):
            result += chunk
        return result

    async def process_notification(self, notification: str) -> Optional[str]:
        """GUIからの通知を処理（必要に応じて応答）"""
        # 通知は会話履歴に追加するが、必ずしも応答しない
        self.messages.append({"role": "user", "content": notification})
        
        response = self.llm.create_chat_completion(
            messages=self.messages,
            max_tokens=256,  # 通知への応答は短めに
        )
        
        content = response["choices"][0]["message"].get("content", "")
        
        # 空でない応答があれば返す
        if content.strip():
            self.messages.append({"role": "assistant", "content": content})
            return content
        
        return None

    def receive_gui_event(self, event: Event):
        """GUIイベントを受信"""
        # イベントを自然言語で説明
        descriptions = {
            "capture_started": "撮影が開始されました",
            "frame_captured": f"フレームが撮影されました",
            "image_selected": f"画像が選択されました",
            "image_deleted": f"画像が削除されました",
            "annotation_updated": "アノテーションが更新されました",
            "road_mapping_changed": "ROADマッピングが変更されました",
            "directory_changed": f"作業ディレクトリが変更されました",
        }
        
        description = descriptions.get(event.action, f"{event.action}が実行されました")
        
        # 詳細を追加
        if event.details:
            if "path" in event.details:
                description += f": {Path(event.details['path']).name}"
            if "directory" in event.details:
                description += f": {event.details['directory']}"
            if "number" in event.details:
                description += f" ({event.details['number']}枚目)"
        
        notification = f"[システム通知] {description}"
        return notification


async def main():
    """CLIモード"""
    agent = YANAAgent()

    try:
        await agent.connect_mcp()

        print("\n" + "="*50)
        print("YANA - Your Autonomous Navigation Assistant")
        print("="*50)
        
        # 起動シーケンス
        print("\n--- 起動シーケンス ---")
        async for chunk in agent.startup():
            print(chunk)
        print("--- 起動完了 ---\n")

        print("Type 'quit' to exit, 'reset' to reset session\n")

        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if user_input.lower() == "reset":
                    agent.session_manager.reset()
                    print("セッションをリセットしました。")
                    continue
                if not user_input:
                    continue

                response = await agent.chat(user_input)
                print(f"YANA: {response}\n")

            except KeyboardInterrupt:
                break

    finally:
        await agent.disconnect_mcp()
        print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())
