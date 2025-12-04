#!/usr/bin/env python3
"""JetRacer MCPエージェント"""

import json
import asyncio
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

from llama_cpp import Llama
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

MODEL_PATH = Path.home() / "projects/jetracer-agent/models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
MCP_SERVER_PATH = Path.home() / "projects/jetracer-agent/mcp_server/server.py"

@dataclass
class Message:
    role: str
    content: Optional[str] = None
    tool_calls: Optional[list] = None
    tool_call_id: Optional[str] = None

@dataclass
class Agent:
    llm: Llama = field(init=False)
    mcp_session: Optional[ClientSession] = None
    messages: list[dict] = field(default_factory=list)
    tools: list[dict] = field(default_factory=list)

    def __post_init__(self):
        print("Loading LLM...")
        self.llm = Llama(
            model_path=str(MODEL_PATH),
            n_gpu_layers=-1,
            n_ctx=4096,
            verbose=False,
        )
        print("LLM loaded.")

        self.messages = [
            {
                "role": "system",
                "content": (
                    "あなたはJetRacerの画像処理アシスタントです。"
                    "ユーザーの指示に従い、利用可能なツールを使って画像の解析やアノテーションを行います。"
                    "ツールの実行結果は日本語で分かりやすく説明してください。"
                )
            }
        ]

    async def connect_mcp(self):
        """MCPサーバーに接続"""
        server_params = StdioServerParameters(
            command="python3",
            args=[str(MCP_SERVER_PATH)],
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

    async def chat(self, user_input: str) -> str:
        """ユーザー入力を処理して応答を返す"""
        self.messages.append({"role": "user", "content": user_input})

        while True:
            response = self.llm.create_chat_completion(
                messages=self.messages,
                tools=self.tools if self.tools else None,
                tool_choice="auto" if self.tools else None,
                max_tokens=1024,
            )

            message = response["choices"][0]["message"]

            # ツール呼び出しがある場合
            if "tool_calls" in message and message["tool_calls"]:
                self.messages.append(message)

                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = json.loads(tool_call["function"]["arguments"])

                    print(f"  [Tool] {func_name}({func_args})")
                    result = await self._execute_tool(func_name, func_args)
                    print(f"  [Result] {result[:100]}...")

                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })
            else:
                # 通常の応答
                content = message.get("content", "")
                self.messages.append({"role": "assistant", "content": content})
                return content

async def main():
    agent = Agent()

    try:
        await agent.connect_mcp()

        print("\nJetRacer Agent Ready")
        print("Type 'quit' to exit\n")

        while True:
            try:
                user_input = input("You: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    continue

                response = await agent.chat(user_input)
                print(f"Agent: {response}\n")

            except KeyboardInterrupt:
                break

    finally:
        await agent.disconnect_mcp()
        print("Goodbye!")

if __name__ == "__main__":
    asyncio.run(main())
