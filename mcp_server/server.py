#!/usr/bin/env python3
"""JetRacer MCP Server"""

import asyncio
import json
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

from tools.image_list import list_images
from tools.image_info import get_image_info
from tools.segment import segment_image
from tools.quality import evaluate_quality

server = Server("jetracer-tools")

TOOL_DEFINITIONS = [
    types.Tool(
        name="list_images",
        description="指定フォルダ内の画像ファイル一覧を取得する",
        inputSchema={
            "type": "object",
            "properties": {
                "folder": {"type": "string", "description": "画像フォルダのパス"}
            },
            "required": ["folder"]
        }
    ),
    types.Tool(
        name="get_image_info",
        description="画像のメタ情報（サイズ、形式等）を取得する",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "画像ファイルのパス"}
            },
            "required": ["image_path"]
        }
    ),
    types.Tool(
        name="segment_image",
        description="画像をセグメンテーション処理する",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "画像ファイルのパス"},
                "output_path": {"type": "string", "description": "出力パス（省略時は自動生成）"}
            },
            "required": ["image_path"]
        }
    ),
    types.Tool(
        name="evaluate_quality",
        description="画像の品質（ブレ、露出）を評価する",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "画像ファイルのパス"}
            },
            "required": ["image_path"]
        }
    ),
]

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return TOOL_DEFINITIONS

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        if name == "list_images":
            result = await list_images(arguments["folder"])
        elif name == "get_image_info":
            result = await get_image_info(arguments["image_path"])
        elif name == "segment_image":
            output_path = arguments.get("output_path")
            result = await segment_image(arguments["image_path"], output_path)
        elif name == "evaluate_quality":
            result = await evaluate_quality(arguments["image_path"])
        else:
            result = {"error": f"Unknown tool: {name}"}

        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))]

    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}))]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
