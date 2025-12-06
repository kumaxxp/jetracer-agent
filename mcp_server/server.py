#!/usr/bin/env python3
"""JetRacer MCP Server - YANA対応版"""

import asyncio
import json
from pathlib import Path
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types

# 既存ツール
from tools.image_list import list_images
from tools.image_info import get_image_info
from tools.segment import segment_image
from tools.quality import evaluate_quality

# 新規ツール（YANA用）
from tools.system import (
    check_camera,
    analyze_frame,
    check_system_resources,
    check_model_files
)
from tools.detection import detect_objects
from tools.calibration import (
    get_calibration_status,
    detect_checkerboard,
    capture_calibration_pair,
    run_calibration,
    get_calibration_instruction,
    clear_calibration_images
)

server = Server("jetracer-tools")

TOOL_DEFINITIONS = [
    # === 既存ツール ===
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
    
    # === 新規ツール（YANA用）===
    types.Tool(
        name="check_camera",
        description="カメラの接続状態を確認し、1フレーム取得して一時保存する",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    types.Tool(
        name="analyze_frame",
        description="画像の明るさ、コントラスト、エッジ量を分析する。レンズキャップや照明問題の検出に使用",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "画像ファイルのパス"}
            },
            "required": ["image_path"]
        }
    ),
    types.Tool(
        name="detect_objects",
        description="YOLOv8で画像内の物体を検出する。カメラに何が映っているかの確認に使用",
        inputSchema={
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "画像ファイルのパス"},
                "confidence": {"type": "number", "description": "信頼度閾値（デフォルト0.5）"}
            },
            "required": ["image_path"]
        }
    ),
    types.Tool(
        name="check_system_resources",
        description="メモリとGPU使用状況を確認する",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    types.Tool(
        name="check_model_files",
        description="必要なモデルファイル（LLM、セグメンテーション、YOLO）の存在を確認する",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    
    # === キャリブレーションツール ===
    types.Tool(
        name="get_calibration_status",
        description="カメラキャリブレーションの状態を取得する。撮影済み画像数、キャリブレーション済みかどうか、RMSエラー等を確認",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    types.Tool(
        name="detect_checkerboard",
        description="指定カメラでチェッカーボードを検出する。キャリブレーション撮影前の確認に使用",
        inputSchema={
            "type": "object",
            "properties": {
                "camera_id": {"type": "integer", "description": "カメラID (0 or 1)"}
            },
            "required": ["camera_id"]
        }
    ),
    types.Tool(
        name="capture_calibration_pair",
        description="両カメラで同時にキャリブレーション用画像を撮影する。20ペア程度必要",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    types.Tool(
        name="run_calibration",
        description="収集した画像でキャリブレーションを実行する。10枚以上の画像が必要",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    types.Tool(
        name="get_calibration_instruction",
        description="次のキャリブレーション撮影の指示を取得する。チェッカーボードをどこに配置すべきかを指示",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
    types.Tool(
        name="clear_calibration_images",
        description="収集したキャリブレーション画像をすべてクリアする",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": []
        }
    ),
]

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    return TOOL_DEFINITIONS

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        # 既存ツール
        if name == "list_images":
            result = await list_images(arguments["folder"])
        elif name == "get_image_info":
            result = await get_image_info(arguments["image_path"])
        elif name == "segment_image":
            output_path = arguments.get("output_path")
            result = await segment_image(arguments["image_path"], output_path)
        elif name == "evaluate_quality":
            result = await evaluate_quality(arguments["image_path"])
        
        # 新規ツール（YANA用）
        elif name == "check_camera":
            result = await check_camera()
        elif name == "analyze_frame":
            result = await analyze_frame(arguments["image_path"])
        elif name == "detect_objects":
            confidence = arguments.get("confidence", 0.5)
            result = await detect_objects(arguments["image_path"], confidence)
        elif name == "check_system_resources":
            result = await check_system_resources()
        elif name == "check_model_files":
            result = await check_model_files()
        
        # キャリブレーションツール
        elif name == "get_calibration_status":
            result = await get_calibration_status()
        elif name == "detect_checkerboard":
            result = await detect_checkerboard(arguments["camera_id"])
        elif name == "capture_calibration_pair":
            result = await capture_calibration_pair()
        elif name == "run_calibration":
            result = await run_calibration()
        elif name == "get_calibration_instruction":
            result = await get_calibration_instruction()
        elif name == "clear_calibration_images":
            result = await clear_calibration_images()
        
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
