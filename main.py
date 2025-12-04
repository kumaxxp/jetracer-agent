#!/usr/bin/env python3
"""
YANA - Your Autonomous Navigation Assistant
JetRacer 自律走行プロジェクト アシスタント

Usage:
    python main.py          # GUIモード（デフォルト）
    python main.py --cli    # CLIモード
    python main.py --port 8081  # ポート指定
"""

import sys
import argparse
import asyncio
from pathlib import Path

# プロジェクトルートをパスに追加
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description='YANA - Your Autonomous Navigation Assistant'
    )
    parser.add_argument(
        '--cli', 
        action='store_true',
        help='CLIモードで起動'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='GUIモードのポート番号（デフォルト: 8080）'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='GUIモードのホスト（デフォルト: 0.0.0.0）'
    )
    
    args = parser.parse_args()
    
    if args.cli:
        # CLIモード
        from agent.agent import main as cli_main
        asyncio.run(cli_main())
    else:
        # GUIモード
        from nicegui import ui
        from gui.app import create_app
        
        create_app()
        ui.run(
            title='YANA - JetRacer Control',
            host=args.host,
            port=args.port,
            reload=False
        )


if __name__ == "__main__":
    main()
