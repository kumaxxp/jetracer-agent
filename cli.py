#!/usr/bin/env python3
"""YANA CLI モード"""

import sys
import asyncio
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.agent import main

if __name__ == "__main__":
    asyncio.run(main())
