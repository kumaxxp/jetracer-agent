"""画像情報取得ツール"""

from pathlib import Path
from typing import Any
from PIL import Image

async def get_image_info(image_path: str) -> dict[str, Any]:
    """画像のメタ情報を取得"""
    path = Path(image_path)

    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    try:
        with Image.open(path) as img:
            return {
                "path": image_path,
                "format": img.format,
                "mode": img.mode,
                "width": img.width,
                "height": img.height,
                "size_bytes": path.stat().st_size
            }
    except Exception as e:
        return {"error": f"Failed to read image: {e}"}
