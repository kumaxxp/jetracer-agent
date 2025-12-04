"""画像一覧取得ツール"""

from pathlib import Path
from typing import Any

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

async def list_images(folder: str) -> dict[str, Any]:
    """指定フォルダ内の画像ファイル一覧を取得"""
    folder_path = Path(folder)

    if not folder_path.exists():
        return {"error": f"Folder not found: {folder}"}

    if not folder_path.is_dir():
        return {"error": f"Not a directory: {folder}"}

    images = [
        str(f) for f in folder_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return {
        "folder": folder,
        "count": len(images),
        "images": sorted(images)
    }
