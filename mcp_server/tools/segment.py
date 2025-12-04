"""セグメンテーションツール（スタブ実装）"""

from pathlib import Path
from typing import Any, Optional

# 将来的にONNXモデルをロード
# import onnxruntime as ort

async def segment_image(image_path: str, output_path: Optional[str] = None) -> dict[str, Any]:
    """画像をセグメンテーション処理する（スタブ）"""
    path = Path(image_path)

    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    if output_path is None:
        output_path = str(path.parent / f"{path.stem}_seg{path.suffix}")

    # TODO: 実際のセグメンテーション処理を実装
    # 現在はスタブとして入力画像情報を返す

    return {
        "status": "stub",
        "message": "Segmentation not yet implemented",
        "input": image_path,
        "output": output_path
    }
