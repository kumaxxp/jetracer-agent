"""画像品質評価ツール"""

from pathlib import Path
from typing import Any
import cv2
import numpy as np

async def evaluate_quality(image_path: str) -> dict[str, Any]:
    """画像の品質（ブレ、露出）を評価"""
    path = Path(image_path)

    if not path.exists():
        return {"error": f"File not found: {image_path}"}

    try:
        img = cv2.imread(str(path))
        if img is None:
            return {"error": "Failed to load image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ブレ検出（Laplacianの分散）
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 100  # 閾値は調整可能

        # 露出評価
        mean_brightness = np.mean(gray)
        is_underexposed = mean_brightness < 50
        is_overexposed = mean_brightness > 200

        return {
            "path": image_path,
            "blur_score": round(laplacian_var, 2),
            "is_blurry": is_blurry,
            "brightness": round(mean_brightness, 2),
            "is_underexposed": is_underexposed,
            "is_overexposed": is_overexposed,
            "quality_ok": not (is_blurry or is_underexposed or is_overexposed)
        }

    except Exception as e:
        return {"error": f"Quality evaluation failed: {e}"}
