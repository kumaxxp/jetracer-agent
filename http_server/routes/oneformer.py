"""POST /oneformer/{camera_id} - OneFormerセグメンテーション"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import cv2
import base64
import time
import numpy as np
from PIL import Image
import io
import sys

# jetracer_annotation_tool のパスを追加
sys.path.insert(0, '/home/jetson/projects/jetracer_annotation_tool')

from ..core.camera_manager import camera_manager

router = APIRouter()

# OneFormerモデル（遅延ロード）
_segmenter = None


def get_segmenter():
    """OneFormerモデルを取得（遅延ロード）"""
    global _segmenter
    if _segmenter is None:
        print("[OneFormer] Loading model... (this may take a minute)")
        try:
            from core.ade20k_segmentation import ADE20KSegmenter
            _segmenter = ADE20KSegmenter()
            print("[OneFormer] Model loaded successfully")
        except Exception as e:
            print(f"[OneFormer] Failed to load model: {e}")
            raise
    return _segmenter


def create_overlay_image(original: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """セグメンテーションマスクからオーバーレイ画像を作成"""
    try:
        from data.ade20k_labels import get_label_color
    except ImportError:
        # フォールバック: 簡易カラーマップ
        def get_label_color(label_id):
            np.random.seed(label_id)
            return tuple(np.random.randint(0, 255, 3).tolist())
    
    height, width = seg_mask.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 各ラベルに色を割り当て
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        color = get_label_color(int(label_id))
        overlay[mask] = color
    
    # 元画像とブレンド
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # サイズが異なる場合はリサイズ
    if original_rgb.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (original_rgb.shape[1], original_rgb.shape[0]))
    
    blended = cv2.addWeighted(original_rgb, 1 - alpha, overlay, alpha, 0)
    return blended


@router.post("/oneformer/{camera_id}")
def run_oneformer(camera_id: int = 0):
    """OneFormerでセグメンテーションを実行"""
    start_time = time.time()
    
    # カメラからフレーム取得
    # TODO: 複数カメラ対応。現在はcamera_idは無視してメインカメラを使用
    frame = camera_manager.read()
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera not available")
    
    capture_time = time.time()
    
    try:
        # モデル取得
        segmenter = get_segmenter()
        
        # フレームを一時ファイルに保存（OneFormerはファイルパスを期待）
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            tmp_path = tmp.name
        
        # セグメンテーション実行
        seg_mask = segmenter.segment_image(tmp_path)
        
        # 一時ファイル削除
        import os
        os.unlink(tmp_path)
        
        seg_time = time.time()
        
        # 検出クラス情報
        unique_classes = np.unique(seg_mask)
        class_names = [segmenter.get_class_name(int(c)) for c in unique_classes]
        
        # オーバーレイ画像作成
        overlay = create_overlay_image(frame, seg_mask, alpha=0.5)
        
        # Base64エンコード
        # オーバーレイ画像
        overlay_pil = Image.fromarray(overlay)
        overlay_buffer = io.BytesIO()
        overlay_pil.save(overlay_buffer, format='PNG')
        overlay_base64 = base64.b64encode(overlay_buffer.getvalue()).decode('utf-8')
        
        # 元画像
        _, orig_buffer = cv2.imencode('.jpg', frame)
        original_base64 = base64.b64encode(orig_buffer).decode('utf-8')
        
        end_time = time.time()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "camera_id": camera_id,
            "overlay_base64": overlay_base64,
            "original_base64": original_base64,
            "classes": class_names,
            "num_classes": len(unique_classes),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "process_time_ms": (end_time - start_time) * 1000,
            "capture_time_ms": (capture_time - start_time) * 1000,
            "segmentation_time_ms": (seg_time - capture_time) * 1000
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OneFormer error: {str(e)}")
