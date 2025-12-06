"""POST /oneformer/{camera_id} - OneFormerセグメンテーション"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import cv2
import base64
import time
import numpy as np
from PIL import Image
import io
from typing import Optional, Dict, Set

from ..core.camera_manager import camera_manager
from ..core.ade20k_labels import get_label_color, get_label_name, ADE20K_LABELS
from ..core.road_mapping import get_road_mapping

router = APIRouter()

# OneFormerモデル（遅延ロード）
_segmenter = None

# 最新のセグメンテーション結果を保持（クリック位置→ラベル変換用）
_latest_seg_masks: Dict[int, np.ndarray] = {}  # camera_id -> seg_mask
_latest_seg_sizes: Dict[int, tuple] = {}  # camera_id -> (height, width)


class ClickRequest(BaseModel):
    """Request body for get_label_at_position."""
    x: float  # 0.0 - 1.0 (relative position)
    y: float  # 0.0 - 1.0 (relative position)


def get_segmenter():
    """OneFormerモデルを取得（遅延ロード）"""
    global _segmenter
    if _segmenter is None:
        print("[OneFormer] Loading model... (this may take a minute)")
        try:
            from ..core.ade20k_segmentation import ADE20KSegmenter
            _segmenter = ADE20KSegmenter()
            print("[OneFormer] Model loaded successfully")
        except Exception as e:
            print(f"[OneFormer] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            raise
    return _segmenter


def create_stripe_pattern(height: int, width: int, stripe_width: int = 10) -> np.ndarray:
    """斜め線パターンを作成（NumPy高速化版）
    
    Args:
        height: 画像の高さ
        width: 画像の幅
        stripe_width: ストライプの幅（ピクセル）
        
    Returns:
        斜め線パターンのブールマスク
    """
    # メッシュグリッドで座標を生成
    y_coords, x_coords = np.ogrid[:height, :width]
    # 斜め線パターン: (y + x) % (stripe_width * 2) < stripe_width
    pattern = ((y_coords + x_coords) % (stripe_width * 2)) < stripe_width
    return pattern


def create_overlay_image(
    original: np.ndarray, 
    seg_mask: np.ndarray, 
    alpha: float = 0.5,
    highlight_road: bool = False,
    road_label_ids: Optional[Set[int]] = None
) -> np.ndarray:
    """セグメンテーションマスクからオーバーレイ画像を作成
    
    Args:
        original: 元画像 (BGR)
        seg_mask: セグメンテーションマスク
        alpha: ブレンド率
        highlight_road: ROADラベルをハイライト表示
        road_label_ids: ROADとしてマークされたラベルIDのセット
    """
    height, width = seg_mask.shape
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 各ラベルに色を割り当て
    for label_id in np.unique(seg_mask):
        mask = seg_mask == label_id
        color = get_label_color(int(label_id))
        overlay[mask] = color
    
    # ROADハイライト（斜め線パターン）
    if highlight_road and road_label_ids:
        # 斜め線パターンを作成
        stripe_pattern = create_stripe_pattern(height, width, stripe_width=10)
        
        # ROADラベルの領域に黄色の斜め線を適用
        for label_id in road_label_ids:
            mask = seg_mask == label_id
            road_stripe_mask = mask & stripe_pattern
            # 黄色でストライプを描画
            overlay[road_stripe_mask] = [255, 255, 0]  # RGB: Yellow
    
    # 元画像をRGBに変換
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # サイズが異なる場合はリサイズ
    if original_rgb.shape[:2] != overlay.shape[:2]:
        overlay = cv2.resize(overlay, (original_rgb.shape[1], original_rgb.shape[0]))
    
    # ブレンド
    blended = cv2.addWeighted(original_rgb, 1 - alpha, overlay, alpha, 0)
    return blended


def get_road_label_ids() -> Set[int]:
    """ROADマッピングからラベルIDのセットを取得"""
    road_mapping = get_road_mapping()
    road_label_names = road_mapping.get_road_labels()
    
    road_ids = set()
    for label_name in road_label_names:
        # ラベル名からIDを逆引き
        for lid, lname in ADE20K_LABELS.items():
            if lname == label_name:
                road_ids.add(lid)
                break
    
    return road_ids


@router.post("/oneformer/{camera_id}")
def run_oneformer(camera_id: int = 0, highlight_road: bool = False):
    """OneFormerでセグメンテーションを実行
    
    Args:
        camera_id: カメラID
        highlight_road: ROADラベルを黄色斜め線でハイライト表示
    """
    global _latest_seg_masks, _latest_seg_sizes
    
    start_time = time.time()
    
    # カメラからフレーム取得
    frame = camera_manager.read(camera_id)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
    
    capture_time = time.time()
    
    try:
        # モデル取得
        segmenter = get_segmenter()
        
        # フレームを一時ファイルに保存（OneFormerはファイルパスを期待）
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            tmp_path = tmp.name
        
        # セグメンテーション実行
        seg_mask = segmenter.segment_image(tmp_path)
        
        # 一時ファイル削除
        os.unlink(tmp_path)
        
        # マスクを保存（クリック位置→ラベル変換用）
        _latest_seg_masks[camera_id] = seg_mask
        _latest_seg_sizes[camera_id] = (seg_mask.shape[0], seg_mask.shape[1])
        
        seg_time = time.time()
        
        # 検出クラス情報
        unique_classes = np.unique(seg_mask)
        class_names = [segmenter.get_class_name(int(c)) for c in unique_classes]
        
        # ROADマッピングを取得して各クラスの状態を追加
        road_mapping = get_road_mapping()
        road_label_ids = get_road_label_ids() if highlight_road else set()
        
        class_info = []
        for class_id in unique_classes:
            name = segmenter.get_class_name(int(class_id))
            class_info.append({
                "id": int(class_id),
                "name": name,
                "is_road": road_mapping.is_road(name)
            })
        
        # オーバーレイ画像作成（ROADハイライト付き）
        overlay = create_overlay_image(
            frame, seg_mask, alpha=0.5, 
            highlight_road=highlight_road,
            road_label_ids=road_label_ids
        )
        
        # Base64エンコード
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
            "class_info": class_info,
            "num_classes": len(unique_classes),
            "width": frame.shape[1],
            "height": frame.shape[0],
            "road_labels": road_mapping.get_road_labels(),
            "highlight_road": highlight_road,
            "process_time_ms": (end_time - start_time) * 1000,
            "capture_time_ms": (capture_time - start_time) * 1000,
            "segmentation_time_ms": (seg_time - capture_time) * 1000
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OneFormer error: {str(e)}")


@router.post("/oneformer/{camera_id}/label-at-position")
def get_label_at_position(camera_id: int, request: ClickRequest):
    """クリック位置のラベルを取得
    
    Args:
        camera_id: カメラID
        request: クリック位置 (x, y は 0.0-1.0 の相対座標)
    """
    if camera_id not in _latest_seg_masks:
        raise HTTPException(
            status_code=404, 
            detail=f"No segmentation result for camera {camera_id}. Run /oneformer/{camera_id} first."
        )
    
    seg_mask = _latest_seg_masks[camera_id]
    height, width = seg_mask.shape
    
    # 相対座標を絶対座標に変換
    x = int(request.x * width)
    y = int(request.y * height)
    
    # 範囲チェック
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    
    # ラベルID取得
    label_id = int(seg_mask[y, x])
    label_name = get_label_name(label_id)
    
    # ROADかどうか確認
    road_mapping = get_road_mapping()
    is_road = road_mapping.is_road(label_name)
    
    return {
        "x": request.x,
        "y": request.y,
        "pixel_x": x,
        "pixel_y": y,
        "label_id": label_id,
        "label_name": label_name,
        "is_road": is_road,
        "color": get_label_color(label_id)
    }


@router.post("/oneformer/{camera_id}/toggle-road-at-position")
def toggle_road_at_position(camera_id: int, request: ClickRequest):
    """クリック位置のラベルのROAD状態をトグル
    
    Args:
        camera_id: カメラID
        request: クリック位置 (x, y は 0.0-1.0 の相対座標)
    """
    # まずラベルを取得
    label_info = get_label_at_position(camera_id, request)
    label_name = label_info["label_name"]
    
    # ROADをトグル
    road_mapping = get_road_mapping()
    new_state = road_mapping.toggle_road(label_name)
    
    return {
        "label_name": label_name,
        "label_id": label_info["label_id"],
        "is_road": new_state,
        "road_labels": road_mapping.get_road_labels()
    }
