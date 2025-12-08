"""距離グリッドAPI"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import base64
import cv2
import numpy as np

from ..core.distance_grid import distance_grid_manager
from ..core.camera_manager import camera_manager


router = APIRouter(prefix="/distance-grid", tags=["distance_grid"])


class GridConfigUpdate(BaseModel):
    """グリッド設定更新リクエスト"""
    camera_height_mm: Optional[float] = None
    camera_pitch_deg: Optional[float] = None
    camera_roll_deg: Optional[float] = None
    grid_depth_min_m: Optional[float] = None
    grid_depth_max_m: Optional[float] = None
    grid_width_m: Optional[float] = None
    grid_depth_lines: Optional[int] = None
    grid_width_lines: Optional[int] = None
    vanishing_point_y: Optional[float] = None
    perspective_strength: Optional[float] = None


class DistanceQueryRequest(BaseModel):
    """距離クエリリクエスト"""
    pixel_x: int
    pixel_y: int


@router.get("/{camera_id}/status")
async def get_grid_status(camera_id: int):
    """グリッドステータスを取得"""
    return distance_grid_manager.get_status(camera_id)


@router.get("/{camera_id}/config")
async def get_grid_config(camera_id: int):
    """グリッド設定を取得"""
    from dataclasses import asdict
    config = distance_grid_manager.get_config(camera_id)
    return asdict(config)


@router.put("/{camera_id}/config")
async def update_grid_config(camera_id: int, update: GridConfigUpdate):
    """グリッド設定を更新"""
    from dataclasses import asdict
    
    # None以外のフィールドのみ更新
    update_dict = {k: v for k, v in update.dict().items() if v is not None}
    
    if not update_dict:
        return {"message": "No changes", "config": asdict(distance_grid_manager.get_config(camera_id))}
    
    config = distance_grid_manager.update_config(camera_id, **update_dict)
    return {"message": "Updated", "config": asdict(config)}


@router.get("/{camera_id}/grid-lines")
async def get_grid_lines(camera_id: int):
    """グリッド線データを取得"""
    grid_data = distance_grid_manager.compute_grid_lines(camera_id)
    return grid_data


@router.get("/{camera_id}/preview")
async def get_grid_preview(camera_id: int, undistort: bool = False):
    """グリッドプレビュー画像を取得
    
    Args:
        camera_id: カメラID
        undistort: 歪み補正を適用するか
    
    Returns:
        Base64エンコードされたプレビュー画像
    """
    # カメラからフレームを取得（歪み補正はcamera_managerの設定に依存）
    frame = camera_manager.read(camera_id, apply_undistort=undistort)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
    
    # グリッドオーバーレイを描画
    result = distance_grid_manager.draw_grid_overlay(
        camera_id, frame,
        color=(0, 255, 0),
        thickness=1,
        show_labels=True
    )
    
    # JPEG圧縮
    _, buffer = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    config = distance_grid_manager.get_config(camera_id)
    
    return {
        "image_base64": img_base64,
        "camera_id": camera_id,
        "undistorted": undistort,
        "grid_config": {
            "camera_height_mm": config.camera_height_mm,
            "camera_pitch_deg": config.camera_pitch_deg,
            "grid_depth_range_m": [config.grid_depth_min_m, config.grid_depth_max_m],
            "grid_width_m": config.grid_width_m
        }
    }


@router.get("/{camera_id}/overlay-on-segmentation")
async def get_grid_overlay_on_segmentation(
    camera_id: int,
    highlight_road: bool = True,
    undistort: bool = False
):
    """セグメンテーション結果にグリッドオーバーレイを合成
    
    Args:
        camera_id: カメラID
        highlight_road: ROAD領域をハイライトするか
        undistort: 歪み補正を適用するか
    
    Returns:
        セグメンテーション + グリッドの合成画像
    """
    from ..core.ade20k_segmentation import ade20k_segmentor
    from ..core.road_mapping import road_mapping_manager
    
    # カメラからフレームを取得
    frame = camera_manager.read(camera_id, apply_undistort=undistort)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
    
    # セグメンテーション実行
    seg_result = ade20k_segmentor.segment(frame)
    if seg_result is None:
        raise HTTPException(status_code=500, detail="Segmentation failed")
    
    # オーバーレイ画像を生成
    overlay = ade20k_segmentor.create_overlay(frame, seg_result["segmentation"])
    
    # ROADハイライト（オプション）
    if highlight_road:
        road_labels = road_mapping_manager.get_road_labels()
        if road_labels:
            overlay = ade20k_segmentor.highlight_road_regions(
                overlay, 
                seg_result["segmentation"],
                road_labels
            )
    
    # グリッドオーバーレイを追加（緑色）
    grid_overlay = distance_grid_manager.draw_grid_overlay(
        camera_id, overlay,
        color=(0, 255, 0),
        thickness=1,
        show_labels=True
    )
    
    # JPEG圧縮
    _, buffer = cv2.imencode('.jpg', grid_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "image_base64": img_base64,
        "camera_id": camera_id,
        "segmentation_classes": seg_result.get("detected_classes", []),
        "num_classes": seg_result.get("num_classes", 0),
        "road_labels": road_mapping_manager.get_road_labels() if highlight_road else []
    }


@router.post("/{camera_id}/distance-at-point")
async def get_distance_at_point(camera_id: int, request: DistanceQueryRequest):
    """ピクセル座標から推定距離を取得"""
    result = distance_grid_manager.get_distance_at_point(
        camera_id, request.pixel_x, request.pixel_y
    )
    
    if result is None:
        return {
            "error": "Could not compute distance",
            "pixel_x": request.pixel_x,
            "pixel_y": request.pixel_y
        }
    
    return {
        "pixel_x": request.pixel_x,
        "pixel_y": request.pixel_y,
        "distance_m": result["distance_m"],
        "lateral_offset_m": result["lateral_offset_m"]
    }


@router.get("/{camera_id}/analyze-segmentation")
async def analyze_segmentation_with_grid(camera_id: int, undistort: bool = False):
    """セグメンテーション結果をグリッドで分析
    
    各グリッドセル内のROAD領域の割合を計算
    
    Returns:
        グリッドセルごとのROAD割合と距離情報
    """
    from ..core.ade20k_segmentation import ade20k_segmentor
    from ..core.road_mapping import road_mapping_manager
    
    # カメラからフレームを取得
    frame = camera_manager.read(camera_id, apply_undistort=undistort)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
    
    # セグメンテーション実行
    seg_result = ade20k_segmentor.segment(frame)
    if seg_result is None:
        raise HTTPException(status_code=500, detail="Segmentation failed")
    
    segmentation = seg_result["segmentation"]
    
    # ROADラベルを取得
    road_labels = road_mapping_manager.get_road_labels()
    road_label_ids = set()
    
    # ラベル名からIDへの変換
    for label_name in road_labels:
        label_id = ade20k_segmentor.get_label_id(label_name)
        if label_id is not None:
            road_label_ids.add(label_id)
    
    # グリッドデータを取得
    config = distance_grid_manager.get_config(camera_id)
    grid_data = distance_grid_manager.compute_grid_lines(camera_id)
    
    # 各奥行きラインでROAD割合を計算
    depth_analysis = []
    
    h, w = segmentation.shape
    
    for line_data in grid_data["horizontal_lines"]:
        depth_m = line_data["depth_m"]
        points = line_data["points"]
        
        if len(points) < 2:
            continue
        
        # この奥行きラインのY座標（平均）
        y_pixels = [int(p[1]) for p in points]
        y_avg = np.mean(y_pixels)
        
        if 0 <= y_avg < h:
            # この行でのROAD割合を計算
            y_idx = int(y_avg)
            row = segmentation[y_idx, :]
            road_pixels = sum(1 for p in row if p in road_label_ids)
            road_ratio = road_pixels / len(row)
            
            depth_analysis.append({
                "depth_m": depth_m,
                "road_ratio": float(road_ratio),
                "pixel_y": y_idx
            })
    
    # 左右のROAD分布を分析
    lateral_analysis = []
    for line_data in grid_data["vertical_lines"]:
        offset_m = line_data["offset_m"]
        points = line_data["points"]
        
        if len(points) < 2:
            continue
        
        # このラインのX座標（平均）
        x_pixels = [int(p[0]) for p in points]
        x_avg = np.mean(x_pixels)
        
        if 0 <= x_avg < w:
            x_idx = int(x_avg)
            col = segmentation[:, x_idx]
            road_pixels = sum(1 for p in col if p in road_label_ids)
            road_ratio = road_pixels / len(col)
            
            lateral_analysis.append({
                "offset_m": offset_m,
                "road_ratio": float(road_ratio),
                "pixel_x": x_idx
            })
    
    # 走行可能な方向を推定
    navigation_hint = _compute_navigation_hint(depth_analysis, lateral_analysis)
    
    return {
        "camera_id": camera_id,
        "depth_analysis": depth_analysis,
        "lateral_analysis": lateral_analysis,
        "navigation_hint": navigation_hint,
        "road_labels": list(road_labels),
        "grid_config": {
            "depth_range_m": [config.grid_depth_min_m, config.grid_depth_max_m],
            "width_m": config.grid_width_m
        }
    }


def _compute_navigation_hint(depth_analysis: list, lateral_analysis: list) -> Dict[str, Any]:
    """走行可能な方向のヒントを計算"""
    hint = {
        "forward_clear": False,
        "max_clear_distance_m": 0.0,
        "recommended_direction": "stop",
        "confidence": 0.0
    }
    
    if not depth_analysis:
        return hint
    
    # 近距離から遠距離へチェック
    sorted_depth = sorted(depth_analysis, key=lambda x: x["depth_m"])
    
    # 前方の道路状況を評価
    road_threshold = 0.3  # ROAD比率がこれ以上なら走行可能と判断
    
    max_clear_distance = 0.0
    for d in sorted_depth:
        if d["road_ratio"] >= road_threshold:
            max_clear_distance = d["depth_m"]
        else:
            break
    
    hint["max_clear_distance_m"] = max_clear_distance
    hint["forward_clear"] = max_clear_distance >= 0.5  # 0.5m以上開いていれば前進可能
    
    # 左右の道路状況を評価
    if lateral_analysis:
        left_ratio = sum(l["road_ratio"] for l in lateral_analysis if l["offset_m"] < 0) / max(1, sum(1 for l in lateral_analysis if l["offset_m"] < 0))
        right_ratio = sum(l["road_ratio"] for l in lateral_analysis if l["offset_m"] > 0) / max(1, sum(1 for l in lateral_analysis if l["offset_m"] > 0))
        
        if hint["forward_clear"]:
            hint["recommended_direction"] = "forward"
            hint["confidence"] = min(max_clear_distance / 2.0, 1.0)
        elif left_ratio > right_ratio and left_ratio > road_threshold:
            hint["recommended_direction"] = "left"
            hint["confidence"] = left_ratio
        elif right_ratio > road_threshold:
            hint["recommended_direction"] = "right"
            hint["confidence"] = right_ratio
        else:
            hint["recommended_direction"] = "stop"
            hint["confidence"] = 1.0 - max(left_ratio, right_ratio)
    
    return hint
