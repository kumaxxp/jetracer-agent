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
    status = distance_grid_manager.get_status(camera_id)
    print(f"[DistanceGrid API] Status for camera {camera_id}: {status}")
    return status


@router.post("/reload-calibration")
async def reload_calibration():
    """キャリブレーションデータを再読み込み"""
    cameras = distance_grid_manager.reload_calibration()
    return {
        "message": "Calibration reloaded",
        "cameras_with_calibration": cameras
    }


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
async def get_grid_preview(camera_id: int, undistort: bool = True):
    """グリッドプレビュー画像を取得"""
    print(f"[DistanceGrid API] Preview request: camera={camera_id}, undistort={undistort}")
    
    status = distance_grid_manager.get_status(camera_id)
    print(f"[DistanceGrid API] Grid status: {status}")
    
    frame = camera_manager.read(camera_id, apply_undistort=undistort)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
    
    print(f"[DistanceGrid API] Frame shape: {frame.shape}")
    
    result = distance_grid_manager.draw_grid_overlay(
        camera_id, frame,
        color=(0, 255, 0),
        thickness=1,
        show_labels=True
    )
    
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
    undistort: bool = True
):
    """セグメンテーション結果にグリッドオーバーレイを合成"""
    from .oneformer import run_oneformer_internal
    
    try:
        print(f"[DistanceGrid] overlay-on-segmentation: camera={camera_id}, highlight_road={highlight_road}")
        
        # OneFormerでセグメンテーション実行
        seg_result = run_oneformer_internal(camera_id, highlight_road)
        print(f"[DistanceGrid] OneFormer result keys: {seg_result.keys()}")
        
        # オーバーレイ画像をデコード
        overlay_base64 = seg_result.get('overlay_base64', '')
        if not overlay_base64:
            raise HTTPException(status_code=500, detail="Failed to get overlay image")
        
        print(f"[DistanceGrid] overlay_base64 length: {len(overlay_base64)}")
        
        overlay_bytes = base64.b64decode(overlay_base64)
        overlay_array = np.frombuffer(overlay_bytes, dtype=np.uint8)
        overlay_image = cv2.imdecode(overlay_array, cv2.IMREAD_COLOR)
        
        if overlay_image is None:
            raise HTTPException(status_code=500, detail="Failed to decode overlay image")
        
        print(f"[DistanceGrid] overlay_image shape: {overlay_image.shape}")
        
        # グリッドオーバーレイを追加（緑色）
        grid_overlay = distance_grid_manager.draw_grid_overlay(
            camera_id, overlay_image,
            color=(0, 255, 0),
            thickness=2,
            show_labels=True
        )
        
        # JPEG圧縮
        _, buffer = cv2.imencode('.jpg', grid_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"[DistanceGrid] Output image base64 length: {len(img_base64)}")
        
        return {
            "image_base64": img_base64,
            "camera_id": camera_id,
            "segmentation_classes": seg_result.get("classes", []),
            "num_classes": seg_result.get("num_classes", 0),
            "road_labels": seg_result.get("road_labels", []),
            "process_time_ms": seg_result.get("process_time_ms", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[DistanceGrid] Error in overlay-on-segmentation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
    
    各グリッドセル内のROAD領域の割合を計算し、ナビゲーションヒントを生成
    """
    from .oneformer import get_segmenter, _latest_seg_masks, run_oneformer_internal
    from ..core.road_mapping import get_road_mapping
    
    try:
        print(f"[DistanceGrid] analyze-segmentation: camera={camera_id}")
        
        # OneFormerでセグメンテーション実行
        seg_result = run_oneformer_internal(camera_id, highlight_road=True)
        
        # 最新のセグメンテーションマスクを取得
        if camera_id not in _latest_seg_masks:
            raise HTTPException(status_code=500, detail="Segmentation mask not available")
        
        segmentation = _latest_seg_masks[camera_id]
        print(f"[DistanceGrid] Segmentation mask shape: {segmentation.shape}")
        
        # ROADラベルを取得
        road_mapping = get_road_mapping()
        road_labels = road_mapping.get_road_labels()
        
        # ラベル名からIDへの変換
        segmenter = get_segmenter()
        road_label_ids = set()
        for label_name in road_labels:
            for lid, lname in segmenter.id2label.items():
                if lname == label_name:
                    road_label_ids.add(int(lid))
                    break
        
        print(f"[DistanceGrid] Road label IDs: {road_label_ids}")
        
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
            
            y_pixels = [int(p[1]) for p in points]
            y_avg = np.mean(y_pixels)
            
            if 0 <= y_avg < h:
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
            },
            "process_time_ms": seg_result.get("process_time_ms", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[DistanceGrid] Error in analyze-segmentation: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


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
    
    sorted_depth = sorted(depth_analysis, key=lambda x: x["depth_m"])
    
    road_threshold = 0.3
    
    max_clear_distance = 0.0
    for d in sorted_depth:
        if d["road_ratio"] >= road_threshold:
            max_clear_distance = d["depth_m"]
        else:
            break
    
    hint["max_clear_distance_m"] = max_clear_distance
    hint["forward_clear"] = max_clear_distance >= 0.5
    
    if lateral_analysis:
        left_count = sum(1 for l in lateral_analysis if l["offset_m"] < 0)
        right_count = sum(1 for l in lateral_analysis if l["offset_m"] > 0)
        
        left_ratio = sum(l["road_ratio"] for l in lateral_analysis if l["offset_m"] < 0) / max(1, left_count)
        right_ratio = sum(l["road_ratio"] for l in lateral_analysis if l["offset_m"] > 0) / max(1, right_count)
        
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
