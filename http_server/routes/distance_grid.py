"""距離グリッドAPI"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import base64
import cv2
import numpy as np
import time

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


def _compute_cell_polygons(grid_data: dict) -> List[List[List[tuple]]]:
    """グリッド線データからセルのポリゴン座標を計算
    
    Returns:
        cell_polygons[row][col] = [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    h_lines = grid_data.get('horizontal_lines', [])
    v_lines = grid_data.get('vertical_lines', [])
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        return []
    
    cell_polygons = []
    
    for row in range(len(h_lines) - 1):
        row_cells = []
        h_line_top = h_lines[row]['points']
        h_line_bottom = h_lines[row + 1]['points']
        
        for col in range(len(v_lines) - 1):
            if col < len(h_line_top) and col + 1 < len(h_line_top):
                top_left = h_line_top[col]
                top_right = h_line_top[col + 1]
            else:
                continue
            
            if col < len(h_line_bottom) and col + 1 < len(h_line_bottom):
                bottom_left = h_line_bottom[col]
                bottom_right = h_line_bottom[col + 1]
            else:
                continue
            
            cell = [
                (int(top_left[0]), int(top_left[1])),
                (int(top_right[0]), int(top_right[1])),
                (int(bottom_right[0]), int(bottom_right[1])),
                (int(bottom_left[0]), int(bottom_left[1]))
            ]
            row_cells.append(cell)
        
        cell_polygons.append(row_cells)
    
    return cell_polygons


def _analyze_cell(segmentation: np.ndarray, cell_polygon: List[tuple], road_label_ids: set) -> float:
    """セル内のROAD比率を計算
    
    Args:
        segmentation: セグメンテーションマスク (H, W)
        cell_polygon: セルの4頂点座標
        road_label_ids: ROADとみなすラベルIDのセット
    
    Returns:
        road_ratio: 0.0〜1.0、画面外の場合は-1.0
    """
    h, w = segmentation.shape
    
    # ポリゴンをNumPy配列に変換
    pts = np.array(cell_polygon, np.int32)
    
    # 元のバウンディングボックス（クリップ前）
    orig_x_min = min(p[0] for p in cell_polygon)
    orig_x_max = max(p[0] for p in cell_polygon)
    orig_y_min = min(p[1] for p in cell_polygon)
    orig_y_max = max(p[1] for p in cell_polygon)
    
    orig_width = max(1, orig_x_max - orig_x_min)
    orig_height = max(1, orig_y_max - orig_y_min)
    
    # クリップ後のバウンディングボックス
    x_min = max(0, orig_x_min)
    x_max = min(w - 1, orig_x_max)
    y_min = max(0, orig_y_min)
    y_max = min(h - 1, orig_y_max)
    
    # セルが完全に画面外
    if x_min >= x_max or y_min >= y_max:
        return -1.0  # 画面外
    
    clipped_width = x_max - x_min
    clipped_height = y_max - y_min
    
    # セルの50%以上が画面外の場合は信頼性なし
    visible_ratio = (clipped_width * clipped_height) / (orig_width * orig_height)
    if visible_ratio < 0.5:
        return -1.0  # 信頼性なし
    
    # マスクを作成
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)
    
    # セル領域のピクセルを抽出
    cell_mask = mask[y_min:y_max+1, x_min:x_max+1]
    cell_seg = segmentation[y_min:y_max+1, x_min:x_max+1]
    
    # セル内のピクセルのみを対象
    cell_pixels = cell_seg[cell_mask == 1]
    
    if len(cell_pixels) == 0:
        return -1.0  # ピクセルがない場合
    
    # ROADピクセルの割合を計算
    road_pixels = sum(1 for p in cell_pixels if p in road_label_ids)
    road_ratio = road_pixels / len(cell_pixels)
    
    return float(road_ratio)


@router.get("/{camera_id}/analyze-segmentation")
async def analyze_segmentation_with_grid(camera_id: int, undistort: bool = False):
    """セグメンテーション結果をグリッドで分析（OneFormer使用）
    
    各グリッドセル内のROAD領域の割合を計算し、ナビゲーションヒントを生成
    高精度だが低速（~15秒）
    """
    from .oneformer import get_segmenter, _latest_seg_masks, run_oneformer_internal
    from ..core.road_mapping import get_road_mapping
    import torch
    
    try:
        print(f"[DistanceGrid] analyze-segmentation: camera={camera_id}")
        
        # CUDAストリームを同期（他のモデルとの競合を防ぐ）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
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
        
        h, w = segmentation.shape
        
        # ========== セル単位の分析（新機能）==========
        cell_polygons = _compute_cell_polygons(grid_data)
        cell_analysis = []
        
        for row, row_cells in enumerate(cell_polygons):
            row_data = []
            for col, cell_polygon in enumerate(row_cells):
                road_ratio = _analyze_cell(segmentation, cell_polygon, road_label_ids)
                row_data.append(road_ratio)
            cell_analysis.append(row_data)
        
        num_rows = len(cell_analysis)
        num_cols = len(cell_analysis[0]) if cell_analysis else 0
        print(f"[DistanceGrid] Cell analysis: {num_rows} rows x {num_cols} cols")
        
        # デバッグ出力
        for i, row in enumerate(cell_analysis):
            print(f"[DistanceGrid]   Row {i}: {[f'{v:.2f}' for v in row]}")
        
        # ========== 従来の行/列分析（後方互換性のため維持）==========
        depth_analysis = []
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
        navigation_hint = _compute_navigation_hint(depth_analysis, lateral_analysis, cell_analysis)
        
        return {
            "camera_id": camera_id,
            "cell_analysis": cell_analysis,  # 新機能: セル単位の分析結果
            "depth_analysis": depth_analysis,
            "lateral_analysis": lateral_analysis,
            "navigation_hint": navigation_hint,
            "road_labels": list(road_labels),
            "grid_config": {
                "depth_range_m": [config.grid_depth_min_m, config.grid_depth_max_m],
                "width_m": config.grid_width_m,
                "num_rows": num_rows,
                "num_cols": num_cols
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


@router.get("/{camera_id}/analyze-segmentation-lightweight")
async def analyze_segmentation_lightweight(camera_id: int, undistort: bool = False):
    """軽量モデル（DeepLabV3+）でセグメンテーション結果をグリッド分析
    
    OneFormerより高速（30-50ms）だが精度は若干低い
    """
    import torch
    from pathlib import Path
    from ..core.road_mapping import get_road_mapping
    
    try:
        print(f"[DistanceGrid] analyze-segmentation-lightweight: camera={camera_id}")
        
        # CUDAストリームを同期（他のモデルとの競合を防ぐ）
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # カメラからフレーム取得
        frame = camera_manager.read(camera_id, apply_undistort=undistort)
        if frame is None:
            raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
        
        # 軽量モデルのパス
        model_dir = Path.home() / "models"
        
        # ONNXモデルを優先、なければPyTorchモデル
        onnx_path = model_dir / "road_segmentation.onnx"
        pth_path = model_dir / "best_model.pth"
        
        start_time = time.time()
        
        if onnx_path.exists():
            # ONNX推論
            segmentation, inference_time, model_type = _run_lightweight_onnx(frame, onnx_path)
        elif pth_path.exists():
            # PyTorch推論
            segmentation, inference_time, model_type = _run_lightweight_pth(frame, pth_path)
        else:
            raise HTTPException(
                status_code=404, 
                detail="No lightweight model found. Train a model first."
            )
        
        print(f"[DistanceGrid] Lightweight model ({model_type}): {inference_time:.1f}ms")
        print(f"[DistanceGrid] Segmentation shape: {segmentation.shape}")
        
        # ROADラベルID（軽量モデルは固定: 0=Other, 1=ROAD, 2=MYCAR）
        road_label_ids = {1}  # ROAD
        
        # グリッドデータを取得
        config = distance_grid_manager.get_config(camera_id)
        grid_data = distance_grid_manager.compute_grid_lines(camera_id)
        
        h, w = segmentation.shape
        
        # セル単位の分析
        cell_polygons = _compute_cell_polygons(grid_data)
        cell_analysis = []
        
        for row, row_cells in enumerate(cell_polygons):
            row_data = []
            for col, cell_polygon in enumerate(row_cells):
                road_ratio = _analyze_cell(segmentation, cell_polygon, road_label_ids)
                row_data.append(road_ratio)
            cell_analysis.append(row_data)
        
        num_rows = len(cell_analysis)
        num_cols = len(cell_analysis[0]) if cell_analysis else 0
        
        # 行/列分析
        depth_analysis = []
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
        
        # ナビゲーションヒント
        navigation_hint = _compute_navigation_hint(depth_analysis, lateral_analysis, cell_analysis)
        
        # オーバーレイ画像作成
        overlay = _create_lightweight_overlay(frame, segmentation)
        
        # グリッドオーバーレイを追加
        grid_overlay = distance_grid_manager.draw_grid_overlay(
            camera_id, overlay,
            color=(0, 255, 0),
            thickness=2,
            show_labels=True
        )
        
        _, buffer = cv2.imencode('.jpg', grid_overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        overlay_base64 = base64.b64encode(buffer).decode('utf-8')
        
        total_time = (time.time() - start_time) * 1000
        
        # ROAD統計
        road_percentage = float(np.sum(segmentation == 1) / segmentation.size * 100)
        
        return {
            "camera_id": camera_id,
            "model_type": model_type,
            "cell_analysis": cell_analysis,
            "depth_analysis": depth_analysis,
            "lateral_analysis": lateral_analysis,
            "navigation_hint": navigation_hint,
            "overlay_base64": overlay_base64,
            "road_percentage": round(road_percentage, 1),
            "grid_config": {
                "depth_range_m": [config.grid_depth_min_m, config.grid_depth_max_m],
                "width_m": config.grid_width_m,
                "num_rows": num_rows,
                "num_cols": num_cols
            },
            "inference_time_ms": round(inference_time, 1),
            "total_time_ms": round(total_time, 1)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[DistanceGrid] Error in analyze-segmentation-lightweight: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _run_lightweight_onnx(frame: np.ndarray, model_path) -> tuple:
    """ONNX軽量モデルで推論"""
    start = time.time()
    
    # OpenCV DNNでロード
    net = cv2.dnn.readNetFromONNX(str(model_path))
    
    # CUDAバックエンドを設定（setInputの前に！）
    backend = "CPU"
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        backend = "CUDA_FP16"
    except Exception as e:
        print(f"[Lightweight] CUDA setup failed: {e}, using CPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # 前処理（ImageNet標準化）
    input_size = (320, 240)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # CHW形式に変換してバッチ次元追加
    blob = img.transpose(2, 0, 1).reshape(1, 3, 240, 320).astype(np.float32)
    
    # 推論
    net.setInput(blob)
    output = net.forward()
    
    inference_time = (time.time() - start) * 1000
    
    # 後処理
    mask = np.argmax(output[0], axis=0).astype(np.uint8)
    
    # 元画像サイズにリサイズ
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask, inference_time, f"ONNX ({backend})"


def _run_lightweight_pth(frame: np.ndarray, model_path) -> tuple:
    """PyTorch軽量モデルで推論"""
    import torch
    
    start = time.time()
    
    # デバイス設定
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルロード
    try:
        import segmentation_models_pytorch as smp
        model = smp.DeepLabV3Plus(
            encoder_name="mobilenet_v2",
            encoder_weights=None,
            in_channels=3,
            classes=3
        )
    except ImportError:
        raise HTTPException(
            status_code=500, 
            detail="segmentation_models_pytorch not installed"
        )
    
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    
    # 前処理
    input_size = (320, 240)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img = img.astype(np.float32) / 255.0
    
    # ImageNet標準化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    
    # テンソル変換
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0).float().to(device)
    
    # 推論
    with torch.no_grad():
        output = model(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    inference_time = (time.time() - start) * 1000
    
    # 後処理
    mask = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    # 元画像サイズにリサイズ
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return mask, inference_time, f"PyTorch ({device})"


def _create_lightweight_overlay(frame: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
    """軽量モデル用のオーバーレイ画像を作成"""
    # カラーマップ: 0=黒(Other), 1=緑(ROAD), 2=赤(MYCAR)
    overlay = np.zeros_like(frame)
    overlay[segmentation == 1] = [0, 255, 0]    # ROAD = 緑 (BGR)
    overlay[segmentation == 2] = [0, 0, 255]    # MYCAR = 赤 (BGR)
    
    # ブレンド
    result = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return result


def _compute_navigation_hint(depth_analysis: list, lateral_analysis: list, cell_analysis: list = None) -> Dict[str, Any]:
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
    
    # セル分析がある場合はそれを使用
    if cell_analysis and len(cell_analysis) > 0 and len(cell_analysis[0]) > 0:
        num_cols = len(cell_analysis[0])
        center_col = num_cols // 2
        
        # 各列の通行可能性（全行の平均）
        col_scores = []
        for col in range(num_cols):
            col_sum = sum(row[col] for row in cell_analysis if col < len(row))
            col_avg = col_sum / len(cell_analysis) if cell_analysis else 0
            col_scores.append(col_avg)
        
        # 最も通行可能な列を見つける
        best_col = center_col
        best_score = col_scores[center_col] if center_col < len(col_scores) else 0
        for col, score in enumerate(col_scores):
            if score > best_score:
                best_score = score
                best_col = col
        
        # 方向を決定
        if best_score >= road_threshold:
            if best_col < center_col - 1:
                hint["recommended_direction"] = "left"
            elif best_col > center_col + 1:
                hint["recommended_direction"] = "right"
            else:
                hint["recommended_direction"] = "forward"
            hint["confidence"] = best_score
        else:
            hint["recommended_direction"] = "stop"
            hint["confidence"] = 1.0 - best_score
    
    elif lateral_analysis:
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
