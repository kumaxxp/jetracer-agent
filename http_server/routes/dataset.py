"""データセット管理API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import base64
import cv2
import numpy as np

from ..core.dataset_manager import dataset_manager
from ..core.camera_manager import camera_manager
from .oneformer import _latest_seg_masks

router = APIRouter(prefix="/dataset", tags=["dataset"])


class CreateDatasetRequest(BaseModel):
    name: str


class SelectDatasetRequest(BaseModel):
    name: str


class AddImageRequest(BaseModel):
    include_segmentation: bool = True


@router.get("/list")
def list_datasets():
    """データセット一覧を取得"""
    datasets = dataset_manager.list_datasets()
    return {
        "datasets": datasets,
        "current": dataset_manager.current_dataset
    }


@router.post("/create")
def create_dataset(request: CreateDatasetRequest):
    """新しいデータセットを作成"""
    result = dataset_manager.create_dataset(request.name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/select")
def select_dataset(request: SelectDatasetRequest):
    """データセットを選択"""
    result = dataset_manager.select_dataset(request.name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.delete("/{name}")
def delete_dataset(name: str):
    """データセットを削除"""
    result = dataset_manager.delete_dataset(name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{name}/info")
def get_dataset_info(name: str):
    """データセット詳細情報を取得"""
    result = dataset_manager.get_dataset_info(name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/{name}/images")
def get_dataset_images(name: str, camera_id: Optional[int] = None):
    """データセットの画像一覧を取得"""
    result = dataset_manager.get_images(name, camera_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.post("/{name}/add/{camera_id}")
def add_image_to_dataset(name: str, camera_id: int, include_segmentation: bool = True):
    """カメラ画像をデータセットに追加
    
    Args:
        name: データセット名
        camera_id: カメラID (0 or 1)
        include_segmentation: セグメンテーション結果も保存するか
    """
    # データセットを選択
    result = dataset_manager.select_dataset(name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    # カメラから画像取得
    frame = camera_manager.read(camera_id)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
    
    # セグメンテーションマスク（あれば）
    seg_mask = None
    if include_segmentation and camera_id in _latest_seg_masks:
        seg_mask = _latest_seg_masks[camera_id]
    
    # データセットに追加
    result = dataset_manager.add_image(camera_id, frame, seg_mask)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result


@router.post("/{name}/add-with-oneformer/{camera_id}")
async def add_image_with_oneformer(name: str, camera_id: int):
    """OneFormerでセグメンテーションしてからデータセットに追加
    
    Args:
        name: データセット名
        camera_id: カメラID (0 or 1)
    """
    from .oneformer import run_oneformer_internal
    
    # データセットを選択
    result = dataset_manager.select_dataset(name)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    
    # カメラから画像取得
    frame = camera_manager.read(camera_id)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
    
    # OneFormerでセグメンテーション実行
    try:
        seg_result = run_oneformer_internal(camera_id, highlight_road=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OneFormer error: {e}")
    
    # セグメンテーションマスク取得
    seg_mask = _latest_seg_masks.get(camera_id)
    
    # データセットに追加
    result = dataset_manager.add_image(camera_id, frame, seg_mask)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    result["segmentation_time_ms"] = seg_result.get("segmentation_time_ms", 0)
    result["num_classes"] = seg_result.get("num_classes", 0)
    
    return result


@router.put("/{name}/road-mapping")
def update_road_mapping(name: str, road_labels: List[str]):
    """ROADマッピングを更新"""
    result = dataset_manager.update_road_mapping(road_labels, name)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/{name}/image/{image_name}")
def get_image(name: str, image_name: str):
    """画像をBase64で取得"""
    from pathlib import Path
    
    dataset_dir = dataset_manager.base_dir / name
    
    # カメラ0, 1両方で探す
    for cam_id in [0, 1]:
        img_path = dataset_dir / "images" / f"camera_{cam_id}" / image_name
        if img_path.exists():
            with open(img_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode()
            
            # セグメンテーションも探す
            seg_name = img_path.stem + "_seg.png"
            seg_path = dataset_dir / "segmentation" / seg_name
            seg_base64 = None
            if seg_path.exists():
                with open(seg_path, "rb") as f:
                    seg_base64 = base64.b64encode(f.read()).decode()
            
            return {
                "name": image_name,
                "image_base64": img_base64,
                "seg_base64": seg_base64,
                "camera_id": cam_id
            }
    
    raise HTTPException(status_code=404, detail=f"Image '{image_name}' not found")
