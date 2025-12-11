"""データセット管理"""
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


class DatasetManager:
    """セグメンテーション学習用データセット管理"""
    
    def __init__(self, base_dir: str = "/home/jetson/jetracer_data/datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.current_dataset: Optional[str] = None
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """データセット一覧を取得"""
        datasets = []
        for d in self.base_dir.iterdir():
            if d.is_dir() and (d / "metadata.json").exists():
                meta = self._load_metadata(d.name)
                datasets.append({
                    "name": d.name,
                    "created_at": meta.get("created_at"),
                    "image_count": meta.get("image_count", 0),
                    "has_segmentation": meta.get("has_segmentation", False)
                })
        return sorted(datasets, key=lambda x: x.get("created_at", ""), reverse=True)
    
    def create_dataset(self, name: str) -> Dict[str, Any]:
        """新しいデータセットを作成"""
        # 名前のバリデーション
        safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
        if not safe_name:
            safe_name = f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        dataset_dir = self.base_dir / safe_name
        if dataset_dir.exists():
            return {"error": f"Dataset '{safe_name}' already exists"}
        
        # ディレクトリ構造を作成
        (dataset_dir / "images" / "camera_0").mkdir(parents=True)
        (dataset_dir / "images" / "camera_1").mkdir(parents=True)
        (dataset_dir / "segmentation").mkdir(parents=True)
        (dataset_dir / "masks").mkdir(parents=True)
        
        # メタデータ作成
        metadata = {
            "name": safe_name,
            "created_at": datetime.now().isoformat(),
            "image_count": 0,
            "has_segmentation": False,
            "road_mapping": {}
        }
        self._save_metadata(safe_name, metadata)
        
        self.current_dataset = safe_name
        
        return {
            "name": safe_name,
            "path": str(dataset_dir),
            "created_at": metadata["created_at"]
        }
    
    def delete_dataset(self, name: str) -> Dict[str, Any]:
        """データセットを削除"""
        dataset_dir = self.base_dir / name
        if not dataset_dir.exists():
            return {"error": f"Dataset '{name}' not found"}
        
        shutil.rmtree(dataset_dir)
        
        if self.current_dataset == name:
            self.current_dataset = None
        
        return {"deleted": name}
    
    def select_dataset(self, name: str) -> Dict[str, Any]:
        """データセットを選択"""
        dataset_dir = self.base_dir / name
        if not dataset_dir.exists():
            return {"error": f"Dataset '{name}' not found"}
        
        self.current_dataset = name
        meta = self._load_metadata(name)
        
        return {
            "name": name,
            "image_count": meta.get("image_count", 0),
            "has_segmentation": meta.get("has_segmentation", False)
        }
    
    def add_image(self, camera_id: int, image: np.ndarray, 
                  seg_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """画像をデータセットに追加
        
        Args:
            camera_id: カメラID (0 or 1)
            image: BGR画像
            seg_mask: セグメンテーションマスク（オプション）
        """
        if not self.current_dataset:
            return {"error": "No dataset selected"}
        
        dataset_dir = self.base_dir / self.current_dataset
        
        # 画像ファイル名生成
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        img_filename = f"cam{camera_id}_{timestamp}.jpg"
        
        # 画像保存
        img_path = dataset_dir / "images" / f"camera_{camera_id}" / img_filename
        cv2.imwrite(str(img_path), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        # セグメンテーションマスク保存（あれば）
        seg_path = None
        if seg_mask is not None:
            seg_filename = f"cam{camera_id}_{timestamp}_seg.png"
            seg_path = dataset_dir / "segmentation" / seg_filename
            cv2.imwrite(str(seg_path), seg_mask)
        
        # メタデータ更新
        meta = self._load_metadata(self.current_dataset)
        meta["image_count"] = meta.get("image_count", 0) + 1
        if seg_mask is not None:
            meta["has_segmentation"] = True
        self._save_metadata(self.current_dataset, meta)
        
        return {
            "dataset": self.current_dataset,
            "camera_id": camera_id,
            "image_path": str(img_path),
            "seg_path": str(seg_path) if seg_path else None,
            "image_count": meta["image_count"]
        }
    
    def get_images(self, name: Optional[str] = None, camera_id: Optional[int] = None) -> Dict[str, Any]:
        """データセットの画像一覧を取得"""
        dataset_name = name or self.current_dataset
        if not dataset_name:
            return {"error": "No dataset specified or selected"}
        
        dataset_dir = self.base_dir / dataset_name
        if not dataset_dir.exists():
            return {"error": f"Dataset '{dataset_name}' not found"}
        
        images = []
        images_dir = dataset_dir / "images"
        
        camera_dirs = []
        if camera_id is not None:
            camera_dirs = [images_dir / f"camera_{camera_id}"]
        else:
            camera_dirs = [images_dir / "camera_0", images_dir / "camera_1"]
        
        for cam_dir in camera_dirs:
            if cam_dir.exists():
                for img_path in sorted(cam_dir.glob("*.jpg")):
                    # 対応するセグメンテーションファイルを探す
                    seg_filename = img_path.stem + "_seg.png"
                    seg_path = dataset_dir / "segmentation" / seg_filename
                    
                    images.append({
                        "name": img_path.name,
                        "path": str(img_path),
                        "camera_id": int(cam_dir.name.split("_")[1]),
                        "has_segmentation": seg_path.exists(),
                        "seg_path": str(seg_path) if seg_path.exists() else None
                    })
        
        return {
            "dataset": dataset_name,
            "images": images,
            "total": len(images)
        }
    
    def get_dataset_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """データセット詳細情報を取得"""
        dataset_name = name or self.current_dataset
        if not dataset_name:
            return {"error": "No dataset specified or selected"}
        
        dataset_dir = self.base_dir / dataset_name
        if not dataset_dir.exists():
            return {"error": f"Dataset '{dataset_name}' not found"}
        
        meta = self._load_metadata(dataset_name)
        
        # 画像数をカメラごとにカウント
        cam0_count = len(list((dataset_dir / "images" / "camera_0").glob("*.jpg")))
        cam1_count = len(list((dataset_dir / "images" / "camera_1").glob("*.jpg")))
        seg_count = len(list((dataset_dir / "segmentation").glob("*.png")))
        
        return {
            "name": dataset_name,
            "path": str(dataset_dir),
            "created_at": meta.get("created_at"),
            "camera_0_images": cam0_count,
            "camera_1_images": cam1_count,
            "total_images": cam0_count + cam1_count,
            "segmentation_count": seg_count,
            "has_segmentation": seg_count > 0,
            "road_mapping": meta.get("road_mapping", {})
        }
    
    def update_road_mapping(self, road_labels: List[str], name: Optional[str] = None) -> Dict[str, Any]:
        """ROADマッピングを更新"""
        dataset_name = name or self.current_dataset
        if not dataset_name:
            return {"error": "No dataset specified or selected"}
        
        meta = self._load_metadata(dataset_name)
        meta["road_mapping"] = {"road_labels": road_labels}
        self._save_metadata(dataset_name, meta)
        
        return {"road_labels": road_labels}
    
    def _load_metadata(self, name: str) -> Dict[str, Any]:
        """メタデータをロード"""
        meta_path = self.base_dir / name / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, name: str, metadata: Dict[str, Any]):
        """メタデータを保存"""
        meta_path = self.base_dir / name / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


# シングルトンインスタンス
dataset_manager = DatasetManager()
