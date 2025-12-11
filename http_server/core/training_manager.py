"""学習管理"""
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class TrainingStatus(str, Enum):
    IDLE = "idle"
    PREPARING = "preparing"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainingState:
    """学習状態"""
    status: TrainingStatus = TrainingStatus.IDLE
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    best_val_loss: float = float('inf')
    start_time: Optional[str] = None
    elapsed_seconds: float = 0.0
    message: str = ""
    history: list = field(default_factory=list)


class TrainingManager:
    """セグメンテーションモデル学習管理"""
    
    def __init__(self, models_dir: str = "/home/jetson/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = TrainingState()
        self._training_thread: Optional[threading.Thread] = None
        self._cancel_requested = False
    
    def get_status(self) -> Dict[str, Any]:
        """学習状態を取得"""
        return {
            "status": self.state.status.value,
            "current_epoch": self.state.current_epoch,
            "total_epochs": self.state.total_epochs,
            "train_loss": self.state.train_loss,
            "val_loss": self.state.val_loss,
            "best_val_loss": self.state.best_val_loss,
            "start_time": self.state.start_time,
            "elapsed_seconds": self.state.elapsed_seconds,
            "message": self.state.message,
            "history": self.state.history[-20:]  # 最新20エポック
        }
    
    def prepare_training_data(self, dataset_name: str) -> Dict[str, Any]:
        """データセットから学習データを準備
        
        Args:
            dataset_name: データセット名
        
        Returns:
            準備結果
        """
        from .dataset_manager import dataset_manager
        
        # データセット情報取得
        info = dataset_manager.get_dataset_info(dataset_name)
        if "error" in info:
            return info
        
        dataset_dir = Path(info["path"])
        
        # セグメンテーション画像があるか確認
        if info["segmentation_count"] == 0:
            return {"error": "No segmentation data. Run OneFormer first."}
        
        # ROADマッピング取得
        road_mapping = info.get("road_mapping", {})
        road_labels = road_mapping.get("road_labels", [])
        
        if not road_labels:
            # グローバルなROADマッピングを使用
            from .road_mapping import get_road_mapping
            road_labels = get_road_mapping().get_road_labels()
        
        if not road_labels:
            return {"error": "No ROAD labels defined. Annotate road areas first."}
        
        # 学習データディレクトリ作成
        training_data_dir = dataset_dir / "training_data"
        train_dir = training_data_dir / "train"
        val_dir = training_data_dir / "val"
        
        (train_dir / "images").mkdir(parents=True, exist_ok=True)
        (train_dir / "labels").mkdir(parents=True, exist_ok=True)
        (val_dir / "images").mkdir(parents=True, exist_ok=True)
        (val_dir / "labels").mkdir(parents=True, exist_ok=True)
        
        # 画像とセグメンテーションをペアで収集
        images_result = dataset_manager.get_images(dataset_name)
        images_with_seg = [
            img for img in images_result["images"]
            if img["has_segmentation"]
        ]
        
        if len(images_with_seg) < 5:
            return {"error": f"Not enough images with segmentation ({len(images_with_seg)}). Need at least 5."}
        
        # Train/Val分割 (80/20)
        import random
        random.seed(42)
        random.shuffle(images_with_seg)
        
        split_idx = int(len(images_with_seg) * 0.8)
        train_images = images_with_seg[:split_idx]
        val_images = images_with_seg[split_idx:]
        
        # ADE20KラベルIDを取得
        road_label_ids = self._get_road_label_ids(road_labels)
        
        # 画像とマスクを処理
        import cv2
        
        def process_split(images, output_dir):
            count = 0
            for img_info in images:
                try:
                    # 元画像をコピー
                    src_img = Path(img_info["path"])
                    dst_img = output_dir / "images" / src_img.name
                    
                    import shutil
                    shutil.copy(src_img, dst_img)
                    
                    # セグメンテーションマスクからバイナリマスク生成
                    seg_path = Path(img_info["seg_path"])
                    seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
                    
                    # ROADマスク作成 (0: Other, 1: ROAD, 2: MYCAR)
                    binary_mask = np.zeros_like(seg_mask, dtype=np.uint8)
                    for label_id in road_label_ids:
                        binary_mask[seg_mask == label_id] = 1  # ROAD
                    
                    # ラベル保存
                    label_path = output_dir / "labels" / f"{src_img.stem}.png"
                    cv2.imwrite(str(label_path), binary_mask)
                    
                    count += 1
                except Exception as e:
                    print(f"[Training] Error processing {img_info['path']}: {e}")
            
            return count
        
        train_count = process_split(train_images, train_dir)
        val_count = process_split(val_images, val_dir)
        
        return {
            "dataset": dataset_name,
            "training_data_dir": str(training_data_dir),
            "train_count": train_count,
            "val_count": val_count,
            "road_labels": road_labels,
            "road_label_ids": list(road_label_ids)
        }
    
    def _get_road_label_ids(self, road_labels: list) -> set:
        """「ROADラベル名からADE20K IDを取得"""
        # ADE20Kラベルをローカルで定義（インポート問題回避）
        ADE20K_LABELS = {
            3: "floor", 4: "floor", 7: "road", 12: "sidewalk",
            29: "rug", 30: "field", 47: "sand", 53: "path",
            55: "runway", 92: "dirt track", 109: "plaything"
        }
        
        road_label_ids = set()
        for label_name in road_labels:
            for lid, lname in ADE20K_LABELS.items():
                if lname == label_name:
                    road_label_ids.add(lid)
                    break
        return road_label_ids
    
    def start_training(
        self,
        dataset_name: str,
        epochs: int = 50,
        batch_size: int = 4,
        learning_rate: float = 1e-4
    ) -> Dict[str, Any]:
        """学習を開始
        
        Args:
            dataset_name: データセット名
            epochs: エポック数
            batch_size: バッチサイズ
            learning_rate: 学習率
        """
        if self.state.status == TrainingStatus.TRAINING:
            return {"error": "Training already in progress"}
        
        # データ準備
        self.state = TrainingState(
            status=TrainingStatus.PREPARING,
            message="Preparing training data..."
        )
        
        prep_result = self.prepare_training_data(dataset_name)
        if "error" in prep_result:
            self.state.status = TrainingStatus.FAILED
            self.state.message = prep_result["error"]
            return prep_result
        
        training_data_dir = Path(prep_result["training_data_dir"])
        
        # 学習スレッド開始
        self._cancel_requested = False
        self._training_thread = threading.Thread(
            target=self._training_loop,
            args=(training_data_dir, epochs, batch_size, learning_rate),
            daemon=True
        )
        self._training_thread.start()
        
        return {
            "status": "started",
            "dataset": dataset_name,
            "epochs": epochs,
            "train_count": prep_result["train_count"],
            "val_count": prep_result["val_count"]
        }
    
    def cancel_training(self) -> Dict[str, Any]:
        """学習をキャンセル"""
        if self.state.status != TrainingStatus.TRAINING:
            return {"error": "No training in progress"}
        
        self._cancel_requested = True
        self.state.message = "Cancelling..."
        
        return {"status": "cancelling"}
    
    def _training_loop(
        self,
        training_data_dir: Path,
        epochs: int,
        batch_size: int,
        learning_rate: float
    ):
        """学習ループ（別スレッドで実行）"""
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import Dataset, DataLoader
            
            # smpがインストールされているか確認
            try:
                import segmentation_models_pytorch as smp
            except ImportError:
                self.state.status = TrainingStatus.FAILED
                self.state.message = "segmentation_models_pytorch not installed"
                return
            
            self.state.status = TrainingStatus.TRAINING
            self.state.total_epochs = epochs
            self.state.start_time = datetime.now().isoformat()
            self.state.message = "Loading model..."
            
            start_time = time.time()
            
            # デバイス設定
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"[Training] Using device: {device}")
            
            # モデル作成
            model = smp.DeepLabV3Plus(
                encoder_name="mobilenet_v2",
                encoder_weights='imagenet',
                in_channels=3,
                classes=3,  # Other, ROAD, MYCAR
                activation=None
            )
            model.to(device)
            
            # データセット作成
            from .training_dataset import ROADDataset, get_transforms
            
            train_dataset = ROADDataset(
                image_dir=training_data_dir / "train" / "images",
                label_dir=training_data_dir / "train" / "labels",
                transform=get_transforms((320, 240), is_train=True)
            )
            val_dataset = ROADDataset(
                image_dir=training_data_dir / "val" / "images",
                label_dir=training_data_dir / "val" / "labels",
                transform=get_transforms((320, 240), is_train=False)
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True
            )
            
            # 損失関数・オプティマイザ
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            self.state.message = "Training..."
            
            # 学習ループ
            for epoch in range(epochs):
                if self._cancel_requested:
                    self.state.status = TrainingStatus.CANCELLED
                    self.state.message = "Training cancelled"
                    return
                
                # Train
                model.train()
                train_loss = 0.0
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validate
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                # 状態更新
                self.state.current_epoch = epoch + 1
                self.state.train_loss = train_loss
                self.state.val_loss = val_loss
                self.state.elapsed_seconds = time.time() - start_time
                self.state.history.append({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })
                
                # ベストモデル保存
                if val_loss < self.state.best_val_loss:
                    self.state.best_val_loss = val_loss
                    torch.save(model.state_dict(), self.models_dir / "best_model.pth")
                
                print(f"[Training] Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
            # 最終モデル保存
            torch.save(model.state_dict(), self.models_dir / "final_model.pth")
            
            # ONNXエクスポート
            self.state.message = "Exporting ONNX..."
            self._export_onnx(model, self.models_dir / "road_segmentation.onnx")
            
            self.state.status = TrainingStatus.COMPLETED
            self.state.message = "Training completed!"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.state.status = TrainingStatus.FAILED
            self.state.message = f"Training failed: {e}"
    
    def _export_onnx(self, model, onnx_path: Path):
        """ONNXエクスポート"""
        import torch
        
        model.eval()
        model.to('cpu')
        
        dummy_input = torch.randn(1, 3, 240, 320)
        
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch'},
                'output': {0: 'batch'}
            }
        )
        
        print(f"[Training] ONNX exported to {onnx_path}")


# シングルトン
training_manager = TrainingManager()
