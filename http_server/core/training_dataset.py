"""学習用データセット"""
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


class ROADDataset(Dataset):
    """ROAD セグメンテーション用データセット"""
    
    def __init__(self, image_dir: Path, label_dir: Path, transform=None):
        """
        Args:
            image_dir: 画像ディレクトリ
            label_dir: ラベルディレクトリ
            transform: Albumentations変換
        """
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.transform = transform
        
        self.image_files = sorted(list(self.image_dir.glob("*.jpg")))
        if not self.image_files:
            self.image_files = sorted(list(self.image_dir.glob("*.png")))
        
        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 画像読み込み
        img_path = self.image_files[idx]
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # ラベル読み込み
        label_path = self.label_dir / f"{img_path.stem}.png"
        if not label_path.exists():
            # フォールバック: 同名のpng
            label_path = self.label_dir / img_path.name.replace('.jpg', '.png')
        
        label = np.array(Image.open(label_path).convert('L'))
        label = label.astype(np.int64)
        
        # 変換適用
        if self.transform:
            transformed = self.transform(image=image, mask=label)
            image = transformed['image']
            label = transformed['mask']
            
            if torch.is_tensor(label):
                label = label.long()
        
        return image, label


def get_transforms(input_size: Tuple[int, int] = (320, 240), is_train: bool = True):
    """データ拡張変換を取得
    
    Args:
        input_size: (width, height)
        is_train: 学習用か（拡張あり）
    
    Returns:
        Albumentations変換
    """
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations is required. Install with: pip install albumentations")
    
    if is_train:
        return A.Compose([
            A.Resize(height=input_size[1], width=input_size[0]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=input_size[1], width=input_size[0]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
