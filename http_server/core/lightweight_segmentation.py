"""Lightweight セグメンテーションモジュール"""
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any
import base64
import time


class LightweightSegmentation:
    """Lightweight セグメンテーション（DeepLabV3+ MobileNetV2）"""
    
    def __init__(self):
        self._model = None
        self._device = None
        self._model_path = None
        self._input_size = (224, 224)  # モデルの入力サイズ
        
        # クラス定義
        self._class_names = ["Other", "ROAD", "MYCAR"]
        self._class_colors = {
            0: (128, 128, 128),  # Other: グレー
            1: (0, 255, 0),      # ROAD: 緑
            2: (255, 0, 0),      # MYCAR: 赤
        }
    
    def load(self, model_path: str = None) -> bool:
        """モデルをロード"""
        try:
            import segmentation_models_pytorch as smp
            
            if model_path is None:
                model_path = Path.home() / "models" / "best_model.pth"
            else:
                model_path = Path(model_path)
            
            if not model_path.exists():
                print(f"[LightweightSeg] Model not found: {model_path}")
                return False
            
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self._model = smp.DeepLabV3Plus(
                encoder_name="mobilenet_v2",
                encoder_weights=None,
                in_channels=3,
                classes=3
            )
            
            self._model.load_state_dict(torch.load(str(model_path), map_location=self._device))
            self._model.to(self._device)
            self._model.eval()
            
            self._model_path = str(model_path)
            print(f"[LightweightSeg] Model loaded on {self._device}")
            return True
            
        except Exception as e:
            print(f"[LightweightSeg] Failed to load model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def is_loaded(self) -> bool:
        """モデルがロードされているか確認"""
        return self._model is not None
    
    def segment(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """セグメンテーション実行
        
        Args:
            frame: BGR画像 (H, W, 3)
            
        Returns:
            セグメンテーション結果
        """
        if not self.is_loaded():
            return None
        
        try:
            start_time = time.perf_counter()
            
            # 前処理
            h, w = frame.shape[:2]
            input_img = cv2.resize(frame, self._input_size)
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
            input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0
            input_tensor = input_tensor.unsqueeze(0).to(self._device)
            
            # 推論
            with torch.no_grad():
                output = self._model(input_tensor)
                pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
            
            inference_time = (time.perf_counter() - start_time) * 1000
            
            # 元のサイズにリサイズ
            pred_resized = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 統計計算
            total_pixels = pred_resized.size
            road_pixels = np.sum(pred_resized == 1)
            road_percentage = (road_pixels / total_pixels) * 100
            
            # オーバーレイ画像を作成
            overlay = frame.copy()
            for class_id, color in self._class_colors.items():
                mask = pred_resized == class_id
                overlay[mask] = cv2.addWeighted(frame, 0.5, np.full_like(frame, color), 0.5, 0)[mask]
            
            # Base64エンコード
            _, buffer = cv2.imencode('.png', overlay)
            overlay_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return {
                "mask": pred_resized,
                "road_percentage": road_percentage,
                "class_counts": {
                    "other": int(np.sum(pred_resized == 0)),
                    "road": int(road_pixels),
                    "mycar": int(np.sum(pred_resized == 2)),
                },
                "inference_time_ms": round(inference_time, 2),
                "overlay_base64": overlay_base64
            }
            
        except Exception as e:
            print(f"[LightweightSeg] Segmentation error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """モデル情報を取得"""
        return {
            "loaded": self.is_loaded(),
            "model_path": self._model_path,
            "device": str(self._device) if self._device else None,
            "input_size": self._input_size,
            "classes": self._class_names
        }


# シングルトンインスタンス
lightweight_segmentation = LightweightSegmentation()
