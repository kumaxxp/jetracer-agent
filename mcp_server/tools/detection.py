"""物体検出ツール（YOLOv8）"""

from pathlib import Path


async def detect_objects(image_path: str, confidence: float = 0.5) -> dict:
    """YOLOv8で画像内の物体を検出する
    
    Args:
        image_path: 画像ファイルパス
        confidence: 信頼度閾値（デフォルト0.5）
    
    Returns:
        detected_objects: 検出された物体とその数
        total_count: 検出総数
        details: 各検出の詳細（位置、信頼度）
    """
    try:
        from ultralytics import YOLO
        import cv2
        
        # モデルパス
        model_path = Path("models/yolov8n.pt")
        if not model_path.exists():
            # プロジェクトルートから探す
            project_root = Path(__file__).parent.parent.parent
            model_path = project_root / "models" / "yolov8n.pt"
        
        if not model_path.exists():
            # 自動ダウンロード
            model = YOLO("yolov8n.pt")
        else:
            model = YOLO(str(model_path))
        
        # 画像読み込み
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"画像を読み込めません: {image_path}"}
        
        # 推論
        results = model(img, verbose=False)
        
        # 結果を集計
        detections = []
        object_counts = {}
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                
                if conf >= confidence:
                    # カウント
                    object_counts[name] = object_counts.get(name, 0) + 1
                    
                    # 詳細
                    xyxy = box.xyxy[0].tolist()
                    detections.append({
                        "object": name,
                        "confidence": round(conf, 2),
                        "bbox": [round(x, 1) for x in xyxy]
                    })
        
        return {
            "detected_objects": object_counts,
            "total_count": len(detections),
            "details": detections
        }
        
    except ImportError:
        return {
            "error": "ultralytics がインストールされていません",
            "detected_objects": {},
            "total_count": 0
        }
    except Exception as e:
        return {
            "error": str(e),
            "detected_objects": {},
            "total_count": 0
        }


async def detect_objects_simple(image_path: str) -> dict:
    """簡易物体検出（YOLOなしの場合のフォールバック）
    
    色とエッジ情報から大まかなシーン推定を行う
    """
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"画像を読み込めません: {image_path}"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 基本統計
        brightness = gray.mean()
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = edges.sum() / (edges.size * 255) * 100
        
        # 色の分布
        hue_std = hsv[:, :, 0].std()
        sat_mean = hsv[:, :, 1].mean()
        
        # シーン推定
        scene_hints = []
        
        if edge_ratio < 2:
            scene_hints.append("単調な表面（壁、床など）")
        elif edge_ratio > 10:
            scene_hints.append("複雑なテクスチャ")
        
        if sat_mean < 30:
            scene_hints.append("彩度が低い（無彩色系）")
        
        if brightness < 50:
            scene_hints.append("暗い環境")
        elif brightness > 200:
            scene_hints.append("明るい環境")
        
        return {
            "detected_objects": {},
            "total_count": 0,
            "scene_hints": scene_hints,
            "note": "YOLOモデルなしの簡易解析"
        }
        
    except Exception as e:
        return {"error": str(e)}
