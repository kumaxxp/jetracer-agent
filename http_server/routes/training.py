"""学習管理API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback
import numpy as np
from pathlib import Path

router = APIRouter(prefix="/training", tags=["training"])


def get_training_manager():
    """遅延ロードでtraining_managerを取得"""
    from ..core.training_manager import training_manager
    return training_manager


class StartTrainingRequest(BaseModel):
    dataset_name: str
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4


@router.get("/status")
def get_training_status():
    """学習状態を取得"""
    try:
        return get_training_manager().get_status()
    except Exception as e:
        print(f"[Training API] get_status error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/prepare/{dataset_name}")
def prepare_training_data(dataset_name: str):
    """学習データを準備"""
    try:
        result = get_training_manager().prepare_training_data(dataset_name)
        if "error" in result:
            print(f"[Training API] prepare error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Training API] prepare exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
def start_training(request: StartTrainingRequest):
    """学習を開始"""
    try:
        print(f"[Training API] start_training: dataset={request.dataset_name}, epochs={request.epochs}")
        result = get_training_manager().start_training(
            dataset_name=request.dataset_name,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate
        )
        print(f"[Training API] start_training result: {result}")
        if "error" in result:
            print(f"[Training API] start error: {result['error']}")
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Training API] start exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel")
def cancel_training():
    """学習をキャンセル"""
    try:
        result = get_training_manager().cancel_training()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Training API] cancel exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models")
def list_models():
    """保存済みモデル一覧"""
    try:
        models_dir = get_training_manager().models_dir
        models = []
        
        for ext in [".pth", ".onnx"]:
            for model_path in models_dir.glob(f"*{ext}"):
                stat = model_path.stat()
                models.append({
                    "name": model_path.name,
                    "path": str(model_path),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": stat.st_mtime
                })
        
        return {"models": sorted(models, key=lambda x: x["modified"], reverse=True)}
    except Exception as e:
        print(f"[Training API] list_models exception: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test/{model_name}")
def test_model(model_name: str, camera_id: int = 0):
    """モデルをテスト（カメラ画像で推論実行）
    
    Args:
        model_name: モデルファイル名（.pthまたは.onnx）
        camera_id: カメラID
    
    Returns:
        result_base64: セグメンテーション結果画像
        inference_time_ms: 推論時間
        road_percentage: ROAD領域の割合
    """
    import time
    import cv2
    import base64
    
    from ..core.camera_manager import camera_manager
    
    models_dir = get_training_manager().models_dir
    model_path = models_dir / model_name
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_name}")
    
    # カメラから画像取得
    frame = camera_manager.read(camera_id)
    if frame is None:
        raise HTTPException(status_code=503, detail=f"Camera {camera_id} not available")
    
    try:
        if model_name.endswith('.pth'):
            result = _test_pth_model(model_path, frame)
        elif model_name.endswith('.onnx'):
            result = _test_onnx_model(model_path, frame)
        else:
            raise HTTPException(status_code=400, detail="Unsupported model format")
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _test_pth_model(model_path: Path, frame: np.ndarray) -> dict:
    """PyTorchモデルでテスト"""
    import torch
    import time
    import cv2
    import base64
    
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        return {"error": "segmentation_models_pytorch not installed"}
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Test] Using device: {device}")
    
    # モデルロード
    load_start = time.perf_counter()
    model = smp.DeepLabV3Plus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,
        in_channels=3,
        classes=3,
        activation=None
    )
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    load_time = (time.perf_counter() - load_start) * 1000
    print(f"[Test] Model loaded in {load_time:.1f}ms")
    
    # 前処理
    original_h, original_w = frame.shape[:2]
    input_size = (320, 240)
    
    resized = cv2.resize(frame, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 正規化（ImageNet）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
    
    # Tensor変換
    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    
    # 推論
    infer_start = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
    if device == 'cuda':
        torch.cuda.synchronize()
    inference_time = (time.perf_counter() - infer_start) * 1000
    
    # 後処理
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    # カラーマスク作成 (0: 黒, 1: 緑=ROAD, 2: 赤=MYCAR)
    color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    color_mask[pred == 1] = [0, 255, 0]  # ROAD = 緑
    color_mask[pred == 2] = [0, 0, 255]  # MYCAR = 赤
    
    # 元サイズにリサイズ
    color_mask = cv2.resize(color_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # オーバーレイ
    overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
    
    # ROAD割合計算
    road_pixels = np.sum(pred == 1)
    total_pixels = pred.size
    road_percentage = (road_pixels / total_pixels) * 100
    
    # Base64エンコード
    _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    
    _, orig_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    original_base64 = base64.b64encode(orig_buffer).decode('utf-8')
    
    return {
        "model_name": model_path.name,
        "model_type": "pth",
        "device": device,
        "load_time_ms": round(load_time, 1),
        "inference_time_ms": round(inference_time, 1),
        "input_size": list(input_size),
        "road_percentage": round(road_percentage, 2),
        "result_base64": result_base64,
        "original_base64": original_base64
    }


def _test_onnx_model(model_path: Path, frame: np.ndarray) -> dict:
    """ONNXモデルでテスト"""
    import time
    import cv2
    import base64
    
    original_h, original_w = frame.shape[:2]
    input_size = (320, 240)
    
    # モデルロード
    load_start = time.perf_counter()
    net = cv2.dnn.readNetFromONNX(str(model_path))
    
    # CUDAが使えるか確認
    try:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
        backend = "CUDA FP16"
    except:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        backend = "CPU"
    
    load_time = (time.perf_counter() - load_start) * 1000
    print(f"[Test] ONNX model loaded in {load_time:.1f}ms, backend: {backend}")
    
    # 前処理
    resized = cv2.resize(frame, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    # 正規化（ImageNet）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (rgb.astype(np.float32) / 255.0 - mean) / std
    
    # blob作成
    blob = normalized.transpose(2, 0, 1).reshape(1, 3, 240, 320).astype(np.float32)
    
    # 推論
    net.setInput(blob)
    infer_start = time.perf_counter()
    output = net.forward()
    inference_time = (time.perf_counter() - infer_start) * 1000
    
    # 後処理
    pred = np.argmax(output[0], axis=0).astype(np.uint8)
    
    # カラーマスク作成
    color_mask = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    color_mask[pred == 1] = [0, 255, 0]  # ROAD = 緑
    color_mask[pred == 2] = [0, 0, 255]  # MYCAR = 赤
    
    # 元サイズにリサイズ
    color_mask = cv2.resize(color_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # オーバーレイ
    overlay = cv2.addWeighted(frame, 0.6, color_mask, 0.4, 0)
    
    # ROAD割合計算
    road_pixels = np.sum(pred == 1)
    total_pixels = pred.size
    road_percentage = (road_pixels / total_pixels) * 100
    
    # Base64エンコード
    _, buffer = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    
    _, orig_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    original_base64 = base64.b64encode(orig_buffer).decode('utf-8')
    
    return {
    "model_name": model_path.name,
    "model_type": "onnx",
    "backend": backend,
    "load_time_ms": round(load_time, 1),
    "inference_time_ms": round(inference_time, 1),
    "input_size": list(input_size),
    "road_percentage": round(road_percentage, 2),
    "result_base64": result_base64,
    "original_base64": original_base64
    }


@router.get("/debug/{dataset_name}/masks")
def get_debug_masks(dataset_name: str, limit: int = 5):
    """学習データのマスクを確認用に取得
    
    Args:
        dataset_name: データセット名
        limit: 取得する画像数
    
    Returns:
        images: 元画像、OneFormerマスク、学習用マスクのリスト
    """
    import cv2
    import base64
    from ..core.dataset_manager import dataset_manager
    from ..core.ade20k_full_labels import ADE20K_ID_TO_NAME
    
    # データセット情報取得
    info = dataset_manager.get_dataset_info(dataset_name)
    if "error" in info:
        raise HTTPException(status_code=404, detail=info["error"])
    
    dataset_dir = Path(info["path"])
    training_data_dir = dataset_dir / "training_data" / "train"
    
    # 学習データがない場合
    if not training_data_dir.exists():
        raise HTTPException(status_code=400, detail="Training data not prepared. Run 'Start Training' first.")
    
    # 画像一覧取得
    images_result = dataset_manager.get_images(dataset_name)
    images_with_seg = [
        img for img in images_result["images"]
        if img["has_segmentation"]
    ][:limit]
    
    results = []
    
    for img_info in images_with_seg:
        try:
            img_path = Path(img_info["path"])
            seg_path = Path(img_info["seg_path"])
            label_path = training_data_dir / "labels" / f"{img_path.stem}.png"
            
            # 元画像
            original = cv2.imread(str(img_path))
            if original is None:
                continue
            _, orig_buf = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 80])
            original_b64 = base64.b64encode(orig_buf).decode('utf-8')
            
            # OneFormerマスク（カラー化）
            seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
            if seg_mask is None:
                continue
            
            # OneFormerマスクをカラー化（各クラスに色付け）
            seg_colored = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
            unique_ids = np.unique(seg_mask)
            for uid in unique_ids:
                # クラスIDに応じた色を生成
                np.random.seed(uid)
                color = np.random.randint(50, 255, 3).tolist()
                seg_colored[seg_mask == uid] = color
            
            _, seg_buf = cv2.imencode('.jpg', seg_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
            seg_b64 = base64.b64encode(seg_buf).decode('utf-8')
            
            # 学習用マスク（カラー化）
            label_b64 = None
            label_stats = None
            if label_path.exists():
                label_mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
                if label_mask is not None:
                    # 0: 黒, 1: 緑(ROAD), 2: 赤(MYCAR)
                    label_colored = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.uint8)
                    label_colored[label_mask == 1] = [0, 255, 0]  # ROAD = 緑
                    label_colored[label_mask == 2] = [0, 0, 255]  # MYCAR = 赤
                    
                    _, label_buf = cv2.imencode('.jpg', label_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    label_b64 = base64.b64encode(label_buf).decode('utf-8')
                    
                    # 統計情報
                    total = label_mask.size
                    label_stats = {
                        "other_pct": round(np.sum(label_mask == 0) / total * 100, 1),
                        "road_pct": round(np.sum(label_mask == 1) / total * 100, 1),
                        "mycar_pct": round(np.sum(label_mask == 2) / total * 100, 1),
                    }
            
            # OneFormerマスクのクラス情報
            class_info = []
            for uid in sorted(unique_ids):
                name = ADE20K_ID_TO_NAME.get(uid, f"unknown_{uid}")
                pct = round(np.sum(seg_mask == uid) / seg_mask.size * 100, 1)
                class_info.append({"id": int(uid), "name": name, "percentage": pct})
            
            results.append({
                "filename": img_path.name,
                "original_base64": original_b64,
                "seg_base64": seg_b64,
                "label_base64": label_b64,
                "label_stats": label_stats,
                "class_info": class_info,
            })
            
        except Exception as e:
            print(f"[Debug] Error processing {img_info['path']}: {e}")
            continue
    
    # ROADマッピング情報も返す
    from ..core.road_mapping import get_road_mapping
    road_mapping = get_road_mapping()
    
    return {
        "dataset": dataset_name,
        "count": len(results),
        "road_labels": road_mapping.get_road_labels(),
        "images": results
    }


@router.post("/preview-mask")
def preview_training_mask(dataset_name: str, image_name: str):
    """指定画像の学習用マスクをプレビュー（学習前に確認）
    
    Args:
        dataset_name: データセット名
        image_name: 画像ファイル名
    
    Returns:
        元画像、OneFormerマスク、生成されるROADマスクのプレビュー
    """
    import cv2
    import base64
    from ..core.dataset_manager import dataset_manager
    from ..core.road_mapping import get_road_mapping
    from ..core.ade20k_full_labels import get_road_label_ids, ADE20K_ID_TO_NAME
    
    # データセット情報取得
    info = dataset_manager.get_dataset_info(dataset_name)
    if "error" in info:
        raise HTTPException(status_code=404, detail=info["error"])
    
    dataset_dir = Path(info["path"])
    
    # 画像を探す
    img_path = None
    seg_path = None
    for cam_id in [0, 1]:
        candidate = dataset_dir / "images" / f"camera_{cam_id}" / image_name
        if candidate.exists():
            img_path = candidate
            seg_path = dataset_dir / "segmentation" / f"{candidate.stem}_seg.png"
            break
    
    if img_path is None or not img_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_name}")
    
    if not seg_path.exists():
        raise HTTPException(status_code=400, detail=f"Segmentation not found for {image_name}")
    
    # ROADマッピング取得
    road_mapping = get_road_mapping()
    road_labels = road_mapping.get_road_labels()
    road_label_ids = get_road_label_ids(road_labels)
    
    # 元画像
    original = cv2.imread(str(img_path))
    _, orig_buf = cv2.imencode('.jpg', original, [cv2.IMWRITE_JPEG_QUALITY, 80])
    original_b64 = base64.b64encode(orig_buf).decode('utf-8')
    
    # OneFormerマスク
    seg_mask = cv2.imread(str(seg_path), cv2.IMREAD_GRAYSCALE)
    
    # OneFormerマスクをカラー化
    seg_colored = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3), dtype=np.uint8)
    unique_ids = np.unique(seg_mask)
    for uid in unique_ids:
        np.random.seed(uid)
        color = np.random.randint(50, 255, 3).tolist()
        seg_colored[seg_mask == uid] = color
    
    _, seg_buf = cv2.imencode('.jpg', seg_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
    seg_b64 = base64.b64encode(seg_buf).decode('utf-8')
    
    # ROADマスクを生成（プレビュー）
    road_mask = np.zeros_like(seg_mask, dtype=np.uint8)
    for label_id in road_label_ids:
        road_mask[seg_mask == label_id] = 1  # ROAD
    
    # ROADマスクをカラー化
    road_colored = np.zeros((road_mask.shape[0], road_mask.shape[1], 3), dtype=np.uint8)
    road_colored[road_mask == 1] = [0, 255, 0]  # ROAD = 緑
    
    _, road_buf = cv2.imencode('.jpg', road_colored, [cv2.IMWRITE_JPEG_QUALITY, 80])
    road_b64 = base64.b64encode(road_buf).decode('utf-8')
    
    # オーバーレイ画像も作成
    overlay = cv2.addWeighted(original, 0.6, road_colored, 0.4, 0)
    _, overlay_buf = cv2.imencode('.jpg', overlay, [cv2.IMWRITE_JPEG_QUALITY, 80])
    overlay_b64 = base64.b64encode(overlay_buf).decode('utf-8')
    
    # クラス情報
    class_info = []
    for uid in sorted(unique_ids):
        name = ADE20K_ID_TO_NAME.get(uid, f"unknown_{uid}")
        pct = round(np.sum(seg_mask == uid) / seg_mask.size * 100, 1)
        is_road = uid in road_label_ids
        class_info.append({
            "id": int(uid), 
            "name": name, 
            "percentage": pct,
            "is_road": is_road
        })
    
    # 統計
    total = road_mask.size
    road_pct = round(np.sum(road_mask == 1) / total * 100, 1)
    
    return {
        "image_name": image_name,
        "original_base64": original_b64,
        "seg_base64": seg_b64,
        "road_mask_base64": road_b64,
        "overlay_base64": overlay_b64,
        "road_labels": road_labels,
        "road_label_ids": list(road_label_ids),
        "road_percentage": road_pct,
        "class_info": class_info,
    }
