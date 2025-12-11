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
