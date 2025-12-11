"""学習管理API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import traceback

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
        from pathlib import Path
        
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
