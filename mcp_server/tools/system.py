"""システムチェックツール"""

import subprocess
from pathlib import Path

# 設定値
DARK_THRESHOLD = 20
DIM_THRESHOLD = 50
BRIGHT_THRESHOLD = 200


async def check_camera() -> dict:
    """カメラの接続状態を確認し、接続されていれば1フレーム取得する
    
    Returns:
        connected: カメラ接続状態
        resolution: 解像度（接続時）
        frame_path: 一時保存した画像パス（接続時）
        error: エラーメッセージ（エラー時）
    """
    try:
        import cv2
        
        # CSIカメラを試行
        gst_pipeline = (
            "nvarguscamerasrc sensor-id=0 ! "
            "video/x-raw(memory:NVMM),width=640,height=480,framerate=15/1 ! "
            "nvvidconv ! video/x-raw,format=BGRx ! "
            "videoconvert ! video/x-raw,format=BGR ! appsink drop=1"
        )
        
        cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        
        if not cap.isOpened():
            # USBカメラにフォールバック
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            return {"connected": False, "error": "カメラが見つかりません"}
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return {"connected": False, "error": "フレーム取得失敗"}
        
        # フレームを一時保存
        temp_path = "/tmp/yana_startup_frame.jpg"
        cv2.imwrite(temp_path, frame)
        
        return {
            "connected": True,
            "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
            "frame_path": temp_path
        }
        
    except Exception as e:
        return {"connected": False, "error": str(e)}


async def analyze_frame(image_path: str) -> dict:
    """画像の明るさ、コントラスト、エッジ量を分析する
    
    Args:
        image_path: 画像ファイルパス
    
    Returns:
        brightness: 平均明るさ (0-255)
        contrast: コントラスト（標準偏差）
        edge_density: エッジ密度（%）
        is_very_dark: 真っ暗判定
        is_dark: 暗め判定
        is_bright: 明るすぎ判定
        has_texture: テクスチャあり判定
    """
    try:
        import cv2
        import numpy as np
        
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"画像を読み込めません: {image_path}"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        brightness = float(gray.mean())
        contrast = float(gray.std())
        
        # エッジ量（テクスチャの指標）
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum() / (edges.size * 255) * 100)
        
        return {
            "brightness": round(brightness, 1),
            "contrast": round(contrast, 1),
            "edge_density": round(edge_density, 2),
            "is_very_dark": brightness < DARK_THRESHOLD,
            "is_dark": brightness < DIM_THRESHOLD,
            "is_bright": brightness > BRIGHT_THRESHOLD,
            "has_texture": edge_density > 3
        }
        
    except Exception as e:
        return {"error": str(e)}


async def check_system_resources() -> dict:
    """メモリとGPU状況を確認する
    
    Returns:
        memory_total_mb: 総メモリ
        memory_free_mb: 空きメモリ
        memory_ok: メモリ状況が正常か
        gpu_info: GPU情報
    """
    try:
        # メモリ
        result = subprocess.run(['free', '-m'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        mem = lines[1].split()
        total_mb = int(mem[1])
        used_mb = int(mem[2])
        free_mb = total_mb - used_mb
        
        # GPU（Jetson / NVIDIA）
        gpu_info = "不明"
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_used, gpu_total = result.stdout.strip().split(', ')
                gpu_info = f"{gpu_used}MB / {gpu_total}MB"
        except:
            pass
        
        return {
            "memory_total_mb": total_mb,
            "memory_free_mb": free_mb,
            "memory_ok": free_mb > 2000,
            "gpu_info": gpu_info
        }
        
    except Exception as e:
        return {"error": str(e)}


async def check_model_files() -> dict:
    """必要なモデルファイルの存在を確認する
    
    Returns:
        各モデルの存在状況
    """
    models = {
        "llm": Path.home() / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "segmentation": Path("models/road_segmentation.onnx"),
        "yolo": Path("models/yolov8n.pt")
    }
    
    # プロジェクトルートからの相対パスも試す
    project_root = Path(__file__).parent.parent
    
    results = {}
    for name, path in models.items():
        exists = path.exists()
        if not exists and not path.is_absolute():
            # プロジェクトルートからの相対パスを試す
            alt_path = project_root / path
            exists = alt_path.exists()
            if exists:
                path = alt_path
        
        results[name] = {
            "path": str(path),
            "exists": exists
        }
    
    return results
