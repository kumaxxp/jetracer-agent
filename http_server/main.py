"""FastAPI HTTP Server エントリーポイント"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from .routes import status, camera, analysis, control, stream, oneformer, road_mapping, calibration, navigation, distance_grid, dataset, training, benchmark
from .core.camera_manager import camera_manager
from .core.sensor_capabilities import sensor_capabilities
from .core.distance_grid import distance_grid_manager
from .config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動・終了処理"""
    # 起動時
    print("[Server] Starting JetRacer HTTP API Server...")
    
    # cuDNN設定（複数モデルの競合を防止）
    try:
        import torch
        if torch.cuda.is_available():
            # ベンチマークを無効化（競合の原因になる）
            torch.backends.cudnn.benchmark = False
            # 決定的な動作を有効化
            torch.backends.cudnn.deterministic = True
            print("[Server] cuDNN: benchmark=False, deterministic=True")
    except Exception as e:
        print(f"[Server] Warning: Could not configure cuDNN: {e}")
    
    # センサー capabilities をプローブ（カメラ起動前）
    print("[Server] Probing sensor capabilities...")
    try:
        probe_results = sensor_capabilities.initialize(camera_ids=[0, 1])
        for cid, ok in probe_results.items():
            modes = sensor_capabilities.get_modes(cid)
            sensor_name = sensor_capabilities.get_sensor_name(cid)
            print(f"[Server] Camera {cid}: {sensor_name}, {len(modes)} modes available")
    except Exception as e:
        print(f"[Server] Warning: Sensor probe failed: {e}")
    
    # 重要: 両方のモデルを先にロード（CUDAストリーム競合防止）
    print("[Server] Pre-loading models to avoid CUDA stream conflicts...")
    try:
        # 1. OneFormerモデルをロード
        print("[Server] Loading OneFormer model...")
        from .routes.oneformer import get_segmenter
        get_segmenter()
        print("[Server] OneFormer model loaded")
        
        # 2. Lightweightモデルをロード（新しいモジュール）
        print("[Server] Loading Lightweight model...")
        from .core.lightweight_segmentation import lightweight_segmentation
        if lightweight_segmentation.load():
            print(f"[Server] Lightweight model loaded on {lightweight_segmentation._device}")
        else:
            print("[Server] Lightweight model not found, skipping")
        
        print("[Server] All models pre-loaded successfully")
    except Exception as e:
        print(f"[Server] Warning: Model pre-loading failed: {e}")
        import traceback
        traceback.print_exc()

    # 複数カメラを起動（カメラ0, 1）
    results = camera_manager.start_all(
        width=config.camera_width,
        height=config.camera_height,
        fps=config.camera_fps,
        camera_ids=[0, 1]
    )
    for cid, ok in results.items():
        print(f"[Server] Camera {cid}: {'ready' if ok else 'failed'}")
    
    # キャリブレーションデータをロードして歪み補正を有効化
    try:
        # JSONから直接ロード（確実）
        camera_manager.load_calibration_from_json()
        
        # 歪み補正を有効化
        for cid in [0, 1]:
            if camera_manager.has_calibration(cid):
                camera_manager.set_undistort_enabled(cid, True)
                print(f"[Server] Camera {cid}: Undistortion enabled")
        
        # Distance Grid Managerのキャリブレーションも再読み込み
        distance_grid_manager.reload_calibration()
        
    except Exception as e:
        print(f"[Server] Failed to load calibration: {e}")
        import traceback
        traceback.print_exc()

    yield

    # 終了時
    print("[Server] Shutting down...")
    camera_manager.stop()  # 全カメラ停止


app = FastAPI(
    title="JetRacer API",
    description="JetRacer センサー・制御 HTTP API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS設定（PCからのアクセス許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ルーター登録
app.include_router(status.router, tags=["status"])
app.include_router(camera.router, tags=["camera"])
app.include_router(analysis.router, tags=["analysis"])
app.include_router(control.router, tags=["control"])
app.include_router(stream.router, tags=["stream"])
app.include_router(oneformer.router, tags=["oneformer"])
app.include_router(road_mapping.router, tags=["road-mapping"])
app.include_router(calibration.router, tags=["calibration"])
app.include_router(navigation.router, tags=["navigation"])
app.include_router(distance_grid.router, tags=["distance-grid"])
app.include_router(dataset.router, tags=["dataset"])
app.include_router(training.router, tags=["training"])
app.include_router(benchmark.router, prefix="/benchmark", tags=["benchmark"])


@app.get("/")
def root():
    """API ルート"""
    return {
        "name": "JetRacer API",
        "version": "1.0.0",
        "endpoints": [
            "GET  /status                - システム状態",
            "POST /capture               - カメラ画像取得",
            "GET  /capabilities          - センサー capabilities（全カメラ）",
            "GET  /capabilities/{id}     - センサー capabilities（カメラ指定）",
            "POST /capabilities/probe    - センサー capabilities 再取得",
            "POST /restart               - カメラ再起動（モード変更）",
            "GET  /settings/{id}         - カメラ設定取得",
            "POST /analyze               - 統合解析",
            "POST /control               - 車両制御",
            "POST /stop                  - 緊急停止",
            "GET  /stream                - MJPEGストリーム（デフォルト: カメラ0）",
            "GET  /stream/{id}           - MJPEGストリーム（カメラ指定）",
            "GET  /snapshot              - 単一JPEG（デフォルト: カメラ0）",
            "GET  /snapshot/{id}         - 単一JPEG（カメラ指定）",
            "POST /oneformer/{camera_id} - OneFormerセグメンテーション",
            "POST /oneformer/{camera_id}/label-at-position - クリック位置のラベル取得",
            "POST /oneformer/{camera_id}/toggle-road-at-position - クリック位置のROADトグル",
            "GET  /road-mapping - ROADマッピング取得",
            "POST /road-mapping/toggle - ROADラベルトグル",
            "GET  /calibration/status - キャリブレーション状態",
            "POST /calibration/detect/{id} - チェッカーボード検出",
            "POST /calibration/capture-stereo - ステレオ撮影",
            "POST /calibration/run - キャリブレーション実行",
            "GET  /navigation/situation - 状況分析",
            "POST /navigation/update-situation - 状況更新（セグメンテーション実行）",
            "POST /navigation/move - 移動コマンド（モック）",
            "GET  /navigation/status - ナビゲーション状態",
            "GET  /dataset/list - データセット一覧",
            "POST /dataset/create - データセット作成",
            "POST /dataset/select - データセット選択",
            "POST /dataset/{name}/add/{camera_id} - 画像追加",
            "POST /dataset/{name}/add-with-oneformer/{camera_id} - OneFormer付き画像追加",
            "POST /benchmark/camera - カメラベンチマーク",
            "POST /benchmark/segmentation - セグメンテーションベンチマーク",
            "GET  /benchmark/fps/{id} - リアルタイムFPS計測"
        ]
    }


def main():
    """サーバー起動"""
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
