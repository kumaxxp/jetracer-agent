"""FastAPI HTTP Server エントリーポイント"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from .routes import status, camera, analysis, control, stream, oneformer
from .core.camera_manager import camera_manager
from .config import config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動・終了処理"""
    # 起動時
    print("[Server] Starting JetRacer HTTP API Server...")
    camera_manager.start(
        width=config.camera_width,
        height=config.camera_height,
        fps=config.camera_fps
    )
    print(f"[Server] Camera ready: {camera_manager.is_ready()}")

    yield

    # 終了時
    print("[Server] Shutting down...")
    camera_manager.stop()


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


@app.get("/")
def root():
    """API ルート"""
    return {
        "name": "JetRacer API",
        "version": "1.0.0",
        "endpoints": [
            "GET  /status             - システム状態",
            "POST /capture            - カメラ画像取得",
            "POST /analyze            - 統合解析",
            "POST /control            - 車両制御",
            "POST /stop               - 緊急停止",
            "GET  /stream/{camera_id} - MJPEGストリーム",
            "GET  /snapshot           - 単一JPEG画像",
            "POST /oneformer/{camera_id} - OneFormerセグメンテーション"
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
