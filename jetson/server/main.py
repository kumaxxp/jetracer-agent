"""Jetson server entry point.

Combines the existing HTTP API (camera, control, calibration, stream)
with new WebSocket sensor streaming and command receiving.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path for shared imports
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from jetson.server.config import config
from jetson.server.ws.sensor_stream import SensorStreamServer
from jetson.server.ws.command_receiver import CommandReceiver

# Reuse existing core modules from http_server
from http_server.core.camera_manager import camera_manager
from http_server.core.vehicle_controller import vehicle_controller

# Reuse existing routes
from http_server.routes import status, camera, control, stream, calibration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger(__name__)

# WebSocket components
sensor_stream = SensorStreamServer(config)
command_receiver = CommandReceiver()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown."""
    logger.info("Starting Jetson Server (HTTP + WebSocket)...")

    # Start cameras
    results = camera_manager.start_all(
        width=config.camera_width,
        height=config.camera_height,
        fps=config.camera_fps,
        camera_ids=config.camera_ids,
    )
    for cid, ok in results.items():
        logger.info(f"Camera {cid}: {'ready' if ok else 'failed'}")

    # Load calibration
    try:
        camera_manager.load_calibration_from_json()
        for cid in config.camera_ids:
            if camera_manager.has_calibration(cid):
                camera_manager.set_undistort_enabled(cid, True)
                logger.info(f"Camera {cid}: Undistortion enabled")
    except Exception as e:
        logger.error(f"Failed to load calibration: {e}")

    # Wire up WebSocket components
    sensor_stream.set_camera_manager(camera_manager)
    sensor_stream.set_vehicle_controller(vehicle_controller)
    command_receiver.set_vehicle_controller(vehicle_controller)

    # Start sensor streaming in background
    streaming_task = asyncio.create_task(sensor_stream.start_streaming())

    yield

    # Shutdown
    logger.info("Shutting down Jetson Server...")
    await sensor_stream.stop_streaming()
    streaming_task.cancel()
    camera_manager.stop()
    vehicle_controller.stop()


app = FastAPI(
    title="JetRacer Jetson API",
    description="Jetson sensor/actuator server with WebSocket streaming",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# HTTP routes (reuse existing)
app.include_router(status.router, tags=["status"])
app.include_router(camera.router, tags=["camera"])
app.include_router(control.router, tags=["control"])
app.include_router(stream.router, tags=["stream"])
app.include_router(calibration.router, tags=["calibration"])


# WebSocket endpoints
@app.websocket("/ws/sensors")
async def ws_sensors(ws: WebSocket):
    """WebSocket endpoint for sensor data streaming (Jetson -> PC)."""
    await sensor_stream.handle_client(ws)


@app.websocket("/ws/commands")
async def ws_commands(ws: WebSocket):
    """WebSocket endpoint for receiving commands (PC -> Jetson)."""
    await command_receiver.handle_client(ws)


@app.get("/")
def root():
    return {
        "name": "JetRacer Jetson Server",
        "version": "2.0.0",
        "ws_clients": sensor_stream.client_count,
        "mode": command_receiver.current_mode.value,
        "endpoints": {
            "http": [
                "GET  /status",
                "POST /capture",
                "POST /control",
                "POST /stop",
                "GET  /stream/{camera_id}",
                "GET  /snapshot/{camera_id}",
                "GET  /calibration/status",
            ],
            "websocket": [
                "WS /ws/sensors   - Sensor data stream (Jetson -> PC)",
                "WS /ws/commands  - Command receiver (PC -> Jetson)",
            ],
        },
    }


def main():
    uvicorn.run(app, host=config.host, port=config.http_port, log_level="info")


if __name__ == "__main__":
    main()
