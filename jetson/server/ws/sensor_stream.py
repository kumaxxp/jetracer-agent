"""WebSocket sensor streaming server.

Streams camera frames, LiDAR scans, YOLO detections, and vehicle status
to connected PC clients.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import TYPE_CHECKING

import cv2
from fastapi import WebSocket, WebSocketDisconnect

from shared.protocol.messages import WSMessage, WSEnvelope

if TYPE_CHECKING:
    from jetson.server.config import JetsonServerConfig

logger = logging.getLogger(__name__)


class SensorStreamServer:
    """Manages WebSocket connections and streams sensor data to PC."""

    def __init__(self, config: JetsonServerConfig):
        self.config = config
        self._clients: set[WebSocket] = set()
        self._running = False
        self._start_time = time.time()
        self._camera_manager = None
        self._vehicle_controller = None
        self._lidar_manager = None

    def set_camera_manager(self, camera_manager):
        self._camera_manager = camera_manager

    def set_vehicle_controller(self, vehicle_controller):
        self._vehicle_controller = vehicle_controller

    def set_lidar_manager(self, lidar_manager):
        self._lidar_manager = lidar_manager

    async def handle_client(self, ws: WebSocket):
        """Handle a new WebSocket client connection."""
        await ws.accept()
        self._clients.add(ws)
        client_addr = ws.client.host if ws.client else "unknown"
        logger.info(f"[WS] PC client connected: {client_addr} (total: {len(self._clients)})")

        try:
            # Keep connection alive, receive commands handled by command_receiver
            while True:
                # We only read here to detect disconnect; commands are handled elsewhere
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            self._clients.discard(ws)
            logger.info(f"[WS] PC client disconnected: {client_addr} (total: {len(self._clients)})")

    async def broadcast(self, message: WSEnvelope):
        """Send a message to all connected clients."""
        if not self._clients:
            return

        data = message.to_json_bytes()
        disconnected = set()

        for ws in self._clients:
            try:
                await ws.send_bytes(data)
            except Exception:
                disconnected.add(ws)

        self._clients -= disconnected

    async def start_streaming(self):
        """Start all sensor streaming tasks."""
        self._running = True
        self._start_time = time.time()

        tasks = [
            asyncio.create_task(self._stream_cameras()),
            asyncio.create_task(self._stream_vehicle_status()),
            asyncio.create_task(self._stream_heartbeat()),
        ]

        # LiDAR streaming added when lidar_manager is set
        if self._lidar_manager is not None:
            tasks.append(asyncio.create_task(self._stream_lidar()))

        logger.info(f"[WS] Sensor streaming started ({len(tasks)} streams)")
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop_streaming(self):
        """Stop streaming."""
        self._running = False

    async def _stream_cameras(self):
        """Stream camera frames as JPEG base64."""
        interval = 1.0 / self.config.stream_camera_fps

        while self._running:
            if not self._clients or self._camera_manager is None:
                await asyncio.sleep(interval)
                continue

            for camera_id in self.config.camera_ids:
                if not self._camera_manager.is_ready(camera_id):
                    continue

                frame = self._camera_manager.read(camera_id)
                if frame is None:
                    continue

                _, jpeg = cv2.imencode(
                    ".jpg", frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self.config.jpeg_quality],
                )
                jpeg_b64 = base64.b64encode(jpeg.tobytes()).decode("ascii")

                msg = WSMessage.camera_frame(
                    camera_id=camera_id,
                    jpeg_b64=jpeg_b64,
                    width=frame.shape[1],
                    height=frame.shape[0],
                )
                await self.broadcast(msg)

            await asyncio.sleep(interval)

    async def _stream_lidar(self):
        """Stream LiDAR scan data."""
        interval = 1.0 / self.config.stream_lidar_hz

        while self._running:
            if not self._clients or self._lidar_manager is None:
                await asyncio.sleep(interval)
                continue

            scan = self._lidar_manager.get_latest_scan()
            if scan is not None:
                points = [
                    {"angle": p.angle, "distance": p.distance, "quality": p.quality}
                    for p in scan.points
                ]
                msg = WSMessage.lidar_scan(points=points, rpm=scan.rpm)
                await self.broadcast(msg)

            await asyncio.sleep(interval)

    async def _stream_vehicle_status(self):
        """Stream vehicle status."""
        interval = 1.0 / self.config.stream_status_hz

        while self._running:
            if not self._clients or self._vehicle_controller is None:
                await asyncio.sleep(interval)
                continue

            status = self._vehicle_controller.get_status()
            msg = WSMessage.vehicle_status(
                steering=status["steering"],
                throttle=status["throttle"],
                connected=status["connected"],
            )
            await self.broadcast(msg)
            await asyncio.sleep(interval)

    async def _stream_heartbeat(self):
        """Send heartbeat every second."""
        while self._running:
            uptime = time.time() - self._start_time
            msg = WSMessage.heartbeat(uptime_s=uptime)
            await self.broadcast(msg)
            await asyncio.sleep(1.0)

    @property
    def client_count(self) -> int:
        return len(self._clients)
