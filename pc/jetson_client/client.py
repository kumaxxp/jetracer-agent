"""WebSocket + HTTP client for communicating with Jetson server."""

from __future__ import annotations

import asyncio
import logging
import time

import websockets
import httpx

from shared.protocol.messages import WSMessage, WSEnvelope

logger = logging.getLogger(__name__)


class JetsonClient:
    """Manages connections to the Jetson server.

    Provides:
    - WebSocket connection for real-time sensor data and commands
    - HTTP client for REST API calls (calibration, config, etc.)
    """

    def __init__(self, jetson_host: str = "jetson", http_port: int = 8000, ws_port: int = 8765):
        self.jetson_host = jetson_host
        self.http_port = http_port
        self.ws_port = ws_port

        self._sensor_ws: websockets.WebSocketClientProtocol | None = None
        self._command_ws: websockets.WebSocketClientProtocol | None = None
        self._http: httpx.AsyncClient | None = None

        self._connected = False
        self._reconnect_interval = 2.0
        self._on_message_callbacks: list = []

        self._heartbeat_task: asyncio.Task | None = None
        self._receive_task: asyncio.Task | None = None

    @property
    def http_base_url(self) -> str:
        return f"http://{self.jetson_host}:{self.http_port}"

    @property
    def sensor_ws_url(self) -> str:
        return f"ws://{self.jetson_host}:{self.http_port}/ws/sensors"

    @property
    def command_ws_url(self) -> str:
        return f"ws://{self.jetson_host}:{self.http_port}/ws/commands"

    @property
    def is_connected(self) -> bool:
        return self._connected

    def on_message(self, callback):
        """Register a callback for incoming sensor messages.

        callback(envelope: WSEnvelope) -> None
        """
        self._on_message_callbacks.append(callback)

    async def connect(self):
        """Connect to Jetson server (WebSocket + HTTP)."""
        logger.info(f"Connecting to Jetson at {self.jetson_host}...")

        self._http = httpx.AsyncClient(base_url=self.http_base_url, timeout=10.0)

        # Connect sensor WebSocket
        try:
            self._sensor_ws = await websockets.connect(self.sensor_ws_url)
            logger.info(f"Sensor WS connected: {self.sensor_ws_url}")
        except Exception as e:
            logger.error(f"Sensor WS connection failed: {e}")
            raise

        # Connect command WebSocket
        try:
            self._command_ws = await websockets.connect(self.command_ws_url)
            logger.info(f"Command WS connected: {self.command_ws_url}")
        except Exception as e:
            logger.error(f"Command WS connection failed: {e}")
            raise

        self._connected = True

        # Start background tasks
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        logger.info("Jetson connection established")

    async def disconnect(self):
        """Disconnect from Jetson server."""
        self._connected = False

        if self._receive_task:
            self._receive_task.cancel()
        if self._heartbeat_task:
            self._heartbeat_task.cancel()

        if self._sensor_ws:
            await self._sensor_ws.close()
        if self._command_ws:
            await self._command_ws.close()
        if self._http:
            await self._http.aclose()

        logger.info("Disconnected from Jetson")

    async def send_control(self, steering: float, throttle: float):
        """Send a control command to Jetson."""
        msg = WSMessage.control_cmd(steering=steering, throttle=throttle)
        await self._send_command(msg)

    async def send_emergency_stop(self, reason: str = "pc_command"):
        """Send emergency stop to Jetson."""
        msg = WSMessage.emergency_stop()
        msg.payload["reason"] = reason
        await self._send_command(msg)

    async def send_mode_change(self, mode: str):
        """Send mode change to Jetson."""
        msg = WSMessage.mode_change(mode=mode)
        await self._send_command(msg)

    async def http_get(self, path: str) -> dict:
        """HTTP GET to Jetson API."""
        resp = await self._http.get(path)
        resp.raise_for_status()
        return resp.json()

    async def http_post(self, path: str, json: dict = None) -> dict:
        """HTTP POST to Jetson API."""
        resp = await self._http.post(path, json=json)
        resp.raise_for_status()
        return resp.json()

    async def get_status(self) -> dict:
        """Get Jetson server status."""
        return await self.http_get("/status")

    async def get_snapshot(self, camera_id: int = 0) -> bytes:
        """Get a single JPEG snapshot."""
        resp = await self._http.get(f"/snapshot/{camera_id}")
        resp.raise_for_status()
        return resp.content

    async def _send_command(self, envelope: WSEnvelope):
        """Send a command via the command WebSocket."""
        if not self._command_ws:
            logger.warning("Command WS not connected")
            return
        try:
            await self._command_ws.send(envelope.to_json_bytes())
        except Exception as e:
            logger.error(f"Failed to send command: {e}")
            self._connected = False

    async def _receive_loop(self):
        """Receive sensor data from Jetson."""
        while self._connected:
            try:
                data = await self._sensor_ws.recv()
                if isinstance(data, str):
                    data = data.encode()
                envelope = WSEnvelope.from_json_bytes(data)

                for cb in self._on_message_callbacks:
                    try:
                        cb(envelope)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

            except websockets.ConnectionClosed:
                logger.warning("Sensor WS connection lost")
                self._connected = False
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                await asyncio.sleep(0.1)

    async def _heartbeat_loop(self):
        """Send heartbeat to Jetson every second."""
        start = time.time()
        while self._connected:
            uptime = time.time() - start
            msg = WSMessage.heartbeat(uptime_s=uptime)
            await self._send_command(msg)
            await asyncio.sleep(1.0)
