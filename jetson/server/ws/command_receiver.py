"""WebSocket command receiver.

Handles incoming commands from PC (control, emergency stop, mode change).
"""

from __future__ import annotations

import logging
import time

from fastapi import WebSocket, WebSocketDisconnect

from shared.protocol.messages import WSEnvelope
from shared.protocol.command_types import ControlCommand, EmergencyStop, ModeChange, DrivingMode

logger = logging.getLogger(__name__)


class CommandReceiver:
    """Receives and dispatches commands from PC via WebSocket."""

    def __init__(self):
        self._vehicle_controller = None
        self._safety_controller = None
        self._last_heartbeat: float = 0.0
        self._current_mode: DrivingMode = DrivingMode.IDLE

    def set_vehicle_controller(self, vc):
        self._vehicle_controller = vc

    def set_safety_controller(self, sc):
        self._safety_controller = sc

    @property
    def last_heartbeat(self) -> float:
        return self._last_heartbeat

    @property
    def current_mode(self) -> DrivingMode:
        return self._current_mode

    async def handle_client(self, ws: WebSocket):
        """Handle incoming commands from a PC client."""
        await ws.accept()
        client_addr = ws.client.host if ws.client else "unknown"
        logger.info(f"[CMD] PC command client connected: {client_addr}")

        try:
            while True:
                data = await ws.receive_bytes()
                envelope = WSEnvelope.from_json_bytes(data)
                self._dispatch(envelope)
        except WebSocketDisconnect:
            logger.info(f"[CMD] PC command client disconnected: {client_addr}")

    def _dispatch(self, envelope: WSEnvelope):
        """Dispatch a received command."""
        msg_type = envelope.type
        payload = envelope.payload

        if msg_type == "heartbeat":
            self._last_heartbeat = time.time()

        elif msg_type == "control_cmd":
            self._handle_control(payload)

        elif msg_type == "emergency_stop":
            self._handle_emergency_stop(payload)

        elif msg_type == "mode_change":
            self._handle_mode_change(payload)

        else:
            logger.warning(f"[CMD] Unknown message type: {msg_type}")

    def _handle_control(self, payload: dict):
        """Apply control command with safety validation."""
        cmd = ControlCommand(**payload)

        # Safety check
        if self._safety_controller and not self._safety_controller.validate_command(cmd):
            logger.warning("[CMD] Control command rejected by safety controller")
            return

        if self._vehicle_controller:
            self._vehicle_controller.set_steering(cmd.steering)
            self._vehicle_controller.set_throttle(cmd.throttle)

    def _handle_emergency_stop(self, payload: dict):
        """Execute emergency stop."""
        reason = payload.get("reason", "pc_command")
        logger.warning(f"[CMD] Emergency stop received: {reason}")

        if self._vehicle_controller:
            self._vehicle_controller.stop()

        self._current_mode = DrivingMode.IDLE

    def _handle_mode_change(self, payload: dict):
        """Change driving mode."""
        mode = DrivingMode(payload["mode"])
        logger.info(f"[CMD] Mode change: {self._current_mode} -> {mode}")
        self._current_mode = mode

        if mode == DrivingMode.IDLE and self._vehicle_controller:
            self._vehicle_controller.stop()
