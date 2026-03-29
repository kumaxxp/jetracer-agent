"""Data models for Jetson client responses."""

from __future__ import annotations

from pydantic import BaseModel


class JetsonStatus(BaseModel):
    """Status response from Jetson /status endpoint."""

    cameras: dict[str, bool] = {}
    vehicle_connected: bool = False
    lidar_connected: bool = False
    ws_clients: int = 0
    mode: str = "idle"
