"""ROAD mapping API endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from ..core.road_mapping import get_road_mapping

router = APIRouter()


class ToggleRequest(BaseModel):
    """Request body for toggle endpoint."""
    label_name: str


class SetRoadRequest(BaseModel):
    """Request body for set_road endpoint."""
    label_name: str
    is_road: bool


@router.get("/road-mapping")
def get_mapping():
    """Get current ROAD mapping."""
    mapping = get_road_mapping()
    return {
        "road_labels": mapping.get_road_labels(),
        "stats": mapping.get_stats()
    }


@router.post("/road-mapping/toggle")
def toggle_road(request: ToggleRequest):
    """Toggle ROAD status for a label."""
    mapping = get_road_mapping()
    new_state = mapping.toggle_road(request.label_name)
    
    return {
        "label_name": request.label_name,
        "is_road": new_state,
        "road_labels": mapping.get_road_labels()
    }


@router.post("/road-mapping/set")
def set_road(request: SetRoadRequest):
    """Set ROAD status for a label."""
    mapping = get_road_mapping()
    mapping.set_road(request.label_name, request.is_road)
    mapping.save()
    
    return {
        "label_name": request.label_name,
        "is_road": request.is_road,
        "road_labels": mapping.get_road_labels()
    }


@router.get("/road-mapping/check/{label_name}")
def check_road(label_name: str):
    """Check if a label is marked as ROAD."""
    mapping = get_road_mapping()
    
    return {
        "label_name": label_name,
        "is_road": mapping.is_road(label_name)
    }


@router.post("/road-mapping/save")
def save_mapping():
    """Save mapping to file."""
    mapping = get_road_mapping()
    mapping.save()
    
    return {
        "status": "saved",
        "road_labels": mapping.get_road_labels()
    }


@router.post("/road-mapping/reset")
def reset_mapping():
    """Reset mapping to defaults."""
    mapping = get_road_mapping()
    mapping._init_default_mapping()
    mapping.save()
    
    return {
        "status": "reset",
        "road_labels": mapping.get_road_labels()
    }
