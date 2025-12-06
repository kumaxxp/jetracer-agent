"""ROAD mapping management for drivable area detection."""

import json
from pathlib import Path
from typing import Dict, Set, Optional


# Default labels likely to be drivable surfaces
DEFAULT_ROAD_LABELS = {
    "road", "floor", "path", "sidewalk", "runway",
    "dirt track", "field", "sand"
}


class ROADMapping:
    """Manages mapping between ADE20K labels and ROAD (drivable) attribute."""

    def __init__(self, mapping_file: Optional[str] = None):
        """
        Initialize ROAD mapping.

        Args:
            mapping_file: Path to JSON mapping file (optional)
        """
        self.mapping_file = Path(mapping_file) if mapping_file else None
        self.mapping: Dict[str, bool] = {}  # label_name -> is_road
        self.road_labels: Set[str] = set()

        if self.mapping_file and self.mapping_file.exists():
            self.load()
        else:
            self._init_default_mapping()

    def _init_default_mapping(self):
        """Initialize with default ROAD labels."""
        from .ade20k_labels import ADE20K_LABELS
        
        for label_id, label_name in ADE20K_LABELS.items():
            self.mapping[label_name] = label_name in DEFAULT_ROAD_LABELS

        self.road_labels = {label for label, is_road in self.mapping.items() if is_road}

    def toggle_road(self, label_name: str) -> bool:
        """
        Toggle ROAD attribute for a label.

        Args:
            label_name: Name of the label

        Returns:
            New ROAD state (True if now ROAD, False otherwise)
        """
        if label_name not in self.mapping:
            self.mapping[label_name] = False

        self.mapping[label_name] = not self.mapping[label_name]

        if self.mapping[label_name]:
            self.road_labels.add(label_name)
        else:
            self.road_labels.discard(label_name)

        # Auto-save
        if self.mapping_file:
            self.save()

        return self.mapping[label_name]

    def set_road(self, label_name: str, is_road: bool):
        """Set ROAD attribute for a label."""
        self.mapping[label_name] = is_road

        if is_road:
            self.road_labels.add(label_name)
        else:
            self.road_labels.discard(label_name)

    def is_road(self, label_name: str) -> bool:
        """Check if a label is marked as ROAD."""
        return self.mapping.get(label_name, False)

    def get_road_labels(self) -> list:
        """Get list of all ROAD label names."""
        return sorted(list(self.road_labels))

    def save(self, filepath: str = None):
        """Save mapping to JSON file."""
        save_path = Path(filepath) if filepath else self.mapping_file
        if not save_path:
            raise ValueError("No save path specified")

        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "road_labels": sorted(list(self.road_labels)),
            "mapping": self.mapping
        }

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[ROADMapping] Saved to {save_path}")

    def load(self, filepath: str = None):
        """Load mapping from JSON file."""
        load_path = Path(filepath) if filepath else self.mapping_file
        if not load_path or not load_path.exists():
            print(f"[ROADMapping] File not found: {load_path}, using defaults")
            self._init_default_mapping()
            return

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.mapping = data.get("mapping", {})
        self.road_labels = set(data.get("road_labels", []))

        print(f"[ROADMapping] Loaded from {load_path}")

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about the mapping."""
        return {
            "total_labels": len(self.mapping),
            "road_labels": len(self.road_labels),
            "non_road_labels": len(self.mapping) - len(self.road_labels)
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "road_labels": self.get_road_labels(),
            "stats": self.get_stats()
        }


# Singleton instance
_road_mapping: Optional[ROADMapping] = None


def get_road_mapping() -> ROADMapping:
    """Get the singleton ROADMapping instance."""
    global _road_mapping
    if _road_mapping is None:
        mapping_path = Path.home() / "jetracer_data" / "road_mapping.json"
        _road_mapping = ROADMapping(str(mapping_path))
    return _road_mapping
