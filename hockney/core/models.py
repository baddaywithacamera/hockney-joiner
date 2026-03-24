"""
models.py — Shared data classes used across core and UI layers.

ImagePlacement and PlacementSnapshot live here so that:
  - core/placement.py can produce ImagePlacement objects
  - ui/tray_view.py can consume them
  - neither layer imports from the other
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ImagePlacement:
    """Mutable placement state for one image on the canvas."""
    image_id: str
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0
    z_order: int = 0
    auto_x: float = 0.0
    auto_y: float = 0.0
    auto_rotation: float = 0.0

    def reset_to_auto(self):
        self.x = self.auto_x
        self.y = self.auto_y
        self.rotation = self.auto_rotation

    def as_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "x": self.x, "y": self.y,
            "rotation": self.rotation,
            "z_order": self.z_order,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ImagePlacement":
        return cls(
            image_id=d["image_id"],
            x=d.get("x", 0.0), y=d.get("y", 0.0),
            rotation=d.get("rotation", 0.0),
            z_order=d.get("z_order", 0),
        )


@dataclass
class PlacementSnapshot:
    """A moment-in-time snapshot of one image's placement (for undo/redo)."""
    image_id: str
    x: float
    y: float
    rotation: float
    z_order: int
    removed: bool = False
