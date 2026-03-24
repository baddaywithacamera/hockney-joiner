"""
models.py — Shared data classes used across core and UI layers.

ImagePlacement and PlacementSnapshot live here so that:
  - core/placement.py can produce ImagePlacement objects
  - ui/tray_view.py can consume them
  - neither layer imports from the other
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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


# ── Project types ─────────────────────────────────────────────────────────────

PROJECT_TYPES = ("perspective", "time_of_day", "seasonal")

SUBJECT_TYPES = ("landscape", "skylife", "urban", "indoor", "people")

# Slot names per project type
PERSPECTIVE_SLOTS = ("standing", "left", "right")
PERSPECTIVE_ADVANCED_SLOTS = ("down_low", "up_high")

TIME_SLOTS = ("morning", "afternoon", "evening", "night")

SEASON_SLOTS = ("spring", "summer", "fall", "winter")

SLOT_LABELS = {
    "standing": "Standing",
    "left": "Left Side",
    "right": "Right Side",
    "down_low": "Down Low",
    "up_high": "Up High",
    "morning": "Morning",
    "afternoon": "Afternoon",
    "evening": "Evening",
    "night": "Night",
    "spring": "Spring",
    "summer": "Summer",
    "fall": "Fall",
    "winter": "Winter",
}


@dataclass
class ReferenceImage:
    """One reference image assigned to a slot."""
    slot: str               # e.g. "standing", "morning", "spring"
    source_path: str        # absolute path to the image file
    image_id: str = ""      # populated after loading into ImageStore

    def as_dict(self) -> dict:
        return {"slot": self.slot, "source_path": self.source_path, "id": self.image_id}

    @classmethod
    def from_dict(cls, d: dict) -> "ReferenceImage":
        return cls(slot=d["slot"], source_path=d["source_path"], image_id=d.get("id", ""))


@dataclass
class ProjectConfig:
    """Project-level settings that drive the placement engine."""
    project_name: str = "Untitled"
    project_type: str = "perspective"     # perspective / time_of_day / seasonal
    subject_type: str = "landscape"       # landscape / skylife / urban / indoor / people
    references: list[ReferenceImage] = field(default_factory=list)

    def slots_for_type(self, include_advanced: bool = False) -> tuple[str, ...]:
        if self.project_type == "perspective":
            base = PERSPECTIVE_SLOTS
            return base + PERSPECTIVE_ADVANCED_SLOTS if include_advanced else base
        elif self.project_type == "time_of_day":
            return TIME_SLOTS
        elif self.project_type == "seasonal":
            return SEASON_SLOTS
        return PERSPECTIVE_SLOTS

    def get_reference(self, slot: str) -> Optional[ReferenceImage]:
        for ref in self.references:
            if ref.slot == slot:
                return ref
        return None

    def set_reference(self, slot: str, source_path: str, image_id: str = ""):
        # Remove existing reference for this slot
        self.references = [r for r in self.references if r.slot != slot]
        self.references.append(ReferenceImage(slot=slot, source_path=source_path, image_id=image_id))

    def clear_reference(self, slot: str):
        self.references = [r for r in self.references if r.slot != slot]

    def has_references(self) -> bool:
        return len(self.references) > 0

    def as_dict(self) -> dict:
        return {
            "project_name": self.project_name,
            "project_type": self.project_type,
            "subject_type": self.subject_type,
            "references": [r.as_dict() for r in self.references],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectConfig":
        cfg = cls(
            project_name=d.get("project_name", "Untitled"),
            project_type=d.get("project_type", "perspective"),
            subject_type=d.get("subject_type", "landscape"),
        )
        for rd in d.get("references", []):
            cfg.references.append(ReferenceImage.from_dict(rd))
        return cfg


@dataclass
class PlacementSnapshot:
    """A moment-in-time snapshot of one image's placement (for undo/redo)."""
    image_id: str
    x: float
    y: float
    rotation: float
    z_order: int
    removed: bool = False
