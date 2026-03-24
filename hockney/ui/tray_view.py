"""
tray_view.py — The central Tray View canvas.

A QGraphicsView/QGraphicsScene-based canvas showing all loaded images
in their auto-detected or manually-placed positions.

Key interactions (per spec):
  Scroll wheel          Zoom in/out
  Middle-mouse drag     Pan canvas
  Hover over image      Highlight edges, dim others
  Click to activate     Selected image becomes focus for transforms
  Arrow keys ←/→        Rotate active image ±0.5° (Shift for ±0.1°)
  Arrow keys ↑/↓        Nudge active image ±1px
  Z / X keys            Send active image backward/forward in layer stack
  Delete key            Remove image (undoable)
  Ctrl+Z / Ctrl+Y       Undo/redo
  R key                 Reset active image to auto-placed position
  G key                 Toggle grid overlay

Misalignment is not a bug. This view embraces it.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QTransform,
    QWheelEvent,
    QColor,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from hockney.core.image_store import ImageStore

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

ROTATE_STEP = 0.5        # degrees per arrow keypress
ROTATE_FINE = 0.1        # degrees with Shift held
NUDGE_STEP = 1.0         # pixels per arrow keypress
GRID_SPACING = 100       # pixels between grid lines
GRID_COLOR = QColor(80, 80, 200, 60)   # subtle blue-grey, low alpha
DIM_OPACITY = 0.35       # opacity of non-active images on hover


# ── Placement state ────────────────────────────────────────────────────────────

@dataclass
class ImagePlacement:
    """Mutable placement state for one image on the canvas."""
    image_id: str
    x: float = 0.0
    y: float = 0.0
    rotation: float = 0.0    # degrees
    z_order: int = 0
    auto_x: float = 0.0      # LightGlue-suggested position (for reset)
    auto_y: float = 0.0
    auto_rotation: float = 0.0

    def reset_to_auto(self):
        self.x = self.auto_x
        self.y = self.auto_y
        self.rotation = self.auto_rotation

    def as_dict(self) -> dict:
        return {
            "image_id": self.image_id,
            "x": self.x,
            "y": self.y,
            "rotation": self.rotation,
            "z_order": self.z_order,
        }


# ── Undo/redo command stack ────────────────────────────────────────────────────

@dataclass
class PlacementSnapshot:
    """A moment-in-time snapshot of one image's placement."""
    image_id: str
    x: float
    y: float
    rotation: float
    z_order: int
    removed: bool = False


class CommandStack:
    """Simple undo/redo stack for placement operations."""

    def __init__(self, maxlen: int = 200):
        self._undo: deque[list[PlacementSnapshot]] = deque(maxlen=maxlen)
        self._redo: deque[list[PlacementSnapshot]] = deque(maxlen=maxlen)

    def push(self, snapshots: list[PlacementSnapshot]):
        self._undo.append(snapshots)
        self._redo.clear()

    def undo(self) -> Optional[list[PlacementSnapshot]]:
        if not self._undo:
            return None
        snap = self._undo.pop()
        self._redo.append(snap)
        return snap

    def redo(self) -> Optional[list[PlacementSnapshot]]:
        if not self._redo:
            return None
        snap = self._redo.pop()
        self._undo.append(snap)
        return snap

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)


# ── Graphics item ──────────────────────────────────────────────────────────────

class PhotoItem(QGraphicsPixmapItem):
    """
    A single photograph on the Tray View canvas.
    Wraps a QGraphicsPixmapItem with extra state (image_id, placement ref).
    """

    def __init__(self, image_id: str, pixmap: QPixmap, placement: ImagePlacement):
        super().__init__(pixmap)
        self.image_id = image_id
        self.placement = placement

        self.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self._sync_from_placement()

    def _sync_from_placement(self):
        self.setPos(self.placement.x, self.placement.y)
        self.setRotation(self.placement.rotation)
        self.setZValue(self.placement.z_order)

    def sync_to_placement(self):
        pos = self.pos()
        self.placement.x = pos.x()
        self.placement.y = pos.y()
        self.placement.rotation = self.rotation()
        self.placement.z_order = int(self.zValue())


# ── Main canvas ────────────────────────────────────────────────────────────────

class TrayView(QGraphicsView):
    """Central canvas. The primary work surface."""

    # Emitted when the user clicks/activates an image
    image_activated = pyqtSignal(str)

    def __init__(self, store: ImageStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._placements: dict[str, ImagePlacement] = {}
        self._items: dict[str, PhotoItem] = {}
        self._active_id: Optional[str] = None
        self._grid_visible = False
        self._grid_items: list[QGraphicsRectItem] = []
        self._commands = CommandStack()
        self._removed_ids: set[str] = set()

        self._configure_view()

    def _configure_view(self):
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(30, 30, 30))   # dark grey canvas background
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # ── Public API ─────────────────────────────────────────────────────────────

    def refresh(self):
        """Reload all images from the store and redraw the canvas."""
        self._scene.clear()
        self._items.clear()

        for record in self.store.all_records():
            if record.id in self._removed_ids:
                continue

            # Create placement if this is a new image
            if record.id not in self._placements:
                self._placements[record.id] = ImagePlacement(image_id=record.id)

            arr = self.store.get_thumbnail(record.id)
            if arr is None:
                continue

            pixmap = _numpy_to_pixmap(arr)
            placement = self._placements[record.id]
            item = PhotoItem(record.id, pixmap, placement)
            self._scene.addItem(item)
            self._items[record.id] = item

        self.arrange_grid()  # default layout until auto-place runs

    def arrange_grid(self):
        """
        Fallback layout: arrange all images in a grid.
        Called when LightGlue isn't available or overlap is insufficient.
        """
        records = [r for r in self.store.all_records() if r.id not in self._removed_ids]
        if not records:
            return

        cols = max(1, int(len(records) ** 0.5))
        thumb_w = 300
        thumb_h = 200
        padding = 20

        for i, record in enumerate(records):
            col = i % cols
            row = i // cols
            placement = self._placements.get(record.id)
            if placement is None:
                continue
            x = col * (thumb_w + padding)
            y = row * (thumb_h + padding)
            placement.x = placement.auto_x = float(x)
            placement.y = placement.auto_y = float(y)
            placement.rotation = placement.auto_rotation = 0.0
            if record.id in self._items:
                self._items[record.id]._sync_from_placement()

        log.info("Grid layout applied to %d images (%d columns)", len(records), cols)

    def set_placements(self, placements: list[ImagePlacement]):
        """Apply LightGlue-computed placements to all images."""
        snapshot_before = self._snapshot_all()
        for p in placements:
            self._placements[p.image_id] = p
            if p.image_id in self._items:
                self._items[p.image_id]._sync_from_placement()
        self._commands.push(snapshot_before)

    def all_placements(self) -> list[ImagePlacement]:
        return list(self._placements.values())

    # ── Active image ───────────────────────────────────────────────────────────

    def _activate(self, image_id: str):
        prev = self._active_id
        self._active_id = image_id

        for iid, item in self._items.items():
            if iid == image_id:
                item.setOpacity(1.0)
            elif prev is not None:
                item.setOpacity(1.0)  # restore on deactivation

        self.image_activated.emit(image_id)

    def _deactivate_all(self):
        self._active_id = None
        for item in self._items.values():
            item.setOpacity(1.0)

    # ── Keyboard control ───────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        if self._active_id is None:
            super().keyPressEvent(event)
            return

        key = event.key()
        mod = event.modifiers()
        placement = self._placements.get(self._active_id)
        if placement is None:
            return

        snap_before = self._snapshot_one(self._active_id)
        changed = True

        if key == Qt.Key.Key_Left:
            step = ROTATE_FINE if mod & Qt.KeyboardModifier.ShiftModifier else ROTATE_STEP
            placement.rotation -= step
        elif key == Qt.Key.Key_Right:
            step = ROTATE_FINE if mod & Qt.KeyboardModifier.ShiftModifier else ROTATE_STEP
            placement.rotation += step
        elif key == Qt.Key.Key_Up:
            placement.y -= NUDGE_STEP
        elif key == Qt.Key.Key_Down:
            placement.y += NUDGE_STEP
        elif key == Qt.Key.Key_Z:
            placement.z_order -= 1
        elif key == Qt.Key.Key_X:
            placement.z_order += 1
        elif key == Qt.Key.Key_R:
            placement.reset_to_auto()
        elif key == Qt.Key.Key_G:
            self.toggle_grid()
            changed = False
        elif key == Qt.Key.Key_Delete:
            self._remove_active()
            changed = False
        else:
            changed = False
            super().keyPressEvent(event)

        if changed:
            self._commands.push([snap_before])
            item = self._items.get(self._active_id)
            if item:
                item._sync_from_placement()

    # ── Undo / redo ────────────────────────────────────────────────────────────

    def undo(self):
        snaps = self._commands.undo()
        if snaps:
            self._apply_snapshots(snaps)

    def redo(self):
        snaps = self._commands.redo()
        if snaps:
            self._apply_snapshots(snaps)

    def _apply_snapshots(self, snaps: list[PlacementSnapshot]):
        for snap in snaps:
            p = self._placements.get(snap.image_id)
            if p is None:
                continue
            p.x = snap.x
            p.y = snap.y
            p.rotation = snap.rotation
            p.z_order = snap.z_order
            item = self._items.get(snap.image_id)
            if item:
                item._sync_from_placement()
                item.setVisible(not snap.removed)

    # ── Remove ─────────────────────────────────────────────────────────────────

    def _remove_active(self):
        if not self._active_id:
            return
        snap = self._snapshot_one(self._active_id, removed=True)
        self._commands.push([snap])
        self._removed_ids.add(self._active_id)
        item = self._items.get(self._active_id)
        if item:
            item.setVisible(False)
        self._active_id = None

    # ── Grid overlay ───────────────────────────────────────────────────────────

    def toggle_grid(self):
        self._grid_visible = not self._grid_visible
        self.viewport().update()

    def drawBackground(self, painter: QPainter, rect: QRectF):
        super().drawBackground(painter, rect)
        if not self._grid_visible:
            return

        pen = QPen(GRID_COLOR)
        pen.setWidth(1)
        painter.setPen(pen)

        left = int(rect.left()) - (int(rect.left()) % GRID_SPACING)
        top = int(rect.top()) - (int(rect.top()) % GRID_SPACING)

        x = left
        while x < rect.right():
            painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += GRID_SPACING

        y = top
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += GRID_SPACING

    # ── Mouse ──────────────────────────────────────────────────────────────────

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            item = self.itemAt(event.pos())
            if isinstance(item, PhotoItem):
                self._activate(item.image_id)
            else:
                self._deactivate_all()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        super().mouseReleaseEvent(event)

    # ── Snapshots ──────────────────────────────────────────────────────────────

    def _snapshot_one(self, image_id: str, removed: bool = False) -> PlacementSnapshot:
        p = self._placements[image_id]
        return PlacementSnapshot(
            image_id=image_id,
            x=p.x, y=p.y,
            rotation=p.rotation,
            z_order=p.z_order,
            removed=removed,
        )

    def _snapshot_all(self) -> list[PlacementSnapshot]:
        return [self._snapshot_one(iid) for iid in self._placements]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _numpy_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert an HxWx3 uint8 numpy array to a QPixmap."""
    h, w, ch = arr.shape
    assert ch == 3
    bytes_per_line = w * 3
    qimage = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage)
