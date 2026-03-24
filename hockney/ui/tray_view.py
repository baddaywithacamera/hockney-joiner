"""
tray_view.py — The central Tray View canvas.

A QGraphicsView/QGraphicsScene canvas showing all loaded images
in their auto-detected or manually-placed positions.

Keyboard controls (per spec):
  Arrow keys ←/→        Rotate active image ±0.5° (Shift = ±0.1° fine)
  Arrow keys ↑/↓        Nudge active image ±1px
  Z / X                 Send active image backward/forward in layer stack
  Delete                Remove image from composition (undoable)
  Ctrl+Z / Ctrl+Y       Undo/redo all refinement operations
  R                     Reset active image to auto-placed position
  G                     Toggle grid overlay
  F                     Fit all images in view

Mouse:
  Scroll wheel          Zoom in/out
  Middle-mouse drag     Pan canvas
  Click image           Activate (keyboard controls apply to it)
  Click empty canvas    Deactivate

Philosophy: misalignment is valid creative output, not an error to fix.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QWidget,
)

from hockney.core.image_store import ImageStore
from hockney.core.models import ImagePlacement, PlacementSnapshot

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

ROTATE_STEP = 0.5
ROTATE_FINE = 0.1
NUDGE_STEP = 1.0
GRID_SPACING = 100
GRID_COLOR = QColor(80, 80, 200, 50)
ZOOM_FACTOR = 1.15
THUMB_LONG_EDGE = 300   # must match image_store.THUMB_LONG_EDGE


# ImagePlacement and PlacementSnapshot are defined in hockney.core.models
# and imported above — do not redefine them here.

# ── Undo/redo ──────────────────────────────────────────────────────────────────


class CommandStack:
    def __init__(self, maxlen: int = 500):
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


# ── Graphics item ──────────────────────────────────────────────────────────────

class PhotoItem(QGraphicsPixmapItem):
    """One photograph on the canvas."""

    def __init__(self, image_id: str, pixmap: QPixmap, placement: ImagePlacement):
        super().__init__(pixmap)
        self.image_id = image_id
        self.placement = placement

        # Rotate/scale around image centre
        self.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._apply_placement()

    def _apply_placement(self):
        self.setPos(self.placement.x, self.placement.y)
        self.setRotation(self.placement.rotation)
        self.setZValue(self.placement.z_order)

    def read_back_placement(self):
        """Push current item position back into the placement dataclass."""
        pos = self.pos()
        self.placement.x = pos.x()
        self.placement.y = pos.y()
        self.placement.rotation = self.rotation()
        self.placement.z_order = int(self.zValue())

    def hoverEnterEvent(self, event):
        # Brighten border on hover via opacity trick on siblings (handled in TrayView)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        super().hoverLeaveEvent(event)


# ── Main canvas ────────────────────────────────────────────────────────────────

class TrayView(QGraphicsView):
    """Central canvas — the primary work surface."""

    image_activated = pyqtSignal(str)   # image_id of newly active image

    def __init__(self, store: ImageStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.store = store

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._placements: dict[str, ImagePlacement] = {}
        self._items: dict[str, PhotoItem] = {}
        self._active_id: Optional[str] = None
        self._grid_visible = False
        self._removed_ids: set[str] = set()
        self._commands = CommandStack()

        self._configure_view()

    def _configure_view(self):
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor(28, 28, 28))
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Enable trackpad pinch-to-zoom on Windows/Mac
        self.viewport().grabGesture(Qt.GestureType.PinchGesture)

    # ── Public API ─────────────────────────────────────────────────────────────

    def refresh(self):
        """Rebuild canvas from current store contents."""
        self._scene.clear()
        self._items.clear()
        self._active_id = None

        for record in self.store.all_records():
            if record.id in self._removed_ids:
                continue

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

    def arrange_grid(self):
        """
        Arrange all images in a grid — used as fallback when LightGlue
        isn't available or overlap between a pair is insufficient.
        Logged so the photographer knows it happened.
        """
        records = [r for r in self.store.all_records() if r.id not in self._removed_ids]
        if not records:
            return

        n = len(records)
        cols = max(1, math.isqrt(n))
        padding = 24

        for i, record in enumerate(records):
            col = i % cols
            row = i // cols

            # Use actual aspect ratio for spacing
            aspect = record.width / max(record.height, 1)
            if aspect >= 1:
                tw = THUMB_LONG_EDGE
                th = int(THUMB_LONG_EDGE / aspect)
            else:
                tw = int(THUMB_LONG_EDGE * aspect)
                th = THUMB_LONG_EDGE

            x = float(col * (THUMB_LONG_EDGE + padding))
            y = float(row * (th + padding))

            p = self._placements.get(record.id)
            if p:
                p.x = p.auto_x = x
                p.y = p.auto_y = y
                p.rotation = p.auto_rotation = 0.0
                p.z_order = i
                item = self._items.get(record.id)
                if item:
                    item._apply_placement()

        log.info("Grid fallback: %d images, %d columns", n, cols)

    def set_placements(self, placements: list[ImagePlacement]):
        """Apply computed placements (from PlacementWorker) to all images."""
        snap_before = self._snapshot_all()
        for p in placements:
            self._placements[p.image_id] = p
            item = self._items.get(p.image_id)
            if item:
                item.placement = p
                item._apply_placement()
        self._commands.push(snap_before)
        self._update_scene_rect()

    def fit_all(self):
        """Zoom and pan so all images are visible."""
        rect = self._scene.itemsBoundingRect()
        if rect.isNull():
            return
        # Add a small margin
        margin = 40
        rect.adjust(-margin, -margin, margin, margin)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def all_placements(self) -> list[ImagePlacement]:
        return list(self._placements.values())

    # ── Activation ─────────────────────────────────────────────────────────────

    def _activate(self, image_id: str):
        self._active_id = image_id
        # Dim everything else, full opacity on active
        for iid, item in self._items.items():
            item.setOpacity(1.0 if iid == image_id else 0.45)
        self.image_activated.emit(image_id)

    def _deactivate_all(self):
        self._active_id = None
        for item in self._items.values():
            item.setOpacity(1.0)

    # ── Keyboard ───────────────────────────────────────────────────────────────

    def keyPressEvent(self, event: QKeyEvent):
        key = event.key()
        mod = event.modifiers()

        # G and F work without an active image
        if key == Qt.Key.Key_G:
            self.toggle_grid()
            return
        if key == Qt.Key.Key_F:
            self.fit_all()
            return

        if self._active_id is None:
            super().keyPressEvent(event)
            return

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
                item._apply_placement()

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
                item._apply_placement()
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
        log.info("Removed: %s (undoable)", self._active_id)
        self._active_id = None
        self._deactivate_all()

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
        factor = ZOOM_FACTOR if delta > 0 else 1 / ZOOM_FACTOR
        self.scale(factor, factor)

    def event(self, event):
        """Catch pinch gestures from trackpad for zoom."""
        from PyQt6.QtCore import QEvent
        if event.type() == QEvent.Type.Gesture:
            gesture = event.gesture(Qt.GestureType.PinchGesture)
            if gesture:
                self.scale(gesture.scaleFactor(), gesture.scaleFactor())
                return True
        return super().event(event)

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

    def _update_scene_rect(self):
        """Expand scene rect to fit all items with margin."""
        rect = self._scene.itemsBoundingRect()
        margin = 200
        self._scene.setSceneRect(rect.adjusted(-margin, -margin, margin, margin))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _numpy_to_pixmap(arr: np.ndarray) -> QPixmap:
    """Convert HxWx3 uint8 numpy array to QPixmap."""
    h, w, ch = arr.shape
    assert ch == 3
    qimage = QImage(arr.data, w, h, w * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage)
