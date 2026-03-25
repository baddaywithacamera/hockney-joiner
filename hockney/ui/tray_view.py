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

Z-order pile controls:
  Ctrl + hover          Highlight the pile of images under the cursor
  Ctrl + click          Cycle the pile — shuffle top to bottom to expose buried images
  Right-click image     Context menu: "Bring Forward" / "Send Backward"
  Up arrow + click      Move clicked image up one step in pile
  Down arrow + click    Move clicked image down one step in pile

Deal Mode (shoot replay):
  D                     Enter deal mode — hide all, reveal one by one
  Spacebar (1st)        Show next photo in corner preview with EXIF
  Spacebar (2nd)        Send it to its calculated position on the table
  ESC                   Exit deal mode, reveal all remaining images

Philosophy: misalignment is valid creative output, not an error to fix.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from enum import Enum, auto
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QPropertyAnimation, QPointF, QEasingCurve, pyqtSignal
from PyQt6.QtGui import (
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QLineEdit,
    QMenu,
    QVBoxLayout,
    QWidget,
)

from hockney.core.image_store import ImageStore
from hockney.core.models import ImagePlacement, PlacementSnapshot

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

ROTATE_STEP = 0.5
ROTATE_FINE = 0.1
NUDGE_STEP = 10.0
GRID_SPACING = 100
GRID_COLOR = QColor(80, 80, 200, 50)
ZOOM_FACTOR = 1.15
THUMB_LONG_EDGE = 300   # must match image_store.THUMB_LONG_EDGE


# ImagePlacement and PlacementSnapshot are defined in hockney.core.models
# and imported above — do not redefine them here.


class DealState(Enum):
    """State machine for Deal Mode's two-tap spacebar flow."""
    IDLE = auto()               # not in deal mode
    WAITING_PREVIEW = auto()    # ready for first tap — will show preview
    SHOWING_PREVIEW = auto()    # preview visible, waiting for second tap to place


class DealOverlay(QWidget):
    """
    Fixed overlay in the lower-right corner of the viewport showing the
    current photo preview plus EXIF info during Deal Mode.
    """

    PREVIEW_SIZE = 220  # max dimension for the corner thumbnail
    MARGIN = 16

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self.setFixedWidth(260)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._progress_label = QLabel()
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet(
            "color: #ccc; font-size: 13px; font-weight: bold;"
        )
        layout.addWidget(self._progress_label)

        self._image_label = QLabel()
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setFixedSize(self.PREVIEW_SIZE + 8, self.PREVIEW_SIZE + 8)
        self._image_label.setStyleSheet(
            "background: #1a1a1a; border: 2px solid #555; border-radius: 4px;"
        )
        layout.addWidget(self._image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self._exif_label = QLabel()
        self._exif_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._exif_label.setWordWrap(True)
        self._exif_label.setStyleSheet(
            "color: #aaa; font-size: 11px; font-family: monospace;"
        )
        layout.addWidget(self._exif_label)

        self._filename_label = QLabel()
        self._filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._filename_label.setStyleSheet(
            "color: #888; font-size: 10px;"
        )
        layout.addWidget(self._filename_label)

        self.setStyleSheet(
            "DealOverlay { background: rgba(20, 20, 20, 200); "
            "border: 1px solid #444; border-radius: 6px; }"
        )
        self.adjustSize()
        self.hide()

    def show_photo(self, pixmap: QPixmap, exif: dict, filename: str,
                   current: int, total: int):
        """Update the overlay with a new photo."""
        self._progress_label.setText(f"{current} / {total}")

        # Scale pixmap to fit preview area
        scaled = pixmap.scaled(
            self.PREVIEW_SIZE, self.PREVIEW_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

        # EXIF line
        parts = []
        if "shutter" in exif:
            parts.append(exif["shutter"])
        if "aperture" in exif:
            parts.append(exif["aperture"])
        if "iso" in exif:
            parts.append(exif["iso"])
        if "focal_length" in exif:
            parts.append(exif["focal_length"])
        self._exif_label.setText("  |  ".join(parts) if parts else "")
        self._exif_label.setVisible(bool(parts))

        self._filename_label.setText(filename)

        self.adjustSize()
        self.show()
        self._reposition()

    def _reposition(self):
        """Anchor to the lower-right of the parent viewport."""
        if self.parent():
            p = self.parent()
            x = p.width() - self.width() - self.MARGIN
            y = p.height() - self.height() - self.MARGIN
            self.move(max(0, x), max(0, y))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition()

class DealModeDialog(QDialog):
    """
    Optional dialog shown when entering Deal Mode.  Lets the user type in
    batch shooting info (shutter, aperture, ISO) that overrides / supplements
    per-file EXIF for the whole set.  All fields are optional — leave blank
    to use whatever EXIF the file contains.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deal Mode — Batch Info")
        self.setMinimumWidth(320)

        layout = QVBoxLayout(self)

        hint = QLabel(
            "Optional: enter shooting info for the whole batch.\n"
            "Leave fields blank to use per-file EXIF (if available)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; margin-bottom: 8px;")
        layout.addWidget(hint)

        form = QFormLayout()
        self.shutter_edit = QLineEdit()
        self.shutter_edit.setPlaceholderText("e.g. 1/125s")
        form.addRow("Shutter:", self.shutter_edit)

        self.aperture_edit = QLineEdit()
        self.aperture_edit.setPlaceholderText("e.g. f/8")
        form.addRow("Aperture:", self.aperture_edit)

        self.iso_edit = QLineEdit()
        self.iso_edit.setPlaceholderText("e.g. ISO 400")
        form.addRow("ISO:", self.iso_edit)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_overrides(self) -> dict[str, str]:
        """Return a dict of non-empty override values."""
        result: dict[str, str] = {}
        s = self.shutter_edit.text().strip()
        if s:
            result["shutter"] = s
        a = self.aperture_edit.text().strip()
        if a:
            result["aperture"] = a
        i = self.iso_edit.text().strip()
        if i:
            result["iso"] = i
        return result


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
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
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
    deal_mode_changed = pyqtSignal(bool)  # True when entering, False when exiting

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
        self._highlighted_ids: set[str] = set()   # AI-flagged images
        self._highlight_timer = None
        self._pile_highlighted_ids: set[str] = set()   # Ctrl-hover pile

        self.setMouseTracking(True)    # needed for Ctrl+hover pile highlight

        # ── Deal Mode state ──────────────────────────────────────────
        self._deal_state = DealState.IDLE
        self._deal_queue: list[str] = []       # image_ids sorted by filename
        self._deal_index: int = 0              # next image to deal
        self._deal_overlay = DealOverlay(self.viewport())
        self._deal_visible_ids: set[str] = set()  # images already dealt onto table
        self._deal_exif_override: dict[str, str] = {}  # batch override for EXIF

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
        tile = self.store.thumb_long_edge

        for i, record in enumerate(records):
            col = i % cols
            row = i // cols

            # Use actual aspect ratio for spacing
            aspect = record.width / max(record.height, 1)
            if aspect >= 1:
                tw = tile
                th = int(tile / aspect)
            else:
                tw = int(tile * aspect)
                th = tile

            x = float(col * (tile + padding))
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
        margin = 40
        rect.adjust(-margin, -margin, margin, margin)
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def highlight_by_index(self, one_based_indices: list[int]):
        """
        Highlight images by their 1-based position in the sorted placement list.
        Called when moondream returns index references in its response.
        Highlighted images pulse with a coloured border. Clears on any keypress
        or after 30 seconds.
        """
        from PyQt6.QtCore import QTimer
        sorted_p = sorted(self._placements.values(), key=lambda p: p.z_order)
        self._highlighted_ids = set()
        for idx in one_based_indices:
            if 1 <= idx <= len(sorted_p):
                self._highlighted_ids.add(sorted_p[idx - 1].image_id)

        self._apply_highlight()

        # Auto-clear after 30 seconds
        if self._highlight_timer:
            self._highlight_timer.stop()
        self._highlight_timer = QTimer(self)
        self._highlight_timer.setSingleShot(True)
        self._highlight_timer.timeout.connect(self.clear_highlight)
        self._highlight_timer.start(30_000)

    def clear_highlight(self):
        """Remove moondream highlight from all images."""
        self._highlighted_ids.clear()
        self._apply_highlight()
        if self._highlight_timer:
            self._highlight_timer.stop()

    def _apply_highlight(self):
        """Set opacity: highlighted=1.0, others=0.35 if any highlighted, else all 1.0."""
        if not self._highlighted_ids:
            for item in self._items.values():
                item.setOpacity(1.0)
            return
        for iid, item in self._items.items():
            item.setOpacity(1.0 if iid in self._highlighted_ids else 0.35)

    def render_contact_sheet(self) -> Optional["Image"]:
        """Render a numbered contact sheet for moondream analysis."""
        from hockney.core.export import render_contact_sheet
        placements = list(self._placements.values())
        if not placements:
            return None
        return render_contact_sheet(placements, self.store)

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

        # Any keypress clears moondream highlight
        if self._highlighted_ids:
            self.clear_highlight()

        # ── Deal Mode keys ────────────────────────────────────────
        if self.in_deal_mode:
            if key == Qt.Key.Key_Space:
                self._deal_spacebar()
                return
            if key == Qt.Key.Key_Escape:
                self.exit_deal_mode()
                return
            # Allow adjustment keys (arrows, Z, X) while in deal mode
            # so user can nudge/z-order the just-placed image before
            # dealing the next one.  Fall through to normal handling.

        # D to enter deal mode (only when not already in it)
        if key == Qt.Key.Key_D and not self.in_deal_mode:
            if self._items:
                self.enter_deal_mode()
            return

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

    # ── Deal Mode ──────────────────────────────────────────────────────────────

    def enter_deal_mode(self):
        """
        Enter Deal Mode: hide all images, prepare a filename-sorted queue,
        and wait for spacebar taps to reveal them one by one.
        """
        records = self.store.all_records()
        if not records:
            return

        # Show batch-info dialog (Cancel aborts entry)
        dlg = DealModeDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        self._deal_exif_override = dlg.get_overrides()

        # Sort by filename (cameras number sequentially)
        records_sorted = sorted(records, key=lambda r: r.source_path.name.lower())
        self._deal_queue = [r.id for r in records_sorted
                            if r.id not in self._removed_ids]

        if not self._deal_queue:
            return

        self._deal_index = 0
        self._deal_visible_ids.clear()
        self._deal_state = DealState.WAITING_PREVIEW

        # Hide all images on the canvas
        for item in self._items.values():
            item.setVisible(False)
            item.setOpacity(1.0)

        self._deal_overlay._reposition()
        self._deal_overlay._progress_label.setText(
            f"0 / {len(self._deal_queue)}"
        )
        self._deal_overlay.show()

        self.deal_mode_changed.emit(True)
        self._update_status_for_deal()
        log.info("Deal Mode entered: %d images queued", len(self._deal_queue))

    def exit_deal_mode(self):
        """Exit Deal Mode: reveal any remaining undealt images."""
        if self._deal_state == DealState.IDLE:
            return

        self._deal_state = DealState.IDLE
        self._deal_overlay.hide()

        # Show all non-removed images
        for iid, item in self._items.items():
            if iid not in self._removed_ids:
                item.setVisible(True)
                item.setOpacity(1.0)

        self._deal_visible_ids.clear()
        self._deal_queue.clear()
        self._deal_index = 0

        self.deal_mode_changed.emit(False)
        log.info("Deal Mode exited")

    @property
    def in_deal_mode(self) -> bool:
        return self._deal_state != DealState.IDLE

    def _deal_spacebar(self):
        """Handle spacebar press during Deal Mode."""
        if self._deal_state == DealState.WAITING_PREVIEW:
            # First tap: show next photo in corner preview
            if self._deal_index >= len(self._deal_queue):
                # All images dealt — exit deal mode
                self.exit_deal_mode()
                return

            image_id = self._deal_queue[self._deal_index]
            record = self.store.get_record(image_id)
            if record is None:
                self._deal_index += 1
                self._deal_spacebar()  # skip missing, try next
                return

            # Get thumbnail pixmap for preview
            arr = self.store.get_thumbnail(image_id)
            if arr is not None:
                pixmap = _numpy_to_pixmap(arr)
            else:
                pixmap = QPixmap(100, 100)
                pixmap.fill(QColor(60, 60, 60))

            # Read EXIF from source file, then apply batch overrides
            from hockney.core.image_store import get_exif_info
            exif = get_exif_info(record.source_path)
            # Batch overrides win over per-file EXIF
            exif.update(self._deal_exif_override)

            self._deal_overlay.show_photo(
                pixmap, exif, record.source_path.name,
                self._deal_index + 1, len(self._deal_queue),
            )
            self._deal_state = DealState.SHOWING_PREVIEW

        elif self._deal_state == DealState.SHOWING_PREVIEW:
            # Second tap: place photo on the table at its calculated position
            image_id = self._deal_queue[self._deal_index]
            item = self._items.get(image_id)
            if item:
                item.setVisible(True)
                item.setOpacity(1.0)
                self._deal_visible_ids.add(image_id)

                # Animate from bottom-right area to target position
                target = QPointF(item.placement.x, item.placement.y)
                # Start position: offset toward bottom-right of current view
                view_br = self.mapToScene(
                    self.viewport().width() - 50,
                    self.viewport().height() - 50,
                )
                self._animate_item_to(item, view_br, target)

                # Activate the just-placed image so user can adjust
                self._activate(image_id)

            self._deal_index += 1
            self._deal_state = DealState.WAITING_PREVIEW
            self._update_status_for_deal()

            # Update overlay progress (but keep it visible for reference)
            if self._deal_index >= len(self._deal_queue):
                self._deal_overlay._progress_label.setText(
                    f"{len(self._deal_queue)} / {len(self._deal_queue)} — done!"
                )
            else:
                self._deal_overlay._progress_label.setText(
                    f"{self._deal_index} / {len(self._deal_queue)}"
                )

    def _animate_item_to(self, item: PhotoItem, start: QPointF, end: QPointF,
                         duration_ms: int = 350):
        """
        Smoothly move item from start to end using a QTimer-based tween.
        QGraphicsPixmapItem doesn't inherit QObject so we can't use
        QPropertyAnimation directly.
        """
        from PyQt6.QtCore import QTimer

        steps = max(1, duration_ms // 16)  # ~60fps
        step_count = [0]
        item.setPos(start)

        def _step():
            step_count[0] += 1
            t = min(1.0, step_count[0] / steps)
            # ease-out cubic
            t_ease = 1.0 - (1.0 - t) ** 3
            x = start.x() + (end.x() - start.x()) * t_ease
            y = start.y() + (end.y() - start.y()) * t_ease
            item.setPos(x, y)
            if t >= 1.0:
                timer.stop()
                item.setPos(end)

        timer = QTimer(self)
        timer.setInterval(16)
        timer.timeout.connect(_step)
        timer.start()
        # prevent GC
        self._deal_anim_timer = timer

    def _update_status_for_deal(self):
        """Update status bar text during deal mode."""
        total = len(self._deal_queue)
        placed = self._deal_index
        if placed < total:
            log.info("Deal Mode: %d / %d placed — spacebar for next", placed, total)
        else:
            log.info("Deal Mode: all %d images placed", total)

    # ── Pile (z-order clump) helpers ──────────────────────────────────────────

    def _items_at_pos(self, view_pos) -> list[PhotoItem]:
        """Return all PhotoItems under a viewport position, sorted by z_order."""
        scene_pos = self.mapToScene(view_pos)
        items = self._scene.items(scene_pos)
        pile = [i for i in items if isinstance(i, PhotoItem)]
        pile.sort(key=lambda i: i.placement.z_order)
        return pile

    def _highlight_pile(self, pile: list[PhotoItem]):
        """Dim everything except the pile; clear if pile is empty."""
        old = self._pile_highlighted_ids
        self._pile_highlighted_ids = {i.image_id for i in pile}
        if self._pile_highlighted_ids == old:
            return  # no change
        if not self._pile_highlighted_ids:
            # Restore normal opacity (respect active image dimming)
            if self._active_id:
                self._activate(self._active_id)
            else:
                self._deactivate_all()
            return
        for iid, item in self._items.items():
            item.setOpacity(1.0 if iid in self._pile_highlighted_ids else 0.25)

    def _clear_pile_highlight(self):
        if self._pile_highlighted_ids:
            self._pile_highlighted_ids.clear()
            if self._active_id:
                self._activate(self._active_id)
            else:
                self._deactivate_all()

    def _cycle_pile(self, pile: list[PhotoItem]):
        """
        Shuffle the pile: take the topmost image and send it to the bottom
        of the pile, exposing the next one. All z_orders within the pile
        rotate down by one step.
        """
        if len(pile) < 2:
            return
        snap_before = [self._snapshot_one(i.image_id) for i in pile]

        # Collect the current z_orders and rotate them
        z_values = [i.placement.z_order for i in pile]
        # Move top to bottom: [1,2,3] → [3,1,2] (highest z goes to lowest)
        rotated = [z_values[-1]] + z_values[:-1]
        # Swap: bottom gets top's z, everyone else shifts down
        rotated = z_values[1:] + [z_values[0]]

        for item, new_z in zip(pile, rotated):
            item.placement.z_order = new_z
            item._apply_placement()

        self._commands.push(snap_before)

    def _move_in_pile(self, item: PhotoItem, direction: int):
        """
        Move item up (+1) or down (-1) in the z-order relative to its
        immediate neighbours in the pile at its position.
        """
        scene_pos = item.mapToScene(item.boundingRect().center())
        all_here = self._scene.items(scene_pos)
        pile = sorted(
            [i for i in all_here if isinstance(i, PhotoItem)],
            key=lambda i: i.placement.z_order,
        )
        if len(pile) < 2:
            return

        idx = next((j for j, p in enumerate(pile) if p.image_id == item.image_id), None)
        if idx is None:
            return

        swap_idx = idx + direction
        if swap_idx < 0 or swap_idx >= len(pile):
            return

        snap_before = [self._snapshot_one(pile[idx].image_id),
                       self._snapshot_one(pile[swap_idx].image_id)]

        # Swap z_orders
        z_a = pile[idx].placement.z_order
        z_b = pile[swap_idx].placement.z_order
        if z_a == z_b:
            z_b += direction  # break tie
        pile[idx].placement.z_order = z_b
        pile[swap_idx].placement.z_order = z_a
        pile[idx]._apply_placement()
        pile[swap_idx]._apply_placement()

        self._commands.push(snap_before)

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

    def mouseMoveEvent(self, event):
        """Ctrl+hover: highlight the pile of images under the cursor."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            pile = self._items_at_pos(event.pos())
            self._highlight_pile(pile)
        elif self._pile_highlighted_ids:
            self._clear_pile_highlight()
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        mod = event.modifiers()

        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            super().mousePressEvent(event)
            return

        item = self.itemAt(event.pos())

        # Ctrl+click: cycle the pile under cursor
        if (event.button() == Qt.MouseButton.LeftButton
                and mod & Qt.KeyboardModifier.ControlModifier):
            pile = self._items_at_pos(event.pos())
            self._cycle_pile(pile)
            return

        # Up/Down arrow held + click: move image in pile
        from PyQt6.QtWidgets import QApplication
        keys = QApplication.queryKeyboardModifiers()
        # (Arrow keys aren't modifiers, so we check via key state in keyPressEvent instead —
        #  handled below in the right-click context menu and Z/X keys)

        if isinstance(item, PhotoItem):
            self._activate(item.image_id)
        else:
            self._deactivate_all()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
        # After drag-move, write the new position back to the placement
        if event.button() == Qt.MouseButton.LeftButton:
            for item in self._scene.selectedItems():
                if isinstance(item, PhotoItem):
                    item.read_back_placement()
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        """Right-click context menu for z-order control."""
        item = self.itemAt(event.pos())
        if not isinstance(item, PhotoItem):
            super().contextMenuEvent(event)
            return

        self._activate(item.image_id)

        menu = QMenu(self)
        bring_fwd = menu.addAction("Bring Forward  (X)")
        send_back = menu.addAction("Send Backward  (Z)")
        menu.addSeparator()
        bring_top = menu.addAction("Bring to Front")
        send_bottom = menu.addAction("Send to Back")

        action = menu.exec(event.globalPos())
        if action is None:
            return

        snap_before = self._snapshot_one(item.image_id)
        if action == bring_fwd:
            self._move_in_pile(item, +1)
        elif action == send_back:
            self._move_in_pile(item, -1)
        elif action == bring_top:
            max_z = max(p.z_order for p in self._placements.values()) + 1
            item.placement.z_order = max_z
            item._apply_placement()
            self._commands.push([snap_before])
        elif action == send_bottom:
            min_z = min(p.z_order for p in self._placements.values()) - 1
            item.placement.z_order = min_z
            item._apply_placement()
            self._commands.push([snap_before])

    def keyReleaseEvent(self, event):
        """Clear pile highlight when Ctrl is released."""
        if event.key() == Qt.Key.Key_Control:
            self._clear_pile_highlight()
        super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        """Reposition deal overlay when viewport resizes."""
        super().resizeEvent(event)
        if self.in_deal_mode:
            self._deal_overlay._reposition()

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
