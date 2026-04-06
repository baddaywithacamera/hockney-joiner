"""
help_dialog.py — In-app help system for Hockney Joiner.

Provides a tabbed dialog with:
  - Quick Start guide
  - Keyboard & mouse controls reference
  - Feature explanations (borders, shadows, crop, etc.)
  - About / credits
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


# ── Content ───────────────────────────────────────────────────────────────────

QUICK_START = """\
<h2>Quick Start</h2>

<p><b>1. Create a project</b><br>
On launch you'll be asked to name your project and choose a type
(single reference, perspective, or freeform). Pick the one that matches
your shooting setup.</p>

<p><b>2. Set a reference image</b><br>
In the <i>Reference</i> panel at the top of the sidebar, load the full
wide-angle shot you're assembling from detail slices. This is the image
the matcher will compare every tile against.</p>

<p><b>3. Load your detail images</b><br>
<i>File &rarr; Load Folder</i> or drag a folder onto the window. These are
the close-up shots that tile together to recreate the reference.</p>

<p><b>4. Auto-Place</b><br>
Click <b>Auto-Place</b> in the toolbar (or sidebar). The matching engine
finds where each tile belongs on the reference and positions it. The
reference backdrop appears behind the tiles so you can see alignment.</p>

<p><b>5. Review in Deal Mode</b><br>
Deal Mode starts automatically after placement. Press <b>Space</b> to
reveal tiles one by one, nudge with arrow keys, then <b>Space</b> again
to place. Press <b>Esc</b> when done.</p>

<p><b>6. Fine-tune</b><br>
Click any tile to activate it. Drag to reposition, use arrow keys to
rotate, <b>Z</b>/<b>X</b> to re-order layers. <b>Shift+click</b>
multiple tiles to move them as a group.</p>

<p><b>7. Export</b><br>
<i>File &rarr; Export</i> to render the composite at screen, medium, or
full resolution. Supports PNG (with optional transparency), JPEG, and
TIFF.</p>
"""

CONTROLS = """\
<h2>Keyboard Controls</h2>

<h3>Image Movement</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>&larr; / &rarr;</b></td><td>Rotate active image &plusmn;0.5&deg;</td></tr>
<tr><td><b>Shift + &larr; / &rarr;</b></td><td>Rotate &plusmn;0.1&deg; (fine)</td></tr>
<tr><td><b>&uarr; / &darr;</b></td><td>Nudge active image &plusmn;10 px</td></tr>
<tr><td><b>R</b></td><td>Reset to auto-placed position</td></tr>
<tr><td><b>Delete</b></td><td>Remove image (undoable)</td></tr>
<tr><td><b>Tab / Shift+Tab</b></td><td>Cycle to next / previous tile</td></tr>
</table>

<h3>Z-Order (Pile Controls)</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>Z</b></td><td>Send active image backward one layer</td></tr>
<tr><td><b>X</b></td><td>Bring active image forward one layer</td></tr>
<tr><td><b>Ctrl + hover</b></td><td>Highlight the pile under cursor</td></tr>
<tr><td><b>Ctrl + click</b></td><td>Cycle the pile &mdash; shuffle top to bottom</td></tr>
<tr><td><b>Right-click</b></td><td>Context menu: Bring Forward / Send Backward / Front / Back / Crop</td></tr>
</table>

<h3>Navigation</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>Scroll wheel</b></td><td>Zoom in / out</td></tr>
<tr><td><b>Trackpad pinch</b></td><td>Zoom in / out</td></tr>
<tr><td><b>Middle-mouse drag</b></td><td>Pan canvas</td></tr>
<tr><td><b>F</b></td><td>Fit all images in view</td></tr>
<tr><td><b>G</b></td><td>Toggle grid overlay</td></tr>
</table>

<h3>Selection</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>Click image</b></td><td>Activate (keyboard controls apply to it)</td></tr>
<tr><td><b>Shift + click</b></td><td>Add / remove from multi-selection</td></tr>
<tr><td><b>Drag selected</b></td><td>Move all selected tiles together</td></tr>
<tr><td><b>Click empty canvas</b></td><td>Deactivate all</td></tr>
<tr><td><b>Escape</b></td><td>Clear selection</td></tr>
</table>

<h3>Deal Mode</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>D</b></td><td>Enter Deal Mode</td></tr>
<tr><td><b>Space (1st tap)</b></td><td>Preview next photo with EXIF info</td></tr>
<tr><td><b>Space (2nd tap)</b></td><td>Place photo at its matched position</td></tr>
<tr><td><b>Esc</b></td><td>Exit Deal Mode, reveal remaining images</td></tr>
</table>

<h3>Undo / Redo</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>Ctrl+Z</b></td><td>Undo last operation</td></tr>
<tr><td><b>Ctrl+Y</b></td><td>Redo</td></tr>
</table>
"""

DISPLAY_SETTINGS = """\
<h2>Display Settings</h2>

<h3>Canvas Colour</h3>
<p>Click the colour swatch to pick a background colour for the canvas.
This also affects the background of non-transparent exports.</p>

<h3>Drag Opacity</h3>
<p>Controls how transparent a tile becomes while you're dragging it
(20%&ndash;100%). Lower values let you see what's underneath, making it
easier to align overlapping tiles.</p>

<h3>Reference Backdrop</h3>
<p>After Auto-Place, the reference image appears as a semi-transparent
backdrop behind the tiles.</p>
<ul>
<li><b>Ref opacity</b> &mdash; How visible the backdrop is (0&ndash;50%).
    Set to 0 to hide it completely.</li>
<li><b>Ref scale</b> &mdash; Shrinks or grows the backdrop. Tiles
    reposition automatically to stay aligned. After Auto-Place the scale
    is set automatically based on tile coverage.</li>
</ul>

<h3>White Print Borders</h3>
<p>Slider (0&ndash;20 px). Adds a white border around every tile,
mimicking Hockney's Polaroid joiners. The border is included in
exports.</p>

<h3>Drop Shadows</h3>
<p>Checkbox. Adds a subtle shadow beneath each tile for that
&ldquo;prints scattered on a table&rdquo; look. Rendered in both the
canvas view and exports.</p>

<h3>Tile Cropping</h3>
<p>Right-click any tile &rarr; <b>Crop Tile&hellip;</b> to trim edges.
Set pixel margins for each side. Cropped areas become invisible and
non-clickable. Fully undoable with Ctrl+Z.</p>
"""

MATCHING = """\
<h2>Matching Engines</h2>

<p>The matching engine determines how tiles are positioned on the
reference image. Choose from the <i>Matching Engine</i> dropdown in
the sidebar.</p>

<h3>Recommended</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>Auto</b></td><td>Tries multiple engines, keeps the best result.
    Slowest but most reliable.</td></tr>
<tr><td><b>DISK + LightGlue</b></td><td>Best all-rounder. Learned
    features, good on most subjects.</td></tr>
<tr><td><b>SIFT + LightGlue</b></td><td>Classic features with learned
    matcher. Strong on textured surfaces.</td></tr>
</table>

<h3>Alternatives</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>SuperPoint + LG</b></td><td>Structural edges and corners.
    Good on architecture.</td></tr>
<tr><td><b>ALIKED + LG</b></td><td>Adaptive descriptors. Varied
    textures.</td></tr>
<tr><td><b>SIFT (classic)</b></td><td>CPU-only, no GPU needed.
    Scale-invariant.</td></tr>
<tr><td><b>ORB</b></td><td>Fast CPU option. Best on strong corners.</td></tr>
<tr><td><b>AKAZE</b></td><td>Nonlinear diffusion. Can work on
    blurry images.</td></tr>
<tr><td><b>BRISK</b></td><td>Fastest CPU option.</td></tr>
</table>

<h3>Orientation Detection</h3>
<p>The matcher automatically detects when a tile was shot in portrait
orientation on a landscape reference (or vice versa) and rotates it to
match. Small camera-shake rotations are ignored &mdash; only genuine
90&deg; or 180&deg; orientation changes are applied.</p>

<h3>Odds &amp; Ends Tray</h3>
<p>Tiles that couldn't be matched with enough confidence are placed in
a row below the main composition. You can drag them into position
manually, or re-run Auto-Place with a different engine.</p>
"""

EXPORT_HELP = """\
<h2>Export</h2>

<p><i>File &rarr; Export</i> (Ctrl+E) renders the composite to a file.</p>

<h3>Scale Modes</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>Screen (1&times;)</b></td><td>Fast preview at thumbnail size
    (300 px per tile slot).</td></tr>
<tr><td><b>Medium (5&times;)</b></td><td>Good for web and social media
    (~1500 px per tile slot).</td></tr>
<tr><td><b>Full</b></td><td>Original source resolution. Print quality.
    Large files.</td></tr>
</table>

<h3>Formats</h3>
<table cellpadding="4" cellspacing="0" border="0">
<tr><td><b>PNG</b></td><td>Lossless. Supports transparent background.</td></tr>
<tr><td><b>JPEG</b></td><td>Smaller files, no transparency. Quality 95.</td></tr>
<tr><td><b>TIFF</b></td><td>Lossless with LZW compression. Best for
    print workflows.</td></tr>
</table>

<h3>Options</h3>
<ul>
<li><b>Transparent background</b> &mdash; PNG only. Canvas areas with no
    tile become transparent.</li>
<li><b>Borders, shadows, crop</b> &mdash; All display settings are
    included in the export exactly as shown on the canvas.</li>
</ul>
"""

DEAL_MODE = """\
<h2>Deal Mode</h2>

<p>Deal Mode replays your shooting sequence, revealing tiles one at a
time in filename order (matching sequential camera numbering).</p>

<h3>Workflow</h3>
<ol>
<li>Press <b>D</b> (or <i>View &rarr; Deal Mode</i>) to enter.</li>
<li>All tiles hide. A progress counter appears.</li>
<li>Press <b>Space</b> &mdash; the next tile appears in a corner preview
    showing its EXIF data (shutter, aperture, ISO, focal length).</li>
<li>Press <b>Space</b> again &mdash; the tile animates to its matched
    position. A dashed orange ghost outline shows where it will land.</li>
<li>While the tile is active you can nudge it with arrow keys, rotate,
    or change its z-order before dealing the next one.</li>
<li>Press <b>Esc</b> at any time to exit and reveal all remaining
    tiles.</li>
</ol>

<h3>Batch EXIF Override</h3>
<p>On entry, a dialog lets you type batch shooting info (shutter,
aperture, ISO) that applies to every photo. Useful for cameras that
don't write EXIF. Leave fields blank to use per-file EXIF.</p>

<h3>Why Deal Mode?</h3>
<p>It's designed for building joiners photo-by-photo &mdash; the same way
you'd lay prints on a table. Great for YouTube assembly videos and for
catching placement errors one tile at a time.</p>
"""

ABOUT = """\
<h2>About Hockney Joiner</h2>

<p>A tool for assembling photographs into David Hockney-style photo
joiners. Load a reference image and detail shots, let the feature
matcher find where each piece belongs, then fine-tune by hand.</p>

<p>The misalignments that remain after placement are not failures.
They are the Hockney in the machine.</p>

<h3>Technology</h3>
<ul>
<li><b>LightGlue</b> &mdash; Learned feature matcher (DISK, SuperPoint,
    ALIKED, SIFT extractors)</li>
<li><b>OpenCV</b> &mdash; Classic feature detectors (SIFT, ORB, AKAZE,
    BRISK) and homography estimation</li>
<li><b>PyQt6</b> &mdash; Application framework</li>
<li><b>PIL / Pillow</b> &mdash; Image processing and export</li>
</ul>
"""

_SECTIONS = [
    ("Quick Start", QUICK_START),
    ("Controls", CONTROLS),
    ("Display", DISPLAY_SETTINGS),
    ("Matching", MATCHING),
    ("Export", EXPORT_HELP),
    ("Deal Mode", DEAL_MODE),
    ("About", ABOUT),
]


# ── Dialog ────────────────────────────────────────────────────────────────────

class HelpDialog(QDialog):
    """Tabbed in-app help dialog."""

    def __init__(self, parent=None, section: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Hockney Joiner — Help")
        self.resize(780, 560)
        self.setMinimumSize(600, 400)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Left: section list ────────────────────────────────────────
        self._list = QListWidget()
        self._list.setFixedWidth(140)
        self._list.setStyleSheet(
            "QListWidget { background: #2a2a2a; border: none; "
            "  border-right: 1px solid #444; padding: 8px 0; }"
            "QListWidget::item { padding: 8px 12px; color: #ccc; }"
            "QListWidget::item:selected { background: #3a5a8a; color: #fff; }"
            "QListWidget::item:hover { background: #333; }"
        )
        for title, _ in _SECTIONS:
            item = QListWidgetItem(title)
            self._list.addItem(item)
        layout.addWidget(self._list)

        # ── Right: content pages ──────────────────────────────────────
        self._stack = QStackedWidget()
        for _, html in _SECTIONS:
            page = self._make_page(html)
            self._stack.addWidget(page)
        layout.addWidget(self._stack, stretch=1)

        self._list.currentRowChanged.connect(self._stack.setCurrentIndex)
        self._list.setCurrentRow(section)

    @staticmethod
    def _make_page(html: str) -> QWidget:
        """Create a scrollable HTML content page."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: #1e1e1e; }")

        label = QLabel()
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignTop)
        label.setContentsMargins(20, 16, 20, 16)
        label.setStyleSheet(
            "QLabel { color: #ddd; font-size: 13px; line-height: 1.5; }"
            " h2 { color: #fff; font-size: 18px; margin-bottom: 8px; }"
            " h3 { color: #eee; font-size: 14px; margin-top: 14px; }"
            " table { margin: 6px 0 10px 0; }"
            " td { padding: 3px 10px 3px 0; color: #ccc; }"
            " b { color: #fff; }"
            " ul, ol { margin: 4px 0 8px 20px; }"
            " li { margin: 2px 0; }"
        )
        label.setText(html)

        scroll.setWidget(label)
        return scroll
