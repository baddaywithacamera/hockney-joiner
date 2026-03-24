"""
main_window.py — Top-level application window.

Layout:
  ┌──────────────────────────────────────────┐
  │  Toolbar (load, process, export buttons) │
  ├───────────────────┬──────────────────────┤
  │                   │                      │
  │   Tray View       │   Sidebar            │
  │   (central        │   (image list,       │
  │    canvas)        │    settings panel)   │
  │                   │                      │
  ├───────────────────┴──────────────────────┤
  │  Status bar                              │
  └──────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressDialog,
    QSplitter,
    QStatusBar,
    QToolBar,
    QWidget,
)
from PyQt6.QtGui import QAction, QKeySequence

from hockney.core.image_store import ImageStore, ScratchSession, check_scratch_disk_space
from hockney.ui.tray_view import TrayView
from hockney.ui.sidebar import Sidebar

log = logging.getLogger(__name__)

WINDOW_TITLE = "Hockney Joiner"


class MainWindow(QMainWindow):
    def __init__(
        self,
        session: ScratchSession,
        models_dir: Path,
        model_ready: bool,
        download_model_on_open: bool,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.session = session
        self.models_dir = models_dir
        self.model_ready = model_ready

        self.store = ImageStore(session)

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1400, 900)

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._connect_signals()

        if download_model_on_open:
            # Defer to after the window is shown
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(500, self._start_model_download)

        self._update_status("Ready. Load images to begin.")

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        self.tray_view = TrayView(self.store, parent=self)
        splitter.addWidget(self.tray_view)

        self.sidebar = Sidebar(self.store, parent=self)
        self.sidebar.setMinimumWidth(240)
        self.sidebar.setMaximumWidth(340)
        splitter.addWidget(self.sidebar)

        splitter.setStretchFactor(0, 3)   # Tray View gets most of the space
        splitter.setStretchFactor(1, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_menus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        load_action = QAction("&Load Images…", self)
        load_action.setShortcut(QKeySequence.StandardKey.Open)
        load_action.triggered.connect(self.load_images)
        file_menu.addAction(load_action)

        load_folder_action = QAction("Load &Folder…", self)
        load_folder_action.triggered.connect(self.load_folder)
        file_menu.addAction(load_folder_action)

        file_menu.addSeparator()

        open_project_action = QAction("&Open Project…", self)
        open_project_action.triggered.connect(self.open_project)
        file_menu.addAction(open_project_action)

        save_project_action = QAction("&Save Project…", self)
        save_project_action.setShortcut(QKeySequence.StandardKey.Save)
        save_project_action.triggered.connect(self.save_project)
        file_menu.addAction(save_project_action)

        file_menu.addSeparator()

        export_action = QAction("&Export…", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.tray_view.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.tray_view.redo)
        edit_menu.addAction(redo_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        grid_action = QAction("Toggle &Grid", self)
        grid_action.setShortcut(QKeySequence("G"))
        grid_action.triggered.connect(self.tray_view.toggle_grid)
        view_menu.addAction(grid_action)

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        load_btn = QAction("Load Images", self)
        load_btn.triggered.connect(self.load_folder)
        toolbar.addAction(load_btn)

        toolbar.addSeparator()

        self.process_btn = QAction("Auto-Place", self)
        self.process_btn.triggered.connect(self.auto_place)
        self.process_btn.setEnabled(False)
        toolbar.addAction(self.process_btn)

        toolbar.addSeparator()

        export_btn = QAction("Export…", self)
        export_btn.triggered.connect(self.export)
        toolbar.addAction(export_btn)

    def _connect_signals(self):
        self.sidebar.process_requested.connect(self._on_process_requested)
        self.tray_view.image_activated.connect(self._on_image_activated)
        self.store  # (store signals wired up as methods are added)

    # ── Load ───────────────────────────────────────────────────────────────────

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not folder:
            return
        self._load_path(Path(folder), is_folder=True)

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.cr2 *.cr3 *.nef *.arw *.dng)",
        )
        if not paths:
            return
        self._load_path([Path(p) for p in paths], is_folder=False)

    def _load_path(self, path, is_folder: bool):
        # Quick space check before committing
        n_estimate = 500  # conservative estimate when loading a folder
        ok, msg = check_scratch_disk_space(self.session.scratch_root, n_estimate)
        if not ok:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Scratch Disk Space", msg)

        self._update_status("Loading images…")

        worker = LoadWorker(self.store, path, is_folder)
        progress = QProgressDialog("Loading images…", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker.finished.connect(lambda count: self._on_load_finished(count, progress))
        worker.error.connect(lambda msg: self._on_load_error(msg, progress))
        worker.start()
        self._load_worker = worker  # keep reference

    def _on_load_finished(self, count: int, progress: QProgressDialog):
        progress.close()
        self._update_status(f"Loaded {count} images.")
        self.tray_view.refresh()
        self.process_btn.setEnabled(count > 0)
        log.info("Load complete: %d images", count)

    def _on_load_error(self, msg: str, progress: QProgressDialog):
        progress.close()
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Load Error", msg)

    # ── Auto-place ─────────────────────────────────────────────────────────────

    def auto_place(self):
        """Run LightGlue matching and position all images."""
        if not self.model_ready:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Model Not Ready",
                "The LightGlue model has not been downloaded yet.\n"
                "Auto-place requires the AI model. Images will be arranged in a grid.",
            )
            self.tray_view.arrange_grid()
            return

        self._update_status("Running LightGlue feature matching…")
        # TODO: wire up PlacementWorker (hockney/core/placement.py)

    def _on_process_requested(self, settings: dict):
        """Called by sidebar when user clicks Process."""
        self.auto_place()

    # ── Project ────────────────────────────────────────────────────────────────

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Hockney Project (*.json)"
        )
        if path:
            # TODO: load project from JSON
            self._update_status(f"Opened: {Path(path).name}")

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "joiner.json", "Hockney Project (*.json)"
        )
        if path:
            self.store.save_meta(Path(path))
            self._update_status(f"Saved: {Path(path).name}")

    # ── Export ─────────────────────────────────────────────────────────────────

    def export(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Composite",
            "joiner.tif",
            "TIFF (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg)",
        )
        if path:
            self._update_status(f"Exporting to {Path(path).name}…")
            # TODO: wire up ExportWorker (hockney/core/export.py)

    # ── Model download ─────────────────────────────────────────────────────────

    def _start_model_download(self):
        from hockney.installer.model_fetch import ModelDownloadWorker
        progress = QProgressDialog(
            "Downloading LightGlue model…\nThis will take a few minutes.",
            "Cancel",
            0,
            100,
            self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker = ModelDownloadWorker(self.models_dir)
        worker.progress.connect(progress.setValue)
        worker.finished.connect(lambda: self._on_model_ready(progress))
        worker.error.connect(lambda msg: self._on_model_error(msg, progress))
        worker.start()
        self._model_worker = worker

    def _on_model_ready(self, progress: QProgressDialog):
        progress.close()
        self.model_ready = True
        self._update_status("LightGlue model ready.")

    def _on_model_error(self, msg: str, progress: QProgressDialog):
        progress.close()
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "Model Download Failed",
            f"Could not download the LightGlue model:\n{msg}\n\n"
            "Auto-place will use grid fallback until the model is available.",
        )

    # ── Misc ───────────────────────────────────────────────────────────────────

    def _on_image_activated(self, image_id: str):
        self.sidebar.set_active_image(image_id)

    def _update_status(self, msg: str):
        self.status_bar.showMessage(msg)
        log.info(msg)

    def closeEvent(self, event):
        # Scratch session cleanup happens in main() after app.exec() returns
        event.accept()


# ── Background workers ─────────────────────────────────────────────────────────

class LoadWorker(QThread):
    """Loads images in a background thread so the UI stays responsive."""
    finished = pyqtSignal(int)
    error = pyqtSignal(str)

    def __init__(self, store: ImageStore, path, is_folder: bool):
        super().__init__()
        self.store = store
        self.path = path
        self.is_folder = is_folder

    def run(self):
        try:
            if self.is_folder:
                records = self.store.load_folder(self.path)
            else:
                records = self.store.load_files(self.path)
            self.finished.emit(len(records))
        except Exception as e:
            self.error.emit(str(e))
