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

Drag-and-drop a folder or image files anywhere onto the window to load.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QSplitter,
    QStatusBar,
    QToolBar,
    QWidget,
)

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
        download_moondream_on_open: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.session = session
        self.models_dir = models_dir
        self.model_ready = model_ready

        self.store = ImageStore(session)

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1400, 900)
        self.setAcceptDrops(True)   # enable drag-and-drop onto the window

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._connect_signals()

        if download_model_on_open:
            QTimer.singleShot(500, self._start_model_download)

        if download_moondream_on_open:
            # Stagger slightly so LightGlue dialog (if also downloading) appears first
            QTimer.singleShot(800, self._start_moondream_download)

        self._update_status(
            "Ready — drag a folder of images here, or use File → Load."
        )

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

        self.sidebar = Sidebar(self.store, models_dir=self.models_dir, parent=self)
        self.sidebar.setMinimumWidth(240)
        self.sidebar.setMaximumWidth(340)
        splitter.addWidget(self.sidebar)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_menus(self):
        menubar = self.menuBar()

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

        edit_menu = menubar.addMenu("&Edit")

        undo_action = QAction("&Undo", self)
        undo_action.setShortcut(QKeySequence.StandardKey.Undo)
        undo_action.triggered.connect(self.tray_view.undo)
        edit_menu.addAction(undo_action)

        redo_action = QAction("&Redo", self)
        redo_action.setShortcut(QKeySequence.StandardKey.Redo)
        redo_action.triggered.connect(self.tray_view.redo)
        edit_menu.addAction(redo_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        dl_lg = QAction("Download LightGlue (auto-place)…", self)
        dl_lg.triggered.connect(self._start_model_download)
        help_menu.addAction(dl_lg)

        dl_md = QAction("Download Moondream (composition chat)…", self)
        dl_md.triggered.connect(self._start_moondream_download)
        help_menu.addAction(dl_md)

        view_menu = menubar.addMenu("&View")

        grid_action = QAction("Toggle &Grid  [G]", self)
        grid_action.triggered.connect(self.tray_view.toggle_grid)
        view_menu.addAction(grid_action)

        fit_action = QAction("&Fit All in View", self)
        fit_action.setShortcut(QKeySequence("F"))
        fit_action.triggered.connect(self.tray_view.fit_all)
        view_menu.addAction(fit_action)

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        load_btn = QAction("Load Folder…", self)
        load_btn.triggered.connect(self.load_folder)
        toolbar.addAction(load_btn)

        load_files_btn = QAction("Load Images…", self)
        load_files_btn.triggered.connect(self.load_images)
        toolbar.addAction(load_files_btn)

        toolbar.addSeparator()

        self.process_btn = QAction("Auto-Place  [LightGlue]", self)
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
        # Connect moondream chat panel to the tray view
        if self.sidebar.chat_panel:
            self.sidebar.chat_panel.set_tray_view(self.tray_view)

    # ── Drag and drop ──────────────────────────────────────────────────────────

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls]

        folders = [p for p in paths if p.is_dir()]
        files = [p for p in paths if p.is_file()]

        if folders:
            # If a folder was dropped, load it (first one wins)
            self._load_path(folders[0], is_folder=True)
        elif files:
            self._load_path(files, is_folder=False)

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
        n_estimate = 200
        ok, msg = check_scratch_disk_space(self.session.scratch_root, n_estimate)
        if not ok:
            QMessageBox.warning(self, "Scratch Disk Space", msg)

        self._update_status("Loading images… (thumbnails generating)")
        self.process_btn.setEnabled(False)

        worker = LoadWorker(self.store, path, is_folder)
        progress = QProgressDialog("Loading images…", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Loading")
        progress.show()

        worker.finished.connect(lambda count: self._on_load_finished(count, progress))
        worker.error.connect(lambda msg: self._on_load_error(msg, progress))
        worker.start()
        self._load_worker = worker

    def _on_load_finished(self, count: int, progress: QProgressDialog):
        progress.close()
        self.tray_view.refresh()
        self.sidebar.refresh()
        self.tray_view.fit_all()
        self.process_btn.setEnabled(count > 0)
        model_note = "" if self.model_ready else " — LightGlue not downloaded, using grid layout"
        self._update_status(f"{count} images loaded{model_note}.")
        log.info("Load complete: %d images", count)

    def _on_load_error(self, msg: str, progress: QProgressDialog):
        progress.close()
        QMessageBox.critical(self, "Load Error", msg)

    # ── Auto-place ─────────────────────────────────────────────────────────────

    def auto_place(self):
        from hockney.core.placement import PlacementWorker

        self._update_status("Computing placements…")
        self.process_btn.setEnabled(False)

        worker = PlacementWorker(self.store, self.model_ready, self.models_dir)
        progress = QProgressDialog("Computing placements…", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Auto-Place")
        progress.show()

        worker.progress.connect(progress.setValue)
        worker.finished.connect(lambda result: self._on_placement_finished(result, progress))
        worker.error.connect(lambda msg: self._on_placement_error(msg, progress))
        worker.start()
        self._placement_worker = worker

    def _on_placement_finished(self, result, progress: QProgressDialog):
        progress.close()
        self.tray_view.set_placements(result.placements)
        self.tray_view.fit_all()
        self.process_btn.setEnabled(True)
        self._update_status(result.message)

    def _on_placement_error(self, msg: str, progress: QProgressDialog):
        progress.close()
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Placement Error", msg)

    def _on_process_requested(self, settings: dict):
        self.auto_place()

    # ── Project ────────────────────────────────────────────────────────────────

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", "Hockney Project (*.json)"
        )
        if not path:
            return

        from hockney.core.project import load_project
        result = load_project(Path(path), self.store)

        self.tray_view.refresh()
        self.tray_view.set_placements(result.placements)
        self.tray_view._removed_ids = result.removed_ids
        self.tray_view.fit_all()
        self.sidebar.refresh()
        self.sidebar.apply_processing_settings(result.processing)
        self.process_btn.setEnabled(self.store.count() > 0)

        if result.missing_files:
            QMessageBox.warning(
                self, "Missing Files",
                f"{len(result.missing_files)} source file(s) could not be found "
                f"and were skipped:\n\n" + "\n".join(result.missing_files[:10]),
            )
        self._update_status(f"Opened: {Path(path).name}")

    def save_project(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "joiner.json", "Hockney Project (*.json)"
        )
        if not path:
            return

        from hockney.core.project import save_project
        save_project(
            Path(path),
            store=self.store,
            placements=self.tray_view.all_placements(),
            removed_ids=self.tray_view._removed_ids,
            processing=self.sidebar.get_processing_settings(),
        )
        self._update_status(f"Saved: {Path(path).name}")

    # ── Export ─────────────────────────────────────────────────────────────────

    def export(self):
        from hockney.ui.export_dialog import ExportDialog
        dialog = ExportDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return

        output_path = dialog.output_path
        scale_mode = dialog.scale_mode
        if not output_path:
            return

        placements = self.tray_view.all_placements()
        if not placements:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Nothing to export", "Load and arrange some images first.")
            return

        from hockney.core.export import ExportWorker
        processing = self.sidebar.get_processing_settings()
        worker = ExportWorker(placements, self.store, Path(output_path), scale_mode, processing)

        progress = QProgressDialog("Rendering composite…", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Exporting")
        progress.show()

        worker.progress.connect(progress.setValue)
        worker.finished.connect(lambda p: self._on_export_finished(p, progress))
        worker.error.connect(lambda msg: self._on_export_error(msg, progress))
        worker.start()
        self._export_worker = worker

    def _on_export_finished(self, path: str, progress: QProgressDialog):
        progress.close()
        self._update_status(f"Exported: {Path(path).name}")
        QMessageBox.information(self, "Export Complete", f"Saved to:\n{path}")

    def _on_export_error(self, msg: str, progress: QProgressDialog):
        progress.close()
        QMessageBox.critical(self, "Export Failed", msg)

    # ── Model download ─────────────────────────────────────────────────────────

    def _start_model_download(self):
        from hockney.installer.model_fetch import ModelDownloadWorker
        progress = QProgressDialog(
            "Downloading LightGlue model…\nThis will take a few minutes.",
            "Cancel",
            0, 100, self,
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
        QMessageBox.warning(
            self,
            "Model Download Failed",
            f"Could not download the LightGlue model:\n{msg}\n\n"
            "Auto-place will use grid layout until the model is available.",
        )

    def _start_moondream_download(self):
        result = QMessageBox.question(
            self,
            "Download Moondream",
            "<b>Download Moondream2 vision model?</b><br><br>"
            "~1.7 GB download. Runs fully offline after download.<br>"
            "Enables the composition chat panel in the sidebar.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        from hockney.core.vision_chat import MoondreamDownloadWorker
        progress = QProgressDialog("Downloading Moondream (~1.7 GB)…", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()

        worker = MoondreamDownloadWorker(self.models_dir)
        worker.progress.connect(progress.setValue)
        worker.finished.connect(lambda: self._on_moondream_ready(progress))
        worker.error.connect(lambda msg: self._on_moondream_error(msg, progress))
        worker.start()
        self._moondream_worker = worker

    def _on_moondream_ready(self, progress: QProgressDialog):
        progress.close()
        self._update_status("Moondream ready — composition chat available.")

    def _on_moondream_error(self, msg: str, progress: QProgressDialog):
        progress.close()
        QMessageBox.warning(self, "Moondream Download Failed", msg)

    # ── Misc ───────────────────────────────────────────────────────────────────

    def _on_image_activated(self, image_id: str):
        self.sidebar.set_active_image(image_id)

    def _update_status(self, msg: str):
        self.status_bar.showMessage(msg)
        log.info(msg)

    def closeEvent(self, event):
        event.accept()


# ── Background workers ─────────────────────────────────────────────────────────

class LoadWorker(QThread):
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
