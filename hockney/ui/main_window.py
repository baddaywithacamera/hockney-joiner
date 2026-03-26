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
from collections import deque
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QDockWidget,
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
from hockney.core.models import ProjectConfig
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
        self._status_history: deque[str] = deque(maxlen=4)
        self._project_config: ProjectConfig | None = None

        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1400, 900)
        self.setAcceptDrops(True)   # enable drag-and-drop onto the window

        self._build_ui()
        self._build_menus()
        self._build_toolbar()
        self._restore_layout()
        self._connect_signals()

        # Prompt for a new project on startup
        QTimer.singleShot(300, self._prompt_new_project)

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
        # Tray canvas is the central (non-dockable) widget
        self.tray_view = TrayView(self.store, parent=self)
        self.setCentralWidget(self.tray_view)

        # ── Tools & settings dock (right) ─────────────────────────────────────
        self.sidebar = Sidebar(self.store, models_dir=self.models_dir, parent=self)
        self.sidebar.setMinimumWidth(240)

        self._sidebar_dock = QDockWidget("Tools", self)
        self._sidebar_dock.setObjectName("sidebar_dock")
        self._sidebar_dock.setWidget(self.sidebar)
        self._sidebar_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea
            | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sidebar_dock)

        # ── Composition chat dock (right, below Tools by default) ─────────────
        if self.sidebar.chat_panel:
            # Pull the chat panel out of the sidebar and give it its own dock
            self.sidebar.layout().removeWidget(self.sidebar.chat_panel)
            self._chat_dock = QDockWidget("Composition Chat", self)
            self._chat_dock.setObjectName("chat_dock")
            self._chat_dock.setWidget(self.sidebar.chat_panel)
            self._chat_dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
                | Qt.DockWidgetArea.BottomDockWidgetArea
            )
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._chat_dock)
            # Stack the chat dock below the sidebar dock by default
            self.splitDockWidget(
                self._sidebar_dock,
                self._chat_dock,
                Qt.Orientation.Vertical,
            )
        else:
            self._chat_dock = None

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def _build_menus(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")

        new_project_action = QAction("&New Project…", self)
        new_project_action.setShortcut(QKeySequence("Ctrl+N"))
        new_project_action.triggered.connect(self._new_project)
        file_menu.addAction(new_project_action)

        file_menu.addSeparator()

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

        clear_action = QAction("&Clear Composition", self)
        clear_action.setToolTip("Remove all images from the canvas and start fresh")
        clear_action.triggered.connect(self._clear_composition)
        file_menu.addAction(clear_action)

        file_menu.addSeparator()

        scratch_action = QAction("Change &Scratch Disk…", self)
        scratch_action.setToolTip("Choose a different scratch disk folder (e.g. a USB SSD)")
        scratch_action.triggered.connect(self._change_scratch_disk)
        file_menu.addAction(scratch_action)

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

        dl_md = QAction("Download Composition AI…", self)
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

        view_menu.addSeparator()

        # Panels — these are created in _build_ui so we add them after the
        # fact using a deferred lambda; the dock's own toggle action is also
        # available via its title-bar context menu when floating.
        show_tools_action = QAction("Show &Tools Panel", self)
        show_tools_action.setCheckable(True)
        show_tools_action.setChecked(True)
        show_tools_action.triggered.connect(
            lambda checked: self._sidebar_dock.setVisible(checked)
        )
        self._sidebar_dock.visibilityChanged.connect(show_tools_action.setChecked)
        view_menu.addAction(show_tools_action)

        if self._chat_dock:
            show_chat_action = QAction("Show &Composition Chat", self)
            show_chat_action.setCheckable(True)
            show_chat_action.setChecked(True)
            show_chat_action.triggered.connect(
                lambda checked: self._chat_dock.setVisible(checked)
            )
            self._chat_dock.visibilityChanged.connect(show_chat_action.setChecked)
            view_menu.addAction(show_chat_action)

        view_menu.addSeparator()

        self._deal_mode_action = QAction("&Deal Mode  [D]", self)
        self._deal_mode_action.setCheckable(True)
        self._deal_mode_action.setToolTip(
            "Reveal images one-by-one in shooting order (spacebar to advance)"
        )
        self._deal_mode_action.triggered.connect(self._toggle_deal_mode)
        view_menu.addAction(self._deal_mode_action)

        view_menu.addSeparator()

        reset_layout_action = QAction("&Reset Panel Layout", self)
        reset_layout_action.setToolTip("Return all panels to their default positions")
        reset_layout_action.triggered.connect(self._reset_panel_layout)
        view_menu.addAction(reset_layout_action)

    def _build_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setObjectName("main_toolbar")
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)
        self._main_toolbar = toolbar

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

    def _restore_layout(self):
        """Restore saved window geometry and dock layout, then force toolbar to top."""
        _LAYOUT_VERSION = 4
        from PyQt6.QtCore import QSettings
        settings = QSettings("HockneyJoiner", "Hockney Joiner")
        if settings.value("layout_version", 0, type=int) == _LAYOUT_VERSION:
            geometry = settings.value("window_geometry")
            state = settings.value("window_state")
            if geometry:
                self.restoreGeometry(geometry)
            if state:
                self.restoreState(state)
        else:
            settings.remove("window_geometry")
            settings.remove("window_state")
            settings.setValue("layout_version", _LAYOUT_VERSION)

        # Always force the toolbar to the top — restoreState can misplace it
        if hasattr(self, '_main_toolbar'):
            self.addToolBar(Qt.ToolBarArea.TopToolBarArea, self._main_toolbar)

    def _connect_signals(self):
        self.sidebar.process_requested.connect(self._on_process_requested)
        self.sidebar.reference_changed.connect(self._on_reference_changed)
        self.sidebar.bg_color_changed.connect(self.tray_view.set_bg_color)
        self.sidebar.ref_backdrop_changed.connect(self.tray_view.update_reference_backdrop)
        self.sidebar.fit_all_requested.connect(self.tray_view.fit_all)
        self.tray_view.image_activated.connect(self._on_image_activated)
        self.tray_view.deal_mode_changed.connect(self._on_deal_mode_changed)
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

    def _last_image_dir(self) -> str:
        """Return the last directory used for loading images, or empty string."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("HockneyJoiner", "Hockney Joiner")
        return settings.value("last_image_dir", "", type=str)

    def _save_image_dir(self, path: Path):
        """Persist the directory so next open starts there."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("HockneyJoiner", "Hockney Joiner")
        folder = str(path if path.is_dir() else path.parent)
        settings.setValue("last_image_dir", folder)

    def _last_project_dir(self) -> str:
        """Return the last directory used for project open/save, or empty string."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("HockneyJoiner", "Hockney Joiner")
        return settings.value("last_project_dir", "", type=str)

    def _save_project_dir(self, path: Path):
        """Persist the project directory so next open/save starts there."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("HockneyJoiner", "Hockney Joiner")
        folder = str(path if path.is_dir() else path.parent)
        settings.setValue("last_project_dir", folder)

    def load_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Image Folder", self._last_image_dir()
        )
        if not folder:
            return
        self._save_image_dir(Path(folder))
        self._load_path(Path(folder), is_folder=True)

    def load_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            self._last_image_dir(),
            "Images (*.jpg *.jpeg *.png *.tif *.tiff *.cr2 *.cr3 *.nef *.arw *.dng)",
        )
        if not paths:
            return
        self._save_image_dir(Path(paths[0]))
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
        self.tray_view.arrange_grid()   # spread images out instead of stacking at origin
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

        worker = PlacementWorker(self.store, self.model_ready, self.models_dir,
                                  config=self._project_config)

        # Build a plain QDialog instead of QProgressDialog to avoid
        # QProgressDialog's auto-cancel / auto-close quirks.
        from PyQt6.QtWidgets import QProgressBar, QPushButton, QVBoxLayout, QDialog, QLabel
        dlg = QDialog(self)
        dlg.setWindowTitle("Auto-Place")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumWidth(350)
        dlg_layout = QVBoxLayout(dlg)
        dlg_label = QLabel("Computing placements…")
        dlg_layout.addWidget(dlg_label)
        dlg_bar = QProgressBar()
        dlg_bar.setRange(0, 100)
        dlg_layout.addWidget(dlg_bar)
        dlg_cancel = QPushButton("Cancel")
        dlg_layout.addWidget(dlg_cancel)
        dlg.show()

        def _on_cancel():
            worker.cancel()
            dlg_label.setText("Cancelling…")
            dlg_cancel.setEnabled(False)

        dlg_cancel.clicked.connect(_on_cancel)
        worker.progress.connect(dlg_bar.setValue)
        worker.finished.connect(lambda result: self._on_placement_finished(result, dlg))
        worker.error.connect(lambda msg: self._on_placement_error(msg, dlg))
        worker.start()
        self._placement_worker = worker

    def _on_placement_finished(self, result, dlg):
        dlg.close()
        if self._placement_worker and self._placement_worker._cancelled:
            self.process_btn.setEnabled(True)
            self._update_status("Placement cancelled.")
            return
        self.tray_view.set_placements(result.placements)
        # Auto-scale so tiles and reference are visually proportionate
        auto_s = self.tray_view.auto_scale_to_fit()
        self.sidebar.set_ref_scale(auto_s)
        # Now show the backdrop at the computed scale
        ref_opacity = self.sidebar._ref_opacity_slider.value() / 100.0
        self.tray_view.show_reference_backdrop(
            self._project_config, opacity=ref_opacity, scale_pct=auto_s)
        self.tray_view.fit_all()
        self.process_btn.setEnabled(True)
        self._update_status(result.message)

        # Auto-enter deal mode after placement so user can review
        # images one by one and adjust positions.
        # Deferred via QTimer so the progress dialog's focus-restoration
        # finishes before we grab keyboard focus on the TrayView.
        if result.placements and self.store.count() > 0:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(150, lambda: self.tray_view.enter_deal_mode(skip_dialog=True))

    def _on_placement_error(self, msg: str, dlg):
        dlg.close()
        self.process_btn.setEnabled(True)
        QMessageBox.critical(self, "Placement Error", msg)

    def _on_process_requested(self, settings: dict):
        # Update matching engine from sidebar selector
        engine = settings.get("matching_engine", "auto")
        if self._project_config:
            self._project_config.matching_engine = engine
        self.auto_place()

    # ── Project ────────────────────────────────────────────────────────────────

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", self._last_project_dir(), "Hockney Project (*.json)"
        )
        if not path:
            return
        self._save_project_dir(Path(path))

        from hockney.core.project import load_project
        result = load_project(Path(path), self.store)

        # Restore project config if present
        if result.config:
            self._project_config = result.config
            self.sidebar.set_project_config(result.config)
            self._update_toolbar_label()

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
        default_name = self._last_project_dir() or "joiner.json"
        if not default_name.endswith(".json"):
            default_name = str(Path(default_name) / "joiner.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", default_name, "Hockney Project (*.json)"
        )
        if not path:
            return
        self._save_project_dir(Path(path))

        from hockney.core.project import save_project
        save_project(
            Path(path),
            store=self.store,
            placements=self.tray_view.all_placements(),
            removed_ids=self.tray_view._removed_ids,
            processing=self.sidebar.get_processing_settings(),
            config=self._project_config,
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
        processing["transparent_bg"] = dialog.transparent_bg
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
            "Download Composition AI",
            "<b>Download BLIP-VQA vision model?</b><br><br>"
            "~1.2 GB download. Runs fully offline after download.<br>"
            "Enables the composition chat panel in the sidebar.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        from hockney.core.vision_chat import MoondreamDownloadWorker
        progress = QProgressDialog("Downloading Composition AI (~1.2 GB)…\nThis may take a few minutes.", None, 0, 100, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.show()

        worker = MoondreamDownloadWorker(self.models_dir)
        worker.progress.connect(progress.setValue)
        worker.finished.connect(lambda: self._on_moondream_ready(progress))
        worker.error.connect(lambda msg: self._on_moondream_error(msg, progress))
        worker.start()
        self._moondream_worker = worker

    def _on_moondream_ready(self, progress: QProgressDialog):
        progress.close()
        self._update_status("Composition AI ready — chat panel available.")

    def _on_moondream_error(self, msg: str, progress: QProgressDialog):
        progress.close()
        QMessageBox.warning(self, "Download Failed", msg)

    # ── Project config ──────────────────────────────────────────────────────

    def _prompt_new_project(self):
        """Show New Project dialog on startup if no project is loaded."""
        if self._project_config is not None:
            return
        self._new_project()

    def _new_project(self):
        from hockney.ui.new_project_dialog import NewProjectDialog
        dlg = NewProjectDialog(self)
        if dlg.exec() != dlg.DialogCode.Accepted:
            return
        config = dlg.get_config()
        self._project_config = config
        self.sidebar.set_project_config(config)
        self._update_toolbar_label()
        self.setWindowTitle(f"{WINDOW_TITLE} — {config.project_name}")
        self._update_status(f"Project: {config.project_name}  ({config.project_type})")

    def _on_reference_changed(self):
        """Sidebar reference panel changed — update toolbar label."""
        self._update_toolbar_label()

    def _update_toolbar_label(self):
        """Update the Auto-Place button text based on whether references are loaded."""
        if self._project_config and self._project_config.has_references():
            self.process_btn.setText("Auto-Place  [Reference]")
        else:
            self.process_btn.setText("Auto-Place  [LightGlue]")

    # ── Misc ───────────────────────────────────────────────────────────────────

    def _on_image_activated(self, image_id: str):
        self.sidebar.set_active_image(image_id)

    def _toggle_deal_mode(self, checked: bool):
        if checked:
            if self.store.count() == 0:
                QMessageBox.information(
                    self, "No Images",
                    "Load some images first before entering Deal Mode."
                )
                self._deal_mode_action.setChecked(False)
                return
            self.tray_view.enter_deal_mode()
            self._update_status(
                "Deal Mode — spacebar: preview next photo, spacebar again: place it. ESC to exit."
            )
        else:
            self.tray_view.exit_deal_mode()
            self._update_status("Deal Mode exited.")

    def _on_deal_mode_changed(self, active: bool):
        """Keep menu checkbox in sync when deal mode is toggled via keyboard."""
        self._deal_mode_action.setChecked(active)
        if not active:
            self._update_status("Deal Mode exited.")

    def _clear_composition(self):
        """File → Clear Composition — dump everything and start fresh."""
        if self.store.count() == 0:
            return
        result = QMessageBox.question(
            self,
            "Clear Composition",
            f"Remove all {self.store.count()} images from the canvas?\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if result != QMessageBox.StandardButton.Yes:
            return

        # Exit deal mode if active
        if self.tray_view.in_deal_mode:
            self.tray_view.exit_deal_mode()

        self.tray_view._scene.clear()
        self.tray_view._items.clear()
        self.tray_view._placements.clear()
        self.tray_view._removed_ids.clear()
        self.tray_view._active_id = None
        self.tray_view._commands = type(self.tray_view._commands)()  # fresh CommandStack
        self.store.clear()
        self._project_config = None
        self.sidebar.set_project_config(None)
        self._update_toolbar_label()
        self.setWindowTitle(WINDOW_TITLE)
        self.sidebar.refresh()
        self.process_btn.setEnabled(False)
        self._update_status("Composition cleared.")

    def _change_scratch_disk(self):
        """File → Change Scratch Disk — pick a new scratch location mid-session."""
        from hockney.main import ScratchDiskDialog
        from PyQt6.QtCore import QSettings
        from hockney.main import APP_ORG, APP_NAME

        current = self.session.scratch_root if hasattr(self.session, "scratch_root") else ""
        dialog = ScratchDiskDialog(self)
        if current:
            dialog.chosen_path = current
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_path = dialog.chosen_path
        new_path.mkdir(parents=True, exist_ok=True)

        # Persist the new choice
        settings = QSettings(APP_ORG, APP_NAME)
        settings.setValue("scratch_disk", str(new_path))

        QMessageBox.information(
            self,
            "Scratch Disk Updated",
            (
                f"Scratch disk changed to:<br><code>{new_path}</code><br><br>"
                "The new location will be used for all future cache writes. "
                "Restart the app to migrate any existing cache files."
            ),
        )
        log.info("Scratch disk changed to: %s", new_path)

    def _reset_panel_layout(self):
        """View → Reset Panel Layout — bring all docks back to defaults."""
        # Re-dock everything in case panels were floated to other monitors
        self._sidebar_dock.setFloating(False)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._sidebar_dock)
        self._sidebar_dock.setVisible(True)
        if self._chat_dock:
            self._chat_dock.setFloating(False)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._chat_dock)
            self.splitDockWidget(
                self._sidebar_dock,
                self._chat_dock,
                Qt.Orientation.Vertical,
            )
            self._chat_dock.setVisible(True)

    def _update_status(self, msg: str):
        self._status_history.append(msg)
        display = "  ·  ".join(self._status_history)
        self.status_bar.showMessage(display)
        log.info(msg)

    def closeEvent(self, event):
        # Persist window geometry and dock layout so panels reopen where the
        # user left them — including on secondary monitors
        from PyQt6.QtCore import QSettings
        settings = QSettings("HockneyJoiner", "Hockney Joiner")
        settings.setValue("layout_version", 4)
        settings.setValue("window_geometry", self.saveGeometry())
        settings.setValue("window_state", self.saveState())
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
