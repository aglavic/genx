from __future__ import annotations

import logging
import sys
import os
import shutil
from dataclasses import dataclass

import platformdirs
from logging import debug, info, warning
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets

from ..version import __version__ as program_version
from .exception_handling import CatchModelError, GuiExceptionHandler


from ..core import config as conf_mod
from ..core.colors import COLOR_CYCLES
from ..core.custom_logging import iprint, numpy_set_options
#from ..plugins import add_on_framework as add_on
from ..version import __version__ as program_version

_path = os.path.dirname(__file__)
if _path[-4:] == ".zip":
    _path, ending = os.path.split(_path)

# Get the configuration path, create if it not exists
config_path = os.path.abspath(platformdirs.user_data_dir("GenX3", "ArturGlavic"))
profile_dest = os.path.abspath(os.path.join(config_path, "profiles"))
profile_src = os.path.abspath(os.path.join(_path, "..", "profiles"))
info(
    f"Paths are:\n        config_path={config_path}\n       profile_dest={profile_dest}\n        profile_src{profile_src}"
)
version_file = os.path.join(config_path, "genx.version")
if not os.path.exists(config_path):
    info(f"Creating path: {config_path}")
    os.makedirs(config_path)
if not os.path.exists(profile_dest):
    info(f"Creating path: {profile_dest}")
    shutil.copytree(profile_src, profile_dest)
    open(version_file, "w").write(program_version + "\n")
elif (
    not os.path.exists(version_file)
    or open(version_file, "r").read().rsplit(".", 1)[0] != program_version.rsplit(".", 1)[0]
):
    # update profiles if major version does not match
    info("Update profiles to default for GenX " + program_version)
    from glob import glob

    for fi in glob(os.path.join(profile_src, "*.conf")):
        info(f"    copy {fi} to {profile_dest}")
        shutil.copy2(fi, profile_dest)
    open(version_file, "w").write(program_version + "\n")
if not os.path.exists(os.path.join(config_path, "genx.conf")):
    info(f"Creating genx.conf at {config_path} from default.profile in {profile_src}")
    shutil.copyfile(os.path.join(profile_src, "default.profile"), os.path.join(config_path, "genx.conf"))

manual_url = "https://aglavic.github.io/genx/doc/"
homepage_url = "https://aglavic.github.io/genx/"


from .main_window_ui import Ui_GenxMainWindowUI
from . import genx_resources_rc  # noqa: F401

@dataclass
class GUIConfig(conf_mod.BaseConfig):
    section = "gui"
    hsize: int = 1200  # stores the width of the window
    vsize: int = 800  # stores the height of the window
    vsplit: int = 300
    hsplit: int = 400
    psplit: int = 550
    solver_update_time: float = 1.5
    editor: str = None
    last_update_check: float = 0.0

class GenxMainWindow(conf_mod.Configurable, QtWidgets.QMainWindow):
    """
    Qt main window.
    - UI layout + actions are defined in main_window.ui (compiled to main_window_ui.py).
    - Business logic lives in this class (methods are stubs for now).
    - Action-to-method wiring is done via Qt auto-connect:
      `on_<objectName>_triggered(...)` for QAction with objectName from the .ui file.
    """
    opt: GUIConfig

    def __init__(self, *, filename: Optional[str] = None):
        self._init_phase = True
        self._startup_filename = filename

        # Load GUI config
        conf_mod.Configurable.__init__(self, GUIConfig)
        debug("setup of MainFrame - config")
        conf_mod.config.load_default(os.path.join(config_path, "genx.conf"))
        self.ReadConfig()

        debug("starting setup of MainFrame")
        QtWidgets.QMainWindow.__init__(self)



        self.ui = Ui_GenxMainWindowUI()
        self.ui.setupUi(self)

        self._install_status_update_hook()
        self._setup_window_basics()
        self._setup_autoconnect()
        self._setup_app_exception_dialogs()

        # Apply window size immediately (safe)
        self.resize(int(self.opt.hsize), int(self.opt.vsize))

        # Apply splitter sizes AFTER the event loop gets a chance to lay out widgets
        QtCore.QTimer.singleShot(0, self._apply_gui_config)

        with self.catch_error(action="init", step="initialize main window"):
            self.initialize()

        self._init_phase = False

    # ----------------------------
    # Core setup helpers
    # ----------------------------

    def _setup_window_basics(self) -> None:
        self.setWindowTitle(f"GenX {program_version}")
        self.setMinimumSize(600, 400)

    def _apply_gui_config(self) -> None:
        """
        Apply GUIConfig values to the Qt layout (window size + splitter positions).
        """
        # Window size
        self.resize(int(self.opt.hsize), int(self.opt.vsize))

        # Splitter sizing helpers
        def _set_splitter_sizes(splitter: QtWidgets.QSplitter, first: int) -> None:
            if splitter is None:
                return
            total = splitter.size().width() if splitter.orientation() == QtCore.Qt.Horizontal else splitter.size().height()
            if total <= 0:
                # Not laid out yet; use window size as a reasonable proxy
                total = int(self.opt.hsize) if splitter.orientation() == QtCore.Qt.Horizontal else int(self.opt.vsize)
            first = max(50, int(first))
            second = max(50, int(total - first))
            splitter.setSizes([first, second])

        # mainSplitter: leftTabWidget | rightSplitter
        _set_splitter_sizes(self.ui.mainSplitter, self.opt.vsplit)

        # rightSplitter: plotSplitter | inputTabWidget
        _set_splitter_sizes(self.ui.rightSplitter, self.opt.hsplit)

        # plotSplitter: plotTabWidget | pluginTabWidget
        _set_splitter_sizes(self.ui.plotSplitter, self.opt.psplit)

    def _store_gui_config_from_widgets(self) -> None:
        """
        Read current Qt widget sizes back into GUIConfig (for persistence).
        """
        size = self.size()
        self.opt.hsize = int(size.width())
        self.opt.vsize = int(size.height())

        def _first_size(splitter: QtWidgets.QSplitter) -> int:
            sizes = splitter.sizes()
            return int(sizes[0]) if sizes else 0

        self.opt.vsplit = _first_size(self.ui.mainSplitter)
        self.opt.hsplit = _first_size(self.ui.rightSplitter)
        self.opt.psplit = _first_size(self.ui.plotSplitter)

    def _setup_autoconnect(self) -> None:
        # Enables automatic connections based on objectName:
        # QAction(objectName="actionOpenModel") -> on_actionOpenModel_triggered()
        QtCore.QMetaObject.connectSlotsByName(self)

    def _setup_app_exception_dialogs(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        handler = GuiExceptionHandler(app)
        logging.getLogger().addHandler(handler)

    def _install_status_update_hook(self) -> None:
        def status_update(text: str) -> None:
            sb = self.statusBar()
            if sb is not None:
                sb.showMessage(text, 5000)

        self._status_update = status_update

    def catch_error(self, action: str = "execution", step: str | None = None, verbose: bool = True) -> CatchModelError:
        return CatchModelError(self, action=action, step=step, status_update=self._status_update if verbose else None)

    # ----------------------------
    # Business logic surface (stubs)
    # ----------------------------

    def initialize(self) -> None:
        """Initialize controllers/services. (Stub)"""
        # Future: create model_control, plugin control, load configs, etc.
        if self._startup_filename:
            self.open_model(self._startup_filename)

    def new_model(self) -> None:
        pass

    def new_from_file(self) -> None:
        pass

    def open_model(self, path: str) -> None:
        pass

    def save_model(self) -> None:
        pass

    def save_model_as(self) -> None:
        pass

    def simulate(self) -> None:
        pass

    def evaluate(self) -> None:
        pass

    def start_fit(self) -> None:
        pass

    def stop_fit(self) -> None:
        pass

    def resume_fit(self) -> None:
        pass

    # ----------------------------
    # QAction auto-connected slots
    # ----------------------------

    @QtCore.Slot(bool)
    def on_actionNewModel_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="create new model"):
            self.new_model()

    @QtCore.Slot(bool)
    def on_actionNewFromFile_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="create new model from file"):
            self.new_from_file()

    @QtCore.Slot(bool)
    def on_actionOpenModel_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="open model"):
            # UI dialog port comes later; stub calls business method
            self.open_model("")

    @QtCore.Slot(bool)
    def on_actionSaveModel_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="save model"):
            self.save_model()

    @QtCore.Slot(bool)
    def on_actionSaveModelAs_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="save model as"):
            self.save_model_as()

    @QtCore.Slot(bool)
    def on_actionQuit_triggered(self, checked: bool = False) -> None:
        self.close()

    @QtCore.Slot(bool)
    def on_actionSimulate_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="simulate"):
            self.simulate()

    @QtCore.Slot(bool)
    def on_actionEvaluate_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="evaluate"):
            self.evaluate()

    @QtCore.Slot(bool)
    def on_actionStartFit_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="start fit"):
            self.start_fit()

    @QtCore.Slot(bool)
    def on_actionStopFit_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="stop fit"):
            self.stop_fit()

    @QtCore.Slot(bool)
    def on_actionResumeFit_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="resume fit"):
            self.resume_fit()

    # Optional: handle checkable actions (stubs)
    @QtCore.Slot(bool)
    def on_actionSimulateAutomatically_triggered(self, checked: bool) -> None:
        pass

    @QtCore.Slot(bool)
    def on_actionUseCuda_triggered(self, checked: bool) -> None:
        pass

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Persist window/splitter sizes
        try:
            self._store_gui_config_from_widgets()
            self.opt.save_config(default=True)
            conf_mod.config.write_default(os.path.join(config_path, "genx.conf"))
        except Exception:
            debug("Could not persist GUIConfig on close", exc_info=True)

        super().closeEvent(event)


def _install_qt_exception_handler(app: QtWidgets.QApplication) -> None:
    handler = GuiExceptionHandler(app)
    logging.getLogger().addHandler(handler)

    def excepthook(exc_type, exc, tb):
        logging.getLogger().critical("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = excepthook


def start_qt_app(*, filename: Optional[str], debug: bool = False) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(":/main_gui/genx.ico"))
    _install_qt_exception_handler(app)

    win = GenxMainWindow(filename=filename)
    win.show()

    sys.exit(app.exec())