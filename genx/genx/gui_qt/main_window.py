import logging
import sys
import os
import shutil
import time
import webbrowser
import tempfile
import subprocess
from dataclasses import dataclass

import platformdirs
from logging import debug, info, warning
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets, QtPrintSupport

from ..version import __version__ as program_version
from .exception_handling import CatchModelError, GuiExceptionHandler
from .message_dialogs import ShowQuestionDialog, ShowNotificationDialog


from ..core import config as conf_mod
from ..core.colors import COLOR_CYCLES
from ..core.custom_logging import iprint, numpy_set_options
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

from . import genx_resources_rc

@dataclass
class GUIConfig(conf_mod.BaseConfig):
    section = "gui"
    hsize: int = 1200  # stores the width of the window
    vsize: int = 800  # stores the height of the window
    vsplit: int = 300
    hsplit: int = 400
    psplit: int = 550
    lsplit: int = 300
    solver_update_time: float = 1.5
    editor: str = None
    last_update_check: float = 0.0


class _ActionCheckable:
    def __init__(self, action: QtGui.QAction) -> None:
        self._action = action

    def Check(self) -> None:
        self._action.setChecked(True)


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
        self._setup_app_exception_dialogs()
        self._splash = None
        self._splash_base_pixmap = None
        self._start_splash()
        self._update_splash("initializing modules...", progress=0.0)
        self._preload_numba_modules()
        self._update_splash("creating main window...", progress=0.7)

        self._init_phase = True
        self._startup_filename = filename
        self._logging_dialogs = []
        self._last_input_tab = None
        self._help_dialogs = []
        self._external_script_file = None
        self._external_editor_proc = None
        self._script_watch_timer = None
        self._auto_simulate_timer = None
        self.mb_checkables = {}

        # Load GUI config
        conf_mod.Configurable.__init__(self, GUIConfig)
        debug("setup of MainFrame - config")
        conf_mod.config.load_default(os.path.join(config_path, "genx.conf"))
        self.ReadConfig()

        debug("starting setup of MainFrame")
        QtWidgets.QMainWindow.__init__(self)

        from .main_window_ui import Ui_GenxMainWindowUI
        self.ui = Ui_GenxMainWindowUI()
        self.ui.setupUi(self)

        self._update_splash("creating main window....", progress=0.75)

        self._install_status_update_hook()
        self._setup_data_view()
        self._setup_window_basics()
        self._setup_auto_color_menu()
        self._update_splash("creating main window.....", progress=0.85)
        self._setup_toolbar_icon_sizes()
        self._update_splash("creating main window......", progress=0.95)

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

    def _start_splash(self) -> None:
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        pixmap = QtGui.QPixmap(":/main_gui/genx.png")
        if pixmap.isNull():
            pixmap = QtGui.QPixmap(400, 400)
            pixmap.fill(QtGui.QColor(255, 255, 255))
        else:
            pixmap = pixmap.scaled(400, 400, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

        splash = QtWidgets.QSplashScreen(pixmap, QtCore.Qt.WindowStaysOnTopHint)
        splash.show()
        app.processEvents()
        try:
            import pyi_splash

            pyi_splash.close()
        except ImportError:
            pass

        self._splash = splash
        self._splash_base_pixmap = pixmap

    def _update_splash(self, text: str, progress: float = 0.0) -> None:
        if self._splash is None or self._splash_base_pixmap is None:
            return
        pixmap = QtGui.QPixmap(self._splash_base_pixmap)
        painter = QtGui.QPainter(pixmap)
        font = QtWidgets.QApplication.font()
        painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        bar_height = metrics.height() + 4
        margin = max(10, int(pixmap.width() * 0.075))
        bar_width = pixmap.width() - 2 * margin
        painter.fillRect(margin, 0, bar_width, bar_height, QtGui.QColor(255, 255, 255))
        if progress > 0.0:
            filled = int(bar_width * min(progress, 1.0))
            painter.fillRect(margin, 0, filled, bar_height, QtGui.QColor(252, 175, 62))
        painter.setPen(QtGui.QColor(0, 0, 0))
        text_width = metrics.horizontalAdvance(text)
        painter.drawText((pixmap.width() - text_width) // 2, metrics.ascent() + 2, text)
        painter.end()
        self._splash.setPixmap(pixmap)
        self._splash.show()
        QtWidgets.QApplication.processEvents()

    def _finish_splash(self) -> None:
        if self._splash is None:
            return
        self._update_splash("Done.", progress=1.0)
        self._splash.finish(self)
        self._splash = None
        self._splash_base_pixmap = None

    def _preload_numba_modules(self) -> None:
        from ..models.lib import USE_NUMBA

        if not USE_NUMBA:
            return
        try:
            import numba
        except ImportError:
            return

        from ..models.lib.numba_integration import configure_numba

        configure_numba()

        import inspect

        real_jit = numba.jit

        class UpdateJit:
            update_counter = 1

            def __call__(self, *args, **opts):
                if inspect.stack()[1][3] != "<lambda>":
                    self._update_status(
                        f"compiling numba functions {self.update_counter}/22",
                        0.0 + 0.7 * (self.update_counter - 1) / 22.0,
                    )
                    self.update_counter += 1
                return real_jit(*args, **opts)

            def __init__(self, update_status):
                self._update_status = update_status

        numba.jit = UpdateJit(self._update_splash)
        try:
            from ..models.lib import instrument_numba, neutron_numba, offspec, paratt_numba, surface_scattering
        finally:
            numba.jit = real_jit

    def _setup_window_basics(self) -> None:
        self.setWindowTitle(f"GenX {program_version}")
        self.setMinimumSize(600, 400)

    def _setup_auto_color_menu(self) -> None:
        menu = self.ui.menuAutoColor
        menu.clear()
        group = QtGui.QActionGroup(self)
        group.setExclusive(True)
        self._auto_color_actions = {}

        for key in COLOR_CYCLES.keys():
            action = QtGui.QAction(key, self)
            action.setActionGroup(group)
            action.setCheckable(True)
            menu.addAction(action)
            action.triggered.connect(lambda _checked=False, k=key: self._on_auto_color_selected(k))
            self._auto_color_actions[key] = action
            self.mb_checkables[key] = _ActionCheckable(action)

    def _on_auto_color_selected(self, key: str) -> None:
        if not hasattr(self, "model_control"):
            return
        self.model_control.update_color_cycle(COLOR_CYCLES.get(key))

    def _setup_toolbar_icon_sizes(self) -> None:
        base = QtWidgets.QApplication.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ToolBarIconSize)
        main_size = QtCore.QSize(int(base * 1.4), int(base * 1.4))
        small_size = QtCore.QSize(int(base * 0.85), int(base * 0.85))

        self.ui.toolbarMain.setIconSize(main_size)
        self.ui.dataListControl.ui.toolbar.setIconSize(small_size)
        self.ui.paramterGrid.toolbar.setIconSize(small_size)

    def _setup_data_view(self) -> None:
        data_list_ctrl = self.ui.dataListControl
        data_grid_panel = self.ui.dataGridPanel
        data_list = data_list_ctrl.list_ctrl.data_cont.get_data()
        data_grid_panel.set_data_list(data_list)
        data_list_ctrl.list_ctrl.data_list_event.connect(data_grid_panel.on_data_list_event)

    def _setup_solver_gui(self) -> None:
        from .solvergui import ModelControlGUI
        self.model_control = ModelControlGUI(self)
        self.model_control.set_update_min_time(self.opt.solver_update_time)

        data_list_ctrl = self.ui.dataListControl
        self.model_control.set_data(data_list_ctrl.list_ctrl.data_cont.data)
        data_list_ctrl.list_ctrl.update_plotsettings.connect(self.model_control.update_plotsettings)
        data_list_ctrl.list_ctrl.data_list_event.connect(self._on_data_list_event)

        plot_data = self.ui.plotDataPanel
        plot_fom = self.ui.plotFomPanel
        plot_pars = self.ui.plotParsPanel
        self.model_control.update_plot.connect(plot_data.OnSolverPlotEvent)
        self.model_control.sim_plot.connect(plot_data.OnSimPlotEvent)
        self.model_control.update_plot.connect(plot_fom.OnSolverPlotEvent)
        self.model_control.update_parameters.connect(plot_pars.OnSolverParameterEvent)

        self.model_control.update_parameters.connect(self.model_control.OnUpdateParameters)
        self.model_control.update_text.connect(self._on_solver_text)
        self.model_control.update_script.connect(self._on_update_script)
        self.model_control.update_plot.connect(self._on_fitting_update)
        self.model_control.sim_plot.connect(self._on_simulate_update)

        self.set_script_text(self.model_control.get_model_script())

        input_tabs = self.ui.inputTabWidget
        self._last_input_tab = input_tabs.currentIndex()
        input_tabs.currentChanged.connect(self._on_input_tab_changed)

        param_grid = self.ui.paramterGrid
        layout = self.ui.inputGridLayout
        from .parametergrid import ParameterGrid

        self.paramter_grid = param_grid
        from .custom_events import SetParameterValueEvent, MoveParameterEvent
        param_grid.SetParameters(self.model_control.get_model_params())
        param_grid.SetFOMFunctions(self.project_fom_parameter, self.scan_parameter)
        param_grid.SetEvalFunc(self.model_control.eval_in_model)
        param_grid.SetSimulateFunc(self.simulate)
        self.ui.actionValueAsSlider.setChecked(param_grid.opt.value_slider)
        self._sync_auto_color_from_model()
        param_grid.set_parameter_value.connect(
            lambda row, col, value: self.model_control.OnSetParameterValue(
                SetParameterValueEvent(row=row, col=col, value=value)
            )
        )
        param_grid.move_parameter.connect(
            lambda row, step: self.model_control.OnMoveParameter(
                MoveParameterEvent(row=row, step=step)
            )
        )
        param_grid.insert_parameter.connect(self.model_control.OnInsertParameter)
        param_grid.delete_parameters.connect(self.model_control.OnDeleteParameter)
        param_grid.sort_and_group_parameters.connect(self.model_control.OnSortAndGroupParameters)
        param_grid.grid_changed.connect(self._on_parameter_grid_change)

    def _setup_plugin_control(self) -> None:
        menu_plugins = self.ui.menuPlugins
        from . import add_on_framework
        self.plugin_control = add_on_framework.PluginController(self, menu_plugins)
        self.plugin_control.LoadDefaultPlugins()

    def _auto_simulate_enabled(self) -> bool:
        return self.ui.actionSimulateAutomatically.isChecked()

    def _ensure_auto_simulate_timer(self) -> QtCore.QTimer:
        if self._auto_simulate_timer is None:
            timer = QtCore.QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(self._run_auto_simulate)
            self._auto_simulate_timer = timer
        return self._auto_simulate_timer

    def _schedule_auto_simulate(self) -> None:
        if not self._auto_simulate_enabled():
            return
        timer = self._ensure_auto_simulate_timer()
        timer.start(150)

    def _run_auto_simulate(self) -> None:
        if not self._auto_simulate_enabled():
            return
        with self.catch_error(action="auto_simulate", step="simulate"):
            self.simulate()

    def _sync_model_to_ui(self, *, update_plugins: bool = True) -> None:
        for panel in (
            self.ui.plotDataPanel,
            self.ui.plotFomPanel,
            self.ui.plotParsPanel,
            self.ui.plotFomScansPanel,
        ):
            if hasattr(panel, "ReadConfig"):
                panel.ReadConfig()
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is not None and hasattr(param_grid, "ReadConfig"):
            param_grid.ReadConfig()
            param_grid.SetParameters(self.model_control.get_model_params(), permanent_change=False)
        model = self.model_control.get_model()
        self.ui.dataListControl.eh_external_new_model(model)
        self.set_script_text(self.model_control.get_model_script())
        self.ui.scriptEditor.EmptyUndoBuffer()
        self.model_control.ModelLoaded()
        if update_plugins and hasattr(self, "plugin_control"):
            self.plugin_control.OnOpenModel(None)

    def _get_external_script_text(self) -> Optional[str]:
        if not self._external_script_file:
            return None
        try:
            return open(self._external_script_file, "r", encoding="utf-8").read()
        except OSError:
            return None

    def _write_external_script_text(self, text: str) -> None:
        if not self._external_script_file:
            return
        try:
            open(self._external_script_file, "w", encoding="utf-8").write(text)
        except OSError:
            warning("Could not write external script file", exc_info=True)

    def _start_external_script_watch(self) -> None:
        if self._script_watch_timer is None:
            self._script_watch_timer = QtCore.QTimer(self)
            self._script_watch_timer.timeout.connect(self._check_external_script)
        if not self._script_watch_timer.isActive():
            self._script_watch_timer.start(1000)

    def _stop_external_script_watch(self) -> None:
        if self._script_watch_timer is not None:
            self._script_watch_timer.stop()

    def _check_external_script(self) -> None:
        text = self._get_external_script_text()
        editor = self.ui.scriptEditor
        if text is not None and text.strip() != editor.toPlainText().strip():
            self.set_script_text(text, from_external=True)
            self.simulate()
        if self._external_editor_proc and self._external_editor_proc.poll() is not None:
            self._external_editor_proc = None
            self._stop_external_script_watch()
            res = ShowQuestionDialog(self, "Editor process exited, reactivate internal editor?", "Editor Closed")
            if res:
                self._deactivate_external_editing()
            else:
                self._start_external_script_watch()

    def _open_external_editor(self) -> None:
        editor = self.ui.scriptEditor
        with tempfile.NamedTemporaryFile(mode="w", prefix="genx", suffix=".py", encoding="utf-8", delete=False) as sfile:
            sfile.write(self.get_script_text())
            self._external_script_file = sfile.name
        if not self.opt.editor:
            if not self._select_external_editor():
                os.remove(self._external_script_file)
                self._external_script_file = None
                return
        try:
            proc = subprocess.Popen([self.opt.editor, self._external_script_file])
        except (subprocess.SubprocessError, OSError):
            os.remove(self._external_script_file)
            self._external_script_file = None
            self.opt.editor = None
            warning("Could not open editor", exc_info=True)
            return

        editor.setReadOnly(True)
        editor.setStyleSheet("QPlainTextEdit { background: #d2d2d2; }")
        self.ui.actionOpenInEditor.setText("Reactivate internal editor\tCtrl+E")
        self._external_editor_proc = proc
        self._start_external_script_watch()

    def _deactivate_external_editing(self) -> None:
        editor = self.ui.scriptEditor
        editor.setReadOnly(False)
        editor.setStyleSheet("")
        self.ui.actionOpenInEditor.setText("Open in Editor\tCtrl+E")
        self._stop_external_script_watch()
        self._external_editor_proc = None
        if self._external_script_file:
            try:
                os.remove(self._external_script_file)
            except OSError:
                warning("Could not remove external script file", exc_info=True)
            self._external_script_file = None

    def _select_external_editor(self) -> bool:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Editor Executable",
            "",
            "Executable (*.exe);;All files (*.*)",
        )
        if not path:
            return False
        self.opt.editor = path
        return True

    def _on_parameter_grid_change(self, permanent_change: bool) -> None:
        if self._init_phase:
            return
        if permanent_change:
            self.model_control.saved = False
            self._schedule_auto_simulate()
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnGridChanged(type("GridEvent", (), {"permanent_change": permanent_change})())

    def _sync_auto_color_from_model(self) -> None:
        if not hasattr(self, "model_control"):
            return
        current = self.model_control.get_color_cycle()
        colors2keys = dict((value, key) for key, value in COLOR_CYCLES.items())
        key = colors2keys.get(current, colors2keys.get(None))
        if key is None:
            return
        action = self._auto_color_actions.get(key) if hasattr(self, "_auto_color_actions") else None
        if action is not None:
            action.blockSignals(True)
            action.setChecked(True)
            action.blockSignals(False)

    def _on_solver_text(self, text: str) -> None:
        if text:
            self._status_update(text)

    def _on_update_script(self, script_text: str) -> None:
        self.set_script_text(script_text)

    def _on_data_list_event(self, event) -> None:
        self.ui.plotDataPanel.OnDataListEvent(event)
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnDataChanged(event)

    def scan_parameter(self, row: int) -> None:
        """
        Scan a parameter by stepping between min/max with user-defined points.
        """
        self.model_control.compile_if_needed()
        points, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Scan Parameter",
            "Steps",
            50,
            2,
            1000,
        )
        if not ok:
            return
        self._status_update("Scanning parameter")
        with self.catch_error(action="scan_parameters", step="scanning parameters"):
            x, y = self.model_control.ScanParameter(row, points)
            bestx = self.model_control.get_parameter_data(row)[1]
            besty = self.model_control.get_fom()
            e_scale = getattr(self.model_control.controller.optimizer.opt, "errorbar_level", 0)
            plot = self.ui.plotFomScansPanel
            plot.SetPlottype("scan")
            plot.Plot((x, y, bestx, besty, e_scale), self.model_control.get_parameter_name(row), "FOM")
            self.ui.plotTabWidget.setCurrentWidget(self.ui.plotTabFomScans)

    def project_fom_parameter(self, row: int) -> None:
        """
        Project FOM onto a parameter axis using stored evaluations.
        """
        import numpy as np

        if not self.model_control.IsFitted():
            ShowNotificationDialog(
                self,
                "Please conduct a fit before scanning a parameter. "
                "The script needs to be compiled and foms have to be collected.",
            )
            return
        self._status_update("Trying to project FOM")
        with self.catch_error(action="project_fom_parameters", step="projecting fom parameters"):
            e_scale = getattr(self.model_control.controller.optimizer.opt, "errorbar_level", None)
            if e_scale is None:
                ShowNotificationDialog(
                    self,
                    "This feature requires a fit with Differential Evolution, consider using FOM scan instead.",
                )
                return
            x, y = self.model_control.ProjectEvals(row)
            if len(x) == 0 or len(y) == 0:
                ShowNotificationDialog(
                    self,
                    "Please conduct a fit before projecting a parameter. "
                    "The script needs to be compiled and foms have to be collected.",
                )
                return
            if self.model_control.get_fom() is None or np.isnan(self.model_control.get_fom()):
                ShowNotificationDialog(self, "The model must be simulated (FOM is not a valid number)")
                return
            _fs, pars = self.model_control.get_sim_pars()
            bestx = pars[row]
            besty = self.model_control.get_fom()
            plot = self.ui.plotFomScansPanel
            plot.SetPlottype("project")
            plot.Plot((x, y, bestx, besty, e_scale), self.model_control.get_parameter_name(row), "FOM")
            self.ui.plotTabWidget.setCurrentWidget(self.ui.plotTabFomScans)

    def _on_fitting_update(self, event) -> None:
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnFittingUpdate(event)

    def _on_simulate_update(self, event) -> None:
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnSimulate(event)

    def set_script_text(self, text: str, from_external: bool = False) -> None:
        editor = self.ui.scriptEditor
        if editor.toPlainText() == text:
            return
        if not from_external and self._external_script_file is not None:
            self._write_external_script_text(text)
        cursor = editor.textCursor()
        vscroll = editor.verticalScrollBar().value()
        hscroll = editor.horizontalScrollBar().value()
        with QtCore.QSignalBlocker(editor):
            editor.setPlainText(text)
        editor.setTextCursor(cursor)
        editor.verticalScrollBar().setValue(vscroll)
        editor.horizontalScrollBar().setValue(hscroll)

    def get_script_text(self) -> str:
        editor = self.ui.scriptEditor
        if self._external_script_file:
            text = self._get_external_script_text()
            if text is not None:
                self.set_script_text(text, from_external=True)
                return text
        return editor.toPlainText()

    def _get_data_list_ctrl(self):
        return self.ui.dataListControl.list_ctrl

    def _get_active_plot_panel(self):
        tab = self.ui.plotTabWidget
        current = tab.currentWidget()
        mapping = {
            self.ui.plotTabData: self.ui.plotDataPanel,
            self.ui.plotTabFom: self.ui.plotFomPanel,
            self.ui.plotTabPars: self.ui.plotParsPanel,
            self.ui.plotTabFomScans: self.ui.plotFomScansPanel,
        }
        return mapping.get(current, None)

    def _update_for_save(self) -> None:
        self.model_control.set_model_script(self.get_script_text())
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is not None and hasattr(param_grid.opt, "save_config"):
            param_grid.opt.save_config(default=True)

    def _set_possible_parameters_in_grid(self) -> None:
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is None:
            return
        with self.catch_error(
            action="set_possible_parameters_in_grid",
            step="getting possible parameters",
            verbose=False,
        ) as mgr:
            par_dict = self.model_control.get_possible_parameters()
        if not mgr.successful:
            return
        param_grid.SetParameterSelections(par_dict)
        param_grid.SetEvalFunc(self.model_control.eval_in_model)

    def _on_input_tab_changed(self, index: int) -> None:
        if self._init_phase:
            self._last_input_tab = index
            return
        input_tabs = self.ui.inputTabWidget
        script_tab = self.ui.inputTabScript
        script_index = input_tabs.indexOf(script_tab)
        if self._last_input_tab == script_index and index != script_index:
            self.model_control.set_model_script(self.get_script_text())
            self.model_control.saved = False
        self._last_input_tab = index
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

        # mainSplitter: leftSplitter | rightSplitter
        _set_splitter_sizes(self.ui.mainSplitter, self.opt.vsplit)

        # rightSplitter: plotSplitter | inputTabWidget
        _set_splitter_sizes(self.ui.rightSplitter, self.opt.hsplit)

        # plotSplitter: plotTabWidget | pluginTabWidget
        _set_splitter_sizes(self.ui.plotSplitter, self.opt.psplit)

        # leftSplitter: leftTabWidget | leftPluginTabWidget
        _set_splitter_sizes(self.ui.leftSplitter, self.opt.lsplit)

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
        self.opt.lsplit = _first_size(self.ui.leftSplitter)

    def _setup_app_exception_dialogs(self) -> None:
        app = QtWidgets.QApplication.instance()
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
        self._setup_solver_gui()
        self._setup_plugin_control()
        if self._startup_filename:
            self.open_model(self._startup_filename)

    def new_model(self) -> None:
        if not self.model_control.saved:
            ans = ShowQuestionDialog(
                self,
                "If you continue any changes in your model will not be saved.",
                "Model not saved",
                yes_no=True,
            )
            if not ans:
                return

        self.model_control.new_model()
        model = self.model_control.get_model()
        self.ui.dataListControl.eh_external_new_model(model)
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is not None:
            param_grid.SetParameters(self.model_control.get_model_params())
        self.set_script_text(self.model_control.get_model_script())
        self.ui.scriptEditor.EmptyUndoBuffer()
        self.model_control.ModelLoaded()
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnNewModel(None)
        self._status_update("New model created")

    def new_from_file(self) -> None:
        pass

    def open_model(self, path: str) -> None:
        if not self.model_control.saved:
            ans = ShowQuestionDialog(
                self,
                "If you continue any changes in your model will not be saved.",
                "Model not saved",
                yes_no=True,
            )
            if not ans:
                return

        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                "Open",
                "",
                "GenX File (*.hgx *.gx)",
            )
            if not path:
                return

        self.model_control.new_model()
        with self.catch_error(action="open_model", step=f"open file {os.path.basename(path)}") as mng:
            self.model_control.load_file(path)
        if not mng.successful:
            return
        self._sync_model_to_ui(update_plugins=True)
        self._status_update("Model loaded from file")

    def save_model(self) -> None:
        self._update_for_save()
        fname = self.model_control.get_filename()
        if not fname:
            self.save_model_as()
            return
        with self.catch_error(action="save_model", step=f"save file {os.path.basename(fname)}"):
            if len(self.model_control.controller.model_store) > 0:
                prog = QtWidgets.QProgressDialog(
                    f"Writing to file\n{fname}\n",
                    "Cancel",
                    0,
                    100,
                    self,
                )
                prog.setWindowTitle("Saving...")
                prog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

                def update_callback(i, n_total):
                    prog.setValue(int(i / n_total * 100))
                    prog.setLabelText(f"Writing to file\n{fname}\ndataset {i} of {n_total}")
                    QtWidgets.QApplication.processEvents()

                try:
                    self.model_control.controller.save_file(fname, update_callback=update_callback)
                finally:
                    prog.close()
            else:
                self.model_control.controller.save_file(fname)
        self._status_update("Model saved")

    def save_model_as(self) -> None:
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save As",
            "",
            "HDF5 GenX File (*.hgx);;GenX File (*.gx)",
        )
        if not fname:
            return
        self._update_for_save()
        base, ext = os.path.splitext(fname)
        if ext == "":
            ext = ".hgx"
        fname = base + ext
        if os.path.exists(fname):
            result = ShowQuestionDialog(
                self,
                f"The file {os.path.basename(fname)} already exists. Do you wish to overwrite it?",
                "Overwrite?",
                yes_no=True,
            )
            if not result:
                return
        with self.catch_error(action="saveas", step=f"saveing file as {os.path.basename(fname)}"):
            if len(self.model_control.controller.model_store) > 0:
                prog = QtWidgets.QProgressDialog(
                    f"Writing to file\n{fname}\n",
                    "Cancel",
                    0,
                    100,
                    self,
                )
                prog.setWindowTitle("Saving...")
                prog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

                def update_callback(i, n_total):
                    prog.setValue(int(i / n_total * 100))
                    prog.setLabelText(f"Writing to file\n{fname}\ndataset {i} of {n_total}")
                    QtWidgets.QApplication.processEvents()

                try:
                    self.model_control.controller.save_file(fname, update_callback=update_callback)
                finally:
                    prog.close()
            else:
                self.model_control.controller.save_file(fname)
        self._status_update("Model saved")

    def simulate(self) -> None:
        self.model_control.set_model_script(self.get_script_text())
        with self.catch_error(action="simulate", step=f"simulating the model") as mgr:
            self.model_control.simulate()
        self._set_possible_parameters_in_grid()

    def evaluate(self) -> None:
        self.model_control.set_model_script(self.get_script_text())
        with self.catch_error(action="evaluate", step=f"evaluate the model") as mgr:
            self.model_control.evaluate()
        self._set_possible_parameters_in_grid()

    def start_fit(self) -> None:
        self.model_control.set_model_script(self.get_script_text())
        with self.catch_error(action="fit_start", step=f"starting fit"):
            self.model_control.StartFit()

    def stop_fit(self) -> None:
        self.model_control.StopFit()

    def resume_fit(self) -> None:
        self.model_control.ResumeFit()

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

    @QtCore.Slot(bool)
    def on_actionCollectDebugInfo_triggered(self, checked: bool = False) -> None:
        import logging

        from ..core import custom_logging
        from .log_dialog import LoggingDialog

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Logfile As",
            "genx.log",
            "GenX logfile (*.log)",
        )
        if filename:
            if os.path.exists(filename):
                res = QtWidgets.QMessageBox.question(
                    self,
                    "Overwrite file?",
                    f"File {os.path.basename(filename)} exists, overwrite?",
                )
                if res == QtWidgets.QMessageBox.StandardButton.Yes:
                    custom_logging.activate_logging(filename)
            else:
                custom_logging.activate_logging(filename)

        dlg = LoggingDialog(self)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._logging_dialogs.append(dlg)
        dlg.destroyed.connect(lambda _=None, d=dlg: self._logging_dialogs.remove(d) if d in self._logging_dialogs else None)
        dlg.show()

    # Optional: handle checkable actions (stubs)
    @QtCore.Slot(bool)
    def on_actionSimulateAutomatically_triggered(self, checked: bool) -> None:
        if checked:
            self._schedule_auto_simulate()

    @QtCore.Slot(bool)
    def on_actionUseCuda_triggered(self, checked: bool) -> None:
        pass

    @QtCore.Slot(bool)
    def on_actionImportTable_triggered(self, checked: bool = False) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import table",
            "",
            "Table File (*.tab);;All files (*.*)",
        )
        if not path:
            return
        with self.catch_error(action="import_table", step=f"importing table {os.path.basename(path)}") as mgr:
            self.model_control.import_table(path)
        if not mgr.successful:
            return
        self._sync_model_to_ui(update_plugins=False)
        self._status_update("Table imported from file")

    @QtCore.Slot(bool)
    def on_actionImportScript_triggered(self, checked: bool = False) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import script",
            "",
            "Python files (*.py);;All files (*.*)",
        )
        if not path:
            return
        with self.catch_error(action="import_script", step=f"importing file {os.path.basename(path)}"):
            self.model_control.import_script(path)
        self._sync_model_to_ui(update_plugins=True)

    @QtCore.Slot(bool)
    def on_actionExportOrso_triggered(self, checked: bool = False) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export data and model",
            "",
            "ORSO Text File (*.ort)",
        )
        if not path:
            return
        res = ShowQuestionDialog(
            self,
            "Convert TTH to Q (ORSO specification)?",
            "Convert to Q",
            yes_no=True,
        )
        with self.catch_error(action="export_orso", step=f"export file {os.path.basename(path)}"):
            self.model_control.export_orso(path, convert_to_q=bool(res))

    @QtCore.Slot(bool)
    def on_actionExportData_triggered(self, checked: bool = False) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export data",
            "",
            "Dat File (*.dat)",
        )
        if not path:
            return
        with self.catch_error(action="export_data", step=f"data file {os.path.basename(path)}"):
            self.model_control.export_data(path)

    @QtCore.Slot(bool)
    def on_actionExportTable_triggered(self, checked: bool = False) -> None:
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export table",
            "",
            "Table File (*.tab)",
        )
        if not fname:
            return
        base, ext = os.path.splitext(fname)
        if ext == "":
            ext = ".tab"
        fname = base + ext
        if os.path.exists(fname):
            result = ShowQuestionDialog(
                self,
                f"The file {os.path.basename(fname)} already exists. Do you wish to overwrite it?",
                "Overwrite?",
                yes_no=True,
            )
            if not result:
                return
        with self.catch_error(action="export_table", step=f"table file {os.path.basename(fname)}"):
            self.model_control.export_table(fname)

    @QtCore.Slot(bool)
    def on_actionExportScript_triggered(self, checked: bool = False) -> None:
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export script",
            "",
            "Python File (*.py)",
        )
        if not fname:
            return
        base, ext = os.path.splitext(fname)
        if ext == "":
            ext = ".py"
        fname = base + ext
        if os.path.exists(fname):
            result = ShowQuestionDialog(
                self,
                f"The file {os.path.basename(fname)} already exists. Do you wish to overwrite it?",
                "Overwrite?",
                yes_no=True,
            )
            if not result:
                return
        with self.catch_error(action="export_script", step=f"export file {os.path.basename(fname)}"):
            self.model_control.export_script(fname)

    @QtCore.Slot(bool)
    def on_actionImportData_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.LoadData()

    @QtCore.Slot(bool)
    def on_actionAddData_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.AddItem()

    @QtCore.Slot(bool)
    def on_actionAddSimulation_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.CreateSimData()

    @QtCore.Slot(bool)
    def on_actionDataInfo_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.ShowInfo()

    @QtCore.Slot(bool)
    def on_actionDelete_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.DeleteItem()

    @QtCore.Slot(bool)
    def on_actionMoveUp_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.MoveItemUp()

    @QtCore.Slot(bool)
    def on_actionMoveDown_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.MoveItemDown()

    @QtCore.Slot(bool)
    def on_actionPlotting_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.OnPlotSettings()

    @QtCore.Slot(bool)
    def on_actionCalc_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.OnCalcEdit()

    @QtCore.Slot(bool)
    def on_actionNewDataSet_triggered(self, checked: bool = False) -> None:
        self.on_actionAddData_triggered(checked)

    @QtCore.Slot(bool)
    def on_actionDeleteDataSet_triggered(self, checked: bool = False) -> None:
        self.on_actionDelete_triggered(checked)

    @QtCore.Slot(bool)
    def on_actionLowerData_triggered(self, checked: bool = False) -> None:
        self.on_actionMoveDown_triggered(checked)

    @QtCore.Slot(bool)
    def on_actionRaiseData_triggered(self, checked: bool = False) -> None:
        self.on_actionMoveUp_triggered(checked)

    @QtCore.Slot(bool)
    def on_actionToggleShow_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.OnShowData()

    @QtCore.Slot(bool)
    def on_actionToggleUse_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.OnUseData()

    @QtCore.Slot(bool)
    def on_actionToggleError_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.OnUseError()

    @QtCore.Slot(bool)
    def on_actionCalculations_triggered(self, checked: bool = False) -> None:
        self.on_actionCalc_triggered(checked)

    @QtCore.Slot(bool)
    def on_actionPlotMarkers_triggered(self, checked: bool = False) -> None:
        self.on_actionPlotting_triggered(checked)

    @QtCore.Slot(bool)
    def on_actionUseToggleShow_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.SetShowToggle(checked)

    @QtCore.Slot(bool)
    def on_actionOptimizer_triggered(self, checked: bool = False) -> None:
        self.model_control.ParametersDialog(self)

    @QtCore.Slot(bool)
    def on_actionUndo_triggered(self, checked: bool = False) -> None:
        self.model_control.OnUndo()

    @QtCore.Slot(bool)
    def on_actionRedo_triggered(self, checked: bool = False) -> None:
        self.model_control.OnRedo()

    @QtCore.Slot(bool)
    def on_actionHistory_triggered(self, checked: bool = False) -> None:
        self.model_control.OnShowHistory()

    @QtCore.Slot(bool)
    def on_actionCopyGraph_triggered(self, checked: bool = False) -> None:
        panel = self._get_active_plot_panel()
        if panel is not None:
            panel.CopyToClipboard()

    @QtCore.Slot(bool)
    def on_actionCopySimulation_triggered(self, checked: bool = False) -> None:
        text_string = self.model_control.get_data_as_asciitable()
        QtWidgets.QApplication.clipboard().setText(text_string)

    @QtCore.Slot(bool)
    def on_actionCopyTable_triggered(self, checked: bool = False) -> None:
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is None:
            return
        pars = getattr(param_grid, "_pars", None)
        if pars is None or not hasattr(pars, "get_ascii_output"):
            return
        QtWidgets.QApplication.clipboard().setText(pars.get_ascii_output())

    @QtCore.Slot(bool)
    def on_actionFindReplace_triggered(self, checked: bool = False) -> None:
        editor = self.ui.scriptEditor
        find_text, ok = QtWidgets.QInputDialog.getText(self, "Find", "Find text:")
        if not ok or not find_text:
            return
        action = QtWidgets.QMessageBox.question(
            self,
            "Find/Replace",
            "Replace all occurrences?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if action == QtWidgets.QMessageBox.StandardButton.Yes:
            replace_text, ok = QtWidgets.QInputDialog.getText(self, "Replace", "Replace with:")
            if not ok:
                return
            text = editor.toPlainText()
            if find_text not in text:
                ShowNotificationDialog(self, f"Could not find text {find_text}")
                return
            self.set_script_text(text.replace(find_text, replace_text))
        else:
            if not editor.find(find_text):
                ShowNotificationDialog(self, f"Could not find text {find_text}")

    @QtCore.Slot(bool)
    def on_actionOpenInEditor_triggered(self, checked: bool = False) -> None:
        if self._external_script_file is not None:
            self._deactivate_external_editing()
            return
        self._open_external_editor()

    @QtCore.Slot(bool)
    def on_actionPrintPlot_triggered(self, checked: bool = False) -> None:
        panel = self._get_active_plot_panel()
        if panel is not None:
            panel.Print()

    @QtCore.Slot(bool)
    def on_actionPrintGrid_triggered(self, checked: bool = False) -> None:
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is None:
            return
        printer = QtPrintSupport.QPrinter()
        dlg = QtPrintSupport.QPrintDialog(printer, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        painter = QtGui.QPainter(printer)
        if not painter.isActive():
            return
        rect = painter.viewport()
        size = param_grid.size()
        if size.width() > 0 and size.height() > 0:
            scale = min(rect.width() / size.width(), rect.height() / size.height())
            painter.scale(scale, scale)
        param_grid.render(painter)
        painter.end()

    @QtCore.Slot(bool)
    def on_actionPrintScript_triggered(self, checked: bool = False) -> None:
        editor = self.ui.scriptEditor
        printer = QtPrintSupport.QPrinter()
        dlg = QtPrintSupport.QPrintDialog(printer, self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        editor.print(printer)

    @QtCore.Slot(bool)
    def on_actionPublishPlot_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="menu", step="publish plot"):
            from .pubgraph_dialog import PublicationDialog

            script_module = self.model_control.get_model()
            dia = PublicationDialog(
                self,
                data=self.model_control.get_data(),
                module=script_module.eval_in_model('globals().get("model")'),
                SLD=script_module.eval_in_model('globals().get("SLD", [])'),
            )
            dia.exec()

    @QtCore.Slot(bool)
    def on_actionDataLoader_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.ChangeDataLoader()

    @QtCore.Slot(bool)
    def on_actionImportSettings_triggered(self, checked: bool = False) -> None:
        ctrl = self._get_data_list_ctrl()
        if ctrl is not None:
            ctrl.OnImportSettings()

    @QtCore.Slot(bool)
    def on_actionSelectExternalEditor_triggered(self, checked: bool = False) -> None:
        self._select_external_editor()

    @QtCore.Slot(bool)
    def on_actionStartupProfile_triggered(self, checked: bool = False) -> None:
        ShowNotificationDialog(self, "Startup profile dialog is not yet available in the Qt UI.")

    @QtCore.Slot(bool)
    def on_actionBatchDialog_triggered(self, checked: bool = False) -> None:
        from .batch_dialog import BatchDialog

        dia = BatchDialog(self, self.model_control)
        size = self.size()
        dia.resize(int(size.width() / 3), int(size.height() * 0.8))
        dia.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dia.show()

    @QtCore.Slot(bool)
    def on_actionErrorStatistics_triggered(self, checked: bool = False) -> None:
        from .bumps_interface import StatisticalAnalysisDialog

        dia = StatisticalAnalysisDialog(self, self.model_control.get_model())
        dia.exec()

    @QtCore.Slot(bool)
    def on_actionCalcErrorBars_triggered(self, checked: bool = False) -> None:
        with self.catch_error(action="calc_error_bars", step="calculating errorbars"):
            error_values = self.model_control.CalcErrorBars()
            self.model_control.set_error_pars(error_values)
            param_grid = getattr(self, "paramter_grid", None)
            if param_grid is not None:
                param_grid.SetParameters(self.model_control.get_parameters())
            self._status_update("Errorbars calculated")

    @QtCore.Slot(bool)
    def on_actionOpenExamples_triggered(self, checked: bool = False) -> None:
        examples_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "")
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Example",
            examples_dir,
            "GenX File (*.hgx *.gx *.ort *.orb)",
        )
        if not path:
            return
        if path.endswith(".ort") or path.endswith(".orb"):
            ShowNotificationDialog(self, "ORSO file import is not yet available in the Qt UI.")
            return
        self.open_model(path)

    @QtCore.Slot(bool)
    def on_actionOpenManual_triggered(self, checked: bool = False) -> None:
        webbrowser.open_new(manual_url)

    @QtCore.Slot(bool)
    def on_actionOpenHomepage_triggered(self, checked: bool = False) -> None:
        webbrowser.open_new(homepage_url)

    @QtCore.Slot(bool)
    def on_actionAbout_triggered(self, checked: bool = False) -> None:
        import platform

        versions = []
        missing = []
        try:
            import numpy
            versions.append(f"NumPy: {numpy.__version__}")
        except ImportError:
            missing.append("numpy")
        try:
            import scipy
            versions.append(f"SciPy: {scipy.__version__}")
        except ImportError:
            missing.append("scipy")
        try:
            import matplotlib
            versions.append(f"Matplotlib: {matplotlib.__version__}")
        except ImportError:
            missing.append("matplotlib")
        try:
            import orsopy
            versions.append(f"ORSOpy: {orsopy.__version__}")
        except ImportError:
            missing.append("orsopy")
        try:
            import numba
            versions.append(f"Numba: {numba.__version__}")
        except ImportError:
            missing.append("numba")
        try:
            import vtk
            versions.append(f"VTK: {vtk.__version__}")
        except ImportError:
            missing.append("vtk")

        info_lines = [
            f"GenX {program_version}",
            f"Python {platform.python_version()}",
        ] + versions
        if missing:
            info_lines.append("Missing: " + ", ".join(sorted(missing)))
        QtWidgets.QMessageBox.about(self, "About GenX", "\n".join(info_lines))

    @QtCore.Slot(bool)
    def on_actionModelsHelp_triggered(self, checked: bool = False) -> None:
        from .help import PluginHelpDialog

        dlg = PluginHelpDialog(self, "models", title="Models help")
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._help_dialogs.append(dlg)
        dlg.destroyed.connect(lambda _=None, d=dlg: self._help_dialogs.remove(d) if d in self._help_dialogs else None)
        dlg.show()

    @QtCore.Slot(bool)
    def on_actionFomHelp_triggered(self, checked: bool = False) -> None:
        from .help import PluginHelpDialog

        dlg = PluginHelpDialog(self, "fom_funcs", title="FOM functions help")
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._help_dialogs.append(dlg)
        dlg.destroyed.connect(lambda _=None, d=dlg: self._help_dialogs.remove(d) if d in self._help_dialogs else None)
        dlg.show()

    @QtCore.Slot(bool)
    def on_actionPluginsHelp_triggered(self, checked: bool = False) -> None:
        from .help import PluginHelpDialog

        dlg = PluginHelpDialog(self, "gui_qt.add_ons", title="Plugins help")
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._help_dialogs.append(dlg)
        dlg.destroyed.connect(lambda _=None, d=dlg: self._help_dialogs.remove(d) if d in self._help_dialogs else None)
        dlg.show()

    @QtCore.Slot(bool)
    def on_actionDataLoadersHelp_triggered(self, checked: bool = False) -> None:
        from .help import PluginHelpDialog

        dlg = PluginHelpDialog(self, "plugins.data_loaders", title="Data loaders help")
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._help_dialogs.append(dlg)
        dlg.destroyed.connect(lambda _=None, d=dlg: self._help_dialogs.remove(d) if d in self._help_dialogs else None)
        dlg.show()

    @QtCore.Slot(bool)
    def on_actionAnalyzeFit_triggered(self, checked: bool = False) -> None:
        warning("Event handler `on_actionAnalyzeFit_triggered` not implemented")

    @QtCore.Slot(bool)
    def on_actionValueAsSlider_triggered(self, checked: bool = False) -> None:
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is not None:
            param_grid.SetValueEditorSlider(bool(checked))

    def check_for_update(self) -> None:
        from .online_update import VersionInfoDialog, check_version

        same_version = True
        with self.catch_error(action="update_check", step="check_version"):
            same_version = check_version()
            self.opt.last_update_check = time.time()
        if same_version:
            return
        with self.catch_error(action="update_check", step="show_dialog"):
            dia = VersionInfoDialog(self)
            res = dia.exec()
            if res == VersionInfoDialog.RESULT_RESTART:
                ShowNotificationDialog(self, "You need to restart GenX for the changes to take effect.")
            elif res == VersionInfoDialog.RESULT_QUIT:
                self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        # Persist window/splitter sizes
        try:
            self._store_gui_config_from_widgets()
            self.opt.save_config(default=True)
            conf_mod.config.write_default(os.path.join(config_path, "genx.conf"))
        except Exception:
            debug("Could not persist GUIConfig on close", exc_info=True)

        if self._external_script_file:
            self._deactivate_external_editing()
        super().closeEvent(event)


def start_qt_app(*, filename: Optional[str], debug: bool = False) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(":/main_gui/genx.ico"))

    win = GenxMainWindow(filename=filename)
    win.show()
    win._finish_splash()
    if time.time() - win.opt.last_update_check > (7 * 24 * 3600):
        QtCore.QTimer.singleShot(1000, win.check_for_update)

    sys.exit(app.exec())
