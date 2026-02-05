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
from .message_dialogs import ShowQuestionDialog


from ..core import config as conf_mod
from ..core.colors import COLOR_CYCLES
from ..core.custom_logging import iprint, numpy_set_options
from . import solvergui
from . import add_on_framework as add_on
from .parametergrid import ParameterGrid
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
        self._logging_dialogs = []
        self._last_input_tab = None

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
        self._setup_data_view()
        self._setup_window_basics()
        self._setup_app_exception_dialogs()
        self._setup_toolbar_icon_sizes()

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

    def _setup_toolbar_icon_sizes(self) -> None:
        base = QtWidgets.QApplication.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ToolBarIconSize)
        main_size = QtCore.QSize(int(base * 1.4), int(base * 1.4))
        small_size = QtCore.QSize(int(base * 0.85), int(base * 0.85))

        main_toolbar = getattr(self.ui, "toolbarMain", None)
        if main_toolbar is not None:
            main_toolbar.setIconSize(main_size)

        data_list_ctrl = getattr(self.ui, "dataListControl", None)
        if data_list_ctrl is not None and hasattr(data_list_ctrl, "ui"):
            data_toolbar = getattr(data_list_ctrl.ui, "toolbar", None)
            if data_toolbar is not None:
                data_toolbar.setIconSize(small_size)

        param_grid = getattr(self.ui, "paramterGrid", None)
        if param_grid is not None and hasattr(param_grid, "toolbar"):
            param_grid.toolbar.setIconSize(small_size)

    def _setup_data_view(self) -> None:
        data_list_ctrl = getattr(self.ui, "dataListControl", None)
        data_grid_panel = getattr(self.ui, "dataGridPanel", None)
        if data_list_ctrl is None or data_grid_panel is None:
            return
        data_list = data_list_ctrl.list_ctrl.data_cont.get_data()
        data_grid_panel.set_data_list(data_list)
        data_list_ctrl.list_ctrl.data_list_event.connect(data_grid_panel.on_data_list_event)

    def _setup_solver_gui(self) -> None:
        self.model_control = solvergui.ModelControlGUI(self)
        self.model_control.set_update_min_time(self.opt.solver_update_time)

        data_list_ctrl = getattr(self.ui, "dataListControl", None)
        if data_list_ctrl is not None:
            self.model_control.set_data(data_list_ctrl.list_ctrl.data_cont.data)
            data_list_ctrl.list_ctrl.update_plotsettings.connect(self.model_control.update_plotsettings)
            data_list_ctrl.list_ctrl.data_list_event.connect(self._on_data_list_event)

        plot_data = getattr(self.ui, "plotDataPanel", None)
        plot_fom = getattr(self.ui, "plotFomPanel", None)
        plot_pars = getattr(self.ui, "plotParsPanel", None)
        if plot_data is not None:
            self.model_control.update_plot.connect(plot_data.OnSolverPlotEvent)
            self.model_control.sim_plot.connect(plot_data.OnSimPlotEvent)
        if plot_fom is not None:
            self.model_control.update_plot.connect(plot_fom.OnSolverPlotEvent)
        if plot_pars is not None:
            self.model_control.update_parameters.connect(plot_pars.OnSolverParameterEvent)

        self.model_control.update_parameters.connect(self.model_control.OnUpdateParameters)
        self.model_control.update_text.connect(self._on_solver_text)
        self.model_control.update_script.connect(self._on_update_script)
        self.model_control.update_plot.connect(self._on_fitting_update)
        self.model_control.sim_plot.connect(self._on_simulate_update)

        self.set_script_text(self.model_control.get_model_script())

        input_tabs = getattr(self.ui, "inputTabWidget", None)
        if input_tabs is not None:
            self._last_input_tab = input_tabs.currentIndex()
            input_tabs.currentChanged.connect(self._on_input_tab_changed)

        param_grid = getattr(self.ui, "paramterGrid", None)
        if param_grid is None:
            layout = getattr(self.ui, "inputGridLayout", None)
            if layout is not None:
                param_grid = ParameterGrid(self)
                layout.addWidget(param_grid)
        if param_grid is not None:
            self.paramter_grid = param_grid
            param_grid.SetParameters(self.model_control.get_model_params())
            param_grid.SetFOMFunctions(self.model_control.ProjectEvals, self.model_control.ScanParameter)
            param_grid.SetEvalFunc(self.model_control.eval_in_model)
            param_grid.SetSimulateFunc(self.simulate)
            param_grid.set_parameter_value.connect(
                lambda row, col, value: self.model_control.OnSetParameterValue(
                    solvergui.SetParameterValueEvent(row=row, col=col, value=value)
                )
            )
            param_grid.move_parameter.connect(
                lambda row, step: self.model_control.OnMoveParameter(
                    solvergui.MoveParameterEvent(row=row, step=step)
                )
            )
            param_grid.insert_parameter.connect(self.model_control.OnInsertParameter)
            param_grid.delete_parameters.connect(self.model_control.OnDeleteParameter)
            param_grid.sort_and_group_parameters.connect(self.model_control.OnSortAndGroupParameters)
            param_grid.grid_changed.connect(self._on_parameter_grid_change)

    def _setup_plugin_control(self) -> None:
        menu_plugins = getattr(self.ui, "menuPlugins", None)
        if menu_plugins is None:
            return
        self.plugin_control = add_on.PluginController(self, menu_plugins)
        self.plugin_control.LoadDefaultPlugins()

    def _on_parameter_grid_change(self, permanent_change: bool) -> None:
        if self._init_phase:
            return
        if permanent_change:
            self.model_control.saved = False
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnGridChanged(type("GridEvent", (), {"permanent_change": permanent_change})())

    def _on_solver_text(self, text: str) -> None:
        if text:
            self._status_update(text)

    def _on_update_script(self, script_text: str) -> None:
        self.set_script_text(script_text)

    def _on_data_list_event(self, event) -> None:
        plot_data = getattr(self.ui, "plotDataPanel", None)
        if plot_data is not None:
            plot_data.OnDataListEvent(event)
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnDataChanged(event)

    def _on_fitting_update(self, event) -> None:
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnFittingUpdate(event)

    def _on_simulate_update(self, event) -> None:
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnSimulate(event)

    def set_script_text(self, text: str) -> None:
        editor = getattr(self.ui, "scriptEditor", None)
        if editor is None:
            return
        if editor.toPlainText() == text:
            return
        cursor = editor.textCursor()
        vscroll = editor.verticalScrollBar().value()
        hscroll = editor.horizontalScrollBar().value()
        with QtCore.QSignalBlocker(editor):
            editor.setPlainText(text)
        editor.setTextCursor(cursor)
        editor.verticalScrollBar().setValue(vscroll)
        editor.horizontalScrollBar().setValue(hscroll)

    def get_script_text(self) -> str:
        editor = getattr(self.ui, "scriptEditor", None)
        if editor is None:
            return ""
        return editor.toPlainText()

    def _get_data_list_ctrl(self):
        return getattr(getattr(self.ui, "dataListControl", None), "list_ctrl", None)

    def _get_active_plot_panel(self):
        tab = getattr(self.ui, "plotTabWidget", None)
        if tab is None:
            return None
        current = tab.currentWidget()
        if current is None:
            return None
        mapping = {
            getattr(self.ui, "plotTabData", None): getattr(self.ui, "plotDataPanel", None),
            getattr(self.ui, "plotTabFom", None): getattr(self.ui, "plotFomPanel", None),
            getattr(self.ui, "plotTabPars", None): getattr(self.ui, "plotParsPanel", None),
            getattr(self.ui, "plotTabFomScans", None): getattr(self.ui, "plotFomScansPanel", None),
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
        input_tabs = getattr(self.ui, "inputTabWidget", None)
        script_tab = getattr(self.ui, "inputTabScript", None)
        if input_tabs is None or script_tab is None:
            self._last_input_tab = index
            return
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
        data_list_ctrl = getattr(self.ui, "dataListControl", None)
        if data_list_ctrl is not None:
            data_list_ctrl.eh_external_new_model(model)
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is not None:
            param_grid.SetParameters(self.model_control.get_model_params())
        self.set_script_text(self.model_control.get_model_script())
        editor = getattr(self.ui, "scriptEditor", None)
        if editor is not None and hasattr(editor, "EmptyUndoBuffer"):
            editor.EmptyUndoBuffer()
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

        for panel in (
            getattr(self.ui, "plotDataPanel", None),
            getattr(self.ui, "plotFomPanel", None),
            getattr(self.ui, "plotParsPanel", None),
            getattr(self.ui, "plotFomScansPanel", None),
        ):
            if panel is not None and hasattr(panel, "ReadConfig"):
                panel.ReadConfig()
        param_grid = getattr(self, "paramter_grid", None)
        if param_grid is not None and hasattr(param_grid, "ReadConfig"):
            param_grid.ReadConfig()
            param_grid.SetParameters(self.model_control.get_model_params(), permanent_change=False)

        model = self.model_control.get_model()
        data_list_ctrl = getattr(self.ui, "dataListControl", None)
        if data_list_ctrl is not None:
            data_list_ctrl.eh_external_new_model(model)
        self.set_script_text(self.model_control.get_model_script())
        editor = getattr(self.ui, "scriptEditor", None)
        if editor is not None and hasattr(editor, "EmptyUndoBuffer"):
            editor.EmptyUndoBuffer()
        self.model_control.ModelLoaded()
        if hasattr(self, "plugin_control"):
            self.plugin_control.OnOpenModel(None)
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
        self.model_control.simulate()
        self._set_possible_parameters_in_grid()

    def evaluate(self) -> None:
        self.model_control.set_model_script(self.get_script_text())
        self.model_control.evaluate()
        self._set_possible_parameters_in_grid()

    def start_fit(self) -> None:
        self.model_control.set_model_script(self.get_script_text())
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
        pass

    @QtCore.Slot(bool)
    def on_actionUseCuda_triggered(self, checked: bool) -> None:
        pass

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
    def on_actionPrintPlot_triggered(self, checked: bool = False) -> None:
        panel = self._get_active_plot_panel()
        if panel is not None:
            panel.Print()

    @QtCore.Slot(bool)
    def on_actionPublishPlot_triggered(self, checked: bool = False) -> None:
        panel = self._get_active_plot_panel()
        if panel is not None:
            panel.PrintPreview()

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
