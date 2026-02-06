"""
Qt port of solvergui from wx.
Uses Qt signals/slots and dataclasses for event payloads.
"""

import time
from dataclasses import dataclass
from logging import debug
from threading import Event, Thread
from typing import TYPE_CHECKING, Union

import numpy as np
from PySide6 import QtCore, QtWidgets

from .. import diffev, fom_funcs, levenberg_marquardt, model_control
from ..core.colors import COLOR_CYCLES
from ..core.custom_logging import iprint
from ..model_actions import ModelAction, ModelInfluence
from ..remote import optimizer as remote_optimizer
from ..solver_basis import GenxOptimizerCallback, SolverParameterInfo, SolverResultInfo, SolverUpdateInfo
from .exception_handling import CatchModelError
from .history_dialog import HistoryDialog
from .message_dialogs import ShowErrorDialog, ShowQuestionDialog, ShowWarningDialog
from .settings_dialog import SettingsDialog
from .utils import ShowInfoDialog

if TYPE_CHECKING:
    from . import main_window


@dataclass(frozen=True)
class UpdatePlotEvent:
    data: object | None = None
    fom_value: float | None = None
    fom_name: str | None = None
    fom_log: object | None = None
    update_fit: bool = False
    desc: str = ""
    model: object | None = None


@dataclass(frozen=True)
class UpdateParametersEvent:
    values: object
    new_best: bool
    population: object
    max_val: object
    min_val: object
    fitting: bool
    desc: str
    update_errors: bool = False
    permanent_change: bool = False


@dataclass(frozen=True)
class FittingEndedEvent:
    start_guess: object
    error_message: str | None
    values: object
    new_best: bool
    population: object
    max_val: object
    min_val: object
    fitting: bool
    desc: str


@dataclass(frozen=True)
class BatchNextEvent:
    last_index: int
    finished: bool


@dataclass(frozen=True)
class UpdatePlotSettingsEvent:
    indices: list[int]
    sim_par: dict
    data_par: dict


@dataclass(frozen=True)
class SetParameterValueEvent:
    row: int
    col: int
    value: object


@dataclass(frozen=True)
class MoveParameterEvent:
    row: int
    step: int


class GuiCallbacks(GenxOptimizerCallback):
    echo = True

    def __init__(self, signal_target: QtCore.QObject):
        self.signal_target = signal_target

    def _emit(self, signal_name: str, event) -> None:
        signal = getattr(self.signal_target, signal_name, None)
        if signal is not None:
            signal.emit(event)

    def text_output(self, text):
        if self.echo:
            debug(f"User Info: {text}", stacklevel=3)
        self._emit("update_text", text)

    def plot_output(self, update_data: SolverUpdateInfo):
        self._emit(
            "update_plot",
            UpdatePlotEvent(
                data=update_data.data,
                fom_value=update_data.fom_value,
                fom_name=update_data.fom_name,
                fom_log=update_data.fom_log,
                update_fit=update_data.new_best,
                desc="Fitting update",
            ),
        )

    def parameter_output(self, param_info: SolverParameterInfo):
        self._emit(
            "update_parameters",
            UpdateParametersEvent(
                values=param_info.values,
                new_best=param_info.new_best,
                population=param_info.population,
                max_val=param_info.max_val,
                min_val=param_info.min_val,
                fitting=True,
                desc="Parameter Update",
                update_errors=False,
                permanent_change=False,
            ),
        )

    def fitting_ended(self, result_data: SolverResultInfo):
        self._emit(
            "fitting_ended",
            FittingEndedEvent(
                start_guess=result_data.start_guess,
                error_message=result_data.error_message,
                values=result_data.values,
                new_best=result_data.new_best,
                population=result_data.population,
                max_val=result_data.max_val,
                min_val=result_data.min_val,
                fitting=True,
                desc="Fitting Ended",
            ),
        )

    def autosave(self):
        signal = getattr(self.signal_target, "autosave", None)
        if signal is not None:
            signal.emit()


class DelayedCallbacks(Thread, GuiCallbacks):
    last_text: Union[list, None] = None
    last_param: Union[SolverParameterInfo, None] = None
    last_update: Union[SolverUpdateInfo, None] = None
    last_endet: Union[SolverResultInfo, None] = None
    min_time = 0.5
    last_iter: float = 0.0
    wait_lock: Event
    stop_thread: Event

    def __init__(self, signal_target: QtCore.QObject):
        GuiCallbacks.__init__(self, signal_target)
        self.echo = False
        Thread.__init__(self, daemon=True, name="GenxDelayedCallbacks")
        self.wait_lock = Event()
        self.stop_thread = Event()

    def run(self):
        self.last_iter = time.time()
        self.stop_thread.clear()
        while not self.stop_thread.is_set():
            time.sleep(max(0.0, (self.last_iter - time.time() + self.min_time)))
            if self.last_text:
                GuiCallbacks.text_output(self, self.last_text)
                self.last_text = None
            if self.last_param:
                GuiCallbacks.parameter_output(self, self.last_param)
                self.last_param = None
            if self.last_update:
                GuiCallbacks.plot_output(self, self.last_update)
                self.last_update = None
            if self.last_endet:
                GuiCallbacks.fitting_ended(self, self.last_endet)
                self.last_endet = None
            self.last_iter = time.time()
            self.wait_lock.clear()
            self.wait_lock.wait()

    def exit(self):
        self.stop_thread.set()
        self.wait_lock.set()
        self.join(timeout=1.0)

    def text_output(self, text):
        debug(f"User Info: {text}", stacklevel=3)
        self.last_text = text
        self.wait_lock.set()

    def fitting_ended(self, result_data):
        self.last_endet = result_data
        self.wait_lock.set()

    def parameter_output(self, param_info):
        self.last_param = param_info
        self.wait_lock.set()

    def plot_output(self, update_data):
        self.last_update = update_data
        self.wait_lock.set()


@dataclass
class BatchOptions:
    keep_last: bool = True
    adjust_bounds: bool = False


class ModelControlGUI(QtCore.QObject):
    """
    Qt solver GUI controller using signals/slots for updates.
    """

    update_script = QtCore.Signal(str)
    update_plot = QtCore.Signal(object)
    sim_plot = QtCore.Signal(object)
    update_text = QtCore.Signal(str)
    update_parameters = QtCore.Signal(object)
    fitting_ended = QtCore.Signal(object)
    autosave = QtCore.Signal()
    batch_next = QtCore.Signal(object)
    value_change = QtCore.Signal()

    batch_running: bool = False
    batch_options: BatchOptions

    def __init__(self, parent: "main_window.GenxMainWindow"):
        super().__init__(parent)
        self.parent = parent
        self.solvers = {
            "Differential Evolution": diffev.DiffEv(),
            "Levenberg-Marquardt": levenberg_marquardt.LMOptimizer(),
        }
        try:
            from ..bumps_optimizer import BumpsOptimizer
        except ImportError:
            pass
        else:
            self.solvers["Bumps"] = BumpsOptimizer()
        self.solvers["Remote DiffEv"] = remote_optimizer.RemoteOptimizer()

        self.controller = model_control.ModelController(self.solvers["Differential Evolution"])
        self.callback_controller = DelayedCallbacks(self)
        self.callback_controller.start()
        self.controller.set_callbacks(self.callback_controller)
        self.controller.set_action_callback(self.OnActionCallback)

        self.fitting_ended.connect(self.OnFittingEnded)
        self.autosave.connect(self.AutoSave)
        self.batch_options = BatchOptions()

        self.ReadConfig()

    def OnActionCallback(self, action: ModelAction):
        self.SetUndoRedoLabels()
        if ModelInfluence.SCRIPT in action.influences:
            if hasattr(self.parent, "set_script_text"):
                self.parent.set_script_text(self.get_model_script())
            self.update_script.emit(self.get_model_script())
        if ModelInfluence.DATA in action.influences:
            cs = self.controller.get_color_cycle()
            colors2keys = dict((value, key) for key, value in COLOR_CYCLES.items())
            mb_checkables = getattr(self.parent, "mb_checkables", None)
            if mb_checkables is not None:
                if cs in colors2keys:
                    mb_checkables[colors2keys[cs]].Check()
                else:
                    mb_checkables[colors2keys[None]].Check()
            data_list_ctrl = getattr(getattr(self.parent, "ui", None), "dataListControl", None)
            if data_list_ctrl is not None:
                dl = data_list_ctrl.list_ctrl
                dl.data_cont.data = self.get_data()
                dl._UpdateImageList()
                dl._UpdateData("Plot settings changed", data_changed=True)
        if ModelInfluence.PARAM in action.influences:
            param_grid = getattr(self.parent, "paramter_grid", None)
            if param_grid is not None and hasattr(param_grid, "SetParameters"):
                param_grid.SetParameters(self.controller.get_parameters(), clear=False, permanent_change=True)
            self.value_change.emit()

    def OnUndo(self, _event=None):
        self.controller.undo_action()

    def OnRedo(self, _event=None):
        self.controller.redo_action()
        param_grid = getattr(self.parent, "paramter_grid", None)
        if param_grid is not None and hasattr(param_grid, "SetParameters"):
            param_grid.SetParameters(self.controller.get_model_params(), permanent_change=False)

    def SetUndoRedoLabels(self):
        undos, redos = self.controller.history_stacks()
        undo_menu = getattr(self.parent, "undo_menu", None)
        redo_menu = getattr(self.parent, "redo_menu", None)
        if undo_menu is not None:
            if len(undos) != 0:
                undo_menu.Enable(True)
                undo_menu.SetItemLabel(f"Undo ({undos[-1].action_name})\tCtrl+Z")
            else:
                undo_menu.Enable(False)
                undo_menu.SetItemLabel("Undo\tCtrl+Z")
        if redo_menu is not None:
            if len(redos) != 0:
                redo_menu.Enable(True)
                redo_menu.SetItemLabel(f"Redo ({redos[-1].action_name})\tCtrl+Shift+Z")
            else:
                redo_menu.Enable(False)
                redo_menu.SetItemLabel("Redo\tCtrl+Shift+Z")

    def new_model(self):
        self.controller.new_model()

    def get_model(self):
        return self.controller.get_model()

    def set_model_script(self, text):
        self.controller.set_model_script(text)

    def set_model_params(self, params):
        self.controller.set_model_params(params)

    def get_model_params(self):
        return self.controller.get_model_params()

    def get_model_script(self):
        return self.controller.get_model_script()

    def set_data(self, data):
        self.controller.set_data(data)

    def get_data(self):
        return self.controller.get_data()

    def update_plotsettings(self, event: UpdatePlotSettingsEvent):
        self.controller.set_data_plotsettings(event.indices, event.sim_par, event.data_par)

    def update_color_cycle(self, source):
        self.controller.update_color_cycle(source)

    def get_parameters(self):
        return self.controller.get_parameters()

    def get_sim_pars(self):
        return self.controller.get_sim_pars()

    def get_color_cycle(self):
        return self.controller.get_color_cycle()

    def get_parameter_data(self, row):
        return self.controller.get_parameter_data(row)

    def get_parameter_name(self, row):
        return self.controller.get_parameter_name(row)

    def get_possible_parameters(self):
        return self.controller.get_possible_parameters()

    def get_fom(self):
        return self.controller.get_fom()

    def get_fom_name(self):
        return self.controller.get_fom_name()

    def set_filename(self, filename):
        self.controller.set_filename(filename)

    def get_filename(self):
        return self.controller.get_filename()

    def get_model_name(self):
        return self.controller.get_model_name()

    def force_compile(self):
        self.controller.force_compile()

    def compile_if_needed(self):
        self.controller.compile_if_needed()

    def simulate(self, recompile=False):
        self.controller.simulate(recompile=recompile)
        self.sim_plot.emit(self.controller.get_model())

    def evaluate(self):
        self.controller.evaluate()

    def set_error_pars(self, error_values):
        self.controller.set_error_pars(error_values)

    def export_data(self, basename):
        self.controller.export_data(basename)

    def export_table(self, basename):
        self.controller.export_table(basename)

    def export_script(self, basename):
        self.controller.export_script(basename)

    def export_orso(self, basename, convert_to_q=True):
        self.controller.export_orso(basename, convert_to_q)

    def import_table(self, filename):
        self.controller.import_table(filename)

    def import_script(self, filename):
        self.controller.import_script(filename)

    def get_data_as_asciitable(self, indices=None):
        return self.controller.get_data_as_asciitable(indices=indices)

    def set_update_min_time(self, new_time):
        self.callback_controller.min_time = new_time

    @property
    def saved(self):
        return self.controller.saved

    @saved.setter
    def saved(self, value):
        self.controller.saved = value

    @property
    def eval_in_model(self):
        return self.controller.eval_in_model

    @property
    def script_module(self):
        return self.controller.script_module

    def get_solvers(self):
        return list(self.solvers.keys())

    def set_solver(self, name):
        self.controller.optimizer = self.solvers[name]
        self.controller.set_callbacks(self.callback_controller)

    def ReadConfig(self):
        self.controller.ReadConfig()

    def WriteConfig(self):
        self.controller.WriteConfig()

    def ParametersDialog(self, frame):
        self.ReadConfig()
        fom_func_name = self.controller.get_fom_name()
        if fom_func_name not in fom_funcs.func_names:
            ShowWarningDialog(
                self.parent,
                "The loaded fom function, "
                + fom_func_name
                + ", does not exist "
                + "in the local fom_funcs file. The fom fucntion has been"
                + " temporary added to the list of availabe fom functions",
            )
            fom_funcs.func_names.append(fom_func_name)
            exectext = "fom_funcs." + fom_func_name + " = self.parent.model.fom_func"
            exec(exectext, {}, {"fom_funcs": fom_funcs, "self": self})

        combined_options = self.controller.get_combined_options()
        dlg = SettingsDialog(frame, combined_options, apply_callback=lambda options: False, title="Optimizer Settings")

        res = dlg.exec()
        if res == QtWidgets.QDialog.DialogCode.Accepted:
            updates = dlg.collect_results()
            self.controller.update_combined_options(updates)

    def ModelLoaded(self):
        param_grid = getattr(self.parent, "paramter_grid", None)
        if param_grid is not None and hasattr(param_grid, "SetParameters"):
            param_grid.SetParameters(self.controller.get_model_params(), permanent_change=False)
        self.update_plot.emit(
            UpdatePlotEvent(
                model=self.controller.get_fitted_model(),
                fom_log=self.controller.get_fom_log(),
                update_fit=False,
                desc="Model loaded",
            )
        )
        self.controller.history_clear()

        if self.controller.is_configured():
            res = self.controller.get_result_info()
            try:
                evt = UpdateParametersEvent(
                    values=res.values,
                    new_best=False,
                    population=res.population,
                    max_val=res.par_max,
                    min_val=res.par_min,
                    fitting=True,
                    desc="Parameter Loaded",
                    update_errors=False,
                    permanent_change=False,
                )
            except AttributeError:
                iprint("Could not create data for parameters")
            else:
                self.update_parameters.emit(evt)
        QtCore.QTimer.singleShot(100, self.SetModelSaved)

    def SetModelSaved(self):
        self.saved = True

    @QtCore.Slot(object)
    def OnFittingEnded(self, evt: FittingEndedEvent):
        if evt.error_message:
            ShowErrorDialog(self.parent, evt.error_message)
            return

        if not self.batch_running:
            message = "Do you want to keep the parameter values from the fit?"
            result = ShowQuestionDialog(self.parent, message, "Keep the fit?", yes_no=True)
        else:
            result = True
        if result:
            self.controller.set_value_pars(evt.values)
            self.update_parameters.emit(
                UpdateParametersEvent(
                    values=evt.values,
                    new_best=True,
                    population=evt.population,
                    max_val=evt.max_val,
                    min_val=evt.min_val,
                    fitting=False,
                    desc="Parameter Improved",
                    update_errors=False,
                    permanent_change=True,
                )
            )
            if self.batch_running:
                QtCore.QTimer.singleShot(0, self.batch_next_step)
        else:
            self.update_parameters.emit(
                UpdateParametersEvent(
                    values=evt.start_guess,
                    new_best=True,
                    population=evt.population,
                    max_val=evt.max_val,
                    min_val=evt.min_val,
                    fitting=False,
                    desc="Parameter Reset",
                    update_errors=False,
                    permanent_change=False,
                )
            )

    def batch_next_step(self):
        idx = self.controller.active_index()
        params = self.get_parameters()
        if idx + 1 == len(self.controller.model_store):
            self.batch_running = False
            self.batch_next.emit(BatchNextEvent(last_index=idx, finished=True))
            return
        self.controller.activate_model(idx + 1)
        new_pars = self.get_parameters()
        if self.batch_options.keep_last:
            row_nmb, funcs, values, min_, max_ = params.get_fit_pars()
            for ri, vi in zip(row_nmb, values):
                new_pars.set_value(ri, 1, vi)
        if self.batch_options.adjust_bounds:
            row_nmb, funcs, values, min_, max_ = new_pars.get_fit_pars()
            for ri, vi, mii, mai in zip(row_nmb, values, min_, max_):
                val_range = mai - mii
                new_pars.set_value(ri, 3, vi - val_range / 2.0)
                new_pars.set_value(ri, 4, vi + val_range / 2.0)
        self.batch_next.emit(BatchNextEvent(last_index=idx, finished=False))
        QtCore.QTimer.singleShot(1000, self.controller.StartFit)

    def OnSetParameterValue(self, evt: SetParameterValueEvent):
        self.controller.set_parameter_value(evt.row, evt.col, evt.value)

    def OnMoveParameter(self, evt: MoveParameterEvent):
        self.controller.move_parameter(evt.row, evt.step)

    def OnInsertParameter(self, row: int):
        self.controller.insert_parameter(row)

    def OnDeleteParameter(self, rows: list[int]):
        self.controller.delete_parameter(rows)

    def OnSortAndGroupParameters(self, sort_params):
        self.controller.sort_and_group_parameters(sort_params)

    def OnUpdateParameters(self, evt: UpdateParametersEvent):
        if evt.desc not in ["Parameter Update", "Parameter Reset"]:
            return
        param_grid = getattr(self.parent, "paramter_grid", None)
        if param_grid is not None and hasattr(param_grid, "ShowParameters"):
            param_grid.ShowParameters(evt.values)

    def OnShowHistory(self, _evt=None):
        dia = HistoryDialog(self.parent, self.controller.history)
        dia.exec()
        if dia.changed_actions:
            self.OnActionCallback(dia.changed_actions)

    def CalcErrorBars(self):
        res = self.controller.CalcErrorBars()
        if (res[:, 0] > 0.0).any() or (res[:, 1] < 0.0).any():
            ShowInfoDialog(
                self.parent,
                "There is something wrong in the error estimation, low/high values don't have the right sign.\n\n"
                "This can be caused by non single-modal parameter statistics, closeness to bounds or too low value of"
                "'burn' before stampling.",
                title="Issue in uncertainty estimation",
            )
        error_strings = []
        for error_low, error_high in res:
            error_str = "(%.3e, %.3e)" % (error_low, error_high)
            error_strings.append(error_str)
        return error_strings

    def ProjectEvals(self, parameter):
        return self.controller.ProjectEvals(parameter)

    def ScanParameter(self, parameter, points):
        row = parameter
        model = self.controller.model
        (funcs, vals) = model.get_sim_pars()
        minval = model.parameters.get_data()[row][3]
        maxval = model.parameters.get_data()[row][4]
        parfunc = funcs[model.parameters.get_sim_pos_from_row(row)]
        par_def_val = vals[model.parameters.get_sim_pos_from_row(row)]
        step = (maxval - minval) / points
        par_vals = np.arange(minval, maxval + step, step)
        fom_vals = np.array([])

        par_name = model.parameters.get_data()[row][0]
        dlg = QtWidgets.QProgressDialog(
            f"Scanning parameter {par_name}",
            "Cancel",
            0,
            len(par_vals),
            self.parent,
        )
        dlg.setWindowTitle("Scan Parameter")
        dlg.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)
        with CatchModelError(self.parent, "ScanParameter", "scan through values") as cme:
            [f(v) for (f, v) in zip(funcs, vals)]
            for i, par_val in enumerate(par_vals, start=1):
                parfunc(par_val)
                fom_vals = np.append(fom_vals, model.evaluate_fit_func())
                dlg.setValue(i)
                QtWidgets.QApplication.processEvents()
                if dlg.wasCanceled():
                    break
        dlg.close()
        parfunc(par_def_val)
        if cme.successful:
            return par_vals, fom_vals

    def ResetOptimizer(self):
        pass

    def StartFit(self):
        self.controller.StartFit()

    def StopFit(self):
        if self.batch_running:
            self.batch_running = False
            return
        self.controller.StopFit()

    def ResumeFit(self):
        self.controller.ResumeFit()

    def IsFitted(self):
        return self.controller.IsFitted()

    @QtCore.Slot()
    def AutoSave(self):
        self.controller.save()

    def load_file(self, fname):
        prog = QtWidgets.QProgressDialog(f"Reading from file\n{fname}\n", "Cancel", 0, 100, self.parent)
        prog.setWindowTitle("Loading...")
        prog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        def update_callback(i, n_total):
            prog.setValue(int(i / n_total * 100))
            prog.setLabelText(f"Reading from file\n{fname}\ndataset {i} of {n_total}")
            QtWidgets.QApplication.processEvents()

        try:
            self.controller.load_file(fname, update_callback=update_callback)
        finally:
            prog.close()
        solver_classes = [si.__class__ for si in self.solvers.values()]
        loaded_solver = self.controller.optimizer.__class__
        if loaded_solver in solver_classes:
            current_solver = list(self.solvers.keys())[solver_classes.index(loaded_solver)]
        else:
            self.solvers[loaded_solver.__name__] = self.controller.optimizer
            current_solver = loaded_solver.__name__
            if hasattr(self.parent, "eh_ex_add_solver_selection"):
                self.parent.eh_ex_add_solver_selection(current_solver)
        if hasattr(self.parent, "eh_ex_set_solver_selection"):
            self.parent.eh_ex_set_solver_selection(current_solver)

    def set_error_bars_level(self, value):
        if value < 1:
            raise ValueError("fom_error_bars_level has to be above 1")
        self.controller.optimizer.opt.errorbar_level = value

    def set_save_all_evals(self, value):
        self.controller.optimizer.opt.save_all_evals = bool(value)
