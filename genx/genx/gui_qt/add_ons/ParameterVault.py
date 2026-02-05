"""
==============
ParameterVault
==============

A plugin that allows the user to quickly store parameter grid values and the corresponding
simulation to compare different fits against each other.

Qt port.
"""

import os

from copy import deepcopy

from PySide6 import QtCore, QtGui, QtWidgets

from genx.parameters import Parameters
from genx.plugins import add_on_framework as framework
from genx.gui_qt.utils import ShowInfoDialog, ShowQuestionDialog, ShowWarningDialog


class Plugin(framework.Template):
    _refplugin = None

    @property
    def refplugin(self):
        if not self._refplugin:
            self._init_refplugin()
        return self._refplugin

    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent

        parameter_panel = self.NewDataFolder("Vault")
        layout = QtWidgets.QHBoxLayout(parameter_panel)
        self.parameter_panel = QtWidgets.QWidget(parameter_panel)
        self.create_parameter_list()
        layout.addWidget(self.parameter_panel, 1)

        menu = self.NewMenu("P. Vault")
        self.action_recalc_all = menu.addAction("Recalculate items")
        self.action_recalc_all.setToolTip("Run simulations for each parameter set again and calculate FOM")

        self.action_auto_store = menu.addAction("Save tables")
        self.action_auto_store.setCheckable(True)
        self.action_auto_store.setChecked(False)
        self.action_auto_store.setToolTip("Save a parameters table file each time you add something to the vault")

        self.action_load_all = menu.addAction("Recall saved tables")
        self.action_load_all.setToolTip("Load all table parameters previously saved")

        self.action_recalc_all.triggered.connect(self.OnRecalcAll)
        self.action_auto_store.triggered.connect(self.OnAutoStore)
        self.action_load_all.triggered.connect(self.OnLoadAll)

    def _init_refplugin(self):
        ph = self.parent.plugin_control.plugin_handler
        if "Reflectivity" in ph.loaded_plugins:
            self._refplugin = ph.loaded_plugins["Reflectivity"]
        else:
            ShowWarningDialog(self.parameter_panel, "Reflectivity plugin must be loaded", "Information")
            self._refplugin = None

    def create_toolbar(self):
        toolbar = QtWidgets.QToolBar(self.parameter_panel)
        toolbar.setIconSize(QtCore.QSize(20, 20))

        action_add = toolbar.addAction(QtGui.QIcon(":/main_gui/add.png"), "Add")
        action_add.setToolTip("Add current parameters to Vault")
        action_delete = toolbar.addAction(QtGui.QIcon(":/main_gui/delete.png"), "Delete")
        action_delete.setToolTip("Delete selected parameter set")
        action_apply = toolbar.addAction(QtGui.QIcon(":/main_gui/start_fit.png"), "Apply")
        action_apply.setToolTip("Apply selected parameter set to the model")
        action_plot = toolbar.addAction(QtGui.QIcon(":/main_gui/plotting.png"), "Plot")
        action_plot.setToolTip("Toggle plotting of the selected parameter set")

        action_add.triggered.connect(self.parameter_list.AddItem)
        action_delete.triggered.connect(self.parameter_list.DeleteItem)
        action_apply.triggered.connect(self.parameter_list.ApplyParameters)
        action_plot.triggered.connect(self.parameter_list.TogglePlot)

        self.toolbar = toolbar

    def create_parameter_list(self):
        self.parameter_list = ParameterList(self.parameter_panel, self)
        layout = QtWidgets.QVBoxLayout(self.parameter_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.create_toolbar()
        layout.addWidget(self.toolbar, 0)
        layout.addWidget(self.parameter_list, 1)

        ppanel = QtWidgets.QWidget(self.parameter_panel)
        psizer = QtWidgets.QGridLayout(ppanel)
        psizer.setContentsMargins(0, 0, 0, 0)

        psizer.addWidget(QtWidgets.QLabel("P1 index:"), 0, 0)
        self.P1_index = QtWidgets.QSpinBox(ppanel)
        self.P1_index.setRange(0, 100)
        psizer.addWidget(self.P1_index, 0, 1)

        psizer.addWidget(QtWidgets.QLabel("P2 index:"), 1, 0)
        self.P2_index = QtWidgets.QSpinBox(ppanel)
        self.P2_index.setRange(0, 100)
        psizer.addWidget(self.P2_index, 1, 1)

        layout.addWidget(ppanel, 0)

        self.P1_index.valueChanged.connect(self.parameter_list.UpdateParams)
        self.P2_index.valueChanged.connect(self.parameter_list.UpdateParams)

    def OnSimulate(self, _event):
        plot = getattr(getattr(self.parent, "ui", None), "plotDataPanel", None)
        if plot is None or self.refplugin is None:
            return
        model = self.GetModel()
        styles = ["--", "-.", ":"]
        k = 0
        for _fom, do_plot, params, data, _slds in self.parameter_list.parameter_list:
            if do_plot and params != model.parameters.data:
                for i, di in enumerate(data):
                    if model.data[i].show:
                        plot.ax.plot(
                            di.x,
                            di.y_sim,
                            c=di.sim_color,
                            lw=di.sim_linethickness,
                            ls=styles[k % len(styles)],
                            marker=di.sim_symbol,
                            ms=di.sim_symbolsize,
                        )
                k += 1
        plot.AutoScale()
        plot.flush_plot()

    def OnRecalcAll(self, _event):
        pl = self.parameter_list
        model = self.GetModel()
        start_params = model.parameters.data
        for i in range(len(pl.parameter_list)):
            pl.SelectRow(i, False)
        for i in range(len(pl.parameter_list)):
            pl.SelectRow(i, True)
            if i > 0:
                pl.SelectRow(i - 1, False)
            pl.ApplyParameters()
            model.simulate()
            slds = deepcopy(model.script_module.SLD)
            params = deepcopy(model.parameters.data)
            data = [di.copy() for di in model.data]
            pl.parameter_list[i] = [model.fom or 0.0, True, params, data, slds]
        pl.RefreshTable()
        model.parameters.data = start_params
        self.OnAutoStore(None)

    def OnAutoStore(self, _event):
        p = Parameters()
        base_name = self.GetModel().filename.rsplit(".", 1)[0]
        if self.action_auto_store.isChecked():
            for i, (_fom, _show, params, _data, _slds) in enumerate(self.parameter_list.parameter_list):
                p.data = params
                txt = p.get_ascii_output()
                with open(base_name + "_%i.tab" % i, "w", encoding="utf-8") as handle:
                    handle.write(txt)

    def OnLoadAll(self, _event):
        pl = self.parameter_list
        base_name = self.GetModel().filename.rsplit(".", 1)[0]
        if not os.path.exists(base_name + "_%i.tab" % 0):
            return
        pl.parameter_list = []
        i = 0
        p = Parameters()
        model = self.GetModel()
        start_params = model.parameters.data
        while os.path.exists(base_name + "_%i.tab" % i):
            with open(base_name + "_%i.tab" % i, "r", encoding="utf-8") as handle:
                txt = handle.read()
            p.set_ascii_input(txt)
            model.parameters.data = deepcopy(p.data)
            model.simulate()
            pl.AddItem(simulate=False)
            i += 1
        model.parameters.data = start_params
        self.parent.simulate()


class ParameterList(QtWidgets.QTableWidget):
    def __init__(self, parent, plugin):
        super().__init__(parent)
        self.plugin = plugin
        self.parameter_list = []
        self._updating = False

        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["#", "FOM", "Show", "P1", "P2"])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)

        self.itemChanged.connect(self._on_item_changed)

    def RefreshTable(self):
        self._updating = True
        try:
            self.setRowCount(len(self.parameter_list))
            for row, data in enumerate(self.parameter_list):
                self._set_row(row, data)
        finally:
            self._updating = False

    def _set_row(self, row, data):
        fom, show, params, _data, _slds = data

        def set_item(col, text, checkable=False, checked=False):
            item = self.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                self.setItem(row, col, item)
            if checkable:
                item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
            item.setText(text)
            return item

        set_item(0, str(row + 1))
        set_item(1, "%.4g" % fom)
        set_item(2, "", checkable=True, checked=show)
        set_item(3, self._get_param_value(params, self.plugin.P1_index.value()))
        set_item(4, self._get_param_value(params, self.plugin.P2_index.value()))

    def _get_param_value(self, params, idx):
        try:
            model = self.plugin.GetModel()
            fit_indices = [i for i, p in enumerate(model.parameters.data) if p[2]]
            real_idx = fit_indices[idx]
            return "%g" % params[real_idx][1]
        except Exception:
            return ""

    def _get_selected_row(self):
        rows = [idx.row() for idx in self.selectionModel().selectedRows()]
        return rows[0] if rows else -1

    def SelectRow(self, row, state=True):
        if 0 <= row < self.rowCount():
            if state:
                self.selectRow(row)
            else:
                self.selectionModel().select(
                    self.model().index(row, 0),
                    QtCore.QItemSelectionModel.SelectionFlag.Deselect | QtCore.QItemSelectionModel.SelectionFlag.Rows,
                )

    def _on_item_changed(self, item):
        if self._updating or item is None:
            return
        if item.column() == 2:
            row = item.row()
            show = item.checkState() == QtCore.Qt.CheckState.Checked
            self.parameter_list[row][1] = show
            self.plugin.parent.simulate()

    def AddItem(self, _=None, simulate=True):
        model = self.plugin.GetModel()
        slds = deepcopy(model.script_module.SLD)
        params = deepcopy(model.parameters.data)
        data = [di.copy() for di in model.data]
        self.parameter_list.append([model.fom or 0.0, True, params, data, slds])
        self.RefreshTable()
        self.plugin.OnAutoStore(None)
        if simulate:
            self.plugin.parent.simulate()

    def DeleteItem(self, _=None):
        index = self._get_selected_row()
        if index < 0:
            return
        self.parameter_list.pop(index)
        self.RefreshTable()
        self.plugin.OnAutoStore(None)

    def ApplyParameters(self, _=None):
        index = self._get_selected_row()
        if index < 0:
            return
        model = self.plugin.GetModel()
        was_stored = any(model.parameters.data == params for _fom, _show, params, _data, _slds in self.parameter_list)
        if not was_stored:
            message = "Do you want to store the current parameters in the Vault before applying the selected ones?"
            result = ShowQuestionDialog(self, message, "Store current parameters?")
            if result:
                self.AddItem()
        model.parameters.data = deepcopy(self.parameter_list[index][2])
        param_grid = getattr(self.plugin.parent, "paramter_grid", None)
        if param_grid is not None:
            param_grid.SetParameters(model.parameters, permanent_change=False)
        self.plugin.parent.simulate()

    def TogglePlot(self, _=None):
        index = self._get_selected_row()
        if index < 0:
            return
        self.parameter_list[index][1] = not self.parameter_list[index][1]
        self.RefreshTable()
        self.plugin.parent.simulate()

    def UpdateParams(self, _=None):
        if len(self.parameter_list) == 0:
            return
        self.RefreshTable()
