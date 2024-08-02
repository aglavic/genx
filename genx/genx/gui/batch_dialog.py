"""
A dialog to import batches of datafiles to create a model sequence and perform
fitting to a common model.
"""

from copy import copy, deepcopy

import numpy as np
import wx.grid

from ..plugins.data_loader_framework import Template as DataLoaderTemplate
from .custom_events import EVT_BATCH_NEXT
from .metadata_dialog import MetaDataDialog
from .plot_dialog import PlotDialog
from .solvergui import ModelControlGUI


def get_value_from_path(data, path):
    if len(path) > 1:
        return get_value_from_path(data[path[0]], path[1:])
    else:
        return data[path[0]]


class CopyGrid(wx.grid.Grid):
    """Grid calss that copies all data on ctrl+C, no matter the selection."""

    def __init__(self, *args, **opts):
        wx.grid.Grid.__init__(self, *args, **opts)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKey)

    def OnKey(self, event):
        # If Ctrl+C is pressed...
        if event.ControlDown() and event.GetKeyCode() == 67:
            self.copy()
        else:
            event.Skip()
            return

    def copy(self):
        output = ""
        for col in range(self.GetNumberCols()):
            name = self.GetColLabelValue(col).replace("\n", ".")
            output += f"{name}\t"
        output += "\n"
        for row in range(self.GetNumberRows()):
            for col in range(self.GetNumberCols()):
                value = self.GetCellValue(row, col)
                output += f"{value}\t"
            output += "\n"

        clipboard = wx.TextDataObject()
        clipboard.SetText(output)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(clipboard)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Can't open the clipboard", "Error")


class BatchDialog(wx.Dialog):
    """
    A dialog to display all models that are stored in the file, to generate sequences from
    data files and to fit batches to a common model.
    """

    model_control: ModelControlGUI

    def __init__(self, parent, model_control: ModelControlGUI):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        self.SetTitle("Model/dataset list for batch fitting")

        self.dpi_scale_factor = wx.GetApp().dpi_scale_factor
        self.model_control = model_control
        self.model_control.Bind(EVT_BATCH_NEXT, self.OnBatchNext)
        self.plots = {}

        vbox.Add(
            wx.StaticText(
                self,
                label="Below is the list of datasets for batch fitting.\n"
                "If you want to activate one dataset, double-click the row label on the left.",
            ),
            proportion=0,
            flag=wx.FIXED_MINSIZE | wx.ALL,
            border=4,
        )

        self.grid = CopyGrid(self)
        self.grid.SetDefaultRowSize(int(self.grid.GetDefaultRowSize() * 1.5))
        self.grid.SetDefaultCellAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTRE)
        vbox.Add(self.grid, proportion=1, flag=wx.EXPAND)
        self.build_grid()

        self.grid.Bind(wx.grid.EVT_GRID_CELL_CHANGING, self.OnUpdateCell)
        self.grid.Bind(wx.grid.EVT_GRID_LABEL_LEFT_DCLICK, self.OnLabelDClick)

        self.keep_last = wx.CheckBox(self, label="Keep result values from last")
        self.keep_last.SetValue(self.model_control.batch_options.keep_last)
        vbox.Add(self.keep_last, proportion=0, flag=wx.FIXED_MINSIZE | wx.TOP | wx.LEFT, border=4)

        self.adjust_bounds = wx.CheckBox(self, label="Adjust boundaries around last values")
        self.adjust_bounds.SetValue(self.model_control.batch_options.adjust_bounds)
        vbox.Add(self.adjust_bounds, proportion=0, flag=wx.FIXED_MINSIZE | wx.LEFT, border=4)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(hbox, proportion=0, flag=wx.EXPAND)

        btn = wx.Button(self, -1, label="Import Data...")
        btn.SetToolTip("Use active dataset as template and load\na list of datasets to generate a batch.")
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnImportData, btn)

        btn = wx.Button(self, -1, label="Clear list")
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnClearData, btn)

        hbox.AddSpacer(int(20 * self.dpi_scale_factor))

        btn = wx.Button(self, -1, label="Fit All")
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnBatchFitModel, btn)

        btn = wx.Button(self, -1, label="Fit From Here")
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnBatchFromHere, btn)

        hbox.AddSpacer(int(20 * self.dpi_scale_factor))

        btn = wx.Button(self, -1, label="Sort by Value")
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnSort, btn)

        btn = wx.Button(self, -1, label="Extract Value")
        btn.SetToolTip("Use metadata from datasets to extract a sequence value (e.g. temperature)")
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnExtract, btn)

    def OnBatchNext(self, evt):
        # evt.finished
        self.switch_line(evt.last_index, self.model_control.controller.active_index())
        g = self.grid
        prev_model = self.model_control.controller.model_store[evt.last_index]
        prev_models = self.model_control.controller.model_store[: evt.last_index + 1]

        row_nmb, funcs, values, min_, max_ = prev_model.parameters.get_fit_pars()
        x = np.array([mi.sequence_value for mi in prev_models])

        for i, (ni, fi, pi) in enumerate(zip(row_nmb, funcs, values)):
            g.SetCellValue(evt.last_index, i + 2, f"{pi:.7g}")
            if fi in self.plots:
                vals = np.array([mi.parameters[ni].value for mi in prev_models])
                plot = self.plots[fi]
                plot.clear_data()
                plot.plot(x, vals, marker="o", ms=2, lw=1, color=(0.0, 0.0, 1.0))
                plot.draw()

    def switch_line(self, prev_row, row):
        g = self.grid
        if prev_row >= 0:
            g.SetCellBackgroundColour(prev_row, 0, "#ffffff")
            g.SetCellBackgroundColour(prev_row, 1, "#ffffff")
        if row >= 0:
            g.SetCellBackgroundColour(row, 0, "#ff8888")
            g.SetCellBackgroundColour(row, 1, "#ff8888")
        g.MakeCellVisible(row, 0)
        g.ForceRefresh()

    def build_grid(self):
        g = self.grid
        models = self.model_control.controller.model_store
        g.CreateGrid(len(models), 2)
        g.SetSelectionMode(wx.grid.Grid.GridSelectNone)

        g.SetColLabelValue(0, "item")
        g.SetColLabelValue(1, "index/value")
        self.fill_grid()

    def fill_grid(self):
        g = self.grid
        models = self.model_control.controller.model_store
        ci = self.model_control.controller.active_index()
        for i, mi in enumerate(models):
            g.SetRowLabelValue(i, f"{i}:")
            g.SetCellValue(i, 0, mi.h5group_name)
            g.SetCellValue(i, 1, f"{mi.sequence_value:.7g}")
            g.DisableRowResize(i)
            if i == ci:
                g.SetCellBackgroundColour(i, 0, "#ff8888")
                g.SetCellBackgroundColour(i, 1, "#ff8888")
            else:
                g.SetCellBackgroundColour(i, 0, "#ffffff")
                g.SetCellBackgroundColour(i, 1, "#ffffff")

        g.AutoSizeColumns(False)

    def OnUpdateCell(self, event: wx.grid.GridEvent):
        if event.GetCol() == 0:
            names = [m.h5group_name for m in self.model_control.controller.model_store]
            new_name = event.GetString()
            if new_name in names:
                event.Veto()
                return
            # change the name of a dataaset
            self.model_control.controller.model_store[event.GetRow()].h5group_name = new_name
        elif event.GetCol() == 1:
            try:
                value = float(event.GetString())
            except ValueError:
                event.Veto()
                return
            else:
                self.model_control.controller.model_store[event.GetRow()].sequence_value = value
        else:
            event.Veto()

    def OnLabelDClick(self, event: wx.grid.GridEvent):
        if event.GetCol() < 0:
            ci = self.model_control.controller.active_index()
            row = event.GetRow()
            self.model_control.controller.activate_model(row)
            self.switch_line(ci, row)

    def set_batch_params(self):
        self.model_control.batch_running = True
        self.model_control.batch_options.keep_last = self.keep_last.IsChecked()
        self.model_control.batch_options.adjust_bounds = self.adjust_bounds.IsChecked()
        self.set_batch_columns()

    def set_batch_columns(self):
        g = self.grid
        if g.GetNumberCols() > 2:
            g.DeleteCols(2, g.GetNumberCols() - 2)
        row_nmb, funcs, values, min_, max_ = self.model_control.get_parameters().get_fit_pars()
        ncols = len(row_nmb)
        g.AppendCols(ncols)
        for i, ci in enumerate(funcs):
            g.SetColLabelValue(i + 2, ci)
            self.create_graph(ci)

    def create_graph(self, parameter):
        if parameter in self.plots:
            self.plots[parameter].Show()
            return
        plot = PlotDialog(self, title=f"Batch parameter {parameter} fit")
        self.plots[parameter] = plot
        plot.Show()
        plot.set_xlabel("batch index/value")
        plot.set_ylabel(parameter)
        plot.draw()

    def OnImportData(self, evt):
        origin = self.model_control.get_model().deepcopy()
        # TODO: There needs to be better separation, data_loader should be accessible through ModelControlGUI
        data_loader: DataLoaderTemplate = self.model_control.parent.data_list.list_ctrl.data_loader
        flists = []
        for i in range(len(origin.data)):
            dia = wx.FileDialog(
                self,
                message=f"Select datafiles for dataset {i}",
                wildcard=data_loader.GetWildcardString(),
                style=wx.FD_OPEN | wx.FD_CHANGE_DIR | wx.FD_MULTIPLE,
            )
            res = dia.ShowModal()
            if res != wx.ID_OK:
                return
            flists.append(dia.GetPaths())

        N = len(flists[0])
        prog = wx.ProgressDialog("Importing...", f"Read from files...\n0/{N}", maximum=100, parent=self)

        def update_callback(i):
            prog.Update(int(i / N * 100), f"Read from files...\n{i}/{N}")

        try:
            self.model_control.controller.read_sequence(
                data_loader, flists, name_by_file=True, callback=update_callback
            )
        finally:
            prog.Destroy()

        self.grid.AppendRows(len(self.model_control.controller.model_store) - self.grid.GetNumberRows())
        self.fill_grid()

    def OnClearData(self, evt):
        self.model_control.controller.model_store = []
        self.grid.DeleteRows(0, self.grid.GetNumberRows())

    def OnBatchFitModel(self, evt):
        evt.Skip()
        # synchronizing current model script and parameters to all datasets
        script = self.model_control.get_model_script()
        params = self.model_control.get_model_params()
        for mi in self.model_control.controller.model_store:
            mi.set_script(copy(script))
            mi.parameters = deepcopy(params)
        # set first model and start fitting
        self.switch_line(self.model_control.controller.active_index(), 0)
        self.model_control.controller.activate_model(0)
        self.set_batch_params()
        wx.CallLater(1000, self.model_control.StartFit)

    def OnBatchFromHere(self, evt):
        evt.Skip()
        # synchronizing current model script and parameters to all datasets
        script = self.model_control.get_model_script()
        params = self.model_control.get_model_params()
        ci = self.model_control.controller.active_index()
        for mi in self.model_control.controller.model_store[ci + 1 :]:
            mi.set_script(copy(script))
            mi.parameters = deepcopy(params)
        # set first model and start fitting
        self.set_batch_params()
        # put values for parameters from previous datasets
        g = self.grid
        fit_rows, *_ = self.model_control.get_parameters().get_fit_pars()
        for i, mi in enumerate(self.model_control.controller.model_store[: ci + 1]):
            params = [mi.parameters[fi].value for fi in fit_rows]
            for j, pj in enumerate(params):
                g.SetCellValue(i, j + 2, f"{pj:.7g}")

        wx.CallLater(1000, self.model_control.StartFit)

    def OnSort(self, evt):
        sort_list = [(mi.sequence_value, i, mi) for i, mi in enumerate(self.model_control.controller.model_store)]
        sort_list.sort()
        self.model_control.controller.model_store = [si[2] for si in sort_list]
        self.fill_grid()

    def OnExtract(self, evt):
        dia = MetaDataDialog(
            self,
            self.model_control.controller.model_store[0].data,
            filter_leaf_types=[int, float],
            close_on_activate=True,
        )
        dia.SetTitle("Select key to use by double-click")
        dia.ShowModal()
        dia.Destroy()
        if dia.activated_leaf is None:
            return

        didx = dia.activated_leaf[0]
        dpath = dia.activated_leaf[1:]
        for mi in self.model_control.controller.model_store:
            try:
                value = float(get_value_from_path(mi.data[didx].meta, dpath))
            except Exception:
                value = 0.0
            mi.sequence_value = value
        self.fill_grid()
