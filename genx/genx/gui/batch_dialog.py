"""
A dialog to import batches of datafiles to create a model sequence and perform
fitting to a common model.
"""
from copy import copy, deepcopy

import wx.grid

from .solvergui import ModelControlGUI
from .custom_events import EVT_BATCH_NEXT


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
        self.SetTitle('Model/dataset list for batch fitting')

        self.dpi_scale_factor = wx.GetApp().dpi_scale_factor
        self.model_control = model_control
        self.model_control.Bind(EVT_BATCH_NEXT, self.OnBatchNext)

        self.grid = wx.grid.Grid(self)
        vbox.Add(self.grid, proportion=1, flag=wx.EXPAND)
        self.build_grid()

        self.grid.Bind(wx.grid.EVT_GRID_CELL_CHANGING, self.OnUpdateCell)
        self.grid.Bind(wx.grid.EVT_GRID_LABEL_LEFT_DCLICK, self.OnLabelDClick)

        self.keep_last = wx.CheckBox(self, label='Keep result values from last')
        self.keep_last.SetValue(self.model_control.batch_options.keep_last)
        vbox.Add(self.keep_last, proportion=0, flag=wx.FIXED_MINSIZE)

        self.adjust_bounds = wx.CheckBox(self, label='Adjust boundaries around last values')
        self.adjust_bounds.SetValue(self.model_control.batch_options.adjust_bounds)
        vbox.Add(self.adjust_bounds, proportion=0, flag=wx.FIXED_MINSIZE)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(hbox, proportion=0, flag=wx.EXPAND)

        btn = wx.Button(self, -1, label='Fit All')
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnBatchFitModel, btn)

        btn = wx.Button(self, -1, label='Fit From Here')
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnBatchFromHere, btn)

        hbox.AddSpacer(20*self.dpi_scale_factor)

        btn = wx.Button(self, -1, label='Sort by Value')
        hbox.Add(btn, proportion=0, flag=wx.FIXED_MINSIZE)
        self.Bind(wx.EVT_BUTTON, self.OnSort, btn)


    def OnBatchNext(self, evt):
        # evt.finished
        self.switch_line(evt.last_index, self.model_control.controller.active_index())
        g = self.grid
        prev_model = self.model_control.controller.model_store[evt.last_index]
        for i, pi in enumerate(prev_model.parameters.get_fit_pars()[2]):
            g.SetCellValue(evt.last_index, i+2, f'{pi:.7g}')

    def switch_line(self, prev_row, row):
        g = self.grid
        if prev_row>=0:
            g.SetCellBackgroundColour(prev_row, 0, "#ffffff")
            g.SetCellBackgroundColour(prev_row, 1, "#ffffff")
        if row>=0:
            g.SetCellBackgroundColour(row, 0, "#ff8888")
            g.SetCellBackgroundColour(row, 1, "#ff8888")
        g.MakeCellVisible(row, 0)
        g.ForceRefresh()

    def build_grid(self):
        g = self.grid
        models = self.model_control.controller.model_store
        g.CreateGrid(len(models), 2)
        g.SetSelectionMode(wx.grid.Grid.GridSelectNone)

        g.SetColLabelValue(0, 'item')
        g.SetColLabelValue(1, 'index/value')
        self.fill_grid()

    def fill_grid(self):
        g = self.grid
        models = self.model_control.controller.model_store
        ci = self.model_control.controller.active_index()
        for i, mi in enumerate(models):
            g.SetRowLabelValue(i, f'{i}:')
            g.SetCellValue(i, 0, mi.h5group_name)
            g.SetCellValue(i, 1, f'{mi.sequence_value:.3g}')
            g.DisableRowResize(i)
            if i==ci:
                g.SetCellBackgroundColour(i, 0, "#ff8888")
                g.SetCellBackgroundColour(i, 1, "#ff8888")
            else:
                g.SetCellBackgroundColour(i, 0, "#ffffff")
                g.SetCellBackgroundColour(i, 1, "#ffffff")

    def OnUpdateCell(self, event: wx.grid.GridEvent):
        if event.GetCol()==0:
            names = [m.h5group_name for m in self.model_control.controller.model_store]
            new_name = event.GetString()
            if new_name in names:
                event.Veto()
                return
            # change the name of a dataaset
            self.model_control.controller.model_store[event.GetRow()].h5group_name = new_name
        else:
            event.Skip()

    def OnLabelDClick(self, event: wx.grid.GridEvent):
        if event.GetCol()<0:
            g = self.grid
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
        if g.GetNumberCols()>2:
            g.DeleteCols(2, g.GetNumberCols()-2)
        row_nmb, funcs, values, min_, max_ = self.model_control.get_parameters().get_fit_pars()
        ncols = len(row_nmb)
        g.AppendCols(ncols)
        for i, ci in enumerate(funcs):
            g.SetColLabelValue(i+2, ci)

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
        for mi in self.model_control.controller.model_store[ci+1:]:
            mi.set_script(copy(script))
            mi.parameters = deepcopy(params)
        # set first model and start fitting
        self.set_batch_params()
        # put values for parameters from previous datasets
        g = self.grid
        fit_rows, *_ = self.model_control.get_parameters().get_fit_pars()
        for i, mi in enumerate(self.model_control.controller.model_store[:ci+1]):
            params = [mi.parameters[fi].value for fi in fit_rows]
            for j, pj in enumerate(params):
                g.SetCellValue(i, j+2, f'{pj:.7g}')

        wx.CallLater(1000, self.model_control.StartFit)

    def OnSort(self, evt):
        sort_list = [(mi.sequence_value, mi) for mi in self.model_control.controller.model_store]
        sort_list.sort()
        self.model_control.controller.model_store = [si[1] for si in sort_list]
        self.fill_grid()
