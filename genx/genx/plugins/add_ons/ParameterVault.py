# -*- coding: utf8 -*-
"""
==============
ParameterVault
==============

A plugin that allows the user to quickly store parameter grid values and the corresponding
simulation to compare different fits agains each other. 

Written by Artur Glavic
Last Changes 04/22/15
"""

import os
import sys

from copy import deepcopy

import wx

from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin

import genx.gui.images as img

from genx.parameters import Parameters

from .. import add_on_framework as framework
from ..utils import ShowInfoDialog, ShowQuestionDialog, ShowWarningDialog


class Plugin(framework.Template):
    _refplugin = None

    @property
    def refplugin(self):
        # check if reflectivity plugin is None or destoryed, try to connect
        if not self._refplugin:
            self._init_refplugin()
        return self._refplugin

    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent

        parameter_panel = self.NewDataFolder("Vault")
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.parameter_panel = wx.Panel(parameter_panel)
        self.create_parameter_list()
        sizer.Add(self.parameter_panel, 1, wx.EXPAND | wx.GROW | wx.ALL)
        parameter_panel.SetSizer(sizer)
        parameter_panel.Layout()

        menu = self.NewMenu("P. Vault")
        self.mb_recalc_all = wx.MenuItem(
            menu,
            wx.NewId(),
            "Recalculate items",
            "Run simulations for each parameter set again and calculate FOM",
            wx.ITEM_NORMAL,
        )
        menu.Append(self.mb_recalc_all)
        self.mb_auto_store = wx.MenuItem(
            menu,
            wx.NewId(),
            "Save tables",
            "Save a parameters table file each time you add something to the vault",
            wx.ITEM_CHECK,
        )
        menu.Append(self.mb_auto_store)
        self.mb_load_all = wx.MenuItem(
            menu, wx.NewId(), "Recall saved tables", "Load all table parameters previously saved", wx.ITEM_NORMAL
        )
        menu.Append(self.mb_load_all)
        self.mb_auto_store.Check(False)

        self.parent.Bind(wx.EVT_MENU, self.OnRecalcAll, self.mb_recalc_all)
        self.parent.Bind(wx.EVT_MENU, self.OnAutoStore, self.mb_auto_store)
        self.parent.Bind(wx.EVT_MENU, self.OnLoadAll, self.mb_load_all)

    def _init_refplugin(self):
        ph = self.parent.plugin_control.plugin_handler
        if "Reflectivity" in ph.loaded_plugins:
            # connect to the reflectivity plugin for layer creation
            self._refplugin = ph.loaded_plugins["Reflectivity"]
        else:
            ShowWarningDialog(self.materials_panel, "Reflectivity plugin must be loaded", "Information")
            self._refplugin = None

    def create_toolbar(self):
        dpi_scale_factor = wx.GetApp().dpi_scale_factor
        tb_bmp_size = int(dpi_scale_factor * 20)
        size = wx.Size(tb_bmp_size, tb_bmp_size)

        self.bitmap_button_add = wx.BitmapButton(
            self.tool_panel,
            -1,
            wx.Bitmap(img.getaddImage().Scale(tb_bmp_size, tb_bmp_size)),
            size=size,
            style=wx.NO_BORDER,
        )
        self.bitmap_button_add.SetToolTip("Add current parameters to Vault")
        self.bitmap_button_delete = wx.BitmapButton(
            self.tool_panel,
            -1,
            wx.Bitmap(img.getdeleteImage().Scale(tb_bmp_size, tb_bmp_size)),
            size=size,
            style=wx.NO_BORDER,
        )
        self.bitmap_button_delete.SetToolTip("Delete selected parameter set")
        self.bitmap_button_apply = wx.BitmapButton(
            self.tool_panel,
            -1,
            wx.Bitmap(img.getstart_fitImage().Scale(tb_bmp_size, tb_bmp_size)),
            size=size,
            style=wx.NO_BORDER,
        )
        self.bitmap_button_apply.SetToolTip("Apply selected parameter set to the model")
        self.bitmap_button_plot = wx.BitmapButton(
            self.tool_panel,
            -1,
            wx.Bitmap(img.getplottingImage().Scale(tb_bmp_size, tb_bmp_size)),
            size=size,
            style=wx.NO_BORDER,
        )
        self.bitmap_button_plot.SetToolTip("Toggle plotting of the selected parameter set")

        space = (2, -1)
        self.sizer_hor = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_hor.Add(self.bitmap_button_add, proportion=0, border=2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_delete, proportion=0, border=2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_apply, proportion=0, border=2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_plot, proportion=0, border=2)

        self.parameter_panel.Bind(wx.EVT_BUTTON, self.parameter_list.AddItem, self.bitmap_button_add)
        self.parameter_panel.Bind(wx.EVT_BUTTON, self.parameter_list.DeleteItem, self.bitmap_button_delete)
        self.parameter_panel.Bind(wx.EVT_BUTTON, self.parameter_list.ApplyParameters, self.bitmap_button_apply)
        self.parameter_panel.Bind(wx.EVT_BUTTON, self.parameter_list.TogglePlot, self.bitmap_button_plot)

    def create_parameter_list(self):
        """
        Create a list of materials and it's graphical representation as
        well as a toolbar.
        """
        self.tool_panel = wx.Panel(self.parameter_panel)
        self.parameter_list = ParameterList(self.parameter_panel, self)
        self.sizer_vert = wx.BoxSizer(wx.VERTICAL)
        self.parameter_panel.SetSizer(self.sizer_vert)

        self.create_toolbar()

        self.sizer_vert.Add(self.tool_panel, proportion=0, flag=wx.EXPAND, border=5)
        self.sizer_vert.Add((-1, 2))
        self.sizer_vert.Add(self.parameter_list, proportion=1, flag=wx.EXPAND, border=5)
        self.tool_panel.SetSizer(self.sizer_hor)

        ppanel = wx.Panel(self.parameter_panel)
        psizer = wx.GridSizer(2, 2, 0)
        ppanel.SetSizer(psizer)

        psizer.Add(wx.StaticText(ppanel, label="P1 index:"), proportion=0, flag=wx.EXPAND, border=2)
        self.P1_index = wx.SpinCtrl(ppanel)
        self.P1_index.SetRange(0, 100)
        psizer.Add(self.P1_index, proportion=1, flag=wx.EXPAND, border=2)

        psizer.Add(wx.StaticText(ppanel, label="P2 index:"), proportion=0, flag=wx.EXPAND, border=2)
        self.P2_index = wx.SpinCtrl(ppanel)
        self.P2_index.SetRange(0, 100)
        psizer.Add(self.P2_index, proportion=1, flag=wx.EXPAND, border=2)

        self.sizer_vert.Add(ppanel, proportion=0, flag=wx.EXPAND, border=5)

        self.parameter_panel.Bind(wx.EVT_SPINCTRL, self.parameter_list.UpdateParams, self.P1_index)
        self.parameter_panel.Bind(wx.EVT_SPINCTRL, self.parameter_list.UpdateParams, self.P2_index)

    def OnSimulate(self, event):
        """OnSimulate(self, event) --> None

        Updates stuff after simulation
        """
        colors = ["b", "r", "g", "c", "m", "y", "k"]
        styles = ["--", "-.", ":"]
        plot = self.parent.plot_data
        sld_plot = self.refplugin.sld_plot
        model = self.GetModel()
        k = 0
        for ignore, do_plot, params, data, slds in self.parameter_list.parameter_list:
            if do_plot and params != model.parameters.data:
                j = 0
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
                        # if len(slds)>i:
                        #     for key in slds[i]:
                        #         is_imag=key[:2]=='Im' or key[:4]=='imag'
                        #         if (is_imag and self.refplugin.show_imag_sld) or not is_imag:
                        #             if key!='z' and key!='SLD unit':
                        #                 sld_plot.ax.plot(slds[i]['z'], slds[i][key],
                        #                                       colors[j%len(colors)], ls=styles[k%len(styles)],
                        #                                       label=None)
                        #                 j+=1
                k += 1
        plot.AutoScale()
        plot.flush_plot()
        # sld_plot.flush_plot()

    def OnRecalcAll(self, event):
        pl = self.parameter_list
        model = self.GetModel()
        start_params = model.parameters.data
        for i in range(len(pl.parameter_list)):
            pl.Select(i, False)
        for i in range(len(pl.parameter_list)):
            pl.Select(i, True)
            if i > 0:
                pl.Select(i - 1, False)
            pl.ApplyParameters(None)
            model.simulate()
            slds = deepcopy(model.script_module.SLD)
            params = deepcopy(model.parameters.data)
            data = [di.copy() for di in model.data]
            pl.parameter_list[i] = [model.fom or 0.0, True, params, data, slds]
        pl.SetItemCount(len(pl.parameter_list))
        model.parameters.data = start_params
        self.OnAutoStore(None)

    def OnAutoStore(self, event):
        p = Parameters()
        base_name = self.GetModel().filename.rsplit(".", 1)[0]
        if self.mb_auto_store.IsChecked():
            for i, (ignore, ignore, params, ignore, ignore) in enumerate(self.parameter_list.parameter_list):
                p.data = params
                txt = p.get_ascii_output()
                open(base_name + "_%i.tab" % i, "w").write(txt)

    def OnLoadAll(self, event):
        pl = self.parameter_list
        base_name = self.GetModel().filename.rsplit(".", 1)[0]
        if not os.path.exists(base_name + "_%i.tab" % 0):
            return
        for ignore in range(len(pl.parameter_list)):
            pl.Select(0, True)
            pl.DeleteItem(None)
        i = 0
        p = Parameters()
        model = self.GetModel()
        start_params = model.parameters.data
        while os.path.exists(base_name + "_%i.tab" % i):
            txt = open(base_name + "_%i.tab" % i, "r").read()
            p.set_ascii_input(txt)
            model.parameters.data = deepcopy(p.data)
            model.simulate()
            pl.AddItem(None, simulate=False)
            i += 1
        model.parameters.data = start_params
        self.parent.eh_tb_simulate(None)


class ParameterList(wx.ListCtrl, ListCtrlAutoWidthMixin):
    """
    The ListCtrl for the materials data.
    """

    def __init__(self, parent, plugin):
        wx.ListCtrl.__init__(self, parent, -1, style=wx.LC_REPORT | wx.LC_VIRTUAL | wx.LC_EDIT_LABELS)
        ListCtrlAutoWidthMixin.__init__(self)
        self.plugin = plugin
        dpi_scale_factor = wx.GetApp().dpi_scale_factor
        if sys.platform.startswith("win"):
            font = wx.Font(
                9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, faceName="Lucida Sans Unicode"
            )
        else:
            font = wx.Font(
                9, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, encoding=wx.FONTENCODING_UNICODE
            )
        self.SetFont(font)

        # Set list length
        self.SetItemCount(0)

        self.parameter_list = []
        # Set the column headers
        for col, (text, width) in enumerate(
            [
                ("#", 24),
                ("FOM", 60),
                ("Show", 60),
                ("P1", 60),
                ("P2", 60),
            ]
        ):
            self.InsertColumn(col, text, width=width)

    def OnSelectionChanged(self, evt):
        if not self.toggleshow:
            indices = self._GetSelectedItems()
            indices.sort()
            if not indices == self.show_indices:
                self.data_cont.show_data(indices)
                self._UpdateData("Show data set flag toggled", data_changed=True)
                # Forces update of list control
                self.SetItemCount(self.data_cont.get_count())
        evt.Skip()

    def OnGetItemText(self, item, col):
        if col == 0:
            return str(item + 1)
        data = self.parameter_list[item]
        if col == 1:
            return "%.4g" % data[0]
        if col == 2:
            return str(data[1])
        if col == 3:
            try:
                idx = self.plugin.P1_index.GetValue()
                model = self.plugin.GetModel()
                idx = [i for i, p in enumerate(model.parameters.data) if p[2]][idx]
                return "%g" % data[2][idx][1]
            except:
                return ""
        if col == 4:
            try:
                idx = self.plugin.P2_index.GetValue()
                model = self.plugin.GetModel()
                idx = [i for i, p in enumerate(model.parameters.data) if p[2]][idx]
                return "%g" % data[2][idx][1]
            except:
                return ""

    def _GetSelectedItems(self):
        """_GetSelectedItems(self) --> indices [list of integers]
        Function that yields a list of the currently selected items
        position in the list. In order of selction, i.e. no order.
        """
        indices = [self.GetFirstSelected()]
        while indices[-1] != -1:
            indices.append(self.GetNextSelected(indices[-1]))

        # Remove the last will be -1
        indices.pop(-1)
        return indices

    def _CheckSelected(self, indices):
        """_CheckSelected(self, indices) --> bool
        Checks so at least data sets are selcted, otherwise show a dialog box
        and return False
        """
        # Check so that one dataset is selected
        if len(indices) == 0:
            ShowInfoDialog(self, "At least one data set has to be selected")
            return False
        return True

    def DeleteItem(self, ignore):
        index = self.GetFirstSelected()
        self.parameter_list.pop(index)
        self.SetItemCount(len(self.parameter_list))
        self.plugin.OnAutoStore(None)

    def AddItem(self, ignore, simulate=True):
        model = self.plugin.GetModel()
        slds = deepcopy(model.script_module.SLD)
        params = deepcopy(model.parameters.data)
        data = [di.copy() for di in model.data]
        self.parameter_list.append([model.fom or 0.0, True, params, data, slds])
        self.SetItemCount(len(self.parameter_list))
        self.plugin.OnAutoStore(None)
        if simulate:
            self.plugin.parent.eh_tb_simulate(None)

    def ApplyParameters(self, ignore):
        index = self.GetFirstSelected()
        model = self.plugin.GetModel()
        was_stored = False
        for ignore, ignore, params, ignore, ignore in self.parameter_list:
            if model.parameters.data == params:
                was_stored = True
                break
        if not was_stored:
            message = "Do you want to store the current parameters in the Vault" + " before applying the selected ones?"
            result = ShowQuestionDialog(self, message, "Store current parameters?")
            if result:
                self.AddItem(None)
        model.parameters.data = deepcopy(self.parameter_list[index][2])
        self.plugin.parent.paramter_grid._grid_changed()
        self.plugin.parent.paramter_grid.grid.ForceRefresh()
        self.plugin.parent.eh_tb_simulate(None)

    def TogglePlot(self, ignore):
        index = self.GetFirstSelected()
        self.parameter_list[index][1] = not self.parameter_list[index][1]
        self.RefreshItem(index)
        self.plugin.parent.eh_tb_simulate(None)

    def UpdateParams(self, ignore):
        if len(self.parameter_list) == 0:
            return
        self.RefreshItems(0, len(self.parameter_list) - 1)
