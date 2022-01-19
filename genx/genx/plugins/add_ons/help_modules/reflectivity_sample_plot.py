"""
Plotting widget for sample SLDs.
"""
import colorsys
from dataclasses import dataclass

import wx
import numpy as np
import time
import os

from genx.core.custom_logging import iprint
from genx.data import DataList
from genx.gui.plotpanel import BasePlotConfig, PlotPanel

@dataclass
class SamplePlotConfig(BasePlotConfig):
    section = 'sample plot'
    data_derived_color: bool = False
    show_single_model: bool = False
    legend_outside: bool = False


class SamplePlotPanel(PlotPanel):
    '''
    Widget for plotting the scattering length density of a sample.
    '''
    opt: SamplePlotConfig

    def __init__(self, parent, plugin, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, SamplePlotConfig, style, **kwargs)
        self.plugin = plugin

        self.update = self.Plot
        self.create_axes()

    def create_axes(self):
        self.ax = self.figure.add_subplot(111)
        box = self.ax.get_position()
        if self.opt.legend_outside:
            self.ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        else:
            self.ax.set_position([box.x0, box.y0, box.width*0.95, box.height])

    def Plot(self):
        '''
        Plotting the sample.
        '''
        if not self.ax:
            # self.ax = self.figure.add_subplot(111)
            self.create_axes()

        data: DataList = self.plugin.GetModel().get_data()
        if self.opt.data_derived_color:
            colors = []
            for di in data:
                h, s, v = colorsys.rgb_to_hsv(*di.sim_color)
                ci = [
                    colorsys.hsv_to_rgb(h, 1.0, 1.0),
                    colorsys.hsv_to_rgb(h, 1.0, 0.5),
                    colorsys.hsv_to_rgb(h, 0.4, 1.0),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 1.0, 1.0),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 1.0, 0.5),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 0.4, 1.0),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 1.0, 1.0),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 1.0, 0.5),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 0.4, 1.0),
                    ]
                colors.append(ci)
        else:
            colors = [['b', 'r', 'g', 'c', 'm', 'y', 'k']]*len(data)

        model = self.plugin.GetModel().script_module
        self.plot_dicts = []
        while len(self.ax.lines)>0:
            self.ax.lines[0].remove()
        i = 0
        sld_units = []

        if self.plugin.sim_returns_sld and model._sim:
            # New style sim function with one sld for each simulation
            self.plot_dicts = model.SLD
            for sim in range(len(self.plot_dicts)):
                if data[sim].show:
                    for key in self.plot_dicts[sim]:
                        try:
                            if key in ['z', 'SLD unit'] or (self.plot_dicts[0][key]==0).all():
                                # skip lines that are all zero to keep legend cleaner
                                continue
                        except KeyError:
                            pass
                        is_imag = key[:2]=='Im' or key[:4]=='imag'
                        if (is_imag and self.plugin.show_imag_sld) or not is_imag:
                            if self.opt.show_single_model:
                                label = key
                            else:
                                label = data[sim].name+'\n'+key
                            self.ax.plot(self.plot_dicts[sim]['z'], self.plot_dicts[sim][key],
                                         color=colors[sim][i%len(colors[sim])], label=label)

                            if 'SLD unit' in self.plot_dicts[sim]:
                                if not self.plot_dicts[sim]['SLD unit'] in sld_units:
                                    sld_units.append(self.plot_dicts[sim]['SLD unit'])
                            i += 1
                if self.opt.show_single_model:
                    break
        else:
            # Old style plotting just one sld
            if self.plugin.GetModel().compiled:
                try:
                    sample = model.sample
                except AttributeError:
                    iprint("Warning: Could not locate the sample in the model")
                    return
                plot_dict = sample.SimSLD(None, None, model.inst)
                self.plot_dicts = [plot_dict]
                for key in self.plot_dicts[0]:
                    if key in ['z', 'SLD unit'] or (self.plot_dicts[0][key]==0).all():
                        # skip lines that are all zero to keep legend cleaner
                        continue
                    is_imag = key[:2]=='Im' or key[:4]=='imag'
                    if (is_imag and self.plugin.show_imag_sld) or not is_imag:
                        label = key
                        self.ax.plot(self.plot_dicts[0]['z'], self.plot_dicts[0][key],
                                     colors[-1][i%len(colors[-1])], label=label)

                        if 'SLD unit' in self.plot_dicts[0]:
                            if not self.plot_dicts[0]['SLD unit'] in sld_units:
                                sld_units.append(self.plot_dicts[0]['SLD unit'])
                        i += 1

        if i>0:
            if self.opt.legend_outside:
                self.ax.legend(loc='center left', framealpha=0.5, bbox_to_anchor=(1, 0.5), fontsize="small", ncol=1)
            else:
                self.ax.legend(loc='upper right', framealpha=0.5, fontsize="small", ncol=1)
            sld_unit = ', '.join(sld_units)
            self.ax.yaxis.label.set_text('$\mathrm{\mathsf{SLD\,[%s]}}$'%sld_unit)
            self.ax.xaxis.label.set_text('$\mathrm{\mathsf{ z\,[\AA]}}$')
            wx.CallAfter(self.flush_plot)
            self.AutoScale()

    def SavePlotData(self, filename):
        ''' Save all the SLD profiles to file with filename.'''
        # Check so that there are a simulation to save
        try:
            self.plot_dicts
        except:
            self.plugin.ShowWarningDialog('No SLD data to save.'
                                          ' Simulate the model first and then save.')
            return
        base, ext = os.path.splitext(filename)
        if ext=='':
            ext = '.dat'
        data = self.plugin.GetModel().get_data()
        for sim in range(len(self.plot_dicts)):
            new_filename = (base+'%03d'%sim+ext)
            save_array = np.array([self.plot_dicts[sim]['z']])
            header = ' z [Å]'+' '*12
            for key in self.plot_dicts[sim]:
                if key!='z' and key!='SLD unit':
                    save_array = np.r_[save_array, [self.plot_dicts[sim][key]]]
                    header += f' {key:19s}'
            with open(new_filename, 'w', encoding='utf-8') as f:
                f.write("# File exported from GenX's Reflectivity plugin\n")
                f.write("# File created: %s\n"%time.ctime())
                f.write("# Simulated SLD for data set: %s\n"%data[sim].name)
                if 'SLD unit' in self.plot_dicts[sim]:
                    sld_unit = self.plot_dicts[sim]['SLD unit']
                    sld_unit = sld_unit.replace('10^{-6}\\AA^{-2}', '10⁻⁶ Å⁻²')
                    sld_unit = sld_unit.replace('r_{e}/\\AA^{3}', 'rₑ/Å⁻³')
                    f.write('# SLD unit: %s\n'%sld_unit)
                f.write("# Coumns: \n")
                f.write('#'+header+'\n')
                np.savetxt(f, save_array.T, fmt='%-19.12e')

    def OnContextMenu(self, event):
        '''
        Callback to show the popmenu for the plot which allows various
        settings to be made.
        '''
        menu = self.generate_context_menu()
        menu.AppendSeparator()

        legendID = wx.NewId()
        menu.AppendCheckItem(legendID, "Legend Outside")
        menu.Check(legendID, self.opt.legend_outside)

        def OnLegend(event):
            self.opt.legend_outside = not self.opt.legend_outside
            self.figure.clear()
            self.create_axes()
            self.Plot()

        self.Bind(wx.EVT_MENU, OnLegend, id=legendID)

        coloringID = wx.NewId()
        menu.AppendCheckItem(coloringID, "Color from Data")
        menu.Check(coloringID, self.opt.data_derived_color)

        def OnColoring(event):
            self.opt.data_derived_color = not self.opt.data_derived_color
            self.Plot()

        self.Bind(wx.EVT_MENU, OnColoring, id=coloringID)

        singlemodelID = wx.NewId()
        menu.AppendCheckItem(singlemodelID, "Single Sample Model")
        menu.Check(singlemodelID, self.opt.show_single_model)

        def OnSingleModel(event):
            self.opt.show_single_model = not self.opt.show_single_model
            self.Plot()

        self.Bind(wx.EVT_MENU, OnSingleModel, id=singlemodelID)

        # Time to show the menu
        self.PopupMenu(menu)
        self.WriteConfig()

        self.Unbind(wx.EVT_MENU)
        menu.Destroy()
