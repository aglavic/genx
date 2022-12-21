# -*- coding: utf8 -*-
'''
=================
FrequencyAnalysis
=================

A plugin to show reflectivity frequency components by using FFT/CWT of reflectivity corrected by critical q-values.

Written by Artur Glavic
'''

import wx
from numpy import isnan, linspace
from .help_modules.frequency_analysis import transform
from genx.gui.plotpanel import PlotPanel, BasePlotConfig
from .. import add_on_framework as framework


class FAPlotConfig(BasePlotConfig):
    section='frequency analysis plot'

class FAPlotPanel(wx.Panel):
    ''' Widget for plotting the frequency analysis of datasets.
    '''

    def __init__(self, parent, plugin, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        wx.Panel.__init__(self, parent)
        self.plot=PlotPanel(self, -1, color, dpi, FAPlotConfig, style, **kwargs)
        self.plugin=plugin

        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.plot, 1, wx.EXPAND | wx.GROW | wx.ALL)

        self.plot.update(None)
        self.plot.ax=self.plot.figure.add_subplot(111)
        box=self.plot.ax.get_position()
        self.plot.ax.set_position([box.x0, box.y0, box.width*0.95, box.height])
        self.plot.ax.set_autoscale_on(True)
        self.plot.update=self.Plot
        self.SetSizer(sizer)
        self.plot_dict={}

    def SetZoom(self, active=False):
        return self.plot.SetZoom(active)

    def GetZoom(self):
        return self.plot.GetZoom()

    def Plot(self):
        ''' Plot(self) --> None

        Plotting the sample Sample.
        '''
        colors=['b', 'r', 'g', 'c', 'm', 'y', 'k']
        # self.plot_dict = model.sample.SimSLD(None, model.inst)
        while len(self.plot.ax.lines)>0:
            self.plot.ax.lines[0].remove()
        i=0
        data=self.plugin.GetModel().get_data()

        d=linspace(0, 500, 250)
        # New style sim function with one sld for each simulation
        for i, di in enumerate(data):
            if i%2!=0:
                continue
            if di.show:
                try:
                    if not isnan(di.y).all() and (di.y>0).any():
                        _,mag=transform(di.x, di.y, Qc=0.05,
                                        trans_type='FT',
                                        log=False, Q4=True,
                                        derivate=False, derivN=3,
                                        Qmin=0.08, Qmax=None,
                                        D=d, wavelet_scaling=0.)
                        self.plot.ax.plot(d, mag,
                                          color=di.data_color,
                                          ls=di.data_linetype, lw=di.data_linethickness,
                                          marker=di.data_symbol, ms=di.data_symbolsize,
                                          label='data '+di.name)

                    if di.y_sim is not None and not isnan(di.y_sim).all() and (di.y_sim>0).any():
                        _,mag=transform(di.x, di.y_sim, Qc=0.05,
                                        trans_type='FT',
                                        log=False, Q4=True,
                                        derivate=False, derivN=3,
                                        Qmin=0.08, Qmax=None,
                                        D=d, wavelet_scaling=0.)
                        self.plot.ax.plot(d, mag,
                                          color=di.sim_color,
                                          ls=di.sim_linetype, lw=di.sim_linethickness,
                                          marker=di.sim_symbol, ms=di.sim_symbolsize,
                                          label='sim '+di.name)
                except KeyError:
                    pass

        self.plot.ax.legend(loc='upper right',  # bbox_to_anchor=(1, 0.5),
                            framealpha=0.5,
                            fontsize="small", ncol=1)

        self.plot.ax.yaxis.label.set_text('')
        self.plot.ax.xaxis.label.set_text('x')
        wx.CallAfter(self.plot.flush_plot)
        self.plot.AutoScale()

    def ReadConfig(self):
        '''ReadConfig(self) --> None

        Reads in the config file
        '''
        return self.plot.ReadConfig()

    def GetYScale(self):
        '''GetYScale(self) --> String

        Returns the current y-scale in use. Currently the string
        'log' or 'linear'. If the axes does not exist it returns None.
        '''
        if self.plot.ax:
            return self.plot.ax.get_yscale()
        else:
            return None

    def GetXScale(self):
        '''GetXScale(self) --> String

        Returns the current x-scale in use. Currently the string
        'log' or 'linear'. If the axes does not exist it returns None.
        '''
        if self.plot.ax:
            return self.plot.ax.get_xscale()
        else:
            return None

class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent=parent

        # Create the SLD plot
        FA_plot_panel=self.NewPlotFolder('Frequency Analysis')
        FA_sizer=wx.BoxSizer(wx.HORIZONTAL)
        FA_plot_panel.SetSizer(FA_sizer)
        self.FA_plot=FAPlotPanel(FA_plot_panel, self)
        FA_sizer.Add(self.FA_plot, 1, wx.EXPAND | wx.GROW | wx.ALL)
        FA_plot_panel.Layout()

    def OnSimulate(self, event):
        '''OnSimulate(self, event) --> None

        Updates stuff after simulation
        '''
        # Calculate and update the sld plot
        wx.CallAfter(self.FA_plot.Plot)

    def OnFittingUpdate(self, event):
        '''OnSimulate(self, event) --> None

        Updates stuff during fitting
        '''
        wx.CallAfter(self.FA_plot.Plot())
