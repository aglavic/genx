# -*- coding: utf8 -*-
'''
=============
SpinAsymmetry
=============

A plugin to show an additional graph to plot the spin-asymmetry of polarized measurements.

Written by Artur Glavic
'''

import wx
from numpy import isnan
from genx.gui.plotpanel import PlotPanel, BasePlotConfig
from .. import add_on_framework as framework

class SAPlotConfig(BasePlotConfig):
    section='spin asymmetry plot'

class SAPlotPanel(wx.Panel):
    ''' Widget for plotting the spin-asymmetry of datasets.
    '''

    def __init__(self, parent, plugin, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        wx.Panel.__init__(self, parent)
        self.plot=PlotPanel(self, -1, color, dpi, SAPlotConfig, style, **kwargs)
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
        sld_units=[]

        # New style sim function with one sld for each simulation
        for i, di in enumerate(data):
            if i%2!=0:
                continue
            if di.show:
                try:
                    dj=data[i+1]
                    li=len(di.x)
                    lj=len(dj.x)
                    l=min(li, lj)
                    SAdata=(di.y[:l]-dj.y[:l])/(di.y[:l]+dj.y[:l])
                    if not isnan(SAdata).all():
                        self.plot.ax.plot(di.x[:l], SAdata,
                                          color=di.data_color,
                                          ls=di.data_linetype, lw=di.data_linethickness,
                                          marker=di.data_symbol, ms=di.data_symbolsize,
                                          label='data '+di.name+'&'+dj.name)
                    SAsim=(di.y_sim[:l]-dj.y_sim[:l])/(di.y_sim[:l]+dj.y_sim[:l])
                    if not isnan(SAsim).all():
                        self.plot.ax.plot(di.x[:l], SAsim,
                                          color=di.sim_color,
                                          ls=di.sim_linetype, lw=di.sim_linethickness,
                                          marker=di.sim_symbol, ms=di.sim_symbolsize,
                                          label='sim '+di.name+'&'+dj.name)
                except:
                    pass

        self.plot.ax.legend(loc='upper right',  # bbox_to_anchor=(1, 0.5),
                            framealpha=0.5,
                            fontsize="small", ncol=1)

        self.plot.ax.yaxis.label.set_text('Spin Asymmetry')
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
        SA_plot_panel=self.NewPlotFolder('Spin-Asymmetry')
        SA_sizer=wx.BoxSizer(wx.HORIZONTAL)
        SA_plot_panel.SetSizer(SA_sizer)
        self.SA_plot=SAPlotPanel(SA_plot_panel, self)
        SA_sizer.Add(self.SA_plot, 1, wx.EXPAND | wx.GROW | wx.ALL)
        SA_plot_panel.Layout()

    def OnSimulate(self, event):
        '''OnSimulate(self, event) --> None

        Updates stuff after simulation
        '''
        # Calculate and update the sld plot
        wx.CallAfter(self.SA_plot.Plot)

    def OnFittingUpdate(self, event):
        '''OnSimulate(self, event) --> None

        Updates stuff during fitting
        '''
        wx.CallAfter(self.SA_plot.Plot())
