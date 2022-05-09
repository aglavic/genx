"""
Native WX replacement for the matplotlib plot panel.
"""
import wx
import numpy as np
import threading
from dataclasses import dataclass
from typing import Type
from logging import debug, getLogger, ERROR
from wx.lib import plot as wxplot

from .custom_events import plot_position, state_changed, skips_event
from ..core.config import BaseConfig, Configurable
from ..model import Model
from ..data import DataList


MARKER_STYLES = {
    'o': 'circle',
    's': 'square',
    '.': 'dot',
    'd': 'triangle_down',
    '<': 'triangle',
    None: 'dot',
    }

LINE_STYLES = {
    '-': wx.PENSTYLE_SOLID,
    ':': wx.PENSTYLE_DOT,
    '--': wx.PENSTYLE_LONG_DASH,
    '.-': wx.PENSTYLE_DOT_DASH,
    None: wx.PENSTYLE_INVALID,
    }

@dataclass
class BasePlotConfig(BaseConfig):
    zoom: bool = False
    autoscale: bool = True
    x_scale: str = 'linear'
    y_scale: str = 'linear'

class PlotPanel(wx.Panel, Configurable):
    '''
    Base class for the plotting in GenX - all the basic functionallity
    should be implemented in this class. The plots should be derived from
    this class. These classes should implement an update method to update
    the plots.
    '''
    opt: BasePlotConfig

    def __init__(self, parent, id=-1, color=None, dpi=None, config_class: Type[BasePlotConfig] = None,
                 style=wx.NO_FULL_REPAINT_ON_RESIZE | wx.EXPAND | wx.ALL, **kwargs):
        debug('start init PlotPanel')
        wx.Panel.__init__(self, parent, id=id, style=style, **kwargs)
        Configurable.__init__(self, config_class)
        if dpi is None:
            dpi = wx.GetApp().dpi_scale_factor*96.  # wx.GetDisplayPPI()[0]
        self.parent = parent

        # Flags and bindings for zooming
        self.opt.load_config()
        self._x_range = None
        self._y_range = None
        self.zooming = False
        self.canvas = wxplot.PlotCanvas(self)
        self.canvas.enableAntiAliasing = True

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, 1, wx.EXPAND)
        self.SetSizer(sizer)
        self.OnPlotDraw(None)

    def OnPlotDraw(self, event):
        """ Sin, Cos, and Points """
        self.resetDefaults()
        self.update()

    def SetCallbackWindow(self, window):
        self.callback_window = window

    def SetAutoScale(self, do_autoscale):
        pass

    def SetZoom(self, active=False):
        pass

    def ExecuteAutoscale(self):
        # Allows to overwrite the behavior when autoscale is active
        self.canvas.xSpec = 'auto'
        self.canvas.ySpec = 'auto'
        if not self.opt.autoscale:
            if self._x_range:
                self.canvas.xSpec=self._x_range
            if self._y_range:
                self.canvas.ySpec=self._y_range

    def resetDefaults(self):
        """Just to reset the fonts back to the PlotCanvas defaults"""
        self.canvas.SetFont(wx.Font(10,
                                    wx.FONTFAMILY_SWISS,
                                    wx.FONTSTYLE_NORMAL,
                                    wx.FONTWEIGHT_NORMAL)
                            )
        self.canvas.fontSizeAxis = 10
        self.canvas.fontSizeLegend = 7
        self.canvas.logScale = (False, False)
        self.ExecuteAutoscale()

    def plot_result(self, graph, delayed=False):
        self._to_plot = graph
        if threading.main_thread() and not delayed:
            self.do_plot_result()
        else:
            wx.CallLater(10, self.do_plot_result)

    def do_plot_result(self):
        if self._to_plot is None:
            return
        graph = self._to_plot
        self._to_plot = None
        # update plot configuration
        self.canvas.logScale = (self.opt.x_scale == 'log', self.opt.y_scale == 'log')
        self.ExecuteAutoscale()
        # draw the new graph
        self.canvas.Draw(graph)
        self.Refresh()

    def update(self, data=None):
        pass

    def get_colour(self, float_colors):
        int_colors=[int(ci*255) for ci in float_colors]
        return wx.Colour(*int_colors)

class DataPanelConfig(BasePlotConfig):
    section = 'data plot'


class DataPlotPanel(PlotPanel):
    ''' Class for plotting the data and the fit
    '''
    _last_poptions = None

    def __init__(self, parent, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        self.main_ax_rect = (0.125, 0.3, 0.8, 0.6)
        self.sub_ax_rect = (0.125, 0.1, 0.8, 0.18)
        PlotPanel.__init__(self, parent, id, color, dpi, DataPanelConfig, style, **kwargs)
        self.update = self.plot_data
        self._last_xlabel = 'x'
        self._last_ylabel = 'y'

    def get_data_plots(self, data:DataList):
        lines = []
        p_datasets = [data_set for data_set in data if data_set.show]
        pe_datasets = [data_set for data_set in data if data_set.use_error and data_set.show]

        for data_set in p_datasets:
            data1 = np.vstack([data_set.x, np.nan_to_num(data_set.y)]).T
            if data_set.data_linetype!='':
                item = wxplot.PolyLine(data1,
                                       colour=self.get_colour(data_set.data_color),
                                       width=data_set.data_linethickness,
                                       style=LINE_STYLES[data_set.data_linetype],
                                       drawstyle='line',
                                       )
                lines.append(item)
            if data_set.data_symbol!='':
                item = wxplot.PolyMarker(data1,
                                         colour=self.get_colour(data_set.data_color),
                                         fillcolour=self.get_colour(data_set.data_color),
                                         marker=MARKER_STYLES[data_set.data_symbol],
                                         width=data_set.data_linethickness,
                                         size=data_set.data_symbolsize*0.2,
                                         fillstyle=wx.BRUSHSTYLE_SOLID,
                                         )
                lines.append(item)

            if data_set in pe_datasets:
                ylow = np.nan_to_num(data_set.y)-np.nan_to_num(data_set.error)
                yhigh = np.nan_to_num(data_set.y)+np.nan_to_num(data_set.error)
                for xi, li, hi in zip(data_set.x, ylow, yhigh):
                    data1 = [[xi, li], [xi, hi]]
                    item = wxplot.PolyLine(data1,
                                           colour=self.get_colour(data_set.data_color),
                                           width=data_set.data_linethickness,
                                           style=wx.PENSTYLE_SOLID,
                                           drawstyle='line',
                                           )
                    lines.append(item)
        return lines

    def get_sim_lines(self, data:DataList):
        lines = []
        p_datasets = [data_set for data_set in data if data_set.show]

        for data_set in p_datasets:
            data1 = np.vstack([data_set.x, np.nan_to_num(data_set.y_sim)]).T
            if data_set.sim_linetype!='':
                item = wxplot.PolyLine(data1,
                            colour=self.get_colour(data_set.sim_color),
                            width=data_set.sim_linethickness,
                            style=LINE_STYLES[data_set.sim_linetype],
                            drawstyle='line',
                            )
                lines.append(item)
            if data_set.sim_symbol!='':
                item = wxplot.PolyMarker(data1,
                                       colour=self.get_colour(data_set.sim_color),
                                       fillcolour=self.get_colour(data_set.sim_color),
                                       marker=MARKER_STYLES[data_set.sim_symbol],
                                       width=data_set.sim_linethickness,
                                       size=data_set.sim_symbolsize*0.2,
                                       fillstyle=wx.BRUSHSTYLE_SOLID,
                                       )

                lines.append(item)
        return lines

    def plot_data(self, data: DataList, xlabel=None, ylabel=None):
        if xlabel is not None:
            self._last_xlabel = xlabel.replace('$^{-1}$', '⁻¹')
        if ylabel is not None:
            self._last_ylabel = ylabel.replace('$^{-1}$', '⁻¹')
        ax1_lines = self.get_data_plots(data)
        ax2_lines = []
        self.plot_result(wxplot.PlotGraphics(ax1_lines, "", xLabel=self._last_xlabel, yLabel=self._last_ylabel))

    def plot_data_fit(self, data: DataList, xlabel=None, ylabel=None):
        return self.plot_data_sim(data, xlabel, ylabel)

    def plot_data_sim(self, data:DataList, xlabel=None, ylabel=None, delayed=False):
        if xlabel is not None:
            self._last_xlabel = xlabel.replace('$^{-1}$', '⁻¹')
        if ylabel is not None:
            self._last_ylabel = ylabel.replace('$^{-1}$', '⁻¹')
        ax1_lines = self.get_data_plots(data)+self.get_sim_lines(data)
        ax2_lines = []
        self.plot_result(wxplot.PlotGraphics(ax1_lines, "", xLabel=self._last_xlabel, yLabel=self._last_ylabel),
                         delayed=delayed)

    @skips_event
    def OnDataListEvent(self, event):
        data_list = event.GetData()
        if event.data_changed:
            self.plot_data(data_list)

    @skips_event
    def OnSimPlotEvent(self, event):
        model: Model = event.GetModel()
        data_list = model.get_data()
        try:
            ylabel = model.eval_in_model('globals().get("__ylabel__", getattr(model, "__ylabel__", "y"))')
        except NameError:
            ylabel = model.eval_in_model('globals().get("__ylabel__", "y")')
        try:
            xlabel = model.eval_in_model('globals().get("__xlabel__", getattr(model, "__xlabel__", "x"))')
        except NameError:
            xlabel = model.eval_in_model('globals().get("__xlabel__", "x")')
        self.plot_data_sim(data_list, xlabel=xlabel, ylabel=ylabel)

    @skips_event
    def OnSolverPlotEvent(self, event):
        if event.update_fit:
            self.plot_data_sim(event.data, delayed=True)


class ErrorPanelConfig(BasePlotConfig):
    section = 'fom plot'


class ErrorPlotPanel(PlotPanel):
    '''
    Class for plotting evolution of the error as a function of the generations.
    '''

    def __init__(self, parent, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, ErrorPanelConfig, style, **kwargs)
        self.update = self.errorplot
        self.update(None)

    def errorplot(self, data):
        if data is None:
            theta = np.arange(0.1, 10, 0.001)
            data1 = np.vstack([theta, np.floor(15-theta)]).T
        else:
            data1 = np.asarray(data)

        item = wxplot.PolyLine(data1,
                               colour='red',
                               width=2,
                               style=wx.PENSTYLE_SOLID,
                               drawstyle='line',
                               )
        self.plot_result(wxplot.PlotGraphics([item], "", xLabel='Iteration', yLabel='FOM'), delayed=True)

    @skips_event
    def OnSolverPlotEvent(self, event):
        '''
        Event handler function to connect to solver update events i.e.
        update the plot with the simulation
        '''
        fom_log = event.fom_log
        self.update(fom_log)


class ParsPanelConfig(BasePlotConfig):
    section = 'pars plot'


class ParsPlotPanel(PlotPanel):
    '''
    Class to plot the diffrent parametervalus during a fit.
    '''

    def __init__(self, parent, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        self.auto_x_range = (-0.5, 1.5)
        self.auto_y_range = (0., 1.)
        PlotPanel.__init__(self, parent, id, color, dpi, ParsPanelConfig, style, **kwargs)
        self.update = self.Plot
        self.update(None)

    def Plot(self, data):
        '''
        Plots each variable and its max and min value in the population.
        '''
        if data is None or not data.fitting:
            pars = np.arange(10)
            pdata = np.vstack([pars, pars*0.01+0.4]).T
            pop_min = pdata[:, 1]-0.05
            pop_max = pdata[:, 1]+0.05
        else:
            pop = np.array(data.population)
            norm = 1.0/(data.max_val-data.min_val)
            best = (np.array(data.values)-data.min_val)*norm
            pop_min = (pop.min(0)-data.min_val)*norm
            pop_max = (pop.max(0)-data.min_val)*norm

            pars = np.arange(len(best))
            pdata = np.vstack([pars, best]).T

        points = wxplot.PolyMarker(pdata,
                                 colour='red',
                                 fillcolour='red',
                                 marker='circle',
                                 width=2.0,
                                 size=1.5,
                                 fillstyle=wx.BRUSHSTYLE_SOLID,
                                 )
        bars = []
        for xi, mini, maxi in zip(pars, pop_min, pop_max):
            points1 = [(xi, mini), (xi, maxi)]
            bars.append(wxplot.PolyLine(points1, colour='blue', width=25))

        self.auto_x_range = (-0.5, pars[-1]+0.5)
        self.auto_y_range = (0., 1.)
        self.plot_result(wxplot.PlotGraphics(bars+[points], "",
                                             xLabel='Parameter Index (only fittable)',
                                             yLabel='Relative value in min/max range'),
                         delayed=True)

    def ExecuteAutoscale(self):
        # Allows to overwrite the behavior when autoscale is active
        self.canvas.xSpec = self.auto_x_range
        self.canvas.ySpec = self.auto_y_range
        if not self.opt.autoscale:
            if self._x_range:
                self.canvas.xSpec=self._x_range
            if self._y_range:
                self.canvas.ySpec=self._y_range

    @skips_event
    def OnSolverParameterEvent(self, event):
        '''
        Event handler function to connect to solver update events i.e.
        update the plot during the fitting
        '''
        self.update(event)


class FomPanelConfig(BasePlotConfig):
    section = 'fom scan plot'


class FomScanPlotPanel(PlotPanel):
    '''
    Class to take care of fom scans.
    '''

    def __init__(self, parent, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        self.type = 'project'
        PlotPanel.__init__(self, parent, id, color, dpi, FomPanelConfig, style, **kwargs)
        self.update = self.Plot
        self.update(None)

    def SetPlottype(self, type):
        '''SetScantype(self, type) --> None

        Sets the type of the scan type = "project" or "scan"
        '''
        if type.lower()=='project':
            self.type = 'project'
            self.opt.autoscale = False
        elif type.lower()=='scan':
            self.type = 'scan'
            self.opt.autoscale = True

    def Plot(self, data, l1='Parameter', l2='FOM'):
        '''
        Plots each variable and its max and min value in the population.
        '''
        if data is None:
            return self.plot_result(wxplot.PlotGraphics([wxplot.PolyLine([(-1, 0.1), (1, 0.1)])], "", xLabel=l1, yLabel=l2))
        x, y, bestx, besty, e_scale = data[:5]
        if self.type.lower()=='project':
            pdata = np.vstack([x,y]).T
            points = wxplot.PolyMarker(pdata,
                                       colour='blue',
                                       fillcolour='blue',
                                       marker='circle',
                                       width=2.0,
                                       size=1.5,
                                       fillstyle=wx.BRUSHSTYLE_SOLID,
                                       )
            bpoint = wxplot.PolyMarker([(bestx, besty)],
                                       colour='red',
                                       fillcolour='red',
                                       marker='circle',
                                       width=2.0,
                                       size=1.5,
                                       fillstyle=wx.BRUSHSTYLE_SOLID,
                                       )
            line = wxplot.PolyLine([(x.min(), besty*e_scale), (x.max(), besty*e_scale)], colour='red', width=1)
            items = [points, line, bpoint]

            self._x_range = (x.min(), x.max())
            self._y_range = (min(y.min(), besty)*0.95, (besty*e_scale-min(y.min(), besty))*2.0+min(y.min(), besty))
        elif self.type.lower()=='scan':
            pdata = np.vstack([x,y]).T
            points = wxplot.PolyLine(pdata, colour='blue', width=1)
            bpoint = wxplot.PolyMarker([(bestx, besty)],
                                       colour='red',
                                       fillcolour='red',
                                       marker='circle',
                                       width=2.0,
                                       size=1.5,
                                       fillstyle=wx.BRUSHSTYLE_SOLID,
                                       )
            line = wxplot.PolyLine([(x.min(), besty*e_scale), (x.max(), besty*e_scale)], colour='red', width=1)
            items = [points, line, bpoint]

            self._x_range = (x.min(), x.max())
            self._y_range = (min(y.min(), besty)*0.95, y.max()*1.05)

        self.plot_result(wxplot.PlotGraphics(items, "", xLabel=l1, yLabel=l2))
