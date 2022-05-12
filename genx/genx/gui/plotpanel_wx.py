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
        self._last_graph = None
        self._initial_scale = True

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
        self.canvas.pointLabelFunc = self.DrawPointLabel

        self.canvas.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseScroll)
        self.canvas.canvas.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.canvas.canvas.Bind(wx.EVT_RIGHT_UP, self.OnContextMenu)
        self.canvas.enablePointLabel=True

    def OnPlotDraw(self, event):
        """ Sin, Cos, and Points """
        self.resetDefaults()
        self.update()

    def SetCallbackWindow(self, window):
        self.callback_window = window

    def SetAutoScale(self, do_autoscale):
        self.opt.autoscale = do_autoscale
        if self._last_graph:
            self.do_replot()

    def GetAutoScale(self):
        return self.opt.autoscale

    def SetZoom(self, active=False):
        self.opt.zoom = active
        if active:
            self.canvas.enableZoom = True
            self.canvas.enablePointLabel = False
        else:
            self.canvas.enableZoom = False
            self.canvas.enablePointLabel = True

    def GetZoom(self):
        return self.opt.zoom

    def GetYScale(self):
        return self.opt.y_scale

    def GetXScale(self):
        return self.opt.x_scale

    def UpdateConfigValues(self):
        self.SetZoom(self.opt.zoom)
        self.SetAutoScale(self.opt.autoscale)

    def ReadConfig(self):
        self._initial_scale = True
        super().ReadConfig()

    def ExecuteAutoscale(self):
        if not self.opt.autoscale and not self._initial_scale and self.opt.zoom:
            try:
                self._x_range = tuple(self.canvas.xCurrentRange)
            except Exception as e:
                pass
            try:
                self._y_range = tuple(self.canvas.yCurrentRange)
            except Exception:
                pass
        elif self.opt.autoscale or self._initial_scale:
            self._x_range = None
            self._y_range = None
            self._initial_scale = False
        self.canvas.logScale = (self.opt.x_scale == 'log', self.opt.y_scale == 'log')
        if self._x_range:
            if self.canvas.logScale[0]:
                self.canvas.xSpec=map(np.log10, self._x_range)
            else:
                self.canvas.xSpec=self._x_range
        else:
            self.canvas.xSpec = 'auto'
        if self._y_range:
            if self.canvas.logScale[1]:
                self.canvas.ySpec=tuple(map(np.log10, self._y_range))
            else:
                self.canvas.ySpec=self._y_range
        else:
            self.canvas.ySpec = 'auto'

    def resetDefaults(self):
        """Just to reset the fonts back to the PlotCanvas defaults"""
        self.canvas.SetFont(wx.Font(10,
                                    wx.FONTFAMILY_SWISS,
                                    wx.FONTSTYLE_NORMAL,
                                    wx.FONTWEIGHT_NORMAL)
                            )
        self.canvas.fontSizeAxis = 10
        self.canvas.fontSizeLegend = 7
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
        self._last_graph = self._to_plot
        self._to_plot = None
        self.do_replot()

    def do_replot(self):
        self.ExecuteAutoscale()
        self.canvas.Draw(self._last_graph)
        self.Refresh()

    def update(self, data=None):
        pass

    def get_colour(self, float_colors):
        int_colors=[int(ci*255) for ci in float_colors]
        return wx.Colour(*int_colors)

    def OnMouseScroll(self, event: wx.MouseEvent):
        if event.GetWheelAxis()==wx.MOUSE_WHEEL_HORIZONTAL:
            event.Skip()
            return
        rot = event.GetWheelRotation()/120.
        if event.ControlDown():
            rot *= 0.1
        if event.AltDown():
            # horizontal scaling
            if self._x_range:
                xmin, xmax = self._x_range
            else:
                xmin, xmax = self.canvas.xCurrentRange
            xrange = xmax-xmin
            if event.ShiftDown():
                if self.opt.x_scale=='log':
                    if rot>0:
                        self._x_range=(xmin*(1+2.33333*rot), xmax)
                    else:
                        self._x_range=(xmin/(1-2.33333*rot), xmax)
                else:
                    self._x_range=(xmin+xrange*0.2*rot, xmax)
            else:
                if self.opt.x_scale=='log':
                    if rot>0:
                        self._x_range=(xmin, xmax*(1+2.33333*rot))
                    else:
                        self._x_range=(xmin, xmax/(1-2.33333*rot))
                else:
                    self._x_range=(xmin, xmax+xrange*0.2*rot)
        else:
            # vertical scaling
            if self._y_range:
                ymin, ymax = self._y_range
            else:
                ymin, ymax = self.canvas.yCurrentRange
            yrange = ymax-ymin
            if event.ShiftDown():
                if self.opt.y_scale=='log':
                    if rot>0:
                        self._y_range=(ymin*(1+2.33333*rot), ymax)
                    else:
                        self._y_range=(ymin/(1-2.33333*rot), ymax)
                else:
                    self._y_range=(ymin+yrange*0.2*rot, ymax)
            else:
                if self.opt.y_scale=='log':
                    if rot>0:
                        self._y_range=(ymin, ymax*(1+2.33333*rot))
                    else:
                        self._y_range=(ymin, ymax/(1-2.33333*rot))
                else:
                    self._y_range=(ymin, ymax+yrange*0.2*rot)
        self.opt.autoscale = False
        tmp = self.opt.zoom
        self.opt.zoom = False
        self.do_replot()
        self.opt.zoom = tmp

    def OnMouseMove(self, event: wx.MouseEvent):
        if self.canvas.enablePointLabel:
            px, py = self.canvas.GetXY(event)
            if self.canvas.logScale[0]:
                px = np.log10(px)
            if self.canvas.logScale[1]:
                py = np.log10(py)
            dlst = self.canvas.GetClosestPoint((px, py), pointScaled=True)
            if dlst != []:
                curveNum, legend, pIndex, pointXY, scaledXY, distance = dlst
                px, py = pointXY
                if self.canvas.logScale[0]:
                    px = 10**px
                if self.canvas.logScale[1]:
                    py = 10**py
                mDataDict = {"curveNum": curveNum,
                             "legend": legend,
                             "pIndex": pIndex,
                             "pointXY": (px, py),
                             "scaledXY": scaledXY}
                self.canvas.UpdatePointLabel(mDataDict)
        event.Skip()

    def DrawPointLabel(self, dc, mDataDict):
        xmin, xmax = self.canvas.xCurrentRange
        ymin, ymax = self.canvas.yCurrentRange

        if self.canvas.logScale[0]:
            xmax = np.log10(xmax)
        if self.canvas.logScale[1]:
            ymax = np.log10(ymax)
        right, top = self.canvas.PositionUserToScreen((xmax, ymax))

        dc.SetPen(wx.Pen(wx.BLACK))
        dc.SetBrush(wx.Brush(wx.BLACK, wx.BRUSHSTYLE_SOLID))

        sx, sy = mDataDict["scaledXY"]  # scaled x,y of closest point
        # 10by10 square centered on point
        dc.DrawRectangle(sx - 5, sy - 5, 10, 10)
        px, py = mDataDict["pointXY"]
        cNum = mDataDict["curveNum"]
        pntIn = mDataDict["pIndex"]
        legend = mDataDict["legend"]
        # make a string to display
        txt=f"x={px:.6g}\ny={py:.6g}"
        txt_width = max([dc.GetTextExtent(ti)[0] for ti in txt.splitlines()])
        dc.DrawText(txt, int(right)-10-txt_width, int(top)+10)

    def OnContextMenu(self, event):
        menu = self.generate_context_menu()
        self.PopupMenu(menu)
        self.Unbind(wx.EVT_MENU)
        menu.Destroy()

    def generate_context_menu(self):
        menu = wx.Menu()
        zoomID = wx.NewId()
        menu.AppendCheckItem(zoomID, "Zoom")
        menu.Check(zoomID, self.GetZoom())

        def OnZoom(event):
            self.SetZoom(not self.GetZoom())

        self.Bind(wx.EVT_MENU, OnZoom, id=zoomID)
        zoomallID = wx.NewId()
        menu.Append(zoomallID, 'Zoom All')

        def zoomall(event):
            tmp = self.opt.autoscale
            self.SetAutoScale(True)
            self.opt.autoscale = tmp

        self.Bind(wx.EVT_MENU, zoomall, id=zoomallID)
        copyID = wx.NewId()
        menu.Append(copyID, "Copy")

        def copy(event):
            context = wx.ClientDC(self.canvas.canvas)
            memory = wx.MemoryDC()
            x, y = self.canvas.canvas.ClientSize
            bitmap = wx.Bitmap(x, y, -1)
            memory.SelectObject(bitmap)
            memory.Blit(0, 0, x, y, context, 0, 0)
            memory.SelectObject(wx.NullBitmap)
            bmp_obj = wx.BitmapDataObject()
            bmp_obj.SetBitmap(bitmap)

            if not wx.TheClipboard.IsOpened():
                open_success = wx.TheClipboard.Open()
                if open_success:
                    wx.TheClipboard.SetData(bmp_obj)
                    wx.TheClipboard.Close()
                    wx.TheClipboard.Flush()


        menu.AppendSeparator()

        self.Bind(wx.EVT_MENU, copy, id=copyID)
        yscalemenu = wx.Menu()
        logID = wx.NewId()
        linID = wx.NewId()
        yscalemenu.AppendRadioItem(logID, "log")
        yscalemenu.AppendRadioItem(linID, "linear")
        menu.Append(-1, "y-scale", yscalemenu)
        if self.opt.y_scale=='log':
            yscalemenu.Check(logID, True)
        else:
            yscalemenu.Check(linID, True)

        def yscale_log(event):
            self.opt.y_scale = 'log'
            self.do_replot()

        def yscale_lin(event):
            self.opt.y_scale = 'linear'
            self.do_replot()

        self.Bind(wx.EVT_MENU, yscale_log, id=logID)
        self.Bind(wx.EVT_MENU, yscale_lin, id=linID)
        xscalemenu = wx.Menu()
        logID = wx.NewId()
        linID = wx.NewId()
        xscalemenu.AppendRadioItem(logID, "log")
        xscalemenu.AppendRadioItem(linID, "linear")
        menu.Append(-1, "x-scale", xscalemenu)
        if self.opt.x_scale=='log':
            xscalemenu.Check(logID, True)
        else:
            xscalemenu.Check(linID, True)

        def xscale_log(event):
            self.opt.x_scale = 'log'
            self.do_replot()

        def xscale_lin(event):
            self.opt.x_scale = 'linear'
            self.do_replot()

        self.Bind(wx.EVT_MENU, xscale_log, id=logID)
        self.Bind(wx.EVT_MENU, xscale_lin, id=linID)
        autoscaleID = wx.NewId()
        menu.AppendCheckItem(autoscaleID, "Autoscale")
        menu.Check(autoscaleID, self.GetAutoScale())

        def OnAutoScale(event):
            self.SetAutoScale(not self.GetAutoScale())

        self.Bind(wx.EVT_MENU, OnAutoScale, id=autoscaleID)
        return menu

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
        sizer = self.GetSizer()
        sizer.GetItem(0).SetProportion(3)

        self.canvas_fom = wxplot.PlotCanvas(self)
        self.canvas_fom.enableAntiAliasing = True
        sizer.Add(self.canvas_fom, 1, wx.EXPAND)

        self.update = self.plot_data
        self._last_xlabel = 'x'
        self._last_ylabel = 'y'
        self.resetDefaultsFOM()

    def resetDefaultsFOM(self):
        self.canvas_fom.SetFont(wx.Font(10,
                                    wx.FONTFAMILY_SWISS,
                                    wx.FONTSTYLE_NORMAL,
                                    wx.FONTWEIGHT_NORMAL)
                            )
        self.canvas_fom.fontSizeAxis = 10
        self.canvas_fom.fontSizeLegend = 7
        self.canvas_fom.enableAxesValues = (False, True)

    def ExecuteAutoscale(self):
        if getattr(self, 'canvas_fom', None):
            self.canvas_fom.logScale = (self.canvas.logScale[0], False)
            self.canvas_fom.xSpec = tuple(self.canvas.xCurrentRange)
            self.canvas_fom.ySpec = 'auto'
            if not self.opt.autoscale and self._x_range and self.canvas.logScale[0]:
                        self.canvas.xSpec=map(np.log10, self.canvas.xCurrentRange)

    def plot_result(self, graph, fom_graph, delayed=False):
        self._to_fom = fom_graph
        super().plot_result(graph, delayed=delayed)

    def do_plot_result(self):
        if self._to_plot is None:
            return
        self._last_graph = (self._to_plot, self._to_fom)
        self._to_plot = None
        self._to_fom = None
        self.do_replot()

    def do_replot(self):
        super().ExecuteAutoscale()
        self.canvas.Draw(self._last_graph[0])
        if self._last_graph[1]:
            self.ExecuteAutoscale()
            self.canvas_fom.Draw(self._last_graph[1])
        self.Refresh()

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

    def get_fom_lines(self, data:DataList):
        lines = []
        p_datasets = [data_set for data_set in data if data_set.show]

        for data_set in p_datasets:
            data1 = np.vstack([data_set.x, np.nan_to_num(data_set.y_fom)]).T
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
        self.plot_result(wxplot.PlotGraphics(ax1_lines, "", xLabel=self._last_xlabel, yLabel=self._last_ylabel),
                         None)

    def plot_data_fit(self, data: DataList, xlabel=None, ylabel=None):
        return self.plot_data_sim(data, xlabel, ylabel)

    def plot_data_sim(self, data:DataList, xlabel=None, ylabel=None, delayed=False):
        if xlabel is not None:
            self._last_xlabel = xlabel.replace('$^{-1}$', '⁻¹')
        if ylabel is not None:
            self._last_ylabel = ylabel.replace('$^{-1}$', '⁻¹')
        ax1_lines = self.get_data_plots(data)+self.get_sim_lines(data)
        ax2_lines = self.get_fom_lines(data)
        self.plot_result(wxplot.PlotGraphics(ax1_lines, "", xLabel=self._last_xlabel, yLabel=self._last_ylabel),
                         wxplot.PlotGraphics(ax2_lines, "", xLabel="", yLabel="FOM"),
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
