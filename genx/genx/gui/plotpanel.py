"""
A extension of panel that implements matplotlib plotting libary.
"""
import warnings
from dataclasses import dataclass
from typing import Type
from logging import debug, getLogger, ERROR

import matplotlib
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.backends import backend_wx
from matplotlib import backend_bases
from matplotlib.figure import Figure

from numpy import *
import wx
import wx.lib.newevent
from wx import PAPER_A4, LANDSCAPE

from .custom_events import plot_position, state_changed, skips_event
from ..core.config import BaseConfig, Configurable
from ..model import Model


# deactivate matplotlib logging that we are not interested in
getLogger('matplotlib.ticker').setLevel(ERROR)
getLogger('matplotlib.font_manager').setLevel(ERROR)

zoom_state = False


# fix a but in wx/matplotlib where keeping a motion event reference breaks scrolling
# see: https://github.com/wxWidgets/Phoenix/issues/2034
def _onMotionFixed(self, event):
    """Start measuring on an axis."""
    x = event.GetX()
    y = self.figure.bbox.height-event.GetY()
    event.Skip()
    backend_bases.FigureCanvasBase.motion_notify_event(self, x, y, guiEvent=event.__class__(event))


backend_wx._FigureCanvasWxBase._onMotion = _onMotionFixed


@dataclass
class BasePlotConfig(BaseConfig):
    zoom: bool = False
    autoscale: bool = True
    x_scale: str = 'linear'
    y_scale: str = 'linear'


# ==============================================================================
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
        self.callback_window = self
        debug('init PlotPanel - setup figure')
        self.figure = Figure(figsize=(1.0, 1.0), dpi=dpi)
        debug('init PlotPanel - setup canvas')
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.canvas.SetExtraStyle(wx.EXPAND)
        self.SetColor(color)
        self._resizeflag = True
        self.print_size = (15./2.54, 12./2.54)
        # self._SetSize()

        # Flags and bindings for zooming
        self.opt.load_config()
        self.zoom = self.opt.zoom
        self.y_scale = self.opt.y_scale
        self.x_scale = self.opt.x_scale
        self.autoscale = self.opt.autoscale
        self.zooming = False

        debug('init PlotPanel - bind events')
        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.OnLeftMouseButtonDown)
        self.canvas.Bind(wx.EVT_LEFT_UP, self.OnLeftMouseButtonUp)
        self.canvas.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.canvas.Bind(wx.EVT_LEFT_DCLICK, self.OnLeftDblClick)
        self.canvas.Bind(wx.EVT_MIDDLE_DOWN, self.OnLeftDblClick)
        self.canvas.Bind(wx.EVT_RIGHT_UP, self.OnContextMenu)
        self.canvas.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseScroll)

        cursor = wx.Cursor(wx.CURSOR_CROSS)
        self.canvas.SetCursor(cursor)
        self.old_scale_state = True
        self.ax = None

        debug('init PlotPanel - FigurePrinter and Bitmap')
        # Init printout stuff
        self.fig_printer = FigurePrinter(self)

        # Create the drawing bitmap
        self.bitmap = wx.Bitmap(1, 1, depth=wx.BITMAP_SCREEN_DEPTH)
        #        DEBUG_MSG("__init__() - bitmap w:%d h:%d" % (w,h), 2, self)
        self._isDrawn = False
        debug('end init PlotPanel')

    def SetColor(self, rgbtuple=None):
        ''' Set the figure and canvas color to be the same '''
        if not rgbtuple:
            rgbtuple = self.parent.GetBackgroundColour()
            # wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE).Get()
        col = [c/255. for c in rgbtuple]
        self.figure.set_facecolor(col)
        self.figure.set_edgecolor(col)
        self.canvas.SetBackgroundColour(wx.Colour(*rgbtuple))

    def _onSize(self, evt):
        self._resizeflag = True
        self._SetSize()
        # self.canvas.draw(repaint = False)

    def _onIdle(self, evt):
        if self._resizeflag:
            self._resizeflag = False
            self._SetSize()
            # self.canvas.gui_repaint(drawDC = wx.PaintDC(self))

    def _SetSize(self, pixels=None):
        '''
        This method can be called to force the Plot to be a desired
        size which defaults to the ClientSize of the Panel.
        '''
        if not pixels:
            pixels = self.GetClientSize()
        if pixels[0]==0 or pixels[1]==0:
            return

        self.canvas.SetSize(pixels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.figure.tight_layout(h_pad=0)
            except ValueError:
                pass
        # self.figure.set_size_inches(pixels[0]/self.figure.get_dpi()
        # , pixels[1]/self.figure.get_dpi())

    def UpdateConfigValues(self):
        self.SetXScale(self.opt.x_scale)
        self.SetYScale(self.opt.y_scale)
        self.SetZoom(self.opt.zoom)
        self.SetAutoScale(self.opt.autoscale)

    def WriteConfig(self):
        self.opt.x_scale = self.x_scale
        self.opt.y_scale = self.y_scale
        self.opt.autoscale = self.autoscale
        self.opt.zoom = self.zoom
        Configurable.WriteConfig(self)

    def SetZoom(self, active=False):
        '''
        set the zoomstate
        '''
        # if not self.zoom_sel:
        # self.zoom_sel = RectangleSelector(self.ax,\
        # self.box_select_callback, drawtype='box',useblit=False)
        # print help(self.zoom_sel.ignore)

        if active:
            # self.zoom_sel.ignore = lambda x: False
            self.zoom = True
            cursor = wx.Cursor(wx.CURSOR_MAGNIFIER)
            self.canvas.SetCursor(cursor)
            if self.callback_window:
                evt = state_changed(zoomstate=True, yscale=self.GetYScale(), autoscale=self.autoscale,
                                    xscale=self.GetXScale())
                wx.PostEvent(self.callback_window, evt)
            if self.ax:
                # self.ax.set_autoscale_on(False)
                self.old_scale_state = self.GetAutoScale()
                self.SetAutoScale(False)

        else:
            # self.zoom_sel.ignore = lambda x: True
            self.zoom = False
            cursor = wx.Cursor(wx.CURSOR_CROSS)
            self.canvas.SetCursor(cursor)
            if self.callback_window:
                evt = state_changed(zoomstate=False, yscale=self.GetYScale(), autoscale=self.autoscale,
                                    xscale=self.GetXScale())
                wx.PostEvent(self.callback_window, evt)
            if self.ax:
                # self.ax.set_autoscale_on(self.autoscale)
                self.SetAutoScale(self.old_scale_state)
        self.WriteConfig()

    def GetZoom(self):
        '''
        Returns the zoom state of the plot panel. 
        '''
        return self.zoom

    def SetAutoScale(self, state):
        '''
        Sets autoscale of the main axes wheter or not it should autoscale
        when plotting
        '''
        # self.ax.set_autoscale_on(state)
        self.autoscale = state
        self.WriteConfig()
        evt = state_changed(zoomstate=self.GetZoom(),
                            yscale=self.GetYScale(), autoscale=self.autoscale,
                            xscale=self.GetXScale())
        wx.PostEvent(self.callback_window, evt)

    def GetAutoScale(self):
        '''
        Returns the autoscale state, true if the plots is automatically
        scaled for each plot command.
        '''
        return self.autoscale

    def AutoScale(self, force=False):
        '''
        A log safe way to autoscale the plots - the ordinary axis tight 
        does not work for negative log data. This works!
        '''

        if not (self.autoscale or force):
            return
        # If nothing is plotted no autoscale use defaults...
        if sum([len(line.get_ydata())>0 for line in self.ax.lines])==0:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(1e-3, 1.0)
            return

        if self.y_scale=='log':
            # Find the lowest possible value of all the y-values that are
            # greater than zero. check so that y data contain data before min
            # is applied
            tmp = [line.get_ydata().compress(line.get_ydata()>0.0).min() \
                   for line in self.ax.lines if array(line.get_ydata()>0.0).sum()>0]
            if len(tmp)>0:
                ymin = min(tmp)
            else:
                ymin = 1e-3
            tmp = [line.get_ydata().compress(line.get_ydata()>0.0).max() \
                   for line in self.ax.lines if array(line.get_ydata()>0.0).sum()>0]
            if len(tmp)>0:
                ymax = max(tmp)
            else:
                ymax = 1
        else:
            ymin = min([array(line.get_ydata()).compress(isfinite(line.get_ydata())).min()
                        for line in self.ax.lines if len(line.get_ydata())>0 and any(isfinite(line.get_ydata()))])
            ymax = max([array(line.get_ydata()).compress(isfinite(line.get_ydata())).max()
                        for line in self.ax.lines if len(line.get_ydata())>0 and any(isfinite(line.get_ydata()))])
        tmp = [array(line.get_xdata()).min() for line in self.ax.lines if len(line.get_ydata())>0]
        if len(tmp)>0:
            xmin = min(tmp)
        else:
            xmin = 0
        tmp = [array(line.get_xdata()).max() \
               for line in self.ax.lines if len(line.get_ydata())>0]
        if len(tmp)>0:
            xmax = max(tmp)
        else:
            xmax = 1
        # Set the limits
        try:
            if xmin!=xmax:
                self.ax.set_xlim(xmin, xmax)
            if ymin!=ymax:
                self.ax.set_ylim(ymin*(1-sign(ymin)*0.05), ymax*(1+sign(ymax)*0.05))
            self.flush_plot()
        except UserWarning:
            pass

    def SetYScale(self, scalestring):
        '''
        Sets the y-scale of the main plotting axes. Currently accepts
        'log' or 'lin'.
        '''
        if not self.ax:
            return
        if scalestring=='log':
            self.y_scale = 'log'
        elif scalestring in ['linear', 'lin']:
            self.y_scale = 'linear'
        else:
            raise ValueError('Not allowed scaling')

        # do nothing if no data in current plot
        self.AutoScale(force=True)

        try:
            self.ax.set_yscale(self.y_scale)
        except OverflowError:
            self.AutoScale(force=True)
        except UserWarning:
            pass

        try:
            self.flush_plot()
        except UserWarning:
            pass
        self.WriteConfig()
        evt = state_changed(zoomstate=self.GetZoom(),
                            yscale=self.GetYScale(), autoscale=self.autoscale,
                            xscale=self.GetXScale())
        wx.PostEvent(self.callback_window, evt)

    def SetXScale(self, scalestring):
        '''
        Sets the x-scale of the main plotting axes. Currently accepts
        'log' or 'lin'.
        '''
        if not self.ax:
            return
        if scalestring=='log':
            self.x_scale = 'log'
            self.AutoScale(force=True)
            try:
                self.ax.set_xscale('log')
            except OverflowError:
                self.AutoScale(force=True)
            except UserWarning:
                pass
        elif scalestring=='linear' or scalestring=='lin':
            self.x_scale = 'linear'
            self.ax.set_xscale('linear')
            self.AutoScale(force=True)
        else:
            raise ValueError('Not allowed scaling')
        try:
            self.flush_plot()
        except UserWarning:
            pass
        self.WriteConfig()
        evt = state_changed(zoomstate=self.GetZoom(),
                            yscale=self.GetYScale(), autoscale=self.autoscale,
                            xscale=self.GetXScale())
        wx.PostEvent(self.callback_window, evt)

    def GetYScale(self):
        '''
        Returns the current y-scale in use. Currently the string
        'log' or 'linear'. If the axes does not exist it returns None.
        '''
        if self.ax:
            return self.ax.get_yscale()
        else:
            return None

    def GetXScale(self):
        '''
        Returns the current x-scale in use. Currently the string
        'log' or 'linear'. If the axes does not exist it returns None.
        '''
        if self.ax:
            return self.ax.get_xscale()
        else:
            return None

    def CopyToClipboard(self, event=None):
        '''
        Copy the plot to the clipboard.
        '''
        self.SetColor((255, 255, 255))
        self.canvas.draw()
        self.canvas.Copy_to_Clipboard(event=event)
        self.SetColor()
        self.canvas.draw()

    def PrintSetup(self, event=None):
        '''
        Sets up the printer. Creates a dialog box
        '''
        self.fig_printer.pageSetup()

    def PrintPreview(self, event=None):
        '''
        Prints a preview on screen.
        '''
        self.fig_printer.previewFigure(self.figure)

    def Print(self, event=None):
        '''
        Print the figure.
        '''
        self.fig_printer.printFigure(self.figure)

    def SetCallbackWindow(self, window):
        '''
        Sets the callback window that should recieve the events from 
        picking.
        '''

        self.callback_window = window

    def OnLeftDblClick(self, event: wx.MouseEvent):
        if self.ax and (self.zoom or event.MiddleDown()):
            tmp = self.GetAutoScale()
            self.SetAutoScale(True)
            self.AutoScale()
            self.SetAutoScale(tmp)
            # self.AutoScale()
            # self.flush_plot()
            # self.ax.set_autoscale_on(False)

    def OnLeftMouseButtonDown(self, event):
        self.start_pos = event.Position

        # print 'Left Mouse button pressed ', self.ax.transData.inverse_xy_tup(self.start_pos)
        class Point:
            pass

        p = Point()
        p.x, p.y = self.start_pos
        size = self.canvas.GetClientSize()
        p.y = (size.height-p.y)
        if self.zoom and self.ax:
            in_axes = self.ax.in_axes(p)
            if in_axes:
                self.zooming = True
                self.cur_rect = None
                self.canvas.CaptureMouse()
                self.overlay = wx.Overlay()
            else:
                self.zooming = False
        elif self.ax:
            size = self.canvas.GetClientSize()
            xy = self.ax.transData.inverted().transform(
                array([self.start_pos[0], size.height-self.start_pos[1]])
                [newaxis, :])
            x, y = xy[0, 0], xy[0, 1]
            if self.callback_window:
                evt = plot_position(text='(%.3e, %.3e)'%(x, y))
                wx.PostEvent(self.callback_window, evt)
            # print x,y

    def OnMouseMove(self, event):
        if self.zooming and event.Dragging() and event.LeftIsDown():
            self.cur_pos = event.Position

            class Point:
                pass

            p = Point()
            p.x, p.y = self.cur_pos
            size = self.canvas.GetClientSize()
            p.y = (size.height-p.y)
            in_axes = self.ax.in_axes(p)
            if in_axes:
                new_rect = (min(self.start_pos[0], self.cur_pos[0]),
                            min(self.start_pos[1], self.cur_pos[1]),
                            abs(self.cur_pos[0]-self.start_pos[0]),
                            abs(self.cur_pos[1]-self.start_pos[1]))
                self._DrawAndErase(new_rect, self.cur_rect)
                self.cur_rect = new_rect
        else:
            event.Skip()

    def OnMouseScroll(self, event: wx.MouseEvent):
        if event.GetWheelAxis()==wx.MOUSE_WHEEL_HORIZONTAL:
            event.Skip()
            return
        self.SetAutoScale(False)
        rot = event.GetWheelRotation()/120.
        if event.ControlDown():
            rot *= 0.1
        if event.AltDown():
            # horizontal scaling
            xmin, xmax = self.ax.get_xlim()
            xrange = xmax-xmin
            if event.ShiftDown():
                if self.x_scale=='log':
                    if rot>0:
                        self.ax.set_xlim(xmin*(1+2.33333*rot), xmax)
                    else:
                        self.ax.set_xlim(xmin/(1-2.33333*rot), xmax)
                else:
                    self.ax.set_xlim(xmin+xrange*0.2*rot, xmax)
            else:
                if self.x_scale=='log':
                    if rot>0:
                        self.ax.set_xlim(xmin, xmax*(1+2.33333*rot))
                    else:
                        self.ax.set_xlim(xmin, xmax/(1-2.33333*rot))
                else:
                    self.ax.set_xlim(xmin, xmax+xrange*0.2*rot)
        else:
            # vertical scaling
            ymin, ymax = self.ax.get_ylim()
            yrange = ymax-ymin
            if event.ShiftDown():
                if self.y_scale=='log':
                    if rot>0:
                        self.ax.set_ylim(ymin*(1+2.33333*rot), ymax)
                    else:
                        self.ax.set_ylim(ymin/(1-2.33333*rot), ymax)
                else:
                    self.ax.set_ylim(ymin+yrange*0.2*rot, ymax)
            else:
                if self.y_scale=='log':
                    if rot>0:
                        self.ax.set_ylim(ymin, ymax*(1+2.33333*rot))
                    else:
                        self.ax.set_ylim(ymin, ymax/(1-2.33333*rot))
                else:
                    self.ax.set_ylim(ymin, ymax+yrange*0.2*rot)
        self.flush_plot()

    def OnLeftMouseButtonUp(self, event):
        if self.canvas.HasCapture():
            # print 'Left Mouse button up'
            self.canvas.ReleaseMouse()
            del self.overlay
            if self.zooming and self.cur_rect:
                # Note: The coordinte system for matplotlib have a different 
                # direction of the y-axis and a different origin!
                size = self.canvas.GetClientSize()
                start = self.ax.transData.inverted().transform(
                    array([self.start_pos[0], size.height-self.start_pos[1]])[newaxis, :])
                end = self.ax.transData.inverted().transform(
                    array([self.cur_pos[0], size.height-self.cur_pos[1]])[newaxis, :])
                xend, yend = end[0, 0], end[0, 1]
                xstart, ystart = start[0, 0], start[0, 1]

                self.ax.set_xlim(min(xstart, xend), max(xstart, xend))
                self.ax.set_ylim(min(ystart, yend), max(ystart, yend))
                self.flush_plot()
            self.zooming = False

    def _DrawAndErase(self, box_to_draw, box_to_erase=None):
        rect = wx.Rect(*box_to_draw)

        # Draw the rubber-band rectangle using an overlay so it
        # will manage keeping the rectangle and the former window
        # contents separate.
        # dc = wx.ClientDC(self)
        dc = wx.ClientDC(self.canvas)
        odc = wx.DCOverlay(self.overlay, dc)
        odc.Clear()

        dc.SetPen(wx.Pen("black", 2, style=wx.DOT_DASH))
        dc.SetBrush(wx.Brush("black", style=wx.BRUSHSTYLE_TRANSPARENT))
        dc.DrawRectangle(rect)

    def OnContextMenu(self, event):
        '''
        Callback to show the popmenu for the plot which allows various 
        settings to be made.
        '''
        menu = self.generate_context_menu()

        # Time to show the menu
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
            tmp = self.GetAutoScale()
            self.SetAutoScale(True)
            self.AutoScale()
            self.SetAutoScale(tmp)
            # self.flush_plot()

        self.Bind(wx.EVT_MENU, zoomall, id=zoomallID)
        copyID = wx.NewId()
        menu.Append(copyID, "Copy")

        def copy(event):
            self.CopyToClipboard()

        menu.AppendSeparator()

        self.Bind(wx.EVT_MENU, copy, id=copyID)
        yscalemenu = wx.Menu()
        logID = wx.NewId()
        linID = wx.NewId()
        yscalemenu.AppendRadioItem(logID, "log")
        yscalemenu.AppendRadioItem(linID, "linear")
        menu.Append(-1, "y-scale", yscalemenu)
        if self.GetYScale()=='log':
            yscalemenu.Check(logID, True)
        else:
            yscalemenu.Check(linID, True)

        def yscale_log(event):
            if self.ax:
                self.SetYScale('log')
                self.AutoScale()
                self.flush_plot()

        def yscale_lin(event):
            if self.ax:
                self.SetYScale('lin')
                self.AutoScale()
                self.flush_plot()

        self.Bind(wx.EVT_MENU, yscale_log, id=logID)
        self.Bind(wx.EVT_MENU, yscale_lin, id=linID)
        xscalemenu = wx.Menu()
        logID = wx.NewId()
        linID = wx.NewId()
        xscalemenu.AppendRadioItem(logID, "log")
        xscalemenu.AppendRadioItem(linID, "linear")
        menu.Append(-1, "x-scale", xscalemenu)
        if self.GetXScale()=='log':
            xscalemenu.Check(logID, True)
        else:
            xscalemenu.Check(linID, True)

        def xscale_log(event):
            if self.ax:
                self.SetXScale('log')
                self.AutoScale()
                self.flush_plot()

        def xscale_lin(event):
            if self.ax:
                self.SetXScale('lin')
                self.AutoScale()
                self.flush_plot()

        self.Bind(wx.EVT_MENU, xscale_log, id=logID)
        self.Bind(wx.EVT_MENU, xscale_lin, id=linID)
        autoscaleID = wx.NewId()
        menu.AppendCheckItem(autoscaleID, "Autoscale")
        menu.Check(autoscaleID, self.GetAutoScale())

        def OnAutoScale(event):
            self.SetAutoScale(not self.GetAutoScale())

        self.Bind(wx.EVT_MENU, OnAutoScale, id=autoscaleID)
        return menu

    def flush_plot(self):
        # self._SetSize()
        # self.canvas.gui_repaint(drawDC = wx.PaintDC(self))
        # self.ax.set_yscale(self.y_scale)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.figure.tight_layout(h_pad=0)
        self.canvas.draw()

    def update(self, data):
        pass


# ==============================================================================
# Print out class borrowed from wxmpl
if callable(getattr(wx, 'PostScriptDC_SetResolution', None)):
    wx.PostScriptDC_SetResolution(300)


class FigurePrinter:
    """
    Provides a simplified interface to the wxPython printing framework that's
    designed for printing matplotlib figures.
    """

    def __init__(self, view, printData=None):
        """
        Create a new C{FigurePrinter} associated with the wxPython widget
        C{view}.  The keyword argument C{printData} supplies a C{wx.PrintData}
        object containing the default printer settings.
        """
        self.view = view

        if printData is None:
            self.pData = wx.PrintData()
        else:
            self.pData = printData

        self.pData.SetPaperId(PAPER_A4)
        self.pData.SetOrientation(LANDSCAPE)
        self.pData.SetNoCopies(1)

    def destroy(self):
        """
        Sets this object's C{view} attribute to C{None}.
        """
        self.view = None

    def getPrintData(self):
        """
        Return the current printer settings in their C{wx.PrintData} object.
        """
        return self.pData

    def setPrintData(self, printData):
        """
        Use the printer settings in C{printData}.
        """
        self.pData = printData

    def pageSetup(self):
        dlg = wx.PrintDialog(self.view)
        pdData = dlg.GetPrintDialogData()
        pdData.SetPrintData(self.pData)
        # pdData.SetSetupDialog(True)

        if dlg.ShowModal()==wx.ID_OK:
            self.pData = pdData.GetPrintData()
        dlg.Destroy()
        self.copyPrintData()

    def previewFigure(self, figure, title=None):
        """
        Open a "Print Preview" window for the matplotlib chart C{figure}.  The
        keyword argument C{title} provides the printing framework with a title
        for the print job.
        """
        window = self.view
        while not isinstance(window, wx.Frame):
            window = window.GetParent()
            assert window is not None

        fpo = FigurePrintout(figure, title)
        fpo4p = FigurePrintout(figure, title)
        preview = wx.PrintPreview(fpo, fpo4p, self.pData)
        frame = wx.PreviewFrame(preview, window, 'Print Preview')
        if self.pData.GetOrientation()==wx.PORTRAIT:
            frame.SetSize(wx.Size(450, 625))
        else:
            frame.SetSize(wx.Size(600, 500))
        frame.Initialize()
        frame.Show(True)
        self.copyPrintData()

    def printFigure(self, figure, title=None):
        """
        Open a "Print" dialog to print the matplotlib chart C{figure}.  The
        keyword argument C{title} provides the printing framework with a title
        for the print job.
        """
        pdData = wx.PrintDialogData()
        pdData.SetFromPage(1)
        pdData.SetToPage(1)
        pdData.SetPrintToFile(True)
        pdData.SetPrintData(self.pData)
        printer = wx.Printer(pdData)
        fpo = FigurePrintout(figure, title)
        self.pData = pdData.GetPrintData()
        if printer.Print(self.view, fpo, True):
            self.pData = pdData.GetPrintData()
        self.copyPrintData()

    def copyPrintData(self):
        '''Create a copy of the print data to avoid seg faults
        '''
        pData_new = wx.PrintData()
        pData_new.SetBin(self.pData.GetBin())
        pData_new.SetCollate(self.pData.GetCollate())
        pData_new.SetColour(self.pData.GetColour())
        pData_new.SetDuplex(self.pData.GetDuplex())
        pData_new.SetFilename(self.pData.GetFilename())
        pData_new.SetNoCopies(self.pData.GetNoCopies())
        pData_new.SetOrientation(self.pData.GetOrientation())
        pData_new.SetPaperId(self.pData.GetPaperId())
        pData_new.SetPaperSize(self.pData.GetPaperSize())
        pData_new.SetPrinterName(self.pData.GetPrinterName())
        pData_new.SetPrintMode(self.pData.GetPrintMode())
        pData_new.SetPrivData(self.pData.GetPrivData())
        pData_new.SetQuality(self.pData.GetQuality())
        self.pData = pData_new


class FigurePrintout(wx.Printout):
    """
    Render a matplotlib C{Figure} to a page or file using wxPython's printing
    framework.
    """

    ASPECT_RECTANGULAR = 1
    ASPECT_SQUARE = 2

    def __init__(self, figure, title=None, size=None, aspectRatio=None):
        """
        Create a printout for the matplotlib chart C{figure}.  The
        keyword argument C{title} provides the printing framework with a title
        for the print job.  The keyword argument C{size} specifies how to scale
        the figure, from 1 to 100 percent.  The keyword argument C{aspectRatio}
        determines whether the printed figure will be rectangular or square.
        """
        self.figure = figure

        figTitle = figure.gca().title.get_text()
        if not figTitle:
            figTitle = title or 'GenX plot'

        if size is None:
            size = 100
        elif size<0 or size>100:
            raise ValueError('invalid figure size')
        self.size = size

        if aspectRatio is None:
            aspectRatio = self.ASPECT_RECTANGULAR
        elif (aspectRatio!=self.ASPECT_RECTANGULAR
              and aspectRatio!=self.ASPECT_SQUARE):
            raise ValueError('invalid aspect ratio')
        self.aspectRatio = aspectRatio

        wx.Printout.__init__(self, figTitle)
        # self.SetPPIPrinter(300, 300)

    def GetPageInfo(self):
        """
        Overrides wx.Printout.GetPageInfo() to provide the printing framework
        with the number of pages in this print job.
        """
        return 1, 1, 1, 1

    def OnPrintPage(self, pageNumber):
        """
        Overrides wx.Printout.OnPrintPage to render the matplotlib figure to
        a printing device context.
        """
        # % of printable area to use
        imgPercent = max(1, min(100, self.size))/100.0

        # ratio of the figure's width to its height
        if self.aspectRatio==self.ASPECT_RECTANGULAR:
            aspectRatio = 1.61803399
        elif self.aspectRatio==self.ASPECT_SQUARE:
            aspectRatio = 1.0
        else:
            raise ValueError('invalid aspect ratio')

        # Device context to draw the page
        dc = self.GetDC()

        # PPI_P: Pixels Per Inch of the Printer
        wPPI_P, hPPI_P = [float(x) for x in self.GetPPIPrinter()]
        PPI_P = (wPPI_P+hPPI_P)/2.0

        # PPI: Pixels Per Inch of the DC
        if self.IsPreview():
            wPPI, hPPI = [float(x) for x in self.GetPPIScreen()]
        else:
            wPPI, hPPI = wPPI_P, hPPI_P
        PPI = (wPPI+hPPI)/2.0

        # Pg_Px: Size of the page (pixels)
        wPg_Px, hPg_Px = [float(x) for x in self.GetPageSizePixels()]

        # Dev_Px: Size of the DC (pixels)
        wDev_Px, hDev_Px = [float(x) for x in self.GetDC().GetSize()]

        # Pg: Size of the page (inches)
        wPg = wPg_Px/PPI_P
        hPg = hPg_Px/PPI_P

        # minimum margins (inches)
        # TODO: make these arguments to __init__()
        wM = 0.75
        hM = 0.75

        # Area: printable area within the margins (inches)
        wArea = wPg-2*wM
        hArea = hPg-2*hM

        # Fig: printing size of the figure
        # hFig is at a maximum when wFig == wArea
        max_hFig = wArea/aspectRatio
        hFig = min(imgPercent*hArea, max_hFig)
        wFig = aspectRatio*hFig

        # scale factor = device size / page size (equals 1.0 for real printing)
        S = ((wDev_Px/PPI)/wPg+(hDev_Px/PPI)/hPg)/2.0

        # Fig_S: scaled printing size of the figure (inches)
        # M_S: scaled minimum margins (inches)
        wFig_S = S*wFig
        hFig_S = S*hFig
        wM_S = S*wM
        hM_S = S*hM

        # Fig_Dx: scaled printing size of the figure (device pixels)
        # M_Dx: scaled minimum margins (device pixels)
        wFig_Dx = int(S*PPI*wFig)
        hFig_Dx = int(S*PPI*hFig)
        wM_Dx = int(S*PPI*wM)
        hM_Dx = int(S*PPI*hM)

        image = self.render_figure_as_image(wFig, hFig, PPI)

        if self.IsPreview():
            image = image.Scale(wFig_Dx, hFig_Dx)
        self.GetDC().DrawBitmap(image.ConvertToBitmap(), wM_Dx, hM_Dx, False)

        return True

    def render_figure_as_image(self, wFig, hFig, dpi):
        """
        Renders a matplotlib figure using the Agg backend and stores the result
        in a C{wx.Image}.  The arguments C{wFig} and {hFig} are the width and
        height of the figure, and C{dpi} is the dots-per-inch to render at.
        """
        figure: matplotlib.figure.Figure = self.figure

        old_dpi = figure.dpi
        figure.dpi = dpi
        old_width = figure.get_figwidth()
        figure.set_figwidth(wFig)
        old_height = figure.get_figheight()
        figure.set_figwidth(hFig)

        wFig_Px = int(figure.bbox.width)
        hFig_Px = int(figure.bbox.height)

        agg = RendererAgg(wFig_Px, hFig_Px, dpi)

        figure.draw(agg)

        figure.dpi = old_dpi
        figure.set_figwidth(old_width)
        figure.set_figheight(old_height)

        image = wx.EmptyImage(wFig_Px, hFig_Px)
        image.SetData(agg.tostring_rgb())
        return image


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
        self.create_axes()
        self.update = self.plot_data
        self.SetAutoScale(True)
        # self.ax = self.figure.add_subplot(111)
        # self.ax.set_autoscale_on(False)

    def SetXScale(self, scalestring):
        ''' SetXScale(self, scalestring) --> None

        Sets the x-scale of the main plotting axes. Currently accepts
        'log' or 'lin'.
        '''
        if self.ax:
            if scalestring=='log':
                self.x_scale = 'log'
                self.AutoScale(force=True)
                try:
                    self.ax.set_xscale('log')
                    self.error_ax.set_xscale('log')
                except OverflowError:
                    self.AutoScale(force=True)
                except UserWarning:
                    pass
            elif scalestring=='linear' or scalestring=='lin':
                self.x_scale = 'linear'
                self.ax.set_xscale('linear')
                self.error_ax.set_xscale('linear')
                self.AutoScale(force=True)
            else:
                raise ValueError('Not allowed scaling')

            try:
                self.flush_plot()
            except UserWarning:
                pass

            self.WriteConfig()
            evt = state_changed(zoomstate=self.GetZoom(),
                                yscale=self.GetYScale(), autoscale=self.autoscale,
                                xscale=self.GetXScale())
            wx.PostEvent(self.callback_window, evt)

    def create_axes(self):
        # self.ax = self.figure.add_axes(self.main_ax_rect)#
        try:
            gs = self.figure.add_gridspec(4, 1)
        except AttributeError:
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(4, 1)
        self.ax = self.figure.add_subplot(gs[:3, 0])
        # self.ax.xaxis.set_visible(False)
        self.ax.get_xaxis().set_visible(False)
        # setp(self.ax.get_xticklabels(), visible=False)
        self.error_ax = self.figure.add_subplot(gs[3, 0], sharex=self.ax)
        # self.error_ax = self.figure.add_axes(self.sub_ax_rect, sharex=self.ax)
        self.ax.set_autoscale_on(False)
        self.error_ax.set_autoscale_on(True)
        self.ax.set_ylabel('y')
        self.error_ax.set_ylabel('FOM')
        self.error_ax.set_xlabel('x')

    def update_labels(self, xlabel=None, ylabel=None, elabel=None):
        if xlabel is not None:
            self.error_ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        if elabel is not None:
            self.error_ax.set_ylabel(elabel)
        self.flush_plot()

    def autoscale_error_ax(self):
        ymin = min([array(line.get_ydata()).min() \
                    for line in self.error_ax.lines if len(line.get_ydata())>0])
        ymax = max([array(line.get_ydata()).max() \
                    for line in self.error_ax.lines if len(line.get_ydata())>0])
        # Set the limits
        if ymin>=ymax:
            return
        self.error_ax.set_ylim(ymin*(1-sign(ymin)*0.05), ymax*(1+sign(ymax)*0.05))
        # self.ax.set_yscale(self.y_scale)

    def singleplot(self, data):
        if not self.ax:
            # self.ax = self.figure.add_subplot(111)
            self.create_axes()
        # theta = arange(0.1,10,0.001)
        # self.ax.plot(theta,1/sin(theta*pi/180)**4,'-')

    def plot_data(self, data):
        if not self.ax:
            # self.ax = self.figure.add_subplot(111)
            self.create_axes()

        # This will be somewhat inefficent since everything is updated
        # at once would be better to update the things that has changed...

        while len(self.ax.lines)>0:
            self.ax.lines[0].remove()
        while len(self.ax.collections)>0:
            self.ax.collections[0].remove()
        # plot the data
        # [self.ax.semilogy(data_set.x,data_set.y) for data_set in data]
        if self.y_scale=='linear':
            [self.ax.plot(data_set.x, data_set.y, color=data_set.data_color,
                          lw=data_set.data_linethickness, ls=data_set.data_linetype,
                          marker=data_set.data_symbol, ms=data_set.data_symbolsize, zorder=1) \
             for data_set in data if not data_set.use_error and data_set.show]
            # With errorbars
            [self.ax.errorbar(data_set.x, data_set.y,
                              yerr=c_[data_set.error*(data_set.error>0),
                                      data_set.error].transpose(),
                              color=data_set.data_color, lw=data_set.data_linethickness,
                              ls=data_set.data_linetype, marker=data_set.data_symbol,
                              ms=data_set.data_symbolsize, zorder=2) \
             for data_set in data if data_set.use_error and data_set.show]
        if self.y_scale=='log':
            [self.ax.plot(data_set.x.compress(data_set.y>0),
                          data_set.y.compress(data_set.y>0), color=data_set.data_color,
                          lw=data_set.data_linethickness, ls=data_set.data_linetype,
                          marker=data_set.data_symbol, ms=data_set.data_symbolsize, zorder=1) \
             for data_set in data if not data_set.use_error and data_set.show]
            # With errorbars
            [self.ax.errorbar(data_set.x.compress(data_set.y
                                                  -data_set.error>0),
                              data_set.y.compress(data_set.y-data_set.error>0),
                              yerr=c_[data_set.error*(data_set.error>0),
                                      data_set.error].transpose().compress(data_set.y-
                                                                           data_set.error>0),
                              color=data_set.data_color, lw=data_set.data_linethickness,
                              ls=data_set.data_linetype, marker=data_set.data_symbol,
                              ms=data_set.data_symbolsize, zorder=2) \
             for data_set in data if data_set.use_error and data_set.show]
        self.AutoScale()
        # Force an update of the plot
        self.flush_plot()

    def plot_data_fit(self, data):
        if not self.ax:
            # self.ax = self.figure.add_subplot(111)
            self.create_axes()

        shown_data = [data_set for data_set in data if data_set.show]
        if len(self.ax.lines)==(2*len(shown_data)):
            for i, data_set in enumerate(shown_data):
                self.ax.lines[i].set_data(data_set.x, data_set.y)
                self.ax.lines[i+len(shown_data)].set_data(data_set.x, data_set.y_sim)
                self.error_ax.lines[i].set_data(data_set.x, ma.fix_invalid(data_set.y_fom, fill_value=0))
        else:
            while len(self.ax.lines)>0:
                self.ax.lines[0].remove()
            while len(self.ax.collections)>0:
                self.ax.collections[0].remove()
            while len(self.error_ax.lines)>0:
                self.error_ax.lines[0].remove()
            while len(self.error_ax.collections)>0:
                self.error_ax.collections[0].remove()
            # plot the data
            # [self.ax.semilogy(data_set.x, data_set.y, '.'\
            # ,data_set.x, data_set.y_sim) for data_set in data]
            [self.ax.plot(data_set.x, data_set.y, color=data_set.data_color,
                          lw=data_set.data_linethickness, ls=data_set.data_linetype,
                          marker=data_set.data_symbol, ms=data_set.data_symbolsize,
                          zorder=1) \
             for data_set in shown_data]
            # The same thing for the simulation
            [self.ax.plot(data_set.x, data_set.y_sim, color=data_set.sim_color,
                          lw=data_set.sim_linethickness, ls=data_set.sim_linetype,
                          marker=data_set.sim_symbol, ms=data_set.sim_symbolsize, zorder=5) \
             for data_set in shown_data]
            # Plot the point by point error:
            [self.error_ax.plot(data_set.x, ma.fix_invalid(data_set.y_fom, fill_value=0), color=data_set.sim_color,
                                lw=data_set.sim_linethickness, ls=data_set.sim_linetype,
                                marker=data_set.sim_symbol, ms=data_set.sim_symbolsize, zorder=2) \
             for data_set in shown_data]
        # Force an update of the plot
        self.autoscale_error_ax()
        self.flush_plot()

    def plot_data_sim(self, data):
        if not self.ax:
            # self.ax = self.figure.add_subplot(111)
            self.create_axes()

        p_options = [self.y_scale]+[[data_set.data_color, data_set.data_linethickness,
                                     data_set.data_linetype, data_set.data_symbol,
                                     data_set.data_symbolsize,
                                     data_set.sim_color, data_set.sim_linethickness,
                                     data_set.sim_linetype, data_set.sim_symbol,
                                     data_set.sim_symbolsize
                                     ] for data_set in data]
        p_datasets = [data_set for data_set in data if data_set.show]
        pe_datasets = [data_set for data_set in data if data_set.use_error and data_set.show]
        s_datasets = [data_set for data_set in data if data_set.show and data_set.use and
                      data_set.x.shape==data_set.y_sim.shape]
        if self._last_poptions==p_options and \
                len(self.ax.lines)==(len(p_datasets)+len(s_datasets)) and \
                len(self.ax.collections)==len(pe_datasets):
            for i, data_set in enumerate(p_datasets):
                if self.y_scale=='linear':
                    self.ax.lines[i].set_data(data_set.x, data_set.y)
                elif data_set.use_error:
                    fltr = (data_set.y-data_set.error)>0
                    self.ax.lines[i].set_data(data_set.x.compress(fltr), data_set.y.compress(fltr))
                else:
                    fltr = data_set.y>0
                    self.ax.lines[i].set_data(data_set.x.compress(fltr), data_set.y.compress(fltr))
            for j, data_set in enumerate(s_datasets):
                self.ax.lines[len(p_datasets)+j].set_data(data_set.x, data_set.y_sim)
                self.error_ax.lines[j].set_data(data_set.x, ma.fix_invalid(data_set.y_fom, fill_value=0))
            for k, data_set in enumerate(pe_datasets):
                ybot = data_set.y-data_set.error
                ytop = data_set.y+data_set.error
                segment_data = hstack([data_set.x, ybot, data_set.x, ytop]).reshape(2, 2, -1).transpose(2, 0, 1)
                if self.y_scale=='log':
                    if data_set.use_error:
                        fltr = (data_set.y-data_set.error)>0
                    else:
                        fltr = data_set.y>0
                    segment_data = segment_data[fltr, :, :]
                self.ax.collections[k].set_segments(segment_data)
        else:
            while len(self.ax.lines)>0:
                self.ax.lines[0].remove()
            while len(self.ax.collections)>0:
                self.ax.collections[0].remove()
            while len(self.error_ax.lines)>0:
                self.error_ax.lines[0].remove()
            while len(self.error_ax.collections)>0:
                self.error_ax.collections[0].remove()
            # plot the data
            # [self.ax.semilogy(data_set.x, data_set.y, '.'\
            # ,data_set.x, data_set.y_sim) for data_set in data]
            # Plot the data sets and take care if it is log scaled data

            if self.y_scale=='linear':
                [self.ax.plot(data_set.x, data_set.y, color=data_set.data_color,
                              lw=data_set.data_linethickness, ls=data_set.data_linetype,
                              marker=data_set.data_symbol, ms=data_set.data_symbolsize, zorder=1) \
                 for data_set in p_datasets if not data_set.use_error]
                # With errorbars
                [self.ax.errorbar(data_set.x, data_set.y,
                                  yerr=c_[data_set.error*(data_set.error>0),
                                          data_set.error].transpose(),
                                  color=data_set.data_color, lw=data_set.data_linethickness,
                                  ls=data_set.data_linetype, marker=data_set.data_symbol,
                                  ms=data_set.data_symbolsize, zorder=2) \
                 for data_set in pe_datasets]
            if self.y_scale=='log':
                [self.ax.plot(data_set.x.compress(data_set.y>0),
                              data_set.y.compress(data_set.y>0), color=data_set.data_color,
                              lw=data_set.data_linethickness, ls=data_set.data_linetype,
                              marker=data_set.data_symbol, ms=data_set.data_symbolsize, zorder=1) \
                 for data_set in p_datasets if not data_set.use_error]
                # With errorbars
                [self.ax.errorbar(data_set.x.compress(data_set.y
                                                      -data_set.error>0),
                                  data_set.y.compress(data_set.y-data_set.error>0),
                                  yerr=c_[data_set.error*(data_set.error>0),
                                          data_set.error].transpose().compress(data_set.y-
                                                                               data_set.error>0),
                                  color=data_set.data_color, lw=data_set.data_linethickness,
                                  ls=data_set.data_linetype, marker=data_set.data_symbol,
                                  ms=data_set.data_symbolsize, zorder=2) \
                 for data_set in pe_datasets]
            # The same thing for the simulation
            [self.ax.plot(data_set.x, data_set.y_sim, color=data_set.sim_color,
                          lw=data_set.sim_linethickness, ls=data_set.sim_linetype,
                          marker=data_set.sim_symbol, ms=data_set.sim_symbolsize, zorder=5) \
             for data_set in s_datasets]
            [self.error_ax.plot(data_set.x, ma.fix_invalid(data_set.y_fom, fill_value=0), color=data_set.sim_color,
                                lw=data_set.sim_linethickness, ls=data_set.sim_linetype, marker=data_set.sim_symbol,
                                ms=data_set.sim_symbolsize) for data_set in s_datasets]
            self._last_poptions = p_options
        try:
            self.autoscale_error_ax()
        except ValueError:
            pass
        self.AutoScale()
        # Force an update of the plot
        self.flush_plot()

    @skips_event
    def OnDataListEvent(self, event):
        '''OnDataListEvent(self, event) --> None
        
        Event handler function for connection  to DataList events...
        i.e. update of the plots when the data has changed
        '''
        # print 'OnDataListEvent runs'
        # print event.data_changed, event.new_data
        data_list = event.GetData()
        if event.data_changed:
            if event.new_data:
                # print 'updating plot'
                self.update = self.plot_data
                self.update(data_list)
                tmp = self.GetAutoScale()
                self.SetAutoScale(True)
                self.AutoScale()
                self.SetAutoScale(tmp)
            else:
                self.update(data_list)
        else:
            # self.update(data_list)
            pass

    @skips_event
    def OnSimPlotEvent(self, event):
        '''OnSimPlotEvent(self, event) --> None
        
        Event handler funciton for connection to simulation events
        i.e. update the plot with the data + simulation
        '''
        model: Model = event.GetModel()
        data_list = model.get_data()
        self.update = self.plot_data_sim
        self.update(data_list)
        try:
            ylabel = model.eval_in_model('globals().get("__ylabel__", getattr(model, "__ylabel__", "y"))')
        except NameError:
            ylabel = model.eval_in_model('globals().get("__ylabel__", "y")')
        try:
            xlabel = model.eval_in_model('globals().get("__xlabel__", getattr(model, "__xlabel__", "x"))')
        except NameError:
            xlabel = model.eval_in_model('globals().get("__xlabel__", "x")')
        self.update_labels(xlabel, ylabel)

    @skips_event
    def OnSolverPlotEvent(self, event):
        ''' OnSolverPlotEvent(self,event) --> None
        
        Event handler function to connect to solver update events i.e.
        update the plot with the simulation
        '''
        if event.update_fit:
            if self.update!=self.plot_data_fit:
                self.update = self.plot_data_fit
                self.SetAutoScale(False)
            self.update(event.data)


class ErrorPanelConfig(BasePlotConfig):
    section = 'fom plot'


class ErrorPlotPanel(PlotPanel):
    ''' Class for plotting evolution of the error as a function of the
        generations.
    '''

    def __init__(self, parent, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, ErrorPanelConfig, style, **kwargs)
        self.update = self.errorplot
        self.update(None)

    def errorplot(self, data):
        if not self.ax:
            self.ax = self.figure.add_subplot(111)

        # self.ax.cla()
        self.ax.set_autoscale_on(False)

        while len(self.ax.lines)>0:
            self.ax.lines[0].remove()
        if data is None:
            theta = arange(0.1, 10, 0.001)
            self.ax.plot(theta, floor(15-theta), '-r')
        else:
            # print 'plotting ...', data
            self.ax.plot(data[:, 0], data[:, 1], '-r')
            if self.GetAutoScale() and len(data)>0:
                self.ax.set_ylim(data[:, 1].min()*0.95, data[:, 1].max()*1.05)
                self.ax.set_xlim(data[:, 0].min(), data[:, 0].max())
                # self.AutoScale()

        self.ax.set_xlabel('Iteration')
        self.ax.set_ylabel('FOM')
        try:
            self.figure.tight_layout(h_pad=0)
        except:
            pass
        self.flush_plot()

    @skips_event
    def OnSolverPlotEvent(self, event):
        ''' OnSolverPlotEvent(self,event) --> None
        Event handler function to connect to solver update events i.e.
        update the plot with the simulation
        '''
        fom_log = event.fom_log
        self.update(fom_log)


class ParsPanelConfig(BasePlotConfig):
    section = 'pars plot'


class ParsPlotPanel(PlotPanel):
    ''' ParsPlotPanel
    
    Class to plot the diffrent parametervalus during a fit.
    '''

    def __init__(self, parent, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, ParsPanelConfig, style, **kwargs)
        self.update(None)
        self.ax = self.figure.add_subplot(111)
        # self.ax.set_autoscale_on(True)
        self.update = self.Plot

    def Plot(self, data):
        ''' Plots each variable and its max and min value in the
        population.
        '''

        if data.fitting:
            pop = array(data.population)
            norm = 1.0/(data.max_val-data.min_val)
            best = (array(data.values)-data.min_val)*norm
            pop_min = (pop.min(0)-data.min_val)*norm
            pop_max = (pop.max(0)-data.min_val)*norm

            self.ax.cla()
            width = 0.8
            x = arange(len(best))
            self.ax.set_autoscale_on(False)
            self.ax.bar(x, pop_max-pop_min, bottom=pop_min, color='b', width=width)
            self.ax.plot(x, best, 'ro')
            if self.GetAutoScale():
                self.ax.axis([x.min()-width, x.max()+width, 0., 1.])

        self.ax.set_xlabel('Parameter Index (only fittable)')
        self.ax.set_ylabel('Relative value in min/max range')
        self.figure.tight_layout(h_pad=0)
        self.flush_plot()

    @skips_event
    def OnSolverParameterEvent(self, event):
        ''' OnSolverParameterEvent(self,event) --> None
        Event handler function to connect to solver update events i.e.
        update the plot during the fitting
        '''
        self.update(event)
        # Do not forget - pass the event on


class FomPanelConfig(BasePlotConfig):
    section = 'fom scan plot'


class FomScanPlotPanel(PlotPanel):
    '''FomScanPlotPanel
    
    Class to take care of fom scans.
    '''

    def __init__(self, parent, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, FomPanelConfig, style, **kwargs)
        self.update(None)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_autoscale_on(False)
        self.update = self.Plot

        self.type = 'project'

    def SetPlottype(self, type):
        '''SetScantype(self, type) --> None
        
        Sets the type of the scan type = "project" or "scan"
        '''
        if type.lower()=='project':
            self.type = 'project'
            self.SetAutoScale(False)
            # self.ax.set_autoscale_on(False)
        elif type.lower()=='scan':
            self.SetAutoScale(True)
            # self.ax.set_autoscale_on(True)
            self.type = 'scan'

    def Plot(self, data, l1='', l2=''):
        ''' Plots each variable and its max and min value in the
        population.
        '''
        self.ax.cla()
        x, y, bestx, besty, e_scale = data[0], data[1], data[2], data[3], \
                                      data[4]
        if self.type.lower()=='project':
            self.ax.set_autoscale_on(False)
            self.ax.plot(x, y, 'ob')
            self.ax.plot([bestx], [besty], 'or')
            self.ax.hlines(besty*e_scale, x.min(), x.max(), 'r')
            self.ax.axis([x.min(), x.max(), min(y.min(), besty)*0.95,
                          (besty*e_scale-min(y.min(), besty))*2.0+min(y.min(), besty)])
        elif self.type.lower()=='scan':
            self.ax.plot(x, y, 'b')
            self.ax.plot([bestx], [besty], 'or')
            self.ax.hlines(besty*e_scale, x.min(), x.max(), 'r')
            if self.GetAutoScale():
                self.ax.set_autoscale_on(False)
                self.ax.axis([x.min(), x.max(), min(y.min(), besty)*0.95, y.max()*1.05])

        self.ax.set_xlabel(l1)
        self.ax.set_ylabel(l2)

        self.flush_plot()
