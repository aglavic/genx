# File: plotpanel.py a extension of panel that implements matplotlib plotting
# libary.
# Programmed by: Matts Bjorck
# Last changed: 2008 09 03

import matplotlib
matplotlib.interactive(False)
# Use WXAgg backend Wx to slow
matplotlib.use('Agg')
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector

from numpy import *
import wx
import wx.lib.newevent

import io

# Event for a click inside an plot which yields a number
(plot_position, EVT_PLOT_POSITION) = wx.lib.newevent.NewEvent()
# Event to tell the main window that the zoom state has changed
(state_changed, EVT_PLOT_SETTINGS_CHANGE) = wx.lib.newevent.NewEvent()

#==============================================================================
class PlotPanel(wx.Panel):
    ''' Base class for the plotting in GenX - all the basic functionallity
        should be implemented in this class. The plots should be derived from
        this class. These classes should implement an update method to update
        the plots. 
    '''
    def __init__(self, parent, id = -1, color = None, dpi = None
            , style = wx.NO_FULL_REPAINT_ON_RESIZE|wx.EXPAND|wx.ALL
            , config = None, config_name = '', **kwargs):
        
        wx.Panel.__init__(self,parent, id = id, style = style, **kwargs)
        
        self.parent = parent
        self.callback_window = self
        self.config = config
        self.config_name = config_name
        self.figure = Figure(None,dpi)
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.canvas.SetExtraStyle(wx.EXPAND)
        self.SetColor(color)
        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)
        self._resizeflag = True
        self.print_size = (15./2.54, 12./2.54)
        #self._SetSize()
        
        # Flags and bindings for zooming
        self.zoom = False
        self.zooming = False
        self.scale = 'linear'
        self.autoscale = True
        
        
        self.canvas.Bind(wx.EVT_LEFT_DOWN, self.OnLeftMouseButtonDown)
        self.canvas.Bind(wx.EVT_LEFT_UP, self.OnLeftMouseButtonUp)
        self.canvas.Bind(wx.EVT_MOTION, self.OnMouseMove)
        self.canvas.Bind(wx.EVT_LEFT_DCLICK, self.OnLeftDblClick)
        self.canvas.Bind(wx.EVT_RIGHT_UP, self.OnContextMenu)
        
        cursor = wx.StockCursor(wx.CURSOR_CROSS)
        self.canvas.SetCursor(cursor)
        self.old_scale_state = True
        self.ax = None
        
    
    def SetColor(self, rgbtuple=None):
        ''' Set the figure and canvas color to be the same '''
        if not rgbtuple:
            rgbtuple = wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE).Get()
        col = [c/255. for c in rgbtuple]
        self.figure.set_facecolor(col)
        self.figure.set_edgecolor(col)
        self.canvas.SetBackgroundColour(wx.Colour(*rgbtuple))
        
    def _onSize(self, evt):
        self._resizeflag = True
        #self._SetSize()
        #self.canvas.draw(repaint = False)
        
    def _onIdle(self, evt):
        if self._resizeflag:
            self._resizeflag = False
            self._SetSize()
            #self.canvas.gui_repaint(drawDC = wx.PaintDC(self))

            
    def _SetSize(self, pixels = None):
        ''' This method can be called to force the Plot to be a desired 
            size which defaults to the ClientSize of the Panel.
        '''
        if not pixels:
            pixels = self.GetClientSize()

        self.canvas.SetSize(pixels)
        self.figure.set_size_inches(pixels[0]/self.figure.get_dpi()
        , pixels[1]/self.figure.get_dpi())
    
    def ReadConfig(self):
        '''ReadConfig(self) --> None
        
        Reads in the config file
        '''
        bool_items = ['zoom', 'autoscale']
        bool_func = [self.SetZoom, self.SetAutoScale]
        
        if not self.config:
            return
        
        
        vals = []
        for index in range(len(bool_items)):
            try:
                val = self.config.get_boolean(self.config_name,\
                        bool_items[index])
            except io.OptionError, e:
                print 'Could not locate option %s.%s'\
                %(self.config_name, bool_items[index])
                vals.append(None)
            else:
                vals.append(val)
                
        try:
            scale = self.config.get(self.config_name, 'y scale')
            string_sucess = True
        except io.OptionError, e:
            string_sucess = False
            print 'Could not locate option %s.%s'\
            %(self.config_name, 'scale')
        else:
            self.SetYScale(scale)
        
        # This is done due to that the zoom and autoscale has to read 
        # before any commands are issued in order not to overwrite 
        # the config
        [bool_func[i](vals[i]) for i in range(len(vals)) if vals[i]]
        
            
    def WriteConfig(self):
        '''WriteConfig(self) --> None
        
        Writes the current settings to the config file
        '''
        if self.config:
            self.config.set(self.config_name, 'zoom', self.GetZoom())
            self.config.set(self.config_name, 'autoscale', self.GetAutoScale())
            self.config.set(self.config_name, 'y scale', self.GetYScale())
    
    def SetZoom(self, active = False):
        '''
        set the zoomstate
        '''
        #if not self.zoom_sel:
            #self.zoom_sel = RectangleSelector(self.ax,\
            # self.box_select_callback, drawtype='box',useblit=False)
        #print help(self.zoom_sel.ignore)

        if active:
            #self.zoom_sel.ignore = lambda x: False
            self.zoom = True
            cursor = wx.StockCursor(wx.CURSOR_MAGNIFIER)
            self.canvas.SetCursor(cursor)
            if self.callback_window:
                evt = state_changed(zoomstate = True,\
                        yscale = self.GetYScale(), autoscale = self.autoscale)
                wx.PostEvent(self.callback_window, evt)
            if self.ax:
                #self.ax.set_autoscale_on(False)
                self.old_scale_state = self.GetAutoScale()
                self.SetAutoScale(False)
                 
        else:
            #self.zoom_sel.ignore = lambda x: True
            self.zoom = False
            cursor = wx.StockCursor(wx.CURSOR_CROSS)
            self.canvas.SetCursor(cursor)
            if self.callback_window:
                evt = state_changed(zoomstate = False,\
                    yscale = self.GetYScale(), autoscale = self.autoscale)
                wx.PostEvent(self.callback_window, evt)
            if self.ax:
                #self.ax.set_autoscale_on(self.autoscale)
                self.SetAutoScale(self.old_scale_state)
        self.WriteConfig()
        
    def GetZoom(self):
        '''GetZoom(self) --> state [bool]
        Returns the zoom state of the plot panel. 
        True = zoom active
        False = zoom inactive
        '''
        return self.zoom
    
    def SetAutoScale(self, state):
        '''SetAutoScale(self, state) --> None
        
        Sets autoscale of the main axes wheter or not it should autoscale
        when plotting
        '''
        #self.ax.set_autoscale_on(state)
        self.autoscale = state
        self.WriteConfig()
        evt = state_changed(zoomstate = self.GetZoom(),\
                        yscale = self.GetYScale(), autoscale = self.autoscale)
        wx.PostEvent(self.callback_window, evt)
        
            
    def GetAutoScale(self):
        '''GetAutoScale(self) --> state [bool]
        
        Returns the autoscale state, true if the plots is automatically
        scaled for each plot command.
        '''
        return self.autoscale
    
    def AutoScale(self, force = False):
        '''AutoScale(self) --> None
        
        A log safe way to autoscale the plots - the ordinary axis tight 
        does not work for negative log data. This works!
        '''
        if not (self.autoscale or force):
            return
        # If nothing is plotted no autoscale use defaults...
        if sum([len(line.get_ydata()) > 0 for line in self.ax.lines]) == 0:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(1e-3, 1.0)
            return
        if self.scale == 'log':
            #print 'log scaling'
            # Find the lowest possible value of all the y-values that are 
            #greater than zero. check so that y data contain data before min
            # is applied
            tmp = [line.get_ydata().compress(line.get_ydata() > 0.0).min()\
                   for line in self.ax.lines if array(line.get_ydata() > 0.0).sum() > 0]
            if len(tmp) > 0:
                ymin = min(tmp)
            else:
                ymin = 1e-3
            tmp = [line.get_ydata().compress(line.get_ydata() > 0.0).max()\
                   for line in self.ax.lines if array(line.get_ydata() > 0.0).sum() > 0]
            if len(tmp) > 0:
                ymax = max(tmp)
            else:
                ymax = 1
        else:
            ymin = min([array(line.get_ydata()).min()\
                     for line in self.ax.lines if len(line.get_ydata()) > 0])
            ymax = max([array(line.get_ydata()).max()\
                   for line in self.ax.lines if len(line.get_ydata()) > 0])
        tmp = [array(line.get_xdata()).min()\
                    for line in self.ax.lines if len(line.get_ydata()) > 0]
        if len(tmp) > 0:
            xmin = min(tmp)
        else:
            xmin = 0
        tmp = [array(line.get_xdata()).max()\
                    for line in self.ax.lines if len(line.get_ydata()) > 0]
        if len(tmp) > 0:
            xmax = max(tmp)
        else:
            xmax = 1
        # Set the limits
        #print 'Autoscaling to: ', ymin, ymax
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        #self.ax.set_yscale(self.scale)
        self.flush_plot()
        
    def SetYScale(self, scalestring):
        ''' SetYScale(self, scalestring) --> None
        
        Sets the y-scale of the main plotting axes. Currently accepts
        'log' or 'lin'.
        '''
        if self.ax:
            if scalestring == 'log':
                self.scale = 'log'
                self.AutoScale(force = True)
                try:
                    self.ax.set_yscale('log')
                except OverflowError:
                    self.AutoScale(force = True)
            elif scalestring == 'linear' or scalestring == 'lin':
                self.scale = 'linear'
                self.ax.set_yscale('linear')
                self.AutoScale(force = True)
            else:
                raise ValueError('Not allowed scaling')
            self.flush_plot()
            evt = state_changed(zoomstate = self.GetZoom(),\
                        yscale = self.GetYScale(), autoscale = self.autoscale)
            wx.PostEvent(self.callback_window, evt)
            self.WriteConfig()
            
    def GetYScale(self):
        '''GetYScale(self) --> String
        
        Returns the current y-scale in use. Currently the string
        'log' or 'linear'. If the axes does not exist it returns None.
        '''
        if self.ax:
            return self.ax.get_yscale()
        else:
            return None
        
    def CopyToClipboard(self):
        '''CopyToClipboard(self, event) --> None
        
        Copy the plot to the clipboard.
        '''
        self.canvas.Copy_to_Clipboard(event=None)
        
    def PrintSetup(self):
        '''PrintSetup(self) --> None
        
        Sets up the printer. Creates a dialog box
        '''
        self.canvas.Printer_Setup(event=None)

    def PrintPreview(self):
        '''PrintPreview(self) --> None
        
        Prints a preview on screen.
        '''
        #size = self.figure.get_size_inches()
        #self.figure.set_size_inches(self.print_size)
        self.canvas.Printer_Preview(event=None)
        #self.figure.set_size_inches(size)
        
    def Print(self):
        '''Print(self) --> None
        
        Print the figure.
        '''
        # TODO: Fix plot ptinting with a better printout class
        #size = self.figure.get_size_inches()
        #self.figure.set_size_inches(self.print_size)
        self.canvas.Printer_Print(event=None)
        #self.figure.set_size_inches(size)
        
    def SetCallbackWindow(self, window):
        '''SetCallbackWindow(self, window) --> None
        
        Sets teh callback window that should recieve the events from 
        picking.
        '''
        
        self.callback_window = window
    
    def OnLeftDblClick(self, event):
        if self.ax and self.zoom:
            tmp = self.GetAutoScale()
            self.SetAutoScale(True)
            self.AutoScale()
            self.SetAutoScale(tmp)
            #self.AutoScale()
            #self.flush_plot()
            #self.ax.set_autoscale_on(False)
    
    def OnLeftMouseButtonDown(self, event):
        self.start_pos = event.GetPositionTuple()
        #print 'Left Mouse button pressed ', self.ax.transData.inverse_xy_tup(self.start_pos)
        if self.zoom and self.ax:
            if self.ax.in_axes(*self.start_pos):
                self.zooming = True
                self.cur_rect = None
                self.canvas.CaptureMouse()
        elif self.ax:
            size = self.canvas.GetClientSize()
            x, y = self.ax.transData.inverse_xy_tup(\
                    (self.start_pos[0], size.height - self.start_pos[1]))
            if self.callback_window:
                evt = plot_position(text = '(%.3e, %.3e)'%(x, y))
                wx.PostEvent(self.callback_window, evt)
            #print x,y
        
    
    def OnMouseMove(self, event):
        if self.zooming and event.Dragging() and event.LeftIsDown():
            self.cur_pos = event.GetPositionTuple()
            #print 'Mouse Move ', self.ax.transData.inverse_xy_tup(self.cur_pos)
            if self.ax.in_axes(*self.cur_pos):
                new_rect = (self.start_pos[0], self.start_pos[1],\
                        self.cur_pos[0] - self.start_pos[0],\
                        self.cur_pos[1] - self.start_pos[1])
                self._DrawAndErase(new_rect, self.cur_rect)
                self.cur_rect = new_rect
        #event.Skip()
        
    def OnLeftMouseButtonUp(self, event):
        if self.canvas.HasCapture():
            #print 'Left Mouse button up'
            self.canvas.ReleaseMouse()
            if self.zooming and self.cur_rect:
                # Note: The coordinte system for matplotlib have a different 
                # direction of the y-axis and a different origin!
                size = self.canvas.GetClientSize()
                xstart, ystart = self.ax.transData.inverse_xy_tup(\
                    (self.start_pos[0], size.height-self.start_pos[1]))
                xend, yend = self.ax.transData.inverse_xy_tup(\
                    (self.cur_pos[0], size.height-self.cur_pos[1]))
                #print xstart, xend
                #print ystart, yend
                self.ax.set_xlim(min(xstart,xend), max(xstart,xend))
                self.ax.set_ylim(min(ystart,yend), max(ystart,yend))
                self.flush_plot()
            self.zooming = False
        
    def _DrawAndErase(self, box_to_draw, box_to_erase = None):
        '''_DrawAndErase(self, box_to_draw, box_to_erase = None) --> None
        '''
        dc = wx.ClientDC(self.canvas)
        dc.BeginDrawing()
        dc.SetPen(wx.Pen(wx.WHITE, 1, wx.DOT))
        dc.SetBrush(wx.TRANSPARENT_BRUSH)
        dc.SetLogicalFunction(wx.XOR)
        if box_to_erase:
            dc.DrawRectangle(*box_to_erase)
        dc.DrawRectangle(*box_to_draw)
        dc.EndDrawing()
        
    def OnContextMenu(self, event):
        '''OnContextMenu(self, event) --> None
        
        Callback to show the popmenu for the plot which allows various 
        settings to be made.
        '''
        menu = wx.Menu()
        
        zoomID = wx.NewId()
        menu.AppendCheckItem(zoomID, "Zoom")
        menu.Check(zoomID, self.GetZoom())
        def OnZoom(event):
            self.SetZoom(not self.GetZoom())
        self.Bind(wx.EVT_MENU, OnZoom, id = zoomID) 
        
        zoomallID = wx.NewId()
        menu.Append(zoomallID, 'Zoom All')
        def zoomall(event):
            tmp = self.GetAutoScale()
            self.SetAutoScale(True)
            self.AutoScale()
            self.SetAutoScale(tmp)
            #self.flush_plot()
        self.Bind(wx.EVT_MENU, zoomall, id = zoomallID)
        
        copyID = wx.NewId()
        menu.Append(copyID, "Copy")
        def copy(event):
            self.CopyToClipboard()
        self.Bind(wx.EVT_MENU, copy, id = copyID)
        
        yscalemenu = wx.Menu()
        logID = wx.NewId()
        linID = wx.NewId()
        yscalemenu.AppendRadioItem(logID, "log")
        yscalemenu.AppendRadioItem(linID, "linear")
        menu.AppendMenu(-1, "y-scale", yscalemenu)
        if self.GetYScale() == 'log':
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
        self.Bind(wx.EVT_MENU, yscale_log, id = logID)
        self.Bind(wx.EVT_MENU, yscale_lin, id = linID)
        
        autoscaleID = wx.NewId()
        menu.AppendCheckItem(autoscaleID, "Autoscale")
        menu.Check(autoscaleID, self.GetAutoScale())
        def OnAutoScale(event):
            self.SetAutoScale(not self.GetAutoScale())
        self.Bind(wx.EVT_MENU, OnAutoScale, id = autoscaleID) 
        
        # Time to show the menu
        self.PopupMenu(menu)
        
        menu.Destroy()
        
    def flush_plot(self):
        #self._SetSize()
        #self.canvas.gui_repaint(drawDC = wx.PaintDC(self))
        #self.ax.set_yscale(self.scale)
        self.canvas.draw()
        
    def update(self, data):
        pass

#==============================================================================
class DataPlotPanel(PlotPanel):
    ''' Class for plotting the data and the fit
    '''
    def __init__(self, parent, id = -1, color = None, dpi = None
    , style = wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, style, **kwargs)
        self.update=self.singleplot
        self.update(None)
        self.update = self.plot_data
        self.SetAutoScale(True)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_autoscale_on(False)
        
    def singleplot(self, data):
        if not self.ax:
                self.ax = self.figure.add_subplot(111)
        #theta = arange(0.1,10,0.001)
        #self.ax.plot(theta,1/sin(theta*pi/180)**4,'-')
    
    def plot_data(self, data):
        
        if not self.ax:
                self.ax = self.figure.add_subplot(111)
        
        # This will be somewhat inefficent since everything is updated
        # at once would be better to update the things that has changed...
        
        # Clear axes
        #self.ax.cla()
        self.ax.lines = []
        self.ax.collections = []
        # plot the data
        #[self.ax.semilogy(data_set.x,data_set.y) for data_set in data]
        if self.scale == 'linear':
            [self.ax.plot(data_set.x, data_set.y, c = data_set.data_color, \
                lw = data_set.data_linethickness, ls = data_set.data_linetype, \
                marker = data_set.data_symbol, ms = data_set.data_symbolsize)\
                for data_set in data if not data_set.use_error and data_set.show]
            # With errorbars
            [self.ax.errorbar(data_set.x, data_set.y,\
                yerr = c_[data_set.error*(data_set.error > 0),\
                 data_set.error].transpose(),\
                c = data_set.data_color, lw = data_set.data_linethickness,\
                ls = data_set.data_linetype, marker = data_set.data_symbol,\
                ms = data_set.data_symbolsize)\
             for data_set in data if data_set.use_error and data_set.show]
        if self.scale == 'log':
            [self.ax.plot(data_set.x.compress(data_set.y > 0),\
             data_set.y.compress(data_set.y > 0), c = data_set.data_color, \
                lw = data_set.data_linethickness, ls = data_set.data_linetype, \
                marker = data_set.data_symbol, ms = data_set.data_symbolsize)\
                for data_set in data if not data_set.use_error and data_set.show]
            # With errorbars
            [self.ax.errorbar(data_set.x.compress(data_set.y\
                    - data_set.error > 0),\
                data_set.y.compress(data_set.y -data_set.error > 0),\
                yerr = c_[data_set.error*(data_set.error > 0),\
                data_set.error].transpose().compress(data_set.y -\
                 data_set.error > 0),\
                c = data_set.data_color, lw = data_set.data_linethickness,\
                ls = data_set.data_linetype, marker = data_set.data_symbol,\
                ms = data_set.data_symbolsize)\
             for data_set in data if data_set.use_error and data_set.show]
        self.AutoScale()
        # Force an update of the plot
        self.flush_plot()
        #self.canvas.draw()
        #print 'Data plotted'
    
    def plot_data_fit(self, data):
        
        if not self.ax:
                self.ax = self.figure.add_subplot(111)
        
        # This will be somewhat inefficent since everything is updated
        # at once would be better to update the things that has changed...
        # Clear axes
        #self.ax.cla()
        self.ax.lines = []
        self.ax.collections = []
        #self.ax.set_title('FOM: None')
        # plot the data
        #[self.ax.semilogy(data_set.x, data_set.y, '.'\
        # ,data_set.x, data_set.y_sim) for data_set in data]
        [self.ax.plot(data_set.x, data_set.y, c = data_set.data_color, \
        lw = data_set.data_linethickness, ls = data_set.data_linetype, \
        marker = data_set.data_symbol, ms = data_set.data_symbolsize)\
         for data_set in data if data_set.show]
        # The same thing for the simulation
        [self.ax.plot(data_set.x, data_set.y_sim, c = data_set.sim_color, \
        lw = data_set.sim_linethickness, ls = data_set.sim_linetype, \
        marker = data_set.sim_symbol, ms = data_set.sim_symbolsize)\
         for data_set in data if data_set.show]
        # Force an update of the plot
        self.flush_plot()
        #self.canvas.draw()
        #print 'Data plotted'
        
    def plot_data_sim(self, data):
        
        if not self.ax:
                self.ax = self.figure.add_subplot(111)
        
        # This will be somewhat inefficent since everything is updated
        # at once would be better to update the things that has changed...
        # Clear axes
        #self.ax.cla()
        self.ax.lines = []
        self.ax.collections = []
        #self.ax.set_title('FOM: None')
        # plot the data
        #[self.ax.semilogy(data_set.x, data_set.y, '.'\
        # ,data_set.x, data_set.y_sim) for data_set in data]
        # Plot the data sets and take care if it is log scaled data
        if self.scale == 'linear':
            [self.ax.plot(data_set.x, data_set.y, c = data_set.data_color, \
                lw = data_set.data_linethickness, ls = data_set.data_linetype, \
                marker = data_set.data_symbol, ms = data_set.data_symbolsize)\
                for data_set in data if not data_set.use_error and data_set.show]
            # With errorbars
            [self.ax.errorbar(data_set.x, data_set.y,\
                yerr = c_[data_set.error*(data_set.error > 0),\
                 data_set.error].transpose(),\
                c = data_set.data_color, lw = data_set.data_linethickness,\
                ls = data_set.data_linetype, marker = data_set.data_symbol,\
                ms = data_set.data_symbolsize)\
             for data_set in data if data_set.use_error and data_set.show]
        if self.scale == 'log':
            [self.ax.plot(data_set.x.compress(data_set.y > 0),\
             data_set.y.compress(data_set.y > 0), c = data_set.data_color, \
                lw = data_set.data_linethickness, ls = data_set.data_linetype, \
                marker = data_set.data_symbol, ms = data_set.data_symbolsize)\
                for data_set in data if not data_set.use_error and data_set.show]
            # With errorbars
            [self.ax.errorbar(data_set.x.compress(data_set.y\
                    - data_set.error > 0),\
                data_set.y.compress(data_set.y -data_set.error > 0),\
                yerr = c_[data_set.error*(data_set.error > 0),\
                data_set.error].transpose().compress(data_set.y -\
                 data_set.error > 0),\
                c = data_set.data_color, lw = data_set.data_linethickness,\
                ls = data_set.data_linetype, marker = data_set.data_symbol,\
                ms = data_set.data_symbolsize)\
             for data_set in data if data_set.use_error and data_set.show]
        # The same thing for the simulation
        [self.ax.plot(data_set.x, data_set.y_sim, c = data_set.sim_color, \
            lw = data_set.sim_linethickness, ls = data_set.sim_linetype, \
            marker = data_set.sim_symbol, ms = data_set.sim_symbolsize)\
            for data_set in data if data_set.show and data_set.use]
        self.AutoScale()
        # Force an update of the plot
        self.flush_plot()
        #self.canvas.draw()
        #print 'Data plotted'
    
    
    def OnDataListEvent(self, event):
        '''OnDataListEvent(self, event) --> None
        
        Event handler function for connection  to DataList events...
        i.e. update of the plots when the data has changed
        '''
        #print 'OnDataListEvent runs'
        #print event.data_changed, event.new_data
        data_list = event.GetData()
        if event.data_changed:
            if event.new_data:
                #print 'updating plot'
                self.update = self.plot_data
                self.update(data_list)
                tmp = self.GetAutoScale()
                self.SetAutoScale(True)
                self.AutoScale()
                self.SetAutoScale(tmp)
            else:
                self.update(data_list)
        else:
            #self.update(data_list)
            pass
        event.Skip()
        
    def OnSimPlotEvent(self, event):
        '''OnSimPlotEvent(self, event) --> None
        
        Event handler funciton for connection to simulation events
        i.e. update the plot with the data + simulation
        '''
        data_list = event.GetModel().get_data()
        self.update = self.plot_data_sim
        self.update(data_list)
    
    def OnSolverPlotEvent(self, event):
        ''' OnSolverPlotEvent(self,event) --> None
        
        Event handler function to connect to solver update events i.e.
        update the plot with the simulation
        '''
        #print 'plotting'
        
        if event.update_fit:
            data_list = event.model.get_data()
            if self.update != self.plot_data_fit:
                self.update = self.plot_data_fit
                self.SetAutoScale(False)
            self.update(data_list)
        # Do not forget - pass the event on
        event.Skip()
    
#==============================================================================
class ErrorPlotPanel(PlotPanel):
    ''' Class for plotting evolution of the error as a function of the
        generations.
    '''
    def __init__(self, parent, id = -1, color = None, dpi = None
    , style = wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, style, **kwargs)
        self.update=self.errorplot
        self.update(None)
        
    def errorplot(self, data):
        if not self.ax:
            self.ax = self.figure.add_subplot(111)
            self.ax.set_autoscale_on(False)
            
        #self.ax.cla()
        
        self.ax.lines = []
        if data == None:
            theta = arange(0.1,10,0.001)
            self.ax.plot(theta,floor(15-theta),'-r')
        else:
            #print 'plotting ...', data
            self.ax.plot(data[:,0],data[:,1], '-r')
            self.ax.set_ylim(data[:,1].min()*0.95, data[:,1].max()*1.05)
            self.ax.set_xlim(data[:,0].min(), data[:,0].max())
            #self.AutoScale()
        
        self.flush_plot()
        self.canvas.draw()
        
    def OnSolverPlotEvent(self, event):
        ''' OnSolverPlotEvent(self,event) --> None
        Event handler function to connect to solver update events i.e.
        update the plot with the simulation
        '''
        #print 'error plot'
        fom_log = event.fom_log
        self.update(fom_log)
        # Do not forget - pass the event on
        event.Skip()

class ParsPlotPanel(PlotPanel):
    ''' ParsPlotPanel
    
    Class to plot the diffrent parametervalus during a fit.
    '''
    def __init__(self, parent, id = -1, color = None, dpi = None
    , style = wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, style, **kwargs)
        self.update(None)
        self.ax = self.figure.add_subplot(111)
        #self.ax.set_autoscale_on(True)
        self.update = self.Plot
    
    def Plot(self, data):
        ''' Plots each variable and its max and min value in the
        population.
        '''
        
        if data.fitting:
            pop = array(data.population)
            norm = 1.0/(data.max_val - data.min_val)
            best = (array(data.values) - data.min_val)*norm
            pop_min = (pop.min(0) - data.min_val)*norm
            pop_max = (pop.max(0) - data.min_val)*norm
            
            self.ax.cla()
            width = 0.8
            x = arange(len(best))
            self.ax.bar(x - width/2.0, pop_max - pop_min, bottom = pop_min,\
                        color = 'b', width = width)
            self.ax.plot(x, best, 'ro')
            self.ax.axis([x.min() - width, x.max() +    width,\
                          0., 1.])
        
        self.flush_plot()
        self.canvas.draw()
        
    def OnSolverParameterEvent(self, event):
        ''' OnSolverParameterEvent(self,event) --> None
        Event handler function to connect to solver update events i.e.
        update the plot during the fitting
        '''
        self.update(event)
        # Do not forget - pass the event on
        event.Skip()
        
class FomScanPlotPanel(PlotPanel):
    '''FomScanPlotPanel
    
    Class to take care of fom scans.
    '''
    def __init__(self, parent, id = -1, color = None, dpi = None
    , style = wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, style, **kwargs)
        self.update(None)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_autoscale_on(True)
        self.update = self.Plot
        
        self.type = 'project'
        
    def SetPlottype(self, type):
        '''SetScantype(self, type) --> None
        
        Sets the type of the scan type = "project" or "scan"
        '''
        if type.lower() == 'project':
            self.type = 'project'
        elif type.lower() == 'scan':
            self.type = 'scan'
        
    def Plot(self, data):
        ''' Plots each variable and its max and min value in the
        population.
        '''
        self.ax.cla()
        x, y = data
        if self.type.lower() == 'project':
            self.ax.plot(x, y, 'ob')
        elif self.type.lower() == 'scan':
            self.ax.plot(x, y, 'b')
        
        self.flush_plot()
        self.canvas.draw()

#=============================================================================
# Test code for the class to be able to independly test the code
if __name__ == '__main__':
    class DemoPlotPanel(PlotPanel):

        def draw(self):
            if not hasattr(self, 'subplot'):
                self.subplot = self.figure.add_subplot(111)
            theta = arange(0, 45*2*pi, 0.02)
            rad = (0.8*theta/2/pi+1)
            r = rad*(8 + sin(theta*7 + rad/1.8))
            x = r * cos(theta)
            y = r * sin(theta)
            
            self.subplot.plot(x,y, '-r')
            
            self.subplot.set_xlim([-400,400])
            self.subplot.set_ylim([-400,400])
            
    app = wx.PySimpleApp(0)
    frame = wx.Frame(None, -1, 'WxPython and Matplotlib')
    panel = DemoPlotPanel(frame)
    sizer = wx.BoxSizer(wx.HORIZONTAL)
    panel.SetSizer(sizer)
    sizer.SetItemMinSize(panel, 300, 300)
    panel.Fit()
    panel._SetSize()
    
    frame.Show()
    app.MainLoop()
            

