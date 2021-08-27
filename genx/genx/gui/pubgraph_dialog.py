import warnings
from dataclasses import dataclass
from logging import debug

import matplotlib
import matplotlib.axes
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure

from numpy import *
import wx
from wx.py.editwindow import EditWindow
from ..core.config import BaseConfig, Configurable
from ..data import DataList, DataSet

@dataclass
class PublicationConfig(BaseConfig):
    section = 'publication graph'

# ==============================================================================
class SimplePlotPanel(wx.Panel):
    ax: matplotlib.axes.Axes

    def __init__(self, parent, id=-1, color=None, dpi=None,
                 style=wx.NO_FULL_REPAINT_ON_RESIZE | wx.EXPAND | wx.ALL, **kwargs):
        wx.Panel.__init__(self, parent, id=id, style=style, **kwargs)
        if dpi is None:
            dpi=wx.GetApp().dpi_scale_factor*96.  # wx.GetDisplayPPI()[0]
        self.parent=parent
        debug('init PlotPanel - setup figure')
        self.figure=Figure(figsize=(1.0, 1.0), dpi=dpi)
        debug('init PlotPanel - setup canvas')
        self.canvas=FigureCanvasWxAgg(self, -1, self.figure)
        self.SetColor(color)
        self._resizeflag=True
        self.print_size=(15./2.54, 12./2.54)
        # self._SetSize()

        # debug('init PlotPanel - bind events')
        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

        cursor=wx.Cursor(wx.CURSOR_CROSS)
        self.canvas.SetCursor(cursor)
        self.ax=self.figure.add_subplot(111)

        # Create the drawing bitmap
        self.bitmap=wx.Bitmap(1, 1, depth=wx.BITMAP_SCREEN_DEPTH)
        #        DEBUG_MSG("__init__() - bitmap w:%d h:%d" % (w,h), 2, self)
        self._isDrawn=False
        debug('end init PlotPanel')

    def clear(self):
        self.figure.clear()
        self.ax=self.figure.add_subplot(111)

    def SetColor(self, rgbtuple=None):
        ''' Set the figure and canvas color to be the same '''
        if not rgbtuple:
            rgbtuple=self.parent.GetBackgroundColour()
            # wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE).Get()
        col=[c/255. for c in rgbtuple]
        self.figure.set_facecolor(col)
        self.figure.set_edgecolor(col)
        self.canvas.SetBackgroundColour(wx.Colour(*rgbtuple))

    def _onSize(self, evt):
        self._resizeflag=True
        self._SetSize()

    def _onIdle(self, evt):
        if self._resizeflag:
            self._resizeflag=False
            self._SetSize()

    def _SetSize(self):
        ''' This method can be called to force the Plot to be a desired
            size which defaults to the ClientSize of the Panel.
        '''
        pixels=self.GetMinSize()

        self.canvas.SetSize(pixels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.figure.tight_layout(h_pad=0)
            except ValueError:
                pass

    def flush_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.figure.tight_layout(h_pad=0)
        self.canvas.draw()

class PublicationDialog(wx.Dialog, Configurable):
    opt: PublicationConfig

    def __init__(self, parent, id=-1, data: DataList=None, text=''):
        wx.Dialog.__init__(self, parent, id=id,
                           style=wx.MAXIMIZE_BOX|wx.MINIMIZE_BOX|wx.RESIZE_BORDER|wx.DEFAULT_DIALOG_STYLE)
        Configurable.__init__(self)
        self.data=data
        self.dpi_scale_factor=parent.dpi_scale_factor
        hbox=wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(hbox)
        vbox=wx.BoxSizer(wx.VERTICAL)
        hbox.Add(vbox, 1, wx.EXPAND|wx.ALL, 4)
        self.tinput=EditWindow(self, wx.ID_ANY)
        self.tinput.SetBackSpaceUnIndents(True)
        self.tinput.SetText(text)

        vbox.Add(self.tinput, 1, wx.EXPAND, 0)
        button=wx.Button(self, label='plot')
        vbox.Add(button, 0, wx.EXPAND, 0)

        vbox2=wx.BoxSizer(wx.VERTICAL)
        hbox.Add(vbox2, 1, wx.EXPAND|wx.ALL, 4)
        self.plot=SimplePlotPanel(self)
        self.plot.SetMinSize(wx.Size(300,300))
        vbox2.Add(self.plot, 1, wx.EXPAND, 0)
        hbox2=wx.BoxSizer(wx.HORIZONTAL)
        vbox2.Add(hbox2, 0, wx.EXPAND, 0)
        self.width=wx.SpinCtrlDouble(self, min=0.1, max=30.0, initial=3.39, inc=0.01, style=wx.SP_ARROW_KEYS)
        self.height=wx.SpinCtrlDouble(self, min=0.1, max=30.0, initial=2.36, inc=0.01, style=wx.SP_ARROW_KEYS)
        hbox2.Add(wx.StaticText(self, label='Width (inches)'))
        hbox2.Add(self.width, 0, wx.FIXED_MINSIZE|wx.RIGHT, 10)
        hbox2.Add(wx.StaticText(self, label='Height (inches)'))
        hbox2.Add(self.height, 0, wx.FIXED_MINSIZE|wx.RIGHT, 10)
        self.font=wx.FontPickerCtrl(self, style=wx.FNTP_FONTDESC_AS_LABEL)
        self.font.SetSelectedFont(wx.Font(int(matplotlib.rcParams['font.size']),
                                          wx.FONTFAMILY_DEFAULT,
                                          wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False,
                                          faceName=matplotlib.rcParams['font.sans-serif'][0]))
        hbox2.Add(wx.StaticText(self, label='Font'))
        hbox2.Add(self.font, 0, wx.EXPAND, 0)

        self.Bind(wx.EVT_BUTTON, self.OnPlot, button)
        self.Bind(wx.EVT_COMMAND_ENTER, self.OnPlot)
        self.Bind(wx.EVT_SPINCTRLDOUBLE, self.OnPlot)
        self.SetSize(wx.Size(800,800))

    def OnPlot(self, event):
        txt=self.tinput.GetText()
        font=self.font.GetSelectedFont()
        weights={wx.FONTWEIGHT_BOLD: 'bold', wx.FONTWEIGHT_NORMAL: 'normal'}
        env=dict(
            rcParams=matplotlib.rcParams,
            data=self.data,
            show=self.plot.flush_plot,
            fig=self.plot.figure,
            canvas=self.plot.canvas,
            savefig=self.plot.figure.savefig,

            font_size=font.GetPointSize(),
            font_face=font.GetFaceName(),
            font_weight=weights[font.GetWeight()],
            )
        def clear():
            self.plot.clear()
            ax=self.plot.ax
            env['plot']=ax.plot,
            env['semilogy']=ax.semilogy
            env['errorbar']=ax.errorbar

            env['xlabel'] = ax.set_xlabel
            env['ylabel'] = ax.set_ylabel
            env['legend'] = ax.legend
            env['axes'] = ax
        env['clear']=clear

        w=int(self.width.GetValue()*72*self.dpi_scale_factor)
        h=int(self.height.GetValue()*72*self.dpi_scale_factor)
        self.plot.figure.dpi=int(72*self.dpi_scale_factor)
        self.plot.SetMinSize(wx.Size(w,h))
        self.plot.SetSize(wx.Size(w, h))
        clear()
        exec(txt, env)

        # make sure we reset the default rc values
        matplotlib.rcdefaults()
        self.Layout()