import warnings

from dataclasses import dataclass
from logging import debug

import matplotlib
import matplotlib.axes
import wx
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
from matplotlib.figure import Figure
from numpy import (arange, arccos, arcsin, arctan, array, cos, exp, hstack, linspace, log, log10, pi, sin, sqrt, tan,
                   vstack)
from wx.py.editwindow import EditWindow

from ..core.config import BaseConfig, Configurable
from ..data import DataList, DataSet


# ==============================================================================
class SimplePlotPanel(wx.Panel):
    ax: matplotlib.axes.Axes

    def __init__(
        self, parent, id=-1, color=None, dpi=None, style=wx.NO_FULL_REPAINT_ON_RESIZE | wx.EXPAND | wx.ALL, **kwargs
    ):
        wx.Panel.__init__(self, parent, id=id, style=style, **kwargs)
        if dpi is None:
            dpi = wx.GetApp().dpi_scale_factor * 96.0  # wx.GetDisplayPPI()[0]
        self.parent = parent
        debug("init PlotPanel - setup figure")
        self.figure = Figure(figsize=(1.0, 1.0), dpi=dpi)
        debug("init PlotPanel - setup canvas")
        self.canvas = FigureCanvasWxAgg(self, -1, self.figure)
        self.SetColor(color)
        self._resizeflag = True
        self.print_size = (15.0 / 2.54, 12.0 / 2.54)
        # self._SetSize()

        # debug('init PlotPanel - bind events')
        self.Bind(wx.EVT_IDLE, self._onIdle)
        self.Bind(wx.EVT_SIZE, self._onSize)

        cursor = wx.Cursor(wx.CURSOR_CROSS)
        self.canvas.SetCursor(cursor)
        self.ax = self.figure.add_subplot(111)

        # Create the drawing bitmap
        self.bitmap = wx.Bitmap(1, 1, depth=wx.BITMAP_SCREEN_DEPTH)
        #        DEBUG_MSG("__init__() - bitmap w:%d h:%d" % (w,h), 2, self)
        self._isDrawn = False
        debug("end init PlotPanel")

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def SetColor(self, rgbtuple=None):
        """Set the figure and canvas color to be the same"""
        if not rgbtuple:
            rgbtuple = self.parent.GetBackgroundColour()
            # wx.SystemSettings.GetColour(wx.SYS_COLOUR_BTNFACE).Get()
        col = [c / 255.0 for c in rgbtuple]
        self.figure.set_facecolor(col)
        self.figure.set_edgecolor(col)
        self.canvas.SetBackgroundColour(wx.Colour(*rgbtuple))

    def _onSize(self, evt):
        self._resizeflag = True
        self._SetSize()

    def _onIdle(self, evt):
        if self._resizeflag:
            self._resizeflag = False
            self._SetSize()

    def _SetSize(self):
        """This method can be called to force the Plot to be a desired
        size which defaults to the ClientSize of the Panel.
        """
        pixels = self.GetMinSize()

        self.canvas.SetSize(pixels)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.figure.tight_layout(h_pad=0)
            except ValueError:
                pass

    def flush_plot(self):
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore", UserWarning)
        #     self.figure.tight_layout(h_pad=0)
        self.canvas.draw()


@dataclass
class PublicationConfig(BaseConfig):
    section = "publication graph"

    width: float = 3.39
    heigth: float = 2.36
    font_size: int = int(matplotlib.rcParams["font.size"])
    font_familty: int = wx.FONTFAMILY_DEFAULT
    font_face: str = matplotlib.rcParams["font.sans-serif"][0]

    start_text: str = repr(
        """## Create plots using matplotlib related to your model.
## Uncomment relevant lines to save/show different graphs.
## For other matplotlib rc style options see 
## https://matplotlib.org/stable/tutorials/introductory/customizing.html
fig.set_facecolor('white')
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = font_size # dialog entry
rcParams['font.sans-serif'] = font_face # dialog entry
rcParams['font.weight'] = font_weight # dialog entry
rcParams['mathtext.fontset'] = 'dejavusans' # 'dejavuserif'|'cm'|'stix'|'stixsans'

#outimg = save_file_dialog('*.png') # select filename for export
############ Plot of Reflectivity curves ###############
clear()
for di in data:
    if not di.show:
        continue
    errorbar(di.x, di.y, yerr=di.error, label="data-"+di.name, **di.data_kwds, zorder=2)
    semilogy(di.x, di.y_sim, label="sim-"+di.name, **di.sim_kwds, zorder=5)
xlabel(model.__xlabel__)
ylabel(model.__ylabel__)
legend()
tight_layout(pad=0.5)
#savefig(outimg, dpi=300) # save to selected image file
show() # show in GUI graph

########### Plot of SLD profiles ##############
clear()
sld = SLD[0] # assuming all dataset represent the same model
for key, value in sld.items():
    if key in ['z', 'SLD unit', 'Mass Density']:
        continue
    plot(sld['z'], value, label=key)
xlabel("z [Å]")
ylabel(f"SLD [${sld['SLD unit']}$]")
legend()
tight_layout(pad=0.5)
#savefig(outimg[:-4]+'_sld.png', dpi=300) # save to image file with changed suffix
#show() # show in GUI graph

########### Plot of SLD profiles ##############
clear()
md = SLD[0]['Mass Density'] # assuming all dataset represent the same model
## replace standard line colors with colormap
## See https://matplotlib.org/stable/users/explain/colors/colormaps.html
axes.set_prop_cycle(color=cm.gist_rainbow(np.linspace(0,1,len(md.keys())-1)))
for key, value in md.items():
    if key in ['z', 'SLD unit']:
        continue
    plot(md['z'], value, label=key)
xlabel("z [Å]")
ylabel(f"density [${md['SLD unit']}$]")
legend()
tight_layout(pad=0.5)
#savefig(outimg[:-4]+'_dens.png', dpi=300) # save to image file with changed suffix
#show() # show in GUI graph

""".splitlines()
    )


class PublicationDialog(wx.Dialog, Configurable):
    opt: PublicationConfig

    def __init__(self, parent, id=-1, data: DataList = None, module=None, SLD: dict = None):
        wx.Dialog.__init__(
            self, parent, id=id, style=wx.MAXIMIZE_BOX | wx.MINIMIZE_BOX | wx.RESIZE_BORDER | wx.DEFAULT_DIALOG_STYLE
        )
        Configurable.__init__(self)
        self.ReadConfig()
        self.SetTitle("Custom plotting for Publication")

        self.data = data
        self.module = module
        self.SLD = SLD
        self.dpi_scale_factor = parent.dpi_scale_factor
        main_box = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(main_box)
        splitter = wx.SplitterWindow(self, wx.ID_ANY, style=wx.SP_3D | wx.SP_BORDER | wx.SP_LIVE_UPDATE)
        main_box.Add(splitter, 1, wx.EXPAND)

        left_panel = wx.Panel(splitter, wx.ID_ANY)
        vbox = wx.BoxSizer(wx.VERTICAL)
        left_panel.SetSizer(vbox)
        right_panel = wx.Panel(splitter, wx.ID_ANY)
        vbox2 = wx.BoxSizer(wx.VERTICAL)
        right_panel.SetSizer(vbox2)
        splitter.SplitVertically(left_panel, right_panel)
        splitter.SetSashPosition(550)
        splitter.SetSashGravity(0.66)

        # ------------ Left Panel items --------------
        self.tinput = EditWindow(left_panel, wx.ID_ANY)
        self.tinput.SetBackSpaceUnIndents(True)
        self.tinput.SetText("\n".join(eval(self.opt.start_text)))

        vbox.Add(self.tinput, 1, wx.EXPAND, 0)
        button = wx.Button(left_panel, label="plot")
        vbox.Add(button, 0, wx.EXPAND, 0)

        # ------------ Right Panel items --------------
        self.plot_area = wx.ScrolledWindow(right_panel)
        self.plot_area.SetScrollbars(1, 1, 1, 1)
        pavbox = wx.BoxSizer(wx.VERTICAL)
        self.plot_area.SetSizer(pavbox)
        self.plot_area.SetAutoLayout(True)
        vbox2.Add(self.plot_area, 1, wx.EXPAND, 0)
        self.plot = SimplePlotPanel(self.plot_area)
        self.plot.SetMinSize(wx.Size(300, 300))
        pavbox.Add(self.plot, 1, wx.EXPAND, 0)
        self.error_text = wx.StaticText(right_panel)
        self.error_text.SetBackgroundColour(wx.Colour(255, 150, 150))
        self.error_text.Hide()
        vbox2.Add(self.error_text, 0, wx.EXPAND, 0)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        vbox2.Add(hbox2, 0, wx.EXPAND, 0)
        self.width = wx.SpinCtrlDouble(
            right_panel, min=0.1, max=30.0, initial=self.opt.width, inc=0.01,
            style=wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER
        )
        self.height = wx.SpinCtrlDouble(
            right_panel, min=0.1, max=30.0, initial=self.opt.heigth, inc=0.01,
            style=wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER
        )
        hbox2.Add(wx.StaticText(right_panel, label="Width (inches)"))
        hbox2.Add(self.width, 0, wx.FIXED_MINSIZE | wx.RIGHT, 10)
        hbox2.Add(wx.StaticText(right_panel, label="Height (inches)"))
        hbox2.Add(self.height, 0, wx.FIXED_MINSIZE | wx.RIGHT, 10)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        vbox2.Add(hbox2, 0, wx.EXPAND, 0)
        self.font = wx.FontPickerCtrl(right_panel, style=wx.FNTP_FONTDESC_AS_LABEL)
        self.font.SetSelectedFont(
            wx.Font(
                self.opt.font_size,
                self.opt.font_familty,
                wx.FONTSTYLE_NORMAL,
                wx.FONTWEIGHT_NORMAL,
                False,
                faceName=self.opt.font_face,
            )
        )
        hbox2.Add(wx.StaticText(right_panel, label="Font"))
        hbox2.Add(self.font, 0, wx.EXPAND, 0)

        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        vbox2.Add(hbox2, 0, wx.EXPAND, 0)
        save_button = wx.Button(right_panel, label="store current script and settings as default")
        hbox2.Add(save_button, 0, wx.EXPAND, 0)

        self.Bind(wx.EVT_BUTTON, self.OnPlot, button)
        self.Bind(wx.EVT_BUTTON, self.OnStoreDefaults, save_button)
        self.Bind(wx.EVT_COMMAND_ENTER, self.OnPlot)
        self.Bind(wx.EVT_SPINCTRLDOUBLE, self.OnPlot)
        self.SetSize(wx.Size(800, 800))

    def SaveFromScript(self, wildcard="*.png"):
        # called from within the script to select file name to save to
        with wx.FileDialog(self, "Save Plot",
                           wildcard=f"Imag files ({wildcard})|{wildcard}",
                           style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as fileDialog:
            if fileDialog.ShowModal()==wx.ID_CANCEL:
                return  # the user changed their mind

            # save the current contents in the file
            return fileDialog.GetPath()

    def OnPlot(self, event):
        txt = self.tinput.GetText()
        font = self.font.GetSelectedFont()
        weights = {wx.FONTWEIGHT_BOLD: "bold", wx.FONTWEIGHT_NORMAL: "normal"}
        env = dict(
            rcParams=matplotlib.rcParams,
            data=self.data,
            model=self.module,
            SLD=self.SLD,
            show=self.plot.flush_plot,
            fig=self.plot.figure,
            canvas=self.plot.canvas,
            savefig=self.plot.figure.savefig,
            tight_layout=self.plot.figure.tight_layout,
            font_size=font.GetPointSize(),
            font_face=font.GetFaceName(),
            font_weight=weights[font.GetWeight()],
            save_file_dialog=self.SaveFromScript,
            np=np,
            cm=plt.cm,
        )

        def clear():
            self.plot.clear()
            ax = self.plot.ax
            env["plot"] = ax.plot
            env["semilogy"] = ax.semilogy
            env["errorbar"] = ax.errorbar
            env["title"] = ax.set_title

            env["xlabel"] = ax.set_xlabel
            env["ylabel"] = ax.set_ylabel
            env["legend"] = ax.legend
            env["axes"] = ax

        env["clear"] = clear

        w = int(self.width.GetValue() * 72 * self.dpi_scale_factor * 2)
        h = int(self.height.GetValue() * 72 * self.dpi_scale_factor * 2)
        self.plot.figure.dpi = int(72 * self.dpi_scale_factor * 2)
        size = wx.Size(w, h)
        self.plot.SetMinSize(size)
        self.plot.SetSize(size)
        self.plot_area.FitInside()
        clear()
        try:
            exec(txt, env)
        except Exception as e:
            self.error_text.SetLabel(e.__class__.__name__ + ": " + str(e))
            self.error_text.Show()
        else:
            self.error_text.Hide()

        # make sure we reset the default rc values
        matplotlib.rcdefaults()
        self.Layout()

    def OnStoreDefaults(self, event):
        self.opt.start_text = repr(self.tinput.GetText().splitlines())
        self.opt.width = self.width.GetValue()
        self.opt.heigth = self.height.GetValue()

        font = self.font.GetSelectedFont()
        self.opt.font_size = font.GetPointSize()
        self.opt.font_face = font.GetFaceName()
        self.opt.font_familty = font.GetFamily()
        self.WriteConfig(default=True)
