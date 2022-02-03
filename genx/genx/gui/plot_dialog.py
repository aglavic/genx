"""
Simple dialog window to show a plot graph.
"""

import wx

from .plotpanel import PlotPanel, BasePlotConfig


class DialogPlotConfig(BasePlotConfig):
    section = 'dialog plot'


class DialogPlot(PlotPanel):

    def __init__(self, parent, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, DialogPlotConfig, style, **kwargs)
        self.update(None)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_autoscale_on(False)


class PlotDialog(wx.Dialog):

    def __init__(self, parent, title='GenX Plot'):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        self.SetTitle(title)

        self._plot_dialog = DialogPlot(self)
        vbox.Add(self._plot_dialog, proportion=1, flag=wx.EXPAND)
        self.SetMinSize(wx.Size(200,200))

    @property
    def plot(self):
        return self._plot_dialog.ax.plot

    @property
    def semilogy(self):
        return self._plot_dialog.ax.semilogy

    def draw(self):
        self._plot_dialog.AutoScale()
        return self._plot_dialog.flush_plot()

    @property
    def legend(self):
        return self._plot_dialog.ax.legend

    def clear(self):
        return self._plot_dialog.ax.clear()

    def clear_data(self):
        ax = self._plot_dialog.ax
        while len(ax.lines)>0:
            ax.lines[0].remove()
        while len(ax.collections)>0:
            ax.collections[0].remove()

    @property
    def set_xlabel(self):
        return self._plot_dialog.ax.set_xlabel

    @property
    def set_ylabel(self):
        return self._plot_dialog.ax.set_ylabel

    @property
    def set_xlim(self):
        return self._plot_dialog.ax.set_xlim

    @property
    def set_ylim(self):
        return self._plot_dialog.ax.set_ylim
