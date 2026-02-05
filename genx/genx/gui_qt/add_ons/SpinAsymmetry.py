"""
=============
SpinAsymmetry
=============

A plugin to show an additional graph to plot the spin-asymmetry of polarized measurements.

Qt port.
"""

from PySide6 import QtCore, QtWidgets

from numpy import isnan

from genx.gui_qt.plotpanel import BasePlotConfig, PlotPanel
from genx.plugins import add_on_framework as framework


class SAPlotConfig(BasePlotConfig):
    section = "spin asymmetry plot"


class SAPlotPanel(QtWidgets.QWidget):
    """Widget for plotting the spin-asymmetry of datasets."""

    def __init__(self, parent, plugin, color=None, dpi=None, **kwargs):
        super().__init__(parent)
        self.plot = PlotPanel(self, color=color, dpi=dpi, config_class=SAPlotConfig, **kwargs)
        self.plugin = plugin

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 1)

        self.plot.update(None)
        self.plot.ax = self.plot.figure.add_subplot(111)
        box = self.plot.ax.get_position()
        self.plot.ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        self.plot.ax.set_autoscale_on(True)
        self.plot.update = self.Plot
        self.plot_dict = {}

    def SetZoom(self, active=False):
        return self.plot.SetZoom(active)

    def GetZoom(self):
        return self.plot.GetZoom()

    def Plot(self):
        while len(self.plot.ax.lines) > 0:
            self.plot.ax.lines[0].remove()

        data = self.plugin.GetModel().get_data()

        for i, di in enumerate(data):
            if i % 2 != 0:
                continue
            if di.show:
                try:
                    dj = data[i + 1]
                    li = len(di.x)
                    lj = len(dj.x)
                    length = min(li, lj)
                    SAdata = (di.y[:length] - dj.y[:length]) / (di.y[:length] + dj.y[:length])
                    if not isnan(SAdata).all():
                        self.plot.ax.plot(
                            di.x[:length],
                            SAdata,
                            color=di.data_color,
                            ls=di.data_linetype,
                            lw=di.data_linethickness,
                            marker=di.data_symbol,
                            ms=di.data_symbolsize,
                            label="data " + di.name + "&" + dj.name,
                        )
                    SAsim = (di.y_sim[:length] - dj.y_sim[:length]) / (di.y_sim[:length] + dj.y_sim[:length])
                    if not isnan(SAsim).all():
                        self.plot.ax.plot(
                            di.x[:length],
                            SAsim,
                            color=di.sim_color,
                            ls=di.sim_linetype,
                            lw=di.sim_linethickness,
                            marker=di.sim_symbol,
                            ms=di.sim_symbolsize,
                            label="sim " + di.name + "&" + dj.name,
                        )
                except Exception:
                    pass

        self.plot.ax.legend(loc="upper right", framealpha=0.5, fontsize="small", ncol=1)
        self.plot.ax.yaxis.label.set_text("Spin Asymmetry")
        self.plot.ax.xaxis.label.set_text("x")
        QtCore.QTimer.singleShot(0, self.plot.flush_plot)
        self.plot.AutoScale()

    def ReadConfig(self):
        return self.plot.ReadConfig()

    def GetYScale(self):
        return self.plot.ax.get_yscale() if self.plot.ax else None

    def GetXScale(self):
        return self.plot.ax.get_xscale() if self.plot.ax else None


class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent

        panel = self.NewPlotFolder("Spin-Asymmetry")
        layout = QtWidgets.QHBoxLayout(panel)
        self.SA_plot = SAPlotPanel(panel, self)
        layout.addWidget(self.SA_plot, 1)

    def OnSimulate(self, _event):
        QtCore.QTimer.singleShot(0, self.SA_plot.Plot)

    def OnFittingUpdate(self, _event):
        QtCore.QTimer.singleShot(0, self.SA_plot.Plot)
