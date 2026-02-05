"""
=================
FrequencyAnalysis
=================

A plugin to show reflectivity frequency components by using FFT/CWT of reflectivity corrected by critical q-values.

Qt port.
"""

from PySide6 import QtCore, QtWidgets

from numpy import isnan, linspace

from genx.gui_qt.plotpanel import BasePlotConfig, PlotPanel
from genx.plugins import add_on_framework as framework
from genx.tools.frequency_analysis import TransformType, transform


class FAPlotConfig(BasePlotConfig):
    section = "frequency analysis plot"


class FAPlotPanel(QtWidgets.QWidget):
    """Widget for plotting the frequency analysis of datasets."""

    def __init__(self, parent, plugin, color=None, dpi=None, **kwargs):
        super().__init__(parent)
        self.plot = PlotPanel(self, color=color, dpi=dpi, config_class=FAPlotConfig, **kwargs)
        self.plugin = plugin

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot, 1)

        controls = QtWidgets.QHBoxLayout()
        layout.addLayout(controls, 0)

        left_col = QtWidgets.QVBoxLayout()
        center_col = QtWidgets.QVBoxLayout()
        right_col = QtWidgets.QVBoxLayout()
        right_labels = QtWidgets.QVBoxLayout()
        controls.addLayout(left_col, 1)
        controls.addLayout(center_col, 1)
        controls.addLayout(right_col, 1)
        controls.addLayout(right_labels, 1)

        self.plot.update(None)
        self.plot.ax = self.plot.figure.add_subplot(111)
        box = self.plot.ax.get_position()
        self.plot.ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        self.plot.ax.set_autoscale_on(True)
        self.plot.update = self.Plot
        self.plot_dict = {}

        self.transform_log = QtWidgets.QCheckBox("log(R)", self)
        left_col.addWidget(self.transform_log)
        self.transform_Q4 = QtWidgets.QCheckBox("R Q^4", self)
        self.transform_Q4.setChecked(True)
        left_col.addWidget(self.transform_Q4)
        self.use_derivative = QtWidgets.QCheckBox("dR/dQ", self)
        left_col.addWidget(self.use_derivative)

        self.tt = {}
        for tt in list(TransformType):
            rb = QtWidgets.QRadioButton(tt.name, self)
            self.tt[tt] = rb
            center_col.addWidget(rb)
            rb.toggled.connect(self.Plot)
        self.tt[TransformType.fourier_transform].setChecked(True)

        self.Qc = QtWidgets.QDoubleSpinBox(self)
        self.Qc.setDecimals(5)
        self.Qc.setRange(0.0, 1.5)
        self.Qc.setSingleStep(0.001)
        self.Qc.setValue(0.05)
        right_labels.addWidget(QtWidgets.QLabel("Qc", self))
        right_col.addWidget(self.Qc)

        self.WLscale = QtWidgets.QDoubleSpinBox(self)
        self.WLscale.setDecimals(3)
        self.WLscale.setRange(0.0, 1.0)
        self.WLscale.setSingleStep(0.01)
        self.WLscale.setValue(0.5)
        right_labels.addWidget(QtWidgets.QLabel("WL-scale", self))
        right_col.addWidget(self.WLscale)

        self.transform_log.toggled.connect(self.Plot)
        self.transform_Q4.toggled.connect(self.Plot)
        self.use_derivative.toggled.connect(self.Plot)
        self.Qc.valueChanged.connect(self.Plot)
        self.WLscale.valueChanged.connect(self.Plot)

    def SetZoom(self, active=False):
        return self.plot.SetZoom(active)

    def GetZoom(self):
        return self.plot.GetZoom()

    def Plot(self, _event=None):
        """Plot the frequency analysis."""
        while len(self.plot.ax.lines) > 0:
            self.plot.ax.lines[0].remove()

        data = self.plugin.GetModel().get_data()
        d = linspace(10.0, 2500.0, 250)

        options = dict(
            Qc=float(self.Qc.value()),
            logI=self.transform_log.isChecked(),
            Q4=self.transform_Q4.isChecked(),
            derivate=self.use_derivative.isChecked(),
            wavelet_scaling=self.WLscale.value(),
        )
        for tt, ctrl in self.tt.items():
            if ctrl.isChecked():
                options["trans_type"] = tt
                break

        for i, di in enumerate(data):
            if i % 2 != 0:
                continue
            if di.show:
                try:
                    if not isnan(di.y).all() and (di.y > 0).any():
                        _, mag = transform(di.x, di.y, derivN=3, Qmin=0.08, Qmax=None, D=d, **options)
                        self.plot.ax.plot(
                            d,
                            mag,
                            color=di.data_color,
                            ls=di.data_linetype,
                            lw=di.data_linethickness,
                            marker=di.data_symbol,
                            ms=di.data_symbolsize,
                            label="data " + di.name,
                        )

                    if di.y_sim is not None and not isnan(di.y_sim).all() and (di.y_sim > 0).any():
                        _, mag = transform(di.x, di.y_sim, derivN=3, Qmin=0.08, Qmax=None, D=d, **options)
                        self.plot.ax.plot(
                            d,
                            mag,
                            color=di.sim_color,
                            ls=di.sim_linetype,
                            lw=di.sim_linethickness,
                            marker=di.sim_symbol,
                            ms=di.sim_symbolsize,
                            label="sim " + di.name,
                        )
                except KeyError:
                    pass

        self.plot.ax.legend(loc="upper right", framealpha=0.5, fontsize="small", ncol=1)
        self.plot.ax.yaxis.label.set_text("magnitude [a.u.]")
        self.plot.ax.xaxis.label.set_text("d [Å]")
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

        panel = self.NewPlotFolder("Frequency Analysis")
        layout = QtWidgets.QHBoxLayout(panel)
        self.FA_plot = FAPlotPanel(panel, self)
        layout.addWidget(self.FA_plot, 1)

    def OnSimulate(self, _event):
        QtCore.QTimer.singleShot(0, self.FA_plot.Plot)

    def OnFittingUpdate(self, _event):
        QtCore.QTimer.singleShot(0, self.FA_plot.Plot)
