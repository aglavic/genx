"""
Qt port of the wx-based plotpanel module.
Implements matplotlib plotting panels using the Qt backend.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from logging import ERROR, debug, getLogger
from typing import Type

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from numpy import arange, array, c_, floor, hstack, isfinite, ma, sign

from PySide6 import QtCore, QtGui, QtPrintSupport, QtWidgets

from ..core.config import BaseConfig, Configurable
from ..data import DataList
from ..model import Model

# deactivate matplotlib logging that we are not interested in
getLogger("matplotlib.ticker").setLevel(ERROR)
getLogger("matplotlib.font_manager").setLevel(ERROR)


@dataclass
class BasePlotConfig(BaseConfig):
    zoom: bool = False
    autoscale: bool = True
    x_scale: str = "linear"
    y_scale: str = "linear"


class QtFigurePrinter:
    """
    Simplified Qt printing helper for matplotlib figures.
    Uses a QPrinter and renders the canvas as a pixmap.
    """

    def __init__(self, view: "PlotPanel") -> None:
        self.view = view
        self.printer = QtPrintSupport.QPrinter()
        self.printer.setPageOrientation(QtGui.QPageLayout.Orientation.Landscape)

    def page_setup(self) -> None:
        dlg = QtPrintSupport.QPageSetupDialog(self.printer, self.view)
        dlg.exec()

    def preview_figure(self) -> None:
        dlg = QtPrintSupport.QPrintPreviewDialog(self.printer, self.view)
        dlg.paintRequested.connect(self._paint_to_printer)
        dlg.exec()

    def print_figure(self) -> None:
        dlg = QtPrintSupport.QPrintDialog(self.printer, self.view)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self._paint_to_printer(self.printer)

    def _paint_to_printer(self, printer: QtPrintSupport.QPrinter) -> None:
        self.view.canvas.draw()
        pixmap = self.view.canvas.grab()
        painter = QtGui.QPainter(printer)
        try:
            page_rect = printer.pageRect(QtPrintSupport.QPrinter.Unit.DevicePixel)
            target = QtCore.QRect(0, 0, page_rect.width(), page_rect.height())
            painter.drawPixmap(target, pixmap)
        finally:
            painter.end()


class PlotPanel(Configurable, QtWidgets.QWidget):
    """
    Base class for the plotting in GenX - all the basic functionality
    should be implemented in this class. The plots should be derived from
    this class. These classes should implement an update method to update
    the plots.
    """

    plot_position = QtCore.Signal(str)
    state_changed = QtCore.Signal(bool, str, bool, str)  # zoomstate, yscale, autoscale, xscale

    opt: BasePlotConfig

    def __init__(
        self,
        parent: QtWidgets.QWidget,
        color=None,
        dpi=None,
        config_class: Type[BasePlotConfig] | None = None,
        **kwargs,
    ) -> None:
        debug("start init PlotPanel")
        QtWidgets.QWidget.__init__(self, parent, **kwargs)
        Configurable.__init__(self, config_class)
        if dpi is None:
            dpi = 96.0
        self.parent = parent
        self.callback_window = self
        debug("init PlotPanel - setup figure")
        self.figure = Figure(figsize=(1.0, 1.0), dpi=dpi)
        debug("init PlotPanel - setup canvas")
        self.canvas = FigureCanvasQTAgg(self.figure)
        debug("init PlotPanel - setup toolbar")
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        self.SetColor(color)
        self.print_size = (15.0 / 2.54, 12.0 / 2.54)

        # Flags and bindings for zooming
        self.opt.load_config()
        self.zoom = self.opt.zoom
        self.y_scale = self.opt.y_scale
        self.x_scale = self.opt.x_scale
        self.autoscale = self.opt.autoscale
        self.zooming = False

        debug("init PlotPanel - bind events")
        self.canvas.mpl_connect("button_press_event", self.OnMPLButton)
        self.canvas.mpl_connect("scroll_event", self.OnMouseScroll)

        self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
        self.old_scale_state = True
        self.ax = None

        debug("init PlotPanel - FigurePrinter and layout")
        sizer = QtWidgets.QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.canvas, 1)
        sizer.addWidget(self.toolbar, 0)

        # Init printout stuff
        self.fig_printer = QtFigurePrinter(self)
        debug("end init PlotPanel")

    def OnMPLButton(self, event):
        mode = getattr(self.toolbar, "mode", None)
        if mode not in (None, "", "None") and str(mode).lower() not in ("none", ""):
            return
        if event.button == 3:
            self.OnContextMenu(event)
        elif event.button == 2:
            pass
        elif event.button == 1:
            self.OnLeftMouseButtonDown(event)

    def SetColor(self, rgbtuple=None):
        """Set the figure and canvas color to be the same."""
        if not rgbtuple:
            if self.parent is not None:
                rgb = self.parent.palette().color(self.parent.backgroundRole())
                rgbtuple = (rgb.red(), rgb.green(), rgb.blue())
            else:
                rgbtuple = (240, 240, 240)
        col = [c / 255.0 for c in rgbtuple]
        self.figure.set_facecolor(col)
        self.figure.set_edgecolor(col)
        self.canvas.setStyleSheet(
            f"background-color: rgb({rgbtuple[0]}, {rgbtuple[1]}, {rgbtuple[2]});"
        )

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

    def SetZoom(self, active: bool = False):
        """
        set the zoomstate
        """
        if active:
            self.zoom = True
            self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ZoomInCursor))
            self.state_changed.emit(self.GetZoom(), self.GetYScale(), self.autoscale, self.GetXScale())
            if self.ax:
                self.old_scale_state = self.GetAutoScale()
                self.SetAutoScale(False)
        else:
            self.zoom = False
            self.canvas.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.CrossCursor))
            self.state_changed.emit(self.GetZoom(), self.GetYScale(), self.autoscale, self.GetXScale())
            if self.ax:
                self.SetAutoScale(self.old_scale_state)
        self.WriteConfig()

    def GetZoom(self):
        """
        Returns the zoom state of the plot panel.
        """
        return self.zoom

    def SetAutoScale(self, state: bool):
        """
        Sets autoscale of the main axes whether or not it should autoscale
        when plotting
        """
        self.autoscale = state
        self.WriteConfig()
        self.state_changed.emit(self.GetZoom(), self.GetYScale(), self.autoscale, self.GetXScale())

    def GetAutoScale(self):
        """
        Returns the autoscale state, true if the plots is automatically
        scaled for each plot command.
        """
        return self.autoscale

    def AutoScale(self, force: bool = False):
        """
        A log safe way to autoscale the plots - the ordinary axis tight
        does not work for negative log data. This works!
        """
        mode = getattr(self.toolbar, "mode", None)
        if not (mode in (None, "", "None") or force):
            return
        if sum([len(line.get_ydata()) > 0 for line in self.ax.lines]) == 0:
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(1e-3, 1.0)
            return

        if self.y_scale == "log":
            tmp = [
                line.get_ydata().compress(line.get_ydata() > 0.0).min()
                for line in self.ax.lines
                if array(line.get_ydata() > 0.0).sum() > 0
            ]
            ymin = min(tmp) if len(tmp) > 0 else 1e-3
            tmp = [
                line.get_ydata().compress(line.get_ydata() > 0.0).max()
                for line in self.ax.lines
                if array(line.get_ydata() > 0.0).sum() > 0
            ]
            ymax = max(tmp) if len(tmp) > 0 else 1
        else:
            ymin = min(
                [
                    array(line.get_ydata()).compress(isfinite(line.get_ydata())).min()
                    for line in self.ax.lines
                    if len(line.get_ydata()) > 0 and any(isfinite(line.get_ydata()))
                ]
            )
            ymax = max(
                [
                    array(line.get_ydata()).compress(isfinite(line.get_ydata())).max()
                    for line in self.ax.lines
                    if len(line.get_ydata()) > 0 and any(isfinite(line.get_ydata()))
                ]
            )
        tmp = [array(line.get_xdata()).min() for line in self.ax.lines if len(line.get_ydata()) > 0]
        xmin = min(tmp) if len(tmp) > 0 else 0
        tmp = [array(line.get_xdata()).max() for line in self.ax.lines if len(line.get_ydata()) > 0]
        xmax = max(tmp) if len(tmp) > 0 else 1
        try:
            if xmin != xmax:
                self.ax.set_xlim(xmin, xmax)
            if ymin != ymax:
                self.ax.set_ylim(ymin * (1 - sign(ymin) * 0.05), ymax * (1 + sign(ymax) * 0.05))
            self.flush_plot()
            self.toolbar.update()
        except UserWarning:
            pass

    def SetYScale(self, scalestring: str):
        """
        Sets the y-scale of the main plotting axes. Currently accepts
        'log' or 'lin'.
        """
        if not self.ax:
            return
        if scalestring == "log":
            self.y_scale = "log"
        elif scalestring in ["linear", "lin"]:
            self.y_scale = "linear"
        else:
            raise ValueError("Not allowed scaling")

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
        self.state_changed.emit(self.GetZoom(), self.GetYScale(), self.autoscale, self.GetXScale())

    def SetXScale(self, scalestring: str):
        """
        Sets the x-scale of the main plotting axes. Currently accepts
        'log' or 'lin'.
        """
        if not self.ax:
            return
        if scalestring == "log":
            self.x_scale = "log"
            self.AutoScale(force=True)
            try:
                self.ax.set_xscale("log")
            except OverflowError:
                self.AutoScale(force=True)
            except UserWarning:
                pass
        elif scalestring in ["linear", "lin"]:
            self.x_scale = "linear"
            self.ax.set_xscale("linear")
            self.AutoScale(force=True)
        else:
            raise ValueError("Not allowed scaling")
        try:
            self.flush_plot()
        except UserWarning:
            pass
        self.WriteConfig()
        self.state_changed.emit(self.GetZoom(), self.GetYScale(), self.autoscale, self.GetXScale())

    def GetYScale(self):
        """
        Returns the current y-scale in use. Currently the string
        'log' or 'linear'. If the axes does not exist it returns None.
        """
        if self.ax:
            return self.ax.get_yscale()
        return None

    def GetXScale(self):
        """
        Returns the current x-scale in use. Currently the string
        'log' or 'linear'. If the axes does not exist it returns None.
        """
        if self.ax:
            return self.ax.get_xscale()
        return None

    def CopyToClipboard(self):
        """
        Copy the plot to the clipboard.
        """
        self.SetColor((255, 255, 255))
        self.canvas.draw()
        pixmap = self.canvas.grab()
        QtWidgets.QApplication.clipboard().setPixmap(pixmap)
        self.SetColor()
        self.canvas.draw()

    def PrintSetup(self):
        """
        Sets up the printer. Creates a dialog box
        """
        self.fig_printer.page_setup()

    def PrintPreview(self):
        """
        Prints a preview on screen.
        """
        self.fig_printer.preview_figure()

    def Print(self):
        """
        Print the figure.
        """
        self.fig_printer.print_figure()

    def SetCallbackWindow(self, window: QtWidgets.QWidget):
        """
        Sets the callback window that should receive the events from picking.
        """
        self.callback_window = window

    def OnLeftDblClick(self, event):
        return

    def OnLeftMouseButtonDown(self, event):
        if event.dblclick:
            return self.OnLeftDblClick(event)
        if event.inaxes:
            self.plot_position.emit("(%.3e, %.3e)" % (event.xdata, event.ydata))

    def _has_modifier(self, event, name: str) -> bool:
        key = getattr(event, "key", None)
        if not key:
            return False
        parts = key.split("+")
        return name in parts

    def OnMouseScroll(self, event):
        self.SetAutoScale(False)
        rot = event.step
        if self._has_modifier(event, "control"):
            rot *= 0.1
        if self._has_modifier(event, "alt"):
            xmin, xmax = self.ax.get_xlim()
            xrange = xmax - xmin
            if self._has_modifier(event, "shift"):
                if self.x_scale == "log":
                    if rot > 0:
                        self.ax.set_xlim(xmin * (1 + 2.33333 * rot), xmax)
                    else:
                        self.ax.set_xlim(xmin / (1 - 2.33333 * rot), xmax)
                else:
                    self.ax.set_xlim(xmin + xrange * 0.2 * rot, xmax)
            else:
                if self.x_scale == "log":
                    if rot > 0:
                        self.ax.set_xlim(xmin, xmax * (1 + 2.33333 * rot))
                    else:
                        self.ax.set_xlim(xmin, xmax / (1 - 2.33333 * rot))
                else:
                    self.ax.set_xlim(xmin, xmax + xrange * 0.2 * rot)
        else:
            ymin, ymax = self.ax.get_ylim()
            yrange = ymax - ymin
            if self._has_modifier(event, "shift"):
                if self.y_scale == "log":
                    if rot > 0:
                        self.ax.set_ylim(ymin * (1 + 2.33333 * rot), ymax)
                    else:
                        self.ax.set_ylim(ymin / (1 - 2.33333 * rot), ymax)
                else:
                    self.ax.set_ylim(ymin + yrange * 0.2 * rot, ymax)
            else:
                if self.y_scale == "log":
                    if rot > 0:
                        self.ax.set_ylim(ymin, ymax * (1 + 2.33333 * rot))
                    else:
                        self.ax.set_ylim(ymin, ymax / (1 - 2.33333 * rot))
                else:
                    self.ax.set_ylim(ymin, ymax + yrange * 0.2 * rot)
        self.flush_plot()
        self.toolbar.push_current()

    def OnContextMenu(self, _event):
        """
        Callback to show the popmenu for the plot which allows various
        settings to be made.
        """
        menu = self.generate_context_menu()
        pos = QtGui.QCursor.pos()
        menu.exec(pos)

    def generate_context_menu(self):
        menu = QtWidgets.QMenu(self)
        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(self.CopyToClipboard)
        menu.addSeparator()

        ymenu = QtWidgets.QMenu("y-scale", menu)
        ygroup = QtGui.QActionGroup(ymenu)
        ylog = ymenu.addAction("log")
        ylog.setCheckable(True)
        ylin = ymenu.addAction("linear")
        ylin.setCheckable(True)
        ygroup.addAction(ylog)
        ygroup.addAction(ylin)
        if self.GetYScale() == "log":
            ylog.setChecked(True)
        else:
            ylin.setChecked(True)
        ylog.triggered.connect(lambda: self.SetYScale("log"))
        ylin.triggered.connect(lambda: self.SetYScale("linear"))
        menu.addMenu(ymenu)

        xmenu = QtWidgets.QMenu("x-scale", menu)
        xgroup = QtGui.QActionGroup(xmenu)
        xlog = xmenu.addAction("log")
        xlog.setCheckable(True)
        xlin = xmenu.addAction("linear")
        xlin.setCheckable(True)
        xgroup.addAction(xlog)
        xgroup.addAction(xlin)
        if self.GetXScale() == "log":
            xlog.setChecked(True)
        else:
            xlin.setChecked(True)
        xlog.triggered.connect(lambda: self.SetXScale("log"))
        xlin.triggered.connect(lambda: self.SetXScale("linear"))
        menu.addMenu(xmenu)
        return menu

    def flush_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            self.figure.tight_layout(h_pad=0)
        self.canvas.draw()

    def update(self, data):
        pass

    # Slots for event-driven updates
    @QtCore.Slot(object)
    def OnDataListEvent(self, event):
        data_list = getattr(event, "data", None)
        if data_list is None and hasattr(event, "GetData"):
            data_list = event.GetData()
        if data_list is None:
            return
        if event.data_changed:
            if event.new_data:
                self.update = self.plot_data
                self.update(data_list)
                tmp = self.GetAutoScale()
                self.SetAutoScale(True)
                self.AutoScale()
                self.SetAutoScale(tmp)
            else:
                self.update(data_list)

    @QtCore.Slot(object)
    def OnSimPlotEvent(self, event):
        model = getattr(event, "model", None)
        if model is None and hasattr(event, "GetModel"):
            model = event.GetModel()
        if model is None and isinstance(event, Model):
            model = event
        if model is None:
            return
        data_list = model.get_data()
        self.update = self.plot_data_sim
        self.update(data_list)
        try:
            ylabel = model.eval_in_model('globals().get(\"__ylabel__\", getattr(model, \"__ylabel__\", \"y\"))')
        except NameError:
            ylabel = model.eval_in_model('globals().get(\"__ylabel__\", \"y\")')
        try:
            xlabel = model.eval_in_model('globals().get(\"__xlabel__\", getattr(model, \"__xlabel__\", \"x\"))')
        except NameError:
            xlabel = model.eval_in_model('globals().get(\"__xlabel__\", \"x\")')
        self.update_labels(xlabel, ylabel)

    @QtCore.Slot(object)
    def OnSolverPlotEvent(self, event):
        if event.update_fit:
            if self.update != self.plot_data_fit:
                self.update = self.plot_data_fit
                self.SetAutoScale(False)
            self.update(event.data)
        elif event.desc == "Model loaded":
            if hasattr(self.toolbar, "set_message"):
                self.toolbar.set_message("")
            self.AutoScale()


class DataPanelConfig(BasePlotConfig):
    section = "data plot"


class DataPlotPanel(PlotPanel):
    """Class for plotting the data and the fit"""

    _last_poptions = None

    def __init__(self, parent, color=None, dpi=None, **kwargs):
        self.main_ax_rect = (0.125, 0.3, 0.8, 0.6)
        self.sub_ax_rect = (0.125, 0.1, 0.8, 0.18)
        super().__init__(parent, color=color, dpi=dpi, config_class=DataPanelConfig, **kwargs)
        self.create_axes()
        self.update = self.plot_data
        self.SetAutoScale(True)

    def SetXScale(self, scalestring: str):
        if self.ax:
            if scalestring == "log":
                self.x_scale = "log"
                self.AutoScale(force=True)
                try:
                    self.ax.set_xscale("log")
                    self.error_ax.set_xscale("log")
                except OverflowError:
                    self.AutoScale(force=True)
                except UserWarning:
                    pass
            elif scalestring in ["linear", "lin"]:
                self.x_scale = "linear"
                self.ax.set_xscale("linear")
                self.error_ax.set_xscale("linear")
                self.AutoScale(force=True)
            else:
                raise ValueError("Not allowed scaling")

            try:
                self.flush_plot()
            except UserWarning:
                pass

            self.WriteConfig()
            self.state_changed.emit(self.GetZoom(), self.GetYScale(), self.autoscale, self.GetXScale())

    def create_axes(self):
        try:
            gs = self.figure.add_gridspec(4, 1)
        except AttributeError:
            from matplotlib.gridspec import GridSpec

            gs = GridSpec(4, 1)
        self.ax = self.figure.add_subplot(gs[:3, 0])
        self.ax.get_xaxis().set_visible(False)
        self.error_ax = self.figure.add_subplot(gs[3, 0], sharex=self.ax)
        self.ax.set_autoscale_on(False)
        self.error_ax.set_autoscale_on(True)
        self.ax.set_ylabel("y")
        self.error_ax.set_ylabel("FOM")
        self.error_ax.set_xlabel("x")

    def update_labels(self, xlabel=None, ylabel=None, elabel=None):
        if xlabel is not None:
            self.error_ax.set_xlabel(xlabel)
        if ylabel is not None:
            self.ax.set_ylabel(ylabel)
        if elabel is not None:
            self.error_ax.set_ylabel(elabel)
        self.flush_plot()

    def autoscale_error_ax(self):
        ymin = min([array(line.get_ydata()).min() for line in self.error_ax.lines if len(line.get_ydata()) > 0])
        ymax = max([array(line.get_ydata()).max() for line in self.error_ax.lines if len(line.get_ydata()) > 0])
        if ymin >= ymax:
            return
        self.error_ax.set_ylim(ymin * (1 - sign(ymin) * 0.05), ymax * (1 + sign(ymax) * 0.05))

    def singleplot(self, data):
        if not self.ax:
            self.create_axes()

    def plot_data(self, data: DataList):
        if not self.ax:
            self.create_axes()

        while len(self.ax.lines) > 0:
            self.ax.lines[0].remove()
        while len(self.ax.collections) > 0:
            self.ax.collections[0].remove()

        if self.y_scale == "linear":
            [
                self.ax.plot(
                    data_set.x,
                    data_set.y,
                    color=data_set.data_color,
                    lw=data_set.data_linethickness,
                    ls=data_set.data_linetype,
                    marker=data_set.data_symbol,
                    ms=data_set.data_symbolsize,
                    zorder=1,
                )
                for data_set in data
                if not data_set.use_error and data_set.show
            ]
            [
                self.ax.errorbar(
                    data_set.x,
                    data_set.y,
                    yerr=c_[data_set.error * (data_set.error > 0), data_set.error].transpose(),
                    color=data_set.data_color,
                    lw=data_set.data_linethickness,
                    ls=data_set.data_linetype,
                    marker=data_set.data_symbol,
                    ms=data_set.data_symbolsize,
                    zorder=2,
                )
                for data_set in data
                if data_set.use_error and data_set.show
            ]
        if self.y_scale == "log":
            [
                self.ax.plot(
                    data_set.x.compress(data_set.y > 0),
                    data_set.y.compress(data_set.y > 0),
                    color=data_set.data_color,
                    lw=data_set.data_linethickness,
                    ls=data_set.data_linetype,
                    marker=data_set.data_symbol,
                    ms=data_set.data_symbolsize,
                    zorder=1,
                )
                for data_set in data
                if not data_set.use_error and data_set.show
            ]
            [
                self.ax.errorbar(
                    data_set.x.compress(data_set.y - data_set.error > 0),
                    data_set.y.compress(data_set.y - data_set.error > 0),
                    yerr=c_[data_set.error * (data_set.error > 0), data_set.error]
                    .transpose()
                    .compress(data_set.y - data_set.error > 0),
                    color=data_set.data_color,
                    lw=data_set.data_linethickness,
                    ls=data_set.data_linetype,
                    marker=data_set.data_symbol,
                    ms=data_set.data_symbolsize,
                    zorder=2,
                )
                for data_set in data
                if data_set.use_error and data_set.show
            ]
        self.AutoScale()
        self.flush_plot()

    def plot_data_fit(self, data: DataList):
        if not self.ax:
            self.create_axes()

        shown_data = [data_set for data_set in data if data_set.show]
        if len(self.ax.lines) == (2 * len(shown_data)):
            for i, data_set in enumerate(shown_data):
                self.ax.lines[i].set_data(data_set.x, data_set.y)
                self.ax.lines[i + len(shown_data)].set_data(data_set.x, data_set.y_sim)
                self.error_ax.lines[i].set_data(data_set.x, ma.fix_invalid(data_set.y_fom, fill_value=0))
        else:
            while len(self.ax.lines) > 0:
                self.ax.lines[0].remove()
            while len(self.ax.collections) > 0:
                self.ax.collections[0].remove()
            while len(self.error_ax.lines) > 0:
                self.error_ax.lines[0].remove()
            while len(self.error_ax.collections) > 0:
                self.error_ax.collections[0].remove()
            [
                self.ax.plot(
                    data_set.x,
                    data_set.y,
                    color=data_set.data_color,
                    lw=data_set.data_linethickness,
                    ls=data_set.data_linetype,
                    marker=data_set.data_symbol,
                    ms=data_set.data_symbolsize,
                    zorder=1,
                )
                for data_set in shown_data
            ]
            [
                self.ax.plot(
                    data_set.x,
                    data_set.y_sim,
                    color=data_set.sim_color,
                    lw=data_set.sim_linethickness,
                    ls=data_set.sim_linetype,
                    marker=data_set.sim_symbol,
                    ms=data_set.sim_symbolsize,
                    zorder=5,
                )
                for data_set in shown_data
            ]
            [
                self.error_ax.plot(
                    data_set.x,
                    ma.fix_invalid(data_set.y_fom, fill_value=0),
                    color=data_set.sim_color,
                    lw=data_set.sim_linethickness,
                    ls=data_set.sim_linetype,
                    marker=data_set.sim_symbol,
                    ms=data_set.sim_symbolsize,
                    zorder=2,
                )
                for data_set in shown_data
            ]
        self.autoscale_error_ax()
        self.flush_plot()

    def plot_data_sim(self, data: DataList):
        if not self.ax:
            self.create_axes()

        p_options = [self.y_scale] + [
            [
                data_set.data_color,
                data_set.data_linethickness,
                data_set.data_linetype,
                data_set.data_symbol,
                data_set.data_symbolsize,
                data_set.sim_color,
                data_set.sim_linethickness,
                data_set.sim_linetype,
                data_set.sim_symbol,
                data_set.sim_symbolsize,
            ]
            for data_set in data
        ]
        p_datasets = [data_set for data_set in data if data_set.show]
        pe_datasets = [data_set for data_set in data if data_set.use_error and data_set.show]
        s_datasets = [
            data_set for data_set in data if data_set.show and data_set.use and data_set.x.shape == data_set.y_sim.shape
        ]
        if (
            self._last_poptions == p_options
            and len(self.ax.lines) == (len(p_datasets) + len(s_datasets))
            and len(self.ax.collections) == len(pe_datasets)
        ):
            for i, data_set in enumerate(p_datasets):
                if self.y_scale == "linear":
                    self.ax.lines[i].set_data(data_set.x, data_set.y)
                elif data_set.use_error:
                    fltr = (data_set.y - data_set.error) > 0
                    self.ax.lines[i].set_data(data_set.x.compress(fltr), data_set.y.compress(fltr))
                else:
                    fltr = data_set.y > 0
                    self.ax.lines[i].set_data(data_set.x.compress(fltr), data_set.y.compress(fltr))
            for j, data_set in enumerate(s_datasets):
                self.ax.lines[len(p_datasets) + j].set_data(data_set.x, data_set.y_sim)
                self.error_ax.lines[j].set_data(data_set.x, ma.fix_invalid(data_set.y_fom, fill_value=0))
            for k, data_set in enumerate(pe_datasets):
                ybot = data_set.y - data_set.error
                ytop = data_set.y + data_set.error
                segment_data = hstack([data_set.x, ybot, data_set.x, ytop]).reshape(2, 2, -1).transpose(2, 0, 1)
                if self.y_scale == "log":
                    if data_set.use_error:
                        fltr = (data_set.y - data_set.error) > 0
                    else:
                        fltr = data_set.y > 0
                    segment_data = segment_data[fltr, :, :]
                self.ax.collections[k].set_segments(segment_data)
        else:
            while len(self.ax.lines) > 0:
                self.ax.lines[0].remove()
            while len(self.ax.collections) > 0:
                self.ax.collections[0].remove()
            while len(self.error_ax.lines) > 0:
                self.error_ax.lines[0].remove()
            while len(self.error_ax.collections) > 0:
                self.error_ax.collections[0].remove()

            if self.y_scale == "linear":
                [
                    self.ax.plot(
                        data_set.x,
                        data_set.y,
                        color=data_set.data_color,
                        lw=data_set.data_linethickness,
                        ls=data_set.data_linetype,
                        marker=data_set.data_symbol,
                        ms=data_set.data_symbolsize,
                        zorder=1,
                    )
                    for data_set in p_datasets
                    if not data_set.use_error
                ]
                [
                    self.ax.errorbar(
                        data_set.x,
                        data_set.y,
                        yerr=c_[data_set.error * (data_set.error > 0), data_set.error].transpose(),
                        color=data_set.data_color,
                        lw=data_set.data_linethickness,
                        ls=data_set.data_linetype,
                        marker=data_set.data_symbol,
                        ms=data_set.data_symbolsize,
                        zorder=2,
                    )
                    for data_set in pe_datasets
                ]
            if self.y_scale == "log":
                [
                    self.ax.plot(
                        data_set.x.compress(data_set.y > 0),
                        data_set.y.compress(data_set.y > 0),
                        color=data_set.data_color,
                        lw=data_set.data_linethickness,
                        ls=data_set.data_linetype,
                        marker=data_set.data_symbol,
                        ms=data_set.data_symbolsize,
                        zorder=1,
                    )
                    for data_set in p_datasets
                    if not data_set.use_error
                ]
                [
                    self.ax.errorbar(
                        data_set.x.compress(data_set.y - data_set.error > 0),
                        data_set.y.compress(data_set.y - data_set.error > 0),
                        yerr=c_[data_set.error * (data_set.error > 0), data_set.error]
                        .transpose()
                        .compress(data_set.y - data_set.error > 0),
                        color=data_set.data_color,
                        lw=data_set.data_linethickness,
                        ls=data_set.data_linetype,
                        marker=data_set.data_symbol,
                        ms=data_set.data_symbolsize,
                        zorder=2,
                    )
                    for data_set in pe_datasets
                ]
            [
                self.ax.plot(
                    data_set.x,
                    data_set.y_sim,
                    color=data_set.sim_color,
                    lw=data_set.sim_linethickness,
                    ls=data_set.sim_linetype,
                    marker=data_set.sim_symbol,
                    ms=data_set.sim_symbolsize,
                    zorder=5,
                )
                for data_set in s_datasets
            ]
            [
                self.error_ax.plot(
                    data_set.x,
                    ma.fix_invalid(data_set.y_fom, fill_value=0),
                    color=data_set.sim_color,
                    lw=data_set.sim_linethickness,
                    ls=data_set.sim_linetype,
                    marker=data_set.sim_symbol,
                    ms=data_set.sim_symbolsize,
                )
                for data_set in s_datasets
            ]
            self._last_poptions = p_options
        try:
            self.autoscale_error_ax()
        except ValueError:
            pass
        self.AutoScale()
        self.flush_plot()


class ErrorPanelConfig(BasePlotConfig):
    section = "fom plot"


class ErrorPlotPanel(PlotPanel):
    """Class for plotting evolution of the error as a function of the generations."""

    def __init__(self, parent, color=None, dpi=None, **kwargs):
        super().__init__(parent, color=color, dpi=dpi, config_class=ErrorPanelConfig, **kwargs)
        self.update = self.errorplot
        self.update(None)

    def errorplot(self, data):
        if not self.ax:
            self.ax = self.figure.add_subplot(111)

        self.ax.set_autoscale_on(False)
        while len(self.ax.lines) > 0:
            self.ax.lines[0].remove()
        if data is None:
            theta = arange(0.1, 10, 0.001)
            self.ax.plot(theta, floor(15 - theta), "-r")
        else:
            self.ax.plot(data[:, 0], data[:, 1], "-r")
            if self.GetAutoScale() and len(data) > 0:
                self.ax.set_ylim(data[:, 1].min() * 0.95, data[:, 1].max() * 1.05)
                xmin, xmax = data[:, 0].min(), data[:, 0].max()
                if xmin == xmax:
                    xmin -= 0.01
                    xmax += 0.01
                self.ax.set_xlim(xmin, xmax)

        self.ax.set_xlabel("Iteration")
        self.ax.set_ylabel("FOM")
        try:
            self.figure.tight_layout(h_pad=0)
        except Exception:
            pass
        self.flush_plot()

    @QtCore.Slot(object)
    def OnSolverPlotEvent(self, event):
        fom_log = event.fom_log
        self.update(fom_log)


class ParsPanelConfig(BasePlotConfig):
    section = "pars plot"


class ParsPlotPanel(PlotPanel):
    """Class to plot the different parametervalues during a fit."""

    def __init__(self, parent, color=None, dpi=None, **kwargs):
        super().__init__(parent, color=color, dpi=dpi, config_class=ParsPanelConfig, **kwargs)
        self.update(None)
        self.ax = self.figure.add_subplot(111)
        self.update = self.Plot

    def Plot(self, data):
        if data.fitting:
            pop = array(data.population)
            norm = 1.0 / (data.max_val - data.min_val)
            best = (array(data.values) - data.min_val) * norm
            pop_min = (pop.min(0) - data.min_val) * norm
            pop_max = (pop.max(0) - data.min_val) * norm

            self.ax.cla()
            width = 0.8
            x = arange(len(best))
            self.ax.set_autoscale_on(False)
            self.ax.bar(x, pop_max - pop_min, bottom=pop_min, color="b", width=width)
            self.ax.plot(x, best, "ro")
            if self.GetAutoScale():
                self.ax.axis([x.min() - width, x.max() + width, 0.0, 1.0])

        self.ax.set_xlabel("Parameter Index (only fittable)")
        self.ax.set_ylabel("Relative value in min/max range")
        self.figure.tight_layout(h_pad=0)
        self.flush_plot()

    @QtCore.Slot(object)
    def OnSolverParameterEvent(self, event):
        self.update(event)


class FomPanelConfig(BasePlotConfig):
    section = "fom scan plot"


class FomScanPlotPanel(PlotPanel):
    """Class to take care of fom scans."""

    def __init__(self, parent, color=None, dpi=None, **kwargs):
        super().__init__(parent, color=color, dpi=dpi, config_class=FomPanelConfig, **kwargs)
        self.update(None)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_autoscale_on(False)
        self.update = self.Plot
        self.type = "project"

    def SetPlottype(self, type):
        if type.lower() == "project":
            self.type = "project"
            self.SetAutoScale(False)
        elif type.lower() == "scan":
            self.SetAutoScale(True)
            self.type = "scan"

    def Plot(self, data, l1="", l2=""):
        self.ax.cla()
        x, y, bestx, besty, e_scale = data[0], data[1], data[2], data[3], data[4]
        if self.type.lower() == "project":
            self.ax.set_autoscale_on(False)
            self.ax.plot(x, y, "ob")
            self.ax.plot([bestx], [besty], "or")
            self.ax.hlines(besty * e_scale, x.min(), x.max(), "r")
            self.ax.axis(
                [
                    x.min(),
                    x.max(),
                    min(y.min(), besty) * 0.95,
                    (besty * e_scale - min(y.min(), besty)) * 2.0 + min(y.min(), besty),
                ]
            )
        elif self.type.lower() == "scan":
            self.ax.plot(x, y, "b")
            self.ax.plot([bestx], [besty], "or")
            self.ax.hlines(besty * e_scale, x.min(), x.max(), "r")
            if self.GetAutoScale():
                self.ax.set_autoscale_on(False)
                self.ax.axis([x.min(), x.max(), min(y.min(), besty) * 0.95, y.max() * 1.05])

        self.ax.set_xlabel(l1)
        self.ax.set_ylabel(l2)
        self.flush_plot()
