"""
Classes to interface with the bumps module for fitting and statistical analysis.
"""

import threading

import wx

from bumps import __version__ as bumps_version
from bumps.dream.corrplot import _hists
from bumps.fitproblem import nllf_scale
from bumps.formatnum import format_uncertainty
from bumps.monitor import TimedUpdate
from matplotlib.colors import LogNorm
from numpy import array, maximum, newaxis, sqrt
from wx.grid import EVT_GRID_CELL_LEFT_DCLICK, Grid

from ..bumps_optimizer import BumpsResult
from ..plugins.utils import ShowInfoDialog
from .exception_handling import CatchModelError
from .plotpanel import BasePlotConfig, PlotPanel


class ProgressMonitor(TimedUpdate):
    """
    Display fit progress on the console
    """

    def __init__(self, problem, pbar, ptxt, progress=0.25, improvement=5.0):
        TimedUpdate.__init__(self, progress=progress, improvement=improvement)
        self.problem = problem
        self.pbar = pbar  # wx.Gauge
        self.ptxt = ptxt  # wx.StaticText
        self.chis = []
        self.steps = []

    def show_progress(self, history):
        scale, err = nllf_scale(self.problem)
        chisq = format_uncertainty(scale * history.value[0], err)
        wx.CallAfter(self.ptxt.SetLabel, "step: %s/%s  cost: %s" % (history.step[0], self.pbar.GetRange(), chisq))
        wx.CallAfter(self.pbar.SetValue, min(history.step[0], self.pbar.GetRange()))
        self.steps.append(history.step[0])
        self.chis.append(scale * history.value[0])

    def show_improvement(self, history):
        return
        p = self.problem.getp()
        try:
            self.problem.setp(history.point[0])
            out = '<table width="50%"><tr>'
            out += "</tr>\n<tr>".join(
                [
                    "<td>%s</td><td>%s</td><td>%s</td>" % (pi.name, pi.value, pi.bounds)
                    for pi in self.problem._parameters
                ]
            )
            self.result_text.value = out + "</td></tr></table>"
        finally:
            self.problem.setp(p)
        with self.plot_out:
            fig = figure()
            plot(self.steps, self.chis)
            clear_output()
            display(fig)
            close()


class HeaderCopyGrid(wx.grid.Grid):
    """Grid calss that copies the column and row headers together with the data."""

    def __init__(self, *args, **opts):
        wx.grid.Grid.__init__(self, *args, **opts)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKey)

    def OnKey(self, event):
        # If Ctrl+C is pressed...
        if event.ControlDown() and event.GetKeyCode() == 67:
            self.copy()
        else:
            event.Skip()
            return

    def copy(self):
        output = ""
        for block in self.GetSelectedBlocks():
            top, left = block.GetTopLeft()
            bottom, right = block.GetBottomRight()
            output += "\t"
            for col in range(left, right + 1):
                name = self.GetColLabelValue(col).replace("\n", ".")
                output += f"{name}\t"
            output = output[:-1] + "\n"
            for row in range(top, bottom + 1):
                name = self.GetRowLabelValue(row).replace("\n", ".")
                output += f"{name}\t"
                for col in range(left, right + 1):
                    output += f"{self.GetCellValue(row, col)}\t"
                output = output[:-1] + "\n"
            output += "\n\n"

        clipboard = wx.TextDataObject()
        clipboard.SetText(output)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(clipboard)
            wx.TheClipboard.Close()
        else:
            wx.MessageBox("Can't open the clipboard", "Error")


class StatisticsPanelConfig(BasePlotConfig):
    section = "statistics plot"


class StatisticalAnalysisDialog(wx.Dialog):
    rel_cov = None
    thread: threading.Thread
    _res: BumpsResult

    def __init__(self, parent, model, prev_result: BumpsResult = None):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        self.SetTitle("Statistical Analysis of Parameters")

        dpi_scale_factor = wx.GetApp().dpi_scale_factor

        self.ptxt = wx.StaticText(self, label="...")
        self.pbar = wx.Gauge(self, range=1000)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(hbox, proportion=1, flag=wx.EXPAND)

        lpanel = wx.Panel(self)
        gpanel = wx.Panel(self)
        gbox = wx.BoxSizer(wx.VERTICAL)
        gpanel.SetSizer(gbox)
        self.grid = HeaderCopyGrid(gpanel)
        gbox.Add(wx.StaticText(gpanel, label="Estimated covariance matrix:"), proportion=0, flag=wx.FIXED_MINSIZE)
        gbox.Add(self.grid, proportion=1, flag=wx.EXPAND)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        gbox.Add(hbox2, proportion=0, flag=wx.FIXED_MINSIZE)
        self.normalize_checkbox = wx.CheckBox(gpanel, label="Normalize value (σ_ij/σ_i/σ_j)")
        self.normalize_checkbox.SetValue(True)
        self.normalize_checkbox.Bind(wx.EVT_CHECKBOX, self.OnToggleNormalize)
        hbox2.Add(self.normalize_checkbox, proportion=0, flag=wx.FIXED_MINSIZE)
        self.chicorrect_checkbox = wx.CheckBox(gpanel, label="Correct by sqrt(chi²)")
        self.chicorrect_checkbox.SetToolTip(
            "Renormalize the parameter errors with sqrt(chi²).\n"
            "This is helpful if you assume the model is correct and "
            "error bars are only a relative measure.\n"
            "The resulting scaled error bars would then lead to chi²=1."
        )
        self.chicorrect_checkbox.SetValue(False)
        self.chicorrect_checkbox.Bind(wx.EVT_CHECKBOX, self.OnToggleChi2)
        hbox2.Add(self.chicorrect_checkbox, proportion=0, flag=wx.FIXED_MINSIZE)

        nfparams = len([pi for pi in model.parameters if pi.fit])
        self.grid.CreateGrid(nfparams + 2, nfparams)
        self.grid.SetColLabelTextOrientation(wx.VERTICAL)
        self.grid.SetColLabelSize(int(dpi_scale_factor * 80))
        for i in range(nfparams):
            self.grid.SetRowSize(i + 2, int(dpi_scale_factor * 80))
            self.grid.SetColSize(i, int(dpi_scale_factor * 80))
            self.grid.SetColLabelValue(i, "%i" % i)
            self.grid.SetRowLabelValue(i + 2, "%i" % i)
        self.grid.SetRowSize(0, int(dpi_scale_factor * 80 / 2))
        self.grid.SetRowSize(1, int(dpi_scale_factor * 80 / 2))
        self.grid.DisableCellEditControl()
        self.grid.SetMinSize((int(dpi_scale_factor * 200), int(dpi_scale_factor * 200)))
        self.grid.Bind(EVT_GRID_CELL_LEFT_DCLICK, self.OnSelectCell)

        rpbox = wx.BoxSizer(wx.VERTICAL)
        rpanel = wx.Panel(self)
        rpanel.SetSizer(rpbox)

        self.fom_text = wx.StaticText(rpanel, label="FOM chi²/bars: -")
        font = wx.Font(wx.FontInfo(2.0 * self.fom_text.GetFont().GetPointSize()))
        self.fom_text.SetFont(font)
        rpbox.Add(self.fom_text, proportion=0, flag=wx.FIXED_MINSIZE | wx.EXPAND)

        plot_panel = PlotPanel(rpanel, config_class=StatisticsPanelConfig)
        plot_panel.SetMinSize((int(dpi_scale_factor * 200), int(dpi_scale_factor * 200)))
        self.plot_panel = plot_panel
        self.ax = plot_panel.figure.add_subplot(111)
        rpbox.Add(plot_panel, proportion=1, flag=wx.EXPAND)

        hbox.Add(lpanel, proportion=0, flag=wx.EXPAND | wx.ALIGN_TOP)
        hbox.Add(gpanel, proportion=1, flag=wx.EXPAND)
        hbox.Add(rpanel, proportion=1, flag=wx.EXPAND)

        lsizer = wx.GridSizer(vgap=1, hgap=2, cols=2)
        lpanel.SetSizer(lsizer)
        self.entries = {}
        for key, emin, emax, val in [
            ("pop", 1, 20 * nfparams, 2 * nfparams),
            ("samples", 1000, 10000000, 10000),
            ("burn", 0, 10000, 200),
        ]:
            lsizer.Add(wx.StaticText(lpanel, label="%s:" % key), flag=wx.FIXED_MINSIZE)
            self.entries[key] = wx.SpinCtrl(lpanel, wx.ID_ANY, min=emin, max=emax, value=str(val))
            lsizer.Add(self.entries[key], flag=wx.FIXED_MINSIZE)

        lsizer.AddStretchSpacer(10)
        self.run_button = wx.Button(self, label="Run Analysis...")
        vbox.Add(self.run_button)
        vbox.Add(self.ptxt, proportion=0, flag=wx.EXPAND)
        vbox.Add(self.pbar, proportion=0, flag=wx.EXPAND)

        self.Bind(wx.EVT_BUTTON, self.OnRunAnalysis, self.run_button)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.model = model
        self.thread = None

        psize = parent.GetSize()
        self.SetSize(int(psize.GetWidth() * 0.75), int(psize.GetHeight() * 0.75))

        if prev_result is not None:
            self._res = prev_result
            self.bproblem = prev_result.bproblem
            self.display_bumps()

    def OnRunAnalysis(self, event):
        if self.thread is not None:
            self.run_button.SetLabel("Run Analysis...")
            self.bproblem.fitness.stop_fit = True
            return
        self.thread = threading.Thread(target=self.run_bumps)
        self.thread.start()
        self.run_button.SetLabel("Stop Run")

    def run_bumps(self):
        self.bproblem = self.model.bumps_problem()
        mon = ProgressMonitor(self.bproblem, self.pbar, self.ptxt)
        pop = self.entries["pop"].GetValue()
        burn = self.entries["burn"].GetValue()
        samples = self.entries["samples"].GetValue()
        self.pbar.SetRange(int(samples / (len(self.bproblem.model_parameters()) * pop)) + burn)

        with CatchModelError(self, "bumps_modeling") as mgr:
            res = self.model.bumps_fit(
                method="dream",
                pop=pop,
                samples=samples,
                burn=burn,
                thin=1,
                alpha=0,
                outliers="none",
                trim=False,
                monitors=[mon],
                problem=self.bproblem,
            )

        self.run_button.SetLabel("Run Analysis...")
        if mgr.successful:
            self._res = res
            wx.CallAfter(self.display_bumps)

    def display_bumps(self):
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None
        self.pbar.SetValue(0)

        res = self._res
        self.chisq = res.chisq
        if self.chicorrect_checkbox.IsChecked():
            scl = sqrt(self.chisq)
        else:
            scl = 1.0

        self.fom_text.SetLabel("FOM chi²/bars: %.3f" % self.chisq)
        self.draw = res.state.draw()
        pnames = list(self.bproblem.model_parameters().keys())
        sort_indices = [pnames.index(ni) for ni in self.draw.labels]

        self.abs_cov = res.cov
        self.rel_cov = res.cov / res.dx[:, newaxis] / res.dx[newaxis, :]
        rel_max = [0, 1, abs(self.rel_cov[0, 1])]
        if self.normalize_checkbox.IsChecked():
            display_cov = self.rel_cov
            fmt = "%.6f"
        else:
            display_cov = self.abs_cov * scl**2
            fmt = "%.4g"
        self.grid.SetRowLabelValue(0, "Value:")
        self.grid.SetRowLabelValue(1, "Error:")
        for i, ci in enumerate(self.rel_cov):
            plabel = "\n".join(self.draw.labels[i].rsplit("_", 1))
            self.grid.SetColLabelValue(sort_indices[i], plabel)
            self.grid.SetRowLabelValue(sort_indices[i] + 2, plabel)
            self.grid.SetCellValue(0, sort_indices[i], "%.8g" % res.x[i])
            self.grid.SetCellAlignment(0, sort_indices[i], wx.ALIGN_CENTRE, wx.ALIGN_CENTRE)
            self.grid.SetCellValue(1, sort_indices[i], "%.4g" % (res.dx[i] * scl))
            self.grid.SetCellAlignment(1, sort_indices[i], wx.ALIGN_CENTRE, wx.ALIGN_CENTRE)
            self.grid.SetReadOnly(0, sort_indices[i])
            self.grid.SetReadOnly(1, sort_indices[i])
            self.grid.SetCellBackgroundColour(0, sort_indices[i], "#cccccc")
            self.grid.SetCellBackgroundColour(1, sort_indices[i], "#cccccc")
            for j, cj in enumerate(ci):
                self.grid.SetCellValue(sort_indices[i] + 2, sort_indices[j], fmt % display_cov[i, j])
                self.grid.SetReadOnly(sort_indices[i] + 2, sort_indices[j])
                self.grid.SetCellAlignment(sort_indices[i] + 2, sort_indices[j], wx.ALIGN_CENTRE, wx.ALIGN_CENTRE)
                if i == j:
                    self.grid.SetCellBackgroundColour(sort_indices[i] + 2, sort_indices[j], "#888888")
                elif abs(cj) > 0.4:
                    self.grid.SetCellBackgroundColour(sort_indices[i] + 2, sort_indices[j], "#ffcccc")
                elif abs(cj) > 0.3:
                    self.grid.SetCellBackgroundColour(sort_indices[i] + 2, sort_indices[j], "#ffdddd")
                elif abs(cj) > 0.2:
                    self.grid.SetCellBackgroundColour(sort_indices[i] + 2, sort_indices[j], "#ffeeee")
                if i != j and abs(cj) > rel_max[2]:
                    rel_max = [min(i, j), max(i, j), abs(cj)]
        self.hists = _hists(self.draw.points.T, bins=50)

        fig = self.plot_panel.figure
        fig.clear()
        ax = fig.add_subplot(111)
        data, x, y = self.hists[(rel_max[0], rel_max[1])]
        vmin, vmax = data[data > 0].min(), data.max()
        ax.pcolorfast(y, x, maximum(vmin, data), norm=LogNorm(vmin, vmax), cmap="inferno")
        ax.set_xlabel(self.draw.labels[rel_max[1]])
        ax.set_ylabel(self.draw.labels[rel_max[0]])
        self.plot_panel.flush_plot()

        # add analysis data to model for later storage in export header
        exdict = {}
        exdict["library"] = "bumps"
        exdict["version"] = bumps_version
        exdict["settings"] = dict(
            pop=self.entries["pop"].GetValue(),
            burn=self.entries["burn"].GetValue(),
            samples=self.entries["samples"].GetValue(),
        )
        exdict["parameters"] = [
            dict(
                name=li,
                value=float(xi),
                error=float(dxi),
                cross_correlations=dict((self.draw.labels[j], float(res.cov[i, j])) for j in range(len(res.x))),
            )
            for i, (li, xi, dxi) in enumerate(zip(self.draw.labels, res.x, res.dx))
        ]
        self.model.extra_analysis["statistics_mcmc"] = exdict
        if (res.dxpm[:, 0] > 0.0).any() or (res.dxpm[:, 1] < 0.0).any():
            ShowInfoDialog(
                self,
                "There is something wrong in the error estimation, low/high values don't have the right sign.\n\n"
                "This can be caused by non single-modal parameter statistics, closeness to bounds or too low value of"
                "'burn' before stampling.\n\n"
                "Using estimated sigma for grid, instead.",
                title="Issue in uncertainty estimation",
            )
            dxpm = array([-res.dx, res.dx]).T * scl
        else:
            dxpm = res.dxpm * scl
        reverse_sort = [sort_indices.index(i) for i in range(len(sort_indices))]
        error_labels = ["(%.3e, %.3e)" % (dxup, dxdown) for dxup, dxdown in dxpm[reverse_sort]]
        self.model.parameters.set_error_pars(error_labels)

    def OnToggleNormalize(self, evt):
        if self.rel_cov is None:
            evt.Skip()
            return
        if self.normalize_checkbox.IsChecked():
            display_cov = self.rel_cov
            fmt = "%.6f"
        else:
            display_cov = self.abs_cov
            if self.chicorrect_checkbox.IsChecked():
                display_cov = display_cov * self.chisq
            fmt = "%.4g"
        pnames = list(self.bproblem.model_parameters().keys())
        sort_indices = [pnames.index(ni) for ni in self.draw.labels]
        for i, ci in enumerate(self.rel_cov):
            for j, cj in enumerate(ci):
                self.grid.SetCellValue(sort_indices[i] + 2, sort_indices[j], fmt % display_cov[i, j])

    def OnToggleChi2(self, evt):
        if self.rel_cov is None:
            evt.Skip()
            return
        if self.chicorrect_checkbox.IsChecked():
            scl = sqrt(self.chisq)
        else:
            scl = 1.0
        pnames = list(self.bproblem.model_parameters().keys())
        sort_indices = [pnames.index(ni) for ni in self.draw.labels]
        res = self._res
        for i, dxi in enumerate(res.dx):
            self.grid.SetCellValue(1, sort_indices[i], "%.4g" % (dxi * scl))

        if (res.dxpm[:, 0] > 0.0).any() or (res.dxpm[:, 1] < 0.0).any():
            dxpm = array([-res.dx, res.dx]).T * scl
        else:
            dxpm = res.dxpm * scl
        reverse_sort = [sort_indices.index(i) for i in range(len(sort_indices))]
        error_labels = ["(%.3e, %.3e)" % (dxup, dxdown) for dxup, dxdown in dxpm[reverse_sort]]
        self.model.parameters.set_error_pars(error_labels)

        if self.normalize_checkbox.IsChecked():
            return
        else:
            display_cov = self.abs_cov
            if self.chicorrect_checkbox.IsChecked():
                display_cov = display_cov * self.chisq
            fmt = "%.4g"
        for i, ci in enumerate(self.rel_cov):
            for j, cj in enumerate(ci):
                self.grid.SetCellValue(sort_indices[i] + 2, sort_indices[j], fmt % display_cov[i, j])

    def OnSelectCell(self, evt):
        ri, rj = evt.GetCol(), evt.GetRow() - 2
        pnames = list(self.bproblem.model_parameters().keys())
        reverse_indices = [self.draw.labels.index(ni) for ni in pnames]
        i = reverse_indices[ri]
        j = reverse_indices[rj]
        if i == j or j < 0:
            return
        elif i > j:
            itmp = i
            i = j
            j = itmp

        fig = self.plot_panel.figure
        fig.clear()
        ax = fig.add_subplot(111)
        data, x, y = self.hists[(i, j)]
        vmin, vmax = data[data > 0].min(), data.max()
        ax.pcolorfast(y, x, maximum(vmin, data), norm=LogNorm(vmin, vmax), cmap="inferno")
        ax.set_xlabel(self.draw.labels[j])
        ax.set_ylabel(self.draw.labels[i])
        self.plot_panel.flush_plot()

    def OnClose(self, event):
        if self.thread is not None and self.thread.is_alive():
            # a running bumps simulation can't be interrupted from other thread, stop window from closing
            self.ptxt.SetLabel("Simulation running")
            self.bproblem.fitness.stop_fit = True
            self.thread.join(timeout=5.0)
        event.Skip()
