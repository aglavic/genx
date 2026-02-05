"""
Qt port of bumps interface for statistical analysis dialogs.
"""

import threading

import bumps
from bumps import __version__ as bumps_version
from bumps.dream.corrplot import _hists
from bumps.fitproblem import nllf_scale
from bumps.formatnum import format_uncertainty
from bumps.monitor import TimedUpdate
from matplotlib.colors import LogNorm
from numpy import array, maximum, newaxis, sqrt
from PySide6 import QtCore, QtGui, QtWidgets

from ..bumps_optimizer import BumpsResult
from .exception_handling import CatchModelError
from .plotpanel import BasePlotConfig, PlotPanel
from .utils import ShowInfoDialog


class ProgressMonitor(TimedUpdate):
    """
    Display fit progress on the dialog widgets.
    """

    def __init__(self, problem, pbar, ptxt, progress=0.25, improvement=5.0):
        super().__init__(progress=progress, improvement=improvement)
        self.problem = problem
        self.pbar = pbar
        self.ptxt = ptxt
        self.chis = []
        self.steps = []

    def show_progress(self, history):
        scale, err = nllf_scale(self.problem)
        chisq = format_uncertainty(scale * history.value[0], err)
        QtCore.QTimer.singleShot(
            0,
            lambda: self.ptxt.setText(f"step: {history.step[0]}/{self.pbar.maximum()}  cost: {chisq}"),
        )
        QtCore.QTimer.singleShot(
            0,
            lambda: self.pbar.setValue(min(history.step[0], self.pbar.maximum())),
        )
        self.steps.append(history.step[0])
        self.chis.append(scale * history.value[0])

    def show_improvement(self, history):
        return


class HeaderCopyTable(QtWidgets.QTableWidget):
    """Table that copies headers together with data on Ctrl+C."""

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier and event.key() == QtCore.Qt.Key.Key_C:
            self.copy()
            return
        super().keyPressEvent(event)

    def copy(self):
        output = ""
        ranges = self.selectedRanges()
        if not ranges:
            return
        for block in ranges:
            top, left = block.topRow(), block.leftColumn()
            bottom, right = block.bottomRow(), block.rightColumn()
            output += "\t"
            for col in range(left, right + 1):
                header = self.horizontalHeaderItem(col)
                name = header.text().replace("\n", ".") if header is not None else ""
                output += f"{name}\t"
            output = output[:-1] + "\n"
            for row in range(top, bottom + 1):
                header = self.verticalHeaderItem(row)
                name = header.text().replace("\n", ".") if header is not None else ""
                output += f"{name}\t"
                for col in range(left, right + 1):
                    item = self.item(row, col)
                    output += f"{item.text() if item is not None else ''}\t"
                output = output[:-1] + "\n"
            output += "\n\n"
        QtWidgets.QApplication.clipboard().setText(output)


class StatisticsPanelConfig(BasePlotConfig):
    section = "statistics plot"


class StatisticalAnalysisDialog(QtWidgets.QDialog):
    rel_cov = None
    thread: threading.Thread
    _res: BumpsResult

    def __init__(self, parent, model, prev_result: BumpsResult = None):
        super().__init__(parent)
        self.setWindowTitle("Statistical Analysis of Parameters")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowMaximizeButtonHint)

        self.model = model
        self.thread = None

        self.ptxt = QtWidgets.QLabel("...")
        self.pbar = QtWidgets.QProgressBar()
        self.pbar.setRange(0, 1000)

        main_row = QtWidgets.QHBoxLayout()

        left_panel = QtWidgets.QWidget()
        left_layout = QtWidgets.QFormLayout(left_panel)
        self.entries = {}
        nfparams = len([pi for pi in model.parameters if pi.fit])
        for key, emin, emax, val in [
            ("pop", 1, 20 * nfparams, 2 * nfparams),
            ("samples", 1000, 10000000, 10000),
            ("burn", 0, 10000, 200),
        ]:
            ctrl = QtWidgets.QSpinBox()
            ctrl.setRange(emin, emax)
            ctrl.setValue(val)
            left_layout.addRow(f"{key}:", ctrl)
            self.entries[key] = ctrl

        grid_panel = QtWidgets.QWidget()
        grid_layout = QtWidgets.QVBoxLayout(grid_panel)
        grid_layout.addWidget(QtWidgets.QLabel("Estimated covariance matrix:"))
        self.grid = HeaderCopyTable(grid_panel)
        self.grid.setRowCount(nfparams + 2)
        self.grid.setColumnCount(nfparams)
        self.grid.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.grid.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.grid.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        for i in range(nfparams):
            self.grid.setHorizontalHeaderItem(i, QtWidgets.QTableWidgetItem(f"{i}"))
            self.grid.setVerticalHeaderItem(i + 2, QtWidgets.QTableWidgetItem(f"{i}"))
        self.grid.setVerticalHeaderItem(0, QtWidgets.QTableWidgetItem("Value:"))
        self.grid.setVerticalHeaderItem(1, QtWidgets.QTableWidgetItem("Error:"))
        self.grid.cellDoubleClicked.connect(self.OnSelectCell)
        grid_layout.addWidget(self.grid, 1)

        check_row = QtWidgets.QHBoxLayout()
        self.normalize_checkbox = QtWidgets.QCheckBox("Normalize value (σ_ij/σ_i/σ_j)")
        self.normalize_checkbox.setChecked(True)
        self.normalize_checkbox.toggled.connect(self.OnToggleNormalize)
        check_row.addWidget(self.normalize_checkbox)
        self.chicorrect_checkbox = QtWidgets.QCheckBox("Correct by sqrt(chi²)")
        self.chicorrect_checkbox.setToolTip(
            "Renormalize the parameter errors with sqrt(chi²).\n"
            "This is helpful if you assume the model is correct and "
            "error bars are only a relative measure.\n"
            "The resulting scaled error bars would then lead to chi²=1."
        )
        self.chicorrect_checkbox.setChecked(False)
        self.chicorrect_checkbox.toggled.connect(self.OnToggleChi2)
        check_row.addWidget(self.chicorrect_checkbox)
        check_row.addStretch(1)
        grid_layout.addLayout(check_row)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self.fom_text = QtWidgets.QLabel("FOM chi²/bars: -")
        font = self.fom_text.font()
        font.setPointSize(int(font.pointSize() * 2.0))
        self.fom_text.setFont(font)
        right_layout.addWidget(self.fom_text)
        self.plot_panel = PlotPanel(right_panel, config_class=StatisticsPanelConfig)
        self.ax = self.plot_panel.figure.add_subplot(111)
        right_layout.addWidget(self.plot_panel, 1)

        main_row.addWidget(left_panel, 0)
        main_row.addWidget(grid_panel, 1)
        main_row.addWidget(right_panel, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(main_row, 1)
        self.run_button = QtWidgets.QPushButton("Run Analysis...")
        self.run_button.clicked.connect(self.OnRunAnalysis)
        layout.addWidget(self.run_button)
        layout.addWidget(self.ptxt)
        layout.addWidget(self.pbar)

        if parent is not None:
            psize = parent.size()
            self.resize(int(psize.width() * 0.75), int(psize.height() * 0.75))

        if prev_result is not None:
            self._res = prev_result
            self.bproblem = prev_result.bproblem
            self.display_bumps()

    def OnRunAnalysis(self):
        if self.thread is not None:
            self.run_button.setText("Run Analysis...")
            if bumps.__version__.startswith("0."):
                self.bproblem.fitness.stop_fit = True
            else:
                self.bproblem.stop_fit = True
            return
        self.thread = threading.Thread(target=self.run_bumps, daemon=True)
        self.thread.start()
        self.run_button.setText("Stop Run")

    def get_bumps_param_names(self):
        if bumps.__version__.startswith("0."):
            return list(self.bproblem.model_parameters().keys())
        return list(self.bproblem.model_parameters()["models"][0].keys())

    def run_bumps(self):
        self.bproblem = self.model.bumps_problem()
        mon = ProgressMonitor(self.bproblem, self.pbar, self.ptxt)
        pop = self.entries["pop"].value()
        burn = self.entries["burn"].value()
        samples = self.entries["samples"].value()
        self.pbar.setRange(0, int(samples / (len(self.get_bumps_param_names()) * pop)) + burn)

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

        QtCore.QTimer.singleShot(0, lambda: self.run_button.setText("Run Analysis..."))
        if mgr.successful:
            self._res = res
            QtCore.QTimer.singleShot(0, self.display_bumps)

    def _set_cell(self, row, col, text, bg=None, read_only=True):
        item = self.grid.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            self.grid.setItem(row, col, item)
        item.setText(text)
        if bg is not None:
            item.setBackground(QtGui.QBrush(QtGui.QColor(bg)))
        if read_only:
            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
        item.setTextAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

    def display_bumps(self):
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread = None
        self.pbar.setValue(0)

        res = self._res
        self.chisq = res.chisq
        if self.chicorrect_checkbox.isChecked():
            scl = sqrt(self.chisq)
        else:
            scl = 1.0

        self.fom_text.setText(f"FOM chi²/bars: {self.chisq:.3f}")
        self.draw = res.state.draw()
        pnames = self.get_bumps_param_names()
        sort_indices = [pnames.index(ni) for ni in self.draw.labels]

        self.abs_cov = res.cov
        self.rel_cov = res.cov / res.dx[:, newaxis] / res.dx[newaxis, :]
        if self.rel_cov.shape[0] > 1:
            rel_max = [0, 1, abs(self.rel_cov[0, 1])]
        else:
            rel_max = [0, 0, 0.0]
        if self.normalize_checkbox.isChecked():
            display_cov = self.rel_cov
            fmt = "%.6f"
        else:
            display_cov = self.abs_cov * scl**2
            fmt = "%.4g"

        for i, ci in enumerate(self.rel_cov):
            plabel = "\n".join(self.draw.labels[i].rsplit("_", 1))
            self.grid.setHorizontalHeaderItem(sort_indices[i], QtWidgets.QTableWidgetItem(plabel))
            self.grid.setVerticalHeaderItem(sort_indices[i] + 2, QtWidgets.QTableWidgetItem(plabel))
            self._set_cell(0, sort_indices[i], "%.8g" % res.x[i], bg="#cccccc")
            self._set_cell(1, sort_indices[i], "%.4g" % (res.dx[i] * scl), bg="#cccccc")
            for j, cj in enumerate(ci):
                bg = None
                if i == j:
                    bg = "#888888"
                elif abs(cj) > 0.4:
                    bg = "#ffcccc"
                elif abs(cj) > 0.3:
                    bg = "#ffdddd"
                elif abs(cj) > 0.2:
                    bg = "#ffeeee"
                self._set_cell(sort_indices[i] + 2, sort_indices[j], fmt % display_cov[i, j], bg=bg)
                if i != j and abs(cj) > rel_max[2]:
                    rel_max = [min(i, j), max(i, j), abs(cj)]

        if self.rel_cov.shape[0] > 1:
            self.hists = _hists(self.draw.points.T, bins=50)
            start_row = sort_indices[rel_max[0]] + 2
            start_col = sort_indices[rel_max[1]]
            self.grid.setRangeSelected(
                QtWidgets.QTableWidgetSelectionRange(start_row, start_col, start_row, start_col), True
            )
            self.plot_histogram(rel_max[0], rel_max[1])

        exdict = {}
        exdict["library"] = "bumps"
        exdict["version"] = bumps_version
        exdict["settings"] = dict(
            pop=self.entries["pop"].value(),
            burn=self.entries["burn"].value(),
            samples=self.entries["samples"].value(),
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

    def OnToggleNormalize(self):
        if self.rel_cov is None:
            return
        if self.normalize_checkbox.isChecked():
            display_cov = self.rel_cov
            fmt = "%.6f"
        else:
            display_cov = self.abs_cov
            if self.chicorrect_checkbox.isChecked():
                display_cov = display_cov * self.chisq
            fmt = "%.4g"
        pnames = self.get_bumps_param_names()
        sort_indices = [pnames.index(ni) for ni in self.draw.labels]
        for i, ci in enumerate(self.rel_cov):
            for j, _cj in enumerate(ci):
                self._set_cell(sort_indices[i] + 2, sort_indices[j], fmt % display_cov[i, j])

    def OnToggleChi2(self):
        if self.rel_cov is None:
            return
        if self.chicorrect_checkbox.isChecked():
            scl = sqrt(self.chisq)
        else:
            scl = 1.0
        pnames = self.get_bumps_param_names()
        sort_indices = [pnames.index(ni) for ni in self.draw.labels]
        res = self._res
        for i, dxi in enumerate(res.dx):
            self._set_cell(1, sort_indices[i], "%.4g" % (dxi * scl), bg="#cccccc")

        if (res.dxpm[:, 0] > 0.0).any() or (res.dxpm[:, 1] < 0.0).any():
            dxpm = array([-res.dx, res.dx]).T * scl
        else:
            dxpm = res.dxpm * scl
        reverse_sort = [sort_indices.index(i) for i in range(len(sort_indices))]
        error_labels = ["(%.3e, %.3e)" % (dxup, dxdown) for dxup, dxdown in dxpm[reverse_sort]]
        self.model.parameters.set_error_pars(error_labels)

        if self.normalize_checkbox.isChecked():
            return
        display_cov = self.abs_cov
        if self.chicorrect_checkbox.isChecked():
            display_cov = display_cov * self.chisq
        fmt = "%.4g"
        for i, ci in enumerate(self.rel_cov):
            for j, _cj in enumerate(ci):
                self._set_cell(sort_indices[i] + 2, sort_indices[j], fmt % display_cov[i, j])

    def OnSelectCell(self, row, col):
        ri, rj = col, row - 2
        pnames = self.get_bumps_param_names()
        reverse_indices = [self.draw.labels.index(ni) for ni in pnames]
        if rj < 0:
            return
        i = reverse_indices[ri]
        j = reverse_indices[rj]
        if i == j:
            return
        if i > j:
            i, j = j, i
        self.plot_histogram(i, j)

    def plot_histogram(self, i, j):
        fig = self.plot_panel.figure
        fig.clear()
        ax = fig.add_subplot(111)
        data, x, y = self.hists[(i, j)]
        vmin, vmax = data[data > 0].min(), data.max()
        ax.pcolorfast(y, x, maximum(vmin, data), norm=LogNorm(vmin, vmax), cmap="inferno")
        p1 = data.sum(axis=1)
        p1 = 0.5 * p1 / p1.max() * (y[-1] - y[0]) + y[0]
        p1x = (x[:-1] + x[1:]) / 2.0
        p2 = data.sum(axis=0)
        p2 = 0.5 * p2 / p2.max() * (x[-1] - x[0]) + x[0]
        p2y = (y[:-1] + y[1:]) / 2.0
        ax.plot(p1, p1x, color="blue")
        ax.plot(p2y, p2, color="blue")
        ax.set_xlabel(self.draw.labels[j])
        ax.set_ylabel(self.draw.labels[i])
        self.plot_panel.flush_plot()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.thread is not None and self.thread.is_alive():
            self.ptxt.setText("Simulation running")
            if bumps.__version__.startswith("0."):
                self.bproblem.fitness.stop_fit = True
            else:
                self.bproblem.stop_fit = True
            self.thread.join(timeout=5.0)
        super().closeEvent(event)
