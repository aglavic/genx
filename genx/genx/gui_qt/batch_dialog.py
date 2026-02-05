"""
Qt dialog for batch fitting over multiple datasets.
Port of gui_wx.batch_dialog.
"""

from copy import copy, deepcopy

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6 import QtCore, QtGui, QtWidgets

from ..plugins.data_loader_framework import Template as DataLoaderTemplate
from .metadata_dialog import MetaDataDialog
from .solvergui import ModelControlGUI


def get_value_from_path(data, path):
    if len(path) > 1:
        return get_value_from_path(data[path[0]], path[1:])
    return data[path[0]]


class CopyTable(QtWidgets.QTableWidget):
    """Table that copies all data on Ctrl+C, regardless of selection."""

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ControlModifier and event.key() == QtCore.Qt.Key.Key_C:
            self.copy()
            return
        super().keyPressEvent(event)

    def copy(self) -> None:
        output = ""
        for col in range(self.columnCount()):
            header = self.horizontalHeaderItem(col)
            name = header.text() if header is not None else ""
            output += f"{name.replace('\n', '.')}\t"
        output += "\n"
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                item = self.item(row, col)
                value = item.text() if item is not None else ""
                output += f"{value}\t"
            output += "\n"
        QtWidgets.QApplication.clipboard().setText(output)


class PlotDialog(QtWidgets.QDialog):
    """Minimal matplotlib dialog for batch parameter traces."""

    def __init__(self, parent, title: str):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.figure = Figure(figsize=(5.0, 4.0), dpi=96)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas, 1)

    def clear_data(self):
        self.ax.cla()

    def plot(self, *args, **kwargs):
        return self.ax.plot(*args, **kwargs)

    def draw(self):
        self.canvas.draw()

    def set_xlabel(self, text):
        self.ax.set_xlabel(text)

    def set_ylabel(self, text):
        self.ax.set_ylabel(text)


class BatchDialog(QtWidgets.QDialog):
    """
    Dialog to display dataset list and run batch fits.
    """

    model_control: ModelControlGUI

    def __init__(self, parent, model_control: ModelControlGUI):
        super().__init__(parent)
        self.setWindowTitle("Model/dataset list for batch fitting")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowMaximizeButtonHint)

        self.model_control = model_control
        self.model_control.batch_next.connect(self.OnBatchNext)
        self.plots = {}
        self._update_guard = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(
            QtWidgets.QLabel(
                "Below is the list of datasets for batch fitting.\n"
                "If you want to activate one dataset, double-click the row label on the left."
            )
        )

        self.grid = CopyTable(self)
        self.grid.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.grid.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.grid.itemChanged.connect(self.OnUpdateCell)
        self.grid.verticalHeader().sectionDoubleClicked.connect(self.OnLabelDClick)
        layout.addWidget(self.grid, 1)
        self.build_grid()

        self.keep_last = QtWidgets.QCheckBox("Keep result values from last")
        self.keep_last.setChecked(self.model_control.batch_options.keep_last)
        layout.addWidget(self.keep_last)

        self.adjust_bounds = QtWidgets.QCheckBox("Adjust boundaries around last values")
        self.adjust_bounds.setChecked(self.model_control.batch_options.adjust_bounds)
        layout.addWidget(self.adjust_bounds)

        button_row = QtWidgets.QHBoxLayout()
        layout.addLayout(button_row)

        btn = QtWidgets.QPushButton("Import Data...")
        btn.setToolTip("Use active dataset as template and load\na list of datasets to generate a batch.")
        btn.clicked.connect(self.OnImportData)
        button_row.addWidget(btn)

        btn = QtWidgets.QPushButton("Clear list")
        btn.clicked.connect(self.OnClearData)
        button_row.addWidget(btn)

        button_row.addSpacing(20)

        btn = QtWidgets.QPushButton("Fit All")
        btn.clicked.connect(self.OnBatchFitModel)
        button_row.addWidget(btn)

        btn = QtWidgets.QPushButton("Fit From Here")
        btn.clicked.connect(self.OnBatchFromHere)
        button_row.addWidget(btn)

        button_row.addSpacing(20)

        btn = QtWidgets.QPushButton("Sort by Value")
        btn.clicked.connect(self.OnSort)
        button_row.addWidget(btn)

        btn = QtWidgets.QPushButton("Extract Value")
        btn.setToolTip("Use metadata from datasets to extract a sequence value (e.g. temperature)")
        btn.clicked.connect(self.OnExtract)
        button_row.addWidget(btn)

    def _set_item(self, row: int, col: int, text: str, editable: bool = True, bg=None) -> None:
        self._update_guard = True
        item = self.grid.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem()
            self.grid.setItem(row, col, item)
        item.setText(text)
        flags = item.flags()
        if editable:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        else:
            flags &= ~QtCore.Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        if bg is not None:
            item.setBackground(QtGui.QBrush(QtGui.QColor(bg)))
        self._update_guard = False

    def OnBatchNext(self, evt):
        self.switch_line(evt.last_index, self.model_control.controller.active_index())
        prev_model = self.model_control.controller.model_store[evt.last_index]
        prev_models = self.model_control.controller.model_store[: evt.last_index + 1]

        row_nmb, funcs, values, _min, _max = prev_model.parameters.get_fit_pars()
        x = np.array([mi.sequence_value for mi in prev_models])

        for i, (ni, fi, pi) in enumerate(zip(row_nmb, funcs, values)):
            self._set_item(evt.last_index, i + 2, f"{pi:.7g}", editable=False)
            if fi in self.plots:
                vals = np.array([mi.parameters[ni].value for mi in prev_models])
                plot = self.plots[fi]
                plot.clear_data()
                plot.plot(x, vals, marker="o", ms=2, lw=1, color=(0.0, 0.0, 1.0))
                plot.draw()

    def switch_line(self, prev_row, row):
        if prev_row >= 0:
            for col in (0, 1):
                item = self.grid.item(prev_row, col)
                self._set_item(prev_row, col, item.text() if item is not None else "", bg="#ffffff")
        if row >= 0:
            for col in (0, 1):
                item = self.grid.item(row, col)
                self._set_item(row, col, item.text() if item is not None else "", bg="#ff8888")
            if self.grid.item(row, 0) is not None:
                self.grid.scrollToItem(self.grid.item(row, 0))

    def build_grid(self):
        models = self.model_control.controller.model_store
        self.grid.setRowCount(len(models))
        self.grid.setColumnCount(2)
        self.grid.setHorizontalHeaderLabels(["item", "index/value"])
        self.fill_grid()

    def fill_grid(self):
        models = self.model_control.controller.model_store
        ci = self.model_control.controller.active_index()
        for i, mi in enumerate(models):
            self.grid.setVerticalHeaderItem(i, QtWidgets.QTableWidgetItem(f"{i}:"))
            self._set_item(i, 0, mi.h5group_name)
            self._set_item(i, 1, f"{mi.sequence_value:.7g}")
            if i == ci:
                self._set_item(i, 0, mi.h5group_name, bg="#ff8888")
                self._set_item(i, 1, f"{mi.sequence_value:.7g}", bg="#ff8888")
            else:
                self._set_item(i, 0, mi.h5group_name, bg="#ffffff")
                self._set_item(i, 1, f"{mi.sequence_value:.7g}", bg="#ffffff")
        self.grid.resizeColumnsToContents()

    def OnUpdateCell(self, item: QtWidgets.QTableWidgetItem):
        if self._update_guard:
            return
        row, col = item.row(), item.column()
        if col == 0:
            names = [m.h5group_name for m in self.model_control.controller.model_store]
            new_name = item.text()
            old_name = self.model_control.controller.model_store[row].h5group_name
            if new_name in names and new_name != old_name:
                self._set_item(row, col, old_name)
                return
            self.model_control.controller.model_store[row].h5group_name = new_name
        elif col == 1:
            old_value = self.model_control.controller.model_store[row].sequence_value
            try:
                value = float(item.text())
            except ValueError:
                self._set_item(row, col, f"{old_value:.7g}")
                return
            self.model_control.controller.model_store[row].sequence_value = value
        else:
            return

    def OnLabelDClick(self, row):
        ci = self.model_control.controller.active_index()
        self.model_control.controller.activate_model(row)
        self.switch_line(ci, row)

    def set_batch_params(self):
        self.model_control.batch_running = True
        self.model_control.batch_options.keep_last = self.keep_last.isChecked()
        self.model_control.batch_options.adjust_bounds = self.adjust_bounds.isChecked()
        self.set_batch_columns()

    def set_batch_columns(self):
        if self.grid.columnCount() > 2:
            for col in range(self.grid.columnCount() - 1, 1, -1):
                self.grid.removeColumn(col)
        row_nmb, funcs, _values, _min, _max = self.model_control.get_parameters().get_fit_pars()
        for i, name in enumerate(funcs):
            self.grid.insertColumn(2 + i)
            self.grid.setHorizontalHeaderItem(2 + i, QtWidgets.QTableWidgetItem(name))
            for row in range(self.grid.rowCount()):
                self._set_item(row, 2 + i, "", editable=False)
            self.create_graph(name)

    def create_graph(self, parameter):
        if parameter in self.plots:
            self.plots[parameter].show()
            self.plots[parameter].raise_()
            return
        plot = PlotDialog(self, title=f"Batch parameter {parameter} fit")
        self.plots[parameter] = plot
        plot.set_xlabel("batch index/value")
        plot.set_ylabel(parameter)
        plot.draw()
        plot.destroyed.connect(lambda _=None, p=parameter: self.plots.pop(p, None))
        plot.show()

    def _qt_filter(self, wildcard: str) -> str:
        parts = wildcard.split("|")
        if len(parts) < 2:
            return wildcard
        labels = [parts[i] for i in range(0, len(parts), 2)]
        return ";;".join(labels)

    def OnImportData(self):
        origin = self.model_control.get_model().deepcopy()
        data_list_ctrl = getattr(getattr(self.model_control.parent, "ui", None), "dataListControl", None)
        if data_list_ctrl is None:
            return
        data_loader: DataLoaderTemplate = data_list_ctrl.list_ctrl.data_loader
        if data_loader is None:
            return
        flists = []
        for i in range(len(origin.data)):
            filters = self._qt_filter(data_loader.GetWildcardString())
            files, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self,
                f"Select datafiles for dataset {i}",
                "",
                filters,
            )
            if not files:
                return
            flists.append(files)

        N = len(flists[0])
        prog = QtWidgets.QProgressDialog("Importing...", f"Read from files...\n0/{N}", 0, 100, self)
        prog.setWindowTitle("Importing...")
        prog.setWindowModality(QtCore.Qt.WindowModality.ApplicationModal)

        def update_callback(i):
            prog.setValue(int(i / N * 100))
            prog.setLabelText(f"Read from files...\n{i}/{N}")
            QtWidgets.QApplication.processEvents()

        try:
            self.model_control.controller.read_sequence(data_loader, flists, name_by_file=True, callback=update_callback)
        finally:
            prog.close()

        rows_needed = len(self.model_control.controller.model_store)
        if rows_needed > self.grid.rowCount():
            self.grid.setRowCount(rows_needed)
        self.fill_grid()

    def OnClearData(self):
        self.model_control.controller.model_store = []
        self.grid.setRowCount(0)

    def OnBatchFitModel(self):
        script = self.model_control.get_model_script()
        params = self.model_control.get_model_params()
        for mi in self.model_control.controller.model_store:
            mi.set_script(copy(script))
            mi.parameters = deepcopy(params)
        self.switch_line(self.model_control.controller.active_index(), 0)
        self.model_control.controller.activate_model(0)
        self.set_batch_params()
        QtCore.QTimer.singleShot(1000, self.model_control.StartFit)

    def OnBatchFromHere(self):
        script = self.model_control.get_model_script()
        params = self.model_control.get_model_params()
        ci = self.model_control.controller.active_index()
        for mi in self.model_control.controller.model_store[ci + 1 :]:
            mi.set_script(copy(script))
            mi.parameters = deepcopy(params)
        self.set_batch_params()
        fit_rows, *_ = self.model_control.get_parameters().get_fit_pars()
        for i, mi in enumerate(self.model_control.controller.model_store[: ci + 1]):
            params = [mi.parameters[fi].value for fi in fit_rows]
            for j, pj in enumerate(params):
                self._set_item(i, j + 2, f"{pj:.7g}", editable=False)
        QtCore.QTimer.singleShot(1000, self.model_control.StartFit)

    def OnSort(self):
        sort_list = [(mi.sequence_value, i, mi) for i, mi in enumerate(self.model_control.controller.model_store)]
        sort_list.sort()
        self.model_control.controller.model_store = [si[2] for si in sort_list]
        self.fill_grid()

    def OnExtract(self):
        if not self.model_control.controller.model_store:
            return
        dia = MetaDataDialog(
            self,
            self.model_control.controller.model_store[0].data,
            filter_leaf_types=[int, float],
            close_on_activate=True,
        )
        dia.setWindowTitle("Select key to use by double-click")
        dia.exec()
        if dia.activated_leaf is None:
            return

        didx = dia.activated_leaf[0]
        dpath = dia.activated_leaf[1:]
        for mi in self.model_control.controller.model_store:
            try:
                value = float(get_value_from_path(mi.data[didx].meta, dpath))
            except Exception:
                value = 0.0
            mi.sequence_value = value
        self.fill_grid()
