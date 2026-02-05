"""
Qt port of the wx-based parametergrid.
Provides a QTableWidget-driven editor for model parameters with
signals mirroring the solver control hooks.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .. import parameters
from ..core.config import BaseConfig, Configurable


@dataclass
class ParameterGridConfig(BaseConfig):
    section = "parameter grid"
    value_slider: bool = False


class ParameterGrid(Configurable, QtWidgets.QWidget):
    opt: ParameterGridConfig

    set_parameter_value = QtCore.Signal(int, int, object)
    move_parameter = QtCore.Signal(int, int)
    insert_parameter = QtCore.Signal(int)
    delete_parameters = QtCore.Signal(list)
    sort_and_group_parameters = QtCore.Signal(object)
    grid_changed = QtCore.Signal(bool)
    value_change = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        QtWidgets.QWidget.__init__(self, parent)
        Configurable.__init__(self)
        self.opt.load_config()

        self._pars = parameters.Parameters()
        self._updating = False
        self._eval_func: Optional[Callable[[str], object]] = None
        self._project_func: Optional[Callable[[int], None]] = None
        self._scan_func: Optional[Callable[[int], None]] = None

        self._build_ui()
        self._populate_from_pars()
        self._apply_value_editor()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.toolbar = QtWidgets.QToolBar(self)
        layout.addWidget(self.toolbar, 0)

        self._act_add = self.toolbar.addAction("Add")
        self._act_delete = self.toolbar.addAction("Delete")
        self._act_up = self.toolbar.addAction("Move Up")
        self._act_down = self.toolbar.addAction("Move Down")
        self.toolbar.addSeparator()
        self._act_sort = self.toolbar.addAction("Sort")
        self._act_sort_name = self.toolbar.addAction("Sort Name")
        self.toolbar.addSeparator()
        self._act_project = self.toolbar.addAction("Project FOM")
        self._act_scan = self.toolbar.addAction("Scan FOM")

        self._act_add.triggered.connect(self._on_add_row)
        self._act_delete.triggered.connect(self._on_delete_rows)
        self._act_up.triggered.connect(lambda: self._on_move_row(-1))
        self._act_down.triggered.connect(lambda: self._on_move_row(1))
        self._act_sort.triggered.connect(lambda: self._on_sort(parameters.SortSplitItem.ATTRIBUTE))
        self._act_sort_name.triggered.connect(lambda: self._on_sort(parameters.SortSplitItem.OBJ_NAME))
        self._act_project.triggered.connect(self._on_project_fom)
        self._act_scan.triggered.connect(self._on_scan_fom)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(self._pars.get_col_headers())
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table, 1)

        self._slider_delegate = SliderValueDelegate(self.table)

    def _populate_from_pars(self) -> None:
        self._updating = True
        try:
            self.table.setRowCount(self._pars.get_len_rows())
            for row, row_data in enumerate(self._pars.get_data()):
                for col, value in enumerate(row_data):
                    if col == 2:
                        item = QtWidgets.QTableWidgetItem("")
                        item.setFlags(
                            QtCore.Qt.ItemFlag.ItemIsEnabled
                            | QtCore.Qt.ItemFlag.ItemIsSelectable
                            | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                        )
                        item.setCheckState(QtCore.Qt.CheckState.Checked if value else QtCore.Qt.CheckState.Unchecked)
                    else:
                        item = QtWidgets.QTableWidgetItem(self._format_value(col, value))
                        item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(row, col, item)
            self._update_row_labels()
        finally:
            self._updating = False
        self._apply_value_editor()

    def _format_value(self, col: int, value: object) -> str:
        if col in (1, 3, 4) and isinstance(value, (float, int)):
            return f"{value:.7g}"
        return str(value)

    def _update_row_labels(self) -> None:
        count = 0
        for row in range(self.table.rowCount()):
            fit_item = self.table.item(row, 2)
            fit = bool(fit_item and fit_item.checkState() == QtCore.Qt.CheckState.Checked)
            if fit:
                count += 1
                label = str(count)
            else:
                label = "-"
            self.table.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(label))

    def _emit_grid_changed(self, permanent_change: bool = True) -> None:
        self.grid_changed.emit(permanent_change)

    def _on_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._updating or item is None:
            return
        row, col = item.row(), item.column()
        if row < 0 or col < 0:
            return
        value = self._item_value(row, col)
        self._set_par_value(row, col, value)
        self.set_parameter_value.emit(row, col, value)
        if col == 2:
            self._update_row_labels()
        if col == 1:
            self.value_change.emit()
        self._emit_grid_changed(permanent_change=True)

    def _item_value(self, row: int, col: int) -> object:
        item = self.table.item(row, col)
        if col == 2:
            return bool(item and item.checkState() == QtCore.Qt.CheckState.Checked)
        text = item.text().strip() if item else ""
        if col in (1, 3, 4):
            try:
                return float(text)
            except ValueError:
                return 0.0
        return text

    def _set_par_value(self, row: int, col: int, value: object) -> None:
        while row >= self._pars.get_len_rows():
            self._pars.append()
        self._pars.set_value(row, col, value)

    def _current_row(self) -> int:
        row = self.table.currentRow()
        return row if row >= 0 else self.table.rowCount() - 1

    def _on_add_row(self) -> None:
        row = self._current_row()
        if row < 0:
            row = 0
        self._pars.insert_row(row)
        self.table.insertRow(row)
        for col in range(self.table.columnCount()):
            value = self._pars.get_value(row, col)
            if col == 2:
                item = QtWidgets.QTableWidgetItem("")
                item.setFlags(
                    QtCore.Qt.ItemFlag.ItemIsEnabled
                    | QtCore.Qt.ItemFlag.ItemIsSelectable
                    | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(QtCore.Qt.CheckState.Checked if value else QtCore.Qt.CheckState.Unchecked)
            else:
                item = QtWidgets.QTableWidgetItem(self._format_value(col, value))
            self.table.setItem(row, col, item)
        self._update_row_labels()
        self.insert_parameter.emit(row)
        self._emit_grid_changed(permanent_change=True)

    def _on_delete_rows(self) -> None:
        rows = [idx.row() for idx in self.table.selectionModel().selectedRows()]
        if not rows:
            row = self._current_row()
            if row >= 0:
                rows = [row]
        rows = sorted(set(rows))
        if not rows:
            return
        self._pars.delete_rows(rows)
        for row in reversed(rows):
            self.table.removeRow(row)
        self._update_row_labels()
        self.delete_parameters.emit(rows)
        self._emit_grid_changed(permanent_change=True)

    def _on_move_row(self, step: int) -> None:
        row = self._current_row()
        if row < 0 or not self._pars.can_move_row(row, step):
            return
        target = row + step
        self._pars.move_row(row, step)
        self._swap_rows(row, target)
        self.table.setCurrentCell(target, 0)
        self._update_row_labels()
        self.move_parameter.emit(row, step)
        self._emit_grid_changed(permanent_change=True)

    def _swap_rows(self, row_a: int, row_b: int) -> None:
        for col in range(self.table.columnCount()):
            item_a = self.table.takeItem(row_a, col)
            item_b = self.table.takeItem(row_b, col)
            self.table.setItem(row_a, col, item_b)
            self.table.setItem(row_b, col, item_a)

    def _on_sort(self, sort_params: parameters.SortSplitItem) -> None:
        self._pars.sort_rows(sort_params=sort_params)
        self._populate_from_pars()
        self.sort_and_group_parameters.emit(sort_params)
        self._emit_grid_changed(permanent_change=True)

    def _on_project_fom(self) -> None:
        if self._project_func is None:
            return
        row = self._current_row()
        if row >= 0:
            self._project_func(row)

    def _on_scan_fom(self) -> None:
        if self._scan_func is None:
            return
        row = self._current_row()
        if row >= 0:
            self._scan_func(row)

    def SetParameters(self, pars, clear: bool = True, permanent_change: bool = True) -> None:
        self._pars = pars.copy() if hasattr(pars, "copy") else pars
        self._populate_from_pars()
        self._emit_grid_changed(permanent_change=permanent_change)

    def ShowParameters(self, values) -> None:
        self._updating = True
        try:
            idx = 0
            for row, row_data in enumerate(self._pars.get_data()):
                if row_data[2] and row_data[0] != "":
                    if idx >= len(values):
                        break
                    value = values[idx]
                    idx += 1
                    item = self.table.item(row, 1)
                    if item is None:
                        item = QtWidgets.QTableWidgetItem("")
                        self.table.setItem(row, 1, item)
                    item.setText(self._format_value(1, value))
        finally:
            self._updating = False
        self._emit_grid_changed(permanent_change=False)

    def GetParameters(self):
        return self._pars

    def SetFOMFunctions(self, projectfunc, scanfunc) -> None:
        self._project_func = projectfunc
        self._scan_func = scanfunc

    def SetEvalFunc(self, func) -> None:
        self._eval_func = func

    def SetParameterSelections(self, _par_dict) -> None:
        pass

    def SetValueEditorSlider(self, active: bool) -> None:
        self.opt.value_slider = bool(active)
        self.opt.save_config(default=True)
        self._apply_value_editor()

    def _apply_value_editor(self) -> None:
        if self.opt.value_slider:
            self.table.setItemDelegateForColumn(1, self._slider_delegate)
        else:
            self.table.setItemDelegateForColumn(1, None)


class SliderValueDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if index.column() != 1:
            return super().createEditor(parent, option, index)
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, parent)
        slider.setMinimum(0)
        slider.setMaximum(1000)
        return slider

    def setEditorData(self, editor, index):
        if not isinstance(editor, QtWidgets.QSlider):
            return super().setEditorData(editor, index)
        row = index.row()
        min_val, max_val = self._get_min_max(index)
        value = self._to_float(index.data())
        editor.setProperty("min_val", min_val)
        editor.setProperty("max_val", max_val)
        if max_val <= min_val:
            editor.setValue(0)
            return
        value = max(min_val, min(max_val, value))
        pos = int((value - min_val) / (max_val - min_val) * editor.maximum())
        editor.setValue(pos)

    def setModelData(self, editor, model, index):
        if not isinstance(editor, QtWidgets.QSlider):
            return super().setModelData(editor, model, index)
        min_val = editor.property("min_val")
        max_val = editor.property("max_val")
        if min_val is None or max_val is None or max_val <= min_val:
            return
        value = min_val + (max_val - min_val) * (editor.value() / editor.maximum())
        model.setData(index, value)

    def updateEditorGeometry(self, editor, option, index):
        editor.setGeometry(option.rect)

    def _get_min_max(self, index):
        model = index.model()
        min_val = self._to_float(model.index(index.row(), 3).data())
        max_val = self._to_float(model.index(index.row(), 4).data())
        return min_val, max_val

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0
