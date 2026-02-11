"""
Qt port of the wx-based parametergrid.
Provides a QTableWidget-driven editor for model parameters with
signals mirroring the solver control hooks.
"""

from dataclasses import dataclass
from typing import Callable, Optional

from PySide6 import QtCore, QtGui, QtWidgets

from .custom_events import UpdateParametersEvent
from .. import parameters
from ..core.config import BaseConfig, Configurable
from .parametergrid_ui import Ui_ParameterGrid


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
        self._simulate_func: Optional[Callable[[], None]] = None
        self._par_dict = {}
        self._variable_span = 0.2

        self._build_ui()
        self._populate_from_pars()
        self._apply_value_editor()

    def UpdateConfigValues(self) -> None:
        if not hasattr(self, "table"):
            return
        self._apply_value_editor()
        self._sync_value_slider_actions()

    def _build_ui(self) -> None:
        self.ui = Ui_ParameterGrid()
        self.ui.setupUi(self)
        self._scope_action_object_names()
        self.toolbar = self.ui.toolbar
        self.table = self.ui.parameterTable
        self._action_value_slider = None

        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(self._pars.get_col_headers())
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        for col in range(self.table.columnCount()):
            if col != 2:
                header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.Interactive)
        self.table.verticalHeader().setVisible(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.itemChanged.connect(self._on_item_changed)
        self.table.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)

        self._value_bar_delegate = ValueBarDelegate(self.table)
        self._opaque_delegate = OpaqueItemDelegate(self.table)
        self._column_weights = {0: 3, 1: 2, 3: 1, 4: 1, 5: 1}
        self._update_column_widths()
        self._setup_value_slider_action()

    def _setup_value_slider_action(self) -> None:
        action = QtGui.QAction(self)
        action.setObjectName("actionValueAsSlider_paramgrid")
        action.setCheckable(True)
        action.setChecked(self.opt.value_slider)
        action.setText("Show sliders")
        action.setToolTip("Show the parameter values as sliders")
        action.setIcon(QtGui.QIcon(":/main_gui/slider.png"))
        action.toggled.connect(self._on_value_slider_toggled)
        project_action = getattr(self.ui, "actionProjectFom", None)
        if project_action is not None:
            self.toolbar.insertAction(project_action, action)
        else:
            self.toolbar.addAction(action)
        self._action_value_slider = action

    def _on_value_slider_toggled(self, checked: bool) -> None:
        self.SetValueEditorSlider(bool(checked))

    def _scope_action_object_names(self) -> None:
        # Avoid QMainWindow auto-connecting to these actions (parameter grid owns them).
        actions = (
            "actionAddRow",
            "actionDeleteRow",
            "actionMoveUp",
            "actionMoveDown",
            "actionSort",
            "actionSortName",
            "actionProjectFom",
            "actionScanFom",
        )
        for name in actions:
            action = getattr(self.ui, name, None)
            if action is not None:
                action.setObjectName(f"{name}_paramgrid")

    @QtCore.Slot()
    def on_actionAddRow_triggered(self) -> None:
        self._on_add_row()

    @QtCore.Slot()
    def on_actionDeleteRow_triggered(self) -> None:
        self._on_delete_rows()

    @QtCore.Slot()
    def on_actionMoveUp_triggered(self) -> None:
        self._on_move_row(-1)

    @QtCore.Slot()
    def on_actionMoveDown_triggered(self) -> None:
        self._on_move_row(1)

    @QtCore.Slot()
    def on_actionSort_triggered(self) -> None:
        self._on_sort(parameters.SortSplitItem.ATTRIBUTE)

    @QtCore.Slot()
    def on_actionSortName_triggered(self) -> None:
        self._on_sort(parameters.SortSplitItem.OBJ_NAME)

    @QtCore.Slot()
    def on_actionProjectFom_triggered(self) -> None:
        self._on_project_fom()

    @QtCore.Slot()
    def on_actionScanFom_triggered(self) -> None:
        self._on_scan_fom()

    @QtCore.Slot()
    def on_update_parameters(self, event: UpdateParametersEvent):
        self.ShowParameters(event.values, event.permanent_change)

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
        self._update_column_widths()
        self._apply_value_editor()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._update_column_widths()

    def _update_column_widths(self) -> None:
        if self.table.columnCount() == 0:
            return
        header = self.table.horizontalHeader()
        fit_col = 2
        fit_width = header.sectionSize(fit_col)
        available = self.table.viewport().width() - fit_width
        if available <= 0:
            return
        total_weight = sum(self._column_weights.values())
        for col, weight in self._column_weights.items():
            width = max(40, int(available * weight / total_weight))
            self.table.setColumnWidth(col, width)

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
        self._populate_from_pars()
        self.table.setCurrentCell(row, 0)
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
        self._populate_from_pars()
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

    @QtCore.Slot(object)
    def on_model_loaded(self, model) -> None:
        self.ReadConfig()
        self.SetParameters(model.parameters, clear=False, permanent_change=False)

    def ShowParameters(self, values, permanent_change=False) -> None:
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
        self._emit_grid_changed(permanent_change=permanent_change)

    def GetParameters(self):
        return self._pars

    def SetFOMFunctions(self, projectfunc, scanfunc) -> None:
        self._project_func = projectfunc
        self._scan_func = scanfunc

    def SetEvalFunc(self, func) -> None:
        self._eval_func = func

    def SetParameterSelections(self, _par_dict) -> None:
        self._par_dict = _par_dict or {}

    def SetValueEditorSlider(self, active: bool) -> None:
        self.opt.value_slider = bool(active)
        self.opt.save_config(default=True)
        self._apply_value_editor()
        self._sync_value_slider_actions()

    def _sync_value_slider_actions(self) -> None:
        action = self._action_value_slider
        if action is not None:
            action.blockSignals(True)
            action.setChecked(self.opt.value_slider)
            action.blockSignals(False)
        window = self.window()
        if window is not None:
            menu_action = window.findChild(QtGui.QAction, "actionValueAsSlider")
            if menu_action is not None:
                menu_action.blockSignals(True)
                menu_action.setChecked(self.opt.value_slider)
                menu_action.blockSignals(False)

    def SetSimulateFunc(self, func: Optional[Callable[[], None]]) -> None:
        self._simulate_func = func

    def _on_context_menu(self, pos: QtCore.QPoint) -> None:
        index = self.table.indexAt(pos)
        if not index.isValid() or index.column() != 0:
            return
        row = index.row()
        self.table.setCurrentCell(row, 0)

        menu = QtWidgets.QMenu(self)
        if not self._par_dict:
            simulate_action = menu.addAction("Simulate to see parameters")
            simulate_action.triggered.connect(self._on_simulate_request)
        else:
            self._populate_parameter_menu(menu, self._par_dict)

        menu.addSeparator()
        manual_action = menu.addAction("Manual Edit")
        manual_action.triggered.connect(lambda: self._edit_cell(row, 0))
        common_action = menu.addAction("Common pars")
        common_action.triggered.connect(self._on_common_pars)

        menu.exec(self.table.viewport().mapToGlobal(pos))

    def _populate_parameter_menu(self, menu: QtWidgets.QMenu, par_dict: dict) -> None:
        for cl in sorted(par_dict.keys(), key=str.lower):
            clmenu = QtWidgets.QMenu(cl, menu)
            entry = par_dict[cl]
            if isinstance(entry, dict):
                for obj in sorted(entry.keys(), key=str.lower):
                    obj_menu = QtWidgets.QMenu(obj, clmenu)
                    funcs = entry[obj]
                    for func in sorted(funcs, key=str.lower):
                        text = f"{obj}.{func}"
                        action = obj_menu.addAction(text)
                        action.triggered.connect(lambda _=False, t=text: self._apply_parameter_choice(t))
                    clmenu.addMenu(obj_menu)
            elif isinstance(entry, list):
                for obj in sorted(entry, key=str.lower):
                    action = clmenu.addAction(obj)
                    action.triggered.connect(lambda _=False, t=obj: self._apply_parameter_choice(t))
            menu.addMenu(clmenu)

    def _apply_parameter_choice(self, text: str) -> None:
        row = self._current_row()
        if row < 0:
            return
        self._set_cell_value(row, 0, text)
        value = self._evaluate_parameter_value(text)
        if value is None:
            return
        self._set_cell_value(row, 1, value)
        minval = value * (1 - self._variable_span)
        maxval = value * (1 + self._variable_span)
        self._set_cell_value(row, 3, min(minval, maxval))
        self._set_cell_value(row, 4, max(minval, maxval))

    def _evaluate_parameter_value(self, text: str):
        if self._eval_func is None:
            return None
        try:
            if ".set" in text:
                getter = text.replace(".set", ".get", 1)
                result = self._eval_func(getter)
                return result() if callable(result) else result
            result = self._eval_func(f"{text}.value")
            return result() if callable(result) else result
        except Exception:
            return None

    def _set_cell_value(self, row: int, col: int, value: object) -> None:
        self._updating = True
        try:
            item = self.table.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem("")
                self.table.setItem(row, col, item)
            if col == 2:
                item.setCheckState(QtCore.Qt.CheckState.Checked if value else QtCore.Qt.CheckState.Unchecked)
            else:
                item.setText(self._format_value(col, value))
            self._set_par_value(row, col, value)
            self.set_parameter_value.emit(row, col, value)
        finally:
            self._updating = False

    def _edit_cell(self, row: int, col: int) -> None:
        item = self.table.item(row, col)
        if item is None:
            item = QtWidgets.QTableWidgetItem("")
            self.table.setItem(row, col, item)
        self.table.editItem(item)

    def _on_simulate_request(self) -> None:
        if self._simulate_func:
            self._simulate_func()

    def _on_common_pars(self) -> None:
        QtWidgets.QMessageBox.information(self, "Common pars", "Common parameter insertion is not available yet.")

    def _apply_value_editor(self) -> None:
        self.table.setItemDelegate(self._opaque_delegate)
        self._value_bar_delegate.set_use_slider(self.opt.value_slider)
        self.table.setItemDelegateForColumn(1, self._value_bar_delegate)

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


class OpaqueItemDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        editor = super().createEditor(parent, option, index)
        if isinstance(editor, QtWidgets.QLineEdit):
            editor.setAutoFillBackground(True)
            palette = editor.palette()
            base = QtWidgets.QApplication.palette().color(QtGui.QPalette.ColorRole.Base)
            palette.setColor(QtGui.QPalette.ColorRole.Base, base)
            editor.setPalette(palette)
        return editor


class WheelValueEditor(QtWidgets.QLineEdit):
    valueStepped = QtCore.Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._min_value = None
        self._max_value = None
        self._steps = 20

    def setRange(self, min_value, max_value) -> None:
        self._min_value = min_value
        self._max_value = max_value

    def setSteps(self, steps: int) -> None:
        self._steps = steps

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if self._min_value is None or self._max_value is None:
            return super().wheelEvent(event)
        low = min(self._min_value, self._max_value)
        high = max(self._min_value, self._max_value)
        span = high - low
        if span <= 0 or self._steps <= 0:
            return super().wheelEvent(event)
        try:
            value = float(self.text())
        except ValueError:
            value = low
        step = span / self._steps
        if event.angleDelta().y() > 0:
            value += step
        elif event.angleDelta().y() < 0:
            value -= step
        value = max(low, min(high, value))
        self.setText(f"{value:.7g}")
        self.valueStepped.emit(value)
        event.accept()


class ValueBarDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._use_slider = False
        self._orig_value = 0.
        self.closeEditor.connect(self.onEditorClosed)

    def set_use_slider(self, enabled: bool) -> None:
        self._use_slider = bool(enabled)

    def createEditor(self, parent, option, index):
        if index.column() != 1:
            return super().createEditor(parent, option, index)
        self._orig_value = self._to_float(index.data())
        if self._use_slider:
            slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, parent)
            slider.setMinimum(0)
            slider.setMaximum(1000)
            model = index.model()
            slider.valueChanged.connect(
                lambda _value, m=model, i=index: self._set_model_from_slider(m, i, slider)
            )
            return slider
        editor = WheelValueEditor(parent)
        min_val, max_val = self._get_min_max(index)
        editor.setRange(min_val, max_val)
        editor.setSteps(20)
        editor.setAutoFillBackground(True)
        palette = editor.palette()
        base = QtWidgets.QApplication.palette().color(QtGui.QPalette.ColorRole.Base)
        palette.setColor(QtGui.QPalette.ColorRole.Base, base)
        editor.setPalette(palette)
        model = index.model()
        editor.valueStepped.connect(lambda value, m=model, i=index: m.setData(i, value))
        return editor

    def setEditorData(self, editor, index):
        if self._use_slider and isinstance(editor, QtWidgets.QSlider):
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
            return
        return super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if self._use_slider and isinstance(editor, QtWidgets.QSlider):
            min_val = editor.property("min_val")
            max_val = editor.property("max_val")
            if min_val is None or max_val is None or max_val <= min_val:
                return
            value = min_val + (max_val - min_val) * (editor.value() / editor.maximum())
            model.setData(index, value)
            return
        return super().setModelData(editor, model, index)

    def paint(self, painter, option, index):
        opt = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget else QtWidgets.QApplication.style()
        text = opt.text
        opt.text = ""
        style.drawControl(QtWidgets.QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)

        if index.column() == 1:
            min_val, max_val = self._get_min_max(index)
            low = min(min_val, max_val)
            high = max(min_val, max_val)
            if high > low:
                value = self._to_float(index.data())
                ratio = (value - low) / (high - low)
                ratio = max(0.0, min(1.0, ratio))
                bar_rect = option.rect.adjusted(2, 1, -2, -1)
                painter.save()
                painter.setPen(QtCore.Qt.NoPen)
                fill_width = int(bar_rect.width() * ratio)
                if fill_width > 0:
                    painter.setBrush(QtGui.QColor(215, 215, 215, 160))
                    painter.drawRect(bar_rect.adjusted(0, 0, -(bar_rect.width() - fill_width), 0))
                painter.restore()

        hide_text = False
        if self._use_slider and opt.widget is not None:
            view = opt.widget
            if (
                isinstance(view, QtWidgets.QAbstractItemView)
                and view.state() == QtWidgets.QAbstractItemView.State.EditingState
                and view.currentIndex() == index
            ):
                hide_text = True
        if text and not hide_text:
            text_role = QtGui.QPalette.ColorRole.Text
            style.drawItemText(painter, option.rect, opt.displayAlignment, opt.palette, True, text, text_role)

    def _get_min_max(self, index):
        model = index.model()
        min_val = self._to_float(model.index(index.row(), 3).data())
        max_val = self._to_float(model.index(index.row(), 4).data())
        return min_val, max_val

    def _set_model_from_slider(self, model, index, slider) -> None:
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        if min_val is None or max_val is None or max_val <= min_val:
            return
        value = min_val + (max_val - min_val) * (slider.value() / slider.maximum())
        model.setData(index, value)

    @staticmethod
    def _to_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def onEditorClosed(self, editor, hint):
        if hint == QtWidgets.QAbstractItemDelegate.EndEditHint.RevertModelCache:
            if isinstance(editor, WheelValueEditor):
                editor.setText(f"{self._orig_value:.7g}")
                editor.valueStepped.emit(self._orig_value)
            else:
                min_val = editor.property("min_val")
                max_val = editor.property("max_val")
                value = max(min_val, min(max_val, self._orig_value))
                pos = int((value-min_val)/(max_val-min_val)*editor.maximum())
                editor.setValue(pos)
