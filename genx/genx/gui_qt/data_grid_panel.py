"""
Qt widget equivalent to the wx data_notebook_pane_2 panel.
"""

from __future__ import annotations

from typing import Optional

from PySide6 import QtCore, QtWidgets, QtGui

from .data_grid_panel_ui import Ui_DataGridPanel


class DataGridPanel(QtWidgets.QWidget):
    """
    Displays the selected dataset in a read-only grid.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._data_list = None

        self.ui = Ui_DataGridPanel()
        self.ui.setupUi(self)
        self._copy_shortcut = QtGui.QShortcut(QtGui.QKeySequence.Copy, self.ui.dataGrid)
        self._copy_shortcut.activated.connect(self.copy_selection_to_clipboard)

    def set_data_list(self, data_list) -> None:
        self._data_list = data_list
        self.update_dataset_choices(data_list)

    def update_dataset_choices(self, data_list) -> None:
        self._data_list = data_list
        names = [data_set.name for data_set in data_list] if data_list is not None else []
        current = self.ui.dataGridChoice.currentText()
        self.ui.dataGridChoice.blockSignals(True)
        self.ui.dataGridChoice.clear()
        self.ui.dataGridChoice.addItems(names)
        if current and current in names:
            self.ui.dataGridChoice.setCurrentText(current)
        self.ui.dataGridChoice.blockSignals(False)

    @QtCore.Slot(object)
    def on_data_list_event(self, event) -> None:
        """
        Qt slot equivalent to wx eh_external_update_data_grid_choice.
        """
        data_list = getattr(event, "data", None)
        if data_list is None:
            return
        self.update_dataset_choices(data_list)
        if self.ui.dataGridChoice.count() > 0 and self.ui.dataGridChoice.currentIndex() < 0:
            self.ui.dataGridChoice.setCurrentIndex(0)

    @QtCore.Slot(int)
    def on_data_grid_choice_changed(self, index: int) -> None:
        """
        Qt slot equivalent to wx eh_data_grid_choice.
        """
        if self._data_list is None:
            return
        if index < 0 or index >= len(self._data_list):
            return
        dataset = self._data_list[index]
        new_rows = max(len(dataset.x), len(dataset.y), len(dataset.x_raw), len(dataset.y_raw))
        self.ui.dataGrid.setRowCount(new_rows)

        for row in range(new_rows):
            for col in range(6):
                item = QtWidgets.QTableWidgetItem("-")
                self.ui.dataGrid.setItem(row, col, item)

        for row in range(len(dataset.x_raw)):
            self.ui.dataGrid.item(row, 0).setText(f"{dataset.x_raw[row]:.3e}")
        for row in range(len(dataset.y_raw)):
            self.ui.dataGrid.item(row, 1).setText(f"{dataset.y_raw[row]:.3e}")
        for row in range(len(dataset.error_raw)):
            self.ui.dataGrid.item(row, 2).setText(f"{dataset.error_raw[row]:.3e}")
        for row in range(len(dataset.x)):
            self.ui.dataGrid.item(row, 3).setText(f"{dataset.x[row]:.3e}")
        for row in range(len(dataset.y)):
            self.ui.dataGrid.item(row, 4).setText(f"{dataset.y[row]:.3e}")
        for row in range(len(dataset.error)):
            self.ui.dataGrid.item(row, 5).setText(f"{dataset.error[row]:.3e}")

    def copy_selection_to_clipboard(self) -> None:
        ranges = self.ui.dataGrid.selectedRanges()
        if not ranges:
            return
        sel = ranges[0]
        lines = []
        for row in range(sel.topRow(), sel.bottomRow() + 1):
            values = []
            for col in range(sel.leftColumn(), sel.rightColumn() + 1):
                item = self.ui.dataGrid.item(row, col)
                values.append(item.text() if item else "")
            lines.append("\t".join(values))
        QtWidgets.QApplication.clipboard().setText("\n".join(lines))
