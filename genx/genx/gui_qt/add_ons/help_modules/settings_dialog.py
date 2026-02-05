from typing import Tuple

from PySide6 import QtWidgets


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, plugin, col_values, misc_values):
        super().__init__(plugin.parent)
        self.setWindowTitle("Data loader settings")

        box_layout = QtWidgets.QHBoxLayout()

        col_group = QtWidgets.QGroupBox("Columns", self)
        col_group_layout = QtWidgets.QGridLayout(col_group)
        self.col_controls = col_values.copy()
        for row, name in enumerate(sorted(col_values.keys())):
            label = QtWidgets.QLabel(f"{name}: ", col_group)
            control = QtWidgets.QSpinBox(col_group)
            control.setRange(0, 100)
            control.setValue(col_values[name])
            col_group_layout.addWidget(label, row, 0, 1, 1)
            col_group_layout.addWidget(control, row, 1, 1, 1)
            self.col_controls[name] = control
        box_layout.addWidget(col_group)

        misc_group = QtWidgets.QGroupBox("Misc", self)
        misc_group_layout = QtWidgets.QGridLayout(misc_group)
        self.misc_controls = misc_values.copy()
        for row, name in enumerate(sorted(misc_values.keys())):
            label = QtWidgets.QLabel(f"{name}: ", misc_group)
            value = misc_values[name]
            if isinstance(value, int):
                control = QtWidgets.QSpinBox(misc_group)
                control.setRange(0, 100)
                control.setValue(value)
            else:
                control = QtWidgets.QLineEdit(str(value), misc_group)
            misc_group_layout.addWidget(label, row, 0, 1, 1)
            misc_group_layout.addWidget(control, row, 1, 1, 1)
            self.misc_controls[name] = control
        box_layout.addWidget(misc_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(box_layout)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))
        layout.addWidget(buttons)

    def GetColumnValues(self):
        values = {}
        for key, ctrl in self.col_controls.items():
            values[key] = ctrl.value()
        return values

    def GetMiscValues(self):
        values = {}
        for key, ctrl in self.misc_controls.items():
            if isinstance(ctrl, QtWidgets.QSpinBox):
                val = ctrl.value()
            else:
                val = ctrl.text()
                if val.lower() == "none":
                    val = None
            values[key] = val
        return values

    def get_results(self) -> Tuple[bool, dict, dict]:
        if self.exec() == QtWidgets.QDialog.Accepted:
            return True, self.GetColumnValues(), self.GetMiscValues()
        return False, {}, {}
