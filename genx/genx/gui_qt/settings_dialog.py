"""
Qt port of settings_dialog.
"""

from math import log10

from PySide6 import QtCore, QtWidgets

from ..core.config import BaseConfig


class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent: QtWidgets.QWidget, settings: BaseConfig, title="Settings", apply_callback: callable = None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.settings = settings
        self.apply_callback = apply_callback or (lambda options: True)

        self.rb_actions = {}
        self.rb_bool_fields = {}
        self.value_entries = {}

        self.build_sizers()

        groups = list(self.settings.groups.keys())
        n_groups = len(groups)
        r_groups = n_groups // 2
        l_groups = n_groups - r_groups
        for lg in groups[:l_groups]:
            self.build_group(lg, self.row_sizer1)
        for rg in groups[l_groups:]:
            self.build_group(rg, self.row_sizer2)

        self.layout().activate()
        self.adjustSize()

    def build_sizers(self, add_apply=False):
        col_sizer = QtWidgets.QHBoxLayout()
        self.row_sizer1 = QtWidgets.QVBoxLayout()
        self.row_sizer2 = QtWidgets.QVBoxLayout()
        col_sizer.addLayout(self.row_sizer1, 1)
        col_sizer.addLayout(self.row_sizer2, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        if add_apply:
            buttons.addButton(QtWidgets.QDialogButtonBox.StandardButton.Apply)
            buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self.on_apply_changes)
        buttons.accepted.connect(self.on_apply_changes)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(col_sizer, 1)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line)
        layout.addWidget(buttons)

    def build_group(self, name: str, sizer: QtWidgets.QVBoxLayout):
        box = QtWidgets.QGroupBox(name, self)
        box_sizer = QtWidgets.QVBoxLayout(box)
        sizer.addWidget(box)

        pnames = self.settings.groups[name]
        fields = dict((fi.name, fi) for fi in self.settings.get_fields())
        for ni in pnames:
            if type(ni) in [list, tuple]:
                if len(ni) == 4 and fields[ni[1]].type is bool:
                    e1_active = fields[ni[1]]
                    e1_field = fields[ni[2]]
                    e2_field = fields[ni[3]]
                    gbox = QtWidgets.QGroupBox(ni[0], box)
                    gbox_sizer = QtWidgets.QVBoxLayout(gbox)
                    box_sizer.addWidget(gbox)
                    self.rb_bool_fields[gbox] = ni[1]
                    e1 = self.get_toggle_entry(e1_field, gbox, getattr(self.settings, e1_active.name, True))
                    gbox_sizer.addLayout(e1)
                    e2 = self.get_toggle_entry(e2_field, gbox, not getattr(self.settings, e1_active.name, True))
                    gbox_sizer.addLayout(e2)
                else:
                    hsizer = QtWidgets.QHBoxLayout()
                    box_sizer.addLayout(hsizer)
                    for nj in ni:
                        field = fields[nj]
                        entry = self.get_entry(field, box)
                        hsizer.addLayout(entry)
            else:
                field = fields[ni]
                entry = self.get_entry(field, box)
                box_sizer.addLayout(entry)

    def get_entry(self, field, parent, add_label=True):
        gmeta = field.metadata.get("genx", {})
        name = gmeta.get("label", field.name.replace("_", " "))
        if field.type is bool:
            entry = QtWidgets.QCheckBox(name, parent)
            entry.setChecked(getattr(self.settings, field.name, False))
            self.value_entries[field.name] = entry
            return self._label_entry(parent, entry, None, labeled=False)
        if "selection" in gmeta:
            choice = QtWidgets.QComboBox(parent)
            choice.addItems(gmeta["selection"])
            choice.setCurrentText(getattr(self.settings, field.name, gmeta["selection"][0]))
            self.value_entries[field.name] = choice
            return self._label_entry(parent, choice, name)
        if "pmin" in gmeta and "pmax" in gmeta:
            if field.type is int:
                spin = QtWidgets.QSpinBox(parent)
                spin.setRange(gmeta["pmin"], gmeta["pmax"])
            else:
                spin = QtWidgets.QDoubleSpinBox(parent)
                spin.setRange(gmeta["pmin"], gmeta["pmax"])
                inc = 10 ** (round(log10(gmeta["pmax"])) - 2)
                spin.setSingleStep(inc)
                spin.setDecimals(4)
            spin.setValue(getattr(self.settings, field.name, gmeta["pmin"]))
            self.value_entries[field.name] = spin
            return self._label_entry(parent, spin, name) if add_label else spin
        txt = QtWidgets.QLineEdit(parent)
        txt.setText(str(getattr(self.settings, field.name, "")))
        self.value_entries[field.name] = txt
        return self._label_entry(parent, txt, name) if add_label else txt

    def get_toggle_entry(self, field, parent, active):
        gmeta = field.metadata.get("genx", {})
        name = gmeta.get("label", field.name.replace("_", " "))
        entry = QtWidgets.QHBoxLayout()
        entry_active = QtWidgets.QRadioButton(name, parent)
        entry_active.setChecked(active)
        entry_active.toggled.connect(self.on_radio)
        entry.addWidget(entry_active)
        field_entry = self.get_entry(field, parent, add_label=False)
        if hasattr(field_entry, "setEnabled"):
            field_entry.setEnabled(active)
        self.rb_actions.setdefault(parent, {})[entry_active] = field_entry
        entry.addWidget(field_entry)
        return entry

    def on_radio(self):
        sender = self.sender()
        parent = sender.parent()
        for btn, field_entry in self.rb_actions.get(parent, {}).items():
            enabled = btn is sender and btn.isChecked()
            if hasattr(field_entry, "setEnabled"):
                field_entry.setEnabled(enabled)

    def _label_entry(self, parent, entry, label, labeled=True):
        out_entry = QtWidgets.QHBoxLayout()
        if labeled and label:
            out_entry.addWidget(QtWidgets.QLabel(label, parent))
        out_entry.addWidget(entry)
        return out_entry

    def collect_results(self):
        out = {}
        fields = dict((fi.name, fi) for fi in self.settings.get_fields())
        for key, entry in self.value_entries.items():
            field = fields[key]
            if isinstance(entry, QtWidgets.QCheckBox):
                out[key] = bool(entry.isChecked())
            elif isinstance(entry, QtWidgets.QComboBox):
                out[key] = entry.currentText()
            elif isinstance(entry, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                out[key] = field.type(entry.value())
            elif isinstance(entry, QtWidgets.QLineEdit):
                try:
                    out[key] = field.type(entry.text())
                except ValueError:
                    pass
            else:
                raise NotImplementedError(f"Could not evaluate Qt Control {entry!r}")
        for parent, rbg in self.rb_actions.items():
            # get the active state of the boolean fields for radio buttons
            self_bool = self.rb_bool_fields[parent]
            active_btn = next((btn for btn in rbg if btn.isChecked()), None)
            if active_btn is not None:
                out[self_bool] = rbg[active_btn].isEnabled()
        return out

    def on_apply_changes(self):
        options = self.collect_results()
        ok_pressed = self.sender() is None or not isinstance(self.sender(), QtWidgets.QAbstractButton) or (
            isinstance(self.sender(), QtWidgets.QAbstractButton)
            and self.sender().text().lower() in ("ok", "&ok")
        )
        if self.apply_callback(options) and ok_pressed:
            for key, value in options.items():
                setattr(self.settings, key, value)
            self.accept()
