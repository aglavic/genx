"""
===========
SimpleLayer
===========

Qt port of the materials helper for reflectivity plugins.
This version focuses on applying materials to the Reflectivity plugin.
"""

from abc import ABC, abstractmethod

from PySide6 import QtCore, QtGui, QtWidgets

from genx.models.lib.refl_base import ReflBase as ReflBaseNew
from genx.plugins import add_on_framework as framework
from genx.gui_qt.utils import ShowInfoDialog, ShowQuestionDialog
from genx.tools.materials_db import MASS_DENSITY_CONVERSION, Formula, mdb


class PluginInterface(ABC):
    def __init__(self, plugin):
        self._plugin = plugin

    @abstractmethod
    def material_apply(self, index, panel: QtWidgets.QWidget | None = None):
        pass


class RefPluginInterface(PluginInterface):
    def Update(self):
        try:
            self._plugin.sample_widget.UpdateListbox()
        except AttributeError:
            self._plugin.sample_widget.Update()

    def get_selected_layer(self):
        layer_idx = self._plugin.sample_widget.listbox.GetSelection()
        if layer_idx < 0:
            raise RuntimeError("No selection")
        active_layer = self._plugin.sampleh.getItem(layer_idx)
        if active_layer.__class__.__name__ == "Stack":
            self._plugin.sampleh.insertItem(layer_idx, "Layer", "WillChange")
            active_layer = self._plugin.sampleh.getItem(layer_idx + 1)
        return active_layer, layer_idx

    def set_layer_name(self, name, layer_idx):
        if self._plugin.sampleh.names[layer_idx] in ["Amb", "Sub"]:
            return
        tmpname = name
        i = 1
        while tmpname in self._plugin.sampleh.names:
            tmpname = "%s_%i" % (name, i)
            i += 1
        self._plugin.sampleh.names[layer_idx] = tmpname

    def material_apply(self, index, panel=None):
        formula, density = mdb[index]
        try:
            layer, layer_idx = self.get_selected_layer()
        except Exception:
            ShowInfoDialog(panel, "You have to select a layer or stack before applying material")
            return
        if isinstance(layer, ReflBaseNew):
            if hasattr(layer, "sld_n"):
                layer._ca["sld_x"] = f"10*{density}*({formula.f()})"
                layer._ca["sld_n"] = f"10*{density}*({formula.b()})"
            else:
                layer._ca["f"] = formula.f()
                layer._ca["b"] = formula.b()
                layer._ca["dens"] = density
        elif layer:
            layer.f = formula.f()
            layer.b = formula.b()
            layer.dens = density
        name = ""
        for element, count in formula:
            element = element.replace("{", "").replace("}", "").replace("^", "i")
            if count == 1:
                name += "%s" % element
            elif float(count) == int(count):
                name += "%s%i" % (element, count)
            else:
                name += ("%s%s" % (element, count)).replace(".", "_")
        self.set_layer_name(name, layer_idx)
        self.Update()


class Plugin(framework.Template):
    _refplugin: PluginInterface | None = None

    @property
    def refplugin(self):
        if not self._refplugin:
            self._init_refplugin()
        return self._refplugin

    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent
        materials_panel = self.NewDataFolder("Materials")
        layout = QtWidgets.QHBoxLayout(materials_panel)
        self.materials_panel = QtWidgets.QWidget(materials_panel)
        self.create_materials_list()
        layout.addWidget(self.materials_panel, 1)

    def _init_refplugin(self):
        ph = self.parent.plugin_control.plugin_handler
        if "Reflectivity" in ph.loaded_plugins:
            self._refplugin = RefPluginInterface(ph.loaded_plugins["Reflectivity"])
        elif "SimpleReflectivity" in ph.loaded_plugins:
            self._refplugin = RefPluginInterface(ph.loaded_plugins["SimpleReflectivity"])
        else:
            self._refplugin = None
            ShowInfoDialog(self.materials_panel, "Reflectivity plugin must be loaded")

    def create_materials_list(self):
        self.known_materials = mdb
        self.materials_list = MaterialsList(self.materials_panel, self.known_materials)
        layout = QtWidgets.QVBoxLayout(self.materials_panel)
        layout.setContentsMargins(0, 0, 0, 0)

        self.create_toolbar()
        layout.addWidget(self.toolbar, 0)
        layout.addWidget(self.materials_list, 1)

    def create_toolbar(self):
        toolbar = QtWidgets.QToolBar(self.materials_panel)
        toolbar.setIconSize(QtCore.QSize(20, 20))

        action_add = toolbar.addAction(QtGui.QIcon(":/main_gui/add.png"), "Add")
        action_delete = toolbar.addAction(QtGui.QIcon(":/main_gui/delete.png"), "Delete")
        action_apply = toolbar.addAction(QtGui.QIcon(":/main_gui/start_fit.png"), "Apply")

        action_add.triggered.connect(self.material_add)
        action_delete.triggered.connect(self.material_delete)
        action_apply.triggered.connect(self.material_apply)

        self.toolbar = toolbar

    def material_add(self):
        dialog = MaterialDialog(self.parent)
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.materials_list.AddItem(dialog.GetResult())
        dialog.close()

    def material_delete(self):
        self.materials_list.DeleteItem()

    def material_apply(self):
        index = self.materials_list.GetFirstSelected()
        if index < 0:
            return
        self.refplugin.material_apply(index, panel=self.materials_panel)


class MaterialsList(QtWidgets.QTableWidget):
    def __init__(self, parent: QtWidgets.QWidget, materials_list):
        super().__init__(parent)
        self.materials_list = materials_list
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["Chemical Formula", "n [10⁻⁶Å⁻²]", "kα [rₑ/Å⁻³]", "FU/Å³", "g/cm³"])
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setStretchLastSection(True)
        self.RefreshTable()

    def RefreshTable(self):
        self.setRowCount(len(self.materials_list))
        for row in range(len(self.materials_list)):
            self._set_row(row)

    def _set_row(self, row):
        def set_item(col, text):
            item = self.item(row, col)
            if item is None:
                item = QtWidgets.QTableWidgetItem()
                self.setItem(row, col, item)
            item.setText(text)

        prev_names = [str(mi[0]) for mi in self.materials_list[:row]]
        name = str(self.materials_list[row][0])
        prev_count = prev_names.count(name)
        if prev_count > 0:
            name += "-%i" % (prev_count + 1)
        set_item(0, name)
        set_item(1, "%.3f" % self.materials_list.SLDn(row).real)
        set_item(2, "%.3f" % self.materials_list.SLDx(row).real)
        set_item(3, "%.4f" % self.materials_list.dens_FU(row).real)
        set_item(4, "%.3f" % self.materials_list.dens_mass(row))

    def GetFirstSelected(self):
        rows = [idx.row() for idx in self.selectionModel().selectedRows()]
        return rows[0] if rows else -1

    def DeleteItem(self, index=None):
        index = self.GetFirstSelected()
        if index < 0:
            return
        item = self.materials_list[index]
        item_formula = ""
        for element, count in item[0]:
            if count == 1:
                item_formula += "%s" % element
            elif float(count) == int(count):
                item_formula += "%s%i" % (element, count)
            else:
                item_formula += "%s(%f)" % (element, count)
        result = ShowQuestionDialog(self, "Remove material %s?" % item_formula, "Remove?")
        if result:
            self.materials_list.pop(index)
            self.RefreshTable()

    def AddItem(self, item):
        index = 0
        while index < len(self.materials_list) and self.materials_list[index][0] < item[0]:
            index += 1
        self.materials_list.insert(index, item)
        self.RefreshTable()


class MaterialDialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("New Material")
        self.extracted_elements = Formula([])
        self._create_entries()

    def _create_entries(self):
        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        self.formula_entry = QtWidgets.QLineEdit(self)
        self.formula_entry.textChanged.connect(self.OnFormulaChanged)
        form.addRow("Formula:", self.formula_entry)

        self.mass_density = QtWidgets.QLineEdit(self)
        self.mass_density.textChanged.connect(self.OnMassDensityChange)
        form.addRow("Mass Density [g/cm³]:", self.mass_density)

        self.result_density = QtWidgets.QLineEdit(self)
        self.result_density.setReadOnly(True)
        form.addRow("Density [FU/Å³]:", self.result_density)

        self.formula_display = QtWidgets.QPlainTextEdit(self)
        self.formula_display.setReadOnly(True)
        self.formula_display.setFixedHeight(100)

        layout.addLayout(form)
        layout.addWidget(QtWidgets.QLabel("Extracted Elements:", self))
        layout.addWidget(self.formula_display, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons, 0)

    def OnFormulaChanged(self, text):
        for ign_char in [" ", "\t", "_", "-"]:
            text = text.replace(ign_char, "")
        if text == "":
            self.extracted_elements = Formula([])
            self.formula_display.setPlainText("")
            return
        try:
            formula = Formula.from_str(text)
        except ValueError:
            self.formula_display.setPlainText(f"?{text}?")
        else:
            self.extracted_elements = formula
            self.formula_display.setPlainText(formula.describe())
        self.OnMassDensityChange(None)

    def OnMassDensityChange(self, _event):
        fu_mass = self.extracted_elements.mFU()
        try:
            mass_density = float(self.mass_density.text())
        except ValueError:
            return
        density = "%g*%g/%g" % (mass_density, MASS_DENSITY_CONVERSION, fu_mass)
        self.result_density.setText(density)

    def GetResult(self):
        return self.extracted_elements, self.result_density.text()
