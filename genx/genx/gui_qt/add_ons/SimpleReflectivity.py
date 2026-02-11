""" <h1>Simple Reflectivity plugin </h1>
Reflectivity is a plugin for beginners just showing one single tab for
the sample and hiding all complex GenX functionality.
"""

import io
import os
import traceback

from logging import debug, warning

from PySide6 import QtCore, QtGui, QtWidgets

from genx.core.custom_logging import iprint
from genx.exceptions import GenxError
from genx.model import Model
from genx.models.lib.refl_base import ReflBase
from genx.plugins import add_on_framework as framework
from genx.tools.materials_db import MASS_DENSITY_CONVERSION, Formula, mdb

from .help_modules.custom_dialog import FloatObjectValidator, ValidateFitDialog
from .help_modules.reflectivity_gui import ReflClassHelpDialog
from .help_modules.reflectivity_sample_plot import SamplePlotPanel
from .help_modules.reflectivity_utils import find_code_segment

try:
    # set locale to prevent some issues with data format
    import locale

    from orsopy.slddb import api
    from orsopy.slddb.material import Formula as MatFormula

    locale.setlocale(locale.LC_ALL, "en_US.utf8")
    # initialize and potentially update local version of ORSO SLD db
    api.check()
except Exception:
    warning("The SimpleReflectivity plugin does not work properly without orsopy installed.")
    api = None


def get_mat_api(frm: Formula):
    # Use ORSO SLD db to query for a material by formula
    if not api:
        return None
    try:
        frm2 = MatFormula(frm.estr())
        res = api.localquery(dict(formula=str(frm2)))
        if len(res):
            mat = api.localmaterial(res[0]["ID"])
            return mat.dens, res[0]["ID"], res[0]["validated"]
    except Exception:
        debug("Error in SLD DB query", exc_info=True)
        return None


class Instrument(ReflBase):
    """
    Specify parameters of the probe and reflectometry instrument in SimpleReflectivity.
    """


_set_func_prefix = "set"
TOP_LAYER = 0
ML_LAYER = 1
BOT_LAYER = 2


class SampleTableData:
    _columns = [
        ("Layer", "string"),
        ("Formula Params:\n---------------\nMixure Params:", "choice"),
        ("Chem. Formula\n-------------\nSLD-1 [1e-6 A^-2]", "string"),
        ("", "bool", False),
        ("Density [g/cm^3]\n---------------\nSLD-2 [1e-6 A^-2]", "string"),
        ("", "bool", False),
        ("Moment [uB/FU]\n--------------\nFraction [% SLD-1]", "string"),
        ("", "bool", True),
        ("d [A]", "string"),
        ("", "bool", False),
        ("sigma [A]", "string"),
    ]

    _last_layer_data = []

    defaults = {
        "Formula": ["Layer", "Formula", "SLD", False, "2.0", False, "0.0", True, "10.0", False, "5.0", ML_LAYER],
        "Mixure": ["MixLayer", "Mixure", "6.0", False, "2.0", False, "100", True, "10.0", False, "5.0", ML_LAYER],
    }

    repetitions = 1

    def __init__(self, panel):
        self.panel = panel
        self.layers = []
        self.invalid_identifiers = []
        self.ResetModel(first=True)

    def ResetModel(self, first=False):
        self.ambient = [None, "Formula", "SLD", False, "0.0", False, "0.0", False, "0", False, "0"]
        self.substrate = [
            None,
            "Formula",
            Formula([["Si", 1.0]]),
            False,
            "2.32998",
            False,
            "0.0",
            False,
            "0",
            False,
            "5.0",
        ]
        self.layers = [
            [
                "Surface_Layer",
                "Formula",
                Formula([["Fe", 2.0], ["O", 2.0]]),
                False,
                "5.25568",
                False,
                "0.0",
                False,
                "20.0",
                False,
                "5.0",
                TOP_LAYER,
            ],
            [
                "Layer_1",
                "Formula",
                Formula([["Fe", 1.0]]),
                False,
                "7.87422",
                False,
                "0.0",
                False,
                "100.0",
                False,
                "5.0",
                ML_LAYER,
            ],
            [
                "Interface_Layer",
                "Formula",
                Formula([["Si", 1.0], ["O", 2.0]]),
                False,
                "4.87479",
                False,
                "0.0",
                False,
                "20.0",
                False,
                "5.0",
                BOT_LAYER,
            ],
        ]
        if not first:
            self.update_model()

    def update_model(self):
        if self.panel is not None:
            self.panel.UpdateModel("table")

    def GetNumberRows(self):
        return len(self.layers) + 3

    def GetNumberCols(self):
        return len(self._columns)

    def GetRowLabelValue(self, row):
        if row == self.repeatedInfo():
            return ""
        row = self.realRow(row)
        if row in [0, self.GetNumberRows() - 2]:
            return "-"
        return f"{row:2d}"

    def repeatedInfo(self):
        info_row = 1
        for layer in self.layers:
            if layer[11] != TOP_LAYER:
                break
            info_row += 1
        return info_row

    def realRow(self, row):
        info_row = self.repeatedInfo()
        if row < info_row:
            return row
        return row - 1

    def GetValue(self, row, col):
        if row == self.repeatedInfo():
            if col == 0:
                return "Repeated layer structure (white background)"
            if col == 8:
                return "Repetitions:"
            if col == 10:
                return str(self.repetitions)
            return ""
        row = self.realRow(row)
        if col == 0:
            if row == 0:
                return "Ambient"
            if row == (self.GetNumberRows() - 2):
                return "Substrate"
            return self.layers[row - 1][col].replace("_", " ")
        if row == 0:
            if col in [7, 8, 9, 10]:
                return ""
            return self.ambient[col]
        if row == self.GetNumberRows() - 2:
            if col in [7, 8]:
                return ""
            return self.substrate[col]
        return self.layers[row - 1][col]

    def get_valid_name(self, name):
        identifier = ""
        for char in name.replace(" ", "_"):
            if (identifier + char).isidentifier():
                identifier += char
        if identifier in self.invalid_identifiers:
            identifier = "_" + identifier
        existing = [li[0] for li in self.layers]
        if identifier not in existing:
            return identifier
        if identifier.split("_")[-1].isdigit():
            identifier = identifier.rsplit("_", 1)[0]
        i = 1
        while "%s_%i" % (identifier, i) in existing:
            i += 1
        return "%s_%i" % (identifier, i)

    def SetValue(self, row, col, value):
        if value == self.GetValue(row, col):
            return
        if row == self.repeatedInfo():
            if col == 10:
                self.repetitions = int(value)
            self.update_model()
            return
        row = self.realRow(row)

        if row == 0:
            to_edit = self.ambient
        elif row == (self.GetNumberRows() - 2):
            to_edit = self.substrate
        else:
            to_edit = self.layers[row - 1]

        if col == 0:
            old_name = to_edit[0]
            to_edit[0] = "AboutToChangeValue"
            to_edit[0] = self.get_valid_name(value)
            self.delete_grid_items(old_name)
        elif col == 2:
            if to_edit[1] == "Formula":
                if value == "SLD":
                    to_edit[2] = value
                else:
                    try:
                        formula = Formula.from_str(value)
                    except Exception:
                        pass
                    else:
                        to_edit[2] = formula
                        if formula in mdb:
                            to_edit[4] = "%g" % mdb.dens_mass(formula)
                        else:
                            res = get_mat_api(formula)
                            if res:
                                to_edit[4] = "%g" % res[0]
            else:
                try:
                    val = float(eval("%s" % value))
                    if 0 <= val <= 100:
                        to_edit[2] = value
                except Exception:
                    pass
                else:
                    to_edit[col] = value
        elif col == 1:
            to_edit[1] = value
            for i in [2, 3, 4, 5, 6]:
                to_edit[i] = self.defaults[value][i]
        elif col in [3, 5, 7, 9]:
            to_edit[col] = value
        elif col in [4, 6, 8, 10]:
            try:
                float(eval("%s" % value))
            except Exception:
                pass
            else:
                to_edit[col] = value
        self.update_model()

    def InsertRow(self, row):
        model_row = self.realRow(row)
        if model_row == (self.GetNumberRows() - 2):
            layer_type = self.substrate[1]
            layer_stack = BOT_LAYER
            model_row -= 1
        elif row == self.repeatedInfo():
            if len(self.layers) == 0:
                layer_type = "Formula"
                layer_stack = ML_LAYER
            else:
                layer_type = self.layers[row - 1][1]
                layer_stack = self.layers[row - 1][11]
        elif model_row > 0:
            layer_type = self.layers[model_row - 1][1]
            layer_stack = self.layers[model_row - 1][11]
        else:
            layer_type = self.ambient[1]
            layer_stack = TOP_LAYER
        newlayer = list(self.defaults[layer_type])
        newlayer[11] = layer_stack
        newlayer[0] = self.get_valid_name(newlayer[0])
        self.layers.insert(model_row, newlayer)
        self.update_model()
        return True

    def DeleteRow(self, row):
        if row == self.repeatedInfo():
            return False
        row = self.realRow(row)
        if row in [0, self.GetNumberRows() - 1]:
            return False
        if self.layers[row - 1][11] == ML_LAYER and len([li for li in self.layers if li[11] == ML_LAYER]) == 1:
            return False
        del_layer = self.layers.pop(row - 1)
        grid_parameters = self.panel.plugin.GetModel().get_parameters()
        for fi in ["dens", "magn", "d", "sigma"]:
            func_name = del_layer[0] + "." + _set_func_prefix + fi.capitalize()
            grid_parameters.set_fit_state_by_name(func_name, 0.0, 0, 0.0, 0.0)
        self.panel.UpdateGrid(grid_parameters)
        self.update_model()
        return True

    def MoveRow(self, row_from, row_to):
        ignore_rows = [0, self.repeatedInfo(), self.GetNumberRows() - 1]
        if row_from in ignore_rows or row_to in ignore_rows or row_to < 0:
            return False
        row_from = self.realRow(row_from)
        row_to = self.realRow(row_to)
        if self.layers[row_from - 1][11] != self.layers[row_to - 1][11]:
            return False
        moved_row = self.layers.pop(row_from - 1)
        self.layers.insert(row_to - 1, moved_row)
        self.update_model()
        return True

    def getLayerCode(self, layer):
        out_param = {}
        output = "model.Layer("
        if layer[1] == "Formula":
            formula = layer[2]
            if formula == "SLD":
                try:
                    nSLD = float(eval(layer[4]))
                except TypeError:
                    nSLD = 0.0
                    layer[4] = "0.0"
                mSLD = float(eval(layer[6]))
                output += "f=%s, " % (10 * nSLD - 10j * mSLD)
                output += "b=%s, " % nSLD
                output += "dens=0.1, magn=%s, " % mSLD
                out_param["dens"] = 0.1
                out_param["magn"] = mSLD
            else:
                dens = eval(layer[4]) * MASS_DENSITY_CONVERSION / formula.mFU()
                output += "f=%s, " % formula.f()
                output += "b=%s, " % formula.b()
                output += "dens=%s, " % dens
                output += "magn=%s, " % layer[6]
                out_param["dens"] = dens
                out_param["magn"] = float(eval(layer[6]))
        else:
            part1 = float(layer[6]) / 100.0
            part2 = 1.0 - part1
            output += "b=(%g*%s + %g*%s), " % (part1, layer[2], part2, layer[4])
            output += "dens=1.0, magn=0.0, "
            out_param["dens"] = 1.0
            out_param["magn"] = 0.0
        output += "d=%s, sigma=%s)" % (layer[8], layer[10])
        out_param["d"] = float(eval(layer[8]))
        out_param["sigma"] = float(eval(layer[10]))
        return output, out_param

    def getModelCode(self):
        grid_parameters = self.panel.plugin.GetModel().get_parameters()
        script = "# BEGIN Sample DO NOT CHANGE\n"
        li, oi = self.getLayerCode(self.ambient)
        script += "Amb = %s\n" % li
        for pi, fi in [(3, "dens"), (5, "magn")]:
            if pi == 5 and self.ambient[1] == "Mixure":
                if not self.ambient[5]:
                    continue
                fi = "dens"
            value = oi[fi]
            minval = value * 0.5
            maxval = value * 2.0
            func_name = "Amb." + _set_func_prefix + fi.capitalize()
            grid_parameters.set_fit_state_by_name(func_name, 0.0, 0, 0.0, 0.0)
            grid_parameters.set_fit_state_by_name(func_name, value, int(self.ambient[pi]), minval, maxval)

        for layer in self.layers:
            li, oi = self.getLayerCode(layer)
            script += "%s = %s\n" % (layer[0], li)
            for pi, fi in [(3, "dens"), (5, "magn"), (7, "d"), (9, "sigma")]:
                if pi == 5 and layer[1] == "Mixure":
                    if not layer[5]:
                        continue
                    fi = "dens"
                value = oi[fi]
                minval = value * 0.5
                maxval = value * 2.0
                func_name = layer[0] + "." + _set_func_prefix + fi.capitalize()
                grid_parameters.set_fit_state_by_name(func_name, 0.0, 0, 0.0, 0.0)
                grid_parameters.set_fit_state_by_name(func_name, value, int(layer[pi]), minval, maxval)

        li, oi = self.getLayerCode(self.substrate)
        script += "\nSub = %s\n" % li
        for pi, fi in [(3, "dens"), (5, "magn"), (9, "sigma")]:
            if pi == 5 and self.substrate[1] == "Mixure":
                if not self.substrate[5]:
                    continue
                fi = "dens"
            value = oi[fi]
            minval = value * 0.5
            maxval = value * 2.0
            func_name = "Sub." + _set_func_prefix + fi.capitalize()
            grid_parameters.set_fit_state_by_name(func_name, 0.0, 0, 0.0, 0.0)
            grid_parameters.set_fit_state_by_name(func_name, value, int(self.substrate[pi]), minval, maxval)

        self.panel.UpdateGrid(grid_parameters)

        top = [li[0] for li in self.layers if li[11] == TOP_LAYER]
        ml = [li[0] for li in self.layers if li[11] == ML_LAYER]
        bot = [li[0] for li in self.layers if li[11] == BOT_LAYER]
        script += "\nTop = model.Stack(Layers=[%s ], Repetitions = 1)\n" % str(", ".join(reversed(top)))
        script += "\nML = model.Stack(Layers=[%s ], Repetitions = %i)\n" % (
            str(", ".join(reversed(ml))),
            self.repetitions,
        )
        script += "\nBot = model.Stack(Layers=[%s ], Repetitions = 1)\n" % str(", ".join(reversed(bot)))
        script += (
            "\nsample = model.Sample(Stacks = [Bot, ML, Top], Ambient = Amb, Substrate = Sub)\n"
            "# END Sample\n\n"
            "# BEGIN Parameters DO NOT CHANGE\n"
        )
        self._last_layer_data = [list(self.ambient)]
        for li in self.layers:
            self._last_layer_data.append(list(li))
        self._last_layer_data.append(list(self.substrate))
        return script

    def delete_grid_items(self, name):
        grid_parameters = self.panel.plugin.GetModel().get_parameters()
        for fi in ["dens", "magn", "d", "sigma"]:
            func_name = name + "." + _set_func_prefix + fi.capitalize()
            grid_parameters.set_fit_state_by_name(func_name, 0.0, 0, 0.0, 0.0)

    def get_name_list(self):
        return ["Amb"] + [li[0] for li in self.layers] + ["Sub"]

    def update_layer_parameters(self, layer, dens=None, magn=None, d=None, sigma=None):
        # Update table values during/after a fit, layer can be index or name.
        if not isinstance(layer, int):
            layer = self.get_name_list().index(layer)
        if layer == 0:
            data = self.ambient
        elif layer == (len(self.layers) + 1):
            data = self.substrate
        else:
            data = self.layers[layer - 1]

        if not self._last_layer_data or len(self._last_layer_data) <= layer:
            self._last_layer_data = [list(self.ambient)] + [list(li) for li in self.layers] + [list(self.substrate)]
        ref_data = self._last_layer_data[layer]

        if data[1] == "Formula":
            formula = data[2]
            if formula == "SLD":
                if dens is not None and data[3]:
                    data[4] = str(eval(ref_data[4]) * dens / 0.1)
                if magn is not None and data[5]:
                    data[6] = str(eval(ref_data[6]) * dens / 0.1)
            else:
                if dens is not None and data[3]:
                    new_dens = float(dens) * formula.mFU() / MASS_DENSITY_CONVERSION
                    data[4] = str(new_dens)
                if magn is not None and data[5]:
                    data[6] = str(float(magn))
        elif dens is not None:
            sld1 = float(eval(ref_data[2]))
            sld2 = float(eval(ref_data[4]))
            frac = float(eval(ref_data[6])) / 100.0
            new_dens = (frac * sld1 + (1 - frac) * sld2) * dens / 0.1
            if data[3]:
                sld2_fraction = new_dens - frac * sld1
                data[4] = str(sld2_fraction / (1.0 - frac))
            if data[5]:
                new_frac = (new_dens - sld2) / (sld1 - sld2)
                data[6] = str(new_frac * 100.0)
        if d is not None and data[7]:
            data[8] = str(float(d))
        if sigma is not None and data[9]:
            data[10] = str(float(sigma))

class FormulaDelegate(QtWidgets.QStyledItemDelegate):
    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel

    def createEditor(self, parent, option, index):
        editor = QtWidgets.QLineEdit(parent)
        editor.textChanged.connect(self.panel.OnFormulaText)
        editor.setAutoFillBackground(True)
        palette = editor.palette()
        base = QtWidgets.QApplication.palette().color(QtGui.QPalette.ColorRole.Base)
        palette.setColor(QtGui.QPalette.ColorRole.Base, base)
        editor.setPalette(palette)
        return editor


class LayerTypeDelegate(QtWidgets.QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        combo = QtWidgets.QComboBox(parent)
        combo.addItems(["Formula", "Mixure"])
        return combo

    def setEditorData(self, editor, index):
        value = index.data(QtCore.Qt.ItemDataRole.EditRole)
        editor.setCurrentText(value)

    def setModelData(self, editor, model, index):
        model.setData(index, editor.currentText(), QtCore.Qt.ItemDataRole.EditRole)


class SampleTableWidget(QtWidgets.QTableWidget):
    def __init__(self, parent, table_data: SampleTableData):
        super().__init__(parent)
        self.table_data = table_data
        self._updating = False
        self.setColumnCount(self.table_data.GetNumberCols())
        self.setHorizontalHeaderLabels([c[0] for c in self.table_data._columns])
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectItems)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
            | QtWidgets.QAbstractItemView.EditTrigger.EditKeyPressed
            | QtWidgets.QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.setItemDelegateForColumn(1, LayerTypeDelegate(self))

        self.itemChanged.connect(self.on_item_changed)
        self.refresh()

    def refresh(self):
        self._updating = True
        try:
            self.setRowCount(self.table_data.GetNumberRows())
            self.clearSpans()
            self.setVerticalHeaderLabels(
                [self.table_data.GetRowLabelValue(row) for row in range(self.table_data.GetNumberRows())]
            )
            for row in range(self.rowCount()):
                for col in range(self.columnCount()):
                    value = self.table_data.GetValue(row, col)
                    item = QtWidgets.QTableWidgetItem("" if value is None else str(value))
                    if col in [3, 5, 7, 9]:
                        item.setText("")
                        item.setFlags(QtCore.Qt.ItemFlag.ItemIsEnabled
                                      | QtCore.Qt.ItemFlag.ItemIsUserCheckable
                                      )
                        item.setCheckState(QtCore.Qt.CheckState.Checked if value else QtCore.Qt.CheckState.Unchecked)
                    self.setItem(row, col, item)
            self._apply_row_spans()
            self._apply_row_styles()
        finally:
            self._updating = False

    def _apply_row_spans(self):
        info_row = self.table_data.repeatedInfo()
        self.setSpan(info_row, 0, 1, 8)
        self.setSpan(info_row, 8, 1, 2)
        self.setSpan(0, 6, 1, 5)
        last_row = self.table_data.GetNumberRows() - 1
        self.setSpan(last_row, 6, 1, 3)

    def _apply_row_styles(self):
        info_row = self.table_data.repeatedInfo()
        last_row = self.table_data.GetNumberRows() - 1
        for row in range(self.rowCount()):
            for col in range(self.columnCount()):
                item = self.item(row, col)
                if item is None:
                    continue
                if row == info_row:
                    if col != 10:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                    if col != 10:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsSelectable)
                    continue
                real_row = self.table_data.realRow(row)
                if row == 0:
                    item.setBackground(QtGui.QColor("#dddddd"))
                elif row == last_row:
                    item.setBackground(QtGui.QColor("#aaaaff"))
                else:
                    if self.table_data.layers[real_row - 1][11] == TOP_LAYER:
                        item.setBackground(QtGui.QColor("#ccffcc"))
                    elif self.table_data.layers[real_row - 1][11] == BOT_LAYER:
                        item.setBackground(QtGui.QColor("#ffaaff"))
                if row in [0, last_row]:
                    if col in [0, 7, 8, 9, 10]:
                        item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if col in [3, 5, 7, 9]:
                    item.setFlags(item.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
                if row in [0, last_row] and col in [9, 10]:
                    item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

                if col in [3, 5] and row not in [0, last_row]:
                    layer = self.table_data.layers[real_row - 1]
                    if layer[1] == "Mixure":
                        if col == 3 and layer[5]:
                            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                        if col == 5 and layer[3]:
                            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if row == 0 and col in [3, 5]:
                    if self.table_data.ambient[1] == "Mixure":
                        if col == 3 and self.table_data.ambient[5]:
                            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                        if col == 5 and self.table_data.ambient[3]:
                            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                if row == last_row and col in [3, 5]:
                    if self.table_data.substrate[1] == "Mixure":
                        if col == 3 and self.table_data.substrate[5]:
                            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)
                        if col == 5 and self.table_data.substrate[3]:
                            item.setFlags(item.flags() & ~QtCore.Qt.ItemFlag.ItemIsEditable)

    def on_item_changed(self, item):
        if self._updating or item is None:
            return
        row = item.row()
        col = item.column()
        if col in [3, 5, 7, 9]:
            value = bool(item.checkState() == QtCore.Qt.CheckState.Checked)
        else:
            value = item.text()
        self.table_data.SetValue(row, col, value)
        self.refresh()

    def current_data(self):
        return self.table_data


class SamplePanel(QtWidgets.QWidget):
    model: Model
    last_sample_script = ""
    help_dialog = None

    inst_params = dict(
        probe="neutron pol",
        I0=1.0,
        res=0.0,
        wavelength=1.54,
        pol="uu",
        Ibkg=0.0,
        samplelen=10.0,
        beamw=0.1,
        footype="no corr",
        name="inst",
        coords="q",
    )

    def __init__(self, parent, plugin):
        super().__init__(parent)
        self.plugin = plugin
        self.variable_span = 0.25
        self.inst_params = dict(SamplePanel.inst_params)
        self.update_callback = None
        self._instrument_dialog = None
        self._last_grid_data = None

        self.toolbar = QtWidgets.QToolBar(self)
        self._build_toolbar()

        self.info_text = QtWidgets.QLabel("", self)
        self.info_text.setWordWrap(True)
        self.info_text.hide()

        self.sample_table = SampleTableData(self)
        self.grid = SampleTableWidget(self, self.sample_table)
        self.grid.setItemDelegateForColumn(2, FormulaDelegate(self, self.grid))
        self.grid.cellActivated.connect(self.OnCellActivated)
        self.grid.currentCellChanged.connect(lambda r, c, _pr, _pc: self.OnCellActivated(r, c))

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.toolbar, 0)
        content = QtWidgets.QHBoxLayout()
        content.addWidget(self.grid, 1)
        content.addWidget(self.info_text, 0)
        layout.addLayout(content, 1)
        self.last_sample_script = self.sample_table.getModelCode()

    def _build_toolbar(self):
        self.action_add = self.toolbar.addAction(
            QtGui.QIcon(":/reflectivity_plugin/insert_layer.png"), "Insert Layer"
        )
        self.action_delete = self.toolbar.addAction(
            QtGui.QIcon(":/reflectivity_plugin/delete.png"), "Delete"
        )
        self.action_up = self.toolbar.addAction(
            QtGui.QIcon(":/reflectivity_plugin/move_up.png"), "Move Up"
        )
        self.action_down = self.toolbar.addAction(
            QtGui.QIcon(":/reflectivity_plugin/move_down.png"), "Move Down"
        )
        self.toolbar.addSeparator()

        self.instrument_button = QtWidgets.QToolButton(self.toolbar)
        self.instrument_button.setText("Instrument Settings")
        self.instrument_button.setIcon(QtGui.QIcon(":/reflectivity_plugin/instrument.png"))
        self.instrument_button.setToolTip("Edit probe and instrument parameters")
        self.instrument_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolbar.addWidget(self.instrument_button)

        self.toolbar.addSeparator()
        spacer = QtWidgets.QWidget(self.toolbar)
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Preferred)
        self.toolbar.addWidget(spacer)

        self.advanced_button = QtWidgets.QToolButton(self.toolbar)
        self.advanced_button.setText("To Advanced Modelling")
        self.advanced_button.setIcon(QtGui.QIcon(":/reflectivity_plugin/custom_parameters.png"))
        self.advanced_button.setToolTip(
            "Switch to Reflectivity plugin for advanced modeling options.\nThis converts the model and can't be undone."
        )
        self.advanced_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.toolbar.addWidget(self.advanced_button)

        self.action_add.triggered.connect(self.OnLayerAdd)
        self.action_delete.triggered.connect(self.OnLayerDelete)
        self.action_up.triggered.connect(self.MoveUp)
        self.action_down.triggered.connect(self.MoveDown)
        self.instrument_button.clicked.connect(self.EditInstrument)
        self.advanced_button.clicked.connect(self.SwitchAdvancedReflectivity)

    def OnLayerAdd(self):
        row = self.grid.currentRow()
        if row < 0:
            row = 0
        self.sample_table.InsertRow(row)
        self.grid.refresh()

    def OnLayerDelete(self):
        row = self.grid.currentRow()
        if row < 0:
            return
        if self.sample_table.DeleteRow(row):
            self.grid.refresh()

    def MoveUp(self):
        row = self.grid.currentRow()
        if row < 0:
            return
        if self.sample_table.MoveRow(row, row - 1):
            self.grid.refresh()
            self.grid.setCurrentCell(max(0, row - 1), 0)

    def MoveDown(self):
        row = self.grid.currentRow()
        if row < 0:
            return
        if self.sample_table.MoveRow(row, row + 1):
            self.grid.refresh()
            self.grid.setCurrentCell(min(self.grid.rowCount() - 1, row + 1), 0)

    def ShowHelp(self):
        if self.help_dialog is not None:
            return
        parent = self._instrument_dialog or self
        self.help_dialog = ReflClassHelpDialog(parent, Instrument())
        self.help_dialog.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.help_dialog.destroyed.connect(lambda _=None: setattr(self, "help_dialog", None))
        self.help_dialog.show()

    def EditInstrument(self):
        validators = {
            "probe": ["x-ray", "neutron", "neutron pol"],
            "coords": ["q", "2θ"],
            "I0": FloatObjectValidator(),
            "res": FloatObjectValidator(),
            "wavelength": FloatObjectValidator(),
            "Ibkg": FloatObjectValidator(),
            "samplelen": FloatObjectValidator(),
            "beamw": FloatObjectValidator(),
            "footype": ["no corr", "gauss beam", "square beam"],
        }
        inst_name = "inst"
        vals = {}
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()
        pars = ["probe", "coords", "wavelength", "I0", "Ibkg", "res", "footype", "samplelen", "beamw"]
        units = {
            "probe": "",
            "wavelength": "A",
            "coords": "",
            "I0": "arb.",
            "res": "[coord]",
            "beamw": "mm",
            "footype": "",
            "samplelen": "mm",
            "Ibkg": "arb.",
            "2θ": "deg",
            "q": "A^-1",
        }
        groups = [
            ("Radiation", ("probe", "wavelength", "I0")),
            ("Data", ("coords", "Ibkg", "res")),
            ("Footprint Correction", ("footype", "beamw", "samplelen")),
        ]
        for key in pars:
            vals[key] = self.inst_params[key]
            if key not in ["probe", "coords", "footype"]:
                func_name = inst_name + "." + _set_func_prefix + key.capitalize()
                editable[key] = int(grid_parameters.get_fit_state_by_name(func_name) or 0)

        dlg = ValidateFitDialog(
            self,
            pars,
            vals,
            validators,
            title="Instrument Editor",
            units=units,
            groups=groups,
            cols=2,
            editable_pars=editable,
            group_boxes=True,
        )
        self._instrument_dialog = dlg
        help_btn = QtWidgets.QPushButton("Show Help", dlg)
        help_btn.setStyleSheet("background-color: rgb(230, 230, 255);")
        help_btn.clicked.connect(self.ShowHelp)
        layout = dlg.layout()
        if layout is not None:
            row = QtWidgets.QHBoxLayout()
            row.addStretch(1)
            row.addWidget(help_btn)
            layout.insertLayout(0, row)

        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            vals = dlg.GetValues()
            states = dlg.GetStates()
            for key, value in vals.items():
                if key in ["probe", "coords", "footype"]:
                    self.inst_params[key] = value
                    continue
                value = float(value)
                minval = value * 0.5
                maxval = value * 2.0
                func_name = inst_name + "." + _set_func_prefix + key.capitalize()
                grid_parameters.set_fit_state_by_name(func_name, value, states.get(key, 0), minval, maxval)
                self.inst_params[key] = value
            self.UpdateGrid(grid_parameters)
            self.UpdateModel(evt="inst")
        self._instrument_dialog = None
        dlg.close()

    def SwitchAdvancedReflectivity(self):
        plugin_control = self.plugin.parent.plugin_control
        try:
            plugin_control.LoadPlugin("Reflectivity")
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
            self.plugin.ShowErrorDialog("Can NOT load Reflectivity plugin:\n\n" + tbtext)

    def SetUpdateCallback(self, func):
        self.update_callback = func

    def Update(self, update_script=True):
        if update_script and self.update_callback:
            self.update_callback(None)

    def UpdateGrid(self, grid_parameters):
        self._last_grid_data = [list(di) for di in grid_parameters.data]
        self.plugin.parent.ui.paramterGrid.SetParameters(grid_parameters)

    def CheckGridUpdate(self, parameters=None):
        if parameters is None:
            new_grid = self.plugin.GetModel().get_parameters()
            new_data = [list(di) for di in new_grid.data]
        else:
            new_data = parameters
        if self._last_grid_data == new_data:
            return
        layers = self.sample_table.get_name_list()
        for pi, val, *_rest in new_data:
            try:
                name, param = pi.split(".", 1)
            except ValueError:
                continue
            if name in layers:
                if param == _set_func_prefix + "Dens":
                    self.sample_table.update_layer_parameters(name, dens=val)
                if param == _set_func_prefix + "Magn":
                    self.sample_table.update_layer_parameters(name, magn=val)
                if param == _set_func_prefix + "D":
                    self.sample_table.update_layer_parameters(name, d=val)
                if param == _set_func_prefix + "Sigma":
                    self.sample_table.update_layer_parameters(name, sigma=val)
        self.grid.refresh()
        self._last_grid_data = new_data

    def UpdateModel(self, evt=None, first=False, re_color=False):
        coords = self.inst_params["coords"]
        if evt in [None, "inst"]:
            sample_script = self.last_sample_script
        else:
            sample_script = self.sample_table.getModelCode()
            self.last_sample_script = sample_script

        script = (
            "from numpy import *\n"
            "import models.spec_nx as model\n"
            "from models.utils import UserVars, fp, fw, bc, bw\n\n"
            "# BEGIN Instrument DO NOT CHANGE\n"
            "from models.utils import create_fp, create_fw\n"
        )
        insts, inst_str = self.instrumentCode()
        script += inst_str
        script += (
            "inst_fp = create_fp(inst.wavelength); inst_fw = create_fw(inst.wavelength)\n"
            "fp.set_wavelength(inst.wavelength); fw.set_wavelength(inst.wavelength)\n"
            "# END Instrument\n\n"
        )
        script += sample_script
        script += (
            "cp = UserVars()\n"
            "# END Parameters\n\n"
            "SLD = []\n"
            "def Sim(data):\n"
            "    I = []\n"
            "    SLD[:] = []\n"
        )
        datasets = self.model.data
        from genx import data

        if coords == "q":
            data.DataSet.simulation_params[0] = 0.001
            data.DataSet.simulation_params[1] = 0.601
        else:
            data.DataSet.simulation_params[0] = 0.01
            data.DataSet.simulation_params[1] = 6.01
        res_set = False
        for i, di in enumerate(datasets):
            di.run_command()
            script += "    # BEGIN Dataset %i DO NOT CHANGE\n" "    d = data[%i]\n" % (i, i)
            inst_id = i % len(insts)
            try:
                pol = di.meta["data_source"]["measurement"]["instrument_settings"]["polarization"]
                if pol == "unpolarized":
                    pass
                elif pol in ["mo", "om", "mm"]:
                    inst_id = 1 % len(insts)
                else:
                    inst_id = 0
            except KeyError:
                pass
            if hasattr(di, "res") and di.res is not None:
                script += "    %s.setRes(data[%i].res)\n" % (insts[inst_id], i)
            elif res_set:
                script += "    %s.setRes(0.001)\n" % (insts[inst_id])
            script += (
                "    I.append(sample.SimSpecular(d.x, %s))\n"
                "    if _sim: SLD.append(sample.SimSLD(None, None, %s))\n"
                "    # END Dataset %i\n" % (insts[inst_id], insts[inst_id], i)
            )
            if re_color and len(insts) > 1:
                if inst_id == 0:
                    di.data_color = (0.7, 0.0, 0.0)
                    di.sim_color = (1.0, 0.0, 0.0)
                else:
                    di.data_color = (0.0, 0.0, 0.7)
                    di.sim_color = (0.0, 0.0, 1.0)
            elif di.name.startswith("Data") and len(insts) > 1:
                prefix = ["Spin Up", "Spin Down"][inst_id]
                di.name = prefix + " %i" % (i // 2 + 1)
                if inst_id == 0:
                    di.data_color = (0.7, 0.0, 0.0)
                    di.sim_color = (1.0, 0.0, 0.0)
                else:
                    di.data_color = (0.0, 0.0, 0.7)
                    di.sim_color = (0.0, 0.0, 1.0)
            elif (di.name.startswith("Spin") and len(insts) == 1) or first:
                di.name = "Data %i" % i
                di.data_color = (0.0, 0.7, 0.0)
                di.sim_color = (0.0, 1.0, 0.0)
        datasets.update_data()
        self.plugin.parent.ui.dataListControl.list_ctrl._UpdateImageList()

        script += "    return I"
        self.plugin.SetModelScript(script)

        if evt is not None and self.plugin.mb_autoupdate_sim.isChecked():
            self.plugin.parent.simulate()

    def instrumentCode(self):
        params = self.inst_params
        inst_id = params.get("name", "inst")
        inst_str = (
            "inst = model.Instrument("
            f"probe='{params['probe']}', wavelength={params['wavelength']}, "
            f"I0={params['I0']}, res={params['res']}, Ibkg={params['Ibkg']}, "
            f"footype='{params['footype']}', beamw={params['beamw']}, "
            f"samplelen={params['samplelen']}, coords='{params['coords']}', "
            f"pol='{params['pol']}'"
            ")\n"
        )
        return [inst_id], inst_str

    def OnFormulaText(self, text):
        if not self.info_text.isVisible():
            return
        if text.strip() == "":
            self.info_text.setText("")
            return
        try:
            frm = Formula.from_str(text)
        except Exception as exc:
            self.info_text.setText("Error in Formula:\n" + str(exc))
        else:
            label = "Analyzed Formula:\n" + frm.describe()
            if frm in mdb:
                dens = mdb.dens_mass(frm)
                label += "\n\nFound in Materials:\n%g g/cm^3" % dens
            else:
                res = get_mat_api(frm)
                if res:
                    label += "\n\nFound in ORSO DB:\n%g g/cm^3" % res[0]
                    if res[2]:
                        label += "\nORSO validated\nID: %s" % res[1]
                    else:
                        label += "\nNOT validated\nID: %s" % res[1]
            self.info_text.setText(label)

    def OnCellActivated(self, row, col):
        if row < 0 or col < 0:
            self.info_text.hide()
            return
        if col == 2 and self.sample_table.GetValue(row, 1) == "Formula":
            self.info_text.show()
        else:
            self.info_text.hide()


class WizardSelectionPage(QtWidgets.QWizardPage):
    def __init__(self, title, intro_text, choices, choices_help=None, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        self._buttons = {}

        layout = QtWidgets.QVBoxLayout(self)
        intro = QtWidgets.QLabel(intro_text, self)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        choices_box = QtWidgets.QGroupBox(self)
        choices_layout = QtWidgets.QGridLayout(choices_box)
        self.button_group = QtWidgets.QButtonGroup(self)
        for idx, choice in enumerate(choices):
            btn = QtWidgets.QRadioButton(choice, choices_box)
            self.button_group.addButton(btn, idx)
            self._buttons[choice] = btn
            row = idx // 2
            col = idx % 2
            choices_layout.addWidget(btn, row, col)
        if choices:
            self._buttons[choices[0]].setChecked(True)

        if choices_help:
            for choice, help_text in zip(choices, choices_help):
                if choice in self._buttons:
                    self._buttons[choice].setToolTip(help_text)
            hint = QtWidgets.QLabel("Hover items for info", self)
            hint.setStyleSheet("color: #555;")
            layout.addWidget(hint)

        layout.addWidget(choices_box)

    def selection(self):
        for choice, btn in self._buttons.items():
            if btn.isChecked():
                return choice
        return ""

class Plugin(framework.Template):
    previous_xaxis = None
    _last_script = None

    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.model_obj = self.GetModel()

        sample_panel = self.NewInputFolder("Sample")
        sample_layout = QtWidgets.QVBoxLayout(sample_panel)
        self.sample_widget = SamplePanel(sample_panel, self)
        sample_layout.addWidget(self.sample_widget, 1)
        self.sample_widget.SetUpdateCallback(self.UpdateScript)
        self.sample_widget.model = self.GetModel()

        sld_plot_panel = self.NewPlotFolder("SLD")
        sld_layout = QtWidgets.QHBoxLayout(sld_plot_panel)
        self.sld_plot = SamplePlotPanel(sld_plot_panel, self)
        sld_layout.addWidget(self.sld_plot, 1)

        menu = self.NewMenu("Reflec")
        self.action_export_sld = menu.addAction("Export SLD...")
        self.action_show_imag = menu.addAction("Show Im SLD")
        self.action_show_imag.setCheckable(True)
        self.action_show_imag.setChecked(self.sld_plot.opt.show_imag)
        self.action_mass_density = menu.addAction("Mass Density")
        self.action_mass_density.setCheckable(True)
        self.action_mass_density.setChecked(self.sld_plot.opt.use_mass_density)
        menu.addSeparator()
        self.action_autoupdate_sld = menu.addAction("Autoupdate SLD")
        self.action_autoupdate_sld.setCheckable(True)
        self.action_autoupdate_sld.setChecked(True)

        self.action_export_sld.triggered.connect(self.OnExportSLD)
        self.action_show_imag.triggered.connect(self.OnShowImagSLD)
        self.action_mass_density.triggered.connect(self.OnShowMassDensity)
        self.action_autoupdate_sld.triggered.connect(self.OnAutoUpdateSLD)

        self.mb_autoupdate_sim = parent.ui.actionSimulateAutomatically
        self.mb_autoupdate_sim.setChecked(True)

        if self.model_obj.script != "":
            if self.model_obj.filename != "":
                iprint("SimpleReflectivity plugin: Reading loaded model")
                self.ReadModel()
            else:
                try:
                    self.ReadModel()
                except Exception:
                    iprint("SimpleReflectivity plugin: Creating new model")
                    self.OnNewModel(None)
        else:
            iprint("SimpleReflectivity plugin: Creating new model")
            self.OnNewModel(None)

        self.HideUIElements()
        self.StatusMessage("Simple Reflectivity plugin loaded")

        self.parent.model_control.update_parameters.connect(self.OnFitParametersUpdated)
        self.parent.model_control.value_change.connect(self.OnGridMayHaveErrors)
        self.parent.model_control.update_script.connect(self.ReadUpdateModel)

    def HideUIElements(self):
        self._hidden_pages = []
        nb = self.parent.ui.inputTabWidget
        for i in reversed(range(nb.count())):
            title = nb.tabText(i)
            if title != "Sample":
                self._hidden_pages.append((title, nb.widget(i), i))
                nb.removeTab(i)
        self._hidden_pages.reverse()

    def ShowUIElements(self):
        nb = self.parent.ui.inputTabWidget
        for title, page, index in self._hidden_pages:
            nb.insertTab(index, page, title)
        self._hidden_pages = None

    def Remove(self):
        if hasattr(self, "_hidden_pages") and self._hidden_pages:
            self.ShowUIElements()
        self.parent.model_control.update_parameters.disconnect(self.OnFitParametersUpdated)
        self.parent.model_control.value_change.disconnect(self.OnGridMayHaveErrors)
        self.parent.model_control.update_script.disconnect(self.ReadUpdateModel)
        framework.Template.Remove(self)

    def UpdateScript(self, _event):
        ... #self.sample_widget.UpdateModel()

    def OnAutoUpdateSLD(self, _evt):
        pass

    def OnShowImagSLD(self, _evt):
        self.sld_plot.opt.show_imag = self.action_show_imag.isChecked()
        self.sld_plot.WriteConfig()
        self.sld_plot.Plot()

    def OnShowMassDensity(self, _evt):
        self.sld_plot.opt.use_mass_density = self.action_mass_density.isChecked()
        self.sld_plot.WriteConfig()
        self.sld_plot.Plot()

    def OnExportSLD(self, _evt):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent,
            "Export SLD to ...",
            "",
            "Dat File (*.dat)",
        )
        if not fname:
            return
        result = True
        if os.path.exists(fname):
            filename = os.path.split(fname)[1]
            result = self.ShowQuestionDialog(
                "The file %s already exists. Do you wish to overwrite it?" % filename
            )
        if result:
            try:
                self.sld_plot.SavePlotData(fname)
            except IOError as exc:
                self.ShowErrorDialog(str(exc))
            except Exception:
                outp = io.StringIO()
                traceback.print_exc(200, outp)
                val = outp.getvalue()
                outp.close()
                self.ShowErrorDialog("Could not save the file. Python Error:\n%s" % (val,))

    def OnNewModel(self, _event):
        data_loader_ctrl = self.parent.ui.dataListControl.list_ctrl.data_loader_cont
        dls = list(sorted(data_loader_ctrl.plugin_handler.get_possible_plugins()))
        if "auto" in dls:
            dls.remove("auto")

        wizard = QtWidgets.QWizard(self.parent)
        wizard.setWindowTitle("Create New Model...")

        p1 = WizardSelectionPage(
            "Select Probe",
            "Please choose the experiment you want to model/fit. "
            "This option can be changed later from the Instrument Settings dialog.",
            ["x-ray", "neutron", "neutron pol"],
            parent=wizard,
        )
        p2 = WizardSelectionPage(
            "Select Data Loader",
            "How to import data into GenX.\n"
            "If your instrument is not listed use the default/resolution loader. "
            "This reads 3/4 columns from an ASCII file. "
            "If the file does not have x,y,dy(,res) column order, use Settings->Import "
            "to select which columns to read.",
            ["auto"] + dls,
            parent=wizard,
        )
        p3 = WizardSelectionPage(
            "Select Data Coordinates",
            "Set the x-coordinates used for simulation.\n"
            "You can define more experimental parameters like wavelength, resolution, "
            "footprint correction or background in the Instrument Settings dialog.",
            ["q", "2θ"],
            choices_help=[
                "Reciprocal lattice vector out-of-plane component qz in A^-1",
                "Detector angle in degrees",
            ],
            parent=wizard,
        )
        wizard.addPage(p1)
        wizard.addPage(p2)
        wizard.addPage(p3)

        if wizard.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            self.sample_widget.inst_params = dict(SamplePanel.inst_params)
            params = self.sample_widget.inst_params
            params["probe"] = p1.selection()
            params["coords"] = p3.selection()
            data_loader_ctrl.LoadPlugin(p2.selection())
            self.sample_widget.sample_table.ResetModel()
            self.sample_widget.UpdateModel(first=True)

    def OnDataChanged(self, event):
        if not event.new_model:
            self.sample_widget.UpdateModel(re_color=True)

    def OnOpenModel(self, _event):
        self.action_show_imag.setChecked(self.sld_plot.opt.show_imag)

    def OnSimulate(self, _event):
        if not self.action_autoupdate_sld.isChecked():
            QtCore.QTimer.singleShot(0, self.sld_plot.Plot)

    def OnFittingUpdate(self, _event):
        if self.action_autoupdate_sld.isChecked():
            QtCore.QTimer.singleShot(0, self.sld_plot.Plot)

    def OnGridChange(self, _event):
        self.sample_widget.Update(update_script=False)

    def ReadUpdateModel(self, *_args):
        try:
            self.ReadModel(verbose=False)
        except GenxError:
            pass
        except Exception as exc:
            self.StatusMessage(f"could not analyze script: {exc}")

    def ReadModel(self, verbose=True):
        if verbose:
            self.StatusMessage("Compiling the script...")
        self.CompileScript()
        if verbose:
            self.StatusMessage("Trying to interpret the script...")

        txt = self.GetModel().script
        grid_parameters = self.GetModel().get_parameters()
        insttxt = find_code_segment(txt, "Instrument")
        items = {}
        for li in insttxt.splitlines():
            if not li.split("=", 1)[0].strip() == "inst":
                continue
            instoptions = li.split("model.Instrument(")[1].strip()[:-1]
            instoptions = instoptions.strip(",").split(",")
            items = dict([tuple(map(str.strip, i.split("=", 1))) for i in instoptions])
        for key, value in self.sample_widget.inst_params.items():
            if key in items:
                newval = type(value)(eval(items[key]))
                self.sample_widget.inst_params[key] = newval

        sampletxt = find_code_segment(txt, "Sample")
        layers = {}
        layer_order = []
        stacks = {}
        repetitions = 1
        for li in sampletxt.splitlines():
            if "model.Layer" in li:
                name, ltxt = map(str.strip, li.split("=", 1))
                layers[name] = analyze_layer_txt(name, ltxt)
                layer_order.append(name)
            if "model.Stack" in li:
                name, stxt = map(str.strip, li.split("=", 1))
                stacks[name] = analyze_stack_txt(stxt)
                if name == "ML":
                    repetitions = int(stxt.split("Repetitions", 1)[1].split("=", 1)[1].split(",", 1)[0].rstrip(")"))
        if "Top" in stacks:
            top = stacks["Top"]
        else:
            top = []
        if "Bot" in stacks:
            bottom = stacks["Bot"]
        else:
            bottom = []
        for ni, di in layers.items():
            if ni in top:
                di.append(TOP_LAYER)
            elif ni in bottom:
                di.append(BOT_LAYER)
            else:
                di.append(ML_LAYER)

        for li in layers.values():
            prefix = li[0] + ".set"
            fit_items = [(3, "dens"), (5, "magn"), (7, "d"), (9, "sigma")]
            for index, si in fit_items:
                if grid_parameters.get_fit_state_by_name(prefix + si.capitalize()):
                    li[index] = True

        table = self.sample_widget.sample_table
        table.ambient = layers["Amb"]
        table.ambient[0] = None
        table.substrate = layers["Sub"]
        table.substrate[0] = None
        table.repetitions = repetitions
        new_layers = [layers[key] for key in layer_order if key not in ["Amb", "Sub"]]
        table.layers = new_layers
        self.sample_widget.grid.refresh()

        if verbose:
            self.StatusMessage("New sample loaded to plugin!")

    def OnFitParametersUpdated(self, event):
        grid_parameters = self.GetModel().get_parameters()
        keys = grid_parameters.get_fit_pars()[1]
        values = event.values
        parameters = [(key, value, None, None, None, None) for key, value in zip(keys, values)]
        self.sample_widget.CheckGridUpdate(parameters=parameters)
        self.sample_widget.Update(update_script=False)
        if event.permanent_change:
            for pi, val in zip(keys, values):
                try:
                    name, param = pi.split(".", 1)
                except ValueError:
                    continue
                if name == "inst":
                    for key in self.sample_widget.inst_params.keys():
                        if param == _set_func_prefix + key.capitalize():
                            value = float(val)
                            self.sample_widget.inst_params[key] = value
                            minval = value * 0.5
                            maxval = value * 2.0
                            grid_parameters.set_fit_state_by_name(pi, value, 0, 0, 0)
                            grid_parameters.set_fit_state_by_name(pi, value, 1, minval, maxval)
            self.sample_widget.sample_table.update_model()

    def OnGridMayHaveErrors(self, _event=None):
        errors = [pi.error for pi in self.model_obj.parameters if pi.fit]
        if len(errors) > 0 and "-" not in errors:
            error_data = []
            st = self.sample_widget.sample_table
            layers = st.get_name_list()
            for pi in self.model_obj.parameters:
                try:
                    name, param = pi.name.split(".", 1)
                except ValueError:
                    continue
                show_data = ["", "", "", ""]
                if name in layers:
                    lidx = st.get_name_list().index(name)
                    if lidx == 0:
                        st_data = st.ambient
                        name = "Ambient"
                    elif lidx == (len(st.layers) + 1):
                        name = "Substrate"
                        st_data = st.substrate
                    else:
                        st_data = st.layers[lidx - 1]
                    if param == _set_func_prefix + "Dens":
                        if st_data[1] == "Formula":
                            formula = st_data[2]
                            if formula == "SLD":
                                show_data = [name, "SLD [1e6 A^-1]", "%.6g" % pi.value, pi.error]
                            else:
                                emin, emax = map(float, pi.error.strip("(").strip(")").strip().split(","))
                                scale = formula.mFU() / MASS_DENSITY_CONVERSION
                                show_data = [
                                    name,
                                    "density [g/cm^3]",
                                    "%.6g" % (pi.value * scale),
                                    "(%.4g, %.4g)" % (emin * scale, emax * scale),
                                ]
                        else:
                            show_data = [name, "density", "%.6g" % pi.value, pi.error]
                    elif param == _set_func_prefix + "Magn":
                        show_data = [name, "magnetization", "%.6g" % pi.value, pi.error]
                    elif param == _set_func_prefix + "D":
                        show_data = [name, "thickness [A]", "%.6g" % pi.value, pi.error]
                    elif param == _set_func_prefix + "Sigma":
                        show_data = [name, "roughness [A]", "%.6g" % pi.value, pi.error]
                    else:
                        show_data = [name, param, "%.6g" % pi.value, pi.error]
                else:
                    show_data = [pi.name, "", "%.6g" % pi.value, pi.error]
                error_data.append(show_data)
            dia = QtWidgets.QDialog(self.parent)
            dia.setWindowTitle("Parameter Error Estimation")
            layout = QtWidgets.QVBoxLayout(dia)
            grid = QtWidgets.QTableWidget(dia)
            grid.setColumnCount(4)
            grid.setRowCount(len(error_data))
            grid.setHorizontalHeaderLabels(["Item", "Parameter", "Value", "Error min/max"])
            grid.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
            for i, row in enumerate(error_data):
                for j, cell_value in enumerate(row):
                    grid.setItem(i, j, QtWidgets.QTableWidgetItem(cell_value))
            grid.resizeColumnsToContents()
            layout.addWidget(grid, 1)
            dia.resize(min(grid.sizeHint().width() + 80, self.parent.width() // 1.5), 400)
            dia.exec()


def analyze_layer_txt(name, txt):
    output = [name]
    layeroptions = txt.split("model.Layer(")[1].strip()[:-1]
    items = layeroptions.strip(",").split(",")
    items = dict([tuple(map(str.strip, i.split("=", 1))) for i in items])

    if "bc." in items["b"] or "bw." in items["b"]:
        output.append("Formula")
        output.append(Formula.from_bstr(items["b"]))
        output.append(False)
        if "bc." in items["b"]:
            dens = float(items["dens"]) * output[2].mFU() / MASS_DENSITY_CONVERSION
        else:
            dens = float(items["dens"])
        output.append(str(dens))
        output.append(False)
        output.append(str(eval(items["magn"])))
    elif "+" in items["b"] and "*" in items["b"]:
        output.append("Mixure")
        part1, part2 = items["b"].strip("()").split("+", 1)
        frac = float(part1.split("*")[0]) * 100
        sld1 = float(part1.split("*")[1])
        sld2 = float(part2.split("*")[1])
        output.append(str(sld1))
        output.append(False)
        output.append(str(sld2))
        output.append(False)
        output.append(str(frac))
    else:
        output.append("Formula")
        output.append("SLD")
        output.append(False)
        output.append(str(eval(items["b"])))
        output.append(False)
        output.append(str(eval(items["magn"])))

    output.append(False)
    output.append(str(eval(items["d"])))
    output.append(False)
    output.append(str(eval(items["sigma"])))
    return output


def analyze_stack_txt(txt):
    txt = txt.split("Layers", 1)[1].split("]", 1)[0].strip("=[ ")
    layers = map(str.strip, txt.split(","))
    return list(layers)
