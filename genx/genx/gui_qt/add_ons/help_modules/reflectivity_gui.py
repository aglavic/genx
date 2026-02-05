"""
GUI support classes for the Reflectivity plug-in (Qt version).
"""

from typing import Callable, Dict, List, Union

from PySide6 import QtCore, QtGui, QtWidgets

from genx.core.custom_logging import iprint
from genx.gui_qt.utils import ShowQuestionDialog, ShowWarningDialog
from genx.model import Model
from genx.models.lib.base import AltStrEnum
from genx.models.lib.refl_base import ReflBase as ReflBaseNew
from genx.parameters import Parameters

try:
    from docutils.core import publish_doctree, publish_from_doctree
    from docutils.parsers.rst import roles
except ImportError:

    def rst_html(text):
        return "For proper display install docutils.<br>\n" + text.replace("\n", "<br>\n")

else:

    def _role_fn(name, rawtext, text, lineno, inliner, options=None, content=None):
        if options is None:
            options = {}
        if content is None:
            content = []
        return [], []

    roles.register_canonical_role("mod", _role_fn)

    def rst_html(text):
        return publish_from_doctree(publish_doctree(text), writer_name="html").decode()

from .custom_dialog import (
    ComplexObjectValidator,
    FloatObjectValidator,
    NoMatchValidTextObjectValidator,
    ParameterExpressionCombo,
    TextObjectValidator,
    ValidateDialog,
    ValidateFitDialog,
    ValidateFitNotebookDialog,
)
from .reflectivity_misc import ReflectivityModule
from .reflectivity_utils import SampleHandler, find_code_segment

_set_func_prefix = "set"


class _HtmlItemDelegate(QtWidgets.QStyledItemDelegate):
    def paint(self, painter, option, index):
        painter.save()
        doc = QtGui.QTextDocument()
        doc.setHtml(index.data(QtCore.Qt.ItemDataRole.DisplayRole))
        doc.setTextWidth(option.rect.width())
        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()
        if option.state & QtWidgets.QStyle.StateFlag.State_Selected:
            painter.fillRect(option.rect, option.palette.highlight())
            ctx.palette.setColor(QtGui.QPalette.ColorRole.Text, option.palette.highlightedText().color())
        painter.translate(option.rect.topLeft())
        doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        doc = QtGui.QTextDocument()
        doc.setHtml(index.data(QtCore.Qt.ItemDataRole.DisplayRole))
        doc.setTextWidth(option.rect.width())
        return QtCore.QSize(int(doc.idealWidth()), int(doc.size().height()))


class MyHtmlListBox(QtWidgets.QListWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setItemDelegate(_HtmlItemDelegate(self))
        self.html_items = []
        self.SetItemList(["Starting up..."])

    def SetItemList(self, html_list):
        self.html_items = html_list
        self.clear()
        for item in html_list:
            self.addItem(QtWidgets.QListWidgetItem(item))
        self.viewport().update()

    def GetSelection(self):
        return self.currentRow()

    def SetSelection(self, row):
        if row is None or row < 0:
            return
        self.setCurrentRow(row)

    def GetStringSelection(self):
        row = self.currentRow()
        if row < 0 or row >= len(self.html_items):
            return ""
        return self.html_items[row]


class ReflClassHelpDialog(QtWidgets.QDialog):
    def __init__(self, parent, item: ReflBaseNew):
        super().__init__(parent)
        self.setWindowTitle(f"{item.__class__.__name__} Help")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)

        layout = QtWidgets.QVBoxLayout(self)
        self.html_win = QtWidgets.QTextBrowser(self)
        layout.addWidget(self.html_win, 1)

        size = parent.size()
        pos = parent.pos()
        self.resize(600, size.height())
        self.move(pos.x() + size.width(), pos.y())

        doc = f"{item.__class__.__name__}\n{'='*len(item.__class__.__name__)}\n\n{item.__class__.__doc__}"
        self.html_win.setHtml(rst_html(doc))
        self.html_win.setStyleSheet("background-color: rgb(230, 230, 255);")


class ReflClassEditor:
    parent: QtWidgets.QWidget
    object_name: str
    eval_func: Callable
    grid_parameters: Parameters

    instance: ReflBaseNew
    ignore_attributes: List[str]
    pars: List[str]
    par_vals: dict
    par_validators: dict
    par_editable: Dict[str, int]

    def __init__(self, parent, object_name, eval_func, grid_parameters=None, ignore_attributes=None):
        self.parent = parent
        self.object_name = object_name
        self.eval_func = eval_func
        self.grid_parameters = grid_parameters
        self.ignore_attributes = ignore_attributes or []
        self.help_dialog = None

        self.initialize()
        self.extract_values()
        self.extract_grid_params()

    def initialize(self):
        self.instance = self.eval_func(self.object_name)
        if not isinstance(self.instance, ReflBaseNew):
            raise RuntimeError(f"{self.instance} is not an instance of a ReflBase derived object")
        self.pars = []
        self.par_vals = {}
        self.par_validators = {}
        self.par_editable = {}

    def extract_values(self):
        for par_name, value_info in self.instance._parameter_info().items():
            if par_name in self.ignore_attributes:
                continue
            value = self.instance._ca[par_name]
            validator = self.get_validator(value_info)
            if isinstance(value_info.type, type) and issubclass(value_info.type, AltStrEnum):
                value = str(value_info.type(self.eval_func(value)))
            self.pars.append(par_name)
            self.par_vals[par_name] = value
            self.par_validators[par_name] = validator

    def extract_grid_params(self):
        if self.grid_parameters is None:
            return
        for par_name, value_info in self.instance._parameter_info().items():
            if par_name in self.ignore_attributes:
                continue
            func_name = self.object_name + "." + _set_func_prefix + par_name.capitalize()
            if value_info.type is not complex:
                grid_value = self.grid_parameters.get_value_by_name(func_name)
                if grid_value is not None:
                    self.par_vals[par_name] = grid_value
                self.par_editable[par_name] = self.grid_parameters.get_fit_state_by_name(func_name)
            else:
                grid_value_real = self.grid_parameters.get_value_by_name(func_name + "real")
                grid_value_imag = self.grid_parameters.get_value_by_name(func_name + "imag")
                if grid_value_imag is not None or grid_value_real is not None:
                    v = self.eval_func(self.par_vals[par_name])
                    if grid_value_real is not None:
                        v = grid_value_real + v.imag * 1.0j
                    if grid_value_imag is not None:
                        v = v.real + grid_value_imag * 1.0j
                    self.par_vals[par_name] = repr(v)
                self.par_editable[par_name] = max(
                    self.grid_parameters.get_fit_state_by_name(func_name + "real"),
                    self.grid_parameters.get_fit_state_by_name(func_name + "imag"),
                )

    def ShowDialog(self, title=None):
        if title is None:
            title = f"{self.instance.__class__.__name__} Editor ({self.object_name})"

        groups = self.instance.Groups
        units = getattr(self.instance, "Units", False)
        dlg = ValidateFitDialog(
            self.parent,
            self.pars,
            self.par_vals,
            self.par_validators,
            title=title,
            groups=groups,
            units=units,
            editable_pars=self.par_editable,
        )

        res = dlg.ShowModal()
        if res == QtWidgets.QDialog.DialogCode.Accepted:
            vals = dlg.GetValues()
            states = dlg.GetStates()
            self.update_object(vals, states)
            self.update_grid(vals, states)
        dlg.close()
        return res == QtWidgets.QDialog.DialogCode.Accepted

    def update_object(self, vals, states):
        for name, value_info in self.instance._parameter_info().items():
            if name not in self.pars:
                continue
            if not states[name]:
                orig_type = value_info.type
                if getattr(orig_type, "__origin__", None) is Union:
                    e_value = self.eval_func(vals[name])
                    for subtype in orig_type.__args__:
                        try:
                            setattr(self.instance, name, subtype(e_value))
                        except (ValueError, TypeError):
                            continue
                        else:
                            self.instance._ca[name] = str(vals[name])
                    return
                elif orig_type is str or issubclass(orig_type, AltStrEnum):
                    self.instance._ca[name] = repr(vals[name])
                    setattr(self.instance, name, orig_type(vals[name]))
                else:
                    e_value = self.eval_func(vals[name])
                    self.instance._ca[name] = str(vals[name])
                    setattr(self.instance, name, orig_type(e_value))

    def update_grid(self, vals, states):
        vspan = self.parent.variable_span
        for name, value_info in self.instance._parameter_info().items():
            if name not in self.pars:
                continue
            if states[name] and self.par_editable[name] != 3 and self.par_editable[name] != states[name]:
                value = self.eval_func(vals[name])
                if value_info.type is complex:
                    func_name = self.object_name + "." + _set_func_prefix + name.capitalize() + "real"
                    val = value.real
                    minval = min(val * (1 - vspan), val * (1 + vspan))
                    maxval = max(val * (1 - vspan), val * (1 + vspan))
                    self.grid_parameters.set_fit_state_by_name(func_name, val, states[name], minval, maxval)
                    val = value.imag
                    minval = min(val * (1 - vspan), val * (1 + vspan))
                    maxval = max(val * (1 - vspan), val * (1 + vspan))
                    func_name = self.object_name + "." + _set_func_prefix + name.capitalize() + "imag"
                    self.grid_parameters.set_fit_state_by_name(func_name, val, states[name], minval, maxval)
                else:
                    val = value
                    minval = min(val * (1 - vspan), val * (1 + vspan))
                    maxval = max(val * (1 - vspan), val * (1 + vspan))
                    func_name = self.object_name + "." + _set_func_prefix + name.capitalize()
                    self.grid_parameters.set_fit_state_by_name(func_name, value, states[name], minval, maxval)
            elif self.par_editable[name] and not states[name]:
                if value_info.type is complex or complex in getattr(value_info.type, "__args__", ()):
                    func_name = self.object_name + "." + _set_func_prefix + name.capitalize() + "real"
                    self.grid_parameters.set_fit_state_by_name(func_name, 0.0, 0, 0, 0)
                    func_name = self.object_name + "." + _set_func_prefix + name.capitalize() + "imag"
                    self.grid_parameters.set_fit_state_by_name(func_name, 0.0, 0, 0, 0)
                else:
                    func_name = self.object_name + "." + _set_func_prefix + name.capitalize()
                    self.grid_parameters.set_fit_state_by_name(func_name, 0.0, 0, 0, 0)

    def get_validator(self, value_info):
        if isinstance(value_info.type, type) and issubclass(value_info.type, AltStrEnum):
            return [i.value for i in value_info.type]
        if value_info.type is complex or complex in getattr(value_info.type, "__args__", ()):
            return ComplexObjectValidator(eval_func=self.eval_func)
        if value_info.type is bool:
            return ["True", "False"]
        return FloatObjectValidator(eval_func=self.eval_func)

class SamplePanel(QtWidgets.QWidget):
    sampleh: SampleHandler
    model: ReflectivityModule

    def __init__(self, parent, plugin, refindexlist=None):
        super().__init__(parent)
        self.plugin = plugin
        self.refindexlist = refindexlist or []
        self.variable_span = 0.25
        self.fit_colour = (245, 121, 0)
        self.const_fit_colour = (114, 159, 207)
        self.update_callback = lambda event: ""

        layout = QtWidgets.QVBoxLayout(self)
        self.toolbar = QtWidgets.QToolBar(self)
        self._setup_toolbar()
        layout.addWidget(self.toolbar, 0)
        self.listbox = MyHtmlListBox(self)
        self.listbox.itemDoubleClicked.connect(self.lbDoubleClick)
        layout.addWidget(self.listbox, 1)

    def _icon(self, name):
        return QtGui.QIcon(f":/reflectivity_plugin/{name}.png")

    def _setup_toolbar(self):
        actions = [
            ("Insert Layer", "insert_layer", self.InsertLay),
            ("Insert Stack", "insert_stack", self.InsertStack),
            ("Delete", "delete", self.DeleteSample),
            ("Rename", "change_name", self.ChangeName),
            ("Move up", "move_up", self.MoveUp),
            ("Move down", "move_down", self.MoveDown),
            ("Edit Sample", "sample", self.EditSampleParameters),
            ("Edit Instrument", "instrument", self.EditInstrument),
        ]
        for label, icon_name, callback in actions:
            action = self.toolbar.addAction(self._icon(icon_name), label)
            action.triggered.connect(callback)

    def SetUpdateCallback(self, func):
        self.update_callback = func

    def set_sampleh(self, sampleh: SampleHandler):
        self.sampleh = sampleh

    def set_model(self, model: ReflectivityModule):
        self.model = model

    def create_html_decorator(self):
        grid_parameters = self.plugin.GetModel().get_parameters()
        dic_lookup = {}
        for par in grid_parameters.get_names():
            l = par.split(".")
            if len(l) == 2:
                name = l[0]
                par_name = l[1][3:].lower()
                dic_lookup[(name, par_name)] = (
                    grid_parameters.get_value_by_name(par),
                    grid_parameters.get_fit_state_by_name(par),
                )
        fit_color_str = "rgb(%d,%d,%d)" % self.fit_colour
        const_fit_color_str = "rgb(%d,%d,%d)" % self.const_fit_colour

        def decorator(name, text):
            try:
                start_index = text.index("(") + 1
            except ValueError:
                start_index = 0
            ret_str = text[:start_index]
            for par_str in text[start_index:].split(","):
                par_name = par_str.split("=")[0].strip()
                if (name, par_name) in dic_lookup:
                    val, state = dic_lookup[(name, par_name)]
                    if state == 1:
                        par_str = " <font color=%s><b>%s=%.2e</b></font>," % (fit_color_str, par_name, val)
                    elif state == 2:
                        par_str = " <font color=%s><b>%s=%.2e</b></font>," % (const_fit_color_str, par_name, val)
                elif (name, par_name + "real") in dic_lookup or (name, par_name + "imag") in dic_lookup:
                    if (name, par_name + "real") in dic_lookup:
                        val, state = dic_lookup[(name, par_name + "real")]
                        if state == 1:
                            par_str = " <font color=%s><b>%s=(%.2e,</b></font>" % (fit_color_str, par_name, val)
                        elif state == 2:
                            par_str = " <font color=%s><b>%s=(%.2e,</b></font>" % (const_fit_color_str, par_name, val)
                    else:
                        par_str = " <b>%s=??+</b>" % par_name
                    if (name, par_name + "imag") in dic_lookup:
                        val, state = dic_lookup[(name, par_name + "imag")]
                        if state == 1:
                            par_str += " <font color=%s><b>%.2e)</b></font>," % (fit_color_str, val)
                        elif state == 2:
                            par_str += " <font color=%s><b>%.2e)</b></font>," % (const_fit_color_str, val)
                    else:
                        par_str += " <b>??)</b>,"
                else:
                    par_str += ","
                ret_str += par_str
            if ret_str[-1] == ",":
                ret_str = ret_str[:-1]
            if text[-1] == ")" and ret_str[-1] != ")":
                ret_str += ")"
            return ret_str

        return decorator

    def Update(self, update_script=True):
        deco = self.create_html_decorator()
        sl = self.sampleh.getStringList(html_encoding=True, html_decorator=deco)
        self.listbox.SetItemList(sl)
        if update_script:
            self.update_callback(None)

    def SetSample(self, sample, names):
        self.sampleh.sample = sample
        self.sampleh.names = names
        self.Update()

    def EditSampleParameters(self, _evt=None):
        obj_name = "sample"
        eval_func = self.plugin.GetModel().eval_in_model
        if isinstance(self.sampleh.sample, ReflBaseNew):
            grid_parameters = self.plugin.GetModel().get_parameters()
            editor = ReflClassEditor(
                self, obj_name, eval_func, grid_parameters, ignore_attributes=["Ambient", "Substrate", "Stacks"]
            )
            if editor.ShowDialog():
                self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
                self.sampleh.sample = self.plugin.GetModel().script_module.sample
                self.Update()
            return

        editable, grid_parameters, groups, pars, units, validators, vals = self.ExtractSampleParameters(obj_name)
        dlg = ValidateFitDialog(
            self, pars, vals, validators, title="Sample Editor", groups=groups, units=units, editable_pars=editable
        )
        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            old_vals = vals
            vals = dlg.GetValues()
            states = dlg.GetStates()
            for par in pars:
                if not states[par]:
                    old_type = type(old_vals[par])
                    setattr(self.sampleh.sample, par, old_type(vals[par]))
                if editable[par] != states[par]:
                    value = eval_func(vals[par])
                    minval = min(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    maxval = max(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    func_name = obj_name + "." + _set_func_prefix + par.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value, states[par], minval, maxval)
                    self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
            self.Update()
        dlg.close()

    def ExtractSampleParameters(self, obj_name):
        grid_parameters = self.plugin.GetModel().get_parameters()
        validators = {}
        vals = {}
        pars = []
        editable = {}
        string_choices = getattr(self.model, "sample_string_choices", {})
        for item in self.model.SampleParameters:
            if item in ["Stacks", "Substrate", "Ambient"]:
                continue
            validators[item] = string_choices.get(item, FloatObjectValidator())
            vals[item] = getattr(self.sampleh.sample, item)
            pars.append(item)
            func_name = obj_name + "." + _set_func_prefix + item.capitalize()
            grid_value = grid_parameters.get_value_by_name(func_name)
            editable[item] = grid_parameters.get_fit_state_by_name(func_name)
            if grid_value is not None:
                vals[item] = grid_value
        groups = getattr(self.model, "SampleGroups", False)
        units = getattr(self.model, "SampleUnits", False)
        return editable, grid_parameters, groups, pars, units, validators, vals

    def SetInstrument(self, instruments):
        self.instruments = instruments

    def EditInstrument(self, _evt=None):
        eval_func = self.plugin.GetModel().eval_in_model
        editable, grid_parameters, model_inst_params, pars, validators, vals, groups, units = (
            self.ExtractInstrumentParams()
        )
        dlg = ValidateFitNotebookDialog(
            self,
            pars,
            vals,
            validators,
            title="Instrument Editor",
            groups=groups,
            units=units,
            editable_pars=editable,
        )
        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            old_vals = vals
            vals = dlg.GetValues()
            states = dlg.GetStates()
            self.UpdateInstrumentConfiguration(
                editable, eval_func, grid_parameters, model_inst_params, old_vals, states, vals
            )
            self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
            self.Update()
        dlg.close()

    def UpdateInstrumentConfiguration(
        self, editable, eval_func, grid_parameters, model_inst_params, old_vals, states, vals
    ):
        if isinstance(self.sampleh.sample, ReflBaseNew):
            return self.UpdateInstrumentConfigurationNew(
                editable, eval_func, grid_parameters, model_inst_params, old_vals, states, vals
            )
        self.instruments = {}
        for inst_name in vals:
            new_instrument = False
            if inst_name not in self.instruments:
                self.instruments[inst_name] = self.model.Instrument()
                new_instrument = True
            for par in self.model.InstrumentParameters:
                if not states[inst_name][par]:
                    orig_type = type(model_inst_params[par])
                    e_value = vals[inst_name][par] if orig_type is str else eval_func(vals[inst_name][par])
                    setattr(self.instruments[inst_name], par, orig_type(e_value))
                else:
                    setattr(self.instruments[inst_name], par, old_vals[inst_name][par])
                if new_instrument and states[inst_name][par] > 0:
                    value = eval_func(vals[inst_name][par])
                    minval = min(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    maxval = max(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    func_name = inst_name + "." + _set_func_prefix + par.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value, states[inst_name][par], minval, maxval)
                elif not new_instrument and editable[inst_name][par] != states[inst_name][par]:
                    value = eval_func(vals[inst_name][par])
                    minval = min(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    maxval = max(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    func_name = inst_name + "." + _set_func_prefix + par.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value, states[inst_name][par], minval, maxval)

    def UpdateInstrumentConfigurationNew(
        self, editable, eval_func, grid_parameters, model_inst_params, old_vals, states, vals
    ):
        self.instruments = {}
        for inst_name in vals:
            new_instrument = False
            if inst_name not in self.instruments:
                self.instruments[inst_name] = self.model.Instrument()
                new_instrument = True
            inst = self.instruments[inst_name]
            for name, value_info in inst._parameter_info().items():
                if not states[inst_name][name]:
                    orig_type = value_info.type
                    e_value = vals[inst_name][name] if orig_type is str else eval_func(vals[inst_name][name])
                    setattr(self.instruments[inst_name], name, orig_type(e_value))
                elif states[inst_name][name] != 3:
                    setattr(self.instruments[inst_name], name, old_vals[inst_name][name])
                if states[inst_name][name] == 3:
                    orig_type = value_info.type
                    e_value = getattr(inst, name)
                    setattr(self.instruments[inst_name], name, orig_type(e_value))
                elif new_instrument and states[inst_name][name] > 0:
                    value = eval_func(vals[inst_name][name])
                    minval = min(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    maxval = max(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    func_name = inst_name + "." + _set_func_prefix + name.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value, states[inst_name][name], minval, maxval)
                elif not new_instrument and editable[inst_name][name] != states[inst_name][name]:
                    value = eval_func(vals[inst_name][name])
                    minval = min(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    maxval = max(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    func_name = inst_name + "." + _set_func_prefix + name.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value, states[inst_name][name], minval, maxval)

    def ExtractInstrumentParams(self):
        if isinstance(self.sampleh.sample, ReflBaseNew):
            return self.ExtractInstrumentParamsNew()
        validators = {}
        vals = {}
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()
        for inst_name in self.instruments:
            vals[inst_name] = {}
            editable[inst_name] = {}
        pars = []
        model_inst_params = self.model.InstrumentParameters
        for item in model_inst_params:
            validators[item] = self.model.instrument_string_choices.get(item, FloatObjectValidator())
            for inst_name in self.instruments:
                val = getattr(self.instruments[inst_name], item)
                vals[inst_name][item] = val
                func_name = inst_name + "." + _set_func_prefix + item.capitalize()
                grid_value = grid_parameters.get_value_by_name(func_name)
                editable[inst_name][item] = grid_parameters.get_fit_state_by_name(func_name)
                if grid_value is not None:
                    vals[inst_name][item] = grid_value
            pars.append(item)
        groups = getattr(self.model, "InstrumentGroups", False)
        units = getattr(self.model, "InstrumentUnits", False)
        return editable, grid_parameters, model_inst_params, pars, validators, vals, groups, units

    def ExtractInstrumentParamsNew(self):
        validators = {}
        vals = {}
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()
        for inst_name in self.instruments:
            vals[inst_name] = {}
            editable[inst_name] = {}
        pars = []
        inst = self.model.Instrument()
        for name, value_info in inst._parameter_info().items():
            validators[name] = self.model.instrument_string_choices.get(name, FloatObjectValidator())
            for inst_name in self.instruments:
                val = getattr(self.instruments[inst_name], name)
                vals[inst_name][name] = val
                func_name = inst_name + "." + _set_func_prefix + name.capitalize()
                grid_value = grid_parameters.get_value_by_name(func_name)
                editable[inst_name][name] = grid_parameters.get_fit_state_by_name(func_name)
                if grid_value is not None:
                    vals[inst_name][name] = grid_value
            pars.append(name)
        groups = getattr(self.model, "InstrumentGroups", False)
        units = getattr(self.model, "InstrumentUnits", False)
        return editable, grid_parameters, inst._parameter_info(), pars, validators, vals, groups, units

    def InsertLay(self, _evt=None):
        pos = self.listbox.GetSelection()
        if pos < 0:
            pos = len(self.sampleh.names) - 2
        sl = self.sampleh.insertItem(pos, "Layer", "L%s" % (len(self.sampleh.names) - 2))
        if sl:
            self.Update()
        else:
            self.plugin.ShowWarningDialog(
                "Can not insert a layer at the current position. Layers has to be part of a stack."
            )

    def InsertStack(self, _evt=None):
        pos = self.listbox.GetSelection()
        if pos < 0:
            pos = len(self.sampleh.names) - 2
        sl = self.sampleh.insertItem(pos, "Stack", "St%s" % (len(self.sampleh.names)))
        if sl:
            self.Update()

    def DeleteSample(self, _evt=None):
        sl = self.sampleh.deleteItem(self.listbox.GetSelection())
        if sl:
            self.Update()

    def ChangeName(self, _evt=None):
        pos = self.listbox.GetSelection()
        if pos == 0 or pos == len(self.sampleh.names) - 1:
            self.plugin.ShowInfoDialog(
                "It is forbidden to change the name of the substrate (Sub) and the Ambient (Amb) layers."
            )
        else:
            unallowed_names = self.sampleh.names[:pos] + self.sampleh.names[max(0, pos - 1) :]
            dlg = ValidateDialog(
                self,
                ["Name"],
                {"Name": self.sampleh.names[pos]},
                {"Name": NoMatchValidTextObjectValidator(unallowed_names)},
                title="Give New Name",
            )
            if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
                vals = dlg.GetValues()
                result = self.sampleh.changeName(pos, vals["Name"])
                if result:
                    self.Update()
                else:
                    iprint("Unexpected problems when changing name...")
            dlg.close()

    def lbDoubleClick(self, _evt):
        sel = self.sampleh.getItem(self.listbox.GetSelection())
        if isinstance(sel, self.model.Layer):
            sl = self.EditLayer(sel)
        else:
            sl = self.EditStack(sel)
        if sl:
            self.Update()

    def EditStack(self, sel):
        if isinstance(sel, ReflBaseNew):
            return self.EditStackNew()
        obj_name = self.sampleh.getName(self.listbox.GetSelection())
        eval_func = self.plugin.GetModel().eval_in_model
        validators = {}
        vals = {}
        pars = []
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()
        for item in list(self.model.StackParameters.keys()):
            if item == "Layers":
                continue
            value = getattr(sel, item)
            if isinstance(value, float):
                validators[item] = FloatObjectValidator(eval_func, alt_types=[self.model.Stack])
            else:
                validators[item] = TextObjectValidator()
            vals[item] = value
            pars.append(item)
            func_name = obj_name + "." + _set_func_prefix + item.capitalize()
            grid_value = grid_parameters.get_value_by_name(func_name)
            editable[item] = grid_parameters.get_fit_state_by_name(func_name)
            if grid_value is not None:
                vals[item] = grid_value
        groups = getattr(self.model, "StackGroups", False)
        units = getattr(self.model, "StackUnits", False)
        dlg = ValidateFitDialog(
            self, pars, vals, validators, title="Layer Editor", groups=groups, units=units, editable_pars=editable
        )
        sl = None
        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            vals = dlg.GetValues()
            states = dlg.GetStates()
            for par in pars:
                if not states[par]:
                    setattr(sel, par, vals[par])
                if editable[par] != states[par]:
                    value = eval_func(vals[par])
                    minval = min(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    maxval = max(value * (1 - self.variable_span), value * (1 + self.variable_span))
                    func_name = obj_name + "." + _set_func_prefix + par.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value, states[par], minval, maxval)
                    self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
            sl = self.sampleh.getStringList()
        dlg.close()
        return sl

    def EditStackNew(self):
        obj_name = self.sampleh.getName(self.listbox.GetSelection())
        eval_func = self.plugin.GetModel().eval_in_model
        grid_parameters = self.plugin.GetModel().get_parameters()
        editor = ReflClassEditor(self, obj_name, eval_func, grid_parameters, ignore_attributes=["Layers"])
        if editor.ShowDialog():
            self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
            self.sampleh.sample = self.plugin.GetModel().script_module.sample
            return self.sampleh.getStringListNew()
        return None

    def EditLayer(self, sel):
        if isinstance(sel, ReflBaseNew):
            return self.EditLayerNew()
        obj_name = self.sampleh.getName(self.listbox.GetSelection())
        eval_func = self.plugin.GetModel().eval_in_model
        validators = {}
        vals = {}
        pars = []
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()
        for item in self.model.LayerParameters.keys():
            value = getattr(sel, item)
            vals[item] = value
            if type(self.model.LayerParameters[item]) != type(1 + 1.0j):
                validators[item] = FloatObjectValidator(eval_func, alt_types=[self.model.Layer])
                func_name = obj_name + "." + _set_func_prefix + item.capitalize()
                grid_value = grid_parameters.get_value_by_name(func_name)
                if grid_value is not None:
                    vals[item] = grid_value
                editable[item] = grid_parameters.get_fit_state_by_name(func_name)
            else:
                validators[item] = ComplexObjectValidator(eval_func, alt_types=[self.model.Layer])
                func_name = obj_name + "." + _set_func_prefix + item.capitalize()
                grid_value_real = grid_parameters.get_value_by_name(func_name + "real")
                grid_value_imag = grid_parameters.get_value_by_name(func_name + "imag")
                if grid_value_real is not None:
                    v = eval_func(vals[item]) if isinstance(vals[item], str) else vals[item]
                    vals[item] = grid_value_real + v.imag * 1.0j
                if grid_value_imag is not None:
                    v = eval_func(vals[item]) if isinstance(vals[item], str) else vals[item]
                    vals[item] = v.real + grid_value_imag * 1.0j
                editable[item] = max(
                    grid_parameters.get_fit_state_by_name(func_name + "real"),
                    grid_parameters.get_fit_state_by_name(func_name + "imag"),
                )
            pars.append(item)
        groups = getattr(self.model, "LayerGroups", False)
        units = getattr(self.model, "LayerUnits", False)
        dlg = ValidateFitDialog(
            self, pars, vals, validators, title="Layer Editor", groups=groups, units=units, editable_pars=editable
        )
        sl = None
        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            vals = dlg.GetValues()
            states = dlg.GetStates()
            for par in pars:
                if not states[par]:
                    setattr(sel, par, vals[par])
                if editable[par] != states[par]:
                    value = eval_func(vals[par])
                    if type(value) is complex:
                        func_name = obj_name + "." + _set_func_prefix + par.capitalize() + "real"
                        val = value.real
                        minval = min(val * (1 - self.variable_span), val * (1 + self.variable_span))
                        maxval = max(val * (1 - self.variable_span), val * (1 + self.variable_span))
                        grid_parameters.set_fit_state_by_name(func_name, val, states[par], minval, maxval)
                        val = value.imag
                        minval = min(val * (1 - self.variable_span), val * (1 + self.variable_span))
                        maxval = max(val * (1 - self.variable_span), val * (1 + self.variable_span))
                        func_name = obj_name + "." + _set_func_prefix + par.capitalize() + "imag"
                        grid_parameters.set_fit_state_by_name(func_name, val, states[par], minval, maxval)
                    else:
                        val = value
                        minval = min(val * (1 - self.variable_span), val * (1 + self.variable_span))
                        maxval = max(val * (1 - self.variable_span), val * (1 + self.variable_span))
                        func_name = obj_name + "." + _set_func_prefix + par.capitalize()
                        grid_parameters.set_fit_state_by_name(func_name, value, states[par], minval, maxval)
                    self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
            sl = self.sampleh.getStringList()
        dlg.close()
        return sl

    def EditLayerNew(self):
        obj_name = self.sampleh.getName(self.listbox.GetSelection())
        eval_func = self.plugin.GetModel().eval_in_model
        grid_parameters = self.plugin.GetModel().get_parameters()
        editor = ReflClassEditor(self, obj_name, eval_func, grid_parameters)
        if editor.ShowDialog():
            self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
            self.sampleh.sample = self.plugin.GetModel().script_module.sample
            return self.sampleh.getStringListNew()
        return None

    def MoveUp(self, _evt=None):
        if self.sampleh.moveUp(self.listbox.GetSelection()):
            self.Update()

    def MoveDown(self, _evt=None):
        if self.sampleh.moveDown(self.listbox.GetSelection()):
            self.Update()

class DataParameterPanel(QtWidgets.QWidget):
    def __init__(self, parent, plugin):
        super().__init__(parent)
        self.plugin = plugin
        self.command_indent = "<pre>   "
        self.script_update_func = None
        self.parameterlist = []

        layout = QtWidgets.QVBoxLayout(self)
        self.toolbar = QtWidgets.QToolBar(self)
        self._setup_toolbar()
        layout.addWidget(self.toolbar, 0)
        self.listbox = MyHtmlListBox(self)
        self.listbox.itemDoubleClicked.connect(self.Edit)
        layout.addWidget(self.listbox, 1)

    def _setup_toolbar(self):
        actions = [
            ("Insert", "add", self.Insert),
            ("Delete", "delete", self.Delete),
            ("User Variables", "custom_parameters", self.EditPars),
        ]
        for label, icon_name, callback in actions:
            action = self.toolbar.addAction(QtGui.QIcon(f":/reflectivity_plugin/{icon_name}.png"), label)
            action.triggered.connect(callback)

    def onsimulate(self, event=None):
        self.plugin.parent.eh_tb_simulate(event)

    def SetDataList(self, datalist):
        self.datalist = datalist

    def GetDataList(self):
        return self.datalist

    def SetParameterList(self, parameterlist):
        self.parameterlist = parameterlist

    def GetParameterList(self):
        return self.parameterlist

    def SetExpressionList(self, expressionlist):
        if len(expressionlist) != len(self.datalist):
            raise ValueError("The list of expression has to have the same length as the data list")
        self.expressionlist = expressionlist

    def GetExpressionList(self):
        return self.expressionlist

    def SetSimArgs(self, sim_funcs, insts, args):
        if len(sim_funcs) != len(self.datalist):
            raise ValueError("The list of sim_funcs has to have the same length as the data list")
        if len(insts) != len(self.datalist):
            raise ValueError("The list of insts has to have the same length as the data list")
        if len(args) != len(self.datalist):
            raise ValueError("The list of args has to have the same length as the data list")
        self.sim_funcs = sim_funcs[:]
        self.insts = insts[:]
        self.args = args[:]

    def GetSimArgs(self):
        return self.sim_funcs, self.insts, self.args

    def AppendSim(self, sim_func, inst, args):
        self.sim_funcs.append(sim_func)
        self.insts.append(inst)
        self.args.append(args)

    def InstrumentNameChange(self, old_name, new_name):
        for i in range(len(self.insts)):
            if self.insts[i] == old_name:
                self.insts[i] = new_name
        self.update_listbox()

    def SetUpdateScriptFunc(self, func):
        self.script_update_func = func

    def UpdateListbox(self, update_script=True):
        self.update_listbox()
        if self.script_update_func and update_script:
            self.script_update_func(None)
        self.update()

    def update_listbox(self):
        list_strings = []
        for i in range(len(self.datalist)):
            str_arg = ", ".join(self.args[i])
            list_strings.append(
                "<code><b>%s</b>: %s(%s, %s)</code> \n" % (self.datalist[i], self.sim_funcs[i], str_arg, self.insts[i])
            )
            for item in self.expressionlist[i]:
                list_strings.append(self.command_indent + "%s</pre>" % item)
        self.listbox.SetItemList(list_strings)

    def get_expression_position(self):
        index = self.listbox.GetSelection()
        if index < 0:
            return -1, -1
        dataindex = -1
        itemindex = -1
        listindex = -1
        for i in range(len(self.datalist)):
            dataindex += 1
            listindex += 1
            itemindex = -1
            if listindex >= index:
                return dataindex, itemindex
            for _item in self.expressionlist[i]:
                listindex += 1
                itemindex += 1
                if listindex >= index:
                    return dataindex, itemindex
        return -1, -1

    def Edit(self, _event):
        data_pos, exp_pos = self.get_expression_position()
        if exp_pos != -1 and data_pos != -1:
            list_item = self.expressionlist[data_pos][exp_pos]
            dlg = ParameterExpressionDialog(self, self.plugin.GetModel(), list_item, sim_func=self.onsimulate)
            if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
                exp = dlg.GetExpression()
                self.expressionlist[data_pos][exp_pos] = exp
                self.UpdateListbox()
        if exp_pos == -1 and data_pos != -1:
            dlg = SimulationExpressionDialog(
                self,
                self.plugin.GetModel(),
                self.plugin.sample_widget.instruments,
                self.sim_funcs[data_pos],
                self.args[data_pos],
                self.insts[data_pos],
                data_pos,
            )
            if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
                self.args[data_pos] = dlg.GetExpressions()
                self.insts[data_pos] = dlg.GetInstrument()
                self.sim_funcs[data_pos] = dlg.GetSim()
                self.UpdateListbox()

    def Insert(self, _event):
        data_pos, exp_pos = self.get_expression_position()
        if data_pos != -1:
            dlg = ParameterExpressionDialog(self, self.plugin.GetModel(), sim_func=self.onsimulate)
            if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
                exp = dlg.GetExpression()
                if exp_pos == -1:
                    self.expressionlist[data_pos].insert(0, exp)
                else:
                    self.expressionlist[data_pos].insert(exp_pos, exp)
                self.UpdateListbox()

    def Delete(self, _event):
        data_pos, exp_pos = self.get_expression_position()
        if exp_pos != -1 and data_pos != -1:
            self.expressionlist[data_pos].pop(exp_pos)
            self.UpdateListbox()

    def EditPars(self, _event):
        dlg = EditCustomParameters(self, self.plugin.GetModel(), self.parameterlist)
        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            self.parameterlist = dlg.GetLines()
            self.UpdateListbox()
        dlg.close()

    def OnDataChanged(self, _event):
        self.UpdateListbox(update_script=False)

class EditCustomParameters(QtWidgets.QDialog):
    model: Model

    def __init__(self, parent, model, lines):
        super().__init__(parent)
        self.setWindowTitle("Custom parameter editor")
        self.model = model
        self.lines = lines
        self.var_name = "cp"

        layout = QtWidgets.QVBoxLayout(self)
        grid = QtWidgets.QGridLayout()
        col_labels = ["Name", "Value", "Sigma (for systematic error)"]
        for idx, text in enumerate(col_labels):
            grid.addWidget(QtWidgets.QLabel(text), 0, idx)
        self.name_ctrl = QtWidgets.QLineEdit(self)
        self.value_ctrl = QtWidgets.QLineEdit(self)
        self.error_ctrl = QtWidgets.QLineEdit(self)
        self.add_button = QtWidgets.QPushButton("Add", self)
        self.add_button.clicked.connect(self.OnAdd)
        grid.addWidget(self.name_ctrl, 1, 0)
        grid.addWidget(self.value_ctrl, 1, 1)
        grid.addWidget(self.error_ctrl, 1, 2)
        grid.addWidget(self.add_button, 1, 3)
        layout.addLayout(grid)

        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))
        self.listbox = MyHtmlListBox(self)
        self.listbox.SetItemList(self.lines)
        layout.addWidget(self.listbox, 1)

        self.delete_button = QtWidgets.QPushButton("Delete", self)
        self.delete_button.clicked.connect(self.OnDelete)
        layout.addWidget(self.delete_button, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def OnAdd(self):
        sigma = self.error_ctrl.text()
        if sigma.strip() == "":
            line = "%s.new_var('%s', %s)" % (self.var_name, self.name_ctrl.text(), self.value_ctrl.text())
        else:
            line = "%s.new_sys_err('%s', %s, %s)" % (
                self.var_name,
                self.name_ctrl.text(),
                self.value_ctrl.text(),
                sigma,
            )
        try:
            self.model.eval_in_model(line)
        except Exception as exc:
            result = "Could not evaluate the expression. The python error is: \n" + repr(exc)
            ShowWarningDialog(self, result, "Error in expression")
        else:
            self.lines.append(line)
            self.listbox.SetItemList(self.lines)

    def OnDelete(self):
        result = "Do you want to delete the expression?\nRemember to check if parameter is used elsewhere!"
        if ShowQuestionDialog(self, result, "Delete expression?"):
            self.lines.pop(self.listbox.GetSelection())
            self.listbox.SetItemList(self.lines)

    def GetLines(self):
        return self.lines

    def ShowModal(self):
        return self.exec()


class SimulationExpressionDialog(QtWidgets.QDialog):
    model: Model

    def __init__(self, parent, model, instruments, sim_func, arguments, inst_name, data_index):
        super().__init__(parent)
        self.setWindowTitle("Simulation editor")
        if not model.compiled:
            model.compile_script()

        self.model = model
        self.instruments = instruments
        self.available_sim_funcs = list(self.model.eval_in_model("model.SimulationFunctions.keys()"))
        self.data_index = data_index

        max_val = -1
        self.sim_args = {}
        self.sim_defaults = {}
        for func in self.available_sim_funcs:
            doc = self.model.eval_in_model("model.SimulationFunctions" '["%s"].__doc__' % func)
            doc_lines = find_code_segment(doc, "Parameters").splitlines()
            max_val = max(len(doc_lines), max_val)
            args = []
            defaults = []
            for line in doc_lines:
                items = line.lstrip().rstrip().split(" ")
                args.append(items[0])
                defaults.append(items[1].replace("data", "d"))
            self.sim_args[func] = args
            self.sim_defaults[func] = defaults

        expressions = {"Instrument": inst_name}
        for arg_name, arg in zip(self.sim_args[sim_func], arguments):
            expressions[arg_name] = arg

        if max_val < 0:
            raise ValueError("Wrongly formatted function docs for the simulation functions")

        grid = QtWidgets.QGridLayout()
        col_labels = ["Simulation", "Instrument"]
        [col_labels.append(arg) for arg in self.sim_args[sim_func] if arg not in col_labels]
        self.labels = []
        self.arg_controls = []
        for index in range(2 + max_val):
            label = QtWidgets.QLabel("", self)
            grid.addWidget(label, 0, index)
            self.labels.append(label)
            if index > 1:
                exp_ctrl = QtWidgets.QLineEdit(self)
                grid.addWidget(exp_ctrl, 1, index)
                self.arg_controls.append(exp_ctrl)

        for item, label in zip(col_labels[:2], self.labels[:2]):
            label.setText(item)
        for item, label, arg_ctrl in zip(col_labels[2:], self.labels[2:], self.arg_controls):
            label.setText(item)
            arg_ctrl.setText(expressions[item])
            arg_ctrl.setReadOnly(False)
        for i in range(len(col_labels) - 2, len(self.arg_controls)):
            self.arg_controls[i].setReadOnly(True)

        self.sim_choice = QtWidgets.QComboBox(self)
        self.sim_choice.addItems(self.available_sim_funcs)
        self.sim_choice.currentIndexChanged.connect(self.on_sim_change)
        self.sim_choice.setCurrentIndex(self.available_sim_funcs.index(sim_func))
        grid.addWidget(self.sim_choice, 1, 0)

        self.inst_choice = QtWidgets.QComboBox(self)
        self.inst_choice.addItems(list(self.instruments.keys()))
        self.inst_choice.setCurrentIndex(list(self.instruments.keys()).index(expressions["Instrument"]))
        grid.addWidget(self.inst_choice, 1, 1)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.on_ok_button)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))
        layout.addWidget(buttons)

    def on_sim_change(self):
        new_sim = self.sim_choice.currentText()
        for label, arg_name in zip(self.labels[2:], self.sim_args[new_sim]):
            label.setText(arg_name)
        for label in self.labels[len(self.sim_args[new_sim]) + 2 :]:
            label.setText("")
        for i in range(len(self.sim_args[new_sim])):
            self.arg_controls[i].setText(self.sim_defaults[new_sim][i])
            self.arg_controls[i].setReadOnly(False)
        for ctrl in self.arg_controls[len(self.sim_args[new_sim]) :]:
            ctrl.setReadOnly(True)
            ctrl.setText("")

    def on_ok_button(self):
        expressions = self.GetExpressions()
        exec("d = data[%d]" % self.data_index, self.model.script_module.__dict__)
        for exp in expressions:
            try:
                self.model.eval_in_model(exp)
            except Exception as exc:
                result = "Could not evaluate expression:\n%s.\n The python error is: \n" % exp + repr(exc)
                ShowWarningDialog(self, result, "Error in expression")
                return
        self.accept()

    def GetExpressions(self):
        return [ctrl.text() for ctrl in self.arg_controls if not ctrl.isReadOnly()]

    def GetInstrument(self):
        return self.inst_choice.currentText()

    def GetSim(self):
        return self.sim_choice.currentText()

    def ShowModal(self):
        return self.exec()


class ParameterExpressionDialog(QtWidgets.QDialog):
    model: Model

    def __init__(self, parent, model, expression=None, sim_func=None):
        super().__init__(parent)
        self.setWindowTitle("Parameter editor")
        self.model = model
        self.sim_func = sim_func
        if not model.compiled:
            model.compile_script()

        grid = QtWidgets.QGridLayout()
        col_labels = ["Object", "Parameter", "Expression"]
        for index, item in enumerate(col_labels):
            grid.addWidget(QtWidgets.QLabel(item, self), 0, index)

        par_dict = model.get_possible_parameters()
        objlist = []
        funclist = []
        for cl in par_dict:
            obj_dict = par_dict[cl]
            for obj in obj_dict:
                objlist.append(obj)
                funclist.append(obj_dict[obj])
        self.objlist = objlist
        self.funclist = funclist

        self.obj_choice = QtWidgets.QComboBox(self)
        self.obj_choice.addItems(objlist)
        self.obj_choice.currentIndexChanged.connect(self.on_obj_change)

        self.func_choice = QtWidgets.QComboBox(self)
        self.obj_choice.setCurrentIndex(0)
        self.on_obj_change(None)

        grid.addWidget(self.obj_choice, 1, 0)
        grid.addWidget(self.func_choice, 1, 1)

        exp_right = ""
        if expression:
            p = expression.find("(")
            exp_left = expression[:p]
            obj = exp_left.split(".")[0]
            func = exp_left.split(".")[1]
            exp_right = expression[p + 1 : -1]
            obj_pos = [i for i in range(len(objlist)) if objlist[i] == obj]
            if obj_pos:
                self.obj_choice.setCurrentIndex(obj_pos[0])
                self.on_obj_change(None)
                func_pos = [i for i in range(len(funclist[obj_pos[0]])) if funclist[obj_pos[0]][i] == func]
                if func_pos:
                    self.func_choice.setCurrentIndex(func_pos[0])
                else:
                    raise ValueError("The function %s for object %s does not exist" % (func, obj))
            else:
                raise ValueError("The object %s does not exist" % obj)

        self.expression_ctrl = ParameterExpressionCombo(par_dict, sim_func, self, exp_right)
        grid.addWidget(self.expression_ctrl, 1, 2)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self.OnApply)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(QtWidgets.QFrame(frameShape=QtWidgets.QFrame.HLine))
        layout.addWidget(buttons)

    def on_obj_change(self, _event):
        index = self.obj_choice.currentIndex()
        self.func_choice.clear()
        self.func_choice.addItems(self.funclist[index])

    def OnApply(self):
        evalstring = self.GetExpression()
        try:
            self.model.eval_in_model(evalstring)
        except Exception as exc:
            result = "Could not evaluate the expression. The python is: \n" + repr(exc)
            ShowWarningDialog(self, result, "Error in expression")
            return
        self.accept()

    def GetExpression(self):
        objstr = self.obj_choice.currentText()
        funcstr = self.func_choice.currentText()
        set_expression = self.expression_ctrl.text()
        return "%s.%s(%s)" % (objstr, funcstr, set_expression)

    def ShowModal(self):
        return self.exec()
