import math
import string
from logging import debug, info

from PySide6 import QtCore, QtGui, QtWidgets

from genx.core.custom_logging import iprint


def is_reflfunction(obj):
    return obj.__class__.__name__ == "ReflFunction"


class BaseValidator:
    def Clone(self):
        return self.__class__()

    def Validate(self, text, parent=None):
        return True


class ValueValidator(BaseValidator):
    def __init__(self, cls):
        self.valid_cls = cls

    def Clone(self):
        return ValueValidator(self.valid_cls)

    def Validate(self, text, parent=None):
        if len(text) == 0:
            QtWidgets.QMessageBox.warning(parent, "Error", "A text object must contain some text!")
            return False
        try:
            self.valid_cls(text)
        except ValueError:
            return False
        return True


class TextObjectValidator(BaseValidator):
    def Clone(self):
        return TextObjectValidator()

    def Validate(self, text, parent=None):
        if len(text) == 0:
            QtWidgets.QMessageBox.warning(parent, "Error", "A text object must contain some text!")
            return False
        return True


class MatchTextObjectValidator(BaseValidator):
    def __init__(self, stringlist):
        self.stringlist = stringlist

    def Clone(self):
        return MatchTextObjectValidator(self.stringlist)

    def Validate(self, text, parent=None):
        if text in self.stringlist:
            return True
        QtWidgets.QMessageBox.warning(parent, "Error", "The name is not defined!")
        return False


class NoMatchTextObjectValidator(BaseValidator):
    def __init__(self, stringlist):
        self.stringlist = stringlist

    def Clone(self):
        return NoMatchTextObjectValidator(self.stringlist)

    def Validate(self, text, parent=None):
        if len(text) == 0:
            QtWidgets.QMessageBox.warning(parent, "Error", "A text object must contain some text!")
            return False
        if text in self.stringlist:
            QtWidgets.QMessageBox.warning(parent, "Error", "Duplicates are not allowed!")
            return False
        return True


class NoMatchValidTextObjectValidator(BaseValidator):
    def __init__(self, stringlist):
        self.stringlist = stringlist
        self.reserved_words = {
            "and",
            "del",
            "from",
            "not",
            "while",
            "as",
            "elif",
            "global",
            "or",
            "with",
            "assert",
            "else",
            "if",
            "pass",
            "yield",
            "break",
            "except",
            "import",
            "print",
            "class",
            "exec",
            "in",
            "raise",
            "continue",
            "finally",
            "is",
            "return",
            "def",
            "for",
            "lambda",
            "try",
        }
        self.allowed_chars = string.digits + string.ascii_letters + "_"

    def Clone(self):
        return NoMatchValidTextObjectValidator(self.stringlist)

    def Validate(self, text, parent=None):
        if len(text) == 0:
            QtWidgets.QMessageBox.warning(parent, "Bad Input", "A text object must contain some text!")
            return False
        if text in self.stringlist:
            QtWidgets.QMessageBox.warning(parent, "Bad Input", "Duplicates are not allowed!")
            return False
        if text in self.reserved_words:
            QtWidgets.QMessageBox.warning(parent, "Bad Input", "Python keywords are not allowed!")
            return False
        if sum([char in self.allowed_chars for char in text]) != len(text) or text[0] in string.digits:
            QtWidgets.QMessageBox.warning(
                parent,
                "Bad Input",
                "Not a valid name. Names can only contain letters, digits and underscores(_) and not start with a digit.",
            )
            return False
        return True


class NoMatchTextCtrlValidator(BaseValidator):
    def __init__(self, textctrls):
        self.textctrls = textctrls[:]

    def set_nomatch_ctrls(self, textctrls):
        self.textctrls = textctrls[:]

    def Clone(self):
        return NoMatchTextCtrlValidator(self.textctrls)

    def Validate(self, text, parent=None):
        stringlist = [ctrl.text() for ctrl in self.textctrls]
        iprint(text, stringlist)
        if len(text) == 0:
            QtWidgets.QMessageBox.warning(parent, "Error", "A text object must contain some text!")
            return False
        if text in stringlist:
            QtWidgets.QMessageBox.warning(parent, "Error", "Duplicates are not allowed!")
            return False
        return True


class FloatObjectValidator(BaseValidator):
    def __init__(self, eval_func=eval, alt_types=None):
        self.value = None
        self.eval_func = eval_func
        self.alt_types = alt_types or []

    def Clone(self):
        return FloatObjectValidator(self.eval_func, self.alt_types)

    def Validate(self, text, parent=None):
        self.value = None
        try:
            val = self.eval_func(text)
            if is_reflfunction(val):
                val = val.validate()
        except Exception as exc:
            info("Can't evaluate the expression", exc_info=True)
            QtWidgets.QMessageBox.warning(parent, "Error", f"Can't evaluate the expression!!\nERROR:\n{exc}")
            return False
        try:
            self.value = float(val)
        except Exception as exc:
            if not any([isinstance(val, typ) for typ in self.alt_types]):
                info("Wrong type of parameter", exc_info=True)
                QtWidgets.QMessageBox.warning(parent, "Error", f"\nERROR:\n{exc}")
                return False
        return True


class ComplexObjectValidator(BaseValidator):
    def __init__(self, eval_func=eval, alt_types=None):
        self.value = None
        self.eval_func = eval_func
        self.alt_types = alt_types or []

    def Clone(self):
        return ComplexObjectValidator(self.eval_func, self.alt_types)

    def Validate(self, text, parent=None):
        self.value = None
        try:
            val = self.eval_func(text)
        except Exception as exc:
            info("Can't evaluate the expression", exc_info=True)
            QtWidgets.QMessageBox.warning(parent, "Error", f"Can't compile the complex expression!!\nERROR:\n{exc}")
            return False
        try:
            if is_reflfunction(val):
                val = val.validate()
            self.value = complex(val.real, val.imag)
        except AttributeError:
            try:
                self.value = complex(val)
            except Exception as exc:
                if not any([isinstance(val, typ) for typ in self.alt_types]):
                    info("Can't evaluate the complex expression, wrong type", exc_info=True)
                    QtWidgets.QMessageBox.warning(parent, "Error", f"Can't evaluate the complex expression!!\nERROR:\n{exc}")
                    return False
        except Exception as exc:
            if not any([isinstance(val, typ) for typ in self.alt_types]):
                info("Can't evaluate the complex expression", exc_info=True)
                QtWidgets.QMessageBox.warning(parent, "Error", f"Can't evaluate the complex expression!!\nERROR:\n{exc}")
                return False
        return True


class FitSelectorCombo(QtWidgets.QWidget):
    def __init__(self, state, parent=None, text="", validator=None):
        super().__init__(parent)
        self.state = state
        self.validator = validator

        self.line_edit = QtWidgets.QLineEdit(str(text), self)
        self.button = QtWidgets.QToolButton(self)
        self.button.setText("F")
        self.button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)

        menu = QtWidgets.QMenu(self)
        self.action_define = menu.addAction("Define parameter here")
        self.action_fit = menu.addAction("Fit parameter")
        self.action_const = menu.addAction("Constant fit parameter")
        self.action_define.setCheckable(True)
        self.action_fit.setCheckable(True)
        self.action_const.setCheckable(True)
        self.action_group = QtGui.QActionGroup(self)
        for action in (self.action_define, self.action_fit, self.action_const):
            self.action_group.addAction(action)
        self.button.setMenu(menu)

        if self.state == 0:
            self.action_define.setChecked(True)
        elif self.state == 1:
            self.action_fit.setChecked(True)
        else:
            self.action_const.setChecked(True)

        self.action_group.triggered.connect(self._on_state_change)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.button, 0)
        self.update_text_state()

    def _on_state_change(self, action):
        if action == self.action_define:
            self.state = 0
        elif action == self.action_fit:
            self.state = 1
        else:
            self.state = 2
        self.update_text_state()

    def update_text_state(self):
        self.line_edit.setReadOnly(self.state != 0)
        font = self.line_edit.font()
        if self.state == 0:
            self.line_edit.setStyleSheet("")
            font.setWeight(QtGui.QFont.Weight.Normal)
        elif self.state == 1:
            self.line_edit.setStyleSheet("color: rgb(245, 121, 0);")
            font.setWeight(QtGui.QFont.Weight.Bold)
        elif self.state == 2:
            self.line_edit.setStyleSheet("color: rgb(114, 159, 207);")
            font.setWeight(QtGui.QFont.Weight.Bold)
        self.line_edit.setFont(font)

    def text(self):
        return self.line_edit.text()

    def setText(self, value):
        self.line_edit.setText(str(value))


class ParameterExpressionCombo(QtWidgets.QWidget):
    def __init__(self, par_dict, sim_func, parent=None, text=""):
        super().__init__(parent)
        self.par_dict = par_dict
        self.sim_func = sim_func

        self.line_edit = QtWidgets.QLineEdit(str(text), self)
        self.button = QtWidgets.QToolButton(self)
        self.button.setText("P")
        self.button.clicked.connect(self._on_button_click)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.button, 0)

    def _on_button_click(self):
        menu = QtWidgets.QMenu(self)
        classes = sorted(self.par_dict.keys(), key=str.lower)
        for cl in classes:
            cl_menu = menu.addMenu(cl)
            obj_dict = self.par_dict[cl]
            objs = sorted(obj_dict.keys(), key=str.lower)
            for obj in objs:
                obj_menu = cl_menu.addMenu(obj)
                funcs = sorted(obj_dict[obj], key=str.lower)
                for func in funcs:
                    action = obj_menu.addAction(obj + "." + func.replace("set", "get"))
                    action.triggered.connect(lambda _=False, text=action.text(): self._insert_text(text))
        menu.exec(QtGui.QCursor.pos())

    def _insert_text(self, text):
        cursor = self.line_edit.cursorPosition()
        current = self.line_edit.text()
        insert_text = text + "()"
        self.line_edit.setText(current[:cursor] + insert_text + current[cursor:])
        self.line_edit.setCursorPosition(cursor + len(insert_text))

    def text(self):
        return self.line_edit.text()

    def setText(self, value):
        self.line_edit.setText(str(value))


class NormalEditMixin:
    def create_edit_ctrl(self, _state, parent, val, validator):
        if isinstance(validator, (list, tuple)):
            ctrl = QtWidgets.QComboBox(parent)
            ctrl.addItems([str(item) for item in validator])
            if isinstance(val, str) and val in validator:
                ctrl.setCurrentIndex(list(validator).index(val))
        else:
            ctrl = QtWidgets.QLineEdit(parent)
            ctrl.setText(str(val))
        return ctrl


class FitEditMixIn:
    def create_edit_ctrl(self, state, parent, val, validator):
        if isinstance(validator, (list, tuple)):
            ctrl = QtWidgets.QComboBox(parent)
            ctrl.addItems([str(item) for item in validator])
            if isinstance(val, str) and val in validator:
                ctrl.setCurrentIndex(list(validator).index(val))
        else:
            ctrl = FitSelectorCombo(state, parent, text=str(val), validator=validator)
        return ctrl

    def GetStates(self):
        p = {}
        for par in list(self.validators.keys()):
            if isinstance(self.validators[par], list):
                p[par] = 0
            else:
                p[par] = self.text_controls[par].state
        return p


class ValidateBaseDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent,
        pars,
        vals,
        validators,
        title="Validated Dialog",
        units=None,
        groups=None,
        cols=2,
        editable_pars=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.pars = pars
        self.validators = validators
        self.cols = cols
        self.vals = vals
        self.units = units or {}
        self.groups = groups or []
        self.text_controls = {}
        self.editable_pars = editable_pars or {}

        main_layout = QtWidgets.QVBoxLayout(self)
        if self.groups:
            tabs = QtWidgets.QTabWidget(self)
            for group_name, group_pars in self.groups:
                tab = QtWidgets.QWidget()
                tabs.addTab(tab, group_name)
                self._layout_params(tab, group_pars)
            main_layout.addWidget(tabs, 1)
        else:
            self._layout_params(self, self.pars)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons, 0)

    def _layout_params(self, parent, pars):
        layout = QtWidgets.QGridLayout(parent)
        for row, par in enumerate(pars):
            label = QtWidgets.QLabel(par, parent)
            layout.addWidget(label, row, 0, 1, 1)

            validator = self.validators[par]
            val = self.vals[par]
            state = self.editable_pars.get(par, 0)
            ctrl = self.create_edit_ctrl(state, parent, val, validator)

            unit = self.units.get(par, "")
            unit_label = QtWidgets.QLabel(unit, parent)
            layout.addWidget(ctrl, row, 1, 1, 1)
            layout.addWidget(unit_label, row, 2, 1, 1)

            self.text_controls[par] = ctrl

    def _on_accept(self):
        for par, validator in self.validators.items():
            ctrl = self.text_controls[par]
            if isinstance(ctrl, QtWidgets.QComboBox):
                text = ctrl.currentText()
            elif isinstance(ctrl, FitSelectorCombo):
                text = ctrl.text()
            else:
                text = ctrl.text()
            if not isinstance(validator, (list, tuple)):
                if not validator.Validate(text, self):
                    return
        self.accept()

    def GetValues(self):
        p = {}
        for par, validator in self.validators.items():
            ctrl = self.text_controls[par]
            if isinstance(validator, list):
                if isinstance(ctrl, QtWidgets.QComboBox):
                    p[par] = ctrl.currentText()
                else:
                    p[par] = ctrl.text()
            else:
                p[par] = ctrl.text() if not isinstance(ctrl, FitSelectorCombo) else ctrl.text()
        return p

    def ShowModal(self):
        return self.exec()


class ValidateDialog(ValidateBaseDialog, NormalEditMixin):
    pass


class ValidateFitDialog(ValidateBaseDialog, FitEditMixIn):
    pass


class ValidateBaseNotebookDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent,
        pars,
        vals,
        validators,
        title="Validated Dialog",
        units=None,
        groups=None,
        cols=2,
        editable_pars=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.pars = pars
        self.validators = validators
        self.cols = cols
        self.vals = vals
        self.units = units or {}
        self.groups = groups or []
        self.text_controls = {}
        self.editable_pars = editable_pars or {}

        main_layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget(self)
        main_layout.addWidget(self.tabs, 1)

        names = list(self.vals.keys())
        names.sort()
        for name in names:
            self.AddPage(name, self.vals[name], self.editable_pars.get(name, {}))

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        main_layout.addWidget(buttons, 0)

    def AddPage(self, name, vals, editable_pars, select=False):
        panel = QtWidgets.QWidget()
        self.tabs.addTab(panel, name)
        layout = QtWidgets.QGridLayout(panel)
        for row, par in enumerate(self.pars):
            label = QtWidgets.QLabel(par, panel)
            layout.addWidget(label, row, 0, 1, 1)
            validator = self.validators[par]
            val = vals[par]
            state = editable_pars.get(par, 0)
            ctrl = self.create_edit_ctrl(state, panel, val, validator)
            unit = self.units.get(par, "")
            unit_label = QtWidgets.QLabel(unit, panel)
            layout.addWidget(ctrl, row, 1, 1, 1)
            layout.addWidget(unit_label, row, 2, 1, 1)
            self.text_controls.setdefault(name, {})[par] = ctrl
        if select:
            self.tabs.setCurrentWidget(panel)

    def _on_accept(self):
        for page_name, controls in self.text_controls.items():
            for par, validator in self.validators.items():
                ctrl = controls[par]
                if isinstance(ctrl, QtWidgets.QComboBox):
                    text = ctrl.currentText()
                elif isinstance(ctrl, FitSelectorCombo):
                    text = ctrl.text()
                else:
                    text = ctrl.text()
                if not isinstance(validator, (list, tuple)):
                    if not validator.Validate(text, self):
                        return
        self.accept()

    def GetValues(self):
        p = {}
        for page, controls in self.text_controls.items():
            p[page] = {}
            for par, validator in self.validators.items():
                ctrl = controls[par]
                if isinstance(validator, list):
                    if isinstance(ctrl, QtWidgets.QComboBox):
                        p[page][par] = ctrl.currentText()
                    else:
                        p[page][par] = ctrl.text()
                else:
                    p[page][par] = ctrl.text() if not isinstance(ctrl, FitSelectorCombo) else ctrl.text()
        return p

    def ShowModal(self):
        return self.exec()


class ValidateNotebookDialog(ValidateBaseNotebookDialog, NormalEditMixin):
    pass


class ValidateFitNotebookDialog(ValidateBaseNotebookDialog, FitEditMixIn):
    def GetStates(self):
        p = {}
        for page, controls in self.text_controls.items():
            p[page] = {}
            for par, validator in self.validators.items():
                if isinstance(validator, list):
                    p[page][par] = 0
                else:
                    p[page][par] = controls[par].state
        return p


class ZoomFrame(QtWidgets.QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("X-Y Scales")

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Min"), 0, 1)
        grid.addWidget(QtWidgets.QLabel("Max"), 0, 2)
        grid.addWidget(QtWidgets.QLabel(" X: "), 1, 0)
        self.xmin = QtWidgets.QLineEdit("0")
        grid.addWidget(self.xmin, 1, 1)
        self.xmax = QtWidgets.QLineEdit("0")
        grid.addWidget(self.xmax, 1, 2)
        grid.addWidget(QtWidgets.QLabel(" Y: "), 2, 0)
        self.ymin = QtWidgets.QLineEdit("0")
        grid.addWidget(self.ymin, 2, 1)
        self.ymax = QtWidgets.QLineEdit("0")
        grid.addWidget(self.ymax, 2, 2)

        apply_button = QtWidgets.QPushButton("Apply", self)
        apply_button.clicked.connect(self.accept)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(apply_button, 0, QtCore.Qt.AlignmentFlag.AlignRight)
