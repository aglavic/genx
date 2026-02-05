"""
=========
UserFuncs
=========

Qt port of the user function plugin.
"""

import inspect
import io
import traceback
import types
import typing

from contextlib import ExitStack
from PySide6 import QtWidgets

from genx.plugins import add_on_framework as framework
from .help_modules.custom_dialog import ValidateDialog, ValueValidator

DEFAULT_VALUE = {int: 0, float: 0.0, str: ""}


class Plugin(framework.Template):
    userfunc_arguments: dict

    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.menu = self.NewMenu("User funcs")
        self.userfunc_arguments = {}
        self.parent = parent
        self.StatusMessage("Successfully loaded UserFuncs...")

    def OnSimulate(self, _event):
        model = self.GetModel()
        funcs = [
            eval("model.script_module.%s" % name, globals(), {"model": model})
            for name in dir(model.script_module)
            if isinstance(eval("model.script_module.%s" % name, globals(), {"model": model}), types.FunctionType)
        ]
        user_funcs = [f.__name__ for f in funcs if f.__module__ == "genx_script_module" and f.__code__.co_argcount == 0]

        self.userfunc_arguments = {}
        for func in [f for f in funcs if f.__module__ == "genx_script_module" and f.__code__.co_argcount != 0]:
            pars = dict(inspect.signature(func).parameters)
            args = pars.keys()
            defaults = dict([(n, p.default) for n, p in pars.items()])

            annots = getattr(func, "__annotations__")
            args_ok = True
            for ai in args:
                ani = annots.get(ai)
                if ani not in [int, float, str, typing.TextIO]:
                    args_ok = False
                    break
            if args_ok:
                name = func.__name__
                self.userfunc_arguments[name] = (args, annots, defaults)
                user_funcs.append(name)

        self.clear_menu()
        for name in user_funcs:
            action = self.menu.addAction(name)
            action.setToolTip("Evaluate %s" % name)
            action.triggered.connect(lambda _=False, fname=name: self.eh_menu_choice(fname))

    def clear_menu(self):
        for action in list(self.menu.actions()):
            self.menu.removeAction(action)

    def eh_menu_choice(self, fname):
        self.StatusMessage("Trying to evaluate %s" % fname)
        if fname in self.userfunc_arguments:
            self.exec_args_function(fname)
        else:
            self.exec_simple_function(fname)

    def exec_simple_function(self, fname):
        model = self.GetModel()
        try:
            model.eval_in_model(f"{fname}()")
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tb_string = outp.getvalue()
            outp.close()
            self.ShowWarningDialog(
                "Error in evaluating the function: %s.\nThe error gave the following traceback:\n%s" % (fname, tb_string)
            )
            self.StatusMessage("error in function %s" % fname)
        else:
            self.StatusMessage("%s successfully evaluated" % fname)

    def get_filename(self, argname):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self.parent, f"Select file for {argname}")
        return fname

    def exec_args_function(self, fname):
        model = self.GetModel()
        args, annots, defaults = self.userfunc_arguments[fname]
        dia_args = [ai for ai in args if annots[ai] is not typing.TextIO]
        dia_vals = dict(
            [
                (ai, defaults[ai]) if defaults[ai] is not inspect.Parameter.empty else (ai, DEFAULT_VALUE[annots[ai]])
                for ai in dia_args
            ]
        )
        dia_validate = dict([(ai, ValueValidator(annots[ai])) for ai in dia_args])

        file_args = [ai for ai in args if annots[ai] is typing.TextIO]

        if len(dia_args) > 0:
            dia = ValidateDialog(self.parent, dia_args, dia_vals, dia_validate, title=f"Evaluate parameters for {fname}")
            res = dia.ShowModal()
            call_values = dia.GetValues()
            dia.close()
            if res != QtWidgets.QDialog.DialogCode.Accepted:
                return
            for key, value in call_values.items():
                call_values[key] = annots[key](value)
        else:
            call_values = {}

        with ExitStack() as stack:
            for fi in file_args:
                filename = self.get_filename(fi)
                if not filename:
                    return
                call_values[fi] = stack.enter_context(open(filename, "w", encoding="utf8"))

            call_pars = [call_values[ai] for ai in args]
            model.set_script_variable("_call_pars", call_pars)

            try:
                model.eval_in_model(f"{fname}(*_call_pars)")
            except Exception:
                outp = io.StringIO()
                traceback.print_exc(200, outp)
                tb_string = outp.getvalue()
                outp.close()
                self.ShowWarningDialog(
                    "Error in evaluating the function: %s.\nThe error gave the following traceback:\n%s" % (fname, tb_string)
                )
                self.StatusMessage("error in function %s" % fname)
            else:
                self.StatusMessage("%s successfully evaluated" % fname)
            finally:
                model.unset_script_variable("_call_pars")
