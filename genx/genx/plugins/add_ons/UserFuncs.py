"""
=========
UserFuncs
=========

A plugin so that users can include their own user function into
the model script and execute them inside the GUI of GenX.<p>

The following design criteria exists:
The function has to be defined inside the script.
The function has to take zero input arguments.

Thus the following example can serve as a template:
::
    def myuserfunc():
        # Do something
        print('It works!')

This should be added somewhere in the script. This provides a great way
to, for example, dump some internal data into a file or for checking the status
of some variables.

For more complex actions you can use type annotation to allow the Plugin
to automatically build a dialog for the user to enter values.
This is implemented for int, float and str types.
If a variable is annotated with **TextIO** a subsequend file
selection dialog is shown to the user to select a file to save to.
The open file object is then passed on to the function:
::
    def my_func_params(a: int, b: float, c: float=1.0, name: str="Name", file_handler: TextIO=None):
        file_handler.write(f'a={a};b={b};c={c};name={name}')

This gives you the flexibility to perform complex tasks while still benefitting from
a GUI integration. Any number of parameters or files can be defined for your funct.
If a parameter is given as keywords, the default dialog value is taken from
the keyword default, otherwise it is 0 for numbers or an empty string.
The default for a TextIO object is ignored.
"""

import inspect
import io
import traceback
import types
import typing

from contextlib import ExitStack

import wx

from .. import add_on_framework as framework
from .help_modules.custom_dialog import ValidateDialog, ValueValidator

DEFAULT_VALUE = {int: 0, float: 0.0, str: ""}


class Plugin(framework.Template):
    userfunc_arguments: dict

    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.menu = self.NewMenu("User funcs")
        self.userfunc_arguments = {}
        self.parent = parent
        self.StatusMessage("Sucessfully loaded UserFuncs...")

    def OnSimulate(self, event):
        """OnSimulate(self, event) --> None

        Updates stuff after simulation
        """
        model = self.GetModel()
        # locate all functions in the model.script_module
        funcs = [
            eval("model.script_module.%s" % name, globals(), {"model": model})
            for name in dir(model.script_module)
            if type(eval("model.script_module.%s" % name, globals(), {"model": model})) == types.FunctionType
        ]
        # Find the functions that are defined in the script and take zero
        # input parameters
        user_funcs = [f.__name__ for f in funcs if f.__module__ == "genx_script_module" and f.__code__.co_argcount == 0]

        # Find the functions that are defined in the script and have type annotation
        # with sufficient information to create and automatic dialog
        self.userfunc_arguments = {}
        for func in [f for f in funcs if f.__module__ == "genx_script_module" and f.__code__.co_argcount != 0]:
            pars = dict(inspect.signature(func).parameters)
            args = pars.keys()
            defaults = dict([(n, p.default) for n, p in pars.items()])

            annots = getattr(func, "__annotations__")
            args_ok = True
            for ai in args:
                ani = annots.get(ai)
                if not ani in [int, float, str, typing.TextIO]:
                    args_ok = False
                    break
            if args_ok:
                name = func.__name__
                self.userfunc_arguments[name] = (args, annots, defaults)
                user_funcs.append(name)

        # Remove all the previous functions
        self.clear_menu()
        # Lets add our user functions to our custo menu
        for name in user_funcs:
            menuitem = wx.MenuItem(self.menu, wx.NewId(), name, "Evaluate %s" % name, wx.ITEM_NORMAL)
            self.menu.Append(menuitem)
            self.parent.Bind(wx.EVT_MENU, self.eh_menu_choice, menuitem)

    def clear_menu(self):
        """clear_menu(self) --> None

        Clears the menu from all items present in it
        """
        [self.menu.Remove(item) for item in self.menu.GetMenuItems()]

    def eh_menu_choice(self, event):
        """eh_menu_choice(self, event)

        event handler for the choice of an menuitem in the User Functions
        menu. Executes the function as defined in the script.
        With error catching.
        """
        menu = event.GetEventObject()
        menuitem = menu.FindItemById(event.Id)
        fname = menuitem.GetItemLabelText()
        # Now try to evaluate the function
        self.StatusMessage("Trying to evaluate %s" % fname)
        if fname in self.userfunc_arguments:
            self.exec_args_function(fname)
        else:
            self.exec_simple_function(fname)

    def exec_simple_function(self, fname):
        model = self.GetModel()
        try:
            model.eval_in_model(f"{fname}()")
        except Exception as e:
            # a bit of traceback
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tb_string = outp.getvalue()
            outp.close()
            self.ShowWarningDialog(
                "Error in evaluating the"
                " function: %s.\n The error gave the following traceback:"
                "\n%s" % (fname, tb_string)
            )
            self.StatusMessage("error in function %s" % fname)
        else:
            self.StatusMessage("%s sucessfully evaluated" % fname)

    def get_filename(self, argname):
        dia = wx.FileDialog(self.parent, f"Select file for {argname}", style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        dia.ShowModal()
        fname = dia.GetPath()
        dia.Destroy()
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
            dia = ValidateDialog(
                self.parent, dia_args, dia_vals, dia_validate, title=f"Evaluate parameters for {fname}"
            )
            res = dia.ShowModal()
            call_values = dia.GetValues()
            dia.Destroy()
            if res != wx.ID_OK:
                return
            for key, value in call_values.items():
                # convert entry value into correct type
                call_values[key] = annots[key](value)
        else:
            call_values = {}

        with ExitStack() as stack:
            for fi in file_args:
                call_values[fi] = stack.enter_context(open(self.get_filename(fi), "w", encoding="utf8"))

            call_pars = [call_values[ai] for ai in args]
            model.set_script_variable("_call_pars", call_pars)

            try:
                model.eval_in_model(f"{fname}(*_call_pars)")
            except Exception as e:
                # a bit of traceback
                outp = io.StringIO()
                traceback.print_exc(200, outp)
                tb_string = outp.getvalue()
                outp.close()
                self.ShowWarningDialog(
                    "Error in evaluating the"
                    " function: %s.\n The error gave the following traceback:"
                    "\n%s" % (fname, tb_string)
                )
                self.StatusMessage("error in function %s" % fname)
            else:
                self.StatusMessage("%s sucessfully evaluated" % fname)
            finally:
                model.unset_script_variable("_call_pars")
