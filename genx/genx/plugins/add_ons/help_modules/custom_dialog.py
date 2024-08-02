import math
import os
import string

from logging import debug, info

import wx

from wx.lib.intctrl import IntCtrl

from genx.core.custom_logging import iprint

from . import reflectivity_images as images


def is_reflfunction(obj):
    """Convenience function to determine whether obj belongs to the ReflFunction class.
    Return boolean.
    """
    return obj.__class__.__name__ == "ReflFunction"


class ValueValidator(wx.Validator):
    """
    Validate a value for a given type.
    """

    def __init__(self, cls):
        wx.Validator.__init__(self)
        self.valid_cls = cls

    def Clone(self):
        return ValueValidator(self.valid_cls)

    def Validate(self, win):
        """Validate the contents of the given text control."""
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()

        if len(text) == 0:
            debug("Must contain a value")
            wx.MessageBox("A text object must contain some text!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            try:
                self.valid_cls(text)
            except ValueError:
                textCtrl.SetBackgroundColour("pink")
                textCtrl.SetFocus()
                textCtrl.Refresh()
                return False
            else:
                textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
                textCtrl.Refresh()
                return True

    def TransferToWindow(self):
        return True

    def TransferFromWindow(self):
        return True


class TextObjectValidator(wx.Validator):
    """This validator is used to ensure that the user has entered something
    into the text object editor dialog's text field.
    """

    def __init__(self):
        """Standard constructor."""
        wx.Validator.__init__(self)

    def Clone(self):
        """Standard cloner.

        Note that every validator must implement the Clone() method.
        """
        return TextObjectValidator()

    def Validate(self, win):
        """Validate the contents of the given text control."""
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()

        if len(text) == 0:
            debug("A text object must contain some text")
            wx.MessageBox("A text object must contain some text!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """Transfer data from validator to window.

        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.

    def TransferFromWindow(self):
        """Transfer data from window to validator.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.


class MatchTextObjectValidator(wx.Validator):
    """This validator is used to ensure that the user has entered something
    into the text object editor dialog's text field.
    """

    def __init__(self, stringlist):
        """Standard constructor."""
        wx.Validator.__init__(self)
        self.stringlist = stringlist

    def Clone(self):
        """Standard cloner.

        Note that every validator must implement the Clone() method.
        """
        return MatchTextObjectValidator(self.stringlist)

    def Validate(self, win):
        """Validate the contents of the given text control."""
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()

        if self.stringlist.__contains__(text):
            textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True
        else:
            debug("The name is not defined")
            wx.MessageBox("The name is not defined!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False

    def TransferToWindow(self):
        """Transfer data from validator to window.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.

    def TransferFromWindow(self):
        """Transfer data from window to validator.

        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.


class NoMatchTextObjectValidator(wx.Validator):
    """This validator is used to ensure that the user has entered something
    into the text object editor dialog's text field.
    """

    def __init__(self, stringlist):
        """Standard constructor."""
        wx.Validator.__init__(self)
        self.stringlist = stringlist

    def Clone(self):
        """Standard cloner.
        Note that every validator must implement the Clone() method.
        """
        return NoMatchTextObjectValidator(self.stringlist)

    def Validate(self, win):
        """Validate the contents of the given text control."""
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        if len(text) == 0:
            debug("A text object must contain some text")
            wx.MessageBox("A text object must contain some text!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif self.stringlist.__contains__(text):
            debug("Duplicates are not allowed")
            wx.MessageBox("Duplicates are not allowed!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """Transfer data from validator to window.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.

    def TransferFromWindow(self):
        """Transfer data from window to validator.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.


class NoMatchValidTextObjectValidator(wx.Validator):
    """This validator is used to ensure that the user has entered something
    into the text object editor dialog's text field. It should not match
    a name in stringlist and it should be a valid varaible name. I.e
    is should start with a letter and can only contains ordinary
    letters as in string.letters as well as numbers as in string.digits
    or _
    """

    def __init__(self, stringlist):
        """Standard constructor."""
        wx.Validator.__init__(self)
        self.stringlist = stringlist
        self.reserved_words = [
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
        ]

        self.allowed_chars = string.digits + string.ascii_letters + "_"

    def Clone(self):
        """Standard cloner.
        Note that every validator must implement the Clone() method.
        """
        return NoMatchValidTextObjectValidator(self.stringlist)

    def Validate(self, win):
        """Validate the contents of the given text control."""
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        # print text, len(text)
        # print sum([char in self.allowed_chars for char in text])
        if len(text) == 0:
            debug("A text object must contain some text")
            wx.MessageBox("A text object must contain some text!", "Bad Input")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif self.stringlist.__contains__(text):
            debug("Duplicates are not allowed")
            wx.MessageBox("Duplicates are not allowed!", "Bad Input")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif text in self.reserved_words:
            debug("Python keywords are not allowed")
            wx.MessageBox("Python keywords are not allowed!", "Bad Input")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif sum([char in self.allowed_chars for char in text]) != len(text) or text[0] in string.digits:
            debug("Not a valid name")
            wx.MessageBox(
                "Not a vaild name. Names can only contain letters"
                ", digits and underscores(_) and not start with a digit.",
                "Bad Input",
            )
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """Transfer data from validator to window.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.

    def TransferFromWindow(self):
        """Transfer data from window to validator.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.


class NoMatchTextCtrlValidator(wx.Validator):
    """This validator is used to ensure that the user has entered something
    into the text object editor dialog's text field and that it does not match any other string in another textctrl.
    """

    def __init__(self, textctrls):
        """Standard constructor."""
        wx.Validator.__init__(self)
        self.textctrls = textctrls[:]

    def set_nomatch_ctrls(self, textctrls):
        self.textctrls = textctrls[:]

    def Clone(self):
        """Standard cloner.
        Note that every validator must implement the Clone() method.
        """
        return NoMatchTextCtrlValidator(self.textctrls)

    def Validate(self, win):
        """Validate the contents of the given text control."""
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        stringlist = [ctrl.GetValue() for ctrl in self.textctrls]
        iprint(text, stringlist)
        if len(text) == 0:
            debug("A text object must contain some text")
            wx.MessageBox("A text object must contain some text!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif text in stringlist:
            debug("Duplicates are not allowed")
            wx.MessageBox("Duplicates are not allowed!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """Transfer data from validator to window.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.

    def TransferFromWindow(self):
        """Transfer data from window to validator.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True  # Prevent wxDialog from complaining.


class FloatObjectValidator(wx.Validator):
    """This validator is used to ensure that the user has entered something
    into the text object editor dialog's text field.
    """

    def __init__(self, eval_func=eval, alt_types=None):
        """Standard constructor."""
        wx.Validator.__init__(self)
        self.value = None
        self.eval_func = eval_func
        self.alt_types = alt_types or []

    def Clone(self):
        """Standard cloner.

        Note that every validator must implement the Clone() method.
        """
        return FloatObjectValidator(self.eval_func, self.alt_types)

    def Validate(self, win):
        """Validate the contents of the given text control."""
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        self.value = None
        try:
            val = self.eval_func(text)
            if is_reflfunction(val):
                val = val.validate()
        except Exception as S:
            info("Can't evaluate the expression", exc_info=True)
            wx.MessageBox("Can't evaluate the expression!!\nERROR:\n%s" % S.__str__(), "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            try:
                self.value = float(val)
            except Exception as S:
                # print type(val), self.alt_types
                if not any([isinstance(val, typ) for typ in self.alt_types]):
                    info("Wrong type of parameter", exc_info=True)
                    wx.MessageBox("\nERROR:\n%s" % S.__str__(), "Error")
                    textCtrl.SetBackgroundColour("pink")
                    textCtrl.SetFocus()
                    textCtrl.Refresh()
                    return False

            textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """Transfer data from validator to window.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True

    def TransferFromWindow(self):
        """Transfer data from window to validator.

        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True


class ComplexObjectValidator(wx.Validator):
    """This validator is used to ensure that the user has entered something
    into the text object editor dialog's text field.
    """

    def __init__(self, eval_func=eval, alt_types=None):
        """Standard constructor."""
        wx.Validator.__init__(self)
        self.value = None
        self.eval_func = eval_func
        self.alt_types = alt_types or []

    def Clone(self):
        """Standard cloner.
        Note that every validator must implement the Clone() method.
        """
        return ComplexObjectValidator(self.eval_func, self.alt_types)

    def Validate(self, win):
        """Validate the contents of the given text control."""
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        self.value = None
        try:
            val = self.eval_func(text)
        except Exception as S:
            info("Can't evaluate the expression", exc_info=True)
            wx.MessageBox("Can't compile the complex expression!!\nERROR:\n%s" % S.__str__(), "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        try:
            # Have to do it differentily to work with proxys
            # self.value=complex(self.eval_func(text))
            if is_reflfunction(val):
                val = val.validate()
            self.value = complex(val.real, val.imag)
        except AttributeError:
            try:
                self.value = complex(val)
            except Exception as S:
                if not any([isinstance(val, typ) for typ in self.alt_types]):
                    info("Can't evaluate the complex expression, wrong type", exc_info=True)
                    wx.MessageBox(
                        "Can't evaluate the complex expression, not the correct type!!\nERROR:\n%s" % S.__str__(),
                        "Error",
                    )
                    textCtrl.SetBackgroundColour("pink")
                    textCtrl.SetFocus()
                    textCtrl.Refresh()
                    return False
                else:
                    textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
                    textCtrl.Refresh()
                    return True
        except Exception as S:
            if not any([isinstance(val, typ) for typ in self.alt_types]):
                info("Can't evaluate the complex expression", exc_info=True)
                wx.MessageBox("Can't evaluate the complex expression!!\nERROR:\n%s" % S.__str__(), "Error")
                textCtrl.SetBackgroundColour("pink")
                textCtrl.SetFocus()
                textCtrl.Refresh()
                return False
            else:
                textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
                textCtrl.Refresh()
                return True
        else:
            # print 'OK'
            textCtrl.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """Transfer data from validator to window.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True

    def TransferFromWindow(self):
        """Transfer data from window to validator.
        The default implementation returns False, indicating that an error
        occurred.  We simply return True, as we don't do any data transfer.
        """
        return True


# ----------------------------------------------------------------------


class ValidateBaseDialog(wx.Dialog):
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
        """A dialog that validates the input when OK are pressed. The validation is done
        through the validators given as input.

        Pars should contain a list of names.
        Vals should be a dictionary of names and values for the different parameters one wish to set.
        editable_pars should be a dictonary indicating if the parameter should be editable, a missing value
        is interpreted that the parameter is editable.
        Groups should either be false or consist of a list of lists which
        in turn consist of the (name, value) tuples where the first item
        in the list should be a string describing the group. This will be layed out
        with subboxes. An example:
        groups = [['Standard', [('f', 25), ('sigma', 7)]], ['Neutron', [('b', 3.0)]]]


        Note validators, values, units and editable pars should be dictionaries of values!
        """
        wx.Dialog.__init__(self, parent, -1, title)
        self.pars = pars
        self.editable_pars = editable_pars or {}
        self.validators = validators
        self.cols = cols
        self.vals = vals
        self.units = units or {}
        self.groups = groups or []
        self.tc = {}
        self.not_editable_bkg_color = wx.GREEN
        self.SetAutoLayout(True)

        if self.groups:
            self.main_sizer, self.tc = self.grid_layout(self, self.vals, self.editable_pars)
        else:
            self.main_sizer, self.tc = self.layout_group(self, self.pars, self.vals, self.editable_pars)

        self.main_layout()
        self.Layout()

    def main_layout(self):
        border = wx.BoxSizer(wx.VERTICAL)
        border.Add(self.main_sizer, 1, wx.GROW | wx.ALL, 5)

        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        border.Add(line, 0, wx.GROW | wx.RIGHT | wx.TOP, 5)

        buttons = self.create_buttons()
        border.Add(buttons, flag=wx.ALIGN_RIGHT | wx.ALL, border=5)
        self.SetSizer(border)
        border.Fit(self)
        self.border_sizer = border

    def create_buttons(self):
        buttons = wx.StdDialogButtonSizer()  # wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self, wx.ID_OK, "OK")
        b.SetDefault()
        buttons.AddButton(b)
        buttons.AddButton(wx.Button(self, wx.ID_CANCEL, "Cancel"))
        buttons.Realize()
        return buttons

    def grid_layout(self, parent, vals, editable_pars):
        """Do an more advanced layout with subboxes"""
        rows = math.ceil(len(self.pars) / (self.cols * 1.0))
        sizer = wx.FlexGridSizer(rows=rows, cols=self.cols, vgap=10, hgap=10)
        tc = {}
        for group in self.groups:
            if type(group[0]) != str:
                raise TypeError("First item in a group has to be a string")
            # Make the box for putting in the columns
            col_box = wx.StaticBox(parent, -1, group[0])
            col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL)
            group, group_tc = self.layout_group(parent, group[1], vals, editable_pars)
            col_box_sizer.Add(group, flag=wx.EXPAND, border=5)
            for item in group_tc:
                tc[item] = group_tc[item]
            sizer.Add(col_box_sizer, flag=wx.EXPAND)
        return sizer, tc

    def layout_group(self, parent, pars, vals, editable_pars):
        if self.units:
            layout_cols = 3
        else:
            layout_cols = 2
        sizer = wx.FlexGridSizer(len(pars) + 1, layout_cols, vgap=10, hgap=5)
        tc = {}
        for par in pars:
            # print par, vals[par]
            label = wx.StaticText(parent, -1, par + ": ")
            validator = self.validators[par]  # .Clone()
            val = vals[par]
            editable = True
            if par in editable_pars:
                editable = editable_pars[par]

            tc[par] = self.crete_edit_ctrl(editable, par, parent, val, validator)

            sizer.Add(label, flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
            sizer.Add(tc[par], flag=wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, border=5)

            # If we include units as well:
            if self.units:
                unit_label = wx.StaticText(parent, -1, " " + self.units[par])
                sizer.Add(unit_label, flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=5)
        return sizer, tc

    def GetValues(self):
        # print dir(self.tc[0])
        # print self.tc[0].GetValue()
        p = {}
        for par in list(self.validators.keys()):
            if type(self.validators[par]) == type([]):
                # have to pad teh text to make it a string inside a string...
                # text = '\''
                text = self.validators[par][self.tc[par].GetSelection()]
                # text += '\''
                p[par] = text
            else:
                p[par] = self.tc[par].GetValue()
        return p


class NormalEditMixin:
    """
    Normal mixin for the Validate Dialog
    """

    not_editable_bkg_color = None

    def crete_edit_ctrl(self, editable, par, parent, val, validator):
        """
        Creates the edit control for a parameter
        :param editable: flag to indicate if the parameter is editable
        :param par: parameter name
        :param parent: parent of the control
        :param val: value of the parameter
        :param validator: validator of the control
        :return: a wx control object (subclass of wx.Control)?
        """
        if type(validator) == type([]):
            # There should be a list of choices
            validator = validator[:]
            ctrl = wx.Choice(parent, -1, choices=validator)
            # Since we work with strings we have to find the right
            # strings positons to initilize the choice box.
            pos = 0
            if type(val) == type(""):
                pos = validator.index(val)
            elif type(par) == type(1):
                pos = par
            ctrl.SetSelection(pos)
        elif isinstance(validator, bool):
            # Parameter is a boolean
            ctrl = wx.CheckBox(self, -1)
            ctrl.SetValue(val)
            # Create a non-editable box if needed
            # ctrl.SetEditable(editable)
        elif isinstance(validator, int):
            # Parameter is an integer
            ctrl = IntCtrl(self, -1, val)
            # Create a non-editable box if needed
            ctrl.SetEditable(editable)
            if not editable:
                ctrl.SetBackgroundColour(self.not_editable_bkg_color)

        # Otherwise it should be a validator ...
        else:
            validator = validator.Clone()
            ctrl = wx.TextCtrl(parent, -1, str(val), validator=validator, style=wx.TE_RIGHT | wx.TE_RICH)

            # Create a non-editable box if needed
            ctrl.SetEditable(editable)
            if not editable:
                ctrl.SetBackgroundColour(self.not_editable_bkg_color)
        return ctrl


class ValidateBaseNotebookDialog(ValidateBaseDialog):
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
        fixed_pages=None,
        editable_pars=None,
    ):
        """A dialog that validates the input when OK are pressed. The validation is done
        through the validators given as input.

        Pars should contain a list of names.
        Vals should be a dictionary of dictionaries of names and values for the different parameters one wish to set.
        editable_pars should be a dictionary of the same form as vals.
        The upper dictionary level should contain the object names.
        Groups should either be false or a consist of a list of lists which
        in turn consist of the (name, value) tuples where the first item
        in the list should be a string describing the group. This will be layed out
        with subboxes. An example:
        groups = [['Standard', ('f', 25), ('sigma', 7)], ['Neutron', ('b', 3.0)]]

        Note validators, values and units should be dictonaries of values!
        """
        wx.Dialog.__init__(self, parent, -1, title)
        self.SetExtraStyle(wx.WS_EX_VALIDATE_RECURSIVELY)
        self.pars = pars
        self.validators = validators
        self.cols = cols
        self.vals = vals
        self.units = units or {}
        self.groups = groups or []
        self.fixed_pages = fixed_pages or []
        self.changes = []
        self.text_controls = {}
        self.SetAutoLayout(True)
        self.editable_pars = editable_pars or {}
        self.not_editable_bkg_color = wx.GREEN

        self.layout_notebook()
        self.do_toolbar()

        self.main_layout()
        self.Layout()

    def main_layout(self):
        border = wx.BoxSizer(wx.VERTICAL)
        border.Add((-1, 5))
        border.Add(self.toolbar)
        # border.Add((-1,2))

        border.Add(self.main_sizer, 1, wx.GROW | wx.ALL, 5)

        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        border.Add(line, 0, wx.GROW | wx.RIGHT | wx.TOP, 5)

        buttons = self.create_buttons()
        border.Add(buttons, flag=wx.ALIGN_RIGHT | wx.ALL, border=5)
        self.SetSizer(border)
        border.Fit(self)
        self.border_sizer = border

    def do_toolbar(self):
        if os.name == "nt":
            size = (24, 24)
        else:
            size = (-1, -1)
        dpi_scale_factor = wx.GetApp().dpi_scale_factor
        tb_bmp_size = int(dpi_scale_factor * 20)

        space = (5, -1)
        button_names = ["Insert", "Delete", "Rename"]
        button_images = [
            wx.Bitmap(images.insert_layer.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.Bitmap(images.delete.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.Bitmap(images.change_name.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
        ]
        callbacks = [self.eh_insert, self.eh_delete, self.eh_rename]
        tooltips = ["Insert", "Delete", "Rename"]

        boxbuttons = wx.BoxSizer(wx.HORIZONTAL)
        boxbuttons.Add((5, -1))
        for i in range(len(button_names)):
            # button = wx.Button(self,-1, button_names[i])
            button = wx.BitmapButton(self, -1, button_images[i], style=wx.NO_BORDER, size=size)
            boxbuttons.Add(button, 1, wx.EXPAND, 5)
            boxbuttons.Add(space)
            button.SetToolTip(tooltips[i])
            self.Bind(wx.EVT_BUTTON, callbacks[i], button)
        self.toolbar = boxbuttons

    def layout_notebook(self):
        """Do the notebook layout in the dialog"""
        self.main_sizer = wx.Notebook(self, -1, style=wx.NB_TOP)
        names = list(self.vals.keys())
        names.sort()
        for item in names:
            editable_pars = {}
            if item in self.editable_pars:
                editable_pars = self.editable_pars[item]
            self.AddPage(item, self.vals[item], editable_pars)

    def eh_insert(self, evt):
        """Eventhandler that creates a new panel"""
        pos = self.main_sizer.GetSelection()
        current_name = self.main_sizer.GetPageText(pos)
        if len(self.fixed_pages) > 0:
            current_name = self.fixed_pages[0]
        index = 1
        while "%s_%d" % (current_name, index) in list(self.vals.keys()):
            index += 1
        new_name = "%s_%d" % (current_name, index)

        state = {}
        for key in self.vals[current_name]:
            state[key] = 0

        self.AddPage(new_name, self.vals[current_name], state, select=True)
        self.changes.append(("", new_name))

    def eh_delete(self, evt):
        """Eventhandler that deletes the current page and its data"""
        pos = self.main_sizer.GetSelection()
        current_name = self.main_sizer.GetPageText(pos)
        self.RemovePage(current_name)
        self.changes.append((current_name, ""))

    def eh_rename(self, evt):
        """Eventhandler that renames the current tab"""
        pos = self.main_sizer.GetSelection()
        current_name = self.main_sizer.GetPageText(pos)
        if current_name in self.fixed_pages:
            msg = f"It is forbidden to change the name of {current_name}"
            debug(msg)
            wx.MessageBox(msg)
        else:
            unallowed_names = []
            for name in self.vals:
                if name != current_name:
                    unallowed_names.append(name)
            dlg = ValidateDialog(
                self,
                ["Name"],
                {"Name": current_name},
                {"Name": NoMatchValidTextObjectValidator(unallowed_names)},
                title="Give New Name",
            )

            if dlg.ShowModal() == wx.ID_OK:
                vals = dlg.GetValues()
                self.vals[vals["Name"]] = self.vals.pop(current_name)
                self.text_controls[vals["Name"]] = self.text_controls.pop(current_name)
                self.main_sizer.SetPageText(pos, vals["Name"])
                self.changes.append((current_name, vals["Name"]))
            dlg.Destroy()

    def AddPage(self, name, vals, editable_pars, select=False):
        """Adds a page to the notebook

        the flag select activates a page change to the current page
        """
        panel = wx.Panel(self.main_sizer, -1)
        self.main_sizer.AddPage(panel, name, select=select)
        if self.groups:
            sizer, page_text_controls = self.grid_layout(panel, vals, editable_pars)
        else:
            sizer, page_text_controls = self.layout_group(panel, self.pars, vals, editable_pars)
        self.text_controls[name] = page_text_controls
        panel.SetSizerAndFit(sizer)
        panel.Layout()
        self.Fit()
        # Make sure that the data is added if we add a new page after init has
        # been executed
        if not name in self.vals:
            self.vals[name] = vals.copy()
            self.editable_pars[name] = editable_pars.copy()

    def RemovePage(self, name):
        """Remove a page from the notebook"""
        sucess = False
        index = -1
        if name in self.fixed_pages:
            return sucess
        for i in range(self.main_sizer.GetPageCount()):
            if self.main_sizer.GetPageText(i) == name:
                sucess = True
                index = i
        if sucess:
            self.main_sizer.DeletePage(index)
            self.vals.pop(name)
        return sucess

    def GetValues(self):
        """Extracts values from the text controls and returns is as a
        two level nested dictionary.
        """
        p = {}
        for page in self.vals:
            p[page] = {}
            for par in list(self.validators.keys()):
                if type(self.validators[par]) == type([]):
                    # have to pad teh text to make it a string inside a string...
                    # text = '\''
                    text = self.validators[par][self.text_controls[page][par].GetSelection()]
                    # text += '\''
                    p[page][par] = text
                else:
                    p[page][par] = self.text_controls[page][par].GetValue()
        return p

    def GetChanges(self):
        """GetChanges(self) --> changes
        Returns the changes as a list consisting of tuples (old_name, new_name).
        if the page has been deleted new_name is an empty string.
        """
        return self.changes


class FitEditMixIn:
    """
    Mixin for the Validate Dialogs to enable the definition of fitable parameters.
    """

    validators = None
    tc = None

    def __init__(self):
        pass

    def crete_edit_ctrl(self, state, par, parent, val, validator):
        """
        Creates the edit control for a parameter
        :param state: flag to indicate if the parameter is editable integer that takes the values
            0 - editable
            1 - fitted in the grid
            2 - constant but defined in the grid
        :param par: parameter name
        :param parent: parent of the control
        :param val: value of the parameter
        :param validator: validator of the control
        :return: a wx control object (subclass of wx.Control)?
        """
        if type(validator) in [list, tuple]:
            # There should be a list of choices
            validator = validator[:]
            ctrl = wx.Choice(parent, -1, choices=validator)
            # Since we work with strings we have to find the right
            # strings positons to initilize the choice box.
            pos = 0
            if type(val) == type(""):
                try:
                    pos = validator.index(val)
                except ValueError:
                    pos = 0
            elif type(par) == type(1):
                pos = par
            ctrl.SetSelection(pos)
        # Otherwise it should be a validator ...
        else:
            validator = validator.Clone()
            ctrl = FitSelectorCombo(state, parent, -1, str(val), validator=validator, style=wx.TE_RICH | wx.TE_RIGHT)
        return ctrl

    def GetStates(self):
        """Returns the states of the parameters in the dialog

        :return: a dictionary with keys representing parameters and values representing state.
        """
        # print dir(self.tc[0])
        # print self.tc[0].GetValue()
        p = {}
        for par in list(self.validators.keys()):
            if isinstance(self.validators[par], list):
                p[par] = 0
            else:
                p[par] = self.tc[par].state
        return p


class ValidateDialog(ValidateBaseDialog, NormalEditMixin):
    """
    The normal Validate Dialog definition
    """


class ValidateFitDialog(ValidateBaseDialog, FitEditMixIn):
    """
    A sub-class of ValidateDialog that uses a custom ComboCtrl to allow a parameter to be
    set for fitting.
    """


class ValidateNotebookDialog(ValidateBaseNotebookDialog, NormalEditMixin):
    """
    The normal Validate Notebook Dialog.
    """


class ValidateFitNotebookDialog(ValidateBaseNotebookDialog, FitEditMixIn):
    """
    A sub-class of ValidateNotebookDialog that uses a custom ComboCtrl to allow a parameter to be
    set for fitting.
    """

    def GetStates(self):
        """Extracts values from the text controls and returns is as a
        two level nested dictionary.
        """
        p = {}
        for page in self.vals:
            p[page] = {}
            for par in list(self.validators.keys()):
                if isinstance(self.validators[par], list):
                    p[page][par] = 0
                else:
                    p[page][par] = self.text_controls[page][par].state
        return p


class FitSelectorCombo(wx.ComboCtrl):
    """Class that defines a Combobox that allows the user to set if the parameter should be handled by the grid
    , fitted or constants from the beginning.
    """

    def update_text_state(self):
        self.SetEditable(not self.state)
        # self.GetTextCtrl().Enable(not self.state)
        font = self.GetFont()

        if self.state == 0:
            self.SetForegroundColour(wx.BLACK)
            font.SetWeight(wx.FONTWEIGHT_NORMAL)
        elif self.state == 1:
            self.SetForegroundColour(self.fit_bkg_color)
            font.SetWeight(wx.FONTWEIGHT_BOLD)
        elif self.state == 2:
            self.SetForegroundColour(self.const_fit_bkg_color)
            font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.SetFont(font)

        self.Refresh()

    def create_button_bitmap(self):
        # make a custom bitmap showing "F"
        bw, bh = 14, 16
        bmp = wx.Bitmap(bw, bh, depth=wx.BITMAP_SCREEN_DEPTH)
        dc = wx.MemoryDC(bmp)
        # clear to a specific background colour
        bgcolor = wx.Colour(255, 254, 255)
        dc.SetBackground(wx.Brush(bgcolor))
        dc.Clear()
        # draw the label onto the bitmap
        label = "F"
        font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        dc.SetFont(font)
        tw, th = dc.GetTextExtent(label)
        dc.DrawText(label, (bw - tw) // 2, (bh - th) // 2)
        del dc
        # now apply a mask using the bgcolor
        bmp.SetMaskColour(bgcolor)
        # and tell the ComboCtrl to use it
        self.SetButtonBitmaps(bmp, True)

    def __init__(self, state, *args, **kw):
        """

        :param state: an int signalling the state 0 - parameter defined in dialog,
            1 - parameter fitted, 2 - parameter constant in grid.
        :param args:
        :param kw:
        :return:
        """
        wx.ComboCtrl.__init__(self, *args, **kw)

        self.state = state
        # Green wx.Colour(138, 226, 52), ORANGE wx.Colour(245, 121, 0)
        self.fit_bkg_color = wx.Colour(245, 121, 0)
        # Tango Sky blue wx.Colour(52, 101, 164), wx.Colour(114, 159, 207)
        self.const_fit_bkg_color = wx.Colour(114, 159, 207)

        self.create_button_bitmap()
        self.update_text_state()

        # Set the id's for the Menu
        self.id_define_par = wx.NewId()
        self.id_fit_par = wx.NewId()
        self.id_const_fit_par = wx.NewId()

        menu = wx.Menu()
        menu.AppendRadioItem(
            self.id_define_par, "Define parameter here", "Unlock the parameter for editing and remove it from the grid."
        )
        menu.AppendRadioItem(self.id_fit_par, "Fit parameter", "Define the parameter in the grid and use fit it.")
        menu.AppendRadioItem(self.id_const_fit_par, "Constant fit parameter", "Define the parameter in the fri")
        if not self.state:
            menu.Check(self.id_define_par, True)
        elif self.state == 1:
            menu.Check(self.id_fit_par, True)
        elif self.state == 2:
            menu.Check(self.id_const_fit_par, True)

        self.menu = menu

    # Overridden from ComboCtrl, called when the combo button is clicked
    def OnButtonClick(self):
        # Create a popup menu when pressing the button.
        self.PopupMenu(self.menu)
        if self.menu.IsChecked(self.id_define_par):
            self.state = 0
        elif self.menu.IsChecked(self.id_fit_par):
            self.state = 1
        elif self.menu.IsChecked(self.id_const_fit_par):
            self.state = 2
        self.update_text_state()

    # Overridden from ComboCtrl to avoid assert since there is no ComboPopup
    def DoSetPopupControl(self, popup):
        pass


class ParameterExpressionCombo(wx.ComboCtrl):
    """Class that defines a Combobox for editing an expression and inserting parameters from the model in that
    expression by choosing them from a popupmenu.
    """

    def create_button_bitmap(self):
        # make a custom bitmap showing "F"
        bw, bh = 14, 16
        bmp = wx.Bitmap(bw, bh, depth=wx.BITMAP_SCREEN_DEPTH)
        dc = wx.MemoryDC(bmp)
        # clear to a specific background colour
        bgcolor = wx.Colour(255, 254, 255)
        dc.SetBackground(wx.Brush(bgcolor))
        dc.Clear()
        # draw the label onto the bitmap
        label = "P"
        font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        dc.SetFont(font)
        tw, th = dc.GetTextExtent(label)
        dc.DrawText(label, (bw - tw) // 2, (bh - th) // 2)
        del dc
        # now apply a mask using the bgcolor
        bmp.SetMaskColour(bgcolor)
        # and tell the ComboCtrl to use it
        self.SetButtonBitmaps(bmp, True)

    def __init__(self, par_dict, sim_func, *args, **kw):
        """

        :param state: an int signalling the state 0 - parameter defined in dialog,
            1 - parameter fitted, 2 - parameter constant in grid.
        :param args:
        :param kw:
        :return:
        """
        wx.ComboCtrl.__init__(self, *args, **kw)

        self.create_button_bitmap()

        # Set the id's for the Menu
        self.id_simulate = wx.NewId()

        self.par_dict = par_dict
        self.sim_func = sim_func

    # Overridden from ComboCtrl, called when the combo button is clicked
    def OnButtonClick(self):
        menu = wx.Menu()
        par_dict = self.par_dict
        classes = list(par_dict.keys())
        classes.sort(key=str.lower)
        for cl in classes:
            # Create a submenu for each class
            clmenu = wx.Menu()
            obj_dict = par_dict[cl]
            objs = list(obj_dict.keys())
            objs.sort(key=str.lower)
            # Create a submenu for each object
            for obj in objs:
                obj_menu = wx.Menu()
                funcs = obj_dict[obj]
                funcs.sort(key=str.lower)
                # Create an item for each method
                for func in funcs:
                    item = obj_menu.Append(wx.NewId(), obj + "." + func.replace("set", "get"))
                    self.Bind(wx.EVT_MENU, self.OnPopUpItemSelected, item)
                clmenu.Append(-1, obj, obj_menu)
            menu.Append(-1, cl, clmenu)
        # Check if there are no available classes
        # TODO: Test implementation if necessary?
        # if len(classes) == 0:
        #    # Add an item to compile the model
        #    item = menu.Append(self.id_simulate, 'Simulate to see parameters')
        #    self.Bind(wx.EVT_MENU, self.OnSimulate, item)
        # Create a popup menu when pressing the button.
        self.menu = menu
        self.PopupMenu(self.menu)
        menu.Destroy()

    def OnSimulate(self, event):
        """Event handler for selection of the simulate option in the popup.
        :param event:
        :return:
        """
        if not self.sim_func is None:
            self.sim_func()

    def OnPopUpItemSelected(self, event):
        """Event handler for selcting a parameters get function.

        :param event:
        :return:
        """
        item = self.menu.FindItemById(event.GetId())
        self.WriteText(item.GetItemLabel() + "()")

    # Overridden from ComboCtrl to avoid assert since there is no ComboPopup
    def DoSetPopupControl(self, popup):
        pass


class ZoomFrame(wx.MiniFrame):
    def __init__(self, parent):
        wx.MiniFrame.__init__(self, parent, -1, "X-Y Scales")

        # self.SetAutoLayout(True)

        VSPACE = 10

        self.panel = wx.Panel(self, -1, style=wx.TAB_TRAVERSAL | wx.CLIP_CHILDREN | wx.FULL_REPAINT_ON_RESIZE)

        gbs = wx.GridBagSizer(3, 3)
        label = wx.StaticText(self.panel, -1, "Min")
        gbs.Add(label, (0, 1), flag=wx.ALIGN_CENTER, border=2)
        label = wx.StaticText(self.panel, -1, "Max")
        gbs.Add(label, (0, 2), flag=wx.ALIGN_CENTER, border=2)
        label = wx.StaticText(self.panel, -1, " X" + ": ")
        gbs.Add(label, (1, 0), flag=wx.ALIGN_RIGHT, border=2)
        self.xmin = wx.TextCtrl(self.panel, -1, "0", validator=FloatObjectValidator())
        gbs.Add(self.xmin, (1, 1), flag=wx.ALIGN_CENTER | wx.EXPAND, border=2)
        self.xmax = wx.TextCtrl(self.panel, -1, "0", validator=FloatObjectValidator())
        gbs.Add(self.xmax, (1, 2), flag=wx.ALIGN_CENTER | wx.EXPAND, border=2)
        label = wx.StaticText(self.panel, -1, " Y" + ": ")
        gbs.Add(label, (2, 0), flag=wx.ALIGN_RIGHT, border=2)
        self.ymin = wx.TextCtrl(self.panel, -1, "0", validator=FloatObjectValidator())
        gbs.Add(self.ymin, (2, 1), flag=wx.ALIGN_CENTER | wx.EXPAND, border=2)
        self.ymax = wx.TextCtrl(self.panel, -1, "0", validator=FloatObjectValidator())
        gbs.Add(self.ymax, (2, 2), flag=wx.ALIGN_CENTER | wx.EXPAND, border=2)

        buttons = wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self.panel, wx.ID_OK, "Apply")
        # b.SetDefault()
        # buttons.AddButton(wx.Button(self, wx.ID_CANCEL, "Cancel"))
        # buttons.Realize()
        buttons.Add((0, 0), 2, wx.EXPAND)
        buttons.Add(b, 1, flag=wx.ALIGN_RIGHT)
        border = wx.BoxSizer(wx.VERTICAL)

        border.Add(gbs, 0, wx.GROW | wx.ALL, 2)
        border.Add(buttons)

        self.panel.SetSizerAndFit(border)
        self.SetClientSize(self.panel.GetSize())
        # self.Layout()
