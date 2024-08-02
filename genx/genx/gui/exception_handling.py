"""
Module to handle different cases of exceptions in the user interface.
"""

import traceback

from logging import CRITICAL, Formatter, Handler, LogRecord, debug, error, warning

import wx

from ..exceptions import GenxError


class CatchModelError:
    """
    A context manager that displays an error dialog if an exception is raised within the 'with' block.
    If the error is a GenX exception the display is slightly different.
    """

    def __init__(
        self,
        parent: wx.TopLevelWindow = None,
        action: str = "execution",
        step: str = None,
        status_update: callable = None,
    ):
        self.parent = parent
        self.action = action
        self.step = step
        self._status_update = status_update
        self.successful = False

    def status_update(self, text):
        if self._status_update:
            self._status_update(text)

    def __enter__(self):
        debug(f"enter {self.action}/{self.step}", stacklevel=3)
        if self.step:
            self.status_update(f"Start {self.step}.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            debug(f"exit {self.action}/{self.step} w/o error", stacklevel=3)
            self.successful = True
            self.status_update(f"Success in {self.step}.")
            return True

        # allow the code to do something with the error, normally not needed
        self.successful = False
        self.exc_type = exc_type
        self.exc_val = exc_val
        self.exc_tb = exc_tb

        message = f"{type(exc_val).__name__} in {self.action}"
        if self.step:
            message += f", could not {self.step}."
        ext_message = f"{exc_val}"

        if isinstance(exc_val, GenxError):
            warning(f"{message}", exc_info=(exc_type, exc_val, exc_tb), stack_info=True)
            self.status_update(f"Error in {self.step}, {type(exc_val).__name__}")
            title = "Warning"
            icon_style = wx.ICON_WARNING
        else:
            error(message, exc_info=(exc_type, exc_val, exc_tb), stack_info=True)
            self.status_update(f"Fatal error in {self.step}, {type(exc_val).__name__}")
            ext_message += "\n\nPython Error (last calls first):\n    "
            ext_message += "\n    ".join(traceback.format_tb(exc_tb)[:2:-1])
            title = "Warning"
            icon_style = wx.ICON_ERROR
        full_trace = message + ":\n\n"
        full_trace += "".join(traceback.format_tb(exc_tb))
        full_trace += f"{type(exc_val).__name__}: {exc_val}"

        # make sure the dialog is shown from main thread and after any queued actions
        wx.CallAfter(self.display_message, title, message, ext_message, full_trace, icon_style)
        return True  # exception is not raised in main context

    def display_message(self, title, message, ext_message, full_trace, icon_style=wx.ICON_ERROR):
        style = wx.OK | wx.HELP
        dlg = wx.MessageDialog(self.parent, message, title, style | icon_style)
        dlg.SetExtendedMessage(ext_message)
        dlg.SetHelpLabel("Copy to Clipboard")

        result = dlg.ShowModal()
        while result == wx.ID_HELP:
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(full_trace))
                wx.TheClipboard.Close()
            result = dlg.ShowModal()
        dlg.Destroy()


class GuiExceptionHandler(Handler):
    app: wx.App
    show_dialog = True

    def __init__(self, app, level=CRITICAL):
        Handler.__init__(self, level=level)
        self.app = app

    def emit(self, record: LogRecord):
        if not self.show_dialog:
            return
        title = "GenX - Unhandled Python Error"
        message = f"GenX encountered an unexpected error.\n{record.exc_info[0].__name__}: {record.exc_info[1]}"
        ext_message = f"{record.exc_text}"
        ext_message += "\n\nYou can suppress any future warnings by choosing 'Cancel' to close this window."
        full_trace = f"{record.exc_text}"

        self.display_message(title, message, ext_message, full_trace)

    def display_message(self, title, message, ext_message, full_trace, icon_style=wx.ICON_ERROR):
        style = wx.OK | wx.CANCEL | wx.HELP
        dlg = wx.MessageDialog(self.app.GetTopWindow(), message, title, style | icon_style)
        dlg.SetExtendedMessage(ext_message)
        dlg.SetHelpLabel("Copy to Clipboard")

        result = dlg.ShowModal()
        while result == wx.ID_HELP:
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(full_trace))
                wx.TheClipboard.Close()
            result = dlg.ShowModal()
        if result == wx.ID_CANCEL:
            self.show_dialog = False
        dlg.Destroy()
