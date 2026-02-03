"""
Exception handling utilities for the Qt GUI.
Mirrors the concepts from the wx GUI version:
- CatchModelError context manager for recoverable UI operations
- GuiExceptionHandler logging.Handler for unhandled exceptions routed through logging
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from logging import CRITICAL, Handler, LogRecord, debug, error, warning
from typing import Callable, Optional

from PySide6 import QtCore, QtWidgets

from ..exceptions import GenxError


class CatchModelError:
    """
    A context manager that displays an error dialog if an exception is raised within the 'with' block.
    If the error is a GenX exception the display is slightly different.
    """

    def __init__(
            self,
            parent: Optional[QtWidgets.QWidget] = None,
            action: str = "execution",
            step: str | None = None,
            status_update: Optional[Callable[[str], None]] = None,
            ):
        self.parent = parent
        self.action = action
        self.step = step
        self._status_update = status_update
        self.successful = False

    def status_update(self, text: str) -> None:
        if self._status_update:
            self._status_update(text)

    def __enter__(self) -> "CatchModelError":
        debug(f"enter {self.action}/{self.step}", stacklevel=3)
        if self.step:
            self.status_update(f"Start {self.step}.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            debug(f"exit {self.action}/{self.step} w/o error", stacklevel=3)
            self.successful = True
            if self.step:
                self.status_update(f"Success in {self.step}.")
            return True

        self.successful = False
        message = f"{type(exc_val).__name__} in {self.action}"
        if self.step:
            message += f", could not {self.step}."
        ext_message = f"{exc_val}"

        if isinstance(exc_val, GenxError):
            warning(message, exc_info=(exc_type, exc_val, exc_tb), stack_info=True)
            title = "Warning"
            icon = QtWidgets.QMessageBox.Icon.Warning
            if self.step:
                self.status_update(f"Error in {self.step}, {type(exc_val).__name__}")
        else:
            error(message, exc_info=(exc_type, exc_val, exc_tb), stack_info=True)
            title = "Error"
            icon = QtWidgets.QMessageBox.Icon.Critical
            if self.step:
                self.status_update(f"Fatal error in {self.step}, {type(exc_val).__name__}")
            ext_message += "\n\nPython Error (last calls first):\n    "
            ext_message += "\n    ".join(traceback.format_tb(exc_tb)[:2:-1])

        full_trace = message + ":\n\n"
        full_trace += "".join(traceback.format_tb(exc_tb))
        full_trace += f"{type(exc_val).__name__}: {exc_val}"

        # Ensure dialogs are shown on the Qt main thread.
        QtCore.QTimer.singleShot(
            0, lambda: self.display_message(title, message, ext_message, full_trace, icon=icon)
            )
        return True  # suppress exception

    def display_message(
            self,
            title: str,
            message: str,
            ext_message: str,
            full_trace: str,
            icon: QtWidgets.QMessageBox.Icon = QtWidgets.QMessageBox.Icon.Critical,
            ) -> None:
        box = QtWidgets.QMessageBox(self.parent)
        box.setWindowTitle(title)
        box.setIcon(icon)
        box.setText(message)
        box.setInformativeText(ext_message)

        copy_btn = box.addButton("Copy to Clipboard", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        ok_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        box.setDefaultButton(ok_btn)

        box.exec()
        if box.clickedButton() == copy_btn:
            QtWidgets.QApplication.clipboard().setText(full_trace)


class GuiExceptionHandler(Handler):
    """
    A custom logging handler that opens a message dialog on critical (unhandled) exceptions.
    """

    show_dialog = True

    def __init__(self, app: QtWidgets.QApplication, level=CRITICAL):
        super().__init__(level=level)
        self.app = app

    def emit(self, record: LogRecord) -> None:
        if not self.show_dialog:
            return
        if not record.exc_info:
            return

        title = "GenX - Unhandled Python Error"
        exc_type, exc_val, _ = record.exc_info
        message = f"{exc_type.__name__}: {exc_val}"
        message = "GenX encountered an unexpected error.\n" + "\n".join(message.splitlines()[-5:])

        exc_text = record.exc_text or ""
        ext_message = "\n".join(exc_text.splitlines()[-30:])
        ext_message += "\n\nYou can suppress any future warnings by choosing 'Cancel' to close this window."
        full_trace = exc_text

        QtCore.QTimer.singleShot(
            0, lambda: self.display_message(title, message, ext_message, full_trace)
            )

    def display_message(
            self,
            title: str,
            message: str,
            ext_message: str,
            full_trace: str,
            icon: QtWidgets.QMessageBox.Icon = QtWidgets.QMessageBox.Icon.Critical,
            ) -> None:
        box = QtWidgets.QMessageBox()
        box.setWindowTitle(title)
        box.setIcon(icon)
        box.setText(message)
        box.setInformativeText(ext_message)

        copy_btn = box.addButton("Copy to Clipboard", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        ok_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Ok)
        cancel_btn = box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(ok_btn)

        box.exec()
        if box.clickedButton() == copy_btn:
            QtWidgets.QApplication.clipboard().setText(full_trace)
        elif box.clickedButton() == cancel_btn:
            self.show_dialog = False