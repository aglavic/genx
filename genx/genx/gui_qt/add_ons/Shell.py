"""<h1>Shell</h1>
A plugin which opens a new tab with a simple shell inside.
Qt port with a basic interactive console.
"""

import code
import io
import traceback

from contextlib import redirect_stderr, redirect_stdout
from PySide6 import QtCore, QtWidgets

from genx.plugins import add_on_framework as framework


class SimpleShell(QtWidgets.QWidget):
    def __init__(self, parent, locals_dict):
        super().__init__(parent)
        self.interpreter = code.InteractiveInterpreter(locals=locals_dict)

        layout = QtWidgets.QVBoxLayout(self)
        self.output = QtWidgets.QPlainTextEdit(self)
        self.output.setReadOnly(True)
        self.input = QtWidgets.QLineEdit(self)
        self.input.returnPressed.connect(self._on_enter)

        layout.addWidget(self.output, 1)
        layout.addWidget(self.input, 0)

        self._append("GenX shell ready. Enter Python expressions.")

    def _append(self, text):
        self.output.appendPlainText(text)

    def _on_enter(self):
        command = self.input.text()
        self.input.clear()
        if not command.strip():
            return
        self._append(f">>> {command}")
        out = io.StringIO()
        err = io.StringIO()
        try:
            with redirect_stdout(out), redirect_stderr(err):
                more = self.interpreter.runsource(command, "<genx-shell>", "single")
        except Exception:
            self._append(traceback.format_exc())
            return
        std = out.getvalue().strip()
        serr = err.getvalue().strip()
        if std:
            self._append(std)
        if serr:
            self._append(serr)
        if more:
            self._append("... (multiline input not supported)")


class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        inputpanel = self.NewInputFolder("Shell")
        layout = QtWidgets.QHBoxLayout(inputpanel)
        self.shell = SimpleShell(
            inputpanel,
            locals_dict={
                "frame": parent,
                "model": self.GetModel(),
                "data": self.GetModel().get_data(),
                "ctrl": parent.model_control.controller,
            },
        )
        layout.addWidget(self.shell, 1)
        self.StatusMessage("Shell plugin loaded")
