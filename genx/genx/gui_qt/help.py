"""
Qt help dialogs for GenX.
"""

import os
import importlib

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from docutils.core import publish_doctree, publish_from_doctree
    from docutils.parsers.rst import roles
except ImportError:

    def rst_html(text: str) -> str:
        return "For proper display install docutils.<br>\n" + text.replace("\n", "<br>\n")

else:

    def _role_fn(name, rawtext, text, lineno, inliner, options=None, content=None):
        if options is None:
            options = {}
        if content is None:
            content = []
        return [], []

    roles.register_canonical_role("mod", _role_fn)

    def rst_html(text: str) -> str:
        return publish_from_doctree(publish_doctree(text), writer_name="html").decode()


class PluginHelpDialog(QtWidgets.QDialog):
    def __init__(self, parent, module: str, title: str = "Models help") -> None:
        super().__init__(parent)
        self.module = module
        self.sub_modules = True

        self.setWindowTitle(title)
        layout = QtWidgets.QVBoxLayout(self)
        choice_row = QtWidgets.QHBoxLayout()
        choice_row.addWidget(QtWidgets.QLabel("Module:"))

        mod_list = self.find_modules(module)
        self.choice = QtWidgets.QComboBox()
        self.choice.addItems(mod_list)
        self.choice.currentIndexChanged.connect(self.on_choice)
        choice_row.addWidget(self.choice, 1)
        layout.addLayout(choice_row)

        self.html_win = QtWidgets.QTextBrowser()
        self.html_win.setOpenExternalLinks(True)
        self.html_win.setStyleSheet("QTextBrowser { background: #e6ffe6; }")
        layout.addWidget(self.html_win, 1)

        if mod_list:
            self.choice.setCurrentIndex(0)
            self.on_choice()

        self.resize(600, max(400, parent.height() if parent is not None else 500))
        if parent is not None:
            pos = parent.pos()
            size = parent.size()
            dest = QtCore.QPoint(pos.x() + size.width(), pos.y())
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                if dest.x() + self.width() > available.right():
                    dest.setX(available.right() - self.width())
            self.move(dest)

    @QtCore.Slot()
    def on_choice(self) -> None:
        sub_module = self.choice.currentText()
        if self.sub_modules:
            doc = self.load_doc(self.module, sub_module)
        else:
            doc = self.load_doc(self.module)
        self.html_win.setHtml(doc)

    def find_modules(self, module: str) -> list[str]:
        mod = importlib.import_module(f"genx.{module}")
        try:
            modules = [
                s[:-3]
                for s in os.listdir(mod.__path__[0])
                if s[0] != "_" and s.endswith(".py")
            ]
        except AttributeError:
            modules = [module]
            self.sub_modules = False
        return modules

    def load_doc(self, module: str, sub_module: str | None = None) -> str:
        try:
            if sub_module is not None:
                mod = importlib.import_module(f"genx.{module}.{sub_module}")
            else:
                mod = importlib.import_module(f"genx.{module}")
        except Exception as exc:
            docs = f"Could not load docstring for {sub_module}."
            docs += f"\n The following exception occured: {exc}"
        else:
            if mod.__doc__ is None:
                docs = "No documentation available for module."
            elif "<h1>" in mod.__doc__:
                docs = mod.__doc__
            else:
                docs = rst_html(mod.__doc__)
        if not isinstance(docs, str):
            docs = f"The doc string is of the wrong type in module {sub_module}"
        return docs
