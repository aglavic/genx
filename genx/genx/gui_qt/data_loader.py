"""data_loader.py

Qt port of the GUI side of data_loader.
"""

import io
import os
import traceback

from PySide6 import QtWidgets

from genx.core.config import config
from genx.core.custom_logging import iprint

from genx.plugins.utils import PluginHandler, __MODULE_DIR__ as plugin_dir
from .message_dialogs import ShowErrorDialog

head, tail = os.path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled...
__FILENAME__ = tail.split(".")[0]
# This assumes that plugin is under the current dir may need
# changing
__MODULE_DIR__ = head
if __MODULE_DIR__ != os.path.sep:
    __MODULE_DIR__ += os.path.sep


class PluginController:
    def __init__(self, parent):
        self.plugin_handler = PluginHandler(parent, plugin_dir, "data_loaders")
        self.parent = parent
        self.plugin_handler.load_plugin("auto")

    def _set_status(self, text: str) -> None:
        if hasattr(self.parent, "statusBar"):
            sb = self.parent.statusBar()
            if sb is not None:
                sb.showMessage(text, 5000)

    def load_default(self):
        try:
            plugin_name = config.get("data handling", "data loader")
            self.LoadPlugin(plugin_name)
        except Exception as exc:
            iprint("Could not locate the data loader parameter or the data loader. Error:")
            iprint(str(exc))
            iprint("Proceeding with loading the default data loader.")
            self.LoadPlugin("default")

    def LoadPlugin(self, plugin):
        """
        Loads a data handler note that there is no UnLoad function
        since only one DataHandler can be plugged in at a time.
        """
        names = self.plugin_handler.loaded_plugins.copy()
        try:
            [self.plugin_handler.unload_plugin(pl) for pl in names]
            if names:
                self._set_status(f"Unloaded data loader {list(names.keys())[0]}")
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
            ShowErrorDialog(
                self.parent,
                "Can NOT unload plugin object" + list(names.keys())[0] + "\nPython traceback below:\n\n" + tbtext,
            )
        try:
            self.plugin_handler.load_plugin(plugin)
            self._set_status(f"Loaded data loader: {plugin}")
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
            ShowErrorDialog(self.parent, "Can NOT load plugin " + plugin + "\nPython traceback below:\n\n" + tbtext)

    def ShowDialog(self):
        """ShowDialog(self) --> None

        Shows a dialog box for the user to choose a data loader.
        """
        cur_plugin = list(self.plugin_handler.loaded_plugins.keys())[0]
        plugin_list = self.plugin_handler.get_possible_plugins()
        dlg = PluginDialog(self.parent, plugin_list, cur_plugin, self.LoadPlugin)
        dlg.exec()


class PluginDialog(QtWidgets.QDialog):
    def __init__(self, parent, plugin_list, current_plugin, load_plugin_func=None):
        super().__init__(parent)
        self.setWindowTitle("Choose a data loader")

        self.load_plugin_func = load_plugin_func

        choice_label = QtWidgets.QLabel("Data loaders:", self)
        self.choice_control = QtWidgets.QComboBox(self)
        self.choice_control.addItems(plugin_list)
        if current_plugin in plugin_list:
            self.choice_control.setCurrentText(current_plugin)

        choice_row = QtWidgets.QHBoxLayout()
        choice_row.addWidget(choice_label)
        choice_row.addWidget(self.choice_control)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.on_ok)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self.on_apply)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addSpacing(20)
        layout.addLayout(choice_row)
        layout.addSpacing(20)
        line = QtWidgets.QFrame(self)
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line)
        layout.addWidget(buttons)

    def on_apply(self):
        if self.load_plugin_func is not None:
            self.load_plugin_func(self.choice_control.currentText())

    def on_ok(self):
        self.on_apply()
        self.accept()
