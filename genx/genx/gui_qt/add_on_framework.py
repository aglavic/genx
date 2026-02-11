"""
Qt port of add_on_framework for plugin handling.
"""

import io
import os
import traceback

from dataclasses import dataclass
from logging import debug, error, info, warning

from PySide6 import QtCore, QtGui, QtWidgets

from genx.core.config import BaseConfig, Configurable
from genx.plugins.utils import PluginHandler
from .utils import ShowErrorDialog, ShowInfoDialog, ShowQuestionDialog, ShowWarningDialog

head, tail = os.path.split(__file__)
__FILENAME__ = tail.split(".")[0]
__MODULE_DIR__ = head
if __MODULE_DIR__ != "/":
    __MODULE_DIR__ += "/"

class PluginPage(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(0, 0, 0, 0)

    def _reset_layout_margins(self) -> None:
        layout = self.layout()
        if layout is not None:
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

    def childEvent(self, event):
        super().childEvent(event)
        if event.type() == QtCore.QEvent.Type.ChildAdded:
            if isinstance(event.child(), QtWidgets.QLayout):
                self._reset_layout_margins()

    def event(self, event):
        if event.type() == QtCore.QEvent.Type.LayoutRequest:
            self._reset_layout_margins()
        return super().event(event)

class Template:
    """
    Qt template class for handling plugins.
    """

    def __init__(self, parent: "GenxMainWindow"):
        self.parent = parent
        self.plot_pages = []
        self.input_pages = []
        self.data_pages = []
        self.menus = []


    def _get_plugin_tab_widget(self) -> QtWidgets.QTabWidget | None:
        return self.parent.ui.pluginTabWidget

    def _get_plugin_empty_tab(self) -> QtWidgets.QWidget | None:
        return self.parent.ui.pluginTabEmpty

    def _plugin_tab_count(self, notebook: QtWidgets.QTabWidget | None) -> int:
        if notebook is None:
            return 0
        empty = self._get_plugin_empty_tab()
        return sum(1 for i in range(notebook.count()) if notebook.widget(i) is not empty)

    def _remove_plugin_empty_tab(self, notebook: QtWidgets.QTabWidget | None) -> None:
        if notebook is None:
            return
        empty = self._get_plugin_empty_tab()
        if empty is None:
            return
        idx = notebook.indexOf(empty)
        if idx >= 0:
            notebook.removeTab(idx)

    def _ensure_plugin_empty_tab(self, notebook: QtWidgets.QTabWidget | None) -> None:
        if notebook is None:
            return
        empty = self._get_plugin_empty_tab()
        if empty is None:
            return
        if notebook.indexOf(empty) < 0:
            notebook.addTab(empty, "Empty Tab")

    def InputPageChanged(self, pname):
        pass

    def NewPlotFolder(self, name, pos=-1):
        panel = PluginPage()
        notebook = self._get_plugin_tab_widget()
        if notebook is None:
            notebook = self.parent.ui.plotTabWidget
        else:
            empty = self._get_plugin_empty_tab()
            if empty is not None and notebook.count() == 1 and notebook.widget(0) is empty:
                self._remove_plugin_empty_tab(notebook)
        if notebook is not None:
            notebook.addTab(panel, name)
        self.plot_pages.append(panel)
        return panel

    def NewInputFolder(self, name, pos=-1):
        panel = PluginPage()
        notebook = self.parent.ui.inputTabWidget
        if notebook is not None:
            notebook.addTab(panel, name)
        self.input_pages.append(panel)
        return panel

    def NewDataFolder(self, name, pos=-1):
        panel = PluginPage()
        notebook = self.parent.ui.leftPluginTabWidget
        if notebook is None:
            notebook = self.parent.ui.leftTabWidget
        if notebook is not None:
            notebook.addTab(panel, name)
        self.data_pages.append(panel)
        return panel

    def NewMenu(self, name):
        menu = QtWidgets.QMenu(name, self.parent)
        self.parent.menuBar().addMenu(menu)
        self.menus.append(menu)
        return menu

    def StatusMessage(self, text):
        debug(text)
        sb = self.parent.statusBar()
        if sb is not None:
            sb.showMessage(text, 5000)

    def ShowErrorDialog(self, message):
        error(message)
        ShowErrorDialog(self.parent, message)

    def ShowInfoDialog(self, message):
        info(message)
        ShowInfoDialog(self.parent, message)

    def ShowWarningDialog(self, message):
        warning(message)
        ShowWarningDialog(self.parent, message)

    def ShowQuestionDialog(self, message):
        return ShowQuestionDialog(self.parent, message)

    def GetModel(self):
        return self.parent.model_control.get_model()

    def GetSolverControl(self):
        return self.parent.model_control

    def SetModelScript(self, script):
        self.parent.model_control.set_model_script(script)

    def GetModelScript(self):
        return self.parent.model_control.get_model_script()

    def CompileScript(self):
        self.parent.model_control.force_compile()

    def GetScriptModule(self):
        return self.parent.model_control.script_module

    def OnNewModel(self, event):
        pass

    def OnDataChanged(self, event):
        pass

    def OnOpenModel(self, event):
        pass

    def OnSimulate(self, event):
        pass

    def OnFittingUpdate(self, event):
        pass

    def OnGridChange(self, event):
        pass

    def Remove(self):
        plot_nb = self._get_plugin_tab_widget()
        input_nb = self.parent.ui.inputTabWidget
        data_nb = self.parent.ui.leftPluginTabWidget

        for panel in self.plot_pages:
            idx = plot_nb.indexOf(panel)
            if self._plugin_tab_count(plot_nb) == 1 and idx >= 0:
                self._ensure_plugin_empty_tab(plot_nb)
            if idx >= 0:
                plot_nb.removeTab(idx)
        for panel in self.input_pages:
            idx = input_nb.indexOf(panel)
            if idx >= 0:
                input_nb.removeTab(idx)
        for panel in self.data_pages:
            idx = data_nb.indexOf(panel)
            if idx >= 0:
                data_nb.removeTab(idx)

        for menu in self.menus:
            self.parent.menuBar().removeAction(menu.menuAction())


@dataclass
class PluginConfig(BaseConfig):
    section = "plugins"
    loaded_plugins: str = ""


class PluginController(Configurable):
    opt: PluginConfig

    def __init__(self, parent, menu: QtWidgets.QMenu):
        self.plugin_handler = PluginHandler(parent, __MODULE_DIR__, "add_ons")
        self.parent = parent
        Configurable.__init__(self)

        self.load_menu = QtWidgets.QMenu("Load", menu)
        self.unload_menu = QtWidgets.QMenu("Unload", menu)
        self.action_update = QtGui.QAction("Update module list", menu)
        self.action_update.triggered.connect(self.update_plugins)

        menu.addMenu(self.load_menu)
        menu.addMenu(self.unload_menu)
        menu.addSeparator()
        menu.addAction(self.action_update)

    def update_plugins(self):
        self.load_menu.clear()
        modlist = self.plugin_handler.get_plugins()
        modlist.sort()

        for mod in modlist:
            action = QtGui.QAction(mod, self.load_menu)
            action.triggered.connect(lambda _=False, name=mod: self.LoadPlugin(name))
            self.load_menu.addAction(action)

        self.update_config()

    def RegisterPlugin(self, plugin):
        action = QtGui.QAction(plugin, self.unload_menu)
        action.triggered.connect(lambda _=False, name=plugin: self.UnLoadPlugin_by_Name(name))
        self.unload_menu.addAction(action)
        self.update_plugins()

    def update_config(self):
        loaded_plugins = self.plugin_handler.get_loaded_plugins()
        self.opt.loaded_plugins = ";".join(loaded_plugins)
        self.WriteConfig()

    def LoadDefaultPlugins(self):
        self.ReadConfig()
        plugin_str = self.opt.loaded_plugins
        if plugin_str == "":
            self.update_plugins()
            return
        existing_plugins = self.plugin_handler.get_possible_plugins()
        not_found = []
        not_loadable = []
        for plugin in plugin_str.split(";"):
            if not self.plugin_handler.is_loaded(plugin):
                if plugin in existing_plugins:
                    try:
                        self.plugin_handler.load_plugin(plugin)
                        self.RegisterPlugin(plugin)
                    except Exception:
                        outp = io.StringIO()
                        traceback.print_exc(200, outp)
                        tbtext = outp.getvalue()
                        outp.close()
                        not_loadable.append((plugin, tbtext))
                else:
                    not_found.append(plugin)
        if not_loadable:
            text = "Can NOT load plugin "
            tbtext = "\nPython traceback below:\n\n"
            for pi, tbti in not_loadable:
                text += pi+'; '
                tbtext += tbti+'\n'
            ShowErrorDialog(
                self.parent, text[:-2]+tbtext
                )
        elif not_found:
            ShowInfoDialog(
                self.parent,
                'Could not find plugin(s) "%s"'
                ". Either there is an error in the config file"
                " or the plugin is not installed."%(";".join(not_found)),
                )
        self.update_plugins()

    def LoadPlugin(self, plugin):
        try:
            self.plugin_handler.load_plugin(plugin)
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
            ShowErrorDialog(self.parent, "Can NOT load plugin " + plugin + "\nPython traceback below:\n\n" + tbtext)
        else:
            self.RegisterPlugin(plugin)

    def UnLoadPlugin_by_Name(self, plugin):
        try:
            self.plugin_handler.unload_plugin(plugin)
        except Exception:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
            ShowErrorDialog(
                self.parent, "Can NOT unload plugin object" + plugin + "\nPython traceback below:\n\n" + tbtext
            )
            return False
        else:
            items = list(self.unload_menu.actions())
            for item in items:
                if item.text() == plugin:
                    self.unload_menu.removeAction(item)
                    break
            self.update_plugins()
            return True

    def OnNewModel(self, event=None):
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnNewModel(event)

    def OnDataChanged(self, event=None):
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnDataChanged(event)

    def OnOpenModel(self, event=None):
        loaded_plugins = list(self.plugin_handler.loaded_plugins.keys())
        for name in loaded_plugins:
            self.plugin_handler.unload_plugin(name)

        self.unload_menu.clear()
        self.LoadDefaultPlugins()
        self.update_plugins()

        loaded_plugins = list(self.plugin_handler.loaded_plugins.keys())
        for name in loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnOpenModel(event)

    @QtCore.Slot(object)
    def on_model_loaded(self, _model) -> None:
        self.OnOpenModel(None)

    def OnSimulate(self, event=None):
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnSimulate(event)

    def OnFittingUpdate(self, event=None):
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnFittingUpdate(event)

    def OnGridChanged(self, event=None):
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnGridChange(event)

    def GetPlugin(self, plugin_name):
        return self.plugin_handler.loaded_plugins[plugin_name]
