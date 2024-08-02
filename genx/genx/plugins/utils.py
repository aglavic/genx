""" utils.py

module for utils needed for the plugin frameworks. This module implemtents a
handler which searches for plugins and loads and removes them.
Programmer: Matts Bjorck
Last changed: 2008 07 23
"""

import os
import sys

from logging import error, info, warning

try:
    import wx
except ImportError:
    wx = None

# ==============================================================================


class PluginHandler:
    """Class that takes care of the loading/unloading of the plugins.
    This is used in the main GUI. Should not be tinkered with.
    The plugin module can only conatain files wich are real plugins i.e.
    that implements the class Template. Of course it is possible to have
    subdirectories that has support libraries.

    Note that the plugins modules has to have an class called Plugin
    and that this class should implement the method Remove. The init function
    should only take the parent (i.e. the original window frame) as input.
    """

    def __init__(self, parent, directory, plugin_folder):
        """__init__(self, parent)
        Create an instance of PluginHandler
        """
        self.parent = parent
        # self.modules = []
        self.loaded_plugins = {}
        self.directory = directory
        self.plugin_folder = plugin_folder

    def get_plugins(self):
        """get_possible_plugins(self) --> list of strings

        search the plugin directory.
        Checks the list for python scripts and returns a list of
        module names that are NOT loaded.
        """
        # Locate all python files in this files directory
        # but excluding this file and not loaded.
        plugins = [
            s.rsplit(".", 1)[0]
            for s in os.listdir(self.directory + self.plugin_folder)
            if s.split(".", 1)[-1] in ["py", "pyo", "pyc"]
            and s[:2] != "__"
            and s.rsplit(".", 1)[0] not in self.loaded_plugins
        ]
        return plugins

    def get_possible_plugins(self):
        """get_possible_plugins(self) --> list of strings

        search the plugin directory.
        Checks the list for python scripts and returns a list of
        module names that are loadable .
        """
        # Locate all python files in this files directory
        # but excluding this file and not loaded.
        plugins = [s[:-3] for s in os.listdir(self.directory + self.plugin_folder) if ".py" == s[-3:] and s[:2] != "__"]
        return plugins

    def get_loaded_plugins(self):
        """get_loaded_plugins(self) --> plugins [list]

        returns a list of the loaded plugins
        """
        return list(self.loaded_plugins.keys())

    def load_plugin(self, plugin_name):
        """load_plugin(self, plugin_name) --> plugin object
        load the plugin given by the plugin_name [string].
        """
        # Load the module
        module = self._load_module(self.plugin_folder + "." + plugin_name)
        # Create the Plugin object from the module
        plugin = module.Plugin(self.parent)
        # Add it to our dictonary over loaded plugins
        self.loaded_plugins[plugin_name] = plugin

    def is_loaded(self, plugin_name):
        """isloaded(self, plugin_name) --> loaded [bool]

        Returns True if plugin with name plugin_name is loaded,
        otherwise False.
        """
        return plugin_name in self.loaded_plugins

    def _load_module(self, module_name):
        """_load_module(self, module) --> module
        Load a module given by name
        """
        # print 'Trying to load module: ', module_name
        module = __import__(module_name, globals(), locals(), ["plugins"], level=1)
        return module

    def unload_plugin(self, plugin_name):
        """unload_plugin(self, plugin_name) --> None
        Used to remove the plugin from the system.
        """
        # print self.loaded_plugins.keys()
        self.loaded_plugins[plugin_name].Remove()
        del self.loaded_plugins[plugin_name]


# END: PluginHandler
# ==============================================================================
# Utility Dialog functions..
def ShowInfoDialog(frame, message, title="Information"):
    if wx is None:
        print(message)
        return
    else:
        exc_info = sys.exc_info()
        info(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_INFORMATION)
    dlg.ShowModal()
    dlg.Destroy()


def ShowWarningDialog(frame, message, title="Warning"):
    if wx is None:
        print(message)
        return
    else:
        exc_info = sys.exc_info()
        warning(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()


def ShowErrorDialog(frame, message, title="ERROR"):
    if wx is None:
        print(message)
        return
    else:
        exc_info = sys.exc_info()
        error(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()


def ShowQuestionDialog(frame, message, title="Question"):
    if wx is None:
        result = input(message + " [y/n]").strip().lower() in ["y", "yes"]
        return result
    dlg = wx.MessageDialog(frame, message, title, wx.YES_NO | wx.ICON_QUESTION)
    result = dlg.ShowModal() == wx.ID_YES
    dlg.Destroy()
    return result
