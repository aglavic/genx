"""data_loader_wx.py

Implements the GUI side of data_loader as DataLoaderController which
takes care of the use of the DataLoaders.
"""

import io
import os
import traceback

import wx

from genx.core.config import config
from genx.core.custom_logging import iprint

from .utils import PluginHandler, ShowErrorDialog

head, tail = os.path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled...
__FILENAME__ = tail.split(".")[0]
# This assumes that plugin is under the current dir may need
# changing
__MODULE_DIR__ = head
if __MODULE_DIR__ != "/":
    __MODULE_DIR__ += "/"


class PluginController:
    def __init__(self, parent):
        self.plugin_handler = PluginHandler(parent, __MODULE_DIR__, "data_loaders")
        self.parent = parent
        self.plugin_handler.load_plugin("auto")

    def load_default(self):
        try:
            plugin_name = config.get("data handling", "data loader")
            self.LoadPlugin(plugin_name)
        except Exception as S:
            iprint("Could not locate the data loader parameter or the data loader. Error:")
            iprint(S.__str__())
            iprint("Proceeding with laoding the default data loader.")
            self.LoadPlugin("default")

    def LoadPlugin(self, plugin):
        """
        Loads a data handler note that there is no UnLoad function
        since only one DataHandler can be plugged in at a time.
        """
        # Unload the plugins
        names = self.plugin_handler.loaded_plugins.copy()
        try:
            [self.plugin_handler.unload_plugin(pl) for pl in names]
            self.parent.SetStatusText("Unloaded data loader %s" % list(names.keys())[0])
        except:
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
            self.parent.SetStatusText("Loaded data loader: %s" % plugin)
        except:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
            ShowErrorDialog(self.parent, "Can NOT load plugin " + plugin + "\nPython traceback below:\n\n" + tbtext)

    def ShowDialog(self):
        """ShowDialog(self) --> None

        Shows a dialog boc for the user to choose a data loader.
        """
        cur_plugin = list(self.plugin_handler.loaded_plugins.keys())[0]
        plugin_list = self.plugin_handler.get_possible_plugins()
        dlg = PluginDialog(self.parent, plugin_list, cur_plugin, self.LoadPlugin)
        dlg.ShowModal()
        dlg.Destroy()


class PluginDialog(wx.Dialog):
    def __init__(self, parent, plugin_list, current_plugin, load_plugin_func=None):
        wx.Dialog.__init__(self, parent, -1, "Choose a data loader")

        self.load_plugin_func = load_plugin_func

        choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer = wx.BoxSizer(wx.VERTICAL)

        choice_text = wx.StaticText(self, -1, "Data loaders: ")
        self.choice_control = wx.Choice(self, -1, choices=plugin_list)
        self.choice_control.SetStringSelection(current_plugin)
        choice_sizer.Add(choice_text, 0, wx.ALIGN_CENTER_VERTICAL)
        choice_sizer.Add(self.choice_control, 0, wx.ALIGN_CENTER_VERTICAL)

        # Add the Dialog buttons
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        apply_button = wx.Button(self, wx.ID_APPLY)
        apply_button.SetDefault()
        button_sizer.AddButton(apply_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        # Add some eventhandlers
        self.Bind(wx.EVT_BUTTON, self.on_apply, okay_button)
        self.Bind(wx.EVT_BUTTON, self.on_apply, apply_button)

        sizer.Add((20, 20), 0)
        sizer.Add(choice_sizer, 0, wx.ALIGN_CENTRE | wx.ALL)
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.TOP, 20)
        sizer.Add(button_sizer)

        self.SetSizer(sizer)

        sizer.Fit(self)
        self.Layout()

    def on_apply(self, event):
        if self.load_plugin_func is not None:
            self.load_plugin_func(self.choice_control.GetStringSelection())
        event.Skip()
