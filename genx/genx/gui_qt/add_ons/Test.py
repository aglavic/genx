"""<h1> Test </h1>
A plugin to test the builtin features for the plugin.
Opens three new tabs, one in each area of the main window.
It also adds a a custom menu.

Qt port.
"""

from genx.core.custom_logging import iprint
from genx.plugins import add_on_framework as framework


class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.NewPlotFolder("Test")
        self.NewInputFolder("Test")
        self.NewDataFolder("Test")
        self.NewMenu("Test")
        iprint("Everything tested, should be visible :-)")
