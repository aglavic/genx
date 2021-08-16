'''<h1> Test </h1>
A plugin to test the builtin features for the plugin.
Opens three new tabs, one in each area of the main window.
It also adds a a custom menu.
'''
from .. import add_on_framework as framework
from genx.core.custom_logging import iprint

class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        plotpanel=self.NewPlotFolder('Test')
        inputpanel=self.NewInputFolder('Test')
        datapanel=self.NewDataFolder('Test')
        menu=self.NewMenu('Test')
        iprint('Everyting tested, should be visible :-)')
