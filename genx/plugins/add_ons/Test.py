import plugins.add_on_framework as framework
import plotpanel

class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        plotpanel = self.NewPlotFolder('Test')
        inputpanel = self.NewInputFolder('Test')
        datapanel = self.NewDataFolder('Test')
        menu = self.NewMenu('Test')
        print 'Everyting tested, should be visible :-)'
    