import plugins.add_on_framework as framework
import wx.py.shell, wx

class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        inputpanel = self.NewInputFolder('Shell')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        inputpanel.SetSizer(sizer)
        self.shell = wx.py.shell.Shell(inputpanel, -1,\
            locals = {'frame': parent, 'model': self.GetModel(),\
                'data': self.GetModel().get_data()})
        sizer.Add(self.shell,1, wx.EXPAND)
        inputpanel.Layout()