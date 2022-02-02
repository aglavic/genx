'''<h1>Shell</h1>
A plugin which opens a up a new tab with a shell inside.
Can be used for introspection in GenX. To acess the module
which holds the script type: <code>model.script_module.[object]</code>.
This is mostly intended for searching for bugs and test expresions
and such when programming. 
The available entry points into GenX is:
<dl>
    <dt><b><code>frame</code></b></dt>
    <dd>The top level frame. All the widgets and windows of GenX can be acessed
    in this object</dd>
    <dt><b><code>data</code></b></dt>
    <dd>The data object. All the loaded data can acessed through this object.
    Each data set is acessed as in a list, i.e., data[0] for the first data set.
    x and y components are acesssed through data[0].x and data[0].y, 
    respectively.</dd>
    <dt><b><code>model</code></b></dt>
    <dd>Access to the model object. Stores the script and the module. 
    Important object.</dd>
</dl>
'''
from .. import add_on_framework as framework
import wx.py.shell, wx

class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        inputpanel=self.NewInputFolder('Shell')
        sizer=wx.BoxSizer(wx.HORIZONTAL)
        inputpanel.SetSizer(sizer)
        self.shell=wx.py.shell.Shell(inputpanel, -1,
                                     locals={'frame': parent, 'model': self.GetModel(),
                                             'data': self.GetModel().get_data(),
                                             'ctrl': parent.model_control.controller})
        sizer.Add(self.shell, 1, wx.EXPAND)
        inputpanel.Layout()
        self.StatusMessage('Shell plugin loaded')
