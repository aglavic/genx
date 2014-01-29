''' <h1>Default data loader</h1>

Loads the data from whitespace seperated column formatted ascii data files.
The module allows the specification of which columns that correspond to the
x, y and y_error columns in the data file. 
If y_error is not used the it can safely be set to the same column as y. <p>

Which columns that are imported are determined from the dialog box in 
import settings. Note that these settings apply to the marked data set(s).
Other possible tunings are the definitions of delimiter, None means any 
white space characthre (default). Skip rows is how many rows are skipped before 
the file is started to be read. Comment is the first chrachter of a commented 
line.
'''

import numpy as np
import wx
from wx.lib.masked import NumCtrl

from plugins.data_loader_framework import Template
from plugins.utils import ShowErrorDialog, ShowWarningDialog, ShowInfoDialog

class Plugin(Template):
    def __init__(self, parent):
        Template.__init__(self, parent)
        self.x_col = 0
        self.y_col = 1
        self.e_col = 1
        self.comment = '#'
        self.skip_rows = 0
        self.delimiter = None
    
    def LoadData(self, data_item_number, filename):
        '''LoadData(self, data_item_number, filename) --> none
        
        Loads the data from filename into the data_item_number.
        '''
        try:
            load_array = np.loadtxt(filename, delimiter = self.delimiter, 
                comments = self.comment, skiprows = self.skip_rows)
        except Exception, e:
            ShowWarningDialog(self.parent, 'Could not load the file: ' +\
                    filename + ' \nPlease check the format.\n\n numpy.loadtxt'\
                    + ' gave the following error:\n'  +  str(e))
        else:
            # For the freak case of only one data point
            if len(load_array.shape) < 2:
                load_array = np.array([load_array])
            # Check so we have enough columns
            if load_array.shape[1]-1 < max(self.x_col, self.y_col, self.e_col):
                ShowWarningDialog(self.parent, 'The data file does not contain'\
                        + 'enough number of columns. It has ' + str(load_array[1])\
                        + ' columns. Rember that the column index start at zero!')
                # Okay now we have showed a dialog lets bail out ...
                return
            # The data is set by the default Template.__init__ function, neat hu
            # Know the loaded data goes into *_raw so that they are not
            # changed by the transforms
            self.data = self.parent.data_cont.get_data()
            self.data[data_item_number].x_raw = load_array[:, self.x_col]
            self.data[data_item_number].y_raw = load_array[:, self.y_col]
            self.data[data_item_number].error_raw = load_array[:, self.e_col]
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            self.data[data_item_number].run_command()
            
            self.UpdateDataList()
            
            # Send an update that new data has been loaded
            self.SendUpdateDataEvent()
        
    def SettingsDialog(self):
        '''SettingsDialog(self) --> None
        
        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        '''
        col_values = {'y': self.y_col,'x': self.x_col,'y error': self.e_col}
        misc_values = {'Comment': str(self.comment), 'Skip rows': self.skip_rows,\
                'Delimiter': str(self.delimiter)}
        dlg = SettingsDialog(self.parent, col_values, misc_values)
        if dlg.ShowModal() == wx.ID_OK:
            col_values = dlg.GetColumnValues()
            misc_values = dlg.GetMiscValues()
            self.y_col = col_values['y']
            self.x_col = col_values['x']
            self.e_col = col_values['y error']
            self.comment = misc_values['Comment']
            self.skip_rows = misc_values['Skip rows']
            self.delimiter = misc_values['Delimiter']
        dlg.Destroy()
        
        
class SettingsDialog(wx.Dialog):
    
    def __init__(self, parent, col_values, misc_values):
        wx.Dialog.__init__(self, parent, -1, 'Data loader settings')
        
        box_sizer = wx.BoxSizer(wx.HORIZONTAL)
        
        # Make the box for putting in the columns
        col_box = wx.StaticBox(self, -1, "Columns" )
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL )
        
        
        #col_values = {'y': 1,'x': 0,'y error': 1}
        col_grid = wx.GridBagSizer(len(col_values), 2)
        self.col_controls = col_values.copy()
        keys = col_values.keys()
        keys.sort()
        for i, name in enumerate(keys):
            text = wx.StaticText(self, -1, name+': ')
            control = wx.SpinCtrl(self)
            control.SetRange(0,100)
            control.SetValue(col_values[name])
            col_grid.Add(text, (i,0),\
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,\
                    border = 5)
            col_grid.Add(control, (i,1),\
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,\
                    border = 5)
            self.col_controls[name] = control
        
        col_box_sizer.Add(col_grid, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        box_sizer.Add(col_box_sizer, 0, wx.ALIGN_CENTRE|wx.ALL|wx.EXPAND, 5)
        
        col_box = wx.StaticBox(self, -1, "Misc" )
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL )
        
        # Lets add another box for comments and rows to skip
        #misc_values = {'Comment': '#', 'Skip rows': 0,'Delimiter': 'None'}
        col_grid = wx.GridBagSizer(len(misc_values), 2)
        self.misc_controls = misc_values.copy()
        keys = misc_values.keys()
        keys.sort()
        for i, name in enumerate(keys):
            text = wx.StaticText(self, -1, name+': ')
            if type(misc_values[name]) == type(1):
                control = wx.SpinCtrl(self)
                control.SetRange(0,100)
                control.SetValue(misc_values[name])
            else:
                control = wx.TextCtrl(self, value = misc_values[name],\
                        style = wx.EXPAND)
            col_grid.Add(text, (i,0),\
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,\
                    border = 5)
            col_grid.Add(control, (i,1),\
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,\
                    border = 5)
            self.misc_controls[name] = control
        
        col_box_sizer.Add(col_grid, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        box_sizer.Add(col_box_sizer, 0, wx.ALIGN_CENTRE|wx.ALL|wx.EXPAND, 5)
        
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(box_sizer, 1, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL, 20)
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL, 30)
        
        sizer.Add(button_sizer,0,\
                flag = wx.ALIGN_RIGHT, border = 20)
        self.SetSizer(sizer)
        
        sizer.Fit(self)
        self.Layout()
        
    def GetColumnValues(self):
        values = {}
        for key in self.col_controls:
            values[key] = self.col_controls[key].GetValue()
        return values
    
    def GetMiscValues(self):
        values = {}
        for key in self.misc_controls:
            val = self.misc_controls[key].GetValue()
            if (type(val) == type(u'') or type(val) == type('')):
                if val.lower() == 'none':
                    val = None
            values[key] = val
        return values
