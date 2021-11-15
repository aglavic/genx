'''
===============================
:mod:`xrdml`  XRDML data loader
===============================

Loads the data from Philips XPert instrument.
'''

import numpy as np
from xml.dom.minidom import parseString

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog

try:
    import wx
    from wx.lib.masked import NumCtrl
except ImportError:
    class void():
        pass

    wx=void()
    wx.Dialog=void

class Plugin(Template):
    wildcard='*.xrdml'

    def __init__(self, parent):
        Template.__init__(self, parent)
        self.x_col=0
        self.y_col=1
        self.e_col=2
        self.xe_col=-1
        self.comment='#'
        self.skip_rows=0
        self.delimiter=None

    def CountDatasets(self, file_path):
        try:
            orso_datasets=self.ReadXpert(file_path)
        except Exception as e:
            return 1
        else:
            return len(orso_datasets)

    def LoadData(self, dataset, filename, data_id=0):
        '''
        Loads the data from filename into the data_item_number.
        '''
        try:
            datasets=ReadXpert(filename)
        except Exception as e:
            import traceback
            ShowWarningDialog(self.parent, 'Could not load the file: '+ \
                              filename+' \nPlease check the format.\n\n Error in ReadXpert:\n'+
                              traceback.format_exc())
        else:
            ds=datasets[data_id]
            dataset.x_raw=ds[0]
            dataset.y_raw=ds[1]
            dataset.error_raw=ds[2]
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta['data_source']['experiment']['instrument']='XRDML'
            dataset.meta['data_source']['experiment']['probe']='xray'
            dataset.meta['data_source']['measurement']['scheme']='angle-dispersive'

    def SettingsDialog(self):
        '''
        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        '''
        col_values={'y': self.y_col, 'x': self.x_col,
                    'y error': self.e_col, 'x error': self.xe_col}
        misc_values={'Comment': str(self.comment), 'Skip rows': self.skip_rows,
                     'Delimiter': str(self.delimiter)}
        dlg=SettingsDialog(self.parent, col_values, misc_values)
        if dlg.ShowModal()==wx.ID_OK:
            col_values=dlg.GetColumnValues()
            misc_values=dlg.GetMiscValues()
            self.y_col=col_values['y']
            self.x_col=col_values['x']
            self.e_col=col_values['y error']
            self.xe_col=col_values['x error']
            self.comment=misc_values['Comment']
            self.skip_rows=misc_values['Skip rows']
            self.delimiter=misc_values['Delimiter']
        dlg.Destroy()

class SettingsDialog(wx.Dialog):

    def __init__(self, parent, col_values, misc_values):
        wx.Dialog.__init__(self, parent, -1, 'Data loader settings')

        box_sizer=wx.BoxSizer(wx.HORIZONTAL)

        # Make the box for putting in the columns
        col_box=wx.StaticBox(self, -1, "Columns")
        col_box_sizer=wx.StaticBoxSizer(col_box, wx.VERTICAL)

        # col_values = {'y': 1,'x': 0,'y error': 1}
        col_grid=wx.GridBagSizer(len(col_values), 2)
        self.col_controls=col_values.copy()
        keys=list(col_values.keys())
        keys.sort()
        for i, name in enumerate(keys):
            text=wx.StaticText(self, -1, name+': ')
            control=wx.SpinCtrl(self)
            control.SetRange(-10, 100)
            control.SetValue(col_values[name])
            col_grid.Add(text, (i, 0),
                         flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL,
                         border=5)
            col_grid.Add(control, (i, 1),
                         flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL,
                         border=5)
            self.col_controls[name]=control

        col_box_sizer.Add(col_grid, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        box_sizer.Add(col_box_sizer, 0, wx.ALL | wx.EXPAND, 5)

        col_box=wx.StaticBox(self, -1, "Misc")
        col_box_sizer=wx.StaticBoxSizer(col_box, wx.VERTICAL)

        # Lets add another box for comments and rows to skip
        # misc_values = {'Comment': '#', 'Skip rows': 0,'Delimiter': 'None'}
        col_grid=wx.GridBagSizer(len(misc_values), 2)
        self.misc_controls=misc_values.copy()
        keys=list(misc_values.keys())
        keys.sort()
        for i, name in enumerate(keys):
            text=wx.StaticText(self, -1, name+': ')
            if type(misc_values[name])==type(1):
                control=wx.SpinCtrl(self)
                control.SetRange(0, 100)
                control.SetValue(misc_values[name])
            else:
                control=wx.TextCtrl(self, value=misc_values[name],
                                    style=wx.EXPAND)
            col_grid.Add(text, (i, 0),
                         flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL,
                         border=5)
            col_grid.Add(control, (i, 1),
                         flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL,
                         border=5)
            self.misc_controls[name]=control

        col_box_sizer.Add(col_grid, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        box_sizer.Add(col_box_sizer, 0, wx.ALL | wx.EXPAND, 5)

        button_sizer=wx.StdDialogButtonSizer()
        okay_button=wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()

        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add(box_sizer, 1, wx.GROW, 20)
        line=wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW, 30)

        sizer.Add(button_sizer, 0,
                  flag=wx.ALIGN_RIGHT, border=20)
        self.SetSizer(sizer)

        sizer.Fit(self)
        self.Layout()

    def GetColumnValues(self):
        values={}
        for key in self.col_controls:
            values[key]=self.col_controls[key].GetValue()
        return values

    def GetMiscValues(self):
        values={}
        for key in self.misc_controls:
            val=self.misc_controls[key].GetValue()
            if type(val)==type('') or type(val)==type(''):
                if val.lower()=='none':
                    val=None
            values[key]=val
        return values

def ReadXpert(file_name):
    '''
      Read the data of a philips X'Pert diffractometer file, exported as text files.
    '''
    raw_data=open(file_name, 'r').read()
    xml_data=parseString(raw_data).firstChild

    # retrieve data
    try:
        sample_name=xml_data.getElementsByTagName('sample')[0].getElementsByTagName('name')[0].firstChild.nodeValue
    except AttributeError:
        sample_name=file_name.rsplit('.', 1)[0]
    datasets=[]
    for xml_scan in xml_data.getElementsByTagName('xrdMeasurement')[0].getElementsByTagName('scan'):
        scan=xml_scan.getElementsByTagName('dataPoints')[0]

        moving_positions={}
        for motor in scan.getElementsByTagName('positions'):
            axis=motor.attributes['axis'].value
            if len(motor.getElementsByTagName('commonPosition'))==0:
                start=float(motor.getElementsByTagName('startPosition')[0].firstChild.nodeValue)
                end=float(motor.getElementsByTagName('endPosition')[0].firstChild.nodeValue)
                moving_positions[axis]=(start, end)

        try:
            atten_factors=scan.getElementsByTagName('beamAttenuationFactors')[0].firstChild.nodeValue
            atten_factors=list(map(float, atten_factors.split()))
            atten=np.array(atten_factors)
        except IndexError:
            atten=1.0
        time=float(scan.getElementsByTagName('commonCountingTime')[0].firstChild.nodeValue)
        data_tags=scan.getElementsByTagName('intensities')+scan.getElementsByTagName('counts')
        data=data_tags[0].firstChild.nodeValue
        data=list(map(float, data.split()))
        I=np.array(data)
        dI=np.sqrt(I*atten)/atten
        I/=time
        dI/=time
        if '2Theta' in moving_positions:
            th=np.linspace(moving_positions['2Theta'][0],
                           moving_positions['2Theta'][1], len(data))/2.
        else:
            th=np.linspace(moving_positions['Omega'][0],
                           moving_positions['Omega'][1], len(data))
        Q=4.*np.pi/1.54*np.sin(th/180.*np.pi)
        datasets.append((Q, I, dI, sample_name))
    return datasets
