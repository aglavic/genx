"""
====================================================
:mod:`nicos_ascii_scan` NICOS ASCII scan data loader
====================================================

Loads the NICOS scan ASCII datafile format used for extracted reflectivity at
the MLZ reflectometer like XRR, TREFF, ... .
"""

import locale
import re
from os.path import basename, splitext

import numpy as np

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog

try:
    import wx
except ImportError:
    # in case of console usinge without wx being installed put a mock class/module
    class void:
        pass

    wx = void()
    wx.Dialog = void

INFO_CATEGORIES = [
    ('experiment', 'Experiment information'),
    ('sample', 'Sample and alignment'),
    ('instrument', 'Instrument setup'),
    ('offsets', 'Offsets'),
    ('limits', 'Limits'),
    ('precisions', 'Precisions/tolerances'),
    ('status', 'Device status'),
    ('general', 'Device positions and sample environment state'),
    ('presets', 'Detector preset information'),
    ('result', 'Updated values after counting'),
]

TIMEFMT = '%Y-%m-%d %H:%M:%S'

try:
    LOCALE_ENCODING = locale.getpreferredencoding(False)
except Exception:
    LOCALE_ENCODING = 'utf-8'


class Plugin(Template):

    wildcard = '*.dat'

    re_number_unit = re.compile(r'^(\d+\.?\d*)(\s(.+)|)')
    re_collection_unit = re.compile(r'^((\(|\[).+(\)|\]))(\s(.+)|)')

    class DevFake:

        def __init__(self, name):
            self.name = name

    def __init__(self, parent):
        Template.__init__(self, parent)
        self.metainfo = {}
        self.x_col = 0
        self.y_col = 1
        self.t_col = 0

    def CanOpen(self, filename):
        if not Template.CanOpen(self, filename):
            return False
        try:
            with open(filename, 'r', encoding=LOCALE_ENCODING) as fh:
                return fh.readline().startswith('### NICOS data file, created at')
        except FileNotFoundError:
            return False

    def getCategory(self, text):
        for cat, txt in INFO_CATEGORIES:
            if text == txt:
                return cat
        return ''

    def getKeyValue(self, line):
        k, v = line[1:].split(':', 1)
        return k.strip(), v.strip()

    def getDevicesDetectors(self, line):
        devs, dets = line.split(';')
        return devs.split(), dets.split()

    def getMetainfoKey(self, line):
        return line.rsplit('_', 1)

    def getMetainfoValue(self, line, entry):
        # session.log.debug(line)
        if line:
            res = self.re_number_unit.match(line)
            if res:
                v = res.group(1)
                return (v, v, res.group(2), entry)
            res = self.re_collection_unit.match(line)
            if res:
                v = res.group(1)
                return (v, v, res.group(5) or '', entry)
        return (line, line, '', entry)

    def LoadData(self, dataset, filename, data_id=0):
        """LoadData(self, data_item_number, filename) --> none

        Loads the data from filename into the data_item_number.
        """

        entry = ''
        scanheader = 0

        def guess_det_type(name, unit):
            if unit in ['s']:
                return 'time'
            elif unit in ['cts']:
                if name.startswith('m'):
                    return 'monitor'
                return 'counter'
            return 'other'

        with open(filename, 'r', encoding=LOCALE_ENCODING) as f:
            name = splitext(basename(filename))[0]
            # Name the dataset according to file name
            dataset.name = name
            x_raw = []
            y_raw = []
            det_raw = {}
            t_raw = []
            data_source = dataset.meta['data_source']
            data_source['import_columns'] = {
                'x': self.x_col,
                'y': self.y_col,
                't': self.t_col
            }
            data_source['experiment']['probe'] = 'neutron'
            for line in f:
                lin = line.strip()
                if lin.startswith('###'):
                    _, t = lin.split(' ', 1)
                    cat = self.getCategory(t)
                    if cat:
                        entry = cat
                    elif t.startswith('NICOS data file,'):
                        entry = 'header'
                        started = t.split(' ', 5)[-1].replace(' ', 'T')
                        data_source['experiment']['start_date'] = started
                    elif t.startswith('Scan data'):
                        entry = 'data'
                    elif t.startswith('End of NICOS data file'):
                        entry = ''
                elif lin.startswith('#'):
                    if entry == 'header':
                        key, value = self.getKeyValue(lin)
                        # if key == 'filename':
                        #     self.scandataset.filenames = [value]
                        # elif key == 'filepath':
                        #     self.scandataset.filepaths = [value]
                        # elif key == 'number':
                        #     self.scandataset.counter = '%s (%s)' % (
                        #         value, basename(filename))
                        # elif key == 'info':
                        #     self.scandataset.userinfo, _, self.scandataset.info = value.rpartition('-')
                        # else:
                        #     session.log.warning('unknown header: %s=%s', key, value)
                    elif entry == 'experiment':
                        key, value = self.getKeyValue(lin)
                        if key.endswith('_title'):
                            data_source['experiment']['title'] = value
                        elif key.endswith('_users'):
                            data_source['owner']['name'] = value
                    elif entry == 'sample':
                        key, value = self.getKeyValue(lin)
                        if key.endswith('samplename'):
                            data_source['sample']['name'] = value
                    elif entry == 'instrument':
                        key, value = self.getKeyValue(lin)
                        if key.endswith('_instrument'):
                            data_source['experiment']['instrument'] = value
                        elif key.endswith('_facility'):
                            data_source['facility'] = value
                        elif key=='image_anode' and value=='Cu':
                            data_source['measurement']['instrument_settings']['wavelength'] = {
                                'magnitude': 1.54056,
                                'unit': 'angstrom',
                                }
                        if key.startswith('xrr_'):
                            data_source['experiment']['probe'] = 'x-ray'
                    elif entry == 'data':
                        # get header lines devices and units
                        if not scanheader:
                            devs, dets = self.getDevicesDetectors(line[1:])
                            for di in dets:
                                if not 'file' in di:
                                    di = di.replace('.', '_')
                                    det_raw[di] = []
                            det_name = dets[self.y_col]
                            self.devs = [self.DevFake(dev) for dev in devs]
                            self.dets = [self.DevFake(det) for det in dets]
                        else:
                            devunits, detunits = self.getDevicesDetectors(line[1:])
                            # devvalueinfo = [(d.name, u)
                            #     for d, u in zip(self.devs, devunits)]
                            # detvalueinfo = [(d.name, unit=u,
                            #           type=guess_det_type(d.name, u))
                            #     for d, u in zip(self.dets, detunits)]
                            # envvalueinfo = []
                        scanheader += 1
                    elif not entry:
                        break
                    else:
                        key, value = self.getKeyValue(lin)
                        devname, devkey = self.getMetainfoKey(key)
                        self.metainfo[devname, devkey] = self.getMetainfoValue(
                            value, entry)
                else:
                    # get scan data
                    devvals, detvals = self.getDevicesDetectors(line)
                    x_raw.append(devvals[self.x_col])
                    y_raw.append(detvals[self.y_col])
                    t_raw.append(detvals[self.t_col])
                    for i, di in enumerate(dets):
                        if not 'file' in di:
                            di = di.replace('.', '_')
                            det_raw[di].append(detvals[i])
                    # values = {dev.name: (None, val)
                    #           for dev, val in zip(self.devs, devvals)}
                    # results = {det.name: [val]
                    #            for det, val in zip(self.dets, detvals)}

            dataset.x_raw = np.array(x_raw, dtype=float)
            data_source['measurement']['instrument_settings']['incident_angle'] = {
                'min': float(dataset.x_raw.min()/2.),
                'max': float(dataset.x_raw.max()/2.),
                'unit': 'deg'
                }
            dataset.y_raw = np.array(y_raw, dtype=float)
            for i, di in enumerate(dets):
                if not 'file' in di:
                    di = di.replace('.', '_')
                    dataset.extra_data_raw[di] = np.array(det_raw[di], dtype=float)
                    dataset.extra_commands[di] = di
            dataset.error_raw = dataset.y_raw * np.nan
            dataset.y_command = f'{dets[self.y_col]}/{dets[self.t_col]}'
            dataset.error_command = f'sqrt({dets[self.y_col]})/{dets[self.t_col]}'

            # Run the commands on the data - this also sets the x,y, error
            # members of the data item.
            dataset.run_extra_commands()
            dataset.run_command()
            # ShowWarningDialog(self.parent, f'metainfo: {dataset.meta}')

    def SettingsDialog(self):
        """SettingsDialog(self) --> None

        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        """
        col_values = {
            'y': self.y_col,
            'x': self.x_col,
            't': self.t_col,
        }
        dlg = SettingsDialog(self.parent, col_values)
        if dlg.ShowModal() == wx.ID_OK:
            col_values = dlg.GetColumnValues()
            self.y_col = col_values['y']
            self.x_col = col_values['x']
            self.t_col = col_values["t"]
        dlg.Destroy()


class SettingsDialog(wx.Dialog):

    def __init__(self, parent, col_values):
        wx.Dialog.__init__(self, parent, -1, 'Data loader settings')

        box_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Make the box for putting in the columns
        col_box = wx.StaticBox(self, -1, 'Columns')
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL)

        col_grid = wx.GridBagSizer(len(col_values), 2)
        self.col_controls = col_values.copy()
        keys = list(col_values.keys())
        keys.sort()
        for i, name in enumerate(keys):
            text = wx.StaticText(self, -1, f'{name}: ')
            control = wx.SpinCtrl(self)
            control.SetRange(0, 100)
            control.SetValue(col_values[name])
            col_grid.Add(text, (i, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
            col_grid.Add(control, (i, 1), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
            self.col_controls[name] = control

        col_box_sizer.Add(col_grid, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        box_sizer.Add(col_box_sizer, 0, wx.ALL | wx.EXPAND, 5)

        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(box_sizer, 1, wx.GROW, 20)
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW, 30)

        sizer.Add(button_sizer, 0, flag=wx.ALIGN_RIGHT, border=20)
        self.SetSizer(sizer)

        sizer.Fit(self)
        self.Layout()

    def GetColumnValues(self):
        values = {}
        for key in self.col_controls:
            values[key] = self.col_controls[key].GetValue()
        return values
