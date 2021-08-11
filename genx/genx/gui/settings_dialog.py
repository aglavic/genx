"""
A general wx.Dialog to deal with user configurable settings.
"""

import wx

from ..core.config import BaseConfig

class SettingsDialog(wx.Dialog):
    def __init__(self, parent: wx.Window, settings: BaseConfig, title='Settings', apply_callback:callable=None):
        '''
        Configuration optitons for a DiffEv solver.
        '''
        wx.Dialog.__init__(self, parent, -1, title)
        self.settings=settings
        if apply_callback is None:
            self.apply_callback=lambda: None
        else:
            self.apply_callback=apply_callback

        self.build_sizers()

        groups=list(self.settings.groups.keys())
        n_groups=len(groups)
        r_groups=n_groups//2
        l_groups=n_groups-r_groups
        for lg in groups[:l_groups]:
            self.build_group(lg, self.row_sizer1)
        for rg in groups[l_groups:]:
            self.build_group(rg, self.row_sizer2)

        self.GetSizer().Fit(self)
        self.Layout()

    def build_sizers(self):
        col_sizer=wx.BoxSizer(wx.HORIZONTAL)
        self.row_sizer1=wx.BoxSizer(wx.VERTICAL)
        self.row_sizer2=wx.BoxSizer(wx.VERTICAL)
        col_sizer.Add(self.row_sizer1, 1, wx.ALIGN_CENTRE | wx.ALL, 5)
        col_sizer.Add(self.row_sizer2, 1, wx.ALIGN_CENTRE | wx.ALL, 5)

        # Add the Dialog buttons
        button_sizer=wx.StdDialogButtonSizer()
        okay_button=wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        apply_button=wx.Button(self, wx.ID_APPLY)
        apply_button.SetDefault()
        button_sizer.AddButton(apply_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        # Add some event handlers
        self.Bind(wx.EVT_BUTTON, self.on_apply_changes, okay_button)
        self.Bind(wx.EVT_BUTTON, self.on_apply_changes, apply_button)

        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add(col_sizer, 1, wx.GROW, 20)
        # sizer.Add(col_sizer, 1, wx.GROW|wx.ALL|wx.EXPAND, 20)
        line=wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.TOP, 20)

        sizer.Add(button_sizer, 0,
                  flag=wx.ALIGN_RIGHT, border=20)
        self.SetSizer(sizer)


    def build_group(self, name:str, sizer:wx.BoxSizer):
        box=wx.StaticBox(self, -1, name)
        box_sizer=wx.StaticBoxSizer(box, orient=wx.VERTICAL)
        sizer.Add(box_sizer, 0, wx.EXPAND)

        pnames=self.settings.groups[name]
        fields=dict((fi.name, fi) for fi in self.settings.get_fields())
        for ni in pnames:
            if type(ni) in [list, tuple]:
                raise NotImplementedError("Horizontal grouping not yet supported")
            field=fields[ni]
            entry=self.get_entry(field, box)
            box_sizer.Add(entry, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

    def get_entry(self, field, parent):
        gmeta=field.metadata.get('genx', {})
        if 'label' in gmeta:
            name=gmeta['label']
        else:
            name=field.name.replace('_', ' ')
        if field.type is bool:
            entry=wx.CheckBox(parent, -1, name)
            entry.SetValue(getattr(self.settings, field.name, False))
        elif 'selection' in gmeta:
            choice = wx.Choice(self, -1, choices=gmeta['selection'])
            choice.SetSelection(gmeta['selection'].index(getattr(self.settings, field.name, gmeta['selection'][0])))
            entry = wx.BoxSizer(wx.HORIZONTAL)
            entry.Add(wx.StaticText(parent, -1, name), 0, wx.ALIGN_CENTER_VERTICAL, 0)
            entry.Add(choice)
        else:
            txt=wx.TextCtrl(parent, -1)
            txt.SetValue(str(getattr(self.settings, field.name, '')))
            entry = wx.BoxSizer(wx.HORIZONTAL)
            entry.Add(wx.StaticText(parent, -1, name), 0, wx.ALIGN_CENTER_VERTICAL, 0)
            entry.Add(txt)
        return entry

    def on_apply_changes(self, event):
        pass