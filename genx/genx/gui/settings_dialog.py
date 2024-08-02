"""
A general wx.Dialog to deal with user configurable settings.
"""

from math import log10

import wx

from ..core.config import BaseConfig


class SettingsDialog(wx.Dialog):
    def __init__(self, parent: wx.Window, settings: BaseConfig, title="Settings", apply_callback: callable = None):
        """
        Configuration dialog for BaseConfig based classes.

        The apply_callback is a function that is called with a dictionary of updated options that
        can be evaluated. It returns a bool to determin if the dialog should apply these options to the
        config object.
        """
        wx.Dialog.__init__(self, parent, -1, title, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
        self.settings = settings
        if apply_callback is None:
            self.apply_callback = lambda options: True
        else:
            self.apply_callback = apply_callback

        self.build_sizers()
        self.rb_actions = {}  # entries for toggling radio button selected entries enabled
        self.rb_bool_fields = {}  # the associated with the above action
        self.value_entries = {}  # all controls for settable values, the key is field.name

        groups = list(self.settings.groups.keys())
        n_groups = len(groups)
        r_groups = n_groups // 2
        l_groups = n_groups - r_groups
        for lg in groups[:l_groups]:
            self.build_group(lg, self.row_sizer1)
        for rg in groups[l_groups:]:
            self.build_group(rg, self.row_sizer2)

        self.GetSizer().Fit(self)
        self.Layout()

    def build_sizers(self, add_apply=False):
        col_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.row_sizer1 = wx.BoxSizer(wx.VERTICAL)
        self.row_sizer2 = wx.BoxSizer(wx.VERTICAL)
        col_sizer.Add(self.row_sizer1, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)
        col_sizer.Add(self.row_sizer2, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        # Add the Dialog buttons
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        if add_apply:
            apply_button = wx.Button(self, wx.ID_APPLY)
            apply_button.SetDefault()
            button_sizer.AddButton(apply_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        # Add some event handlers
        self.Bind(wx.EVT_BUTTON, self.on_apply_changes)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(col_sizer, 1, wx.GROW | wx.LEFT | wx.RIGHT, 10)
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.TOP, 10)

        sizer.Add(button_sizer, 0, flag=wx.ALIGN_RIGHT, border=4)
        self.SetSizer(sizer)

    def build_group(self, name: str, sizer: wx.BoxSizer):
        box = wx.StaticBox(self, -1, name)
        box_sizer = wx.StaticBoxSizer(box, orient=wx.VERTICAL)
        sizer.Add(box_sizer, 0, wx.EXPAND)

        pnames = self.settings.groups[name]
        fields = dict((fi.name, fi) for fi in self.settings.get_fields())
        for ni in pnames:
            if type(ni) in [list, tuple]:
                if len(ni) == 4 and fields[ni[1]].type is bool:
                    # the entry is toggling type
                    e1_active = fields[ni[1]]
                    e1_field = fields[ni[2]]
                    e2_field = fields[ni[3]]
                    gbox = wx.StaticBox(box, -1, ni[0])
                    gbox_sizer = wx.StaticBoxSizer(gbox, orient=wx.VERTICAL)
                    box_sizer.Add(gbox_sizer, 0, wx.EXPAND)
                    self.rb_bool_fields[gbox.GetId()] = ni[1]
                    e1 = self.get_toggle_entry(e1_field, gbox, getattr(self.settings, e1_active.name, True))
                    gbox_sizer.Add(e1, 1, wx.EXPAND | wx.ALL, 2)
                    e2 = self.get_toggle_entry(e2_field, gbox, not getattr(self.settings, e1_active.name, True))
                    gbox_sizer.Add(e2, 1, wx.EXPAND | wx.BOTTOM, 5)
                    gbox.Bind(wx.EVT_RADIOBUTTON, self.on_radio)
                else:
                    hsizer = wx.BoxSizer(wx.HORIZONTAL)
                    box_sizer.Add(hsizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
                    for nj in ni:
                        field = fields[nj]
                        entry = self.get_entry(field, box)
                        hsizer.Add(entry, 0, wx.EXPAND | wx.ALL, 2)
            else:
                field = fields[ni]
                entry = self.get_entry(field, box)
                box_sizer.Add(entry, 1, wx.FIXED_MINSIZE | wx.ALIGN_CENTRE | wx.ALL, 5)
        box_sizer.Fit(box)

    def get_entry(self, field, parent, add_label=True):
        gmeta = field.metadata.get("genx", {})
        if "label" in gmeta:
            name = gmeta["label"]
        else:
            name = field.name.replace("_", " ")
        if field.type is bool:
            entry = wx.CheckBox(parent, -1, name, style=wx.ALIGN_CENTER)
            entry.SetValue(getattr(self.settings, field.name, False))
            self.value_entries[field.name] = entry
        elif "selection" in gmeta:
            choice = wx.Choice(parent, -1, choices=gmeta["selection"])
            choice.SetSelection(gmeta["selection"].index(getattr(self.settings, field.name, gmeta["selection"][0])))
            self.value_entries[field.name] = choice
            entry = self.label_entry(parent, choice, name)
        elif "pmin" in gmeta and "pmax" in gmeta:
            # a numerical parameter with given range
            if field.type is int:
                spin = wx.SpinCtrl(parent, -1, min=gmeta["pmin"], max=gmeta["pmax"])
            else:
                inc = 10 ** (round(log10(gmeta["pmax"])) - 2)
                spin = wx.SpinCtrlDouble(parent, -1, min=gmeta["pmin"], max=gmeta["pmax"], inc=inc)
                spin.SetDigits(4)
            spin.SetValue(getattr(self.settings, field.name, gmeta["pmin"]))

            spin.SetMinSize(spin.GetSizeFromTextSize(spin.GetTextExtent("%.4f" % gmeta["pmax"] + " ")))
            self.value_entries[field.name] = spin
            if add_label:
                entry = self.label_entry(parent, spin, name)
            else:
                entry = spin
        else:
            txt = wx.TextCtrl(parent, -1, style=0)
            txt.SetValue(str(getattr(self.settings, field.name, "")))
            txt.SetMinSize(txt.GetSizeFromTextSize(txt.GetTextExtent(10 * " ")))
            self.value_entries[field.name] = txt
            if add_label:
                entry = self.label_entry(parent, txt, name)
            else:
                entry = txt
        return entry

    def get_toggle_entry(self, field, parent, active):
        gmeta = field.metadata.get("genx", {})
        if "label" in gmeta:
            name = gmeta["label"]
        else:
            name = field.name.replace("_", " ")
        # generate an entries with a radio button to toggle it active
        entry = wx.BoxSizer(wx.HORIZONTAL)
        entry_active = wx.RadioButton(parent, -1, name, style=wx.ALIGN_RIGHT)
        entry_active.SetValue(active)
        entry.Add(entry_active, 1, wx.FIXED_MINSIZE | wx.RIGHT, 8)
        field_entry = self.get_entry(field, parent, add_label=False)
        field_entry.Enable(active)
        parent_id = parent.GetId()
        if parent_id in self.rb_actions:
            self.rb_actions[parent_id][entry_active.GetId()] = field_entry
        else:
            self.rb_actions[parent_id] = {entry_active.GetId(): field_entry}
        entry.Add(field_entry, 1, wx.FIXED_MINSIZE, 0)
        return entry

    def on_radio(self, event: wx.CommandEvent):
        # activates the entry next to the selected radio button and deactivates all others in that group
        parent_id = event.GetEventObject().GetParent().GetId()
        sender_id = event.GetId()
        for id, field_entry in self.rb_actions[parent_id].items():
            field_entry.Enable(sender_id == id)
        event.Skip()

    def label_entry(self, parent, entry, label):
        out_entry = wx.BoxSizer(wx.HORIZONTAL)
        out_entry.Add(
            wx.StaticText(parent, -1, label, style=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL),
            0,
            wx.FIXED_MINSIZE | wx.ALIGN_CENTER_VERTICAL | wx.RIGHT,
            8,
        )
        out_entry.Add(entry, 0, wx.EXPAND, 0)
        return out_entry

    def collect_results(self):
        # create a dictionary with all values from the dialog entry
        out = {}
        fields = dict((fi.name, fi) for fi in self.settings.get_fields())
        for key, entry in self.value_entries.items():
            field = fields[key]
            if hasattr(entry, "GetValue"):
                try:
                    out[key] = field.type(entry.GetValue())
                except ValueError:
                    # TODO: add handling of entry errors
                    pass
            elif hasattr(entry, "GetString") and hasattr(entry, "GetSelection"):
                out[key] = entry.GetString(entry.GetSelection())
            else:
                raise NotImplementedError(f"Could not evaluate wx Control {entry!r}")
        for pid, rbg in self.rb_actions.items():
            # get the active state of the boolean fields for radio buttons
            out[self.rb_bool_fields[pid]] = list(rbg.values())[0].IsEnabled()
        return out

    def on_apply_changes(self, event):
        event.Skip()
        options = self.collect_results()
        OK_pressed = event.GetId() == wx.ID_OK
        if self.apply_callback(options) and OK_pressed:
            for key, value in options.items():
                setattr(self.settings, key, value)
