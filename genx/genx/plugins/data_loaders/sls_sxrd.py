"""
====================================
:mod:`sls_sxrd` SLD sxrd data loader
====================================

Loads the data from whitespace seperated column formatted ascii data files.
It is intended for surface x-ray diffraction data where the data sets consists
of rod scans along the l-direction (perpendicular to the surface). The plugin
sorts each rod with equal h and k values into one data sets. The l-direction 
is also sorted.

The default columns are the following:

* First column h values
* Second column k values
* Third values l values
* Fourth column Intensites
* Fifth column The standard deviation of the intensities.

The other settings are just as in the default data loader.

The h,k values is stored as extra data in data.extra_data dictonary as
h and k. 
"""

import numpy as np

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog

try:
    import wx
except ImportError:
    # in case of console usinge withou wx beeing installed put a mock class/module
    class void:
        pass

    wx = void()
    wx.Dialog = void


class Plugin(Template):
    def __init__(self, parent):
        Template.__init__(self, parent)
        self.h_col = 0
        self.k_col = 1
        self.l_col = 2
        self.I_col = 3
        self.eI_col = 4
        self.LB_col = 5
        self.dL_col = 6
        self.comment = "#"
        self.skip_rows = 0
        self.delimiter = None

    def LoadData(self, dataset, filename, data_id=0):
        """LoadData(self, dataset, filename) --> none

        Loads the data from filename into the dataset.
        """
        try:
            load_array = np.loadtxt(
                filename, delimiter=self.delimiter, encoding="utf8", comments=self.comment, skiprows=self.skip_rows
            )
        except Exception as e:
            ShowWarningDialog(
                self.parent,
                "Could not load the file: "
                + filename
                + " \nPlease check the format.\n\n numpy.loadtxt"
                + " gave the following error:\n"
                + str(e),
            )
        else:
            # For the freak case of only one data point
            if len(load_array.shape) < 2:
                load_array = np.array([load_array])
            # Check so we have enough columns
            if load_array.shape[1] - 1 < max(self.h_col, self.k_col, self.l_col, self.I_col, self.eI_col):
                ShowWarningDialog(
                    self.parent,
                    "The data file does not contain"
                    + "enough number of columns. It has "
                    + str(load_array.shape[1])
                    + " columns. Rember that the column index start at zero!",
                )
                # Okay now we have showed a dialog lets bail out ...
                return
            # The data is set by the default Template.__init__ function, neat hu
            # Note that the loaded data goes into *_raw so that they are not
            # changed by the transforms

            # Create an record array so we can sort the data properly
            if load_array.shape[1] - 1 >= max(
                self.h_col, self.k_col, self.l_col, self.I_col, self.eI_col, self.LB_col, self.dL_col
            ):
                # dataset with LB and dL columns
                data = np.rec.fromarrays(
                    [
                        load_array[:, self.h_col].round().astype(type(1)),
                        load_array[:, self.k_col].round().astype(type(1)),
                        load_array[:, self.l_col],
                        load_array[:, self.I_col],
                        load_array[:, self.eI_col],
                        load_array[:, self.LB_col],
                        load_array[:, self.dL_col],
                    ],
                    names="h, k, l, I, eI, LB, dL",
                )
                has_LB = True
            else:
                data = np.rec.fromarrays(
                    [
                        load_array[:, self.h_col].round().astype(type(1)),
                        load_array[:, self.k_col].round().astype(type(1)),
                        load_array[:, self.l_col],
                        load_array[:, self.I_col],
                        load_array[:, self.eI_col],
                    ],
                    names="h, k, l, I, eI",
                )
                has_LB = False
            # Sort the data
            data.sort(order=("h", "k", "l"))
            i = 0
            while i < len(data):
                # Find all the data for each rod
                tmp = data.compress(np.bitwise_and(data["h"] == data[i]["h"], data["k"] == data[i]["k"]))
                self.data.add_new("(%i, %i)" % (tmp["h"][0], tmp["k"][0]))
                self.data[-1].x_raw = tmp["l"]
                self.data[-1].y_raw = tmp["I"]
                self.data[-1].error_raw = tmp["eI"]
                # Run the commands on the data - this also sets the x,y, error memebers
                # of that data item.
                self.data[-1].run_command()
                self.data[-1].set_extra_data("h", tmp["h"], "h")
                self.data[-1].set_extra_data("k", tmp["k"], "k")
                if has_LB:
                    self.data[-1].set_extra_data("LB", tmp["LB"], "LB")
                    self.data[-1].set_extra_data("dL", tmp["dL"], "dL")
                # Increase the index
                i += len(tmp)

            # Update the data list
            self.UpdateDataList()
            # Send an update that new data has been loaded
            self.SendUpdateDataEvent()

    def SettingsDialog(self):
        """SettingsDialog(self) --> None

        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        """
        col_values = {
            "I": self.I_col,
            "h": self.h_col,
            "k": self.k_col,
            "l": self.l_col,
            "I error": self.eI_col,
            "LB": self.LB_col,
            "dL": self.dL_col,
        }
        misc_values = {"Comment": str(self.comment), "Skip rows": self.skip_rows, "Delimiter": str(self.delimiter)}
        dlg = SettingsDialog(self.parent, col_values, misc_values)
        if dlg.ShowModal() == wx.ID_OK:
            col_values = dlg.GetColumnValues()
            misc_values = dlg.GetMiscValues()
            self.h_col = col_values["h"]
            self.k_col = col_values["k"]
            self.l_col = col_values["l"]
            self.I_col = col_values["I"]
            self.eI_col = col_values["I error"]
            self.comment = misc_values["Comment"]
            self.skip_rows = misc_values["Skip rows"]
            self.delimiter = misc_values["Delimiter"]
            self.SetStatusText("Changed import settings")
        else:
            self.SetStatusText("No change to import settings")
        dlg.Destroy()


class SettingsDialog(wx.Dialog):

    def __init__(self, parent, col_values, misc_values):
        wx.Dialog.__init__(self, parent, -1, "Data loader settings")

        box_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Make the box for putting in the columns
        col_box = wx.StaticBox(self, -1, "Columns")
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL)

        # col_values = {'y': 1,'x': 0,'y error': 1}
        col_grid = wx.GridBagSizer(len(col_values), 2)
        self.col_controls = col_values.copy()
        keys = list(col_values.keys())
        keys.sort()
        for i, name in enumerate(keys):
            text = wx.StaticText(self, -1, name + ": ")
            control = wx.SpinCtrl(self)
            control.SetRange(0, 100)
            control.SetValue(col_values[name])
            col_grid.Add(text, (i, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
            col_grid.Add(control, (i, 1), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
            self.col_controls[name] = control

        col_box_sizer.Add(col_grid, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        box_sizer.Add(col_box_sizer, 0, wx.ALL | wx.EXPAND, 5)

        col_box = wx.StaticBox(self, -1, "Misc")
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL)

        # Lets add another box for comments and rows to skip
        # misc_values = {'Comment': '#', 'Skip rows': 0,'Delimiter': 'None'}
        col_grid = wx.GridBagSizer(len(misc_values), 2)
        self.misc_controls = misc_values.copy()
        keys = list(misc_values.keys())
        keys.sort()
        for i, name in enumerate(keys):
            text = wx.StaticText(self, -1, name + ": ")
            if type(misc_values[name]) == type(1):
                control = wx.SpinCtrl(self)
                control.SetRange(0, 100)
                control.SetValue(misc_values[name])
            else:
                control = wx.TextCtrl(self, value=misc_values[name], style=wx.EXPAND)
            col_grid.Add(text, (i, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
            col_grid.Add(control, (i, 1), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
            self.misc_controls[name] = control

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

    def GetMiscValues(self):
        values = {}
        for key in self.misc_controls:
            val = self.misc_controls[key].GetValue()
            if type(val) == type("") or type(val) == type(""):
                if val.lower() == "none":
                    val = None
            values[key] = val
        return values
