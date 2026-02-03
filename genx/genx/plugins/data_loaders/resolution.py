"""
==========================================================
:mod:`resolution` Data loader that includes the resoultion
==========================================================

Loads files in a four column format where the fourth column contains the resolution of the experiment.
Note that all reflectivity modules uses the standard deviation as a the resolution. Some instruments might
use the FWHM instead.

The default columns are the following:

* First column q values
* Second column Intensitiy values
* Third values The uncertainty in the Intensities
* Fourth column resolution

The other settings are just as in the default data loader.

The resolution is stored as the member variable res. For data set 0 it can accessed as
data[0].res
"""

import numpy as np

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog

class Plugin(Template):
    def __init__(self, parent):
        Template.__init__(self, parent)
        self.q_col = 0
        self.I_col = 1
        self.eI_col = 2
        self.res_col = 3
        self.comment = "#"
        self.skip_rows = 0
        self.delimiter = None

    def LoadData(self, dataset, filename, data_id=0):
        """LoadData(self, dataset, filename) --> none

        Loads the data from filename into the data_item_number.
        """
        try:
            with open(filename, encoding="utf-8", errors="ignore") as fh:
                load_array = np.loadtxt(fh, delimiter=self.delimiter, comments=self.comment, skiprows=self.skip_rows)
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
            if load_array.shape[1] - 1 < max(self.q_col, self.res_col, self.I_col, self.eI_col):
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

            # print load_array
            dataset.x_raw = load_array[:, self.q_col]
            dataset.y_raw = load_array[:, self.I_col]
            dataset.error_raw = load_array[:, self.eI_col]
            dataset.set_extra_data("res", load_array[:, self.res_col], "res")
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta["data_source"]["facility"] = "GenX resolution data loader"
            dataset.meta["data_source"]["experiment"]["instrument"] = "unkown"
            dataset.meta["data_source"]["import_columns"] = dict(
                x=self.q_col, y=self.I_col, error=self.eI_col, res=self.res_col
            )

    def SettingsDialog(self):
        """SettingsDialog(self) --> None

        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        """
        from genx.gui_generic.add_ons.help_modules.settings_dialog import SettingsDialog
        col_values = {"I": self.I_col, "q": self.q_col, "Resolution": self.res_col, "I error": self.eI_col}
        misc_values = {"Comment": str(self.comment), "Skip rows": self.skip_rows, "Delimiter": str(self.delimiter)}
        dlg = SettingsDialog(self.parent, col_values, misc_values)
        success, col_values, misc_values = dlg.get_results()
        if success:
            self.q_col = col_values["q"]
            self.res_col = col_values["Resolution"]
            self.I_col = col_values["I"]
            self.eI_col = col_values["I error"]
            self.comment = misc_values["Comment"]
            self.skip_rows = misc_values["Skip rows"]
            self.delimiter = misc_values["Delimiter"]
            self.SetStatusText("Changed import settings")
        else:
            self.SetStatusText("No change to import settings")
