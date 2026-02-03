"""
==================================
:mod:`default` Default data loader
==================================

Loads the data from whitespace seperated column formatted ascii data files.
The module allows the specification of which columns that correspond to the
x, y and y_error columns in the data file. 
If y_error is not used the it can safely be set to the same column as y.

Which columns that are imported are determined from the dialog box in 
import settings. Note that these settings apply to the marked data set(s).
Other possible tunings are the definitions of delimiter, None means any 
white space character (default). Skip rows is how many rows are skipped before
the file is started to be read. Comment is the first character of a commented
line.
"""

import numpy as np

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog

class Plugin(Template):
    def __init__(self, parent):
        Template.__init__(self, parent)
        self.x_col = 0
        self.y_col = 1
        self.e_col = 2
        self.comment = "#"
        self.skip_rows = 0
        self.delimiter = None

    def LoadData(self, dataset, filename, data_id=0):
        """LoadData(self, dataset, filename) --> none

        Loads the data from filename into the dataset object.
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
            if load_array.shape[1] - 1 < max(self.x_col, self.y_col):
                ShowWarningDialog(
                    self.parent,
                    "The data file does not contain enough number of columns. It has "
                    + str(load_array.shape[1])
                    + " columns. Rember that the column index start at zero!",
                )
                self.SetStatusText("Could not load data - not enough columns")
                return

            # The data is set by the default Template.__init__ function
            # Know the loaded data goes into *_raw so that they are not
            # changed by the transforms
            dataset.x_raw = load_array[:, self.x_col]
            dataset.y_raw = load_array[:, self.y_col]

            # Check if we have errors in the data - if not handle it with nan's
            if load_array.shape[1] - 1 < self.e_col:
                dataset.error_raw = load_array[:, self.y_col] * np.nan
                self.SetStatusText("Could not load error column - setting it to nan")
            else:
                dataset.error_raw = load_array[:, self.e_col]

            # Run the commands on the data - this also sets the x,y, error members of the data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta["data_source"]["facility"] = "GenX default data loader"
            dataset.meta["data_source"]["experiment"]["instrument"] = "unknown"
            dataset.meta["data_source"]["import_columns"] = dict(x=self.x_col, y=self.y_col, error=self.e_col)

    def SettingsDialog(self):
        """SettingsDialog(self) --> None

        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        """
        from genx.gui_generic.add_ons.help_modules.settings_dialog import SettingsDialog
        col_values = {"y": self.y_col, "x": self.x_col, "y error": self.e_col}
        misc_values = {"Comment": str(self.comment), "Skip rows": self.skip_rows, "Delimiter": str(self.delimiter)}
        dlg = SettingsDialog(self.parent, col_values, misc_values)
        success, col_values, misc_values = dlg.get_results()
        if success:
            self.y_col = col_values["y"]
            self.x_col = col_values["x"]
            self.e_col = col_values["y error"]
            self.comment = misc_values["Comment"]
            self.skip_rows = misc_values["Skip rows"]
            self.delimiter = misc_values["Delimiter"]
            self.SetStatusText("Changed import settings")
        else:
            self.SetStatusText("No change to import settings")
