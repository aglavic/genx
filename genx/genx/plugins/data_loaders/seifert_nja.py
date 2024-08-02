"""
===================================
:mod:`seifert_nja` NJA ASCII format
===================================

Loads the data from *.nja ASCII files with limited header analysis.
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
    wildcard = "*.nja"

    def __init__(self, parent):
        Template.__init__(self, parent)

    def CanOpen(self, file_path):
        if not Template.CanOpen(self, file_path):
            return False
        with open(file_path, "r", encoding="utf-8") as fh:
            l1 = fh.readline()
            l2 = fh.readline()
        return l2.startswith("#ScanTableParameter")

    def evaluate_header(self, header):
        output = {}
        for line in header:
            if not line.startswith("&"):
                continue
            items = line.split("&")
            if items[1].startswith("ScanAxis"):
                output["ScanAxis"] = items[1].split("=")[1].strip()
            elif items[1].startswith("Axis"):
                output[items[1]] = dict([tuple(ei.strip().split("=", 2)) for ei in items[1:]])
            else:
                for ei in items[1:]:
                    output[ei.split("=")[0]] = ei.split("=", 2)[-1].strip()

        return output

    def LoadData(self, dataset, filename, data_id=0):
        """LoadData(self, dataset, filename) --> none

        Loads the data from filename into the dataset object.
        """
        try:
            with open(filename, encoding="utf-8", errors="ignore") as fh:
                header = [fh.readline()]
                while not header[-1].startswith("&NoValues="):
                    header.append(fh.readline())
                load_array = np.loadtxt(fh, delimiter=None)
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
            if load_array.shape[1] < 2:
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
            dataset.x_raw = load_array[:, 0]
            dataset.y_raw = load_array[:, 1]

            header = self.evaluate_header(header)
            if header.get("ScanAxis", "") == "O":
                dataset.x_command = "2*x"
            else:
                dataset.x_command = "x"

            # Check if we have errors in the data - if not handle it with nan's
            dataset.error_raw = np.sqrt(load_array[:, 1])

            # Run the commands on the data - this also sets the x,y, error members of the data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta["data_source"]["facility"] = "Seifert nja loader"
            dataset.meta["data_source"]["experiment"]["probe"] = "xray"
            dataset.meta["data_source"]["measurement"]["scheme"] = "angle-dispersive"
            # header evaluations
            inst = dataset.meta["data_source"]["measurement"]["instrument_settings"]
            inst["incident_angle"] = {
                "min": float(dataset.x_raw.min()),
                "max": float(dataset.x_raw.max()),
                "unit": "degree",
            }
            inst["wavelength"] = {"magnitude": 1.54, "unit": "angstrom"}
            # insert all metadata into user field
            dataset.meta["nja_header"] = header
