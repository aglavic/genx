"""
======================================================
:mod:`rigaku`  Rigaku ASCII file from Smartlab (*.ras)
======================================================

Loads the data exported by Rigaku Smartlab.
"""

import numpy as np

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog

try:
    import wx

    from wx.lib.masked import NumCtrl
except ImportError:

    class void:
        pass

    wx = void()
    wx.Dialog = void


class Plugin(Template):
    wildcard = "*.ras"

    def __init__(self, parent):
        Template.__init__(self, parent)
        self.x_col = 0
        self.y_col = 1
        self.e_col = 1
        self.xe_col = -1
        self.comment = "*"
        self.skip_rows = 0
        self.delimiter = None

    def ReadHeader(self, fh):
        header = {}
        hl = fh.readline()
        while hl.startswith("*"):
            if hl.startswith("*RAS_INT_START"):
                break
            if '"' in hl:
                keys, value = hl.split('"', 1)
                value = value.split('"')[0]
            else:
                keys = hl
                value = None
            hl = fh.readline()

            if value and "|" in value:
                value = value.split("|", 1)[-1]
            try:
                value = float(value)
            except (ValueError, TypeError):
                pass
            keys = keys[1:].lower().strip().split("_")
            if keys[0] not in header:
                header[keys[0]] = {}
            subhdr = header[keys.pop(0)]
            while len(keys) > 1:
                if keys[0] not in subhdr:
                    subhdr[keys[0]] = {}
                elif not isinstance(subhdr[keys[0]], dict):
                    subhdr[keys[0]] = {"_value": subhdr[keys[0]]}
                subhdr = subhdr[keys.pop(0)]
            subhdr[keys.pop(0)] = value
        del header["ras"]
        return header

    def get_from_header(self, header, keys):
        value = header.get(keys[0], None)
        if value is None or len(keys) == 1:
            return value
        else:
            if isinstance(value, dict):
                return self.get_from_header(value, keys[1:])
            else:
                return None

    def LoadData(self, dataset, filename, data_id=0):
        """
        Loads the data from filename into the data_item_number.
        """
        try:
            with open(filename, encoding="utf-8", errors="ignore") as fh:
                header = self.ReadHeader(fh)
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
            dataset.x_raw = load_array[:, self.x_col]
            dataset.y_raw = load_array[:, self.y_col]
            dataset.error_raw = np.sqrt(load_array[:, self.y_col])
            if self.get_from_header(header, ["meas", "scan", "axis", "x", "_value"]) == "Omega/2-Theta":
                dataset.x_command = "2*x"
            else:
                dataset.x_command = "x"
            if load_array.shape[1] > 2 and self.get_from_header(header, ["hw", "r", "attenuater", "automode"]) == 1:
                # measurement uses attenuators
                dataset.set_extra_data("atten", load_array[:, 2])
                dataset.y_command = "y*atten"
                dataset.error_command = "e*atten"
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta["data_source"]["experiment"]["instrument"] = "RAS"
            dataset.meta["data_source"]["experiment"]["probe"] = "x-ray"
            dataset.meta["data_source"]["measurement"]["scheme"] = "angle-dispersive"
            inst = dataset.meta["data_source"]["measurement"]["instrument_settings"]
            inst["incident_angle"] = {
                "min": float(dataset.x_raw.min()),
                "max": float(dataset.x_raw.max()),
                "unit": self.get_from_header(header, ["meas", "scan", "unit", "x"]),
            }
            inst["wavelength"] = {
                "magnitude": self.get_from_header(header, ["hw", "xg", "wave", "length", "alpha1"]),
                "unit": "angstrom",
            }
            dataset.meta["data_source"]["sample"] = {"name": self.get_from_header(header, ["file", "sample"])}
            # insert all metadata into user field
            dataset.meta["ras_header"] = header
