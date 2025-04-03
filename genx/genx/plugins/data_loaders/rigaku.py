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


class MeasurementCondition:
    operator = "unknown"
    system = "Unknown Rigaku"
    RASHeader = {}
    Axes = {}
    axis = "TwoThetaOmega"
    mode = "unknown"
    comment = ""
    memo = ""
    sample = ""
    file_type = "RAS_RAW"
    generator = {}
    detector = {}
    optics = {}
    auto_atten = False
    scan_unit = "deg"
    start_time = None

    # used to parse and represent the MeasurementCondition0.xml file content
    def __init__(self, root):
        for element in root:
            if element.tag == "RASHeader":
                self.parse_ras(element)
            elif element.tag == "GeneralInformation":
                self.parse_GI(element)
            elif element.tag == "ScanInformation":
                self.parse_scan(element)
            elif element.tag == "Axes":
                self.parse_axes(element)
            elif element.tag == "HWConfigurations":
                self.parse_hw(element)

    def parse_ras(self, element):
        RASHeader = {}
        for child in element:
            try:
                key, value = self.parse_pair(child)
            except ValueError:
                continue
            else:
                RASHeader[key] = value
        self.RASHeader = RASHeader

    def parse_GI(self, element):
        for child in element:
            if child.tag == "Operator":
                self.operator = child.text
            elif child.tag == "SystemName":
                self.system = child.text
            elif child.tag == "Comment":
                self.comment = child.text or ""
            elif child.tag == "Memo":
                self.memo = child.text or ""
            elif child.tag == "SampleName":
                self.sample = child.text

    def parse_scan(self, element):
        for child in element:
            if child.tag == "AxisName":
                self.axis = child.text
            elif child.tag == "Mode":
                self.mode = child.text
            elif child.tag == "AttenuatorAutoMode":
                self.auto_atten = child.text.strip().lower() == "true"
            elif child.tag == "PositionUnit":
                self.scan_unit = child.text
            elif child.tag == "StartTime":
                self.start_time = child.text

    def parse_axes(self, element):
        Axes = {}
        for child in element:
            try:
                key, value = self.parse_axis(child)
            except ValueError:
                continue
            else:
                Axes[key] = value
        self.Axes = Axes

    def parse_hw(self, element):
        for child in element:
            if child.tag == "XrayGenerator":
                generator = {}
                for sc in child:
                    generator[sc.tag] = sc.text
                self.generator = generator
            elif child.tag == "Detector":
                detector = {}
                for sc in child:
                    detector[sc.tag] = sc.text
                self.detector = detector
            elif child.tag == "Optics":
                optics = {}
                for sc in child:
                    optics[sc.tag] = sc.text
                self.optics = optics

    def parse_pair(self, element):
        if element.tag != "Pair":
            raise ValueError
        return element[0].text.lstrip("*"), element[1].text

    def parse_axis(self, element):
        if element.tag != "Axis":
            raise ValueError
        return element.attrib["Name"], dict(element.attrib)

    def to_dict(self):
        output = {}
        output["Generator"] = self.generator
        output["Detector"] = self.detector
        output["Optics"] = self.optics
        output["Axes"] = self.Axes
        output["RASHeader"] = self.RASHeader
        return output


class Plugin(Template):
    wildcard = "*.ras;*.rasx"

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
        if filename.endswith(".rasx"):
            return self.LoadRasX(dataset, filename, data_id)
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
            dataset.meta["data_source"]["experiment"]["probe"] = "xray"
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

    def LoadRasX(self, dataset, filename, data_id):
        import io
        import xml.etree.ElementTree as ET
        import zipfile

        try:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                profile_txt = zip_ref.open("Data0/Profile0.txt", "r").read()
                header_xml = zip_ref.open("Data0/MesurementConditions0.xml", "r").read()
            load_array = np.loadtxt(io.StringIO(profile_txt.decode("utf-8-sig")))
            info = MeasurementCondition(ET.fromstring(header_xml))
        except Exception as e:
            ShowWarningDialog(
                self.parent,
                "Could not load the file: "
                + filename
                + " \nPlease check the format.\n\n numpy.loadtxt / header analysis "
                + " gave the following error:\n"
                + str(e),
            )
        else:
            dataset.x_raw = load_array[:, 0]
            dataset.y_raw = load_array[:, 1]
            dataset.error_raw = np.sqrt(load_array[:, 1])
            if info.axis == "OmegaTwoTheta":
                dataset.x_command = "2*x"
            else:
                dataset.x_command = "x"
            if load_array.shape[1] > 2 and info.auto_atten:
                # measurement uses attenuators
                dataset.set_extra_data("atten", load_array[:, 2])
                dataset.y_command = "y*atten"
                dataset.error_command = "e*atten"
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta["data_source"]["owner"]["name"] = info.operator
            dataset.meta["data_source"]["experiment"]["instrument"] = info.system
            dataset.meta["data_source"]["experiment"]["probe"] = "x-ray"
            dataset.meta["data_source"]["experiment"]["title"] = info.memo + " - " + info.comment
            dataset.meta["data_source"]["experiment"]["start_date"] = info.start_time
            dataset.meta["data_source"]["measurement"]["scheme"] = "angle-dispersive"
            inst = dataset.meta["data_source"]["measurement"]["instrument_settings"]
            inst["incident_angle"] = {
                "min": float(dataset.x_raw.min()),
                "max": float(dataset.x_raw.max()),
                "unit": info.scan_unit,
            }
            inst["wavelength"] = {
                "magnitude": float(info.generator.get("WavelengthKalpha1", 1.540593)),
                "unit": "angstrom",
            }
            dataset.meta["data_source"]["sample"] = {"name": info.sample}
            # insert all metadata into user field
            dataset.meta["rasx_header"] = info.to_dict()
