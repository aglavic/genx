"""
===========================================
:mod:`bruker`  Bruker format (.brml) reader
===========================================

Loads the data exported by e.g. Bruker D8 instruments.
"""

import io
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog


class Plugin(Template):
    wildcard = "*.brml"

    def __init__(self, parent):
        Template.__init__(self, parent)

    def get_datasets(self, zip_ref):
        # parse the DataContainer.xml file to get a list of all measurements.
        data_container = ET.fromstring(zip_ref.open("Experiment0/DataContainer.xml", "r").read())
        return [ds.text for ds in data_container.find("RawDataReferenceList").iter("string")]

    def CountDatasets(self, file_path):
        try:
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                return len(self.get_datasets(zip_ref))
        except Exception as e:
            return 1

    def get_columns(self, raw_data_views: ET.Element):
        columns = []
        for raw_data_view in raw_data_views:

            if "LogicName" in raw_data_view.attrib and raw_data_view.attrib["Length"] == "1":
                columns.append(raw_data_view.attrib["LogicName"])
            elif raw_data_view.attrib["Length"] == "1":
                if raw_data_view[0].attrib.get("Category", None) == "Count":
                    columns.append("Count")
                else:
                    columns.append(raw_data_view[0].attrib["LogicName"])
            else:
                for fd in raw_data_view.iter("FieldDefinitions"):
                    if "AxisId" in fd.attrib:
                        columns.append(fd.attrib["AxisId"])
                    else:
                        columns.append(fd.attrib["FieldName"])
        return columns

    def LoadData(self, dataset, filename, data_id=0):
        """
        Loads the data from filename into the data_item_number.
        """
        try:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                # get a list of datasets in this file
                datasets = self.get_datasets(zip_ref)
                data_to_load = datasets[data_id]
                data_xml = ET.fromstring(zip_ref.open(data_to_load, "r").read())
            data_lines = [dt.text for dt in data_xml.find("DataRoutes").find("DataRoute").iter("Datum")]
            load_array = np.loadtxt(io.StringIO("\n".join(data_lines)), delimiter=",")
            columns = self.get_columns(data_xml.find("DataRoutes").find("DataRoute").find("DataViews"))
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
            try:
                x_idx = columns.index("Theta")
                tth_only = False
            except ValueError:
                x_idx = columns.index("TwoTheta")
                tth_only = True
            y_idx = columns.index("Count")
            try:
                att_idx = columns.index("AbsorptionFactor")
                atten = 1.0 / load_array[:, att_idx]
            except ValueError:
                att_idx = None
                atten = 1.0

            dataset.x_raw = load_array[:, x_idx]
            dataset.y_raw = load_array[:, y_idx] * atten
            dataset.error_raw = np.sqrt(load_array[:, y_idx]) * atten
            if not tth_only:
                dataset.x_command = "2*x"
            else:
                dataset.x_command = "x"
            if att_idx is not None:
                # measurement reports attenuators
                dataset.set_extra_data("atten", atten)
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            try:
                info_data = list(data_xml.find("FixedInformation").find("MethodInformation").iter("InfoData"))
            except AttributeError:
                info_data = []
                info_list = []
            for id in info_data:
                try:
                    wavelength = float(id.find("MethodAlignment").find("WaveLength").attrib["Value"])
                except (AttributeError, ValueError):
                    wavelength = 1.54
                try:
                    info_list = list(id.find("InfoList").iter())
                except (AttributeError, ValueError):
                    pass
            for il in info_list:
                if il.attrib.get("Name", "") == "SampleName":
                    dataset.meta["data_source"]["sample"] = {"name": il.attrib["Value"]}
                if il.attrib.get("Name", "") == "User":
                    dataset.meta["data_source"]["owner"]["name"] = il.attrib["Value"]
            dataset.meta["data_source"]["experiment"]["instrument"] = "Bruker"
            dataset.meta["data_source"]["experiment"]["probe"] = "xray"
            try:
                dataset.meta["data_source"]["experiment"]["start_date"] = data_xml.find("TimeStampStarted").text
            except AttributeError:
                pass
            dataset.meta["data_source"]["measurement"]["scheme"] = "angle-dispersive"
            inst = dataset.meta["data_source"]["measurement"]["instrument_settings"]
            inst["incident_angle"] = {
                "min": float(dataset.x_raw.min()),
                "max": float(dataset.x_raw.max()),
                "unit": "deg",
            }
            inst["wavelength"] = {
                "magnitude": wavelength,
                "unit": "angstrom",
            }
