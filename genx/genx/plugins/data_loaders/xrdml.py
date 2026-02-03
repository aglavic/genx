"""
===============================
:mod:`xrdml`  XRDML data loader
===============================

Loads the data from Philips XPert instrument.
"""

from xml.dom.minidom import parseString

import numpy as np

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog

class Plugin(Template):
    wildcard = "*.xrdml"

    def __init__(self, parent):
        Template.__init__(self, parent)

    def CountDatasets(self, file_path):
        try:
            orso_datasets = ReadXpert(file_path)
        except Exception as e:
            return 1
        else:
            return len(orso_datasets)

    def LoadData(self, dataset, filename, data_id=0):
        """
        Loads the data from filename into the data_item_number.
        """
        try:
            datasets = ReadXpert(filename)
        except Exception as e:
            import traceback

            ShowWarningDialog(
                self.parent,
                "Could not load the file: "
                + filename
                + " \nPlease check the format.\n\n Error in ReadXpert:\n"
                + traceback.format_exc(),
            )
        else:
            ds = datasets[data_id]
            dataset.x_raw = ds[0]
            dataset.y_raw = ds[1]
            dataset.error_raw = ds[2]
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta["data_source"]["experiment"]["instrument"] = "XRDML"
            dataset.meta["data_source"]["experiment"]["probe"] = "x-ray"
            dataset.meta["data_source"]["measurement"]["scheme"] = "angle-dispersive"
            dataset.meta["data_source"]["sample"]["name"] = ds[3]
            dataset.meta["data_source"]["experiment"].update(ds[4])
            dataset.meta["data_source"]["measurement"].update(ds[5])


def ReadXpert(file_name):
    """
    Read the data of a philips X'Pert diffractometer file, exported as text files.
    """
    raw_data = open(file_name, "r").read()
    while len(raw_data) > 0 and raw_data[0] != "<":
        # some files are written using UTF-8 BOM format that has extra bytes before the text starts
        # this will ignore any characters at the beginning that are not "<", the tag opening for XML
        raw_data = raw_data[1:]
    xml_data = parseString(raw_data).firstChild

    # retrieve data
    try:
        sample_name = xml_data.getElementsByTagName("sample")[0].getElementsByTagName("name")[0].firstChild.nodeValue
    except AttributeError:
        sample_name = file_name.rsplit(".", 1)[0]
    datasets = []

    try:
        lamda = float(xml_data.getElementsByTagName("kAlpha1")[0].firstChild.nodeValue)
    except (IndexError, ValueError):
        lamda = 1.54  # Cu k-alpha

    for xml_scan in xml_data.getElementsByTagName("xrdMeasurement")[0].getElementsByTagName("scan"):
        meta_experiment = {}
        meta_measurement = {}
        scan = xml_scan.getElementsByTagName("dataPoints")[0]

        moving_positions = {}
        for motor in scan.getElementsByTagName("positions"):
            axis = motor.attributes["axis"].value
            if len(motor.getElementsByTagName("commonPosition")) == 0:
                start = float(motor.getElementsByTagName("startPosition")[0].firstChild.nodeValue)
                end = float(motor.getElementsByTagName("endPosition")[0].firstChild.nodeValue)
                moving_positions[axis] = (start, end)

        try:
            atten_factors = scan.getElementsByTagName("beamAttenuationFactors")[0].firstChild.nodeValue
            atten_factors = list(map(float, atten_factors.split()))
            atten = np.array(atten_factors)
        except IndexError:
            atten = 1.0
        try:
            meta_experiment["start_date"] = xml_scan.getElementsByTagName("startTimeStamp")[0].firstChild.nodeValue
        except IndexError:
            pass

        time = float(scan.getElementsByTagName("commonCountingTime")[0].firstChild.nodeValue)
        data_tags = scan.getElementsByTagName("intensities") + scan.getElementsByTagName("counts")
        data_tag = data_tags[0]
        data = list(map(float, data_tag.firstChild.nodeValue.split()))

        I = np.array(data)
        if data_tag.nodeName=="counts":
            dI = np.sqrt(I)
            I /= time / atten
            dI /= time / atten
        else:
            dI = np.sqrt(I*atten)
            I /= time
            dI /= time

        if "2Theta" in moving_positions:
            th = np.linspace(moving_positions["2Theta"][0], moving_positions["2Theta"][1], len(data)) / 2.0
        else:
            th = np.linspace(moving_positions["Omega"][0], moving_positions["Omega"][1], len(data))
        meta_measurement["instrument_settings"] = {
            "incident_angle": {"min": float(th.min()), "max": float(th.max()), "unit": "deg"},
            "wavelength": {"magnitude": lamda, "unit": "angstrom"},
        }
        datasets.append((2*th, I, dI, sample_name, meta_experiment, meta_measurement))
    return datasets
