import numpy as np

from genx.core.custom_logging import iprint
from genx.gui_generic.add_ons.specloader_wizard import DataLoadWizard

from ..data_loader_framework import Template
from ..utils import ShowErrorDialog, ShowWarningDialog
from .help_modules import spec

_maxWidth = 450


class Plugin(Template):
    def __init__(self, parent):
        Template.__init__(self, parent)
        self.specfile = None
        self.specfile_name = None
        self.scan = None
        # self.dataset = data.DataSet()
        self.datalist = self.data

    def LoadDataFile(self, selected_items, data_id=0):
        """Called when GenX wants to load data"""

        if len(selected_items) == 0:
            ShowWarningDialog(self.parent, "Please select a data set before trying to load a spec file.")
            return False

        DataLoadWizard(self, selected_items)

    def load_specfile(self, filename):
        self.specfile = spec.SpecDataFile(filename)
        names = list(self.specfile.findex.keys())
        names.sort()
        try:
            sc = self.specfile.scan_commands
            names = ["%s " % (name,) + sc[name] for name in names]
        except Exception as e:
            iprint("Could not create full names, error: ", e.__str__())
            ShowErrorDialog(self.parent, "Could not create full names, error:\n %s" % e.__str__())
            names = ["%s " % name for name in names]
        self.specfile_name = filename
        self.scan_names = names
        return self.scan_names

    def load_scans(self, scanlist):
        """Function to load the spec scans as a list of scan numbers"""
        # self.dataset = data.DataSet()
        try:
            self.scan = self.specfile[scanlist]
            self.dataset.set_extra_data("values", self.scan.values)
        except Exception as e:
            iprint("Could not load the desired scans")
            iprint("Error: ", e.__str__())
            iprint("scanlist: ", scanlist)
            ShowErrorDialog(self.parent, "Could not load the desired scans, error:\n%s " % e.__str__())
        # for key in self.scan.values:
        #    try:
        #        self.dataset.set_extra_data(key, self.scan.values[key])
        #    except Exception, e:
        #        print "Could not load ", key
        #        print error

    def get_data_choices(self):
        """Function to return the data choices"""
        try:
            choices = self.scan.cols
        except Exception as e:
            iprint("Could not load the scan cols error: ", e.__str__())
            ShowErrorDialog(self.parent, "Could not load the scan cols, error:\n%s " % e.__str__())
            choices = []

        return choices

    def update_data_cols(self, cols, autom=False):
        """Update the choices for the different values"""
        iprint(cols)
        self.x_val = cols[0]
        self.det_val = cols[1]
        self.mon_val = cols[2]
        self.error_val = cols[3]
        xval = self.scan.values[self.x_val]
        yvals = [
            self.scan.values[self.det_val],
        ]
        if self.mon_val != "None":
            yvals.append(self.scan.values[self.mon_val])
        if self.error_val != "None" and self.error_val != "":
            yvals.append(self.scan.values[self.error_val])
        if autom:
            xval, yvals = automerge(xval, yvals)
        # self.dataset.set_extra_data('det', self.scan.values[self.det_val])
        self.dataset.set_extra_data("det", yvals[0])
        if self.mon_val != "None":
            # self.dataset.set_extra_data('mon', self.scan.values[self.mon_val])
            self.dataset.set_extra_data("mon", yvals[1])

            if self.error_val != "None" and self.error_val != "":
                self.dataset.error_raw = yvals[2]
        elif self.error_val != "None" and self.error_val != "":
            self.dataset.error_raw = yvals[1]

        # self.dataset.x_raw = self.scan.values[self.x_val]
        self.dataset.x_raw = xval
        # if self.error_val != 'None' and self.error_val != '':
        #    self.dataset.error_raw = self.scan.values[self.error_val]

    def get_dataset_names(self):
        return [d.name for d in self.datalist]

    def set_commands(self, commands):
        """Sets the data commands if valid to commands"""
        result = self.dataset.try_commands(commands)
        if result != "":
            ShowErrorDialog(self.parent, result)
            return False
        self.dataset.set_commands(commands)
        self.dataset.run_command()
        return True

