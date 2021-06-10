'''
==========================================
:mod:`orso` ORSO standard file data loader
==========================================

Loads the format following the specification of the Open Reflectometry Standards Organization (ORSO).
See https://www.reflectometry.org/working_groups/file_formats/ for more details
'''

import numpy as np
import os

from ..data_loader_framework import Template
from ..utils import ShowWarningDialog
from genx.lib.orso_io.ort import read_file

class Plugin(Template):
    wildcard='*.ort'

    def __init__(self, parent):
        Template.__init__(self, parent)
        self.x_col=0
        self.y_col=1
        self.e_col=3
        self.xe_col=2
        self.comment='#'
        self.skip_rows=0
        self.delimiter=None

    def CanOpen(self, file_path):
        if not Template.CanOpen(self, file_path):
            return False
        l1=open(file_path, 'r', encoding='utf-8').readline()
        return l1.startswith('# ORSO')

    def LoadData(self, dataset, filename, data_id=0):
        '''LoadData(self, data_item_number, filename) --> none

        Loads the data from filename into the data_item_number.
        '''
        try:
            ord_data=read_file(filename)
        except Exception as e:
            ShowWarningDialog(self.parent, 'Could not load the file: '+filename+
                              ' \nPlease check the format.\n\n ORSO reader gave the following error:\n'+str(e))
            return
        else:
            if type(ord_data) is list:
                ord_data=ord_data[data_id]

            dataset.x_raw=ord_data.x
            dataset.y_raw=ord_data.y
            dataset.error_raw=ord_data.dy
            if ord_data.dx is not None:
                dataset.set_extra_data('res', ord_data.dx, 'res')
            for col in ord_data[4:]:
                dataset.set_extra_data(col.name, np.asarray(col), col.name)
            # Name the dataset accordign to file name
            if ord_data.name.strip()!='':
                dataset.name=ord_data.name
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta=dict(ord_data.header)
            del(dataset.meta['columns'])
            if 'data_set' in dataset.meta:
                del(dataset.meta['data_set'])
