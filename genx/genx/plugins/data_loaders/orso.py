'''
==========================================
:mod:`orso` ORSO standard file data loader
==========================================

Loads the format following the specification of the Open Reflectometry Standards Organization (ORSO).
See https://www.reflectometry.org/working_groups/file_formats/ for more details
'''

import numpy as np
import re

from typing import List
from ..data_loader_framework import Template
from ..utils import ShowWarningDialog
from orsopy.fileio import load_orso, OrsoDataset

class Plugin(Template):
    wildcard='*.ort'
    _cached_file = ''
    _cached_data: List[OrsoDataset] = None

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
        return l1.startswith('# # ORSO')

    def LoadCached(self, file_path):
        if self._cached_file==file_path:
            return self._cached_data
        else:
            orso_datasets = load_orso(file_path)
            self._cached_file=file_path
            self._cached_data=orso_datasets
            return orso_datasets

    def CountDatasets(self, file_path):
        try:
            orso_datasets=self.LoadCached(file_path)
        except Exception as e:
            return 1
        else:
            return len(orso_datasets)

    def LoadData(self, dataset, filename, data_id=0):
        '''
        Loads the data from filename into the data_item_number.
        '''
        try:
            orso_datasets=self.LoadCached(filename)
        except Exception as e:
            ShowWarningDialog(self.parent, 'Could not load the file: '+filename+
                              ' \nPlease check the format.\n\n ORSO reader gave the following error:\n'+str(e))
            return
        else:
            orso_dataset: OrsoDataset=orso_datasets[data_id]
            data: np.ndarray=orso_dataset.data.T

            cols=orso_dataset.info.columns

            start_usercols=3
            dataset.x_raw=np.asarray(data[0])
            dataset.y_raw=np.asarray(data[1])
            dataset.error_raw=np.asarray(data[2])
            if data.shape[0] > 3 and cols[3].name==('s'+cols[0].name):
                start_usercols=4
                dataset.set_extra_data('res', np.asarray(data[3]), 'res')
            for i, col in enumerate(data[start_usercols:]):
                colname=cols[i+start_usercols].name
                if not colname.isidentifier() or colname in ['lambda', 'x', 'y', 'error', 'res']:
                    colname='col_'+re.sub('[^0-9a-zA-Z_]', '', colname)
                dataset.set_extra_data(colname, np.asarray(col), colname)
            # Name the dataset accordign to file name
            if orso_dataset.info.data_set:
                dataset.name=str(orso_dataset.info.data_set)
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta=orso_dataset.info.to_dict()
            if 'data_set' in dataset.meta:
                del(dataset.meta['data_set'])
