'''
===========================================
:mod:`orso` SINQ SIX standard file from PSI
===========================================

Load the simple text format used at the PSI SINQ neutron source.
Reads all clumns but uses s2t as standard x.
'''

import numpy as np
import re

from typing import List
from ..data_loader_framework import Template
from ..utils import ShowWarningDialog
from orsopy.fileio import load_orso, OrsoDataset

class Plugin(Template):
    wildcard='*.dat'

    def CanOpen(self, file_path):
        if not Template.CanOpen(self, file_path):
            return False
        l1=open(file_path, 'r', encoding='utf-8').readline()
        return 'Data File' in l1

    def LoadData(self, dataset, filename, data_id=0):
        '''
        Loads the data from filename into the data_item_number.
        '''
        try:
            lines=open(filename, 'r').readlines()
        except Exception as e:
            ShowWarningDialog(self.parent, 'Could not load the file: '+filename+
                              ' \nPlease check the format.\n\n ORSO reader gave the following error:\n'+str(e))
            return
        else:
            header=[]
            while len(lines)>0:
                ln=lines.pop(0)
                if '******* DATA ********' in ln:
                    lines.pop(0);lines.pop(0)
                    cols=[c.strip().lower() for c in lines.pop(0).split()]
                    break
                else:
                    header.append(ln.strip())
            while len(lines)>0:
                le=lines.pop(-1)
                if 'END-OF-DATA' in le:
                    break
            data=np.array([li.split() for li in lines], dtype=float).T


            dataset.x_raw=data[0]
            dataset.y_raw=data[cols.index('counts')]/data[cols.index('monitor1')]
            dataset.error_raw=np.sqrt(data[cols.index('counts')])/data[cols.index('monitor1')]

            for i, col in enumerate(data[1:]):
                colname=cols[i++1]
                if not colname.isidentifier() or colname in ['lambda', 'x', 'y', 'error', 'res']:
                    colname='col_'+re.sub('[^0-9a-zA-Z_]', '', colname)
                dataset.set_extra_data(colname, np.asarray(col), colname)
            # Name the dataset accordign to file name
            dataset.name='#'+filename[:-4].rsplit('n',1)[1]

            if 's2t' in cols:
                dataset.x_command='s2t'
            elif 'som' in cols:
                dataset.x_command='2.*som'

            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            dataset.run_command()

            # insert metadata into ORSO compatible fields
            dataset.meta['data_source']['measurement']['data_file_header']='\n'.join(header)
