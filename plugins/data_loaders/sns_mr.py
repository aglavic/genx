'''
=====================================================
:mod:`sns_mr` SNS magnetism reflectometer data loader
=====================================================

Loads the default datafile format used for extracted reflectivity at
the SNS magnetism reflectometer. It allows to import several channels at
once, automatically naming the datasets according to the imported
polarizations.
'''

import numpy as np
import wx

from plugins.data_loader_framework import Template
from plugins.utils import ShowWarningDialog
import event_handlers

class Plugin(Template):
    wildcard = '*.dat'

    def __init__(self, parent):
        Template.__init__(self, parent)
        self.x_col = 0
        self.y_col = 1
        self.e_col = 2
        self.xe_col = 3
        self.ai_col = 4
        self.comment = '#'
        self.skip_rows = 0
        self.delimiter = None

    def LoadData(self, data_item_number, filename):
        '''LoadData(self, data_item_number, filename) --> none

        Loads the data from filename into the data_item_number.
        '''
        # get scan number and polarization channel from filename
        name = ''
        fhandle = open(filename, 'r')
        fline = fhandle.readline()
        while not '[Data]' in fline:
            if 'Input file indices:' in fline:
                name += fline.split(':')[1].strip()
            if 'Extracted states:' in fline:
                name += ' (%s)'%(fline.split(':')[1].strip().split(' ')[0])
            fline = fhandle.readline()
            if not fline:
                ShowWarningDialog(self.parent, 'Could not load the file: ' + filename+' \nWrong format.\n')
                return

        try:
            load_array = np.loadtxt(fhandle, delimiter=self.delimiter, comments=self.comment, skiprows=self.skip_rows)
        except Exception, e:
            ShowWarningDialog(self.parent, 'Could not load the file: ' + filename +
                              ' \nPlease check the format.\n\n numpy.loadtxt' + ' gave the following error:\n' + str(e))
            return
        else:
            # For the freak case of only one data point
            if len(load_array.shape) < 2:
                load_array = np.array([load_array])
            # Check so we have enough columns
            if load_array.shape[1] - 1 < max(self.x_col, self.y_col, self.e_col):
                ShowWarningDialog(self.parent, 'The data file does not contain' + 'enough number of columns. It has '
                                  + str(load_array[1]) + ' columns. Rember that the column index start at zero!')
                # Okay now we have showed a dialog lets bail out ...
                return
            # The data is set by the default Template.__init__ function, neat hu
            # Know the loaded data goes into *_raw so that they are not
            # changed by the transforms
            self.data[data_item_number].x_raw=load_array[:, self.x_col]
            self.data[data_item_number].y_raw=load_array[:, self.y_col]
            self.data[data_item_number].error_raw=load_array[:, self.e_col]
            self.data[data_item_number].extra_data_raw['xe'] = load_array[:, self.xe_col]
            self.data[data_item_number].extra_commands['xe'] = 'xe'
            self.data[data_item_number].extra_data_raw['ai'] = load_array[:, self.ai_col]
            self.data[data_item_number].extra_commands['ai'] = 'ai'
            # Name the dataset accordign to file name
            self.data[data_item_number].name = name
            # Run the commands on the data - this also sets the x,y, error memebers
            # of that data item.
            self.data[data_item_number].run_command()

            # Send an update that new data has been loaded
            self.SendUpdateDataEvent()
            self.UpdateDataList()

    def LoadDataFile(self, selected_items):
        '''LoadDataFile(self, seleceted_items) --> None

        Selected items is the selcted items in the items in the current DataList
        into which data from file(s) should be loaded. Note that the default
        implementation only allows the loading of a single file!
        Overriding this function in subclasses can of course change this
        behaviour. This function calls the LoadData function which implements
        the io function by it self. The LoadData has to be overloaded in
        order to have a working plugin.
        '''
        n_selected = len(selected_items)
        dlg = wx.FileDialog(self.parent, message="Choose your Datafile", wildcard="All files (%s)|%s" % (self.wildcard,
                                                                                                         self.wildcard)
                           , style=wx.FD_OPEN|wx.FD_MULTIPLE|wx.FD_CHANGE_DIR)

        if dlg.ShowModal() == wx.ID_OK:
            files = dlg.GetPaths()
            dlg.Destroy()
            if len(files) != n_selected:
                dlg = wx.MessageDialog(self.parent, 'Please select %i dataset(s)' % n_selected,
                                       caption='Wrong number of selections', style=wx.OK | wx.ICON_INFORMATION)
                dlg.ShowModal()
                dlg.Destroy()
                return self.LoadDataFile(selected_items)
            for i, fname in enumerate(files):
                self.LoadData(selected_items[i], fname)
            return True
        return False
