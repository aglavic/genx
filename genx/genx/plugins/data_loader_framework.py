'''data_loader_framework.py

Library that implements a template (Template) class for classes that
loads data into GenX.
'''
from ..data import DataSet, META_DEFAULT
from .utils import ShowInfoDialog


class Template:
    wildcard=None

    def __init__(self, parent=None):
        self.parent=parent

        if parent is not None:
            # This is made for the virtual datalist controller...
            self.data=self.parent.data_cont.get_data()
            self.Register()

    def Register(self):
        '''
        Register the data loader with the parent
        '''
        self.parent.data_loader=self

    def SetData(self, data):
        '''
        Sets the data connection to the plugin.
        '''
        self.data=data

    def UpdateDataList(self):
        '''
        Forces the data list to update. This is only necessary if new
        data sets have been added when the data has been loaded
        '''
        # Just a force update of the data_list
        self.parent.SetItemCount(
            self.parent.data_cont.get_count())
        # Updating the imagelist as well
        self.parent._UpdateImageList()

    def SetStatusText(self, text):
        '''
        Set a status text in the main frame for user information
        '''
        self.parent.SetStatusText(text)

    def CountDatasets(self, file_path):
        '''
        Count the number of dataset in a file. Default implementation of
        dataloader allows for only one item.
        '''
        return 1

    def GetWildcardString(self):
        if self.wildcard is None:
            wc = "All files (*.*)|*.*"
        else:
            wc = "%s (%s)|%s|All files (*.*)|*.*"%(self.__module__.rsplit('.', 1)[1].capitalize(),
                                                   self.wildcard, self.wildcard)
        return wc

    def LoadDataFile(self, selected_items):
        '''
        Selected items is the selcted items in the items in the current DataList
        into which data from file(s) should be loaded. Note that the default
        implementation only allows the loading of a single file! 
        Overriding this function in subclasses can of course change this 
        behaviour. This function calls the LoadData function which implements 
        the io function by it self. The LoadData has to be overloaded in
        order to have a working plugin.
        '''
        import wx
        n_selected=len(selected_items)
        if n_selected>0:
            dlg=wx.FileDialog(self.parent, message="Choose your Datafile"
                              , defaultFile="", wildcard=self.GetWildcardString()
                              , style=wx.FD_OPEN | wx.FD_CHANGE_DIR)

            if dlg.ShowModal()==wx.ID_OK:
                file_path=dlg.GetPath()
                dlg.Destroy()
                dcount=self.CountDatasets(file_path)
                if n_selected>dcount:
                    ShowInfoDialog(self.parent, f'You can only load {dcount} dataset(s) from this file',
                                   'Too many selections')
                    dlg.ShowModal()
                    dlg.Destroy()
                else:
                    self.SetData(self.parent.data_cont.get_data())
                    for di in range(n_selected):
                        dataset=DataSet(copy_from=self.data[selected_items[di]])
                        self.LoadDataset(dataset, file_path, data_id=di)

                        if len(dataset.x)==0:
                            continue

                        self.data[selected_items[di]]=dataset
                    # In case the dataset name has changed
                    self.UpdateDataList()

                    # Send an update that new data has been loaded
                    self.SendUpdateDataEvent()

                    return True
            else:
                dlg.Destroy()
        else:
            ShowInfoDialog(self.parent, 'Please select a dataset', 'No active dataset')
        return False

    def LoadDataset(self, dataset, file_path, data_id=0):
        '''
        A wrapper around LoadData that makes sure there is minimal metadata generated.
        '''
        dataset.meta = META_DEFAULT.copy()
        # in case the data loader does not define any metadata
        # at least set the instrument to data loader name
        dataset.meta['data_source']['experiment']['instrument'] = self.__module__.rsplit('.', 1)[1]
        self.LoadData(dataset, file_path, data_id=data_id)
        dataset.meta['data_source']['file_name'] = file_path

    def LoadData(self, dataset, file_path, data_id=0):
        '''
        This file should load a single data file into data object of
        the model. Please overide this function. It is called by the 
        LoadFile function.
        '''
        pass

    def SettingsDialog(self):
        '''
        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        '''
        pass

    def SendUpdateDataEvent(self):
        '''
        Sends an event that new data has been loaded and 
        things such as plotting should be updated.
        '''
        self.parent._UpdateData('New data added')

    def Remove(self):
        '''
        Removes the plugin from knowledge of the parent.
        '''
        self.parent.data_loader=None

    def CanOpen(self, file_path):
        """
        Return if the data loader class can open the given file.
        Default implementation just checks if the filename matches the wildcard, if it exists.
        """
        if self.wildcard is None:
            return True
        for wc in self.wildcard.split(';'):
            if file_path.endswith(wc[1:]):
                return True
        return False
