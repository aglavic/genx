'''data_loader_framework.py

Library that implements a template (Template) class for classes that
loads data into GenX.
'''


class Template:
    wildcard=None

    def __init__(self, parent):
        self.parent = parent
        # This is made for the virtual datalist controller...
        self.data = self.parent.data_cont.get_data()
        self.Register()
        
    def Register(self):
        '''Register(self) --> None
        
        Register the data loader with the parent
        '''
        self.parent.data_loader = self
        
    def SetData(self, data):
        '''SetData(self, data) --> None
        
        Sets the data connection to the plugin.
        '''
        self.data = data
        
    def UpdateDataList(self):
        '''UpdateDataList(self) --> None
        
        Forces the data list to update. This is only necessary if new
        data sets have been added when the data has been loaded
        '''
        # Just a force update of the data_list
        self.parent.SetItemCount(
            self.parent.data_cont.get_count())
        # Updating the imagelist as well
        self.parent._UpdateImageList()
        
    def SetStatusText(self, text):
        '''SetStatusText(self, text) --> None
        
        Set a status text in the main frame for user information 
        '''
        self.parent.SetStatusText(text)
    
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
        import wx
        n_selected = len(selected_items)
        if n_selected == 1:
            if self.wildcard is None:
                wc="All files (*.*)|*.*"
            else:
                wc="%s (%s)|%s|All files (*.*)|*.*"%(self.__module__.rsplit('.', 1)[1].capitalize(),
                                                     self.wildcard, self.wildcard)
            dlg = wx.FileDialog(self.parent, message = "Choose your Datafile"
                    , defaultFile = "", wildcard = wc
                    , style=wx.FD_OPEN | wx.FD_CHANGE_DIR)
                    
            if dlg.ShowModal() == wx.ID_OK:
                self.LoadData(selected_items[0], dlg.GetPath())
                return True
            dlg.Destroy()
        else:
            if n_selected > 1:
                dlg = wx.MessageDialog(self.parent,
                                       'Please select only one dataset'
                                       , caption = 'Too many selections'
                    , style = wx.OK|wx.ICON_INFORMATION)
            else:
                dlg = wx.MessageDialog(self.parent, 'Please select a dataset'
                    , caption = 'No active dataset'
                    , style = wx.OK|wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
        
        return False
    
    def LoadData(self, data_item, file_path):
        '''LoadData(self, data_item, file_path) --> None
        
        This file should load a single data file into data object of
        the model. Please overide this function. It is called by the 
        LoadFile function.
        '''
        pass
        
    def SettingsDialog(self):
        '''SettingsDialog(self) --> None
        
        This function should - if necessary implement a dialog box
        that allows the user set import settings for example.
        '''
        pass
    
    def SendUpdateDataEvent(self):
        '''SendUpdateDataEvent(self) --> None
        
        Sends an event that new data has been loaded and 
        things such as plotting should be updated.
        '''
        self.parent._UpdateData('New data added')
        
    def Remove(self):
        '''Remove(self) --> None
        
        Removes the plugin from knowledge of the parent.
        '''
        self.parent.data_loader = None
