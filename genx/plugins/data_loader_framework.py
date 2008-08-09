'''framework.py

Library that implements a template (Template) class for classes that
loads data into GenX. Also included here is a DataLoaderController which
takes care of the use of the DataLoaders. 
'''
import wx, os

from utils import PluginHandler

head, tail = os.path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled... 
__FILENAME__ = tail.split('.')[0]
# This assumes that plugin is under the current dir may need 
# changing
__MODULE_DIR__ = head

class Template:
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
        if n_selected == 1:
            dlg = wx.FileDialog(self.parent, message = "Choose your Datafile"
                    , defaultFile = "", wildcard = "All files (*.*)|*.*"
                    , style=wx.OPEN | wx.CHANGE_DIR)
                    
            if dlg.ShowModal() == wx.ID_OK:
                self.LoadData(selected_items[0], dlg.GetPath())
            dlg.Destroy()
        else:
            if n_selected > 1:
                dlg = wx.MessageDialog(self.parent,\
                    'Please select only one dataset'\
                    , caption = 'Too many selections'
                    , style = wx.OK|wx.ICON_INFORMATION)
            else:
                dlg = wx.MessageDialog(self.parent, 'Please select a dataset'
                    , caption = 'No active dataset'
                    , style = wx.OK|wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()    
    
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
        
class PluginController:
    def __init__(self, parent):
        self.plugin_handler = PluginHandler(parent, __MODULE_DIR__ \
                            , 'data_loaders')
        self.parent = parent
        self.plugin_handler.load_plugin('default')
        
    def LoadPlugin(self, plugin):
        '''LoadPlugin(self, plugin) --> None
        
        Loads a data handler note that there is no UnLoad function
        since only one DataHandler can be plugged in at a time.
        '''
        # Unload the plugins
        names = self.plugin_handler.loaded_plugins.copy()
        try:
            [self.plugin_handler.unload_plugin(plugin) for plugin\
                in names]
        except:
            outp = StringIO.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
            ShowErrorDialog(self.parent, 'Can NOT unload plugin object'+ \
            self.plugin_handler.loaded_plugins[0]\
             + '\nPython traceback below:\n\n' + tbtext)
        try:
            self.plugin_handler.load_plugin(plugin)
        except:
            outp = StringIO.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
            ShowErrorDialog(self.parent, 'Can NOT load plugin ' + plugin\
             + '\nPython traceback below:\n\n' + tbtext)
        
    def ShowDialog(self):
        '''ShowDialog(self) --> None
        
        Shows a dialog boc for the user to choose a data loader.
        '''
        cur_plugin = self.plugin_handler.loaded_plugins.keys()[0]
        plugin_list = self.plugin_handler.get_possible_plugins()
        dlg = PluginDialog(self.parent, plugin_list, cur_plugin,\
                                self.LoadPlugin)
        dlg.ShowModal()
        dlg.Destroy()
        
        
class PluginDialog(wx.Dialog):
    def __init__(self, parent, plugin_list, current_plugin,\
            load_plugin_func = None):
        wx.Dialog.__init__(self, parent, -1, 'Choose a data loader')
        
        self.load_plugin_func = load_plugin_func
        
        choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer = wx.BoxSizer(wx.VERTICAL)
        
        choice_text = wx.StaticText(self, -1, 'Data loaders: ')
        self.choice_control = wx.Choice(self, -1, choices = plugin_list)
        self.choice_control.SetStringSelection(current_plugin)
        choice_sizer.Add(choice_text, 0, wx.ALIGN_CENTER_VERTICAL)
        choice_sizer.Add(self.choice_control, 0, wx.ALIGN_CENTER_VERTICAL)
        
        # Add the Dialog buttons
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        apply_button = wx.Button(self, wx.ID_APPLY)
        apply_button.SetDefault()
        button_sizer.AddButton(apply_button)    
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        # Add some eventhandlers
        self.Bind(wx.EVT_BUTTON, self.on_apply, okay_button)
        self.Bind(wx.EVT_BUTTON, self.on_apply, apply_button)
        
        
        sizer.Add((20, 20), 0)
        sizer.Add(choice_sizer, 0, wx.ALIGN_CENTRE|wx.ALL)
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL|wx.TOP, 20)
        sizer.Add(button_sizer)
        
        self.SetSizer(sizer)
        
        sizer.Fit(self)
        self.Layout()
        
    def on_apply(self, event):
        if self.load_plugin_func != None:
            self.load_plugin_func(self.choice_control.GetStringSelection())
        event.Skip()
        