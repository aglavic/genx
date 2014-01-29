''' utils.py

module for utils needed for the plugin frameworks. This module implemtents a
handler which searches for plugins and loads and removes them.
Programmer: Matts Bjorck
Last changed: 2008 07 23
'''

import os, wx

#==============================================================================

class PluginHandler:
    ''' Class that takes care of the loading/unloading of the plugins.
    This is used in the main GUI. Should not be tinkered with. 
    The plugin module can only conatain files wich are real plugins i.e.
    that implements the class Template. Of course it is possible to have
    subdirectories that has support libraries.
    
    Note that the plugins modules has to have an class called Plugin
    and that this class should implement the method Remove. The init function
    should only take the parent (i.e. the original window frame) as input.
    '''
    def __init__(self, parent, directory, plugin_folder):
        '''__init__(self, parent)
        Create an instance of PluginHandler
        '''
        self.parent = parent
        #self.modules = []
        self.loaded_plugins = {}
        self.directory = directory
        self.plugin_folder = plugin_folder
        
    def get_plugins(self):
        '''get_possible_plugins(self) --> list of strings
        
        search the plugin directory. 
        Checks the list for python scripts and returns a list of 
        module names that are NOT loaded.
        '''
        # Locate all python files in this files directory
        # but excluding this file and not loaded.
        plugins = [s[:-3] for s in os.listdir(self.directory\
                        + self.plugin_folder) if '.py' == s[-3:] 
                        and s[:2] != '__' and \
                        not self.loaded_plugins.has_key(s[:-3])]
        return plugins
    
    def get_possible_plugins(self):
        '''get_possible_plugins(self) --> list of strings
        
        search the plugin directory. 
        Checks the list for python scripts and returns a list of 
        module names that are loadable .
        '''
        # Locate all python files in this files directory
        # but excluding this file and not loaded.
        plugins = [s[:-3] for s in os.listdir(self.directory\
                        + self.plugin_folder) if '.py' == s[-3:] 
                        and s[:2] != '__']
        return plugins
    
    def get_loaded_plugins(self):
        '''get_loaded_plugins(self) --> plugins [list]
        
        returns a list of the loaded plugins
        '''
        return self.loaded_plugins.keys()
        
    def load_plugin(self, plugin_name):
        '''load_plugin(self, plugin_name) --> plugin object
        load the plugin given by the plugin_name [string].
        '''
        # Loafd the module
        module = self._load_module(self.plugin_folder + '.' + plugin_name)
        # Create the Plugin object from the module
        plugin = module.Plugin(self.parent)
        # Add it to our dictonary over loaded plugins
        self.loaded_plugins[plugin_name] = plugin
            
    def is_loaded(self, plugin_name):
        '''isloaded(self, plugin_name) --> loaded [bool]
        
        Returns True if plugin with name plugin_name is loaded, 
        otherwise False.
        '''
        return self.loaded_plugins.has_key(plugin_name)
    
    def _load_module(self, module_name):
        ''' _load_module(self, module) --> module
        Load a module given by name
        '''
        #print 'Trying to load module: ', module_name
        module = __import__(module_name, globals(), locals(), ['plugins'])
        return module
        
    def unload_plugin(self, plugin_name):
        ''' unload_plugin(self, plugin_name) --> None
        Used to remove the plugin from the system.
        '''
        #print self.loaded_plugins.keys()
        self.loaded_plugins[plugin_name].Remove()
        del self.loaded_plugins[plugin_name]
        

# END: PluginHandler
#==============================================================================
# Utility Dialog functions..
def ShowInfoDialog(frame, message):
    dlg = wx.MessageDialog(frame, message,
                               'Information',
                               wx.OK | wx.ICON_INFORMATION
                               )
    dlg.ShowModal()
    dlg.Destroy()
    
def ShowErrorDialog(frame, message, position = ''):
    dlg = wx.MessageDialog(frame, message,
                               'ERROR',
                               wx.OK | wx.ICON_ERROR
                               )
    dlg.ShowModal()
    dlg.Destroy()

def ShowWarningDialog(frame, message):
    dlg = wx.MessageDialog(frame, message, 'Warning',
                               wx.OK | wx.ICON_ERROR
                               )
    dlg.ShowModal()
    dlg.Destroy()
