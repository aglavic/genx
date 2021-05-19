''' Module to handle plugins including a template class for writing 
plugins. 

Programmer: Matts Bjorck
Last changed: 2008 07 23
'''

import os
import wx, io, traceback

from .utils import PluginHandler

head, tail=os.path.split(__file__)
# Look only after the file name and not the ending since
# the file ending can be pyc if compiled... 
__FILENAME__=tail.split('.')[0]
# This assumes that plugin is under the current dir may need 
# changing
__MODULE_DIR__=head
if __MODULE_DIR__!='/':
    __MODULE_DIR__+='/'

class Template:
    ''' A template class for handling plugins. Note that using the 
    New* methods will automatically remove them when removing the plugin.
    Otherwise the programmer, if he/she makes more advanced changes in the gui,
    have to take care of the deletion of objects.
    '''

    # TODO: Implement an Bind/Unbind mehtods.
    # TODO: Add a Load data function Needs change in data as well.
    def __init__(self, parent):
        '''__init__(self, parent)
        This method should be overloaded.
        '''
        self.parent=parent
        self.plot_pages=[]
        self.input_pages=[]
        self.data_pages=[]
        self.menus=[]
        self.parent.input_notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnInputPageChanged)

    def OnInputPageChanged(self, evt):
        tpage, fpage=evt.GetSelection(), evt.GetOldSelection()
        if fpage!=tpage:
            tpanel=self.parent.input_notebook.GetPage(tpage)
            if tpanel in self.input_pages:
                self.InputPageChanged(self.parent.input_notebook.GetPageText(tpage))
        evt.Skip()

    def InputPageChanged(self, pname):
        # Can be used to react to own plugin pages being selected
        pass

    def NewPlotFolder(self, name, pos=-1):
        '''NewPlotFolder(self, name) --> wx.Panel
        
        Creates a new Folder in the Plot part of the panels. Returns
        a wx.Panel which can be used to create custom controls. 
        o not forget to use RemovePlotfolder in the Destroy method.
        '''
        panel=wx.Panel(self.parent.plot_notebook, -1)
        self.parent.plot_notebook.AddPage(panel, name)
        self.plot_pages.append(panel)

        return panel

    def NewInputFolder(self, name, pos=-1):
        '''NewInputFolder(self, name, pos = -1) --> wx.Panel
        
        Creates a new Folder in the Input part of the panels. Returns
        a wx.Panel which can be used to create custom controls. 
        o not forget to use RemoveInputfolder in the Destroy method.
        '''
        panel=wx.Panel(self.parent.input_notebook, -1)
        self.parent.input_notebook.AddPage(panel, name)
        self.input_pages.append(panel)

        return panel

    def NewDataFolder(self, name, pos=-1):
        '''NewDataFolder(self, name, pos = -1) --> wx.Panel
        
        Creates a new Folder in the data part of the panels. Returns
        a wx.Panel which can be used to create custom controls. 
        o not forget to use RemoveInputfolder in the Destroy method.
        '''
        panel=wx.Panel(self.parent.data_notebook, -1)
        self.parent.data_notebook.AddPage(panel, name)
        self.data_pages.append(panel)

        return panel

    def NewMenu(self, name):
        '''NewMenu(self, name) --> wx.Menu
        
        Creates an top menu that can be used to control the plugin. Remeber
        to also implement RemoveMenu in the Destroy method.
        '''
        menu=wx.Menu()
        self.parent.main_frame_menubar.Append(menu, name)
        self.menus.append(name)

        return menu

    def StatusMessage(self, text):
        '''StatusMessage(self, text) --> None
        
        Method that sets the staustext in the main window
        '''
        self.parent.main_frame_statusbar.SetStatusText(text, 1)

    def ShowErrorDialog(self, message):
        '''ShowErrorDialog(self, message) --> None
        
        Shows an error dialog with message [string]
        '''
        ShowErrorDialog(self.parent, message)

    def ShowInfoDialog(self, message):
        '''ShowInfoDialog(self, message) --> None
        
        Shows an info dialog with message [string]
        '''
        ShowInfoDialog(self.parent, message)

    def ShowWarningDialog(self, message):
        '''ShowWarningDialog(self, message) --> None
        
        Shows an warning dialog with message [string]
        '''
        ShowWarningDialog(self.parent, message)

    def ShowQuestionDialog(self, message):
        '''ShowWarningDialog(self, message) --> None
        
        Shows an warning dialog with message [string]
        '''
        return ShowQuestionDialog(self.parent, message)

    def GetModel(self):
        '''GetModel(self) --> model 
        
        Returns the model currently in use. This is a pointer to the model
        object thus it will automatically always conatin the newest information.        
        '''
        return self.parent.model

    def GetSolverControl(self):
        '''GetSolverControl(self) --> solver_control
        
        Returns the solver_control object that controls all aspects of
        the calculational part of the fitting.
        '''
        return self.parent.solver_control

    def SetModelScript(self, script):
        '''SetModelScript(self, script) --> None
        
        Sets the script of the current model. This overwrite the current 
        script.
        '''

        self.parent.script_editor.SetText(script)
        self.parent.model.set_script(script)

    def GetModelScript(self):
        """Returns the model script"""
        return self.parent.model.get_script()

    def CompileScript(self):
        '''CompileScript(self) --> None
        
        Compiles the model script
        '''
        self.parent.model.compile_script()

    def GetScriptModule(self):
        """Returns the script module"""
        return self.parent.model.script_module

    def OnNewModel(self, event):
        '''OnNewModel(self) --> None
        
        Function to be overridden. Called when a new model is being created.
        '''
        pass

    def OnDataChanged(self, event):
        '''OnDataChanged(self) --> None
        
        Function to be overridden. Called when a new data set has been loaded
        or deleted.
        '''
        pass

    def OnOpenModel(self, event):
        '''OnOpenModel(self, event) --> None
        
        Function that is called after a new model has been loaded.
        Used to set up plugin specific model stuff. To be overridden
        '''
        pass

    def OnSimulate(self, event):
        '''OnSimulate(self, event) --> None
        
        Function that is called after a simulation has been done.
        To be overridden
        '''
        pass

    def OnFittingUpdate(self, event):
        '''OnFittingUpdate(self, event) --> None
        
        Function that is called when the fitting algorithm pushes an update event.
        To be overridden
        '''
        pass

    def OnGridChange(self, event):
        """Responds to a change in the grid.

        :param event:
        :return:
        """
        pass

    def Remove(self):
        '''Remove(self) --> None
        Removes all components.
        '''
        pnb=self.parent.plot_notebook
        inb=self.parent.input_notebook
        dnb=self.parent.data_notebook

        # remove all pages from the notebooks
        for panel in self.plot_pages:
            pages=[pnb.GetPage(i) for i in range(pnb.GetPageCount())]
            idx=pages.index(panel)
            pnb.DeletePage(idx)
        for panel in self.input_pages:
            pages=[inb.GetPage(i) for i in range(inb.GetPageCount())]
            idx=pages.index(panel)
            inb.DeletePage(idx)
        for panel in self.data_pages:
            pages=[dnb.GetPage(i) for i in range(dnb.GetPageCount())]
            idx=pages.index(panel)
            dnb.DeletePage(idx)
        # Remove the menus
        for name in self.menus:
            idx=self.parent.main_frame_menubar.FindMenu(name)
            self.parent.main_frame_menubar.Remove(idx)

# END: Template
# ==============================================================================

# ==============================================================================
class PluginController:
    ''' A controller class to interact with the gui 
    so we can load and unload modules as well as
    update the module list.
    '''

    def __init__(self, parent, menu, config):
        '''__init__(self, parent, menu) --> None
        
        Insert menu items for controlling plugins in menu. 
        Parent is the main window.
        '''
        self.plugin_handler=PluginHandler(parent, __MODULE_DIR__, 'add_ons')
        self.parent=parent
        self.config=config

        # make the menus
        self.load_menu=wx.Menu()
        menu.Insert(0, -1, 'Load', self.load_menu, 'Load a plugin')
        self.unload_menu=wx.Menu()
        menu.Insert(1, -1, 'Unload', self.unload_menu, 'Unload a plugin')

        menu.Append(-1, 'Update module list')

        # wx.CallAfter(self.LoadDefaultPlugins)
        # self.update_plugins()

    def update_plugins(self):
        '''update_modules(self) --> None
        
        Updates the list of modules that can be loaded.
        '''
        # Remove all the items in load_menu
        items=self.load_menu.GetMenuItems()
        [self.load_menu.Delete(item) for item in items]

        # Get the new list of plugin modules
        modlist=self.plugin_handler.get_plugins()
        modlist.sort()

        # Add new menu items
        for mod in modlist:
            menu=self.load_menu.Append(-1, mod)
            self.parent.Bind(wx.EVT_MENU, self.LoadPlugin, menu)

        self.update_config()

    def RegisterPlugin(self, plugin):
        ''' RegisterPlugin(self, plugin) --> None
        
        Adds a plugin to the unload list so that it can be removed later.
        '''
        menu=self.unload_menu.Append(-1, plugin)
        self.parent.Bind(wx.EVT_MENU, self.UnLoadPlugin, menu)
        self.update_plugins()

    def update_config(self):
        '''update_config(self) --> None
        
        Updates the config object
        '''
        loaded_plugins=self.plugin_handler.get_loaded_plugins()
        plugins_str=';'.join(loaded_plugins)
        self.config.set('plugins', 'loaded plugins', plugins_str)

    def LoadDefaultPlugins(self):
        '''LoadDefaultPlugins(self) --> None
        
        Tries to load the default plugins from the config object
        if they are not already loaded.
        '''
        plugin_str=self.config.get('plugins', 'loaded plugins')
        # print' plugins:', plugin_str
        # Check so we have any plugins to load else bail out
        # print 'Plugin string:', plugin_str
        if plugin_str=='':
            self.update_plugins()
            return
        existing_plugins=self.plugin_handler.get_possible_plugins()

        for plugin in plugin_str.split(';'):
            # Check so the plugin is not loaded and exists 
            if not self.plugin_handler.is_loaded(plugin):
                if plugin in existing_plugins:
                    try:
                        self.plugin_handler.load_plugin(plugin)
                        self.RegisterPlugin(plugin)
                    except:
                        outp=io.StringIO()
                        traceback.print_exc(200, outp)
                        tbtext=outp.getvalue()
                        outp.close()
                        ShowErrorDialog(self.parent, 'Can NOT load plugin ' \
                                        +plugin+'\nPython traceback below:\n\n'+tbtext)
                        self.RegisterPlugin(plugin)
                else:
                    ShowInfoDialog(self.parent, 'Could not find plugin "%s"' \
                                                '. Either there is an error in the config file' \
                                                ' or the plugin is not installed.'%plugin)
        self.update_plugins()

    # Callbacks
    def LoadPlugin(self, event):
        '''OnLoadPlugin(self, event) --> None
        
        Loads a plugin from a menu choice.
        '''
        # Get the name of the plugin
        menuitem=self.load_menu.FindItemById(event.GetId())
        plugin=menuitem.GetItemLabel()
        try:
            self.plugin_handler.load_plugin(plugin)
        except:
            outp=io.StringIO()
            traceback.print_exc(200, outp)
            tbtext=outp.getvalue()
            outp.close()
            ShowErrorDialog(self.parent, 'Can NOT load plugin '+plugin \
                            +'\nPython traceback below:\n\n'+tbtext)
        else:
            self.RegisterPlugin(plugin)

    def UnLoadPlugin(self, event):
        '''UnLoadPlugin(self, event) --> None
        
        UnLoads (removes) a plugin module.
        '''
        menuitem=self.unload_menu.FindItemById(event.GetId())
        plugin=menuitem.GetItemLabel()
        if self.UnLoadPlugin_by_Name(plugin):
            # Remove the item from the list
            self.unload_menu.Delete(menuitem)
            # Update the available plugins
            self.update_plugins()

    def UnLoadPlugin_by_Name(self, plugin):
        try:
            self.plugin_handler.unload_plugin(plugin)
        except Exception as e:
            outp=io.StringIO()
            traceback.print_exc(200, outp)
            tbtext=outp.getvalue()
            outp.close()
            ShowErrorDialog(self.parent, 'Can NOT unload plugin object'+ \
                            plugin+'\nPython traceback below:\n\n'+tbtext)
            return False
        else:
            return True

    def OnNewModel(self, event):
        '''OnNewModel(self, event) --> None
        
        Runs plugin code when the user tries to load a new model 
        '''
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnNewModel(event)

    def OnDataChanged(self, event):
        '''OnNewModel(self, event) --> None
        
        Runs plugin code when the user tries to load new data
        '''
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnDataChanged(event)

    def OnOpenModel(self, event):
        '''OnOpenModel(self, event) --> None
        
        Runs plugin code when the user tries to open a model 
        '''
        loaded_plugins=list(self.plugin_handler.loaded_plugins.keys())
        items=self.unload_menu.GetMenuItems()
        for item in items:
            # Remove the item from the list
            self.unload_menu.Delete(item)

        for name in loaded_plugins:
            # self.plugin_handler.loaded_plugins[name].OnOpenModel(event)
            self.plugin_handler.unload_plugin(name)

        self.LoadDefaultPlugins()
        # Update the available plugins
        self.update_plugins()

        loaded_plugins=list(self.plugin_handler.loaded_plugins.keys())
        for name in loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnOpenModel(event)

    def OnSimulate(self, event):
        '''OnOpenModel(self, event) --> None
        
        Runs plugin code when the user has simulated the model
        '''
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnSimulate(event)

    def OnFittingUpdate(self, event):
        '''OnOpenModel(self, event) --> None
        
        Runs plugin code when the Fitting algorithm wants to update its output.
        '''
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnFittingUpdate(event)

    def OnGridChanged(self, event):
        """ Runs plugin code to respond to a changed grid.

        :param event:
        :return:
        """
        for name in self.plugin_handler.loaded_plugins:
            self.plugin_handler.loaded_plugins[name].OnGridChange(event)

# ==============================================================================
# Utility Dialog functions..
def ShowInfoDialog(frame, message):
    dlg=wx.MessageDialog(frame, message,
                         'Information',
                         wx.OK | wx.ICON_INFORMATION
                         )
    dlg.ShowModal()
    dlg.Destroy()

def ShowErrorDialog(frame, message, position=''):
    dlg=wx.MessageDialog(frame, message,
                         'ERROR',
                         wx.OK | wx.ICON_ERROR
                         )
    dlg.ShowModal()
    dlg.Destroy()

def ShowWarningDialog(frame, message):
    dlg=wx.MessageDialog(frame, message, 'Warning',
                         wx.OK | wx.ICON_ERROR
                         )
    dlg.ShowModal()
    dlg.Destroy()

def ShowQuestionDialog(frame, message, title='Question'):
    dlg=wx.MessageDialog(frame, message,
                         title,
                         wx.YES_NO | wx.ICON_QUESTION
                         )
    result=dlg.ShowModal()==wx.ID_YES
    dlg.Destroy()
    return result
