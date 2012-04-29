#!/usr/bin/env python
'''
Library for GUI+interface layer for the data class. 
Implements one Controller and a customiized ListController
for data. The class that should be used for the outside world
is the DataListController. This has a small toolbar ontop.
File started by: Matts Bjorck

$Rev::                                  $:  Revision of last commit
$Author::                               $:  Author of last commit
$Date::                                 $:  Date of last commit
'''

import wx, os
import wx.lib.colourselect as  csel


import data
import filehandling as io
import images as img
import plugins.data_loader_framework as dlf
#==============================================================================

class DataController:
    '''
    Interface layer class between the VirtualDataList and the Data class
    '''

    def __init__(self,data_list):
        self.data=data_list
    
    def get_data(self):
        return self.data
    
    def get_column_headers(self):
        return ['Name','Show','Use','Errors']
        
    def get_count(self):
        return self.data.get_len()
    
    def get_item_text(self, item, col):
        bool_output = {True: 'Yes', False: 'No'}
        if col == 0:
            return self.data.get_name(item)
        if col == 1:
            return bool_output[self.data[item].show]
        if col == 2:
            return bool_output[self.data.get_use(item)]
        if col == 3:
            return bool_output[self.data.get_use_error(item)]
        #if col == 3:
        #    return '(%i,%i,%i)'%self.data.get_cols(item)
            
        else:
            return ''
    def set_data(self, data):
        self.data = data
           
    def set_name(self,pos,name):
        self.data.set_name(pos,name)
        #print self.data.items[pos].name
        
    def move_up(self, pos):
        self.data.move_up(pos)
        
    def move_down(self, pos):
        self.data.move_down(pos)
        
    def add_item(self):
        self.data.add_new()
        
    def delete_item(self, pos):
        self.data.delete_item(pos)
        
    def get_colors(self):
        colors = []
        for data_set in self.data:
            dc = data_set.data_color
            sc = data_set.sim_color
            colors.append(((int(dc[0]*255), int(dc[1]*255), int(dc[2]*255)),\
                (int(sc[0]*255), int(sc[1]*255), int(sc[2]*255))))
        return colors
        
    def load(self, pos, path):
        self.data[pos].loadfile(path)
        
        #print self.data[pos].x
        
    def get_items_plotsettings(self, pos):
        ''' get_items_plotsettings(self, pos) --> (sim_list, data_list)
        Used as an interface between the DataList and GUI eventhandler 
        for the PlotSettings dialog.
        returns a two lists of dictonaries for the plot settings.
        '''
        sim_list = [self.data[i].get_sim_plot_items() for i in pos]
        data_list = [self.data[i].get_data_plot_items() for i in pos]
        return (sim_list, data_list)
    
    def test_commands(self, command, pos):
        '''test_commands(self, pos) --> result string
        Function to test the commands to check for errors in them. 
        I.e. safe execution. If the string is empty everything is ok other
        wise the string will contain information about the FIRST error occured.
        '''
        for i in pos:
            result = self.data[i].try_commands(command)
            if result != '':
                break
            
        return result
    
    def run_commands(self, command, pos):
        '''run_commands(self, commnd, pos) --> string
        Function that runs the command [dict] for items in position pos. The 
        string contains information if something went wrong. Should be impossible.
        '''
        result =  ''
        for i in pos:
            try:
                self.data[i].set_commands(command)
                self.data[i].run_command()
            except Exception, e:
                result += 'Error occured for data set %i: '%i + e.__str__()
                break
        return result
    
    def compare_sim_y_length(self, pos):
        '''compare_sim_y_length(self, pos) --> bool
        
        Method that compares the length of the simulation and y data to see
        if they are the same pos is the position that should be compared [list]
        '''
        result = True
        for index in pos:
            if self.data[index].y.shape != self.data[index].y_sim.shape:
                result = False
                break
            
        return result
    
    def get_items_commands(self, pos):
        ''' get_items_commands(self, pos) --> list of dicts
        Returns a list of dictonaries with keys x,y,z containing strings
        of commands which can be executed. pos is a list of integer.
        '''
        command_list = [self.data[i].get_commands() for i in pos]
        return command_list
        
    def get_items_names(self):
        ''' get_items_commands(self, pos) --> list of strings
        Returns a list of the names of the data sets.
        '''
        name_list = [self.data.get_name(i) for i in range(self.get_count())]
        return name_list
    
    def set_items_plotsettings(self, pos, sim_list, data_list):
        ''' set_items_plotsettings(self, pos) --> None
        Used as an interface between the DataList and GUI eventhandler 
        for the PlotSettings dialog.
        sim_list and data_list has to have the right elements. See data.py
        '''
        lpos = range(len(sim_list))
        [self.data[i].set_sim_plot_items(sim_list[j]) for\
                    i,j in zip(pos, lpos)]
        [self.data[i].set_data_plot_items(data_list[j]) for\
                     i,j in zip(pos, lpos)]
    
    def show_data(self, positions):
        '''Show only data at the indices given in position
        all other should be hidden.
        '''
        self.data.show_items(positions)

    def toggle_show_data(self, positions):
        '''toggle_show_data(self, pos) --> None
        toggles the show value for the data elements at positions.
        positions should be an iteratable yielding integers
        , i.e. list of integers
        '''
        [self.data.toggle_show(pos) for pos in positions]
        
    def toggle_use_data(self, positions):
        '''toggle_use_data(self, pos) --> None
        toggles the use_data value for the data elements at positions.
        positions should be an iteratable yielding integers
        , i.e. list of integers
        '''
        [self.data.toggle_use(pos) for pos in positions]
        
    def toggle_use_error(self, positions):
        '''toggle_use_error(self, pos) --> None
        toggles the use_data value for the data elements at positions.
        positions should be an iteratable yielding integers
        , i.e. list of integers
        '''
        [self.data.toggle_use_error(pos) for pos in positions]
        
# END: DataController
#==============================================================================

class DataListEvent(wx.PyCommandEvent):
    '''
    Event class for the data list - in order to deal with 
    updating of the plots and such.
    '''
    def __init__(self,evt_type, id, data):
        wx.PyCommandEvent.__init__(self, evt_type, id)
        self.data = data
        self.data_changed = True
        self.new_data = False
        self.description = ''
        self.data_moved = False
        self.position = 0
        self.up = False
        self.deleted = False
        self.name_change = False
        
    def GetData(self):
        return self.data
        
    def SetData(self, data):
        self.data = data
        
    def SetDataChanged(self, data_changed):
        self.data_changed = data_changed
        
    def SetDataMoved(self, position, up = True):
        self.data_moved = True
        self.position = position
        self.up = up
        
    def SetDataDeleted(self, position):
        self.deleted = True
        self.position = position
        
    def SetNewData(self, new_data = True):
        self.new_data = new_data
        
    def SetDescription(self, desc):
        '''
        Set a string that describes the event that has occurred
        '''
        self.description = desc

    def SetNameChange(self):
        '''SetNameChange(self) --> None
        
        Sets that a name of the has changed
        '''
        self.name_change = True
        
# Generating an event type:
myEVT_DATA_LIST = wx.NewEventType()
# Creating an event binder object
EVT_DATA_LIST = wx.PyEventBinder(myEVT_DATA_LIST)

# END: DataListEvent
#==============================================================================

class VirtualDataList(wx.ListCtrl):
    '''
    The listcontrol for the data
    '''
    def __init__(self, parent, data_controller, config = None,\
            status_text = None):
        wx.ListCtrl.__init__(self,parent,-1,\
        style=wx.LC_REPORT|wx.LC_VIRTUAL|wx.LC_EDIT_LABELS)
        self.data_cont = data_controller
        self.config = config
        self.parent = parent
        self.status_text = status_text
        # This will set by the register function in the
        # plugin function !
        self.data_loader = None 
        self.data_loader_cont = dlf.PluginController(self)
        self.show_indices = []
        
        # Set list length
        self.SetItemCount(self.data_cont.get_count())      
        
        # Set the column headers
        cols=self.data_cont.get_column_headers()
        for col,text in enumerate(cols):
            self.InsertColumn(col,text)
            
        # Trying to get images out...
        self._UpdateImageList()
    
        self.Bind(wx.EVT_LIST_BEGIN_LABEL_EDIT,self.OnBeginEdit)
        self.Bind(wx.EVT_LIST_END_LABEL_EDIT,self.OnEndEdit)
        self.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK,self.OnListRightClick)
        # For binding selction showing data sets
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelectionChanged)
        self.toggleshow = self.config.get_boolean('data handling', 
                                                  'toggle show')
            
    def SetShowToggle(self, toggle):
        '''Sets the selction type of the show. If toggle is true
        then the selection is via toggle if false via selection of
        data set only.
        '''
        self.toggleshow = bool(toggle)
        self.config.set('data handling', 'toggle show', toggle)
        

    def OnSelectionChanged(self, evt):
        if not self.toggleshow:
            indices = self._GetSelectedItems()
            indices.sort()
            if not indices == self.show_indices:
                self.data_cont.show_data(indices)
                self._UpdateData('Show data set flag toggled', 
                                 data_changed = True)
                # Forces update of list control
                self.SetItemCount(self.data_cont.get_count())
        evt.Skip()

    def OnGetItemText(self,item,col):
        return self.data_cont.get_item_text(item,col)
    
    def OnGetItemImage(self, item):
        #return self.image_list.GetBitmap(item)
        return item
        
    def _CreateBmpIcon(self, color_fit, color_data):
        '''_CreateBmpIcon(color_fit, color_data) --> bmp
        
        Creates an bmp icon for decorating the list
        '''
        bmp = wx.EmptyBitmap(16, 16)
        dc = wx.MemoryDC()
        dc.SelectObject(bmp)
        dc.SetBackground(wx.Brush(color_fit))
        dc.Clear()
        dc.SetBrush(wx.Brush(color_data))
        dc.SetPen(wx.Pen(color_data,0.0))
        #dc.DrawRectangle(3,3,11,11)
        dc.DrawCircle(8, 8, 7)
        dc.SelectObject(wx.NullBitmap)
        return bmp
    
    def SetStatusText(self, text):
        '''SetStatusText(self, text) --> None
        
        Sets the status text of the frame
        '''
        class event:
            pass
        event.text = text
        self.status_text(event)
    
    def _UpdateImageList(self):
        '''_UpdateImageList(self) --> None
        
        Updates the image list so that all items has the right icons
        '''
        self.image_list = wx.ImageList(16, 16)
        for data_color, sim_color in self.data_cont.get_colors():
            bmp = self._CreateBmpIcon(sim_color, data_color)
            self.image_list.Add(bmp)
            
        self.SetImageList(self.image_list, wx.IMAGE_LIST_SMALL)
    
    def _GetSelectedItems(self):
        ''' _GetSelectedItems(self) --> indices [list of integers]
        Function that yields a list of the currently selected items
        position in the list. In order of selction, i.e. no order.
        '''
        indices = [self.GetFirstSelected()]
        while indices[-1] != -1:
            indices.append(self.GetNextSelected(indices[-1]))

        # Remove the last will be -1
        indices.pop(-1)
        return indices
        
    def _UpdateData(self, desc, data_changed = True, new_data = False,\
            position = None, moved = False, direction_up = True,\
             deleted = False, name_change = False):
        '''
        Internal funciton to send an event to update data
        '''
        # Send an event that a new data set ahs been loaded
        evt = DataListEvent(myEVT_DATA_LIST, self.GetId(),\
        self.data_cont.get_data())
        evt.SetDataChanged(data_changed)
        evt.SetNewData(new_data)
        evt.SetNameChange()
        evt.SetDescription(desc)
        if moved:
            evt.SetDataMoved(position, direction_up)
        if deleted:
            evt.SetDataDeleted(position)
        # Process the event!
        self.GetEventHandler().ProcessEvent(evt)
    
    def _CheckSelected(self, indices):
        '''_CheckSelected(self, indices) --> bool
        Checks so at least data sets are selcted, otherwise show a dialog box 
        and return False
        '''
        # Check so that one dataset is selected
        if len(indices) == 0:
            dlg = wx.MessageDialog(self, \
                'At least one data set has to be selected'
                , caption = 'Information', style = wx.OK|wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            return False
        return True
        
    def DeleteItem(self):
        # Count the number of selected items
        index = self.GetFirstSelected()
        count = 0
        while index != -1:
            count += 1
            index = self.GetNextSelected(index)
        
        # Create the dialog box        
        dlg = wx.MessageDialog(self, 'Remove %d dataset(s) ?'%(count), 
        caption = 'Remove?', style = wx.YES_NO|wx.ICON_QUESTION)

        # Show the dialog box
        if dlg.ShowModal() == wx.ID_YES:
            #Get selected items
            indices = self._GetSelectedItems()
            #Sort the list in descending order, this maintains the 
            # indices in the list
            indices.sort()
            indices.reverse()
            [self.data_cont.delete_item(index) for index in indices]
            self._UpdateImageList()
            # Update the list
            self.SetItemCount(self.data_cont.get_count())
            # Send update event
            self._UpdateData('Data Deleted', deleted = True,\
                position = indices)

        dlg.Destroy()
        
        
    def AddItem(self):
        self.data_cont.add_item()
        self._UpdateImageList()
        self.SetItemCount(self.data_cont.get_count())
        self._UpdateData('Item added', data_changed = True, new_data = True)
        
        
    def MoveItemUp(self):
        # Get selected items
        indices = self._GetSelectedItems()
        # Sort them in ascending order
        indices.sort()
        # Move only if all elements can be moved!
        if indices[0] != 0:
            # Move the items in the DataSet
            [self.data_cont.move_up(index) for index in indices]
            # Deselect the currently selected items
            [self.Select(index,0) for index in indices]
            # Select the new items/positions
            [self.Select(index-1,1) for index in indices]
            self._UpdateImageList()
            # Update the list
            self.SetItemCount(self.data_cont.get_count())
            self._UpdateData('Item moved', data_changed = False, moved = True,\
                direction_up = True, position = index)
            
        else:
            dlg = wx.MessageDialog(self, \
                'The first dataset can not be moved up'
                , caption = 'Information', style = wx.OK|wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            
    def MoveItemDown(self):
        # Get selected items
        indices = self._GetSelectedItems()
        # Sort them in ascending order
        indices.sort()
        indices.reverse()
        # Move only if all elements can be moved!
        if indices[0] != self.data_cont.get_count()-1:
            # Move the items in the DataSet
            [self.data_cont.move_down(index) for index in indices]
            # Deselect the currently selected items
            [self.Select(index,0) for index in indices]
            # Select the new items/positions
            [self.Select(index+1,1) for index in indices]
            self._UpdateImageList()
            # Update the list
            self.SetItemCount(self.data_cont.get_count())
            self._UpdateData('Item moved', data_changed = False, moved = True,\
                direction_up = False, position = index)
            
        else:
            dlg = wx.MessageDialog(self, \
                'The last dataset can not be moved down',\
                caption = 'Information', style = wx.OK|wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            
    def Old_LoadData(self):
        # Keep this one if I need to go back...
        # check so only one item is checked
        n_selected = len(self._GetSelectedItems())
        if n_selected == 1:
            dlg = wx.FileDialog(self, message="Choose your Datafile"
                    , defaultFile="", wildcard="All files (*.*)|*.*"
                    , style=wx.OPEN | wx.CHANGE_DIR)
                    
            if dlg.ShowModal() == wx.ID_OK:
                self.data_cont.load(self.GetFirstSelected(), dlg.GetPath())
                self._UpdateData('New data added')
            dlg.Destroy()
        else:
            if n_selected > 1:
                dlg = wx.MessageDialog(self, 'Please select only one dataset'
                , caption = 'Too many selections'
                , style = wx.OK|wx.ICON_INFORMATION)
            else:
                dlg = wx.MessageDialog(self, 'Please select a dataset'
                , caption = 'No active dataset'
                , style = wx.OK|wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            
    def LoadData(self):
        '''LoadData(self, evt) --> None
        
        Loads data into the the model
        '''
        self.data_loader.SetData(self.data_cont.get_data())
        if self.data_loader.LoadDataFile(self._GetSelectedItems()):
            self._UpdateData('New data added', new_data = True)
            
    def ChangeDataLoader(self):
        '''ChangeDataLoader(self, evt) --> None
        
        To show the DataLoader dialog box.
        '''
        self.data_loader_cont.ShowDialog()

    def OnNewModel(self, evt):
        '''
        OnNewModel(self, evt) --> None
        
        Callback for updating the data when a new model has been loaded.
        '''
        # Set the data in the data_cont.
        self.data_cont.set_data(evt.GetModel().get_data())
        self._UpdateImageList()
        self.SetItemCount(self.data_cont.get_count())
        self._UpdateData('Data from model loaded', data_changed = True,\
                new_data = True)
        self.toggleshow = self.config.get_boolean('data handling', 
                                                  'toggle show')
        self.data_loader_cont.load_default()
        #print "new data from model loaded"
        
    def OnBeginEdit(self,evt):
        #print (evt.GetIndex(),evt.GetColumn())
        #print evt.GetText()        
        evt.Skip()
        
    def OnEndEdit(self, evt):
        if not evt.IsEditCancelled():
            self.data_cont.set_name(evt.GetIndex(),evt.GetLabel())
        self._UpdateData('Data set name changed', data_changed = False,\
            name_change = True)
        evt.Skip()
    
    def OnPlotSettings(self, evt):
        '''OnPlotSettings(self, evt) --> None
        Eventhandler for the Plot Settings dialog popup. Handles updating
        of the data from the listbox. Special care is taken for multiple 
        selections
        '''
        indices = self._GetSelectedItems()
        
        if not self._CheckSelected(indices):
            return None

        (sim_list, data_list) = self.data_cont.get_items_plotsettings(indices)
        # Find which values are the same for all lists in sim and data.
        # Note that the lists are treated seperately...
        sim_par = sim_list[0].copy()
        data_par = data_list[0].copy()
        keys = sim_par.keys()
        for sim_dict, data_dict in zip(sim_list[1:],data_list[1:]):
            # Iterate through the keys and mark the one that are
            # not identical with None!
            for key in keys:
                if not sim_dict[key] == sim_par[key]:
                    sim_par[key] = None
                if not data_dict[key] == data_par[key]:
                    data_par[key] = None

        def apply_plotsettings(sim_par, data_par):
            self.data_cont.set_items_plotsettings(indices,\
                    [sim_par]*len(indices), [data_par]*len(indices))
            self._UpdateImageList()
            self._UpdateData('Plot settings changed', data_changed = True)
        
        # Dialog business start here
        dlg = PlotSettingsDialog(self, sim_par, data_par)
        dlg.SetApplyFunc(apply_plotsettings)
        
        if dlg.ShowModal() == wx.ID_OK:
            sim_par = dlg.GetSimPar()
            data_par = dlg.GetDataPar()
            apply_plotsettings(sim_par, data_par)
        dlg.Destroy()

    def OnShowData(self, evt):
        '''OnShowData(self, evt) --> None
        Callback for toggling the state of all selected data.
        '''
        indices = self._GetSelectedItems()
        self.data_cont.toggle_show_data(indices)
        self._UpdateData('Show data set flag toggled', data_changed = True)
        # Forces update of list control
        self.SetItemCount(self.data_cont.get_count())
            
    def OnUseData(self, evt):
        '''OnUseData(self, evt) --> None
        Callback for toggling the state of all selected data.
        '''
        indices = self._GetSelectedItems()
        self.data_cont.toggle_use_data(indices)
        self._UpdateData('Use data set flag toggled', data_changed = True)
        # Forces update of list control
        self.SetItemCount(self.data_cont.get_count())
        
    def OnUseError(self, evt):
        '''OnUseData(self, evt) --> None
        Callback for toggling the state of all selected data.
        '''
        indices = self._GetSelectedItems()
        self.data_cont.toggle_use_error(indices)
        self._UpdateData('Use error in data set toggeled', data_changed = True)
        # Forces update of list control
        self.SetItemCount(self.data_cont.get_count())
        
    def OnCalcEdit(self, evt):
        '''OnCalcEdit(self, evt) --> None
        Callback for starting the editor to edit the transformations/
        calculations on the data.
        '''
        
        indices = self._GetSelectedItems()
        
        if not self._CheckSelected(indices):
            return None
        
        # Get the commands for the selcted values
        commands = self.data_cont.get_items_commands(indices)
        # Get all commands
        all_commands = self.data_cont.get_items_commands(\
                                    range(self.data_cont.get_count()))
        all_names = self.data_cont.get_items_names()
        
        # Find which values are the same for all lists in sim and data.
        # Note that the lists are treated seperately...
        command_par = commands[0].copy()
        
        for command_dict in commands:
            # Iterate through the keys and mark the one that are
            # not identical with None!
            for key in command_dict:
                # Check if the key exist in my commmand dict
                if command_par.has_key(key):
                    # Check so the command is the same
                    if not command_dict[key] == command_par[key]:
                        command_par[key] = ''
                else:
                    # Add a new key and set it to ''
                    command_par[key] = ''
        
        #Check if we have a config file:
        if self.config:
            try:
                predef_names = self.config.get('data commands', 'names').split(';')
                cmds_x = self.config.get('data commands', 'x commands').split(';')
                cmds_y = self.config.get('data commands', 'y commands').split(';')
                cmds_e = self.config.get('data commands', 'e commands').split(';')
            except io.OptionError, e:
                ShowWarningDialog(self.parent, str(e), 'datalist.OnCalcEdit')
                predef_names = None
                predef_commands = None
            else:
                predef_commands = []
                for cmd_x, cmd_y, cmd_e in zip(cmds_x,cmds_y,cmds_e):
                    command = {'x':cmd_x, 'y':cmd_y, 'e':cmd_e}
                    predef_commands.append(command)
        else:
            predef_names = None
            predef_commands = None
        
        # Dialog business start here
        dlg = CalcDialog(self, command_par, all_names, all_commands, \
                        predef_names, predef_commands)
        
        # Some currying for the set functions
        command_tester = \
            lambda command: self.data_cont.test_commands(command, indices)
        def command_runner(command):
            result = self.data_cont.run_commands(command, indices)
            if not self.data_cont.compare_sim_y_length(indices):
                self._UpdateData('New calculation', data_changed = True,\
                    new_data = True)
            else:
                self._UpdateData('New calculation', data_changed = True)
            return result
            
        dlg.SetCommandTester(command_tester)
        dlg.SetCommandRunner(command_runner)
        dlg.ShowModal()
        dlg.Destroy()
        
    def OnImportSettings(self, evt):
        '''OnImportSettings(self, evt) --> None
        Callback to start the dialog box for the iport settings.
        '''
        self.data_loader.SettingsDialog()
        
    def OnListRightClick(self, evt):
        '''OnListRightClick(self, evt) --> None
        Callback for rightclicking on one row. Creates an popupmenu.
        '''
        #print 'On Right Click', (evt.GetIndex(),evt.GetColumn())
        menu = wx.Menu()
        check_showID = wx.NewId()
        check_fitID = wx.NewId()
        check_errorID = wx.NewId()
        calcID = wx.NewId()
        import_settingsID = wx.NewId()
        plot_settingsID = wx.NewId()
        # Create the menu
        menu = wx.Menu()
        menu.Append(check_showID, "Toggle show")
        menu.Append(check_fitID, "Toggle active")
        menu.Append(check_errorID, "Toggle errorbars")
        menu.Append(calcID, "Calculations")
        menu.Append(import_settingsID, "Import settings")
        menu.Append(plot_settingsID, "Plot settings")
        
        self.Bind(wx.EVT_MENU, self.OnShowData, id = check_showID)
        self.Bind(wx.EVT_MENU, self.OnUseData, id = check_fitID)
        self.Bind(wx.EVT_MENU, self.OnUseError, id = check_errorID)
        self.Bind(wx.EVT_MENU, self.OnImportSettings, id = import_settingsID)
        self.Bind(wx.EVT_MENU, self.OnCalcEdit, id = calcID)
        self.Bind(wx.EVT_MENU, self.OnPlotSettings, id = plot_settingsID)
        
        self.PopupMenu(menu)
        menu.Destroy()  
#END: VirtualDataList
#==============================================================================

class DataListControl(wx.Panel):
    '''
    The Control window for the whole Data list including a small toolbar
    '''
    def __init__(self, parent, id=-1, config = None, status_text = None):
        wx.Panel.__init__(self,parent)
        # The two major windows:
        self.tool_panel=wx.Panel(self)
        mydata=data.DataList()
        self.data_cont=DataController(mydata)
        self.list_ctrl=VirtualDataList(self, self.data_cont, config = config,\
            status_text = status_text)
        
        self.sizer_vert=wx.BoxSizer(wx.VERTICAL)
        self.sizer_hor=wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.sizer_vert)
        
        self.do_toolbar()
        self.sizer_vert.Add(self.tool_panel, proportion = 0, flag = wx.EXPAND
        , border = 5)
        self.sizer_vert.Add((-1,2))
        self.sizer_vert.Add(self.list_ctrl, proportion = 1, flag = wx.EXPAND
        , border = 5)
        
        self.tool_panel.SetSizer(self.sizer_hor)
        
        
        #self.sizer_vert.Fit(self)
        
    def do_toolbar(self):
        if os.name == 'nt':
            size = (24, 24)
        else:
            size = (-1, -1)
        self.bitmap_button_open = wx.BitmapButton(self.tool_panel, -1
        , img.getopen_smallBitmap(), size = size, style = wx.NO_BORDER)
        self.bitmap_button_open.SetToolTipString('Import a data set')
        self.bitmap_button_add = wx.BitmapButton(self.tool_panel, -1
        , img.getaddBitmap(), size = size, style = wx.NO_BORDER)
        self.bitmap_button_add.SetToolTipString('Add a new data set')
        self.bitmap_button_delete = wx.BitmapButton(self.tool_panel, -1
        , img.getdeleteBitmap(), size = size, style = wx.NO_BORDER)
        self.bitmap_button_delete.SetToolTipString('Delete a data set')
        self.bitmap_button_move_up = wx.BitmapButton(self.tool_panel, -1
        , img.getmove_upBitmap(), size = size, style = wx.NO_BORDER)
        self.bitmap_button_move_up.SetToolTipString('Move up')
        self.bitmap_button_move_down = wx.BitmapButton(self.tool_panel, -1
        , img.getmove_downBitmap(), size = size, style = wx.NO_BORDER)
        self.bitmap_button_move_down.SetToolTipString('Move down')
        self.bitmap_button_plotting = wx.BitmapButton(self.tool_panel, -1
        , img.getplottingBitmap(), size = size, style = wx.NO_BORDER)
        self.bitmap_button_open.SetToolTipString('Plot settings')
        self.bitmap_button_calc = wx.BitmapButton(self.tool_panel, -1
        , img.getcalcBitmap(), size = size, style = wx.NO_BORDER)
        self.bitmap_button_open.SetToolTipString('Data Calculations')
        
        space = (2, -1)
        self.sizer_hor.Add(self.bitmap_button_open, proportion = 0,
                           border = 2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_add,proportion = 0, border = 2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_delete,proportion = 0, border = 2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_move_up,proportion = 0, border = 2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_move_down,proportion = 0, border = 2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_plotting,proportion = 0, border = 2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_calc,proportion = 0, border = 2)
        
        
        self.Bind(wx.EVT_BUTTON, self.eh_tb_open, self.bitmap_button_open)
        self.Bind(wx.EVT_BUTTON, self.eh_tb_add, self.bitmap_button_add)
        self.Bind(wx.EVT_BUTTON, self.eh_tb_delete, self.bitmap_button_delete)
        self.Bind(wx.EVT_BUTTON, self.eh_tb_move_up
        , self.bitmap_button_move_up)
        self.Bind(wx.EVT_BUTTON, self.eh_tb_move_down
        , self.bitmap_button_move_down)
        self.Bind(wx.EVT_BUTTON, self.eh_tb_plotting
        , self.bitmap_button_plotting)
        self.Bind(wx.EVT_BUTTON, self.eh_tb_calc, self.bitmap_button_calc)
        
    # Callbacks
    def eh_tb_open(self, event):
        #print "eh_tb_open not implemented yet"
        #pass
        self.list_ctrl.LoadData()
        
    def eh_tb_add(self, event):
        #print "eh_tb_add not implemented yet"
        #pass
        self.list_ctrl.AddItem()
        
        
    def eh_tb_delete(self, event):
        #print "eh_tb_delete not implemented yet"
        #pass
        self.list_ctrl.DeleteItem()

    def eh_tb_move_up(self, event):
        #print "eh_tb_move_up not implemented yet"
        #pass
        self.list_ctrl.MoveItemUp()
        
    
    def eh_tb_move_down(self, event):
        #print "eh_tb_move_down not implemented yet"
        #pass
        self.list_ctrl.MoveItemDown()

    
    def eh_tb_plotting(self, event):
        '''eh_tb_plotting(self, event) --> None
        Callback for the creation of a plotting settings dialog box
        '''
        #print "eh_tb_plotting not implemented yet"
        self.list_ctrl.OnPlotSettings(event)
    
    def eh_tb_calc(self, event):
        self.list_ctrl.OnCalcEdit(event)
        
    def eh_external_new_model(self, event):
        self.list_ctrl.OnNewModel(event)
        event.Skip()
        
    def DataLoaderSettingsDialog(self):
        self.list_ctrl.ChangeDataLoader()

# END: DataListControl
#=============================================================================

class PlotSettingsDialog(wx.Dialog):
    def __init__(self, parent, sim_pars, data_pars):
        wx.Dialog.__init__(self, parent, -1, 'Plot Settings')
        self.SetAutoLayout(True)
        
        # Just default value for apply button function
        def func(sim_par, data_par):
            pass
            
        self.apply_func = func
        
        
        # Layout
        gbs = wx.GridBagSizer(3, 6)
        
        # Do the labels first
        col_labels = ['Color', 'Line type', 'Thickness', 'Symbol', 'Size']
        row_labels = ['Simulation: ', 'Data: ']
        
        for item, index in zip(col_labels, range(len(col_labels))):
            label = wx.StaticText(self, -1, item)
            gbs.Add(label,(0, index+1),flag=wx.ALIGN_LEFT,border=5)
            
        for item, index in zip(row_labels, range(len(row_labels))):
            label = wx.StaticText(self, -1, item)
            gbs.Add(label,(index+1,0),\
                flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
        
        # The Color choosers
        # Some None checking i.e. check for not defined values
        if sim_pars['color'] == None:
            color = (255, 255, 255, 255)
        else:
            color = sim_pars['color']
        self.sim_colorbutton = csel.ColourSelect(self, -1, '', color)
        # Some None checking for data.
        if data_pars['color'] == None:
            color = (255, 255, 255, 255)
        else:
            color = data_pars['color']
        self.data_colorbutton = csel.ColourSelect(self, -1, '', color)
        # Add it to the grid bag sizer
        gbs.Add(self.sim_colorbutton, (1,1))
        gbs.Add(self.data_colorbutton, (2,1))
        
        # The Choics boxes for line type
        self.line_type = ['', '-', ':', '--', '.-', None]
        line_type = ['No line','full', 'dotted', 'dashed', 'dash dotted', ' ']
        # Create sim choice and set the current selcetion
        self.sim_linetype_choice = wx.Choice(self, -1, choices = line_type)
        self.sim_linetype_choice.SetSelection(\
            self._get_first_match(sim_pars['linetype'], self.line_type))
        # Create data choice and set the current selcetion
        self.data_linetype_choice = wx.Choice(self, -1, choices = line_type)
        self.data_linetype_choice.SetSelection(\
            self._get_first_match(data_pars['linetype'], self.line_type))
        # Add them to the grid sizer
        gbs.Add(self.sim_linetype_choice, (1,2))
        gbs.Add(self.data_linetype_choice, (2,2))
        
        # The Spin Controls for the Line thickness
        self.sim_linethick_ctrl = wx.SpinCtrl(self, -1, "")
        self.data_linethick_ctrl = wx.SpinCtrl(self, -1, "")
        if sim_pars['linethickness'] != None:
            self.sim_linethick_ctrl.SetRange(1,20)
            self.sim_linethick_ctrl.SetValue(sim_pars['linethickness'])
        else:
            self.sim_linethick_ctrl.SetRange(-1,20)
            self.sim_linethick_ctrl.SetValue(-1)
        if data_pars['linethickness'] != None:
            self.data_linethick_ctrl.SetRange(1,20)
            self.data_linethick_ctrl.SetValue(data_pars['linethickness'])
        else:
            self.data_linethick_ctrl.SetRange(-1,20)
            self.data_linethick_ctrl.SetValue(-1)
            
        gbs.Add(self.sim_linethick_ctrl, (1,3))
        gbs.Add(self.data_linethick_ctrl, (2,3))
        
        # The Choics boxes for symbol type
        self.symbol_type = ['', 's', 'o', '.', 'd', '<', None]
        symbol_type = ['No symbol','squares', 'circles', 'dots', 'diamonds',\
         'triangle', ' ']
        # Create sim choice and set the current selcetion
        self.sim_symboltype_choice = wx.Choice(self, -1, choices = symbol_type)
        self.sim_symboltype_choice.SetSelection(\
            self._get_first_match(sim_pars['symbol'], self.symbol_type))
        # Create data choice and set the current selcetion
        self.data_symboltype_choice = wx.Choice(self, -1,\
                                                choices = symbol_type)
        self.data_symboltype_choice.SetSelection(\
            self._get_first_match(data_pars['symbol'], self.symbol_type))
        # Add them to the grid sizer
        gbs.Add(self.sim_symboltype_choice, (1,4))
        gbs.Add(self.data_symboltype_choice, (2,4))
        
        # The Spin Controls for the symbol size
        self.sim_symbolsize_ctrl = wx.SpinCtrl(self, -1, "")
        self.data_symbolsize_ctrl = wx.SpinCtrl(self, -1, "")
        if sim_pars['symbolsize'] != None:
            self.sim_symbolsize_ctrl.SetRange(1, 20)
            self.sim_symbolsize_ctrl.SetValue(sim_pars['symbolsize'])
        else:
            self.sim_symbolsize_ctrl.SetRange(1, 20)
            self.sim_symbolsize_ctrl.SetValue(-1)
        if data_pars['symbolsize'] != None:
            self.data_symbolsize_ctrl.SetRange(1, 20)
            self.data_symbolsize_ctrl.SetValue(data_pars['symbolsize'])
        else:
            self.data_symbolsize_ctrl.SetRange(0, 20)
            self.data_symbolsize_ctrl.SetValue(-1)
        gbs.Add(self.sim_symbolsize_ctrl, (1,5))
        gbs.Add(self.data_symbolsize_ctrl, (2,5))
        
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        apply_button = wx.Button(self, wx.ID_APPLY)
        apply_button.SetDefault()
        button_sizer.AddButton(apply_button)
        button_sizer.Realize()
        
        self.Bind(wx.EVT_BUTTON, self.OnApply, apply_button)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(gbs, 1, wx.GROW|wx.ALL, 10)
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)
        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(button_sizer,0, wx.ALIGN_RIGHT, 5)
        sizer.Add((-1, 4), 0, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()
    
        
    def _get_first_match(self, item, list1):
        '''_get_first_match(item, list1) --> position [int]
        Finds the first occuruence of item in list1. If not found
        returns the first (default?) item.
        '''
        position = 0
        for i in range(len(list1)):
            if list1[i] == item:
                position = i
                break
        return position
    
    def SetApplyFunc(self, func):
        '''SetApplyFunc(self, func) --> None
        
        Set the function that should be executed when the apply button
        is pressed the function should be on the form:
        func(sim_par, data_par), the *_par is dictonaries that can be passed
        to data.
        '''
        
        self.apply_func = func
        
    def OnApply(self, event):
        '''OnApply(self, event) --> None
        
        Callback for apply button.
        '''
        sim_par = self.GetSimPar()
        data_par = self.GetDataPar()
        
        self.apply_func(sim_par, data_par)
    
    def GetSimPar(self):
        '''GetSimPar(self) --> sim_par [dict]
        Returns a dictonary containing the present values of the choosen
        values for the simulation.
        '''
        # Do some checking so that None is returned if an "invalid" choice 
        # is made
        color = self.sim_colorbutton.GetColour()
        if  color == (255, 255, 255, 255):
            color = None
        symbolsize = self.sim_symbolsize_ctrl.GetValue()
        if symbolsize < 0:
            symbolsize = None
        linethickness = self.sim_linethick_ctrl.GetValue()
        if linethickness < 0:
            linethickness = None
            
        return {'color': color,\
                'symbol': \
                self.symbol_type[self.sim_symboltype_choice.GetSelection()],\
                'symbolsize': symbolsize,\
                'linetype':\
                 self.line_type[self.sim_linetype_choice.GetSelection()],\
                'linethickness': linethickness
               }
        
    def GetDataPar(self):
        '''GetdataPar(self) --> data_par [dict]
        Returns a dictonary containing the present values of the choosen
        values for the data.
        '''
        # Do some checking so that None is returned if an "invalid" choice 
        # is made
        color = self.data_colorbutton.GetColour()
        if  color == (255, 255, 255, 255):
            color = None
        symbolsize = self.sim_symbolsize_ctrl.GetValue()
        if symbolsize < 0:
            symbolsize = None
        linethickness = self.sim_linethick_ctrl.GetValue()
        if linethickness < 0:
            linethickness = None
            
        return {'color': color,\
                'symbol': \
                self.symbol_type[self.data_symboltype_choice.GetSelection()],\
                'symbolsize': self.data_symbolsize_ctrl.GetValue(),\
                'linetype':\
                 self.line_type[self.data_linetype_choice.GetSelection()],\
                'linethickness': self.data_linethick_ctrl.GetValue()
               }
# END: PlotSettingsDialog
#==============================================================================

class CalcDialog(wx.Dialog):
    def __init__(self, parent, commands, data_names, data_commands,\
                predef_commands_names = None, predef_commands = None):
        wx.Dialog.__init__(self, parent, -1, 'Data Calculations')
        self.SetAutoLayout(True)
        
        # Some initlization, shold be function inorder for full function
        # of the dialog box
        self.command_runner = None
        self.command_tester = None
        
        #self.data_list = ['Data1', 'data2']
        # Define the availabel data sets and their commands
        self.data_list = data_names
        self.data_commands = data_commands
        
        # Create a nice static box
        box_choice = wx.StaticBox(self, -1, "Import from: ")
        box_choice_sizer = wx.StaticBoxSizer(box_choice, wx.HORIZONTAL)
        
        # Layout for some of the controlboxes 
        choice_gbs = wx.GridBagSizer(1, 4)
        box_choice_sizer.Add(choice_gbs, flag = wx.ALIGN_CENTER, border = 5)
        col_labels = ['  Predefined: ', ' Data set: ']
            
        for item, index in zip(col_labels, range(len(col_labels))):
            label = wx.StaticText(self, -1, item)
            choice_gbs.Add(label,(0, 2*index),\
                flag = wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL, border = 5)
        
        
        # Make the choice boxes we want to have:
        
        # check wheter or not the user has put any thing for predefined 
        # commands
        if predef_commands and predef_commands_names:
            self.predef_list = predef_commands_names
            self.predef_commands = predef_commands
        else:
            self.predef_list = ['Example', 'Default']
            self.predef_commands = [{'x':'x*2','y':'y/1000.0','e':'e/1000'},\
                {'x':'x','y':'y','e':'e'}]
                
        self.predef_choice = wx.Choice(self, 1, choices = self.predef_list)
        
        self.data_choice = wx.Choice(self, 1, choices = self.data_list)
        # Add them to the sizer
        choice_gbs.Add(self.predef_choice, (0,1))
        choice_gbs.Add(self.data_choice, (0,3))
        # Bind event to the choice boxes
        self.Bind(wx.EVT_CHOICE, self.OnPredefChoice, self.predef_choice)
        self.Bind(wx.EVT_CHOICE, self.OnDataChoice, self.data_choice)
        
        
        
        
        
        # Layout for the command controls
        gbs = wx.GridBagSizer(len(commands), 2)
        
        # Do the labels first
        command_names_standard = ['x', 'y', 'e']
        # We should for simplicity and layout beuty treat x,y,e seperate from
        #the rest
        self.command_ctrl = {}
        for name, index in zip(command_names_standard,\
                                    range(len(command_names_standard))):
            if commands.has_key(name):
                label = wx.StaticText(self, -1, '%s = '%name)
                gbs.Add(label,(index,0),\
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
                self.command_ctrl[name] = wx.TextCtrl(self, -1,\
                                    commands[name], size=(300, -1))
                gbs.Add(self.command_ctrl[name], (index, 1), flag = wx.EXPAND)

        command_names = commands.keys()
        command_names.sort()
        #index_offset = len(command_names_standard) - 1
        #for name, index in zip(command_names, range(len(command_names))):
        index_offset = len(command_names_standard)
        index = 0
        for name in command_names:
            if not (name in command_names_standard):
                label = wx.StaticText(self, -1, '%s = '%name)
                gbs.Add(label,(index + index_offset, 0),\
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
                self.command_ctrl[name] = wx.TextCtrl(self, -1,\
                        commands[name], size=(300, -1))
                gbs.Add(self.command_ctrl[name], (index + index_offset, 1),\
                            flag = wx.EXPAND)
                index += 1
        
        # Add the Dilaog buttons
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
        self.Bind(wx.EVT_BUTTON, self.OnClickExecute, okay_button)
        self.Bind(wx.EVT_BUTTON, self.OnClickExecute, apply_button)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add((-1, 10), 0, wx.EXPAND)
        sizer.Add(box_choice_sizer, 0, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL, 20)
        sizer.Add(gbs, 1, wx.GROW|wx.ALL|wx.EXPAND, 20)
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL|wx.TOP, 20)
        
        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(button_sizer,0,\
                flag = wx.ALIGN_RIGHT, border = 20)
        sizer.Add((-1, 4), 0, wx.EXPAND)
        self.SetSizer(sizer)
        
        sizer.Fit(self)
        self.Layout()
        
    def SetCommandRunner(self, function):
        self.command_runner = function
    
    def SetCommandTester(self, function):
        self.command_tester = function
        
    def OnPredefChoice(self, event):
        '''OnPredefChoice(self, event) --> None
        Callback for the Predefined choice box
        '''
        item = self.predef_choice.GetSelection()
        self.command_ctrl['x'].SetValue(self.predef_commands[item]['x'])
        self.command_ctrl['y'].SetValue(self.predef_commands[item]['y'])
        self.command_ctrl['e'].SetValue(self.predef_commands[item]['e'])
        
        
    def OnDataChoice(self, event):
        ''' OnDataChoice(self, event) --> None
        Callback for the data coiche box
        '''
        item = self.data_choice.GetSelection()
        failed = []
        for name in self.command_ctrl:
            val = self.command_ctrl[name].GetValue()
            try:
                val = self.data_commands[item][name]
            except KeyError:
                failed.append(name)
            self.command_ctrl[name].SetValue(val)
        if len(failed) > 0:
            dlg = wx.MessageDialog(self, 'The data operations for the' + \
                'following memebers of the data set could not be copied: '+
                ' ,'.join(failed),
                               'Copy failed',
                               wx.OK | wx.ICON_WARNING
                               )
            dlg.ShowModal()
            dlg.Destroy()
        #self.command_ctrl['x'].SetValue(self.data_commands[item]['x'])
        #self.command_ctrl['y'].SetValue(self.data_commands[item]['y'])
        #self.command_ctrl['e'].SetValue(self.data_commands[item]['e'])
        
    def OnClickExecute(self, event):
        #current_command = {'x':  self.xcommand_ctrl.GetValue(),\
        #                    'y':  self.ycommand_ctrl.GetValue(), \
        #                    'e':  self.ecommand_ctrl.GetValue() }
        current_command = {}
        for name in self.command_ctrl:
            current_command[name] = self.command_ctrl[name].GetValue()
        
        if self.command_tester and self. command_runner: 
            result = self.command_tester(current_command)
            if result == '':
                result = self.command_runner(current_command)
                if result != '':
                    result = 'There is an error that the command tester did' + \
                     ' not catch please give the following information to' + \
                     ' the developer:\n\n' +  result
                    dlg = wx.MessageDialog(self, result, 'Error in GenX',
                               wx.OK | wx.ICON_ERROR)
                    dlg.ShowModal()
                    dlg.Destroy()
            else:
                result = 'There is an error in the typed expression.\n' + \
                 result
                dlg = wx.MessageDialog(self, result, 'Expression not correct',
                               wx.OK | wx.ICON_WARNING)
                dlg.ShowModal()
                dlg.Destroy()
                
        event.Skip()
        

# END: CalcDialog
#==============================================================================

def ShowWarningDialog(frame, message, position = ''):
    dlg = wx.MessageDialog(frame, message + '\n' + 'Position: ' + position,
                               'WARNING',
                               wx.OK | wx.ICON_WARNING
                               )
    dlg.ShowModal()
    dlg.Destroy()
    
#==============================================================================

# Test code for the class to be able to independly test the code
if __name__=='__main__':
    import data
    
    class MainFrame(wx.Frame):
        def __init__(self,*args,**kwds):
            kwds["style"] = wx.DEFAULT_FRAME_STYLE
            wx.Frame.__init__(self, *args, **kwds)
            mydata=data.DataList()
            mydata.add_new()
            data_cont=DataController(mydata)
            datalist=DataListControl(self,data_cont)

    class MyApp(wx.App):
        def OnInit(self):
            wx.InitAllImageHandlers()
            main_frame = MainFrame(None, -1, "")
            self.SetTopWindow(main_frame)
            main_frame.Show()
            return 1
            
    
    app = MyApp(0)
    app.MainLoop()
