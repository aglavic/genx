"""
Library for GUI+interface layer for the data class. 
Implements one Controller and a customiized ListController
for data. The class that should be used for the outside world
is the DataListController. This has a small toolbar ontop.
"""

from dataclasses import dataclass

import wx
import wx.lib.colourselect as csel
import wx.lib.intctrl as intctrl
import wx.lib.scrolledpanel as scrolled

from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin

try:
    from wx import wizard
except ImportError:
    from wx import adv as wizard

from .. import data
from ..core.config import BaseConfig, Configurable
from ..core.custom_logging import iprint
from ..plugins import data_loader_wx as dlf
from . import images as img
from .custom_events import data_list_type, update_plotsettings
from .message_dialogs import ShowErrorDialog, ShowNotificationDialog, ShowQuestionDialog, ShowWarningDialog
from .metadata_dialog import MetaDataDialog

# ==============================================================================


class DataController:
    """
    Interface layer class between the VirtualDataList and the DataList class
    """

    data: data.DataList

    def __init__(self, data_list):
        self.data = data_list

    def get_data(self):
        return self.data

    def get_column_headers(self):
        return ["Name", "Show", "Use", "Errors"]

    def get_count(self):
        return self.data.get_len()

    def has_data(self, index):
        if index < 0 or index >= self.get_count():
            return False
        return self.data[index].has_data()

    def get_item_text(self, item, col):
        bool_output = {True: "Yes", False: "No"}
        if col == 0:
            return self.data.get_name(item)
        if col == 1:
            return bool_output[self.data[item].show]
        if col == 2:
            return bool_output[self.data.get_use(item)]
        if col == 3:
            return bool_output[self.data.get_use_error(item)]
        # if col == 3:
        #    return '(%i,%i,%i)'%self.data.get_cols(item)

        else:
            return ""

    def set_data(self, data):
        self.data = data

    def set_name(self, pos, name):
        self.data.set_name(pos, name)

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
            colors.append(
                (
                    (int(dc[0] * 255), int(dc[1] * 255), int(dc[2] * 255)),
                    (int(sc[0] * 255), int(sc[1] * 255), int(sc[2] * 255)),
                )
            )
        return colors

    def get_items_plotsettings(self, pos):
        """get_items_plotsettings(self, pos) --> (sim_list, data_list)
        Used as an interface between the DataList and GUI eventhandler
        for the PlotSettings dialog.
        returns a two lists of dictonaries for the plot settings.
        """
        sim_list = [self.data[i].get_sim_plot_items() for i in pos]
        data_list = [self.data[i].get_data_plot_items() for i in pos]
        return sim_list, data_list

    def test_commands(self, command, pos):
        """test_commands(self, pos) --> result string
        Function to test the commands to check for errors in them.
        I.e. safe execution. If the string is empty everything is ok other
        wise the string will contain information about the FIRST error occured.
        """
        for i in pos:
            result = self.data[i].try_commands(command)
            if result != "":
                break

        return result

    def run_commands(self, command, pos):
        """run_commands(self, commnd, pos) --> string
        Function that runs the command [dict] for items in position pos. The
        string contains information if something went wrong. Should be impossible.
        """
        result = ""
        for i in pos:
            try:
                self.data[i].set_commands(command)
                self.data[i].run_command()
            except Exception as e:
                result += "Error occured for data set %i: " % i + e.__str__()
                break
        return result

    def compare_sim_y_length(self, pos):
        """compare_sim_y_length(self, pos) --> bool

        Method that compares the length of the simulation and y data to see
        if they are the same pos is the position that should be compared [list]
        """
        result = True
        for index in pos:
            if self.data[index].y.shape != self.data[index].y_sim.shape:
                result = False
                break

        return result

    def get_items_commands(self, pos):
        """get_items_commands(self, pos) --> list of dicts
        Returns a list of dictonaries with keys x,y,z containing strings
        of commands which can be executed. pos is a list of integer.
        """
        command_list = [self.data[i].get_commands() for i in pos]
        return command_list

    def get_items_names(self):
        """get_items_commands(self, pos) --> list of strings
        Returns a list of the names of the data sets.
        """
        name_list = [self.data.get_name(i) for i in range(self.get_count())]
        return name_list

    def set_items_plotsettings(self, pos, sim_list, data_list):
        """set_items_plotsettings(self, pos) --> None
        Used as an interface between the DataList and GUI eventhandler
        for the PlotSettings dialog.
        sim_list and data_list has to have the right elements. See data.py
        """
        lpos = list(range(len(sim_list)))
        [self.data[i].set_sim_plot_items(sim_list[j]) for i, j in zip(pos, lpos)]
        [self.data[i].set_data_plot_items(data_list[j]) for i, j in zip(pos, lpos)]

    def show_data(self, positions):
        """Show only data at the indices given in position
        all other should be hidden.
        """
        self.data.show_items(positions)

    def toggle_show_data(self, positions):
        """toggle_show_data(self, pos) --> None
        toggles the show value for the data elements at positions.
        positions should be an iteratable yielding integers
        , i.e. list of integers
        """
        [self.data.toggle_show(pos) for pos in positions]

    def toggle_use_data(self, positions):
        """toggle_use_data(self, pos) --> None
        toggles the use_data value for the data elements at positions.
        positions should be an iteratable yielding integers
        , i.e. list of integers
        """
        [self.data.toggle_use(pos) for pos in positions]

    def toggle_use_error(self, positions):
        """toggle_use_error(self, pos) --> None
        toggles the use_data value for the data elements at positions.
        positions should be an iteratable yielding integers
        , i.e. list of integers
        """
        [self.data.toggle_use_error(pos) for pos in positions]


# END: DataController
# ==============================================================================


class DataListEvent(wx.CommandEvent):
    """
    Event class for the data list - in order to deal with
    updating of the plots and such.
    """

    def __init__(self, evt_type, id, data):
        wx.CommandEvent.__init__(self, evt_type, id)
        self.data = data
        self.data_changed = True
        self.new_data = False
        self.new_model = False
        self.description = ""
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

    def SetDataMoved(self, position, up=True):
        self.data_moved = True
        self.position = position
        self.up = up

    def SetDataDeleted(self, position):
        self.deleted = True
        self.position = position

    def SetNewData(self, new_data=True):
        self.new_data = new_data

    def SetNewModel(self, new_model=True):
        self.new_model = new_model

    def SetDescription(self, desc):
        """
        Set a string that describes the event that has occurred
        """
        self.description = desc

    def SetNameChange(self):
        """SetNameChange(self) --> None

        Sets that a name of the has changed
        """
        self.name_change = True


@dataclass
class VDataListConfig(BaseConfig):
    section = "data handling"
    toggle_show: bool = True


@dataclass
class DataCommandConfig(BaseConfig):
    section = "data commands"
    names: str = "A Example;Default;Simulation;Sustematic Errors"
    x_commands: str = "x+33;x;arange(0.01, 6, 0.01);x"
    y_commands: str = "y/1e5;y;arange(0.01, 6, 0.01)*0;y"
    e_commands: str = "e/2.;e;arange(0.01, 6, 0.01)*0;rms(e, fpe(1.0, 0.02), 0.01*dydx())"


class DataFileDropTarget(wx.FileDropTarget):

    def __init__(self, parent):
        self.parent = parent
        wx.FileDropTarget.__init__(self)

    def OnDropFiles(self, x, y, filenames):
        first_name = filenames[0].lower()
        if first_name.endswith(".hgx") or first_name.endswith(".gx"):
            return False
        else:
            return self.parent.load_from_files(filenames)


class VirtualDataList(wx.ListCtrl, ListCtrlAutoWidthMixin, Configurable):
    """
    The ListCtrl for the data
    """

    opt: VDataListConfig

    def __init__(self, parent, data_controller: DataController, status_text: str = None):
        wx.ListCtrl.__init__(self, parent, -1, style=wx.LC_REPORT | wx.LC_VIRTUAL | wx.LC_EDIT_LABELS)
        ListCtrlAutoWidthMixin.__init__(self)
        Configurable.__init__(self)

        self.drop_target = DataFileDropTarget(self)
        self.SetDropTarget(self.drop_target)

        self.data_cont = data_controller
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
        cols = self.data_cont.get_column_headers()
        for col, text in enumerate(cols):
            self.InsertColumn(col, text)
            tw = self.GetFullTextExtent(text)
            self.SetColumnWidth(col, max(tw[0] + 4, 48))

        self.setResizeColumn(0)

        # Trying to get images out...
        self._UpdateImageList()

        self.Bind(wx.EVT_LIST_BEGIN_LABEL_EDIT, self.OnBeginEdit)
        self.Bind(wx.EVT_LIST_END_LABEL_EDIT, self.OnEndEdit)
        self.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK, self.OnListRightClick)
        # For binding selction showing data sets
        self.Bind(wx.EVT_LIST_ITEM_SELECTED, self.OnSelectionChanged)
        self.ReadConfig()

    def SetShowToggle(self, toggle):
        """Sets the selction type of the show. If toggle is true
        then the selection is via toggle if false via selection of
        data set only.
        """
        self.opt.toggle_show = bool(toggle)
        self.WriteConfig()

    def OnSelectionChanged(self, evt):
        if not self.opt.toggle_show:
            indices = self._GetSelectedItems()
            indices.sort()
            if not indices == self.show_indices:
                self.data_cont.show_data(indices)
                self._UpdateData("Show data set flag toggled", data_changed=True)
                # Forces update of list control
                self.SetItemCount(self.data_cont.get_count())
        evt.Skip()

    def OnGetItemText(self, item, col):
        return self.data_cont.get_item_text(item, col)

    def OnGetItemImage(self, item):
        # return self.image_list.GetBitmap(item)
        return item

    def _CreateBmpIcon(self, color_fit, color_data):
        """_CreateBmpIcon(color_fit, color_data) --> bmp

        Creates an bmp icon for decorating the list
        """
        color_fit = wx.Colour(*color_fit)
        color_data = wx.Colour(*color_data)
        bmp = wx.Bitmap(16, 16, depth=wx.BITMAP_SCREEN_DEPTH)
        dc = wx.MemoryDC()
        dc.SelectObject(bmp)
        dc.SetBackground(wx.Brush(color_fit))
        dc.Clear()
        dc.SetBrush(wx.Brush(color_data))
        dc.SetPen(wx.Pen(color_data))
        # dc.DrawRectangle(3,3,11,11)
        dc.DrawCircle(8, 8, 7)
        dc.SelectObject(wx.NullBitmap)
        return bmp

    def SetStatusText(self, text):
        """SetStatusText(self, text) --> None

        Sets the status text of the frame
        """

        class event:
            Skip = lambda: None

        event.text = text
        self.status_text(event)

    def _UpdateImageList(self):
        """_UpdateImageList(self) --> None

        Updates the image list so that all items has the right icons
        """
        self.image_list = wx.ImageList(16, 16)
        for data_color, sim_color in self.data_cont.get_colors():
            bmp = self._CreateBmpIcon(sim_color, data_color)
            self.image_list.Add(bmp)

        self.SetImageList(self.image_list, wx.IMAGE_LIST_SMALL)

    def _GetSelectedItems(self):
        """_GetSelectedItems(self) --> indices [list of integers]
        Function that yields a list of the currently selected items
        position in the list. In order of selction, i.e. no order.
        """
        indices = [self.GetFirstSelected()]
        while indices[-1] != -1:
            indices.append(self.GetNextSelected(indices[-1]))

        # Remove the last will be -1
        indices.pop(-1)
        return indices

    def _UpdateData(
        self,
        desc,
        data_changed=True,
        new_data=False,
        position=None,
        moved=False,
        direction_up=True,
        deleted=False,
        name_change=False,
        new_model=False,
    ):
        """
        Internal funciton to send an event to update data
        """
        # Send an event that a new data set ahs been loaded
        evt = DataListEvent(data_list_type, self.GetId(), self.data_cont.get_data())
        evt.SetDataChanged(data_changed)
        evt.SetNewData(new_data)
        evt.SetNewModel(new_model)
        if name_change:
            evt.SetNameChange()
        evt.SetDescription(desc)
        if moved:
            evt.SetDataMoved(position, direction_up)
        if deleted:
            evt.SetDataDeleted(position)
        # Process the event!
        self.GetEventHandler().ProcessEvent(evt)

    def _CheckSelected(self, indices):
        """_CheckSelected(self, indices) --> bool
        Checks so at least data sets are selcted, otherwise show a dialog box
        and return False
        """
        # Check so that one dataset is selected
        if len(indices) == 0:
            ShowNotificationDialog(self, "At least one data set has to be selected")
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
        result = ShowQuestionDialog(self, "Remove %d dataset(s) ?" % count, title="Remove?")

        # Show the dialog box
        if result:
            # Get selected items
            indices = self._GetSelectedItems()
            # Sort the list in descending order, this maintains the
            # indices in the list
            indices.sort()
            indices.reverse()
            [self.data_cont.delete_item(index) for index in indices]
            self._UpdateImageList()
            # Update the list
            self.SetItemCount(self.data_cont.get_count())
            # Send update event
            self._UpdateData("Data Deleted", deleted=True, position=indices)

    def AddItem(self):
        self.data_cont.add_item()
        self._UpdateImageList()
        self.SetItemCount(self.data_cont.get_count())
        self._UpdateData("Item added", data_changed=True, new_data=True)

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
            [self.Select(index, 0) for index in indices]
            # Select the new items/positions
            [self.Select(index - 1, 1) for index in indices]
            self._UpdateImageList()
            # Update the list
            self.SetItemCount(self.data_cont.get_count())
            self._UpdateData("Item moved", data_changed=False, moved=True, direction_up=True, position=indices)

        else:
            ShowNotificationDialog(self, "The first dataset can not be moved up")

    def MoveItemDown(self):
        # Get selected items
        indices = self._GetSelectedItems()
        # Sort them in ascending order
        indices.sort()
        indices.reverse()
        # Move only if all elements can be moved!
        if indices[0] != self.data_cont.get_count() - 1:
            # Move the items in the DataSet
            [self.data_cont.move_down(index) for index in indices]
            # Deselect the currently selected items
            [self.Select(index, 0) for index in indices]
            # Select the new items/positions
            [self.Select(index + 1, 1) for index in indices]
            self._UpdateImageList()
            # Update the list
            self.SetItemCount(self.data_cont.get_count())
            self._UpdateData("Item moved", data_changed=False, moved=True, direction_up=False, position=indices)

        else:
            ShowNotificationDialog(
                self,
                "The last dataset can not be moved down",
            )

    def LoadData(self):
        """LoadData(self, evt) --> None

        Loads data into the the model
        """
        self.data_loader.SetData(self.data_cont.get_data())
        if self.data_loader.LoadDataFile(self._GetSelectedItems()):
            self._UpdateData("New data added", new_data=True)

    def load_from_files(self, files, do_update=True):
        offset = self.data_cont.get_count()
        while offset > 0 and not self.data_cont.has_data(offset - 1):
            offset -= 1
        i = 0
        for fi in files:
            # load all datasets in file, but limit to 25 to prohibit accidental load of huge datasets
            for di in range(min(self.data_loader.CountDatasets(fi), 25)):
                if self.data_cont.get_count() < (i + offset + 1):
                    self.data_cont.add_item()
                    self._UpdateImageList()
                    self.SetItemCount(self.data_cont.get_count())
                self.data_loader.LoadDataset(self.data_cont.get_data()[i + offset], fi, data_id=di)
                i += 1

        if do_update:
            # In case the dataset name has changed
            self.data_loader.UpdateDataList()
            # Send an update that new data has been loaded
            self.data_loader.SendUpdateDataEvent()
            self._UpdateData("Item added", data_changed=True, new_data=True)
        return True

    def ShowInfo(self):
        """
        Show a dialog with the dataset meta dictionary information.
        """
        sel = self._GetSelectedItems()
        if len(sel) == 0:
            selidx = -1
        else:
            selidx = sel[0]

        dia = MetaDataDialog(self, self.data_cont.get_data(), selidx)
        dia.ShowModal()
        dia.Destroy()

    def CreateSimData(self):
        """Create Simulation data through a wizard

        :return:
        """
        wiz = CreateSimDataWizard(self)
        if wiz.run():
            xstr, ystr, names = wiz.GetValues()
            for name in names:
                self.data_cont.add_item()
                self.data_cont.set_name(-1, name)
                self.data_cont.run_commands({"x": xstr, "y": ystr, "e": ystr}, [-1])
                self._UpdateImageList()
                self.SetItemCount(self.data_cont.get_count())
                self._UpdateData("Item added", data_changed=True, new_data=True)

    def ChangeDataLoader(self):
        """ChangeDataLoader(self, evt) --> None

        To show the DataLoader dialog box.
        """
        self.data_loader_cont.ShowDialog()

    def OnNewModel(self, evt):
        """
        OnNewModel(self, evt) --> None

        Callback for updating the data when a new model has been loaded.
        """
        # Set the data in the data_cont.
        self.data_cont.set_data(evt.GetModel().get_data())
        self._UpdateImageList()
        self.SetItemCount(self.data_cont.get_count())
        self._UpdateData("Data from model loaded", data_changed=True, new_data=True, new_model=True)
        self.ReadConfig()
        self.data_loader_cont.load_default()
        # print "new data from model loaded"

    def OnBeginEdit(self, evt):
        evt.Skip()

    def OnEndEdit(self, evt):
        if not evt.IsEditCancelled():
            self.data_cont.set_name(evt.GetIndex(), evt.GetLabel())
        self._UpdateData("Data set name changed", data_changed=False, name_change=True)
        evt.Skip()

    def OnPlotSettings(self, evt):
        """OnPlotSettings(self, evt) --> None
        Eventhandler for the Plot Settings dialog popup. Handles updating
        of the data from the listbox. Special care is taken for multiple
        selections
        """
        indices = self._GetSelectedItems()

        if not self._CheckSelected(indices):
            return None

        (sim_list, data_list) = self.data_cont.get_items_plotsettings(indices)
        # Find which values are the same for all lists in sim and data.
        # Note that the lists are treated seperately...
        sim_par = sim_list[0].copy()
        data_par = data_list[0].copy()
        keys = list(sim_par.keys())
        for sim_dict, data_dict in zip(sim_list[1:], data_list[1:]):
            # Iterate through the keys and mark the one that are
            # not identical with None!
            for key in keys:
                if not sim_dict[key] == sim_par[key]:
                    sim_par[key] = None
                if not data_dict[key] == data_par[key]:
                    data_par[key] = None

        def apply_plotsettings(sim_par, data_par):
            evt = update_plotsettings(indices=indices, sim_par=sim_par, data_par=data_par)
            wx.PostEvent(self.parent, evt)

        # Dialog business start here
        dlg = PlotSettingsDialog(self, sim_par, data_par)
        dlg.SetApplyFunc(apply_plotsettings)

        if dlg.ShowModal() == wx.ID_OK:
            sim_par = dlg.GetSimPar()
            data_par = dlg.GetDataPar()
            apply_plotsettings(sim_par, data_par)
        dlg.Destroy()

    def OnShowData(self, evt):
        """OnShowData(self, evt) --> None
        Callback for toggling the state of all selected data.
        """
        indices = self._GetSelectedItems()
        self.data_cont.toggle_show_data(indices)
        self._UpdateData("Show data set flag toggled", data_changed=True)
        # Forces update of list control
        self.SetItemCount(self.data_cont.get_count())

    def OnUseData(self, evt):
        """OnUseData(self, evt) --> None
        Callback for toggling the state of all selected data.
        """
        indices = self._GetSelectedItems()
        self.data_cont.toggle_use_data(indices)
        self._UpdateData("Use data set flag toggled", data_changed=True)
        # Forces update of list control
        self.SetItemCount(self.data_cont.get_count())

    def OnUseError(self, evt):
        """OnUseData(self, evt) --> None
        Callback for toggling the state of all selected data.
        """
        indices = self._GetSelectedItems()
        self.data_cont.toggle_use_error(indices)
        self._UpdateData("Use error in data set toggeled", data_changed=True)
        # Forces update of list control
        self.SetItemCount(self.data_cont.get_count())

    def OnCalcEdit(self, evt):
        """OnCalcEdit(self, evt) --> None
        Callback for starting the editor to edit the transformations/
        calculations on the data.
        """

        indices = self._GetSelectedItems()

        if not self._CheckSelected(indices):
            return None

        # Get the commands for the selcted values
        commands = self.data_cont.get_items_commands(indices)
        # Get all commands
        all_commands = self.data_cont.get_items_commands(list(range(self.data_cont.get_count())))
        all_names = self.data_cont.get_items_names()

        # Find which values are the same for all lists in sim and data.
        # Note that the lists are treated seperately...
        command_par = commands[0].copy()

        for command_dict in commands:
            # Iterate through the keys and mark the one that are
            # not identical with None!
            for key in command_dict:
                # Check if the key exist in my commmand dict
                if key in command_par:
                    # Check so the command is the same
                    if not command_dict[key] == command_par[key]:
                        command_par[key] = ""
                else:
                    # Add a new key and set it to ''
                    command_par[key] = ""

        # Read commands from config
        dcfg = DataCommandConfig()
        dcfg.load_config()
        predef_names = dcfg.names.split(";")
        cmds_x = dcfg.x_commands.split(";")
        cmds_y = dcfg.y_commands.split(";")
        cmds_e = dcfg.e_commands.split(";")

        predef_commands = [{"x": cmd_x, "y": cmd_y, "e": cmd_e} for cmd_x, cmd_y, cmd_e in zip(cmds_x, cmds_y, cmds_e)]

        # Dialog business start here
        dlg = CalcDialog(self, command_par, all_names, all_commands, predef_names, predef_commands)

        # Some currying for the set functions
        command_tester = lambda command: self.data_cont.test_commands(command, indices)

        def command_runner(command):
            result = self.data_cont.run_commands(command, indices)
            if not self.data_cont.compare_sim_y_length(indices):
                self._UpdateData("New calculation", data_changed=True, new_data=True)
            else:
                self._UpdateData("New calculation", data_changed=True)
            return result

        dlg.SetCommandTester(command_tester)
        dlg.SetCommandRunner(command_runner)
        dlg.ShowModal()
        dlg.Destroy()

    def OnImportSettings(self, evt):
        """Callback to start the dialog box for the import settings."""
        self.data_loader.SettingsDialog()

    def OnListRightClick(self, evt):
        """Callback for right clicking on one row. Creates an popup menu."""
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

        self.Bind(wx.EVT_MENU, self.OnShowData, id=check_showID)
        self.Bind(wx.EVT_MENU, self.OnUseData, id=check_fitID)
        self.Bind(wx.EVT_MENU, self.OnUseError, id=check_errorID)
        self.Bind(wx.EVT_MENU, self.OnImportSettings, id=import_settingsID)
        self.Bind(wx.EVT_MENU, self.OnCalcEdit, id=calcID)
        self.Bind(wx.EVT_MENU, self.OnPlotSettings, id=plot_settingsID)

        self.PopupMenu(menu)
        menu.Destroy()


# END: VirtualDataList
# ==============================================================================


class DataListControl(wx.Panel):
    """
    The Control window for the whole Data list including a small toolbar
    """

    def __init__(self, parent, id=-1, status_text=None):
        wx.Panel.__init__(self, parent)
        # The two major windows:
        self.toolbar = wx.ToolBar(self, style=wx.TB_FLAT | wx.TB_HORIZONTAL)
        mydata = data.DataList()
        self.data_cont = DataController(mydata)
        self.list_ctrl = VirtualDataList(self, self.data_cont, status_text=status_text)

        self.sizer_vert = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer_vert)

        self.do_toolbar()
        self.sizer_vert.Add(self.toolbar, proportion=0, flag=wx.EXPAND, border=0)
        self.sizer_vert.Add(self.list_ctrl, proportion=1, flag=wx.EXPAND, border=0)

        self.toolbar.Realize()

    def do_toolbar(self):
        dpi_scale_factor = wx.GetApp().dpi_scale_factor
        tb_bmp_size = int(dpi_scale_factor * 20)

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Add data set",
            bitmap=wx.Bitmap(img.add.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Insert empty data set",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_add, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Import data set",
            bitmap=wx.Bitmap(img.open.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Import data into selected data set",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_open, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Add simulation data set",
            bitmap=wx.Bitmap(img.add_simulation.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Insert a data set for simulation",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_add_simulation, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Datset information",
            bitmap=wx.Bitmap(img.info.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Show the meta data information for the selected dataset",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_data_info, id=newid)

        self.toolbar.AddSeparator()

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Move up",
            bitmap=wx.Bitmap(img.move_up.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Move selected data set(s) up",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_move_up, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Move_down",
            bitmap=wx.Bitmap(img.move_down.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Move selected data set(s) down",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_move_down, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Delete data set",
            bitmap=wx.Bitmap(img.delete.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Delete selected data set",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_delete, id=newid)

        self.toolbar.AddSeparator()

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Plot settings",
            bitmap=wx.Bitmap(img.plotting.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Plot settings",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_plotting, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(
            newid,
            label="Calculate",
            bitmap=wx.Bitmap(img.calc.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
            shortHelp="Calculation on selected data set(s)",
        )
        self.Bind(wx.EVT_TOOL, self.eh_tb_calc, id=newid)

    # Callbacks
    def eh_tb_open(self, event):
        # print "eh_tb_open not implemented yet"
        # pass
        self.list_ctrl.LoadData()

    def eh_tb_add(self, event):
        # print "eh_tb_add not implemented yet"
        # pass
        self.list_ctrl.AddItem()

    def eh_tb_add_simulation(self, event):
        self.list_ctrl.CreateSimData()

    def eh_tb_data_info(self, event):
        self.list_ctrl.ShowInfo()

    def eh_tb_delete(self, event):
        # print "eh_tb_delete not implemented yet"
        # pass
        self.list_ctrl.DeleteItem()

    def eh_tb_move_up(self, event):
        # print "eh_tb_move_up not implemented yet"
        # pass
        self.list_ctrl.MoveItemUp()

    def eh_tb_move_down(self, event):
        # print "eh_tb_move_down not implemented yet"
        # pass
        self.list_ctrl.MoveItemDown()

    def eh_tb_plotting(self, event):
        """Callback for the creation of a plotting settings dialog box"""
        self.list_ctrl.OnPlotSettings(event)

    def eh_tb_calc(self, event):
        self.list_ctrl.OnCalcEdit(event)

    def eh_external_new_model(self, event):
        self.list_ctrl.OnNewModel(event)
        event.Skip()

    def DataLoaderSettingsDialog(self):
        self.list_ctrl.ChangeDataLoader()


# END: DataListControl
# =============================================================================


class PlotSettingsDialog(wx.Dialog):

    def __init__(self, parent, sim_pars, data_pars):
        wx.Dialog.__init__(self, parent, -1, "Plot Settings")
        self.SetAutoLayout(True)

        # Just default value for apply button function
        def func(sim_par, data_par):
            pass

        self.apply_func = func

        # Layout
        gbs = wx.GridBagSizer(3, 6)

        # Do the labels first
        col_labels = ["Color", "Line type", "Thickness", "Symbol", "Size"]
        row_labels = ["Simulation: ", "Data: "]

        for item, index in zip(col_labels, list(range(len(col_labels)))):
            label = wx.StaticText(self, -1, item)
            gbs.Add(label, (0, index + 1), flag=wx.ALIGN_LEFT, border=5)

        for item, index in zip(row_labels, list(range(len(row_labels)))):
            label = wx.StaticText(self, -1, item)
            gbs.Add(label, (index + 1, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)

        # The Color choosers
        # Some None checking i.e. check for not defined values
        color = tuple(map(int, sim_pars.get("color", (255, 255, 255, 255))))
        self.sim_colorbutton = csel.ColourSelect(self, -1, "", color)
        # Some None checking for data.
        color = tuple(map(int, data_pars.get("color", (255, 255, 255, 255))))
        self.data_colorbutton = csel.ColourSelect(self, -1, "", color)
        # Add it to the grid bag sizer
        gbs.Add(self.sim_colorbutton, (1, 1))
        gbs.Add(self.data_colorbutton, (2, 1))

        # The Choics boxes for line type
        self.line_type = ["", "-", ":", "--", ".-", None]
        line_type = ["No line", "full", "dotted", "dashed", "dash dotted", " "]
        # Create sim choice and set the current selcetion
        self.sim_linetype_choice = wx.Choice(self, -1, choices=line_type)
        self.sim_linetype_choice.SetSelection(self._get_first_match(sim_pars["linetype"], self.line_type))
        # Create data choice and set the current selcetion
        self.data_linetype_choice = wx.Choice(self, -1, choices=line_type)
        self.data_linetype_choice.SetSelection(self._get_first_match(data_pars["linetype"], self.line_type))
        # Add them to the grid sizer
        gbs.Add(self.sim_linetype_choice, (1, 2))
        gbs.Add(self.data_linetype_choice, (2, 2))

        # The Spin Controls for the Line thickness
        self.sim_linethick_ctrl = wx.SpinCtrl(self, -1, "")
        self.data_linethick_ctrl = wx.SpinCtrl(self, -1, "")
        if sim_pars["linethickness"] is not None:
            self.sim_linethick_ctrl.SetRange(1, 20)
            self.sim_linethick_ctrl.SetValue(sim_pars["linethickness"])
        else:
            self.sim_linethick_ctrl.SetRange(-1, 20)
            self.sim_linethick_ctrl.SetValue(-1)
        if data_pars["linethickness"] is not None:
            self.data_linethick_ctrl.SetRange(1, 20)
            self.data_linethick_ctrl.SetValue(data_pars["linethickness"])
        else:
            self.data_linethick_ctrl.SetRange(-1, 20)
            self.data_linethick_ctrl.SetValue(-1)

        gbs.Add(self.sim_linethick_ctrl, (1, 3))
        gbs.Add(self.data_linethick_ctrl, (2, 3))

        # The Choics boxes for symbol type
        self.symbol_type = ["", "s", "o", ".", "d", "<", None]
        symbol_type = ["No symbol", "squares", "circles", "dots", "diamonds", "triangle", " "]
        # Create sim choice and set the current selcetion
        self.sim_symboltype_choice = wx.Choice(self, -1, choices=symbol_type)
        self.sim_symboltype_choice.SetSelection(self._get_first_match(sim_pars["symbol"], self.symbol_type))
        # Create data choice and set the current selcetion
        self.data_symboltype_choice = wx.Choice(self, -1, choices=symbol_type)
        self.data_symboltype_choice.SetSelection(self._get_first_match(data_pars["symbol"], self.symbol_type))
        # Add them to the grid sizer
        gbs.Add(self.sim_symboltype_choice, (1, 4))
        gbs.Add(self.data_symboltype_choice, (2, 4))

        # The Spin Controls for the symbol size
        self.sim_symbolsize_ctrl = wx.SpinCtrl(self, -1, "")
        self.data_symbolsize_ctrl = wx.SpinCtrl(self, -1, "")
        if sim_pars["symbolsize"] is not None:
            self.sim_symbolsize_ctrl.SetRange(1, 20)
            self.sim_symbolsize_ctrl.SetValue(sim_pars["symbolsize"])
        else:
            self.sim_symbolsize_ctrl.SetRange(1, 20)
            self.sim_symbolsize_ctrl.SetValue(-1)
        if data_pars["symbolsize"] is not None:
            self.data_symbolsize_ctrl.SetRange(1, 20)
            self.data_symbolsize_ctrl.SetValue(data_pars["symbolsize"])
        else:
            self.data_symbolsize_ctrl.SetRange(0, 20)
            self.data_symbolsize_ctrl.SetValue(-1)
        gbs.Add(self.sim_symbolsize_ctrl, (1, 5))
        gbs.Add(self.data_symbolsize_ctrl, (2, 5))

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
        sizer.Add(gbs, 1, wx.GROW | wx.ALL, 10)
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.RIGHT | wx.TOP, 5)
        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT, 5)
        sizer.Add((-1, 4), 0, wx.EXPAND)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()

    def _get_first_match(self, item, list1):
        """_get_first_match(item, list1) --> position [int]
        Finds the first occuruence of item in list1. If not found
        returns the first (default?) item.
        """
        position = 0
        for i in range(len(list1)):
            if list1[i] == item:
                position = i
                break
        return position

    def SetApplyFunc(self, func):
        """SetApplyFunc(self, func) --> None

        Set the function that should be executed when the apply button
        is pressed the function should be on the form:
        func(sim_par, data_par), the *_par is dictonaries that can be passed
        to data.
        """

        self.apply_func = func

    def OnApply(self, event):
        """OnApply(self, event) --> None

        Callback for apply button.
        """
        sim_par = self.GetSimPar()
        data_par = self.GetDataPar()

        self.apply_func(sim_par, data_par)

    def GetSimPar(self):
        """GetSimPar(self) --> sim_par [dict]
        Returns a dictonary containing the present values of the choosen
        values for the simulation.
        """
        # Do some checking so that None is returned if an "invalid" choice
        # is made
        color = self.sim_colorbutton.GetColour()
        if color == (255, 255, 255, 255):
            color = None
        symbolsize = self.sim_symbolsize_ctrl.GetValue()
        if symbolsize < 0:
            symbolsize = None
        linethickness = self.sim_linethick_ctrl.GetValue()
        if linethickness < 0:
            linethickness = None

        return {
            "color": color,
            "symbol": self.symbol_type[self.sim_symboltype_choice.GetSelection()],
            "symbolsize": symbolsize,
            "linetype": self.line_type[self.sim_linetype_choice.GetSelection()],
            "linethickness": linethickness,
        }

    def GetDataPar(self):
        """GetdataPar(self) --> data_par [dict]
        Returns a dictonary containing the present values of the choosen
        values for the data.
        """
        # Do some checking so that None is returned if an "invalid" choice
        # is made
        color = self.data_colorbutton.GetColour()
        if color == (255, 255, 255, 255):
            color = None
        symbolsize = self.sim_symbolsize_ctrl.GetValue()
        if symbolsize < 0:
            symbolsize = None
        linethickness = self.sim_linethick_ctrl.GetValue()
        if linethickness < 0:
            linethickness = None

        return {
            "color": color,
            "symbol": self.symbol_type[self.data_symboltype_choice.GetSelection()],
            "symbolsize": self.data_symbolsize_ctrl.GetValue(),
            "linetype": self.line_type[self.data_linetype_choice.GetSelection()],
            "linethickness": self.data_linethick_ctrl.GetValue(),
        }


# END: PlotSettingsDialog
# ==============================================================================


class CalcDialog(wx.Dialog):

    def __init__(self, parent, commands, data_names, data_commands, predef_commands_names=None, predef_commands=None):
        wx.Dialog.__init__(self, parent, -1, "Data Calculations")
        self.SetAutoLayout(True)

        # Some initlization, shold be function inorder for full function
        # of the dialog box
        self.command_runner = None
        self.command_tester = None

        # self.data_list = ['Data1', 'data2']
        # Define the availabel data sets and their commands
        self.data_list = data_names
        self.data_commands = data_commands

        # Create a nice static box
        box_choice = wx.StaticBox(self, -1, "Import from: ")
        box_choice_sizer = wx.StaticBoxSizer(box_choice, wx.HORIZONTAL)

        # Layout for some of the controlboxes
        choice_gbs = wx.GridBagSizer(1, 4)
        box_choice_sizer.Add(choice_gbs, flag=wx.ALIGN_CENTER, border=5)
        col_labels = ["  Predefined: ", " Data set: "]

        for item, index in zip(col_labels, list(range(len(col_labels)))):
            label = wx.StaticText(self, -1, item)
            choice_gbs.Add(label, (0, 2 * index), flag=wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=5)

        # Make the choice boxes we want to have:

        # check wheter or not the user has put any thing for predefined
        # commands
        if predef_commands and predef_commands_names:
            self.predef_list = predef_commands_names
            self.predef_commands = predef_commands
        else:
            self.predef_list = ["Example", "Default"]
            self.predef_commands = [{"x": "x*2", "y": "y/1000.0", "e": "e/1000"}, {"x": "x", "y": "y", "e": "e"}]

        self.predef_choice = wx.Choice(self, choices=self.predef_list)

        self.data_choice = wx.Choice(self, choices=self.data_list)
        # Add them to the sizer
        choice_gbs.Add(self.predef_choice, (0, 1))
        choice_gbs.Add(self.data_choice, (0, 3))
        # Bind event to the choice boxes
        self.Bind(wx.EVT_CHOICE, self.OnPredefChoice, self.predef_choice)
        self.Bind(wx.EVT_CHOICE, self.OnDataChoice, self.data_choice)

        # Layout for the command controls
        gbs = wx.GridBagSizer(len(commands), 2)

        # Do the labels first
        command_names_standard = ["x", "y", "e"]
        # We should for simplicity and layout beuty treat x,y,e seperate from
        # the rest
        self.command_ctrl = {}
        for name, index in zip(command_names_standard, list(range(len(command_names_standard)))):
            if name in commands:
                label = wx.StaticText(self, -1, "%s = " % name)
                gbs.Add(label, (index, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
                self.command_ctrl[name] = wx.TextCtrl(self, -1, commands[name], size=(300, -1))
                gbs.Add(self.command_ctrl[name], (index, 1), flag=wx.EXPAND)

        command_names = list(commands.keys())
        command_names.sort()
        # index_offset = len(command_names_standard) - 1
        # for name, index in zip(command_names, range(len(command_names))):
        index_offset = len(command_names_standard)
        index = 0
        for name in command_names:
            if not (name in command_names_standard):
                label = wx.StaticText(self, -1, "%s = " % name)
                gbs.Add(label, (index + index_offset, 0), flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
                self.command_ctrl[name] = wx.TextCtrl(self, -1, commands[name], size=(300, -1))
                gbs.Add(self.command_ctrl[name], (index + index_offset, 1), flag=wx.EXPAND)
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
        sizer.Add(box_choice_sizer, 0, wx.GROW, 20)
        sizer.Add(gbs, 1, wx.GROW | wx.ALL | wx.EXPAND, 20)
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.TOP, 20)

        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(button_sizer, 0, flag=wx.ALIGN_RIGHT, border=20)
        sizer.Add((-1, 4), 0, wx.EXPAND)
        self.SetSizer(sizer)

        sizer.Fit(self)
        self.Layout()

    def SetCommandRunner(self, function):
        self.command_runner = function

    def SetCommandTester(self, function):
        self.command_tester = function

    def OnPredefChoice(self, event):
        """OnPredefChoice(self, event) --> None
        Callback for the Predefined choice box
        """
        item = self.predef_choice.GetSelection()
        self.command_ctrl["x"].SetValue(self.predef_commands[item]["x"])
        self.command_ctrl["y"].SetValue(self.predef_commands[item]["y"])
        self.command_ctrl["e"].SetValue(self.predef_commands[item]["e"])

    def OnDataChoice(self, event):
        """OnDataChoice(self, event) --> None
        Callback for the data coiche box
        """
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
            ShowWarningDialog(
                self,
                "The data operations for the"
                + "following memebers of the data set could not be copied: "
                + " ,".join(failed),
                "Copy failed",
            )

    def OnClickExecute(self, event):
        event.Skip()
        current_command = {}
        for name in self.command_ctrl:
            current_command[name] = self.command_ctrl[name].GetValue()

        if self.command_tester and self.command_runner:
            result = self.command_tester(current_command)
            if result == "":
                result = self.command_runner(current_command)
                if result != "":
                    result = (
                        "There is an error that the command tester did"
                        + " not catch please give the following information to"
                        + " the developer:\n\n"
                        + result
                    )
                    ShowErrorDialog(self, result, "Error in GenX")
            else:
                result = "There is an error in the typed expression.\n" + result
                ShowWarningDialog(self, result, "Expression not correct")


# END: CalcDialog
# ==============================================================================
# BEGIN: Sim data Wizard


class TitledPage(wizard.WizardPageSimple):

    def __init__(self, parent, title):
        wizard.WizardPageSimple.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        titleText = wx.StaticText(self, -1, title)
        titleText.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(titleText, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND | wx.ALL, 5)


class CreateSimDataWizard(wizard.Wizard):

    def __init__(self, parent):
        wizard.Wizard.__init__(self, parent, -1, "Create Simulation Data Sets")
        step_types = ["const", "log"]
        self.pages = []
        self.min_val = None
        self.max_val = None

        page1 = TitledPage(self, "X-values")
        page2 = TitledPage(self, "Number of data sets")
        page3 = TitledPage(self, "Data set names")

        dataSizer = wx.FlexGridSizer(rows=5, cols=2, vgap=0, hgap=10)

        dataSizer.Add(wx.StaticText(page1, -1, "Start "), 0, wx.EXPAND | wx.ALL, 5)
        minCtrl = wx.TextCtrl(page1, -1, value="0.0")
        dataSizer.Add(minCtrl, 0, wx.EXPAND | wx.ALL, 5)
        dataSizer.Add(wx.StaticText(page1, -1, "Stop "), 0, wx.EXPAND | wx.ALL, 5)
        maxCtrl = wx.TextCtrl(page1, -1, value="1.0")
        dataSizer.Add(maxCtrl, 0, wx.EXPAND | wx.ALL, 5)
        dataSizer.Add(wx.StaticText(page1, -1, "Step type"), 0, wx.EXPAND | wx.ALL, 5)
        stepChoice = wx.Choice(page1, -1, (-1, -1), choices=step_types)
        stepChoice.SetSelection(0)
        dataSizer.Add(stepChoice, 0, wx.EXPAND | wx.ALL, 5)
        dataSizer.Add(wx.StaticText(page1, -1, "Num steps"), 0, wx.EXPAND | wx.ALL, 5)
        stepCtrl = intctrl.IntCtrl(page1, value=100)
        dataSizer.Add(stepCtrl, 0, wx.EXPAND | wx.ALL, 5)
        page1.sizer.Add(dataSizer)

        dataSizer = wx.FlexGridSizer(rows=1, cols=2, vgap=0, hgap=10)
        dataSizer.Add(wx.StaticText(page2, -1, "Data sets "), 0, wx.EXPAND | wx.ALL, 5)
        setsCtrl = intctrl.IntCtrl(page2, value=1)
        dataSizer.Add(setsCtrl, 0, wx.EXPAND | wx.ALL, 5)
        page2.sizer.Add(dataSizer)

        page3.sizer.Add(wx.StaticText(page3, -1, "Change the name of the data sets"), 0, wx.EXPAND | wx.ALL, 5)
        self.scrollPanel = scrolled.ScrolledPanel(page3, -1, size=(150, 200), style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER)
        self.nameSizer = wx.FlexGridSizer(cols=1, vgap=4, hgap=0)
        self.nameCtrls = []
        page3.sizer.Add(self.scrollPanel, 0, wx.CENTER | wx.ALL | wx.EXPAND, 5)

        self.maxCtrl = maxCtrl
        self.minCtrl = minCtrl
        self.stepChoice = stepChoice
        self.stepCtrl = stepCtrl
        self.setsCtrl = setsCtrl

        self.add_page(page1)
        self.add_page(page2)
        self.add_page(page3)

        self.Bind(wizard.EVT_WIZARD_PAGE_CHANGING, self.on_page_changing)

    def on_page_changing(self, evt):
        """Event handler for changing page

        :param evt:
        :return:
        """
        if evt.GetDirection() and evt.GetPage() is self.pages[0]:
            if not self.min_max_values_valid():
                evt.Veto()
                return
        if evt.GetDirection() and evt.GetPage() is self.pages[1]:
            for item in self.nameCtrls:
                item.Destroy()
            # self.nameSizer.Destroy()
            # self.nameSizer = wx.FlexGridSizer(cols=1, vgap=4, hgap=4)
            self.nameCtrls = []
            # We are moving from page1 to page2 populate the scroll panel
            for i in range(self.setsCtrl.GetValue()):
                self.nameCtrls.append(wx.TextCtrl(self.scrollPanel, -1, "Sim%d" % i))
                self.nameSizer.Add(self.nameCtrls[-1], 0, wx.CENTRE | wx.ALL, 5)
            self.scrollPanel.SetSizer(self.nameSizer)
            # self.scrollPanel.SetAutoLayout(1)
            self.scrollPanel.SetupScrolling()

    def min_max_values_valid(self):
        """checks so that min and max values are vaild floats

        :return:
        """
        try:
            self.min_val = float(eval(self.minCtrl.GetValue()))
        except Exception:
            self.min_val = None
            ShowWarningDialog(self, "The minimum value can not be evaluated to a numerical value")
            return False
        try:
            self.max_val = float(eval(self.maxCtrl.GetValue()))
        except Exception:
            self.max_val = None
            ShowWarningDialog(self, "The minimum value can not be evaluated to a numerical value")
            return False

        if self.min_val < 1e-20 and self.stepChoice.GetStringSelection() == "log":
            ShowWarningDialog(self, "The minimum value have to be larger than 1e-20 when using log step size")
            return False

        return True

    def add_page(self, page):
        """Add a wizard page"""
        if self.pages:
            previous_page = self.pages[-1]
            page.SetPrev(previous_page)
            previous_page.SetNext(page)
        self.pages.append(page)

    def run(self):
        return self.RunWizard(self.pages[0])

    def GetValues(self):
        """Returns the values to be used for simulations.

        :return: xstr, ystr, namestrs
        """
        if self.stepChoice.GetStringSelection() == "log":
            xstr = "logspace(log10(%s), log10(%s), %d)" % (
                self.minCtrl.GetValue(),
                self.maxCtrl.GetValue(),
                self.stepCtrl.GetValue(),
            )
        else:
            xstr = "linspace(%s, %s, %d)" % (self.minCtrl.GetValue(), self.maxCtrl.GetValue(), self.stepCtrl.GetValue())

        ystr = "zeros(%d)*nan" % self.stepCtrl.GetValue()

        namestrs = []
        for ctrl in self.nameCtrls:
            namestrs.append(ctrl.GetValue())

        return xstr, ystr, namestrs
