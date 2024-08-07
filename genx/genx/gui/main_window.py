"""
Main GenX window and functionality.
"""

import _thread
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser

from copy import deepcopy
from dataclasses import dataclass
from logging import debug, info, warning
from typing import List

import appdirs
import wx
import wx.adv
import wx.grid
import wx.py
import wx.stc

from wx.lib.wordwrap import wordwrap

from ..core import config as conf_mod
from ..core.colors import COLOR_CYCLES
from ..core.custom_logging import iprint, numpy_set_options
from ..plugins import add_on_framework as add_on
from ..version import __version__ as program_version
from . import custom_ids, datalist, help
from . import images as img
from . import parametergrid, pubgraph_dialog, solvergui
from .batch_dialog import BatchDialog
from .custom_events import *
from .exception_handling import CatchModelError, GuiExceptionHandler
from .message_dialogs import ShowNotificationDialog, ShowQuestionDialog
from .online_update import VersionInfoDialog, check_version

_path = os.path.dirname(__file__)
if _path[-4:] == ".zip":
    _path, ending = os.path.split(_path)

# Get the configuration path, create if it not exists
config_path = os.path.abspath(appdirs.user_data_dir("GenX3", "ArturGlavic"))
profile_dest = os.path.abspath(os.path.join(config_path, "profiles"))
profile_src = os.path.abspath(os.path.join(_path, "..", "profiles"))
info(
    f"Paths are:\n        config_path={config_path}\n       profile_dest={profile_dest}\n        profile_src{profile_src}"
)
version_file = os.path.join(config_path, "genx.version")
if not os.path.exists(config_path):
    info(f"Creating path: {config_path}")
    os.makedirs(config_path)
if not os.path.exists(profile_dest):
    info(f"Creating path: {profile_dest}")
    shutil.copytree(profile_src, profile_dest)
    open(version_file, "w").write(program_version + "\n")
elif (
    not os.path.exists(version_file)
    or open(version_file, "r").read().rsplit(".", 1)[0] != program_version.rsplit(".", 1)[0]
):
    # update profiles if major version does not match
    info("Update profiles to default for GenX " + program_version)
    from glob import glob

    for fi in glob(os.path.join(profile_src, "*.conf")):
        info(f"    copy {fi} to {profile_dest}")
        shutil.copy2(fi, profile_dest)
    open(version_file, "w").write(program_version + "\n")
if not os.path.exists(os.path.join(config_path, "genx.conf")):
    info(f"Creating genx.conf at {config_path} from default.profile in {profile_src}")
    shutil.copyfile(os.path.join(profile_src, "default.profile"), os.path.join(config_path, "genx.conf"))

manual_url = "https://aglavic.github.io/genx/doc/"
homepage_url = "https://aglavic.github.io/genx/"


@dataclass
class GUIConfig(conf_mod.BaseConfig):
    section = "gui"
    hsize: int = 1200  # stores the width of the window
    vsize: int = 800  # stores the height of the window
    vsplit: int = 300
    hsplit: int = 400
    psplit: int = 550
    solver_update_time: float = 1.5
    editor: str = None
    last_update_check: float = 0.0


@dataclass
class WindowStartup(conf_mod.BaseConfig):
    section = "startup"
    show_profiles: bool = True
    widescreen: bool = False
    wx_plotting: bool = False


class GenxMainWindow(wx.Frame, conf_mod.Configurable):
    opt: GUIConfig
    script_file: str = None

    def __init__(self, parent: wx.App, dpi_overwrite=None):
        self._init_phase = True
        self.parent = parent
        debug("starting setup of MainFrame")
        conf_mod.Configurable.__init__(self)
        self.wstartup = WindowStartup()

        self.flag_simulating = False
        self.simulation_queue_counter = 0

        debug("setup of MainFrame - config")
        conf_mod.config.load_default(os.path.join(config_path, "genx.conf"))
        self.ReadConfig()
        self.wstartup.load_config()
        if self.wstartup.wx_plotting:
            from . import plotpanel_wx as plotpanel
        else:
            from . import plotpanel

        debug("setup of MainFrame - wx.Frame\n")
        wx.Frame.__init__(
            self,
            None,
            id=wx.ID_ANY,
            title="GenX " + program_version,
            size=wx.Size(self.opt.hsize, self.opt.vsize),
            style=wx.DEFAULT_FRAME_STYLE,
        )

        if dpi_overwrite:
            dpi_scale_factor = float(dpi_overwrite)
            debug("Overwrite DPI scale factor as %s" % dpi_scale_factor)
        elif sys.platform == "darwin":
            dpi_scale_factor = 1.0
            debug("Setting DPI scale to MacOS default of 1.0")
        else:
            try:
                dpi_scale_factor = self.GetDPIScaleFactor()
                debug("Detected DPI scale factor %s from GetDPIScaleFactor" % dpi_scale_factor)
            except AttributeError:
                dpi_scale_factor = self.GetContentScaleFactor()
                debug("Detected DPI scale factor %s from GetContentScaleFactor" % dpi_scale_factor)
        self.dpi_scale_factor = dpi_scale_factor
        wx.GetApp().dpi_scale_factor = dpi_scale_factor

        # GenX objects
        self.model_control = solvergui.ModelControlGUI(self)
        self.model_control.set_update_min_time(self.opt.solver_update_time)  # update time from configuration

        self.create_menu()

        self.main_frame_statusbar = self.CreateStatusBar(3)

        debug("setup of MainFrame - tool bar")
        self.create_toolbar()

        debug("setup of MainFrame - splitters and panels")
        self.ver_splitter = wx.SplitterWindow(self, wx.ID_ANY, style=wx.SP_3D | wx.SP_BORDER | wx.SP_LIVE_UPDATE)
        self.data_panel = wx.Panel(self.ver_splitter, wx.ID_ANY)
        self.data_notebook = wx.Notebook(self.data_panel, wx.ID_ANY)
        self.data_notebook_data = wx.Panel(self.data_notebook, wx.ID_ANY)
        self.data_list = datalist.DataListControl(self.data_notebook_data, wx.ID_ANY, self.eh_ex_status_text)
        self.data_notebook_pane_2 = wx.Panel(self.data_notebook, wx.ID_ANY)
        self.label_2 = wx.StaticText(self.data_notebook_pane_2, wx.ID_ANY, "  Data set: ")
        self.data_grid_choice = wx.Choice(self.data_notebook_pane_2, wx.ID_ANY, choices=["test2", "test1"])
        self.static_line_1 = wx.StaticLine(self.data_notebook_pane_2, wx.ID_ANY)
        self.data_grid = wx.grid.Grid(self.data_notebook_pane_2, wx.ID_ANY, size=(1, 1))
        self.main_panel = wx.Panel(self.ver_splitter, wx.ID_ANY)
        self.hor_splitter = wx.SplitterWindow(
            self.main_panel, wx.ID_ANY, style=wx.SP_3D | wx.SP_BORDER | wx.SP_LIVE_UPDATE
        )
        self.plot_panel = wx.Panel(self.hor_splitter, wx.ID_ANY)
        self.plot_splitter = wx.SplitterWindow(self.plot_panel, wx.ID_ANY)
        self.plot_notebook = wx.Notebook(self.plot_splitter, wx.ID_ANY, style=wx.NB_BOTTOM)
        self.plot_notebook_data = wx.Panel(self.plot_notebook, wx.ID_ANY)
        self.plot_data = plotpanel.DataPlotPanel(self.plot_notebook_data)
        self.plot_notebook_fom = wx.Panel(self.plot_notebook, wx.ID_ANY)
        self.plot_fom = plotpanel.ErrorPlotPanel(self.plot_notebook_fom)
        self.plot_notebook_Pars = wx.Panel(self.plot_notebook, wx.ID_ANY)
        self.plot_pars = plotpanel.ParsPlotPanel(self.plot_notebook_Pars)
        self.plot_notebook_foms = wx.Panel(self.plot_notebook, wx.ID_ANY)
        self.plot_fomscan = plotpanel.FomScanPlotPanel(self.plot_notebook_foms)
        self.wide_plugin_notebook = wx.Notebook(self.plot_splitter, wx.ID_ANY, style=wx.NB_BOTTOM)
        self.panel_1 = wx.Panel(self.wide_plugin_notebook, wx.ID_ANY)
        self.input_panel = wx.Panel(self.hor_splitter, wx.ID_ANY)
        self.input_notebook = wx.Notebook(self.input_panel, wx.ID_ANY, style=wx.NB_BOTTOM)
        self.input_notebook_grid = wx.Panel(self.input_notebook, wx.ID_ANY)
        self.paramter_grid = parametergrid.ParameterGrid(self.input_notebook_grid, self, self.model_control.controller)
        self.input_notebook_script = wx.Panel(self.input_notebook, wx.ID_ANY)
        self.script_editor = wx.py.editwindow.EditWindow(self.input_notebook_script, wx.ID_ANY)
        self.script_editor.SetBackSpaceUnIndents(True)
        self.script_editor.AutoCompSetChooseSingle(True)
        self.script_editor.AutoCompSetIgnoreCase(False)
        self.script_editor.Bind(wx.EVT_KEY_DOWN, self.ScriptEditorKeyEvent)

        debug("setup of MainFrame - properties and layout")
        self.__set_properties()
        self.__do_layout()

        debug("setup of MainFrame - bind")
        self.bind_menu()
        self.bind_toolbar()
        self.Bind(wx.EVT_CHOICE, self.eh_data_grid_choice, self.data_grid_choice)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.eh_plot_page_changed, self.plot_notebook)
        self.paramter_grid.grid.Bind(EVT_PARAMETER_SET_VALUE, self.model_control.OnSetParameterValue)
        self.paramter_grid.grid.Bind(EVT_MOVE_PARAMETER, self.model_control.OnMoveParameter)
        self.paramter_grid.grid.Bind(EVT_INSERT_PARAMETER, self.model_control.OnInsertParameter)
        self.paramter_grid.grid.Bind(EVT_DELETE_PARAMETERS, self.model_control.OnDeleteParameter)
        self.paramter_grid.Bind(EVT_SORT_AND_GROUP_PARAMETERS, self.model_control.OnSortAndGroupParameters)

        self.SetDropTarget(GenxFileDropTarget(self))
        debug("setup of MainFrame - manual config")

        # GenX objects
        self.model_control.set_data(self.data_list.data_cont.data)
        self.paramter_grid.SetParameters(self.model_control.get_model_params())
        self.set_script_text(self.model_control.get_model_script())
        self.script_editor.EmptyUndoBuffer()

        # Bind all the events that are needed to occur when a new model has
        # been loaded
        # Update the parameter grid
        self.Bind(EVT_NEW_MODEL, self.paramter_grid.OnNewModel, self)
        self.Bind(EVT_NEW_MODEL, self.data_list.eh_external_new_model, self)
        self.data_list.Bind(EVT_UPDATE_PLOTSETTINGS, self.model_control.update_plotsettings)
        # Update the script
        self.Bind(EVT_NEW_MODEL, self.eh_new_model, self)
        # Event that the plot should respond to
        self.Bind(EVT_DATA_LIST, self.plot_data.OnDataListEvent, self.data_list.list_ctrl)
        self.Bind(EVT_DATA_LIST, self.eh_external_update_data_grid_choice, self.data_list.list_ctrl)
        self.Bind(EVT_DATA_LIST, self.eh_external_update_data, self.data_list.list_ctrl)

        self.Bind(EVT_SIM_PLOT, self.plot_data.OnSimPlotEvent, self)
        self.Bind(EVT_SIM_PLOT, self.eh_external_fom_value, self)
        # Update events from the solver
        self.Bind(EVT_UPDATE_PLOT, self.eh_external_fom_value)
        self.Bind(EVT_UPDATE_PLOT, self.plot_data.OnSolverPlotEvent)
        self.Bind(EVT_UPDATE_PLOT, self.plot_fom.OnSolverPlotEvent)

        self.Bind(EVT_SOLVER_UPDATE_TEXT, self.eh_ex_status_text)
        self.Bind(EVT_UPDATE_PARAMETERS, self.plot_pars.OnSolverParameterEvent)
        self.Bind(EVT_UPDATE_PARAMETERS, self.model_control.OnUpdateParameters)

        # For picking a point in a plot
        self.Bind(EVT_PLOT_POSITION, self.eh_ex_point_pick)
        # This is needed to be able to create the events
        self.plot_data.SetCallbackWindow(self)
        self.plot_fom.SetCallbackWindow(self)
        self.plot_pars.SetCallbackWindow(self)
        self.plot_fomscan.SetCallbackWindow(self)
        self.Bind(EVT_PLOT_SETTINGS_CHANGE, self.eh_ex_plot_settings_changed)

        # Binding events which means model changes
        self.Bind(EVT_PARAMETER_GRID_CHANGE, self.eh_external_model_changed)
        self.Bind(wx.stc.EVT_STC_MODIFIED, self.eh_external_model_changed, self.script_editor)
        self.Bind(EVT_DATA_LIST, self.eh_external_model_changed, self.data_list.list_ctrl)

        # Event for when a value of a parameter in the parameter grid has been updated
        self.Bind(EVT_PARAMETER_VALUE_CHANGE, self.eh_external_parameter_value_changed)

        # Stuff for the find and replace functionality
        self.findreplace_data = wx.FindReplaceData()
        # Make search down as default
        self.findreplace_data.SetFlags(1)
        self.findreplace_dlg = wx.FindReplaceDialog(self, self.findreplace_data, "Find & replace", wx.FR_REPLACEDIALOG)
        self.Bind(wx.EVT_FIND, self.eh_external_find)
        self.Bind(wx.EVT_FIND_NEXT, self.eh_external_find)
        self.Bind(wx.EVT_FIND_REPLACE, self.eh_external_find)
        self.Bind(wx.EVT_FIND_REPLACE_ALL, self.eh_external_find)
        self.Bind(wx.EVT_FIND_CLOSE, self.eh_external_find)
        self.Bind(wx.EVT_CLOSE, self.eh_mb_quit)

        self.input_notebook.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.OnInputPageChanged)
        self.paramter_grid.SetFOMFunctions(self.project_fom_parameter, self.scan_parameter)

        # Initializations..
        # To force an update of the menubar...
        self.plot_data.SetZoom(False)

        with self.catch_error(action="init", step=f"reading plot config"):
            for p in [self.plot_data, self.plot_fom, self.plot_pars, self.plot_fomscan]:
                p.ReadConfig()

        debug("finished setup of MainFrame")

    def create_menu(self):
        debug("setup of MainFrame - menu bar")
        # Menu Bar
        self.main_frame_menubar = wx.MenuBar()
        self.mb_checkables = {}
        mfmb = self.main_frame_menubar
        mb_file = wx.Menu()
        mb_file.Append(custom_ids.MenuId.NEW_MODEL, "New...\tCtrl+N", "Creates a new model")
        mb_file.Append(
            custom_ids.MenuId.NEW_FROM_FILE,
            "New from file...\tCtrl+Shift+N",
            "Creates a new reflectivity model based on datafile",
        )
        mb_file.Append(custom_ids.MenuId.OPEN_MODEL, "Open...\tCtrl+O", "Opens an existing model")
        mb_file.Append(custom_ids.MenuId.SAVE_MODEL, "Save...\tCtrl+S", "Saves the current model")
        mb_file.Append(custom_ids.MenuId.SAVE_MODEL_AS, "Save As...", "Saves the active model with a new name")
        mb_file.AppendSeparator()
        mb_file.Append(custom_ids.MenuId.MODEL_BATCH, "Batch dialog...", "Run sequencial refinements of many datasets")
        mb_file.AppendSeparator()
        mb_import = wx.Menu()
        mb_import.Append(custom_ids.MenuId.IMPORT_DATA, "Import Data...\tCtrl+D", "Import data to the active data set")
        mb_import.Append(custom_ids.MenuId.IMPORT_TABLE, "Import Table...", "Import a table from an ASCII file")
        mb_import.Append(custom_ids.MenuId.IMPORT_SCRIPT, "Import Script...", "Import a python model script")
        mb_file.Append(wx.ID_ANY, "Import", mb_import, "")
        mb_export = wx.Menu()
        mb_export.Append(
            custom_ids.MenuId.EXPORT_ORSO,
            "Export ORSO...",
            "Export data and header in ORSO compatible text format (*.ort)",
        )
        mb_export.Append(custom_ids.MenuId.EXPORT_DATA, "Export Data...", "Export data in ASCII format")
        mb_export.Append(custom_ids.MenuId.EXPORT_TABLE, "Export Table...", "Export table to an ASCII file")
        mb_export.Append(custom_ids.MenuId.EXPORT_SCRIPT, "Export Script...", "Export the script to a python file")
        mb_file.Append(wx.ID_ANY, "Export", mb_export, "")
        mb_file.AppendSeparator()
        mb_print = wx.Menu()
        mb_print.Append(
            custom_ids.MenuId.PUBLISH_PLOT, "Publish Plot...\tCtrl+Shift+P", "Generate publication quality graph"
        )
        mb_print.Append(custom_ids.MenuId.PRINT_PLOT, "Print Plot...\tCtrl+P", "Print the current plot")
        mb_print.Append(custom_ids.MenuId.PRINT_GRID, "Print Grid...", "Prints the grid")
        mb_print.Append(custom_ids.MenuId.PRINT_SCRIPT, "Print Script...", "Prints the model script")
        mb_file.Append(wx.ID_ANY, "Print", mb_print, "")
        mb_file.AppendSeparator()
        mb_file.Append(custom_ids.MenuId.QUIT, "&Quit\tAlt+Q", "Quit the program")
        mfmb.Append(mb_file, "File")
        mb_edit = wx.Menu()
        self.undo_menu = mb_edit.Append(
            custom_ids.MenuId.UNDO, "Undo\tCtrl+Z", "Undo last action on model, not all changes supported"
        )
        self.redo_menu = mb_edit.Append(custom_ids.MenuId.REDO, "Redo\tCtrl+Shift+Z", "Redo last undone model change")
        mb_edit.Append(custom_ids.MenuId.HISTORY, "History...", "Show list of possible undo actions")
        self.undo_menu.Enable(False)
        self.redo_menu.Enable(False)
        mb_edit.AppendSeparator()
        mb_edit.Append(
            custom_ids.MenuId.COPY_GRAPH, "Copy Graph", "Copy the current graph to the clipboard as a bitmap"
        )
        mb_edit.Append(
            custom_ids.MenuId.COPY_SIM, "Copy Simulation", "Copy the current simulation and data as ASCII text"
        )
        mb_edit.Append(custom_ids.MenuId.COPY_TABLE, "Copy Table", "Copy the parameter grid")
        mb_edit.AppendSeparator()
        mb_edit.Append(custom_ids.MenuId.FIND_REPLACE, "&Find/Replace...\tCtrl+F", "Find and replace in the script")
        self.mb_editor = mb_edit.Append(
            custom_ids.MenuId.OPEN_IN_EDITOR, "Open in Editor\tCtrl+E", "Opens the current script in an external editor"
        )
        mb_edit_sub = wx.Menu()
        mb_edit_sub.Append(custom_ids.MenuId.NEW_DATA, "&New data set\tAlt+N", "Appends a new data set")
        mb_edit_sub.Append(custom_ids.MenuId.DELETE_DATA, "&Delete\tAlt+D", "Deletes the selected data sets")
        mb_edit_sub.Append(custom_ids.MenuId.LOWER_DATA, "&Lower item\tAlt+L", "Move selected item down")
        mb_edit_sub.Append(custom_ids.MenuId.RAISE_DATA, "&Raise item\tAlt+R", "Moves selected data sets up")
        mb_edit_sub.AppendSeparator()
        mb_edit_sub.Append(
            custom_ids.MenuId.TOGGLE_SHOW, "Toggle &Show\tAlt+S", "Toggle show on and off for the selected data set"
        )
        mb_edit_sub.Append(
            custom_ids.MenuId.TOGGLE_USE, "Toggle &Use\tAlt+U", "Toggle use on and off for the selected data sets"
        )
        mb_edit_sub.Append(custom_ids.MenuId.TOGGLE_ERROR, "Toggle &Error\tAlt+E", "Turn the use of error on and off")
        mb_edit_sub.AppendSeparator()
        mb_edit_sub.Append(
            custom_ids.MenuId.CALCS_DATA, "&Calculations\tAlt+C", "Opens dialog box to define dataset calculations"
        )
        mb_edit.Append(wx.ID_ANY, "Data", mb_edit_sub, "")
        mfmb.Append(mb_edit, "Edit")
        mb_view = wx.Menu()
        self.mb_checkables[custom_ids.MenuId.TOGGLE_SLIDER] = mb_view.Append(
            custom_ids.MenuId.TOGGLE_SLIDER, "Value as slider", "Control the grid value as a slider", wx.ITEM_CHECK
        )
        mb_view.AppendSeparator()
        mb_view_colors = wx.Menu()
        for key in COLOR_CYCLES.keys():
            self.mb_checkables[key] = mb_view_colors.Append(wx.ID_ANY, key, key, wx.ITEM_RADIO)
        mb_view.Append(
            custom_ids.MenuId.SET_PLOT,
            "Plot Markers\tShift+Ctrl+P",
            "Set the symbols and lines of data and simulations",
        )
        mb_view.Append(wx.ID_ANY, "Auto Color", mb_view_colors, "")
        mb_view.AppendSeparator()
        self.mb_checkables[custom_ids.MenuId.ZOOM] = mb_view.Append(
            custom_ids.MenuId.ZOOM, "Zoom\tCtrl+Z", "Turn the zoom on/off", wx.ITEM_CHECK
        )
        mb_view.Append(custom_ids.MenuId.ZOOM_ALL, "Zoom All\tCtrl+Shift+Z", "Zoom to fit all data points")
        self.mb_checkables[custom_ids.MenuId.AUTO_SCALE] = mb_view.Append(
            custom_ids.MenuId.AUTO_SCALE, "Autoscale", "Sets autoscale on when plotting", wx.ITEM_CHECK
        )
        self.mb_checkables[custom_ids.MenuId.USE_TOGGLE_SHOW] = mb_view.Append(
            custom_ids.MenuId.USE_TOGGLE_SHOW,
            "Use Toggle Show",
            "Set if the plotted data shold be toggled or selected by the mouse",
            wx.ITEM_CHECK,
        )
        mb_view.AppendSeparator()
        mb_view_yscale = wx.Menu()
        self.mb_checkables[custom_ids.MenuId.Y_SCALE_LOG] = mb_view_yscale.Append(
            custom_ids.MenuId.Y_SCALE_LOG, "log", "Set y-scale logarithmic", wx.ITEM_RADIO
        )
        self.mb_checkables[custom_ids.MenuId.Y_SCALE_LIN] = mb_view_yscale.Append(
            custom_ids.MenuId.Y_SCALE_LIN, "lin", "Set y-scale linear", wx.ITEM_RADIO
        )
        mb_view.Append(wx.ID_ANY, "y scale", mb_view_yscale, "")
        mb_view_xscale = wx.Menu()
        self.mb_checkables[custom_ids.MenuId.X_SCALE_LOG] = mb_view_xscale.Append(
            custom_ids.MenuId.X_SCALE_LOG, "log", "Set x-scale logarithmic", wx.ITEM_RADIO
        )
        self.mb_checkables[custom_ids.MenuId.X_SCALE_LIN] = mb_view_xscale.Append(
            custom_ids.MenuId.X_SCALE_LIN, "lin", "Set x-scale linear", wx.ITEM_RADIO
        )
        mb_view.Append(wx.ID_ANY, "x scale", mb_view_xscale, "")
        mfmb.Append(mb_view, "View")
        mb_fit = wx.Menu()
        self.mb_checkables[custom_ids.MenuId.AUTO_SIM] = mb_fit.Append(
            custom_ids.MenuId.AUTO_SIM,
            "Simulate Automatically",
            "Update simulation on model changes automatically",
            wx.ITEM_CHECK,
        )
        mb_fit.AppendSeparator()
        mb_fit.Append(custom_ids.MenuId.SIM_MODEL, "&Simulate\tF9", "Compile the script and run the Sim function")
        mb_fit.Append(
            custom_ids.MenuId.EVAL_MODEL,
            "&Evaluate\tF5",
            "Evaluate the Sim function twice and compre result to find inconsitancy",
        )
        self.mb_checkables[custom_ids.MenuId.TOGGLE_CUDA] = mb_fit.Append(
            custom_ids.MenuId.TOGGLE_CUDA, "Use CUDA", "Make use of Nvidia GPU computing with CUDA", wx.ITEM_CHECK
        )
        mb_fit.AppendSeparator()
        mb_fit.Append(custom_ids.MenuId.START_FIT, "Start &Fit\tCtrl+F", "Start fitting")
        mb_fit.Append(custom_ids.MenuId.STOP_FIT, "&Halt Fit\tCtrl+H", "Stop fitting")
        mb_fit.Append(
            custom_ids.MenuId.RESTART_FIT,
            "&Resume Fit\tCtrl+R",
            "Resumes fitting without reinitilazation of the optimizer",
        )
        mb_fit.AppendSeparator()
        mb_fit.Append(custom_ids.MenuId.ANALYZE, "Analyze fit", "Analyze the fit")
        mfmb.Append(mb_fit, "Model")
        mb_set = wx.Menu()
        mb_set_plugins = wx.Menu()
        mb_set_plugins.AppendSeparator()
        mb_set.Append(wx.ID_ANY, "Plugins", mb_set_plugins, "")
        mb_set.AppendSeparator()
        mb_set.Append(custom_ids.MenuId.SET_OPTIMIZER, "Optimizer\tShift+Ctrl+O", "")
        mb_set.Append(custom_ids.MenuId.SET_DATA_LOADER, "Data Loader\tShift+Ctrl+D", "")
        mb_set.Append(custom_ids.MenuId.SET_IMPORT, "Import\tShift+Ctrl+I", "Import settings for the data sets")
        mb_set.AppendSeparator()
        mb_set.Append(custom_ids.MenuId.SET_PROFILE, "Startup Profile...", "")
        mb_set.Append(custom_ids.MenuId.SET_EDITOR, "Select External Editor...", "")
        mfmb.Append(mb_set, "Settings")
        help_menu = wx.Menu()
        help_menu.Append(custom_ids.MenuId.HELP_MODEL, "Models Help...", "Show help for the models")
        help_menu.Append(custom_ids.MenuId.HELP_FOM, "FOM Help", "Show help about the fom")
        help_menu.Append(custom_ids.MenuId.HELP_PLUGINS, "Plugins Helps...", "Show help for the plugins")
        help_menu.Append(custom_ids.MenuId.HELP_DATA_LOADERS, "Data loaders Help...", "Show help for the data loaders")
        help_menu.AppendSeparator()
        help_menu.Append(
            custom_ids.MenuId.HELP_EXAMPLES, "Open Examples...", "Show load model dialog in examples folder"
        )
        help_menu.Append(custom_ids.MenuId.HELP_MANUAL, "Open Manual...", "Show the manual")
        help_menu.Append(custom_ids.MenuId.HELP_HOMEPAGE, "Open Homepage...", "Open the homepage")
        help_menu.Append(custom_ids.MenuId.HELP_ABOUT, "About...", "Shows information about GenX")
        help_menu.AppendSeparator()
        help_menu.Append(
            custom_ids.MenuId.HELP_DEBUG,
            "Collect Debug Info...\tCtrl+L",
            "Record debug information to file and show console",
        )
        mfmb.Append(help_menu, "Help")
        self.SetMenuBar(mfmb)
        # Plugin controller builds own menu entries
        self.plugin_control = add_on.PluginController(self, mb_set_plugins)

    def bind_menu(self):
        self.Bind(wx.EVT_MENU, self.eh_mb_new, id=custom_ids.MenuId.NEW_MODEL)
        self.Bind(wx.EVT_MENU, self.eh_mb_new_from_file, id=custom_ids.MenuId.NEW_FROM_FILE)
        self.Bind(wx.EVT_MENU, self.eh_mb_open, id=custom_ids.MenuId.OPEN_MODEL)
        self.Bind(wx.EVT_MENU, self.eh_mb_save, id=custom_ids.MenuId.SAVE_MODEL)
        self.Bind(wx.EVT_MENU, self.eh_mb_saveas, id=custom_ids.MenuId.SAVE_MODEL_AS)
        self.Bind(wx.EVT_MENU, self.eh_mb_batch, id=custom_ids.MenuId.MODEL_BATCH)
        self.Bind(wx.EVT_MENU, self.eh_mb_import_data, id=custom_ids.MenuId.IMPORT_DATA)
        self.Bind(wx.EVT_MENU, self.eh_mb_import_table, id=custom_ids.MenuId.IMPORT_TABLE)
        self.Bind(wx.EVT_MENU, self.eh_mb_import_script, id=custom_ids.MenuId.IMPORT_SCRIPT)
        self.Bind(wx.EVT_MENU, self.eh_mb_export_orso, id=custom_ids.MenuId.EXPORT_ORSO)
        self.Bind(wx.EVT_MENU, self.eh_mb_export_data, id=custom_ids.MenuId.EXPORT_DATA)
        self.Bind(wx.EVT_MENU, self.eh_mb_export_table, id=custom_ids.MenuId.EXPORT_TABLE)
        self.Bind(wx.EVT_MENU, self.eh_mb_export_script, id=custom_ids.MenuId.EXPORT_SCRIPT)
        self.Bind(wx.EVT_MENU, self.eh_mb_publish_plot, id=custom_ids.MenuId.PUBLISH_PLOT)
        self.Bind(wx.EVT_MENU, self.eh_mb_print_plot, id=custom_ids.MenuId.PRINT_PLOT)
        self.Bind(wx.EVT_MENU, self.eh_mb_print_grid, id=custom_ids.MenuId.PRINT_GRID)
        self.Bind(wx.EVT_MENU, self.eh_mb_print_script, id=custom_ids.MenuId.PRINT_SCRIPT)
        self.Bind(wx.EVT_MENU, self.eh_mb_quit, id=custom_ids.MenuId.QUIT)
        self.Bind(wx.EVT_MENU, self.model_control.OnUndo, id=custom_ids.MenuId.UNDO)
        self.Bind(wx.EVT_MENU, self.model_control.OnRedo, id=custom_ids.MenuId.REDO)
        self.Bind(wx.EVT_MENU, self.model_control.OnShowHistory, id=custom_ids.MenuId.HISTORY)
        self.Bind(wx.EVT_MENU, self.eh_mb_copy_graph, id=custom_ids.MenuId.COPY_GRAPH)
        self.Bind(wx.EVT_MENU, self.eh_mb_copy_sim, id=custom_ids.MenuId.COPY_SIM)
        self.Bind(wx.EVT_MENU, self.eh_mb_copy_table, id=custom_ids.MenuId.COPY_TABLE)
        self.Bind(wx.EVT_MENU, self.eh_mb_findreplace, id=custom_ids.MenuId.FIND_REPLACE)
        self.Bind(wx.EVT_MENU, self.eh_mb_open_editor, id=custom_ids.MenuId.OPEN_IN_EDITOR)
        self.Bind(wx.EVT_MENU, self.eh_data_new_set, id=custom_ids.MenuId.NEW_DATA)
        self.Bind(wx.EVT_MENU, self.eh_data_delete, id=custom_ids.MenuId.DELETE_DATA)
        self.Bind(wx.EVT_MENU, self.eh_data_move_down, id=custom_ids.MenuId.RAISE_DATA)
        self.Bind(wx.EVT_MENU, self.eh_data_move_up, id=custom_ids.MenuId.LOWER_DATA)
        self.Bind(wx.EVT_MENU, self.eh_data_toggle_show, id=custom_ids.MenuId.TOGGLE_SHOW)
        self.Bind(wx.EVT_MENU, self.eh_data_toggle_use, id=custom_ids.MenuId.TOGGLE_USE)
        self.Bind(wx.EVT_MENU, self.eh_data_toggle_error, id=custom_ids.MenuId.TOGGLE_ERROR)
        self.Bind(wx.EVT_MENU, self.eh_data_calc, id=custom_ids.MenuId.CALCS_DATA)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_grid_slider, id=custom_ids.MenuId.TOGGLE_SLIDER)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_zoom, id=custom_ids.MenuId.ZOOM)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_zoomall, id=custom_ids.MenuId.ZOOM_ALL)
        for key in COLOR_CYCLES.keys():
            self.Bind(wx.EVT_MENU, self.eh_mb_view_color_cycle, id=self.mb_checkables[key].GetId())
        self.Bind(wx.EVT_MENU, self.eh_mb_view_yscale_log, id=custom_ids.MenuId.Y_SCALE_LOG)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_yscale_linear, id=custom_ids.MenuId.Y_SCALE_LIN)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_xscale_log, id=custom_ids.MenuId.X_SCALE_LOG)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_xscale_linear, id=custom_ids.MenuId.X_SCALE_LOG)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_autoscale, id=custom_ids.MenuId.AUTO_SCALE)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_use_toggle_show, id=custom_ids.MenuId.USE_TOGGLE_SHOW)
        self.Bind(wx.EVT_MENU, self.eh_tb_simulate, id=custom_ids.MenuId.SIM_MODEL)
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_evaluate, id=custom_ids.MenuId.EVAL_MODEL)
        self.Bind(wx.EVT_MENU, self.eh_mb_use_cuda, id=custom_ids.MenuId.TOGGLE_CUDA)
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_start, id=custom_ids.MenuId.START_FIT)
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_stop, id=custom_ids.MenuId.STOP_FIT)
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_resume, id=custom_ids.MenuId.RESTART_FIT)
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_analyze, id=custom_ids.MenuId.ANALYZE)
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_autosim, id=custom_ids.MenuId.AUTO_SIM)
        self.Bind(wx.EVT_MENU, self.eh_mb_set_opt, id=custom_ids.MenuId.SET_OPTIMIZER)
        self.Bind(wx.EVT_MENU, self.eh_mb_set_dal, id=custom_ids.MenuId.SET_DATA_LOADER)
        self.Bind(wx.EVT_MENU, self.eh_data_import, id=custom_ids.MenuId.SET_IMPORT)
        self.Bind(wx.EVT_MENU, self.eh_data_plots, id=custom_ids.MenuId.SET_PLOT)
        self.Bind(wx.EVT_MENU, self.eh_show_startup_dialog, id=custom_ids.MenuId.SET_PROFILE)
        self.Bind(wx.EVT_MENU, self.eh_mb_select_editor, id=custom_ids.MenuId.SET_EDITOR)
        self.Bind(wx.EVT_MENU, self.eh_mb_models_help, id=custom_ids.MenuId.HELP_MODEL)
        self.Bind(wx.EVT_MENU, self.eh_mb_fom_help, id=custom_ids.MenuId.HELP_FOM)
        self.Bind(wx.EVT_MENU, self.eh_mb_plugins_help, id=custom_ids.MenuId.HELP_PLUGINS)
        self.Bind(wx.EVT_MENU, self.eh_mb_data_loaders_help, id=custom_ids.MenuId.HELP_DATA_LOADERS)
        self.Bind(wx.EVT_MENU, self.eh_mb_misc_showman, id=custom_ids.MenuId.HELP_MANUAL)
        self.Bind(wx.EVT_MENU, self.eh_mb_misc_examples, id=custom_ids.MenuId.HELP_EXAMPLES)
        self.Bind(wx.EVT_MENU, self.eh_mb_misc_openhomepage, id=custom_ids.MenuId.HELP_HOMEPAGE)
        self.Bind(wx.EVT_MENU, self.eh_mb_misc_about, id=custom_ids.MenuId.HELP_ABOUT)
        self.Bind(wx.EVT_MENU, self.eh_mb_debug_dialog, id=custom_ids.MenuId.HELP_DEBUG)

    def create_toolbar(self):
        tb_bmp_size = int(32 * self.dpi_scale_factor)
        self.main_frame_toolbar = wx.ToolBar(self, -1, style=wx.TB_DEFAULT_STYLE)
        self.SetToolBar(self.main_frame_toolbar)
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.NEW_MODEL,
            "tb_new",
            wx.Bitmap(img.getnewImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "New model | Ctrl+N",
            "Create a new model | Ctrl+N",
        )
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.NEW_FROM_FILE,
            "tb_new_from_file",
            wx.Bitmap(img.getnew_orsoImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "New from file | Ctrl+Shift+N",
            "Create a new reflectivity model based on datafile | Ctrl+Shift+N",
        )
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.OPEN_MODEL,
            "tb_open",
            wx.Bitmap(img.getopenImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Open | Ctrl+O",
            "Open an existing model | Ctrl+O",
        )
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.SAVE_MODEL,
            "tb_save",
            wx.Bitmap(img.getsaveImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Save | Ctrl+S",
            "Save model to file | Ctrl+S",
        )
        self.main_frame_toolbar.AddSeparator()
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.SIM_MODEL,
            "tb_simulate",
            wx.Bitmap(img.getsimulateImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Simulate | F9",
            "Simulate the model | F9",
        )
        self.main_frame_toolbar.AddSeparator()
        sselect = self.model_control.get_solvers()
        solver_select = wx.ComboBox(
            self.main_frame_toolbar,
            id=custom_ids.ToolId.SOLVER_SELECT,
            value=sselect[0],
            choices=sselect,
            style=wx.CB_READONLY,
        )
        self.main_frame_toolbar.AddControl(solver_select, "solver")
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.START_FIT,
            "tb_start_fit",
            wx.Bitmap(img.getstart_fitImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Start fit | Ctrl+F",
            "Start fitting | Ctrl+F",
        )
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.STOP_FIT,
            "tb_stop_fit",
            wx.Bitmap(img.getstop_fitImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Stop fit | Ctrl+H",
            "Stop fitting | Ctrl+H",
        )
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.RESTART_FIT,
            "tb_restart_fit",
            wx.Bitmap(img.getrestart_fitImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Restart fit | Ctrl+R",
            "Restart the fit | Ctrl+R",
        )
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.CALC_ERROR,
            "tb_calc_error_bars",
            wx.Bitmap(img.getcalc_error_barImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Calculate errorbars",
            "Calculate errorbars",
        )
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.ERROR_STATS,
            "tb_error_stats",
            wx.Bitmap(img.getpar_projImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_NORMAL,
            "Error Statistics",
            "Error Statistics",
        )
        self.main_frame_toolbar.AddSeparator()
        self.main_frame_toolbar.AddTool(
            custom_ids.ToolId.ZOOM,
            "tb_zoom",
            wx.Bitmap(img.getzoomImage().Scale(tb_bmp_size, tb_bmp_size)),
            wx.NullBitmap,
            wx.ITEM_CHECK,
            "Zoom | Ctrl+Z",
            "Turn zoom on/off  | Ctrl+Z",
        )

    def bind_toolbar(self):
        self.Bind(wx.EVT_TOOL, self.eh_tb_new, id=custom_ids.ToolId.NEW_MODEL)
        self.Bind(wx.EVT_TOOL, self.eh_tb_new_from_file, id=custom_ids.ToolId.NEW_FROM_FILE)
        self.Bind(wx.EVT_TOOL, self.eh_tb_open, id=custom_ids.ToolId.OPEN_MODEL)
        self.Bind(wx.EVT_TOOL, self.eh_tb_save, id=custom_ids.ToolId.SAVE_MODEL)
        self.Bind(wx.EVT_COMBOBOX, self.eh_tb_select_solver, id=custom_ids.ToolId.SOLVER_SELECT)
        self.Bind(wx.EVT_TOOL, self.eh_tb_simulate, id=custom_ids.ToolId.SIM_MODEL)
        self.Bind(wx.EVT_TOOL, self.eh_tb_start_fit, id=custom_ids.ToolId.START_FIT)
        self.Bind(wx.EVT_TOOL, self.eh_tb_stop_fit, id=custom_ids.ToolId.STOP_FIT)
        self.Bind(wx.EVT_TOOL, self.eh_tb_restart_fit, id=custom_ids.ToolId.RESTART_FIT)
        self.Bind(wx.EVT_TOOL, self.eh_tb_calc_error_bars, id=custom_ids.ToolId.CALC_ERROR)
        self.Bind(wx.EVT_TOOL, self.eh_tb_error_stats, id=custom_ids.ToolId.ERROR_STATS)
        self.Bind(wx.EVT_TOOL, self.eh_tb_zoom, id=custom_ids.ToolId.ZOOM)

    def OnInputPageChanged(self, evt):
        tpage, fpage = evt.GetSelection(), evt.GetOldSelection()
        # check for odd case, that either page does not exist anymore
        if self.input_notebook.GetPageCount() >= max(tpage, fpage):
            return
        if fpage != tpage and self.input_notebook.GetPageText(fpage) == "Script":
            self.model_control.set_model_script(self.script_editor.GetText())

    def scan_parameter(self, row):
        """
        Scans the parameter in row row [int] from max to min in the number
        of steps given by dialog input.
        """
        self.model_control.compile_if_needed()

        dlg = wx.NumberEntryDialog(self, "Input the number of evaluation points for the scan", "Steps", "", 50, 2, 1000)
        if dlg.ShowModal() == wx.ID_OK:
            self.main_frame_statusbar.SetStatusText("Scanning parameter", 1)
            with self.catch_error(action="scan_parameters", step=f"scanning parameters"):
                x, y = self.model_control.ScanParameter(row, dlg.GetValue())

                bestx = self.model_control.get_parameter_data(row)[1]
                besty = self.model_control.get_fom()

                self.plot_fomscan.SetPlottype("scan")
                e_scale = getattr(self.model_control.controller.optimizer.opt, "errorbar_level", 0)
                self.plot_fomscan.Plot((x, y, bestx, besty, e_scale), self.model_control.get_parameter_name(row), "FOM")
                self.sep_plot_notebook.SetSelection(3)

        dlg.Destroy()

    def __set_properties(self):
        self.main_frame_fom_text = wx.StaticText(
            self.main_frame_toolbar, -1, "        FOM:                    ", size=(400, -1)
        )
        try:
            font = wx.Font(wx.FontInfo(15 * self.dpi_scale_factor))
        except TypeError:
            pass
        else:
            self.main_frame_fom_text.SetFont(font)
        self.main_frame_fom_text.SetLabel("        FOM: None")
        # self.main_frame_fom_text.SetEditable(False)
        self.main_frame_toolbar.AddSeparator()
        self.main_frame_toolbar.AddSeparator()
        self.main_frame_toolbar.AddControl(self.main_frame_fom_text)

        _icon = wx.NullIcon
        _icon.CopyFromBitmap(img.genx.GetBitmap())
        self.SetIcon(_icon)
        self.main_frame_statusbar.SetStatusWidths([-2, -3, -2])

        # statusbar fields
        main_frame_statusbar_fields = ["", "", "x,y"]
        for i in range(len(main_frame_statusbar_fields)):
            self.main_frame_statusbar.SetStatusText(main_frame_statusbar_fields[i], i)
        self.main_frame_toolbar.Realize()
        self.data_grid_choice.SetSelection(0)
        self.static_line_1.SetMinSize((-1, 5))
        self.data_grid.CreateGrid(10, 6)
        self.data_grid.EnableEditing(0)
        self.data_grid.EnableDragRowSize(0)
        self.data_grid.SetColLabelValue(0, "x_raw")
        self.data_grid.SetColLabelValue(1, "y_raw")
        self.data_grid.SetColLabelValue(2, "Error_raw")
        self.data_grid.SetColLabelValue(3, "x")
        self.data_grid.SetColLabelValue(4, "y")
        self.data_grid.SetColLabelValue(5, "Error")
        self.plot_splitter.SetMinimumPaneSize(20)
        self.hor_splitter.SetMinimumPaneSize(20)
        self.ver_splitter.SetMinimumPaneSize(20)

        # Turn Line numbering on for the editor
        self.script_editor.setDisplayLineNumbers(True)
        self.ver_splitter.SetMinimumPaneSize(1)
        self.hor_splitter.SetMinimumPaneSize(1)

    def __do_layout(self):
        frame_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        input_sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_8 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_7 = wx.BoxSizer(wx.HORIZONTAL)
        plot_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer_6 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_5 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3 = wx.BoxSizer(wx.HORIZONTAL)
        data_sizer = wx.BoxSizer(wx.VERTICAL)
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        sizer_2 = wx.BoxSizer(wx.HORIZONTAL)
        data_list_sizer = wx.BoxSizer(wx.HORIZONTAL)
        data_list_sizer.Add(self.data_list, 1, wx.EXPAND, 0)
        self.data_notebook_data.SetSizer(data_list_sizer)
        sizer_1.Add((20, 5), 0, 0, 0)
        sizer_2.Add(self.label_2, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_2.Add(self.data_grid_choice, 3, wx.EXPAND, 0)
        sizer_2.Add((20, 20), 0, wx.EXPAND, 0)
        sizer_1.Add(sizer_2, 0, wx.EXPAND, 0)
        sizer_1.Add(self.static_line_1, 0, wx.EXPAND, 0)
        sizer_1.Add(self.data_grid, 1, wx.EXPAND, 0)
        self.data_notebook_pane_2.SetSizer(sizer_1)
        self.data_notebook.AddPage(self.data_notebook_data, "Data")
        self.data_notebook.AddPage(self.data_notebook_pane_2, "View")
        data_sizer.Add(self.data_notebook, 1, wx.EXPAND, 0)
        self.data_panel.SetSizer(data_sizer)
        sizer_3.Add(self.plot_data, 2, wx.EXPAND, 0)
        self.plot_notebook_data.SetSizer(sizer_3)
        sizer_4.Add(self.plot_fom, 1, wx.EXPAND, 0)
        self.plot_notebook_fom.SetSizer(sizer_4)
        sizer_5.Add(self.plot_pars, 1, wx.EXPAND, 0)
        self.plot_notebook_Pars.SetSizer(sizer_5)
        sizer_6.Add(self.plot_fomscan, 1, wx.EXPAND, 0)
        self.plot_notebook_foms.SetSizer(sizer_6)
        self.plot_notebook.AddPage(self.plot_notebook_data, "Data")
        self.plot_notebook.AddPage(self.plot_notebook_fom, "FOM")
        self.plot_notebook.AddPage(self.plot_notebook_Pars, "Pars")
        self.plot_notebook.AddPage(self.plot_notebook_foms, "FOM scans")
        self.wide_plugin_notebook.AddPage(self.panel_1, "Empty Tab")
        self.plot_splitter.SplitVertically(self.plot_notebook, self.wide_plugin_notebook)
        plot_sizer.Add(self.plot_splitter, 1, wx.EXPAND, 0)
        self.plot_panel.SetSizer(plot_sizer)
        sizer_7.Add(self.paramter_grid, 1, wx.EXPAND, 0)
        self.input_notebook_grid.SetSizer(sizer_7)
        sizer_8.Add(self.script_editor, 1, wx.EXPAND, 0)
        self.input_notebook_script.SetSizer(sizer_8)
        self.input_notebook.AddPage(self.input_notebook_grid, "Grid")
        self.input_notebook.AddPage(self.input_notebook_script, "Script")
        input_sizer.Add(self.input_notebook, 1, wx.EXPAND, 0)
        self.input_panel.SetSizer(input_sizer)
        self.hor_splitter.SplitHorizontally(self.plot_panel, self.input_panel)
        main_sizer.Add(self.hor_splitter, 1, wx.EXPAND, 0)
        self.main_panel.SetSizer(main_sizer)
        self.ver_splitter.SplitVertically(self.data_panel, self.main_panel)
        frame_sizer.Add(self.ver_splitter, 1, wx.EXPAND, 0)
        self.SetSizer(frame_sizer)
        frame_sizer.Fit(self)
        frame_sizer.SetSizeHints(self)
        self.Layout()
        self.Centre()

        self.sep_plot_notebook = self.plot_notebook
        if self.wstartup.widescreen:
            # test adding new notebooks for plugins in wide screen layout
            self.plot_notebook = self.wide_plugin_notebook
            self.plot_notebook.DeletePage(0)
            self.plot_splitter.SetSashGravity(0.75)
            self.sep_data_notebook = self.data_notebook
            self.data_notebook = wx.Notebook(self.data_panel, wx.ID_ANY, style=wx.NB_TOP | wx.BORDER_SUNKEN)
            data_sizer.Add(self.data_notebook, 1, wx.EXPAND | wx.ALL, 4)
        else:
            self.plot_splitter.Unsplit()

    def Show(self, **kwargs):
        """
        Overiding the default method since any resizing has to come AFTER
        the calls to Show
        """
        display_size = wx.DisplaySize()
        hsize = self.opt.hsize or int(display_size[0] * 0.85)
        vsize = self.opt.vsize or int(display_size[1] * 0.9)
        self.SetSize(hsize, vsize)
        self.CenterOnScreen()
        self.ver_splitter.SetSashPosition(200)
        self.hor_splitter.SetSashPosition(200)
        # Gravity sets how much the upper/left window is resized default 0
        self.ver_splitter.SetSashGravity(0.25)
        self.hor_splitter.SetSashGravity(0.75)

        wx.Frame.Show(self)
        ## Begin Manual Config
        wx.CallAfter(self.LayoutSplitters)
        wx.CallAfter(self.EndInit)

    def LayoutSplitters(self):
        if not sys.platform.startswith("win"):
            display_size = wx.DisplaySize()
            hsize = self.opt.hsize or int(display_size[0] * 0.85)
            vsize = self.opt.vsize or int(display_size[1] * 0.9)
            self.SetSize(hsize, vsize)
        size = self.GetSize()
        vsplit = self.opt.vsplit or size[0] / 4
        hsplit = self.opt.hsplit or size[1] - 450
        self.ver_splitter.SetSashPosition(vsplit)
        self.hor_splitter.SetSashPosition(hsplit)

        if self.wstartup.widescreen:
            psplit = self.opt.psplit or int(size[1] * 0.6)
            self.plot_splitter.SetSashPosition(psplit)

    def EndInit(self):
        wx.YieldIfNeeded()  # make sure that all GUI layout is performed before _init_phase is unset
        self.SetMinSize(wx.Size(600, 400))
        self._init_phase = False

    def startup_dialog(self, profile_path, force_show=False):
        if self.wstartup.show_profiles or force_show:
            prev_gui = (self.wstartup.wx_plotting, self.wstartup.widescreen)
            startup_dialog = StartUpConfigDialog(
                self,
                os.path.join(profile_path, "profiles"),
                show_cb=self.wstartup.show_profiles,
                wide=self.wstartup.widescreen,
                wx_ploting=self.wstartup.wx_plotting,
            )
            startup_dialog.ShowModal()
            config_file = startup_dialog.GetConfigFile()
            if config_file:
                conf_mod.config.load_default(os.path.join(profile_path, "profiles", config_file), reset=True)
                self.wstartup.show_profiles = startup_dialog.GetShowAtStartup()
                self.wstartup.widescreen = startup_dialog.GetWidescreen()
                self.wstartup.wx_plotting = startup_dialog.GetWxPloting()
                self.wstartup.save_config(default=True)
                conf_mod.config.write_default(os.path.join(config_path, "genx.conf"))
                if (self.wstartup.wx_plotting, self.wstartup.widescreen) != prev_gui and force_show:
                    # in case the GUI has to be re-build completely
                    app: GenxApp = wx.GetApp()
                    wx.CallAfter(app.OnRebuild)
                elif self.wstartup.wx_plotting != prev_gui[0]:
                    app: GenxApp = wx.GetApp()
                    wx.CallLater(2000, app.OnRebuild)
                else:
                    debug("Changed profile, plugins to load=%s" % conf_mod.config.get("plugins", "loaded plugins"))
                    with self.catch_error(action="startup_dialog", step=f"open model"):
                        self.plugin_control.OnOpenModel(None)

    def ScriptEditorKeyEvent(self, evt: wx.KeyEvent):
        if evt.GetKeyCode() in [wx.WXK_RETURN, wx.WXK_NUMPAD_ENTER]:
            if self.script_editor.AutoCompActive():
                self.script_editor.AutoCompComplete()
                return
            pos = self.script_editor.GetCurrentPos()
            line = self.script_editor.GetCurrentLine()
            idn = self.script_editor.GetLineIndentation(line)
            txt = self.script_editor.GetLine(line).strip()
            if evt.controlDown:
                if self.script_editor.AutoCompActive():
                    self.script_editor.AutoCompCancel()
                if evt.shiftDown:
                    # try to show context help for object on current line
                    txt = txt.split("(", 1)[0]
                    tt = self.get_tooltip(txt)
                    tip = wx.adv.RichToolTip(f"Object help for {txt}", tt)
                    tip.SetIcon(wx.ICON_INFORMATION)
                    tip.ShowFor(self.script_editor)
                else:
                    txt = txt.split(",")[-1].split("(")[-1]
                    if "." in txt:
                        obj, subtxt = txt.rsplit(".", 1)
                        self.script_editor.AutoCompShow(
                            len(subtxt), self.build_autocomplete_items(obj=obj, start=subtxt)
                        )
                    else:
                        self.script_editor.AutoCompShow(len(txt), self.build_autocomplete_items(start=txt))
            else:
                if (
                    txt.startswith("for ")
                    or txt.startswith("if ")
                    or txt.startswith("elif ")
                    or txt.startswith("else:")
                ):
                    idn += 4
                if evt.shiftDown:
                    pos += len(self.script_editor.GetLine(line)) - self.script_editor.GetColumn(pos) - 1
                self.script_editor.InsertText(pos, "\n" + " " * idn)
                self.script_editor.GotoPos(pos + idn + 1)
        elif evt.controlDown and evt.GetUnicodeKey() in [ord("z"), ord("Z")]:
            if evt.shiftDown:
                self.script_editor.Redo()
            else:
                self.script_editor.Undo()
        else:
            evt.Skip()

    def get_tooltip(self, obj):
        try:
            objitem = self.model_control.get_model().eval_in_model(obj)
        except Exception:
            return "Error"
        ds = getattr(objitem, "__doc__", None) or "No documentation found"
        ds = "\n".join(ds.splitlines()[:20])
        return ds

    def build_autocomplete_items(self, obj=None, start=""):
        if obj is None:
            output = []
            for key in dir(self.model_control.get_model().script_module):
                if key.startswith("_") or not key.startswith(start):
                    continue
                output.append(key)
            output.sort()
            return " ".join(output)
        else:
            try:
                objitem = self.model_control.get_model().eval_in_model(obj)
            except NameError:
                return ""
            output = []
            for key in dir(objitem):
                if key.startswith("_") or not key.startswith(start):
                    continue
                output.append(key)
            output.sort()
            return " ".join(output)

    def project_fom_parameter(self, row):
        """project_fom_parameter(frame, row) --> None

        Plots the project fom given by the row row [int]
        """
        import numpy as np

        if not self.model_control.IsFitted():
            ShowNotificationDialog(
                self,
                "Please conduct a fit before"
                + " scanning a parameter. The script needs to be compiled and foms have"
                + " to be collected.",
            )
            return

        self.main_frame_statusbar.SetStatusText("Trying to project fom", 1)
        with self.catch_error(action="project_fom_parameters", step=f"projecting fom parameters"):
            e_scale = getattr(self.model_control.controller.optimizer.opt, "errorbar_level", None)
            if e_scale is None:
                ShowNotificationDialog(
                    self, "This feature requires a fit with Differential Evolution, " "consider using fom scan instead."
                )
                return
            x, y = self.model_control.ProjectEvals(row)
            if len(x) == 0 or len(y) == 0:
                ShowNotificationDialog(
                    self,
                    "Please conduct a fit before"
                    + " projecting a parameter. The script needs to be compiled and foms have"
                    + " to be collected.",
                )
                return
            elif self.model_control.get_fom() is None or np.isnan(self.model_control.get_fom()):
                ShowNotificationDialog(self, "The model must be simulated (FOM is not a valid number)")
                return
            fs, pars = self.model_control.get_sim_pars()
            bestx = pars[row]
            besty = self.model_control.get_fom()
            self.plot_fomscan.SetPlottype("project")
            self.plot_fomscan.Plot((x, y, bestx, besty, e_scale), self.model_control.get_parameter_name(row), "FOM")
            self.sep_plot_notebook.SetSelection(3)

    def update_title(self):
        filepath, filename = os.path.split(self.model_control.get_filename())
        if filename != "":
            if self.model_control.saved:
                self.SetTitle(filename + " - " + filepath + " - GenX " + program_version)
            else:
                self.SetTitle(filename + "* - " + filepath + " - GenX " + program_version)
        else:
            self.SetTitle("GenX " + program_version)

    def get_pages(self):
        # Get all plot panel objects in GUI
        pages = []
        if self.sep_plot_notebook is not self.plot_notebook:
            for page in self.sep_plot_notebook.GetChildren():
                pages += page.GetChildren()
        for page in self.plot_notebook.GetChildren():
            pages += page.GetChildren()
        return pages

    def _set_status_text(self, text):
        wx.CallAfter(self.main_frame_statusbar.SetStatusText, text)

    def catch_error(self, action="execution", step=None, verbose=True):
        if verbose:
            return CatchModelError(self, action=action, step=step, status_update=self._set_status_text)
        else:
            return CatchModelError(self, action=action, step=step, status_update=None)

    def new_from_file(self, paths):
        debug("new_from_file: clear model")
        self.model_control.new_model()
        self.paramter_grid.PrepareNewModel()
        self.data_list.data_cont.set_data(self.model_control.get_data())

        # read data from file
        debug("new_from_file: load datafile")
        with self.catch_error(action="read_data", step=f"read file {os.path.basename(paths[0])}") as mng:
            self.data_list.list_ctrl.data_loader_cont.LoadPlugin("orso")
            self.data_list.list_ctrl.load_from_files(paths, do_update=False)

        debug("new_from_file: build model script")
        # if this was exported from genx, use the embedded script
        meta = deepcopy(self.data_list.data_cont.data[0].meta)
        ana_meta = meta.get("analysis", {})
        if ana_meta.get("software", {}).get("name", "") == "GenX":
            self.model_control.set_model_script(ana_meta["script"])
            params = ana_meta["parameters"]
            pdata = [[pi["Parameter"], pi["Value"], pi["Fit"], pi["Min"], pi["Max"], pi["Error"]] for pi in params]
            self.model_control.get_model_params().data = pdata
            self.paramter_grid.table.UpdateView()
            for di in self.data_list.data_cont.data:
                del di.meta["analysis"]
        else:
            from genx.plugins.data_loaders.help_modules.orso_analyzer import OrsoHeaderAnalyzer

            header_analyzed = OrsoHeaderAnalyzer(meta)

            from ..plugins.add_ons.SimpleReflectivity import Plugin as SRPlugin

            if "SimpleReflectivity" in self.plugin_control.plugin_handler.loaded_plugins:
                refl: SRPlugin = self.plugin_control.GetPlugin("SimpleReflectivity")
                header_analyzed.build_simple_model(refl)
            else:
                from ..plugins.add_ons.Reflectivity import Plugin as ReflPlugin

                if not "Reflectivity" in self.plugin_control.plugin_handler.loaded_plugins:
                    self.plugin_control.plugin_handler.load_plugin("Reflectivity")
                # create a new script with the reflectivity plugin
                refl: ReflPlugin = self.plugin_control.GetPlugin("Reflectivity")
                header_analyzed.build_reflectivity(refl)

        with self.catch_error(action="open_model", step=f"processing plugins"):
            self.plugin_control.OnOpenModel(None)
        debug("open_model: post new model event")
        _post_new_model_event(self, self.model_control.get_model())
        self.update_title()

    def open_model(self, path):
        debug("open_model: clear model")
        self.paramter_grid.PrepareNewModel()
        self.model_control.new_model()
        # Update all components so all the traces are gone.
        # _post_new_model_event(frame, frame.model)
        debug("open_model: load_file")
        with self.catch_error(action="open_model", step=f"open file {os.path.basename(path)}") as mng:
            self.model_control.load_file(path)
        if not mng.successful:
            return  # don't continue after error

        debug("open_model: read config")
        with self.catch_error(action="open_model", step=f"loading config for plots"):
            [p.ReadConfig() for p in self.get_pages() if hasattr(p, "ReadConfig")]
        with self.catch_error(action="open_model", step=f"loading config for parameter grid"):
            self.paramter_grid.ReadConfig()
            self.mb_checkables[custom_ids.MenuId.TOGGLE_SLIDER].Check(bool(self.paramter_grid.GetValueEditorSlider()))
        debug("open_model: update plugins")
        with self.catch_error(action="open_model", step=f"processing plugins"):
            self.plugin_control.OnOpenModel(None)
        self.main_frame_statusbar.SetStatusText("Model loaded from file", 1)
        self.mb_checkables["none"].Check()  # reset color cycle to None

        # Post an event to update everything else
        debug("open_model: post new model event")
        _post_new_model_event(self, self.model_control.get_model())
        # Needs to put it to saved since all the widgets will have
        # been updated
        self.update_title()

    def get_script_text(self):
        if self.script_file is None:
            return self.script_editor.GetText()
        else:
            text = open(self.script_file, "r", encoding="utf-8").read()
            self.set_script_text(text, from_script=True)
            return text

    def set_script_text(self, text, from_script=False):
        if not from_script and self.script_file is not None:
            open(self.script_file, "w", encoding="utf-8").write(text)

        was_editable = self.script_editor.IsEditable()
        self.script_editor.SetReadOnly(False)

        current_view = self.script_editor.GetFirstVisibleLine()
        current_cursor = self.script_editor.GetCurrentPos()
        current_selection = self.script_editor.GetSelection()
        self.script_editor.SetText(text)
        self.script_editor.SetCurrentPos(current_cursor)
        self.script_editor.SetFirstVisibleLine(current_view)
        self.script_editor.SetSelection(*current_selection)

        self.script_editor.SetReadOnly(not was_editable)

    def open_external_editor(self):
        """
        Save the script as temporary file and open it in an external editor.
        This will block entry in the GUI script editor and read the file on simulate.
        """
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="genx", suffix=".py", encoding="utf-8", delete=False
        ) as sfile:
            sfile.write(self.get_script_text())
            self.script_file = sfile.name
        if not self.opt.editor:
            if not self.eh_mb_select_editor(None):
                os.remove(self.script_file)
                self.script_file = None
                return
        try:
            proc = subprocess.Popen([self.opt.editor, self.script_file])
        except (subprocess.SubprocessError, OSError):
            os.remove(self.script_file)
            self.script_file = None
            self.opt.editor = None
            warning("Could not open editor", exc_info=True)
            return

        self.script_editor.SetReadOnly(True)
        self.script_editor.StyleSetBackground(wx.stc.STC_STYLE_DEFAULT, wx.Colour(210, 210, 210))
        self.mb_editor.SetItemLabel("Reactivate internal editor\tCtrl+E")
        self._editor_proc = proc
        self._script_watcher = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.check_script_file, self._script_watcher)
        self._script_watcher.Start(1000)

    def check_script_file(self, evt):
        txt = open(self.script_file, "r", encoding="utf-8").read()
        if txt.strip() != self.script_editor.GetText().strip():
            self.eh_tb_simulate(None)
        if self._editor_proc and self._editor_proc.poll() is not None:
            self._editor_proc = None
            self._script_watcher.Stop()
            res = ShowQuestionDialog(self, "Editor process exited, reactivate internal editor?", "Editor Closed")
            if res:
                self.deactivate_external_editing()
            else:
                self._script_watcher.Start(1000)

    def deactivate_external_editing(self):
        self.script_editor.SetReadOnly(False)
        self.script_editor.StyleSetBackground(wx.stc.STC_STYLE_DEFAULT, wx.Colour(255, 255, 255))
        self.mb_editor.SetItemLabel("Open in Editor\tCtrl+E")
        self._script_watcher.Stop()
        self._script_watcher = None
        self._editor_proc = None
        os.remove(self.script_file)
        self.script_file = None

    def update_for_save(self):
        """Updates the various objects for a save"""
        self.model_control.set_model_script(self.get_script_text())
        self.paramter_grid.opt.auto_sim = self.mb_checkables[custom_ids.MenuId.AUTO_SIM].IsChecked()
        self.paramter_grid.WriteConfig()

    def do_simulation(self, from_thread=False):
        if not from_thread:
            self.main_frame_statusbar.SetStatusText("Simulating...", 1)
        currecnt_script = self.get_script_text()
        self.model_control.set_model_script(currecnt_script)
        with self.catch_error(action="do_simulation", step=f"simulating the model") as mgr:
            self.model_control.simulate(recompile=not from_thread)

        if mgr.successful:
            wx.CallAfter(_post_sim_plot_event, self, self.model_control.get_model(), "Simulation")
            wx.CallAfter(self.plugin_control.OnSimulate, None)
            if not from_thread:
                self.main_frame_statusbar.SetStatusText("Simulation Sucessful", 1)

    def set_possible_parameters_in_grid(self):
        # Now we should find the parameters that we can use to
        # in the grid
        with self.catch_error(
            action="set_possible_parameters_in_grid", step=f"getting possible parameters", verbose=False
        ) as mgr:
            pardict = self.model_control.get_possible_parameters()
        if not mgr.successful:
            return

        with self.catch_error(
            action="set_possible_parameters_in_grid", step=f"setting parameter selections", verbose=False
        ):
            self.paramter_grid.SetParameterSelections(pardict)
            # Set the function for which the parameter can be evaluated with
            self.paramter_grid.SetEvalFunc(self.model_control.eval_in_model)

    def view_yscale(self, value):
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel < len(pages):
            pages[sel].SetYScale(value)

    def view_xscale(self, value):
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel < len(pages):
            pages[sel].SetXScale(value)

    def activate_cuda(self):
        dlg = wx.ProgressDialog(
            parent=self,
            maximum=3,
            message="Compiling CUDA GPU computing functions with Numba",
            title="Activating CUDA...",
        )
        dlg.Show()

        with self.catch_error("activate CUDA") as eh:
            dlg.Update(1)
            from ..models.lib import paratt_cuda

            dlg.Update(2)
            from ..models.lib import neutron_cuda

            dlg.Update(3)
        if eh.successful:
            from ..models.lib import neutron_refl, paratt

            paratt.Refl = paratt_cuda.Refl
            paratt.ReflQ = paratt_cuda.ReflQ
            paratt.Refl_nvary2 = paratt_cuda.Refl_nvary2
            neutron_refl.Refl = neutron_cuda.Refl
        dlg.Destroy()

    @staticmethod
    def deactivate_cuda():
        from ..models.lib import neutron_numba, neutron_refl, paratt, paratt_numba

        paratt.Refl = paratt_numba.Refl
        paratt.ReflQ = paratt_numba.ReflQ
        paratt.Refl_nvary2 = paratt_numba.Refl_nvary2
        neutron_refl.Refl = neutron_numba.Refl

    def simulation_loop(self):
        """
        Simulation loop for threading to increase the speed of the interactive simulations
        """
        self.flag_simulating = True
        numpy_set_options()  # has to be set, as options are thread dependent
        while self.simulation_queue_counter > 0:
            self.do_simulation(from_thread=True)
            time.sleep(0.1)
            self.simulation_queue_counter = min(1, self.simulation_queue_counter - 1)
        self.flag_simulating = False

    @skips_event
    def eh_external_parameter_value_changed(self, event):
        """
        Event handler for when a value of a parameter in the grid has been updated.
        """
        self.simulation_queue_counter += 1
        if self.mb_checkables[custom_ids.MenuId.AUTO_SIM].IsChecked() and not self.flag_simulating:
            _thread.start_new_thread(self.simulation_loop, ())

    @skips_event
    def eh_external_update_data_grid_choice(self, event):
        """
        Updates the choices of the grids to display from the data.
        """
        data = event.GetData()
        names = [data_set.name for data_set in data]
        self.data_grid_choice.Clear()
        self.data_grid_choice.AppendItems(names)

    @skips_event
    def eh_external_update_data(self, event):
        self.plugin_control.OnDataChanged(event)

    @skips_event
    def eh_new_model(self, event):
        """
        Callback for NEW_MODEL event. Used to update the script for
        a new model i.e. put the string to the correct value.
        """
        # Set the string in the script_editor
        self.set_script_text(event.GetModel().get_script())
        self.script_editor.EmptyUndoBuffer()
        # Let the solver gui do its loading and updating:
        self.model_control.ModelLoaded()
        # Lets update the mb_use_toggle_show Menu item
        self.mb_checkables[custom_ids.MenuId.USE_TOGGLE_SHOW].Check(self.data_list.list_ctrl.opt.toggle_show)
        self.mb_checkables[custom_ids.MenuId.AUTO_SIM].Check(self.paramter_grid.opt.auto_sim)
        # Let other event handlers receive the event as well

    def eh_mb_new(self, event):
        """
        Event handler for creating a new model
        """
        if not self.model_control.saved:
            ans = ShowQuestionDialog(
                self, "If you continue any changes in" " your model will not be saved.", "Model not saved"
            )
            if not ans:
                return

        # Reset the model - remove everything from the previous model
        self.model_control.new_model()
        # Update all components so all the traces are gone.
        _post_new_model_event(self, self.model_control.get_model(), desc="Fresh model")
        self.plugin_control.OnNewModel(None)
        self.main_frame_statusbar.SetStatusText("New model created", 1)
        self.update_title()

    def eh_mb_new_from_file(self, event):
        """
        Event handler for opening a model file...
        """
        # Check so the model is saved before quitting
        if not self.model_control.saved:
            ans = ShowQuestionDialog(
                self, "If you continue any changes in" " your model will not be saved.", "Model not saved"
            )
            if not ans:
                return

        dlg = wx.FileDialog(
            self,
            message="New from file",
            defaultFile="",
            wildcard="Suppoerted types (*.ort/*.orb)|*.ort;*.orb",
            style=wx.FD_OPEN | wx.FD_MULTIPLE | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            paths = dlg.GetPaths()
            debug("new_from_file: path retrieved")
            self.new_from_file(paths)

        dlg.Destroy()

    def eh_mb_open(self, event, directory=""):
        """
        Event handler for opening a model file...
        """
        self.paramter_grid.ClearEditing()
        wx.CallAfter(self.eh_action_open, directory)

    def eh_action_open(self, directory):
        # Check so the model is saved before loading
        if not self.model_control.saved:
            ans = ShowQuestionDialog(
                self, "If you continue any changes in" " your model will not be saved.", "Model not saved"
            )
            if not ans:
                return
        dlg = wx.FileDialog(
            self,
            message="Open",
            defaultDir=directory,
            wildcard="GenX File (*.hgx;*.gx)|*.hgx;*.gx",
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            debug("open: path retrieved")
            self.open_model(path)

        dlg.Destroy()

    def eh_mb_save(self, event):
        """
        Event handler for saving a model file ...
        """
        self.update_for_save()
        fname = self.model_control.get_filename()
        # If model hasn't been saved
        if fname == "":
            # Proceed with calling save as
            self.eh_mb_saveas(event)
        else:
            with self.catch_error(action="save_model", step=f"save file {os.path.basename(fname)}"):
                if len(self.model_control.controller.model_store) > 0:
                    prog = wx.ProgressDialog("Saving...", f"Writing to file\n{fname}\n", maximum=100, parent=self)

                    def update_callback(i, N):
                        prog.Update(int(i / N * 100), f"Writing to file\n{fname}\ndataset {i} of {N}")

                    try:
                        self.model_control.controller.save_file(fname, update_callback=update_callback)
                    finally:
                        prog.Destroy()
                else:
                    self.model_control.controller.save_file(fname)
                self.update_title()

    def eh_mb_publish_plot(self, event):
        dia = pubgraph_dialog.PublicationDialog(
            self,
            data=self.model_control.get_data(),
            module=self.model_control.get_model().eval_in_model('globals().get("model")'),
        )
        dia.ShowModal()

    def eh_mb_print_plot(self, event):
        """
        prints the current plot in the plot notebook.
        """
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel < len(pages):
            pages[sel].Print()

    def eh_mb_print_grid(self, event):
        self.paramter_grid.Print()

    @skips_event
    def eh_mb_print_script(self, event):
        warning("Event handler `eh_mb_print_script' not implemented")

    def eh_mb_export_orso(self, event):
        """
        Exports the data to one file per data set with a basename with
        extention given by a save dialog.
        """
        dlg = wx.FileDialog(
            self,
            message="Export data and model",
            defaultFile="",
            wildcard="ORSO Text File (*.ort)|*.ort",
            style=wx.FD_SAVE | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with self.catch_error(action="export_orso", step=f"export file {os.path.basename(path)}"):
                self.model_control.export_orso(path)
        dlg.Destroy()

    def eh_mb_export_data(self, event):
        """
        Exports the data to one file per data set with a basename with
        extension given by a save dialog.
        """
        dlg = wx.FileDialog(
            self,
            message="Export data",
            defaultFile="",
            wildcard="Dat File (*.dat)|*.dat",
            style=wx.FD_SAVE | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with self.catch_error(action="export_data", step=f"data file {os.path.basename(path)}"):
                self.model_control.export_data(path)

        dlg.Destroy()

    def eh_mb_export_table(self, event):
        """
        Exports the table to a dat file given by a filedialog.
        """
        dlg = wx.FileDialog(
            self,
            message="Export table",
            defaultFile="",
            wildcard="Table File (*.tab)|*.tab",
            style=wx.FD_SAVE | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            fname = dlg.GetPath()
            base, ext = os.path.splitext(fname)
            if ext == "":
                ext = ".tab"
            fname = base + ext
            result = True
            if os.path.exists(fname):
                filepath, filename = os.path.split(fname)
                result = ShowQuestionDialog(
                    self, "The file %s already exists. Do you wish to overwrite it?" % filename, "Overwrite?"
                )
            if result:
                with self.catch_error(action="export_table", step=f"table file {os.path.basename(fname)}"):
                    self.model_control.export_table(fname)

        dlg.Destroy()

    def eh_mb_export_script(self, event):
        """
        Exports the script to a python file given by a filedialog.
        """
        dlg = wx.FileDialog(
            self,
            message="Export script",
            defaultFile="",
            wildcard="Python File (*.py)|*.py",
            style=wx.FD_SAVE | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            fname = dlg.GetPath()
            base, ext = os.path.splitext(fname)
            if ext == "":
                ext = ".py"
            fname = base + ext
            result = True
            if os.path.exists(fname):
                filepath, filename = os.path.split(fname)
                result = ShowQuestionDialog(
                    self, "The file %s already exists. Do you wish to overwrite it?" % filename, "Overwrite?"
                )
            if result:
                with self.catch_error(action="export_orso", step=f"export file {os.path.basename(fname)}"):
                    self.model_control.export_script(fname)

        dlg.Destroy()

    def eh_mb_quit(self, event: wx.CloseEvent):
        """
        Quit the program
        """
        # Check so the model is saved before quitting
        if (not isinstance(event, wx.CloseEvent) or event.CanVeto()) and not self.model_control.saved:
            # stop window from closing if canceled
            ans = ShowQuestionDialog(
                self, "If you continue any changes in your model will not be saved.", "Model not saved"
            )
            if not ans:
                if isinstance(event, wx.CloseEvent):
                    event.Veto()
                return

        self.opt.hsize, self.opt.vsize = self.GetSize()
        self.opt.vsplit = self.ver_splitter.GetSashPosition()
        self.opt.hsplit = self.hor_splitter.GetSashPosition()
        self.opt.psplit = self.plot_splitter.GetSashPosition()
        self.opt.save_config(default=True)

        conf_mod.config.write_default(os.path.join(config_path, "genx.conf"))

        self.findreplace_dlg.Destroy()
        self.findreplace_dlg = None

        if self.script_file:
            os.remove(self.script_file)
            self.script_file = None

        self.Destroy()

    def eh_mb_copy_graph(self, event):
        """
        Callback that copies the current graph in the plot notebook to
        the clipboard.
        """
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel < len(pages):
            pages[sel].CopyToClipboard()

    def eh_mb_copy_sim(self, event):
        """
        Copies the simulation and the data to the clipboard. Note that this
        copies ALL data.
        """
        text_string = self.model_control.get_data_as_asciitable()
        text = wx.TextDataObject(text_string)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(text)
            wx.TheClipboard.Close()

    def eh_mb_copy_table(self, event):
        """
        Copies the table as ascii text to the clipboard
        """
        ascii_table = self.paramter_grid.table.pars.get_ascii_output()
        text_table = wx.TextDataObject(ascii_table)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(text_table)
            wx.TheClipboard.Close()

    @skips_event
    def eh_mb_view_zoom(self, event):
        """
        Takes care of clicks on the toolbar zoom button and the menu item zoom.
        """
        if event.GetId() == custom_ids.ToolId.ZOOM:
            zoom_state = self.main_frame_toolbar.GetToolState(custom_ids.ToolId.ZOOM)
            self.mb_checkables[custom_ids.MenuId.ZOOM].Check(zoom_state)
        else:
            zoom_state = self.mb_checkables[custom_ids.MenuId.ZOOM].IsChecked()
            self.main_frame_toolbar.ToggleTool(custom_ids.ToolId.ZOOM, zoom_state)

        # Synchronize all plots with zoom state
        pages = self.get_pages()
        for page in pages:
            page.SetZoom(zoom_state)

    @skips_event
    def eh_mb_view_grid_slider(self, event):
        """
        Change the state of the grid value input, either as slider or as a number.
        """
        val = self.mb_checkables[custom_ids.MenuId.TOGGLE_SLIDER].IsChecked()
        self.paramter_grid.SetValueEditorSlider(val)
        self.paramter_grid.toggle_slider_tool(val)
        self.paramter_grid.Refresh()

    def eh_mb_fit_start(self, event):
        """
        Event handler to start fitting
        """
        with self.catch_error(action="fit_start", step=f"starting fit"):
            self.model_control.StartFit()

    def eh_mb_fit_stop(self, event):
        """
        Event handler to stop the fitting routine
        """
        self.model_control.StopFit()

    def eh_mb_fit_resume(self, event):
        """
        Event handler to resume the fitting routine. No initilization.
        """
        with self.catch_error(action="fit_resume", step=f"resume fit"):
            self.model_control.ResumeFit()

    @skips_event
    def eh_mb_fit_analyze(self, event):
        warning("Event handler `eh_mb_fit_analyze' not implemented")

    def eh_mb_misc_examples(self, event):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples", "")
        self.eh_mb_open(event, directory=path)

    def eh_mb_misc_showman(self, event):
        webbrowser.open_new(manual_url)

    def eh_mb_misc_about(self, event):
        """
        Show an about box about GenX with some info...
        """
        import platform

        import matplotlib
        import numpy
        import scipy

        useful = ""
        try:
            import orsopy

            useful += "ORSOpy: %s, " % orsopy.__version__
        except ImportError:
            pass
        try:
            # noinspection PyUnresolvedReferences
            import numba

            useful += "Numba: %s, " % numba.__version__
        except ImportError:
            pass
        try:
            # noinspection PyUnresolvedReferences
            import vtk

            # noinspection PyUnresolvedReferences
            useful += "VTK: %s, " % vtk.vtkVersion.GetVTKVersion()
        except ImportError:
            pass
        try:
            # noinspection PyUnresolvedReferences
            import bumps

            useful += "Bumps: %s, " % bumps.__version__
        except ImportError:
            pass

        info_dilog = wx.adv.AboutDialogInfo()
        info_dilog.SetName("GenX")
        info_dilog.SetVersion(program_version)
        info_dilog.SetCopyright("(C) 2008 Matts Bjorck; 2020 Artur Glavic")
        info_dilog.SetDescription(
            wordwrap(
                "GenX is a multipurpose refinement program using the differential "
                "evolution algorithm. It is developed  mainly for refining x-ray reflectivity "
                "and neutron reflectivity data."
                "\n\nConfiguration files stored in %s" % config_path
                + "\n\nThe versions of the mandatory libraries are:\n"
                "Python: %s, wxPython: %s, Numpy: %s, Scipy: %s, Matplotlib: %s"
                "\n\nThe non-mandatory but useful packages:\n%s"
                ""
                % (
                    platform.python_version(),
                    wx.__version__,
                    numpy.__version__,
                    scipy.__version__,
                    matplotlib.__version__,
                    useful,
                ),
                500,
                wx.ClientDC(self),
            )
        )
        info_dilog.WebSite = (homepage_url, "GenX homepage")
        # No developers yet
        info_dilog.SetDevelopers(["Artur Glavic <artur.glavic@psi.ch>"])
        info_dilog.SetLicence(
            wordwrap(
                "This program is free software: you can redistribute it and/or modify "
                "it under the terms of the GNU General Public License as published by "
                "the Free Software Foundation, either version 3 of the License, or "
                "(at your option) any later version. "
                "\n\nThis program is distributed in the hope that it will be useful, "
                "but WITHOUT ANY WARRANTY; without even the implied warranty of "
                "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the "
                "GNU General Public License for more details. "
                "\n\nYou should have received a copy of the GNU General Public License "
                "along with this program.  If not, see <http://www.gnu.org/licenses/>. ",
                400,
                wx.ClientDC(self),
            )
        )

        wx.adv.AboutBox(info_dilog)

    def eh_mb_debug_dialog(self, event):
        import logging

        from ..core import custom_logging

        logger = logging.getLogger()
        level = logger.getEffectiveLevel()
        logger.setLevel(logging.DEBUG)

        dlg = wx.FileDialog(
            self,
            message="Save Logfile As",
            defaultFile="genx.log",
            wildcard="GenX logfile (*.log)|*.log",
            style=wx.FD_SAVE,
        )
        if dlg.ShowModal() == wx.ID_OK:
            fname = dlg.GetPath()
            dlg.Destroy()
            if os.path.exists(fname):
                res = ShowQuestionDialog(self, f"File {os.path.basename(fname)} exists, overwrite?", "Overwrite file?")
                if res:
                    custom_logging.activate_logging(fname)
            else:
                custom_logging.activate_logging(fname)
        # open logging console dialog
        from .log_dialog import LoggingDialog

        dlg = LoggingDialog(self)
        dlg.Show()

    def eh_mb_saveas(self, event):
        """
        Event handler for save as ...
        """
        dlg = wx.FileDialog(
            self,
            message="Save As",
            defaultFile="",
            wildcard="HDF5 GenX File (*.hgx)|*.hgx|GenX File (*.gx)|*.gx",
            style=wx.FD_SAVE | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            self.update_for_save()
            fname = dlg.GetPath()
            base, ext = os.path.splitext(fname)
            if ext == "":
                ext = ".hgx"
            fname = base + ext
            result = True
            if os.path.exists(fname):
                filepath, filename = os.path.split(fname)
                result = ShowQuestionDialog(
                    self, "The file %s already exists. Do you wish to overwrite it?" % filename, "Overwrite?"
                )
            if result:
                with self.catch_error(action="saveas", step=f"saveing file as {os.path.basename(fname)}"):
                    if len(self.model_control.controller.model_store) > 0:
                        prog = wx.ProgressDialog("Saving...", f"Writing to file\n{fname}\n", maximum=100, parent=self)

                        def update_callback(i, N):
                            prog.Update(int(i / N * 100), f"Writing to file\n{fname}\ndataset {i} of {N}")

                        try:
                            self.model_control.controller.save_file(fname, update_callback=update_callback)
                        finally:
                            prog.Destroy()
                    else:
                        self.model_control.controller.save_file(fname)
                self.update_title()
        dlg.Destroy()

    def eh_mb_batch(self, event):
        dia = BatchDialog(self, self.model_control)
        ssize = self.GetSize()
        dia.SetSize(wx.Size(ssize.width // 3, int(ssize.height * 0.8)))
        dia.Show()

    def eh_mb_view_color_cycle(self, event):
        id2colors = dict(((self.mb_checkables[key].GetId(), value) for key, value in COLOR_CYCLES.items()))
        self.model_control.update_color_cycle(id2colors[event.GetId()])

    def eh_mb_view_yscale_log(self, event):
        self.view_yscale("log")

    def eh_mb_view_yscale_linear(self, event):
        self.view_yscale("linear")

    def eh_mb_view_xscale_log(self, event):
        """
        Set the x-scale of the current plot. type should be linear or log, strings.
        """
        self.view_xscale("log")

    def eh_mb_view_xscale_linear(self, event):
        self.view_xscale("linear")

    def eh_mb_view_autoscale(self, event):
        """on_autoscale(frame, event) --> None

        Toggles the autoscale of the current plot.
        """
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel < len(pages):
            pages[sel].SetAutoScale(not pages[sel].GetAutoScale())

    def eh_data_grid_choice(self, event):
        """
        change the data displayed in the grid...
        """
        dataset = self.model_control.get_data()[event.GetSelection()]
        rows = self.data_grid.GetNumberRows()
        new_rows = max(len(dataset.x), len(dataset.y), len(dataset.x_raw), len(dataset.y_raw))
        self.data_grid.DeleteRows(numRows=rows)
        self.data_grid.AppendRows(new_rows)
        [[self.data_grid.SetCellValue(row, col, "-") for col in range(6)] for row in range(new_rows)]
        [self.data_grid.SetCellValue(row, 0, "%.3e" % dataset.x_raw[row]) for row in range(len(dataset.x_raw))]
        [self.data_grid.SetCellValue(row, 1, "%.3e" % dataset.y_raw[row]) for row in range(len(dataset.y_raw))]
        [self.data_grid.SetCellValue(row, 2, "%.3e" % dataset.error_raw[row]) for row in range(len(dataset.error_raw))]
        [self.data_grid.SetCellValue(row, 3, "%.3e" % dataset.x[row]) for row in range(len(dataset.x))]
        [self.data_grid.SetCellValue(row, 4, "%.3e" % dataset.y[row]) for row in range(len(dataset.y))]
        [self.data_grid.SetCellValue(row, 5, "%.3e" % dataset.error[row]) for row in range(len(dataset.error))]

    def eh_tb_new(self, event):
        self.eh_mb_new(event)

    def eh_tb_new_from_file(self, event):
        self.eh_mb_new_from_file(event)

    def eh_tb_open(self, event):
        self.eh_mb_open(event)

    def eh_tb_save(self, event):
        self.eh_mb_save(event)

    def eh_tb_simulate(self, event):
        """
        Event handler for simulation.
        """
        self.flag_simulating = True
        self.do_simulation()
        self.set_possible_parameters_in_grid()
        self.flag_simulating = False

    def eh_tb_select_solver(self, event):
        self.model_control.set_solver(event.GetString())
        self.paramter_grid.Refresh()

    def eh_tb_start_fit(self, event):
        self.eh_mb_fit_start(event)

    def eh_tb_stop_fit(self, event):
        self.eh_mb_fit_stop(event)

    def eh_tb_restart_fit(self, event):
        self.eh_mb_fit_resume(event)

    def eh_tb_zoom(self, event):
        self.eh_mb_view_zoom(event)

    def eh_ex_set_solver_selection(self, selection):
        select: wx.ComboBox = self.FindWindowById(custom_ids.ToolId.SOLVER_SELECT, self.main_frame_toolbar)
        select.SetValue(selection)

    def eh_ex_add_solver_selection(self, selection):
        select: wx.ComboBox = self.FindWindowById(custom_ids.ToolId.SOLVER_SELECT, self.main_frame_toolbar)
        select.Append(selection)

    @skips_event
    def eh_ex_status_text(self, event):
        self.main_frame_statusbar.SetStatusText(event.text, 1)

    def eh_ex_point_pick(self, event):
        self.main_frame_statusbar.SetStatusText(event.text, 2)

    @skips_event
    def eh_ex_plot_settings_changed(self, event):
        """
        Callback for the settings change event for the current plot
         - change the toggle for the zoom icon and change the menu items.
        """
        self.main_frame_toolbar.ToggleTool(custom_ids.ToolId.ZOOM, event.zoomstate)
        self.mb_checkables[custom_ids.MenuId.ZOOM].Check(event.zoomstate)
        if event.yscale == "log":
            self.mb_checkables[custom_ids.MenuId.Y_SCALE_LOG].Check(True)
        elif event.yscale == "linear":
            self.mb_checkables[custom_ids.MenuId.Y_SCALE_LIN].Check(True)
        if event.xscale == "log":
            self.mb_checkables[custom_ids.MenuId.X_SCALE_LOG].Check(True)
        elif event.xscale == "linear":
            self.mb_checkables[custom_ids.MenuId.X_SCALE_LIN].Check(True)
        self.mb_checkables[custom_ids.MenuId.AUTO_SCALE].Check(event.autoscale)

    def eh_tb_calc_error_bars(self, event):
        """
        callback to calculate the error bars on the data.
        """
        with self.catch_error(action="calc_error_bars", step=f"calculating errorbars"):
            error_values = self.model_control.CalcErrorBars()
            self.model_control.set_error_pars(error_values)
            self.paramter_grid.SetParameters(self.model_control.get_parameters())
            self.main_frame_statusbar.SetStatusText("Errorbars calculated", 1)

    def eh_tb_error_stats(self, event):
        with self.catch_error(action="error_stats", step=f"opening Bumps analysis dialog"):
            from .bumps_interface import StatisticalAnalysisDialog

            prev_result = getattr(self.model_control.controller.optimizer, "last_result", None)
            if prev_result and len(prev_result.dx) != len(self.model_control.get_parameters().get_fit_pars()[0]):
                # fit parameters were changed since result was computed
                prev_result = None
            dia = StatisticalAnalysisDialog(self, self.model_control.get_model(), prev_result=prev_result)
            dia.ShowModal()
        self.paramter_grid.grid.ForceRefresh()

    @skips_event
    def eh_plot_page_changed(self, event):
        """plot_page_changed(frame, event) --> None

        Callback for page change in plot notebook. Changes the state of
        the zoom toggle button.
        """
        sel = event.GetSelection()
        pages = self.get_pages()
        if sel < len(pages):
            zoom_state = pages[sel].GetZoom()
            # Set the zoom button to the correct value
            self.main_frame_toolbar.ToggleTool(custom_ids.ToolId.ZOOM, zoom_state)
            self.mb_checkables[custom_ids.MenuId.ZOOM].Check(zoom_state)

            yscale = pages[sel].GetYScale()
            if yscale == "log":
                self.mb_checkables[custom_ids.MenuId.Y_SCALE_LOG].Check(True)
            elif yscale == "linear":
                self.mb_checkables[custom_ids.MenuId.Y_SCALE_LIN].Check(True)
            xscale = pages[sel].GetXScale()
            if xscale == "log":
                self.mb_checkables[custom_ids.MenuId.X_SCALE_LOG].Check(True)
            elif xscale == "linear":
                self.mb_checkables[custom_ids.MenuId.X_SCALE_LIN].Check(True)

    @skips_event
    def eh_mb_view_zoomall(self, event):
        """zoomall(self, event) --> None

        Zoom out and show all data points
        """
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel < len(pages):
            tmp = pages[sel].GetAutoScale()
            pages[sel].SetAutoScale(True)
            pages[sel].AutoScale()
            pages[sel].SetAutoScale(tmp)
            pages[sel].AutoScale()

    def eh_mb_use_cuda(self, event):
        if self.mb_checkables[custom_ids.MenuId.TOGGLE_CUDA].IsChecked():
            self.activate_cuda()
        else:
            self.deactivate_cuda()

    def eh_mb_set_opt(self, event):
        self.model_control.ParametersDialog(self)

    def eh_mb_import_data(self, event):
        """
        callback to import data into the program
        """
        with self.catch_error(action="import_data", step=f"open data file"):
            self.data_list.eh_tb_open(event)

    def eh_mb_import_table(self, event):
        """
        imports a table from the file given by a file dialog box
        """
        dlg = wx.FileDialog(
            self,
            message="Import script",
            defaultFile="",
            wildcard="Table File (*.tab)|*.tab|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with self.catch_error(action="import_table", step=f"importing table {os.path.basename(path)}") as mgr:
                self.model_control.import_table(path)
            if not mgr.successful:
                dlg.Destroy()
                return
        dlg.Destroy()
        # Post event to tell that the model has changed
        _post_new_model_event(self, self.model_control.get_model())
        self.main_frame_statusbar.SetStatusText("Table imported from file", 1)

    def eh_mb_import_script(self, event):
        """
        imports a script from the file given by a file dialog box
        """
        dlg = wx.FileDialog(
            self,
            message="Import script",
            defaultFile="",
            wildcard="Python files (*.py)|*.py|All files (*.*)|*.*",
            style=wx.FD_OPEN | wx.FD_CHANGE_DIR,
        )
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            with self.catch_error(action="import_script", step=f"importing file {os.path.basename(path)}"):
                self.model_control.import_script(path)
                self.plugin_control.OnOpenModel(None)
        dlg.Destroy()
        # Post event to tell that the model has changed
        _post_new_model_event(self, self.model_control.get_model())

    @skips_event
    def eh_external_fom_value(self, event):
        """
        Callback to update the fom_value displayed by the gui
        """
        if hasattr(event, "fom_value"):
            fom_value = event.fom_value
            fom_name = event.fom_name
        else:
            # workaround for GenericModelEvent, TODO: fix this in the future with better event
            fom_value = self.model_control.get_fom()
            fom_name = self.model_control.get_fom_name()
        if fom_value:
            self.main_frame_fom_text.SetLabel("        FOM %s: %.4e" % (fom_name, fom_value))
        else:
            self.main_frame_fom_text.SetLabel("        FOM %s: None" % fom_name)
        # Hard code the events for the plugins so that they can be run synchronously.
        # This is important since the Reflectivity model, for example, relies on the
        # current state of the model.
        try:
            self.plugin_control.OnFittingUpdate(event)
        except Exception as e:
            iprint("Error in plot output:\n" + repr(e))

    def eh_mb_set_dal(self, event):
        self.data_list.DataLoaderSettingsDialog()

    def eh_mb_fit_evaluate(self, event):
        """
        Event handler for only evaluating the Sim function - no recompiling
        """
        self.flag_simulating = True
        self.main_frame_statusbar.SetStatusText("Simulating...", 1)
        # Compile is not necessary when using simulate...
        with self.catch_error(action="fit_evaluate", step=f"simulating model"):
            self.do_simulation()
            self.set_possible_parameters_in_grid()
            data = self.model_control.get_data()
            sims1 = [di.y_sim for di in data]
            self.model_control.simulate(recompile=False)
            data = self.model_control.get_data()
            sims2 = [di.y_sim for di in data]
            _post_sim_plot_event(self, self.model_control.get_model(), "Evaluation")
            diffs = [(si1 != si2).any() for si1, si2 in zip(sims1, sims2)]
            if any(diffs):
                ShowNotificationDialog(
                    self,
                    f"Issue in simulation, there were differences in the "
                    f"first and second evaluation\n of the model for datasets "
                    f"{[i for i, diff in enumerate(diffs) if diff]}\n"
                    f"This is often caused by changing a parameter"
                    f"in the Sim function\nwithout resetting it at the "
                    f"bottom/top.\n\n"
                    f"If you try to fit this will lead to unpredictable results.",
                )
            self.plugin_control.OnSimulate(None)
        self.flag_simulating = False

    def eh_data_new_set(self, event):
        self.data_list.eh_tb_add(event)

    def eh_data_new_simulation_set(self, event):
        self.data_list.eh_tb_add_simulation(event)

    def eh_data_delete(self, event):
        self.data_list.eh_tb_delete(event)

    def eh_data_move_down(self, event):
        self.data_list.list_ctrl.MoveItemDown()

    def eh_data_move_up(self, event):
        self.data_list.list_ctrl.MoveItemUp()

    def eh_data_toggle_show(self, event):
        self.data_list.list_ctrl.OnShowData(event)

    def eh_data_toggle_use(self, event):
        self.data_list.list_ctrl.OnUseData(event)

    def eh_data_toggle_error(self, event):
        self.data_list.list_ctrl.OnUseError(event)

    def eh_data_calc(self, event):
        self.data_list.list_ctrl.OnCalcEdit(event)

    def eh_data_import(self, event):
        self.data_list.list_ctrl.OnImportSettings(event)

    def eh_data_plots(self, event):
        self.data_list.list_ctrl.OnPlotSettings(event)

    def eh_mb_models_help(self, event):
        """
        Show a help dialog for information about the different models.
        """
        dlg = help.PluginHelpDialog(self, "models", title="Models help")
        current_model = self.model_control.get_model_name()
        if current_model in dlg.choice.GetStrings():
            dlg.choice.SetStringSelection(current_model)
            dlg.on_choice(None)
        dlg.Show()

    @skips_event
    def eh_external_model_changed(self, event):
        """
        callback when something has changed in the model so that the
        user can be made aware that the model needs saving.
        """
        if self._init_phase:
            return
        try:
            self.model_control.saved = (not event.permanent_change) and self.model_control.saved
        except AttributeError:
            self.model_control.saved = False
        else:
            self.plugin_control.OnGridChanged(event)
        self.update_title()

    def eh_mb_plugins_help(self, event):
        """
        Show a help dialog for information about the different plugins.
        """
        dlg = help.PluginHelpDialog(self, "plugins.add_ons", title="Plugins help")
        dlg.Show()

    def eh_mb_data_loaders_help(self, event):
        """
        Show a help dialog for information about the different data_loaders.
        """
        dlg = help.PluginHelpDialog(self, "plugins.data_loaders", title="Data loaders help")
        dlg.Show()

    def eh_mb_findreplace(self, event):
        self.findreplace_dlg.Show(True)

    def eh_mb_open_editor(self, event):
        if self.script_file is not None:
            self.deactivate_external_editing()
            return
        self.open_external_editor()

    def eh_external_find(self, event):
        """callback for find events - coupled to the script"""
        evtype = event.GetEventType()

        def find():
            find_str = event.GetFindString()
            flags = event.GetFlags()
            if flags & 1:
                ##print "Searching down"
                pos = self.script_editor.SearchNext(flags, find_str)
            else:
                ##print "Searching up"
                pos = self.script_editor.SearchPrev(flags, find_str)
            if pos == -1:
                self.main_frame_statusbar.SetStatusText("Could not find text %s" % find_str, 1)
            return pos

        def replace():
            replace_str = event.GetReplaceString()
            self.script_editor.ReplaceSelection(replace_str)

        # Deal with the different cases
        if evtype == wx.wxEVT_COMMAND_FIND:
            self.script_editor.SearchAnchor()
            find()

        elif evtype == wx.wxEVT_COMMAND_FIND_NEXT:
            pnew = self.script_editor.GetSelectionEnd()
            self.script_editor.GotoPos(pnew)
            self.script_editor.SetAnchor(pnew)
            self.script_editor.SearchAnchor()
            find()

        elif evtype == wx.wxEVT_COMMAND_FIND_REPLACE:
            # If we do not have found text already
            # or if we have marked other text by mistake...
            if self.script_editor.GetSelectedText() != event.GetFindString():
                find()
            # We already have found and marked text that we should
            # replace
            else:
                self.script_editor.ReplaceSelection(event.GetReplaceString())
                # Find a new text to replace
                find()
        elif evtype == wx.wxEVT_COMMAND_FIND_REPLACE_ALL:
            if self.script_editor.GetSelectedText() != event.GetFindString():
                pos = find()
            else:
                pos = -1
            i = 0
            while pos != -1:
                self.script_editor.ReplaceSelection(event.GetReplaceString())
                i += 1
                pos = find()
            self.main_frame_statusbar.SetStatusText("Replaces %d occurancies of  %s" % (i, event.GetFindString()), 1)

        else:
            raise ValueError(f"Faulty event supplied in find and repalce functionallity: {event}")
        # This will scroll the editor to the right position so we can see
        # the text
        self.script_editor.EnsureCaretVisible()

    def eh_mb_fom_help(self, event):
        """
        Show a help dialog for information about the different fom.
        """
        dlg = help.PluginHelpDialog(self, "fom_funcs", title="FOM functions help")
        dlg.Show()

    def eh_mb_view_use_toggle_show(self, event):
        new_val = self.mb_checkables[custom_ids.MenuId.USE_TOGGLE_SHOW].IsChecked()
        self.data_list.list_ctrl.SetShowToggle(new_val)

    def eh_mb_misc_openhomepage(self, event):
        webbrowser.open_new(homepage_url)

    def eh_show_startup_dialog(self, event):
        pre_dia = self.wstartup.copy()
        self.startup_dialog(config_path, force_show=True)
        # print(pre_dia==self.wstartup)

    def eh_mb_select_editor(self, event):
        dlg = wx.FileDialog(self, message="Select Editor Executable", defaultFile="", style=wx.FD_OPEN)
        if dlg.ShowModal() == wx.ID_OK:
            path = dlg.GetPath()
            debug("open_external_editor: path retrieved")
            self.opt.editor = path
            return True
        else:
            return False

    def eh_mb_fit_autosim(self, event):
        event.Skip()

    def check_for_update(self):
        same_version = True
        with self.catch_error(action="update_check", step=f"check_version"):
            same_version = check_version()
            self.opt.last_update_check = time.time()
        if same_version:
            return
        with self.catch_error(action="update_check", step=f"show_dialog"):
            dia = VersionInfoDialog(self)
            res = dia.ShowModal()
            if res == wx.ID_OK:
                ShowNotificationDialog(self, "You need to restart GenX for the changes to take effect.")
            elif res == wx.ID_DELETE:
                self.Destroy()


class GenxFileDropTarget(wx.FileDropTarget):
    parent: GenxMainWindow

    def __init__(self, parent):
        self.parent = parent
        wx.FileDropTarget.__init__(self)

    def OnDropFiles(self, x, y, filenames):
        model_file = filenames[0]
        if model_file.lower().endswith(".hgx") or model_file.lower().endswith(".gx"):
            # Check so the model is saved before quitting
            if not self.parent.model_control.saved:
                ans = ShowQuestionDialog(
                    self.parent, "If you continue any changes in your model will not be saved.", "Model not saved"
                )
                if not ans:
                    return False
            self.parent.open_model(model_file)
            return True
        if model_file.lower().endswith(".ort") or model_file.lower().endswith(".orb"):
            # Check so the model is saved before quitting
            if not self.parent.model_control.saved:
                ans = ShowQuestionDialog(
                    self.parent, "If you continue any changes in your model will not be saved.", "Model not saved"
                )
                if not ans:
                    return False
            self.parent.new_from_file(filenames)
            return True
        return False


class GenxApp(wx.App):

    def __init__(self, filename=None, dpi_overwrite=None):
        debug("App init started")
        self.open_file = filename
        self.dpi_overwrite = dpi_overwrite
        self._first_init = True
        wx.App.__init__(self, redirect=False)
        if hasattr(wx, "OSX_FILEDIALOG_ALWAYS_SHOW_TYPES"):
            wx.SystemOptions.SetOption(wx.OSX_FILEDIALOG_ALWAYS_SHOW_TYPES, 1)
        debug("App init complete")

    def ConnectExceptionHandler(self):
        """
        Create a custom logging handler that opens a message dialog on critical (unhandled) exceptions.
        """
        self._exception_handler = GuiExceptionHandler(self)
        logging.getLogger().addHandler(self._exception_handler)

    def ShowSplash(self):
        debug("Display Splash Screen")
        image = wx.Bitmap(img.getgenxImage().Scale(400, 400))
        self.splash = wx.adv.SplashScreen(image, wx.adv.SPLASH_CENTER_ON_SCREEN, wx.adv.SPLASH_NO_TIMEOUT, None)
        wx.YieldIfNeeded()

    def WriteSplash(self, text, progress=0.0):
        image = self.splash.GetBitmap()
        self._draw_bmp(image, text, progress=progress)
        self.splash.Refresh()
        self.splash.Update()
        wx.YieldIfNeeded()

    @staticmethod
    def _draw_bmp(bmp, txt, progress=0.0):
        w, h = 400, 400
        dc = wx.MemoryDC()
        dc.SelectObject(bmp)
        gc = wx.GraphicsContext.Create(dc)
        font = wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT)
        gc.SetFont(font, wx.Colour(0, 0, 0))
        gc.SetBrush(wx.Brush(wx.Colour(255, 255, 255)))
        gc.DrawRectangle(30, 0, 370, font.GetPixelSize().height + 4)
        if progress > 0:
            gc.SetBrush(wx.Brush(wx.Colour(252, 175, 62)))
            gc.DrawRectangle(30, 0, int(progress * 370), font.GetPixelSize().height + 4)
        tw, th = gc.GetTextExtent(txt)
        gc.DrawText(txt, (w - tw) // 2, 0)
        dc.SelectObject(wx.NullBitmap)

    def OnInit(self):
        first_init = self._first_init
        if first_init:
            locale = wx.Locale(wx.LANGUAGE_ENGLISH_US)
            self.locale = locale
            self._first_init = False
        self.ConnectExceptionHandler()
        self.ShowSplash()
        debug("entering init phase")

        self.WriteSplash("initializeing main window...")
        main_frame = GenxMainWindow(self, dpi_overwrite=self.dpi_overwrite)
        self.SetTopWindow(main_frame)
        main_frame.SetMinSize(wx.Size(600, 400))

        from genx.models.lib import USE_NUMBA

        if USE_NUMBA:
            try:
                import numba
            except ImportError:
                pass
            else:
                from ..models.lib.numba_integration import configure_numba

                configure_numba()

                import inspect

                # load numba modules, show progress as in case they aren't cached it takes some seconds
                self.WriteSplash("compiling numba functions...", progress=0.25)
                real_jit = numba.jit

                class UpdateJit:
                    update_counter = 1
                    WriteSplash = self.WriteSplash

                    def __call__(self, *args, **opts):
                        if inspect.stack()[1][3] != "<lambda>":
                            self.WriteSplash(
                                f"compiling numba functions {self.update_counter}/21",
                                progress=0.25 + 0.5 * (self.update_counter - 1) / 21.0,
                            )
                            self.update_counter += 1
                            wx.YieldIfNeeded()
                        return real_jit(*args, **opts)

                numba.jit = UpdateJit()
                from ..models.lib import instrument_numba, neutron_numba, offspec, paratt_numba, surface_scattering

                numba.jit = real_jit

        if self.open_file is None:
            self.splash.Destroy()
            if first_init:
                main_frame.startup_dialog(config_path)
            self.ShowSplash()
        else:
            wx.CallAfter(self.WriteSplash, f"loading file {os.path.basename(self.open_file)}...", progress=0.8)
            if self.open_file.endswith(".ort"):
                wx.CallAfter(self.WriteSplash, "load default plugins...", progress=0.9)
                wx.CallAfter(main_frame.plugin_control.LoadDefaultPlugins)
                wx.CallAfter(main_frame.new_from_file, [self.open_file])
            else:
                wx.CallAfter(main_frame.open_model, self.open_file)
            wx.CallAfter(self.WriteSplash, "display main window...", progress=0.9)
            wx.CallAfter(main_frame.Show)
            wx.CallAfter(self.splash.Destroy)
            return 1

        debug("init complete")
        wx.CallAfter(self.WriteSplash, "load default plugins...", progress=0.8)
        wx.CallAfter(main_frame.plugin_control.LoadDefaultPlugins)
        wx.CallAfter(self.WriteSplash, "display main window...", progress=0.9)
        wx.CallAfter(main_frame.Show)
        if time.time() - main_frame.opt.last_update_check > (7 * 24 * 3600):
            wx.CallAfter(self.WriteSplash, "checking for update...", progress=0.95)
            wx.CallAfter(main_frame.check_for_update)
        wx.CallAfter(self.splash.Destroy)
        wx.CallLater(100, main_frame.model_control.SetModelSaved)
        return 1

    def OnRebuild(self):
        # close current main window and create new one from scratch
        main_frame: GenxMainWindow = self.GetTopWindow()
        wx.CallAfter(main_frame.Close, True)
        wx.CallAfter(self.Rebuild)

    def Rebuild(self):
        custom_ids.rebuild_IDs()
        self.OnInit()

    def Restart(self):
        import subprocess
        import sys

        subprocess.Popen([sys.executable, "-m", "genx.run"])


class StartUpConfigDialog(wx.Dialog):

    def __init__(self, parent, config_folder, show_cb=True, wide=False, wx_ploting=False):
        wx.Dialog.__init__(self, parent, -1, "Change Startup Configuration")

        self.config_folder = config_folder
        self.selected_config = None

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add((-1, 10), 0, wx.EXPAND)

        sizer.Add(
            wx.StaticText(self, label="Choose the profile you want GenX to use:            "), 0, wx.ALIGN_LEFT, 5
        )
        self.profiles = self.get_possible_configs()
        self.config_list = wx.ListBox(self, size=(-1, 200), choices=self.profiles, style=wx.LB_SINGLE)
        self.config_list.SetSelection(self.profiles.index("SimpleReflectivity"))
        sizer.Add(self.config_list, 1, wx.GROW | wx.TOP, 5)

        startup_cb = wx.CheckBox(self, -1, "Show at startup", style=wx.ALIGN_LEFT)
        startup_cb.SetValue(show_cb)
        self.startup_cb = startup_cb
        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(startup_cb, 0, wx.EXPAND, 5)
        wide_cb = wx.CheckBox(self, -1, "Widescreen (need restart)", style=wx.ALIGN_LEFT)
        wide_cb.SetValue(wide)
        self.wide_cb = wide_cb
        sizer.Add(wide_cb, 0, wx.EXPAND, 5)
        wx_plot = wx.CheckBox(self, -1, "wxPlot (tech preview)", style=wx.ALIGN_LEFT)
        wx_plot.SetValue(wx_ploting)
        self.wx_plot = wx_plot
        sizer.Add(wx_plot, 0, wx.EXPAND, 5)

        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(
            wx.StaticText(self, label="These settings can be changed at the menu:\n Options/Startup Profile"),
            0,
            wx.ALIGN_LEFT,
            5,
        )

        # Add the Dialog buttons
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        # Add some event handlers
        self.Bind(wx.EVT_BUTTON, self.OnClickOkay, okay_button)

        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.TOP, 20)

        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(button_sizer, 0, flag=wx.ALIGN_RIGHT, border=20)
        sizer.Add((-1, 4), 0, wx.EXPAND)

        main_sizer = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add((10, -1), 0, wx.EXPAND)
        main_sizer.Add(sizer, 1, wx.EXPAND)
        main_sizer.Add((10, -1), 0, wx.EXPAND)
        self.SetSizer(main_sizer)

        sizer.Fit(self)
        self.Layout()
        self.CentreOnScreen()

    @skips_event
    def OnClickOkay(self, event):
        self.selected_config = self.profiles[self.config_list.GetSelection()]
        self.show_at_startup = self.startup_cb.GetValue()
        self.widescreen = self.wide_cb.GetValue()
        self.wxPlot = self.wx_plot.GetValue()

    def GetConfigFile(self):
        if self.selected_config:
            return self.selected_config + ".conf"
        else:
            return None

    def GetShowAtStartup(self):
        return self.show_at_startup

    def GetWidescreen(self):
        return self.widescreen

    def GetWxPloting(self):
        return self.wxPlot

    def get_possible_configs(self) -> List[str]:
        """
        search the plugin directory.
        Checks the list for python scripts and returns a list of
        module names that are loadable .
        """
        plugins = [s[:-5] for s in os.listdir(self.config_folder) if ".conf" == s[-5:] and s[:2] != "__"]
        return plugins


# =============================================================================
# Custom events needed for updating and message parsing between the different
# modules.


class GenericModelEvent(wx.CommandEvent):
    """
    Event class for a new model - for updating
    of the paramters, plots and script.
    """

    def __init__(self, evt_type, evt_id, model):
        wx.CommandEvent.__init__(self, evt_type, evt_id)
        self.model = model
        self.description = ""

    def GetModel(self):
        return self.model

    def SetModel(self, model):
        self.model = model

    def SetDescription(self, desc):
        """
        Set a string that describes the event that has occurred
        """
        self.description = desc


def _post_new_model_event(parent, model, desc=""):
    # Send an event that a new data set has been loaded
    evt = GenericModelEvent(new_model_type, parent.GetId(), model)
    evt.SetDescription(desc)
    # Process the event!
    parent.GetEventHandler().ProcessEvent(evt)


def _post_sim_plot_event(parent, model, desc=""):
    # Send an event that a new data set ahs been loaded
    evt = GenericModelEvent(sim_plot_type, parent.GetId(), model)
    evt.SetDescription(desc)
    # Process the event!
    parent.GetEventHandler().ProcessEvent(evt)
