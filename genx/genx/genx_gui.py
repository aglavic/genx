'''
Main GenX window and functionality.
'''

import appdirs
import io
import os
import shutil
import sys
import traceback
import webbrowser
import _thread, time
from logging import debug, error, info, warning
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from typing import List

import wx
import wx.adv
import wx.grid
import wx.py
import wx.stc
from wx.lib.wordwrap import wordwrap

from genx.plugins import add_on_framework as add_on
from . import datalist, filehandling as io, images as img, model, parametergrid, plotpanel, solvergui, help
from .version import __version__ as program_version
from .exceptions import GenxError, GenxIOError, ErrorBarError
from .gui_logging import iprint

# import wx.lib.agw.aui as aui

# Add current path to the system paths
# just in case some user make a directory change
sys.path.append(os.getcwd())
_path, _file=os.path.split(__file__)
if _path[-4:]=='.zip':
    _path, ending=os.path.split(_path)

# Get the configuration path, create if it not exists
config_path=appdirs.user_data_dir('GenX3', 'ArturGlavic')+'/'
info(config_path)
version_file=os.path.join(config_path, 'genx.version')
if not os.path.exists(config_path):
    info('Creating path: %s'%config_path)
    os.makedirs(config_path)
if not os.path.exists(os.path.join(config_path, 'profiles')):
    info('Creating path: %s'%os.path.join(config_path, 'profiles'))
    shutil.copytree(os.path.join(_path, 'profiles'),
                    os.path.join(config_path, 'profiles'))
    open(version_file, 'w').write(program_version+'\n')
elif not os.path.exists(version_file) or \
        open(version_file, 'r').read().rsplit('.', 1)[0]!=program_version.rsplit('.', 1)[0]:
    # update profiles if major version does not match
    info('Update profiles to default for GenX '+program_version)
    from glob import glob

    for fi in glob(os.path.join(_path, 'profiles', '*.conf')):
        shutil.copy2(fi, os.path.join(config_path, 'profiles'))
    open(version_file, 'w').write(program_version+'\n')
if not os.path.exists(os.path.join(config_path, 'genx.conf')):
    info('Creating genx.conf at %s by copying config from %s'%(config_path,
                                                               os.path.join(_path, 'genx.conf')))
    shutil.copyfile(os.path.join(_path, 'genx.conf'),
                    os.path.join(config_path, 'genx.conf'))

manual_url='https://aglavic.github.io/genx/doc/'
homepage_url='https://aglavic.github.io/genx/'

class ToolId(int, Enum):
    NEW_MODEL=10001
    OPEN_MODEL=10002
    SAVE_MODEL=10003
    SIM_MODEL=10004
    START_FIT=10005
    STOP_FIT=10006
    RESTART_FIT=10007
    CALC_ERROR=10008
    ZOOM=10009
    ERROR_STATS=10010

@dataclass
class GUIConfig(io.BaseConfig):
    section='gui'
    hsize: int=None # stores the width of the window
    vsize: int=None # stores the height of the window
    vsplit: int=None
    hsplit: int=None
    psplit: int=None

@dataclass
class WindowStartup(io.BaseConfig):
    section='startup'
    show_profiles: bool=True
    widescreen: bool=False

class MainFrame(wx.Frame, io.Configurable):
    opt: GUIConfig

    def __init__(self, parent: wx.App):
        self.parent=parent
        debug('starting setup of MainFrame')
        io.Configurable.__init__(self)
        self.wstartup=WindowStartup()

        self.flag_simulating=False
        self.simulation_queue_counter=0

        debug('setup of MainFrame - config')
        io.config.load_default(os.path.join(config_path, 'genx.conf'))
        self.ReadConfig()
        self.wstartup.load_config()

        debug('setup of MainFrame - wx.Frame\n')
        wx.Frame.__init__(self, None, id=wx.ID_ANY, title='GenX '+program_version,
                          size=wx.Size(self.opt.hsize, self.opt.vsize),
                          style=wx.DEFAULT_FRAME_STYLE)

        try:
            dpi_scale_factor=self.GetDPIScaleFactor()
            debug("Detected DPI scale factor %s from GetDPIScaleFactor"%dpi_scale_factor)
        except AttributeError:
            dpi_scale_factor=self.GetContentScaleFactor()
        debug("Detected DPI scale factor %s from GetContentScaleFactor"%dpi_scale_factor)
        self.dpi_scale_factor=dpi_scale_factor
        wx.GetApp().dpi_scale_factor=dpi_scale_factor
        tb_bmp_size=int(32*self.dpi_scale_factor)

        debug('setup of MainFrame - menu bar')
        # Menu Bar
        self.main_frame_menubar=wx.MenuBar()
        self.mb_file=wx.Menu()
        self.main_frame_menubar.mb_new=self.mb_file.Append(wx.ID_ANY, "New...\tCtrl+N", "Creates a new model")
        self.Bind(wx.EVT_MENU, self.eh_mb_new, id=self.main_frame_menubar.mb_new.GetId())
        self.main_frame_menubar.mb_open=self.mb_file.Append(wx.ID_ANY, "Open...\tCtrl+O", "Opens an existing model")
        self.Bind(wx.EVT_MENU, self.eh_mb_open, id=self.main_frame_menubar.mb_open.GetId())
        self.main_frame_menubar.mb_save=self.mb_file.Append(wx.ID_ANY, "Save...\tCtrl+S", "Saves the current model")
        self.Bind(wx.EVT_MENU, self.eh_mb_save, id=self.main_frame_menubar.mb_save.GetId())
        self.main_frame_menubar.mb_saveas=self.mb_file.Append(wx.ID_ANY, "Save As...",
                                                              "Saves the active model with a new name")
        self.Bind(wx.EVT_MENU, self.eh_mb_saveas, id=self.main_frame_menubar.mb_saveas.GetId())
        self.mb_file.AppendSeparator()
        mb_import=wx.Menu()
        self.main_frame_menubar.mb_import_data=mb_import.Append(wx.ID_ANY, "Import Data...\tCtrl+D",
                                                                "Import data to the active data set")
        self.Bind(wx.EVT_MENU, self.eh_mb_import_data, id=self.main_frame_menubar.mb_import_data.GetId())
        self.main_frame_menubar.mb_import_table=mb_import.Append(wx.ID_ANY, "Import Table...",
                                                                 "Import a table from an ASCII file")
        self.Bind(wx.EVT_MENU, self.eh_mb_import_table, id=self.main_frame_menubar.mb_import_table.GetId())
        self.main_frame_menubar.mb_import_script=mb_import.Append(wx.ID_ANY, "Import Script...",
                                                                  "Import a python model script")
        self.Bind(wx.EVT_MENU, self.eh_mb_import_script, id=self.main_frame_menubar.mb_import_script.GetId())
        self.mb_file.Append(wx.ID_ANY, "Import", mb_import, "")
        mb_export=wx.Menu()
        self.main_frame_menubar.mb_export_orso=mb_export.Append(wx.ID_ANY, "Export ORT (alpha)...",
                                                                "Export data and header in ORSO compatible ASCII format")
        self.Bind(wx.EVT_MENU, self.eh_mb_export_orso, id=self.main_frame_menubar.mb_export_orso.GetId())
        self.main_frame_menubar.mb_export_data=mb_export.Append(wx.ID_ANY, "Export Data...",
                                                                "Export data in ASCII format")
        self.Bind(wx.EVT_MENU, self.eh_mb_export_data, id=self.main_frame_menubar.mb_export_data.GetId())
        self.main_frame_menubar.mb_export_table=mb_export.Append(wx.ID_ANY, "Export Table...",
                                                                 "Export table to an ASCII file")
        self.Bind(wx.EVT_MENU, self.eh_mb_export_table, id=self.main_frame_menubar.mb_export_table.GetId())
        self.main_frame_menubar.mb_export_script=mb_export.Append(wx.ID_ANY, "Export Script...",
                                                                  "Export the script to a python file")
        self.Bind(wx.EVT_MENU, self.eh_mb_export_script, id=self.main_frame_menubar.mb_export_script.GetId())
        self.mb_file.Append(wx.ID_ANY, "Export", mb_export, "")
        self.mb_file.AppendSeparator()
        mb_print=wx.Menu()
        self.main_frame_menubar.mb_print_plot=mb_print.Append(wx.ID_ANY, "Print Plot...\tCtrl+P",
                                                              "Print the current plot")
        self.Bind(wx.EVT_MENU, self.eh_mb_print_plot, id=self.main_frame_menubar.mb_print_plot.GetId())
        self.main_frame_menubar.mb_print_grid=mb_print.Append(wx.ID_ANY, "Print Grid...", "Prints the grid")
        self.Bind(wx.EVT_MENU, self.eh_mb_print_grid, id=self.main_frame_menubar.mb_print_grid.GetId())
        self.main_frame_menubar.mb_print_script=mb_print.Append(wx.ID_ANY, "Print Script...", "Prints the model script")
        self.Bind(wx.EVT_MENU, self.eh_mb_print_script, id=self.main_frame_menubar.mb_print_script.GetId())
        self.mb_file.Append(wx.ID_ANY, "Print", mb_print, "")
        self.mb_file.AppendSeparator()
        self.main_frame_menubar.mb_quit=self.mb_file.Append(wx.ID_ANY, "&Quit\tAlt+Q", "Quit the program")
        self.Bind(wx.EVT_MENU, self.eh_mb_quit, id=self.main_frame_menubar.mb_quit.GetId())
        self.main_frame_menubar.Append(self.mb_file, "File")
        self.mb_edit=wx.Menu()
        self.main_frame_menubar.mb_copy_graph=self.mb_edit.Append(wx.ID_ANY, "Copy Graph",
                                                                  "Copy the current graph to the clipboard as a bitmap")
        self.Bind(wx.EVT_MENU, self.eh_mb_copy_graph, id=self.main_frame_menubar.mb_copy_graph.GetId())
        self.main_frame_menubar.mb_copy_sim=self.mb_edit.Append(wx.ID_ANY, "Copy Simulation",
                                                                "Copy the current simulation and data as ASCII text")
        self.Bind(wx.EVT_MENU, self.eh_mb_copy_sim, id=self.main_frame_menubar.mb_copy_sim.GetId())
        self.main_frame_menubar.mb_copy_table=self.mb_edit.Append(wx.ID_ANY, "Copy Table", "Copy the parameter grid")
        self.Bind(wx.EVT_MENU, self.eh_mb_copy_table, id=self.main_frame_menubar.mb_copy_table.GetId())
        self.main_frame_menubar.mb_findreplace=self.mb_edit.Append(wx.ID_ANY, "&Find/Replace...\tCtrl+F",
                                                                   "Find and replace in the script")
        self.Bind(wx.EVT_MENU, self.eh_mb_findreplace, id=self.main_frame_menubar.mb_findreplace.GetId())
        self.mb_edit_sub=wx.Menu()
        self.main_frame_menubar.mb_new_data_set=self.mb_edit_sub.Append(wx.ID_ANY, "&New data set\tAlt+N",
                                                                        "Appends a new data set")
        self.Bind(wx.EVT_MENU, self.eh_data_new_set, id=self.main_frame_menubar.mb_new_data_set.GetId())
        self.main_frame_menubar.mb_data_delete=self.mb_edit_sub.Append(wx.ID_ANY, "&Delete\tAlt+D",
                                                                       "Deletes the selected data sets")
        self.Bind(wx.EVT_MENU, self.eh_data_delete, id=self.main_frame_menubar.mb_data_delete.GetId())
        self.main_frame_menubar.mb_data_move_down=self.mb_edit_sub.Append(wx.ID_ANY, "&Lower item\tAlt+L",
                                                                          "Move selected item down")
        self.Bind(wx.EVT_MENU, self.eh_data_move_down, id=self.main_frame_menubar.mb_data_move_down.GetId())
        self.main_frame_menubar.mb_data_move_up=self.mb_edit_sub.Append(wx.ID_ANY, "&Raise item\tAlt+R",
                                                                        "Moves selected data sets up")
        self.Bind(wx.EVT_MENU, self.eh_data_move_up, id=self.main_frame_menubar.mb_data_move_up.GetId())
        self.mb_edit_sub.AppendSeparator()
        self.main_frame_menubar.mb_toggle_show=self.mb_edit_sub.Append(wx.ID_ANY, "Toggle &Show\tAlt+S",
                                                                       "Toggle show on and off for the selected data set")
        self.Bind(wx.EVT_MENU, self.eh_data_toggle_show, id=self.main_frame_menubar.mb_toggle_show.GetId())
        self.main_frame_menubar.mb_toggle_use=self.mb_edit_sub.Append(wx.ID_ANY, "Toggle &Use\tAlt+U",
                                                                      "Toggle use on and off for the selected data sets")
        self.Bind(wx.EVT_MENU, self.eh_data_toggle_use, id=self.main_frame_menubar.mb_toggle_use.GetId())
        self.main_frame_menubar.mb_toggle_error=self.mb_edit_sub.Append(wx.ID_ANY, "Toggle &Error\tAlt+E",
                                                                        "Turn the use of error on and off")
        self.Bind(wx.EVT_MENU, self.eh_data_toggle_error, id=self.main_frame_menubar.mb_toggle_error.GetId())
        self.mb_edit_sub.AppendSeparator()
        self.main_frame_menubar.mb_toggle_calc=self.mb_edit_sub.Append(wx.ID_ANY, "&Calculations\tAlt+C",
                                                                       "OPens dialog box to define dataset calculations")
        self.Bind(wx.EVT_MENU, self.eh_data_calc, id=self.main_frame_menubar.mb_toggle_calc.GetId())
        self.mb_edit.Append(wx.ID_ANY, "Data", self.mb_edit_sub, "")
        self.main_frame_menubar.Append(self.mb_edit, "Edit")
        self.mb_view=wx.Menu()
        self.main_frame_menubar.mb_view_grid_slider=self.mb_view.Append(wx.ID_ANY, "Value as slider",
                                                                        "View and control the value as a slider",
                                                                        wx.ITEM_CHECK)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_grid_slider, id=self.main_frame_menubar.mb_view_grid_slider.GetId())
        self.main_frame_menubar.mb_view_zoom=self.mb_view.Append(wx.ID_ANY, "Zoom\tCtrl+Z", "Turn the zoom on/off",
                                                                 wx.ITEM_CHECK)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_zoom, id=self.main_frame_menubar.mb_view_zoom.GetId())
        self.main_frame_menubar.mb_view_zoomall=self.mb_view.Append(wx.ID_ANY, "Zoom All\tCtrl+A",
                                                                    "Zoom to fit all data points")
        self.Bind(wx.EVT_MENU, self.eh_mb_view_zoomall, id=self.main_frame_menubar.mb_view_zoomall.GetId())
        mb_view_yscale=wx.Menu()
        self.main_frame_menubar.mb_view_yscale_log=mb_view_yscale.Append(wx.ID_ANY, "log", "Set y-scale logarithmic",
                                                                         wx.ITEM_RADIO)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_yscale_log, id=self.main_frame_menubar.mb_view_yscale_log.GetId())
        self.main_frame_menubar.mb_view_yscale_lin=mb_view_yscale.Append(wx.ID_ANY, "lin", "Set y-scale linear",
                                                                         wx.ITEM_RADIO)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_yscale_linear, id=self.main_frame_menubar.mb_view_yscale_lin.GetId())
        self.mb_view.Append(wx.ID_ANY, "y scale", mb_view_yscale, "")
        mb_view_xscale=wx.Menu()
        self.main_frame_menubar.mb_view_xscale_log=mb_view_xscale.Append(wx.ID_ANY, "log", "Set x-scale logarithmic",
                                                                         wx.ITEM_RADIO)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_xscale_log, id=self.main_frame_menubar.mb_view_xscale_log.GetId())
        self.main_frame_menubar.mb_view_xscale_lin=mb_view_xscale.Append(wx.ID_ANY, "lin", "Set x-scale linear",
                                                                         wx.ITEM_RADIO)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_xscale_linear, id=self.main_frame_menubar.mb_view_xscale_lin.GetId())
        self.mb_view.Append(wx.ID_ANY, "x scale", mb_view_xscale, "")
        self.main_frame_menubar.mb_view_autoscale=self.mb_view.Append(wx.ID_ANY, "Autoscale",
                                                                      "Sets autoscale on when plotting", wx.ITEM_CHECK)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_autoscale, id=self.main_frame_menubar.mb_view_autoscale.GetId())
        self.main_frame_menubar.mb_use_toggle_show=self.mb_view.Append(wx.ID_ANY, "Use Toggle Show",
                                                                       "Set if the plotted data shold be toggled or selcted by the mouse",
                                                                       wx.ITEM_CHECK)
        self.Bind(wx.EVT_MENU, self.eh_mb_view_use_toggle_show, id=self.main_frame_menubar.mb_use_toggle_show.GetId())
        self.main_frame_menubar.Append(self.mb_view, "View")
        self.mb_fit=wx.Menu()
        self.main_frame_menubar.mb_fit_simulate=self.mb_fit.Append(wx.ID_ANY, "&Simulate\tF9",
                                                                   "Compile the script and run the Sim function")
        self.Bind(wx.EVT_MENU, self.eh_tb_simulate, id=self.main_frame_menubar.mb_fit_simulate.GetId())
        self.main_frame_menubar.mb_fit_evaluate=self.mb_fit.Append(wx.ID_ANY, "&Evaluate\tF5",
                                                                   "Evaluate the Sim function only - no recompiling")
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_evaluate, id=self.main_frame_menubar.mb_fit_evaluate.GetId())
        self.main_frame_menubar.mb_use_cuda=self.mb_fit.Append(wx.ID_ANY, "Use CUDA",
                                                               "Make use of Nvidia GPU computing with CUDA",
                                                               wx.ITEM_CHECK)
        self.Bind(wx.EVT_MENU, self.eh_mb_use_cuda, id=self.main_frame_menubar.mb_use_cuda.GetId())
        self.mb_fit.AppendSeparator()
        self.main_frame_menubar.mb_fit_start=self.mb_fit.Append(wx.ID_ANY, "Start &Fit\tCtrl+F", "Start fitting")
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_start, id=self.main_frame_menubar.mb_fit_start.GetId())
        self.main_frame_menubar.mb_fit_stop=self.mb_fit.Append(wx.ID_ANY, "&Halt Fit\tCtrl+H", "Stop fitting")
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_stop, id=self.main_frame_menubar.mb_fit_stop.GetId())
        self.main_frame_menubar.mb_fit_resume=self.mb_fit.Append(wx.ID_ANY, "&Resume Fit\tCtrl+R",
                                                                 "Resumes fitting without reinitilazation of the optimizer")
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_resume, id=self.main_frame_menubar.mb_fit_resume.GetId())
        self.main_frame_menubar.mb_fit_analyze=self.mb_fit.Append(wx.ID_ANY, "Analyze fit", "Analyze the fit")
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_analyze, id=self.main_frame_menubar.mb_fit_analyze.GetId())
        self.mb_fit.AppendSeparator()
        self.main_frame_menubar.mb_fit_autosim=self.mb_fit.Append(wx.ID_ANY, "Simulate Automatically",
                                                                  "Update simulation on model changes automatically",
                                                                  wx.ITEM_CHECK)
        self.Bind(wx.EVT_MENU, self.eh_mb_fit_autosim, id=self.main_frame_menubar.mb_fit_autosim.GetId())
        self.main_frame_menubar.Append(self.mb_fit, "Fit")
        self.mb_set=wx.Menu()
        mb_set_plugins=wx.Menu()
        mb_set_plugins.AppendSeparator()
        self.mb_set.Append(wx.ID_ANY, "Plugins", mb_set_plugins, "")
        self.main_frame_menubar.mb_set_opt=self.mb_set.Append(wx.ID_ANY, "Optimizer\tShift+Ctrl+O", "")
        self.Bind(wx.EVT_MENU, self.eh_mb_set_opt, id=self.main_frame_menubar.mb_set_opt.GetId())
        self.main_frame_menubar.mb_set_dataloader=self.mb_set.Append(wx.ID_ANY, "Data Loader\tShift+Ctrl+D", "")
        self.Bind(wx.EVT_MENU, self.eh_mb_set_dal, id=self.main_frame_menubar.mb_set_dataloader.GetId())
        self.main_frame_menubar.mb_set_import=self.mb_set.Append(wx.ID_ANY, "Import\tShift+Ctrl+I",
                                                                 "Import settings for the data sets")
        self.Bind(wx.EVT_MENU, self.eh_data_import, id=self.main_frame_menubar.mb_set_import.GetId())
        self.main_frame_menubar.mb_set_dataplot=self.mb_set.Append(wx.ID_ANY, "Plot Markers\tShift+Ctrl+P",
                                                                   "Set the symbols and lines of data and simulations")
        self.Bind(wx.EVT_MENU, self.eh_data_plots, id=self.main_frame_menubar.mb_set_dataplot.GetId())
        self.main_frame_menubar.show_startup_dialog=self.mb_set.Append(wx.ID_ANY, "Startup Profile...", "")
        self.Bind(wx.EVT_MENU, self.eh_show_startup_dialog, id=self.main_frame_menubar.show_startup_dialog.GetId())
        self.main_frame_menubar.Append(self.mb_set, "Settings")
        wxglade_tmp_menu=wx.Menu()
        self.main_frame_menubar.mb_models_help=wxglade_tmp_menu.Append(wx.ID_ANY, "Models Help...",
                                                                       "Show help for the models")
        self.Bind(wx.EVT_MENU, self.eh_mb_models_help, id=self.main_frame_menubar.mb_models_help.GetId())
        self.main_frame_menubar.mb_fom_help=wxglade_tmp_menu.Append(wx.ID_ANY, "FOM Help", "Show help about the fom")
        self.Bind(wx.EVT_MENU, self.eh_mb_fom_help, id=self.main_frame_menubar.mb_fom_help.GetId())
        self.main_frame_menubar.mb_plugins_help=wxglade_tmp_menu.Append(wx.ID_ANY, "Plugins Helps...",
                                                                        "Show help for the plugins")
        self.Bind(wx.EVT_MENU, self.eh_mb_plugins_help, id=self.main_frame_menubar.mb_plugins_help.GetId())
        self.main_frame_menubar.mb_data_loaders_help=wxglade_tmp_menu.Append(wx.ID_ANY, "Data loaders Help...",
                                                                             "Show help for the data loaders")
        self.Bind(wx.EVT_MENU, self.eh_mb_data_loaders_help, id=self.main_frame_menubar.mb_data_loaders_help.GetId())
        wxglade_tmp_menu.AppendSeparator()
        self.main_frame_menubar.mb_misc_showman=wxglade_tmp_menu.Append(wx.ID_ANY, "Open Manual...", "Show the manual")
        self.Bind(wx.EVT_MENU, self.eh_mb_misc_showman, id=self.main_frame_menubar.mb_misc_showman.GetId())
        self.main_frame_menubar.mb_open_homepage=wxglade_tmp_menu.Append(wx.ID_ANY, "Open Homepage...",
                                                                         "Open the homepage")
        self.Bind(wx.EVT_MENU, self.eh_mb_misc_openhomepage, id=self.main_frame_menubar.mb_open_homepage.GetId())
        self.main_frame_menubar.mb_misc_about=wxglade_tmp_menu.Append(wx.ID_ANY, "About...",
                                                                      "Shows information about GenX")
        self.Bind(wx.EVT_MENU, self.eh_mb_misc_about, id=self.main_frame_menubar.mb_misc_about.GetId())
        self.main_frame_menubar.Append(wxglade_tmp_menu, "Help")
        self.SetMenuBar(self.main_frame_menubar)
        # Menu Bar end
        self.main_frame_statusbar=self.CreateStatusBar(3)

        debug('setup of MainFrame - tool bar')
        # Tool Bar
        self.main_frame_toolbar=wx.ToolBar(self, -1, style=wx.TB_DEFAULT_STYLE)
        self.SetToolBar(self.main_frame_toolbar)
        self.main_frame_toolbar.AddTool(ToolId.NEW_MODEL, "tb_new", wx.Bitmap(img.getnewImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "New model | Ctrl+N",
                                        "Create a new model | Ctrl+N")
        self.main_frame_toolbar.AddTool(ToolId.OPEN_MODEL, "tb_open", wx.Bitmap(img.getopenImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "Open | Ctrl+O",
                                        "Open an existing model | Ctrl+O")
        self.main_frame_toolbar.AddTool(ToolId.SAVE_MODEL, "tb_save", wx.Bitmap(img.getsaveImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "Save | Ctrl+S", "Save model to file | Ctrl+S")
        self.main_frame_toolbar.AddSeparator()
        self.main_frame_toolbar.AddTool(ToolId.SIM_MODEL, "tb_simulate",
                                        wx.Bitmap(img.getsimulateImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "Simulate | F9", "Simulate the model | F9")
        self.main_frame_toolbar.AddTool(ToolId.START_FIT, "tb_start_fit",
                                        wx.Bitmap(img.getstart_fitImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "Start fit | Ctrl+F", "Start fitting | Ctrl+F")
        self.main_frame_toolbar.AddTool(ToolId.STOP_FIT, "tb_stop_fit",
                                        wx.Bitmap(img.getstop_fitImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "Stop fit | Ctrl+H", "Stop fitting | Ctrl+H")
        self.main_frame_toolbar.AddTool(ToolId.RESTART_FIT, "tb_restart_fit",
                                        wx.Bitmap(img.getrestart_fitImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "Restart fit | Ctrl+R",
                                        "Restart the fit | Ctrl+R")
        self.main_frame_toolbar.AddTool(ToolId.CALC_ERROR, "tb_calc_error_bars",
                                        wx.Bitmap(img.getcalc_error_barImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "Calculate errorbars", "Calculate errorbars")
        self.main_frame_toolbar.AddTool(ToolId.ERROR_STATS, "tb_error_stats",
                                        wx.Bitmap(img.getpar_projImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_NORMAL, "Error Statistics", "Error Statistics")
        self.main_frame_toolbar.AddSeparator()
        self.main_frame_toolbar.AddTool(ToolId.ZOOM, "tb_zoom", wx.Bitmap(img.getzoomImage().Scale(tb_bmp_size, tb_bmp_size)),
                                        wx.NullBitmap, wx.ITEM_CHECK, "Zoom | Ctrl+Z", "Turn zoom on/off  | Ctrl+Z")
        # Tool Bar end
        debug('setup of MainFrame - splitters and panels')
        self.ver_splitter=wx.SplitterWindow(self, wx.ID_ANY, style=wx.SP_3D | wx.SP_BORDER | wx.SP_LIVE_UPDATE)
        self.data_panel=wx.Panel(self.ver_splitter, wx.ID_ANY)
        self.data_notebook=wx.Notebook(self.data_panel, wx.ID_ANY)
        self.data_notebook_data=wx.Panel(self.data_notebook, wx.ID_ANY)
        self.data_list=datalist.DataListControl(self.data_notebook_data, wx.ID_ANY, self.eh_ex_status_text)
        self.data_notebook_pane_2=wx.Panel(self.data_notebook, wx.ID_ANY)
        self.label_2=wx.StaticText(self.data_notebook_pane_2, wx.ID_ANY, "  Data set: ")
        self.data_grid_choice=wx.Choice(self.data_notebook_pane_2, wx.ID_ANY, choices=["test2", "test1"])
        self.static_line_1=wx.StaticLine(self.data_notebook_pane_2, wx.ID_ANY)
        self.data_grid=wx.grid.Grid(self.data_notebook_pane_2, wx.ID_ANY, size=(1, 1))
        self.main_panel=wx.Panel(self.ver_splitter, wx.ID_ANY)
        self.hor_splitter=wx.SplitterWindow(self.main_panel, wx.ID_ANY,
                                            style=wx.SP_3D | wx.SP_BORDER | wx.SP_LIVE_UPDATE)
        self.plot_panel=wx.Panel(self.hor_splitter, wx.ID_ANY)
        self.plot_splitter=wx.SplitterWindow(self.plot_panel, wx.ID_ANY)
        self.plot_notebook=wx.Notebook(self.plot_splitter, wx.ID_ANY, style=wx.NB_BOTTOM)
        self.plot_notebook_data=wx.Panel(self.plot_notebook, wx.ID_ANY)
        self.plot_data=plotpanel.DataPlotPanel(self.plot_notebook_data)
        self.plot_notebook_fom=wx.Panel(self.plot_notebook, wx.ID_ANY)
        self.plot_fom=plotpanel.ErrorPlotPanel(self.plot_notebook_fom)
        self.plot_notebook_Pars=wx.Panel(self.plot_notebook, wx.ID_ANY)
        self.plot_pars=plotpanel.ParsPlotPanel(self.plot_notebook_Pars)
        self.plot_notebook_foms=wx.Panel(self.plot_notebook, wx.ID_ANY)
        self.plot_fomscan=plotpanel.FomScanPlotPanel(self.plot_notebook_foms)
        self.wide_plugin_notebook=wx.Notebook(self.plot_splitter, wx.ID_ANY, style=wx.NB_BOTTOM)
        self.panel_1=wx.Panel(self.wide_plugin_notebook, wx.ID_ANY)
        self.input_panel=wx.Panel(self.hor_splitter, wx.ID_ANY)
        self.input_notebook=wx.Notebook(self.input_panel, wx.ID_ANY, style=wx.NB_BOTTOM)
        self.input_notebook_grid=wx.Panel(self.input_notebook, wx.ID_ANY)
        self.paramter_grid=parametergrid.ParameterGrid(self.input_notebook_grid, self)
        self.input_notebook_script=wx.Panel(self.input_notebook, wx.ID_ANY)
        self.script_editor=wx.py.editwindow.EditWindow(self.input_notebook_script, wx.ID_ANY)
        self.script_editor.SetBackSpaceUnIndents(True)
        self.script_editor.Bind(wx.EVT_KEY_DOWN, self.ScriptEditorKeyEvent)

        debug('setup of MainFrame - properties and layout')
        self.__set_properties()
        self.__do_layout()

        debug('setup of MainFrame - bind')
        self.Bind(wx.EVT_TOOL, self.eh_tb_new, id=ToolId.NEW_MODEL)
        self.Bind(wx.EVT_TOOL, self.eh_tb_open, id=ToolId.OPEN_MODEL)
        self.Bind(wx.EVT_TOOL, self.eh_tb_save, id=ToolId.SAVE_MODEL)
        self.Bind(wx.EVT_TOOL, self.eh_tb_simulate, id=ToolId.SIM_MODEL)
        self.Bind(wx.EVT_TOOL, self.eh_tb_start_fit, id=ToolId.START_FIT)
        self.Bind(wx.EVT_TOOL, self.eh_tb_stop_fit, id=ToolId.STOP_FIT)
        self.Bind(wx.EVT_TOOL, self.eh_tb_restart_fit, id=ToolId.RESTART_FIT)
        self.Bind(wx.EVT_TOOL, self.eh_tb_calc_error_bars, id=ToolId.CALC_ERROR)
        self.Bind(wx.EVT_TOOL, self.eh_tb_error_stats, id=ToolId.ERROR_STATS)
        self.Bind(wx.EVT_TOOL, self.eh_tb_zoom, id=ToolId.ZOOM)
        self.Bind(wx.EVT_CHOICE, self.eh_data_grid_choice, self.data_grid_choice)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGED, self.eh_plot_page_changed, self.plot_notebook)

        debug('setup of MainFrame - manual config')

        # GenX objects
        self.model=model.Model()
        self.model.data=self.data_list.data_cont.data
        self.paramter_grid.SetParameters(self.model.parameters)

        if self.model.script!='':
            self.script_editor.SetText(self.model.script)
        self.solver_control=solvergui.SolverController(self)

        self.plugin_control= \
            add_on.PluginController(self, mb_set_plugins)

        # Bind all the events that are needed to occur when a new model has
        # been loaded
        # Update the parameter grid
        self.Bind(EVT_NEW_MODEL, self.paramter_grid.OnNewModel, self)
        self.Bind(EVT_NEW_MODEL, self.data_list.eh_external_new_model, self)
        # Update the script
        self.Bind(EVT_NEW_MODEL, self.eh_new_model, self)
        # Event that the plot should respond to
        self.Bind(datalist.EVT_DATA_LIST, self.plot_data.OnDataListEvent, self.data_list.list_ctrl)
        self.Bind(datalist.EVT_DATA_LIST, self.eh_external_update_data_grid_choice, self.data_list.list_ctrl)
        self.Bind(datalist.EVT_DATA_LIST, self.eh_external_update_data, self.data_list.list_ctrl)

        self.Bind(EVT_SIM_PLOT, self.plot_data.OnSimPlotEvent,
                  self)
        self.Bind(EVT_SIM_PLOT, self.eh_external_fom_value,
                  self)
        # Update events from the solver
        self.Bind(solvergui.EVT_UPDATE_PLOT, self.eh_external_fom_value)
        self.Bind(solvergui.EVT_UPDATE_PLOT, self.plot_data.OnSolverPlotEvent)
        self.Bind(solvergui.EVT_UPDATE_PLOT, self.plot_fom.OnSolverPlotEvent)

        self.Bind(solvergui.EVT_SOLVER_UPDATE_TEXT,
                  self.eh_ex_status_text)
        self.Bind(solvergui.EVT_UPDATE_PARAMETERS,
                  self.paramter_grid.OnSolverUpdateEvent)
        self.Bind(solvergui.EVT_UPDATE_PARAMETERS,
                  self.plot_pars.OnSolverParameterEvent)
        # For picking a point in a plot
        self.Bind(plotpanel.EVT_PLOT_POSITION,
                  self.eh_ex_point_pick)
        # This is needed to be able to create the events
        self.plot_data.SetCallbackWindow(self)
        self.plot_fom.SetCallbackWindow(self)
        self.plot_pars.SetCallbackWindow(self)
        self.plot_fomscan.SetCallbackWindow(self)
        self.Bind(plotpanel.EVT_PLOT_SETTINGS_CHANGE,
                  self.eh_ex_plot_settings_changed)

        # Binding events which means model changes
        self.Bind(parametergrid.EVT_PARAMETER_GRID_CHANGE,
                  self.eh_external_model_changed)
        self.Bind(wx.stc.EVT_STC_MODIFIED, self.eh_external_model_changed,
                  self.script_editor)
        self.Bind(datalist.EVT_DATA_LIST, self.eh_external_model_changed,
                  self.data_list.list_ctrl)

        # Event for when a value of a parameter in the parameter grid has been updated
        self.Bind(parametergrid.EVT_PARAMETER_VALUE_CHANGE,
                  self.eh_external_parameter_value_changed)

        # Stuff for the find and replace functionallity
        self.findreplace_data=wx.FindReplaceData()
        # Make search down as default
        self.findreplace_data.SetFlags(1)
        self.findreplace_dlg=wx.FindReplaceDialog(self,
                                                  self.findreplace_data,
                                                  "Find & replace",
                                                  wx.FR_REPLACEDIALOG)
        self.Bind(wx.EVT_FIND, self.eh_external_find)
        self.Bind(wx.EVT_FIND_NEXT, self.eh_external_find)
        self.Bind(wx.EVT_FIND_REPLACE, self.eh_external_find)
        self.Bind(wx.EVT_FIND_REPLACE_ALL, self.eh_external_find)
        self.Bind(wx.EVT_FIND_CLOSE, self.eh_external_find)
        self.Bind(wx.EVT_CLOSE, self.eh_mb_quit)

        self.paramter_grid.SetFOMFunctions(self.project_fom_parameter, self.scan_parameter)

        # Initiializations..
        # To force an update of the menubar...
        self.plot_data.SetZoom(False)

        try:
            for p in [self.plot_data, self.plot_fom,
                      self.plot_pars, self.plot_fomscan]:
                p.ReadConfig()
        except Exception as e:
            outp=io.StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            error('Error in loading config for the plots. Pyton tractback:\n %s'%val)
            ShowErrorDialog(self, 'Could not read the config for the plots. Python Error:\n%s'%(val,))

        self.model.saved=True
        debug('finished setup of MainFrame')

    def scan_parameter(self, row):
        ''' scan_parameter(frame, row) --> None

        Scans the parameter in row row [int] from max to min in the number
        of steps given by dialog input.
        '''
        if not self.model.is_compiled():
            ShowNotificationDialog(self, 'Please conduct a simulation before'+ \
                                   ' scanning a parameter. The script needs to be compiled.')
            return

        dlg = wx.NumberEntryDialog(self,
                                   'Input the number of evaluation points for the scan',
                                   'Steps', '', 50, 2, 1000)
        if dlg.ShowModal()==wx.ID_OK:
            self.main_frame_statusbar.SetStatusText('Scanning parameter', 1)
            try:
                x, y = self.solver_control.ScanParameter(row, dlg.GetValue())
                fs, pars = self.model.get_sim_pars()
                bestx = self.model.parameters.get_data()[row][1]
                besty = self.model.fom

                self.plot_fomscan.SetPlottype('scan')
                self.plot_fomscan.Plot((x, y, bestx, besty,
                                        self.solver_control.fom_error_bars_level),
                                       self.model.parameters.get_names()[row],
                                       'FOM')
                self.sep_plot_notebook.SetSelection(3)
            except Exception as e:
                outp = StringIO()
                traceback.print_exc(200, outp)
                val = outp.getvalue()
                outp.close()
                ShowErrorDialog(self, val)
                self.main_frame_statusbar.SetStatusText('Fatal Error - scan fom', 1)
            else:
                self.main_frame_statusbar.SetStatusText('Scanning finished', 1)

        dlg.Destroy()

    def project_fom_parameter(self, row):
        '''project_fom_parameter(frame, row) --> None

        Plots the project fom given by the row row [int]
        '''
        import numpy as np
        if not self.solver_control.IsFitted():
            ShowNotificationDialog(self, 'Please conduct a fit before'+
                                   ' scanning a parameter. The script needs to be compiled and foms have'
                                   +' to be collected.')
            return

        self.main_frame_statusbar.SetStatusText('Trying to project fom', 1)
        try:
            x, y = self.solver_control.ProjectEvals(row)
            if len(x)==0 or len(y)==0:
                ShowNotificationDialog(self, 'Please conduct a fit before'+
                                       ' projecting a parameter. The script needs to be compiled and foms have'
                                       +' to be collected.')
                return
            elif self.model.fom is None or np.isnan(self.model.fom):
                ShowNotificationDialog(self, 'The model must be simulated (FOM is not a valid number)')
                return
            fs, pars = self.model.get_sim_pars()
            bestx = pars[row]
            besty = self.model.fom
            self.plot_fomscan.SetPlottype('project')
            self.plot_fomscan.Plot((x, y, bestx, besty,
                                    self.solver_control.fom_error_bars_level),
                                   self.model.parameters.get_names()[row],
                                   'FOM')
            self.sep_plot_notebook.SetSelection(3)
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            ShowErrorDialog(self, val)
            self.main_frame_statusbar.SetStatusText('Fatal Error - project fom', 1)
        else:
            self.main_frame_statusbar.SetStatusText('Projected fom plotted', 1)

    def ScriptEditorKeyEvent(self, evt):
        if evt.GetKeyCode()==13:
            pos=self.script_editor.GetCurrentPos()
            line=self.script_editor.GetCurrentLine()
            idn=self.script_editor.GetLineIndentation(line)
            txt=self.script_editor.GetLine(line).strip()
            if txt.startswith('for ') or txt.startswith('if ') or \
                    txt.startswith('elif ') or txt.startswith('else:'):
                idn+=4
            self.script_editor.InsertText(pos, '\n'+' '*idn)
            self.script_editor.GotoPos(pos+idn+1)
        else:
            evt.Skip()

    def __set_properties(self):
        self.main_frame_fom_text=wx.StaticText(self.main_frame_toolbar, -1,
                                               '        FOM:                    ', size=(400, -1))
        font=wx.Font(wx.FontInfo(15*self.dpi_scale_factor))
        self.main_frame_fom_text.SetFont(font)
        self.main_frame_fom_text.SetLabel('        FOM: None')
        # self.main_frame_fom_text.SetEditable(False)
        self.main_frame_toolbar.AddSeparator()
        self.main_frame_toolbar.AddSeparator()
        self.main_frame_toolbar.AddControl(self.main_frame_fom_text)

        # begin wxGlade: MainFrame.__set_properties
        _icon=wx.NullIcon
        _icon.CopyFromBitmap(img.genx.GetBitmap())
        self.SetIcon(_icon)
        self.main_frame_statusbar.SetStatusWidths([-2, -3, -2])

        # statusbar fields
        main_frame_statusbar_fields=["", "", "x,y"]
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
        frame_sizer=wx.BoxSizer(wx.VERTICAL)
        main_sizer=wx.BoxSizer(wx.HORIZONTAL)
        input_sizer=wx.BoxSizer(wx.VERTICAL)
        sizer_8=wx.BoxSizer(wx.HORIZONTAL)
        sizer_7=wx.BoxSizer(wx.HORIZONTAL)
        plot_sizer=wx.BoxSizer(wx.HORIZONTAL)
        sizer_6=wx.BoxSizer(wx.HORIZONTAL)
        sizer_5=wx.BoxSizer(wx.HORIZONTAL)
        sizer_4=wx.BoxSizer(wx.HORIZONTAL)
        sizer_3=wx.BoxSizer(wx.HORIZONTAL)
        data_sizer=wx.BoxSizer(wx.VERTICAL)
        sizer_1=wx.BoxSizer(wx.VERTICAL)
        sizer_2=wx.BoxSizer(wx.HORIZONTAL)
        data_list_sizer=wx.BoxSizer(wx.HORIZONTAL)
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

        self.sep_plot_notebook=self.plot_notebook
        if self.wstartup.widescreen:
            # test adding new notebooks for plugins in wide screen layout
            self.plot_notebook=self.wide_plugin_notebook
            self.plot_notebook.RemovePage(0)
            self.plot_splitter.SetSashGravity(0.75)
            self.sep_data_notebook=self.data_notebook
            self.data_notebook=wx.Notebook(self.data_panel, wx.ID_ANY,
                                           style=wx.NB_TOP | wx.BORDER_SUNKEN)
            data_sizer.Add(self.data_notebook, 1, wx.EXPAND | wx.ALL, 4)
        else:
            self.plot_splitter.Unsplit()

    def Show(self, **kwargs):
        '''
        Overiding the default method since any resizing has to come AFTER
        the calls to Show
        '''
        display_size=wx.DisplaySize()
        hsize=self.opt.hsize or int(display_size[0]*0.85)
        vsize=self.opt.vsize or int(display_size[1]*0.9)
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

    def LayoutSplitters(self):
        size=self.GetSize()
        vsplit=self.opt.vsplit or size[0]/4
        hsplit=self.opt.hsplit or size[1]-450
        self.ver_splitter.SetSashPosition(vsplit)
        self.hor_splitter.SetSashPosition(hsplit)

        if self.wstartup.widescreen:
            psplit=self.opt.psplit or int(size[1]*0.6)
            self.plot_splitter.SetSashPosition(psplit)

    def startup_dialog(self, profile_path, force_show=False):
        if self.wstartup.show_profiles or force_show:
            startup_dialog=StartUpConfigDialog(self, profile_path+'profiles/',
                                               show_cb=self.wstartup.show_profiles,
                                               wide=self.wstartup.widescreen)
            startup_dialog.ShowModal()
            config_file=startup_dialog.GetConfigFile()
            if config_file:
                io.config.load_default(profile_path+'profiles/'+config_file, reset=True)
                self.wstartup.show_profiles=startup_dialog.GetShowAtStartup()
                self.wstartup.widescreen=startup_dialog.GetWidescreen()
                self.wstartup.safe_config(default=True)
                io.config.write_default(os.path.join(config_path, 'genx.conf'))
                debug('Changed profile, plugins to load=%s'%io.config.get('plugins', 'loaded plugins'))
                try:
                    self.plugin_control.OnOpenModel(None)
                except Exception as e:
                    outp=io.StringIO()
                    traceback.print_exc(200, outp)
                    val=outp.getvalue()
                    outp.close()
                    error("Exception:\n%s"%outp)
                    ShowErrorDialog(self, 'Problems when plugins processed model.' \
                                             ' Python Error:\n%s'%(val,))

    def set_title(self):
        filepath, filename = os.path.split(self.model.filename)
        if filename!='':
            if self.model.saved:
                self.SetTitle(filename+' - '+filepath+' - GenX ' \
                              +program_version)
            else:
                self.SetTitle(filename+'* - '+filepath+' - GenX ' \
                              +program_version)
        else:
            self.SetTitle('GenX '+program_version)

    def eh_mb_new(self, event):
        '''
        Event handler for creating a new model
        '''
        if not self.model.saved:
            ans = ShowQuestionDialog(self, 'If you continue any changes in'
                                           ' your model will not be saved.',
                                     'Model not saved')
            if not ans:
                return

        # Reset the model - remove everything from the previous model
        self.model.new_model()
        # Update all components so all the traces are gone.
        _post_new_model_event(self, self.model, desc='Fresh model')
        self.plugin_control.OnNewModel(None)
        self.main_frame_statusbar.SetStatusText('New model created', 1)
        self.set_title()
        self.model.saved = True

    def eh_mb_open(self, event):
        '''
        Event handler for opening a model file...
        '''
        # Check so the model is saved before quitting
        if not self.model.saved:
            ans = ShowQuestionDialog(self, 'If you continue any changes in'
                                           ' your model will not be saved.',
                                     'Model not saved')
            if not ans:
                return

        dlg = wx.FileDialog(self, message="Open", defaultFile="",
                            wildcard="GenX File (*.hgx;*.gx)|*.hgx;*.gx",
                            style=wx.FD_OPEN  # | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            path = dlg.GetPath()
            debug('open: path retrieved')
            self.open_model(path)

        dlg.Destroy()

    def get_pages(self):
        # Get all plot panel objects in GUI
        pages = []
        for page in self.plot_notebook.GetChildren():
            pages += page.GetChildren()
        if self.sep_plot_notebook is not self.plot_notebook:
            for page in self.sep_plot_notebook.GetChildren():
                pages += page.GetChildren()

        # pages = [frame.plot_data, frame.plot_fom, frame.plot_pars,\
        #             frame.plot_fomscan]
        return pages

    def open_model(self, path):
        debug('open_model: clear model')
        self.model.new_model()
        self.paramter_grid.PrepareNewModel()
        # Update all components so all the traces are gone.
        # _post_new_model_event(frame, frame.model)
        debug('open_model: load_file')
        try:
            io.load_file(path, self.model, self.solver_control.optimizer)
        except GenxIOError as e:
            ShowModelErrorDialog(self, e.__str__())
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            iprint('Error in loading the file ', path, '. Pyton traceback:\n ', val)
            ShowErrorDialog(self, 'Could not open the file. Python Error:\n%s'%(val,))
            return
        debug('open_model: read config')
        try:
            [p.ReadConfig() for p in self.get_pages()]
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            iprint('Error in loading config for the plots. Pyton traceback:\n ', val)
            ShowErrorDialog(self, 'Could not read the config for the plots. Python Error:\n%s'%(val,))
        try:
            self.paramter_grid.ReadConfig()
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            iprint('Error in loading config for parameter grid. Pyton traceback:\n ', val)
            ShowErrorDialog(self, 'Could not read the config for the parameter grid. Python Error:\n%s'%(val,))
        else:
            # Update the Menu choice
            self.main_frame_menubar.mb_view_grid_slider.Check(self.paramter_grid.GetValueEditorSlider())
        debug('open_model: update plugins')
        try:
            self.plugin_control.OnOpenModel(None)
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            ShowErrorDialog(self, 'Problems when plugins processed model.' \
                                  ' Python Error:\n%s'%(val,))
        self.main_frame_statusbar.SetStatusText('Model loaded from file',
                                                1)
        # Post an event to update everything else
        debug('open_model: post new model event')
        _post_new_model_event(self, self.model)
        # Needs to put it to saved since all the widgets will have
        # been updated
        self.model.saved = True
        self.set_title()

    def models_changed(self, event):
        '''models_changed(frame, event) --> None

        callback when something has changed in the model so that the
        user can be made aware that the model needs saving.
        '''
        try:
            self.model.saved = not event.permanent_change
        except AttributeError:
            self.model.saved = False
        else:
            self.plugin_control.OnGridChanged(event)
        self.set_title()

    def update_for_save(self):
        """Updates the various objects for a save"""
        self.model.set_script(self.script_editor.GetText())
        self.paramter_grid.opt.auto_sim=self.main_frame_menubar.mb_fit_autosim.IsChecked()
        self.paramter_grid.WriteConfig()

    def eh_mb_save(self, event):
        '''
        Event handler for saving a model file ...
        '''
        self.update_for_save()
        fname = self.model.get_filename()
        # If model hasn't been saved
        if fname=='':
            # Proceed with calling save as
            self.eh_mb_saveas(event)
        else:
            # If it has been saved just save it
            try:
                io.save_file(fname, self.model, self.solver_control.optimizer)
            except GenxIOError as e:
                ShowModelErrorDialog(self, e.__str__())
            except Exception as e:
                outp = StringIO()
                traceback.print_exc(200, outp)
                val = outp.getvalue()
                outp.close()
                ShowErrorDialog(self, 'Could not save the file. Python Error: \n%s'%(val,))
            self.set_title()
        self.main_frame_statusbar.SetStatusText('Model saved to file', 1)

    def eh_mb_print_plot(self, event):
        '''
        prints the current plot in the plot notebook.
        '''
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel<len(pages):
            pages[sel].Print()

    def eh_mb_print_grid(self, event):
        self.paramter_grid.Print()

    def eh_mb_print_script(self, event):
        warning("Event handler `eh_mb_print_script' not implemented")
        event.Skip()

    def eh_mb_export_orso(self, event):
        '''
        Exports the data to one file per data set with a basename with
        extention given by a save dialog.
        '''
        dlg = wx.FileDialog(self, message="Export data and model", defaultFile="",
                            wildcard="ORSO Text File (*.ort)|*.ort",
                            style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            try:
                self.model.export_orso(dlg.GetPath())
            except GenxIOError as e:
                ShowModelErrorDialog(self, str(e))
                self.main_frame_statusbar.SetStatusText(
                    'Error when exporting data', 1)
            except Exception as e:
                ShowErrorDialog(self, str(e), 'export data - model.export_orso')
                self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            else:
                self.main_frame_statusbar.SetStatusText('Data exported', 1)
        dlg.Destroy()

    def eh_mb_export_data(self, event):
        '''
        Exports the data to one file per data set with a basename with
        extension given by a save dialog.
        '''
        dlg = wx.FileDialog(self, message="Export data", defaultFile="",
                            wildcard="Dat File (*.dat)|*.dat",
                            style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            try:
                self.model.export_data(dlg.GetPath())
            except GenxIOError as e:
                ShowModelErrorDialog(self, str(e))
                self.main_frame_statusbar.SetStatusText(
                    'Error when exporting data', 1)
            except Exception as e:
                ShowErrorDialog(self, str(e), 'export data - model.export_data')
                self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            else:
                self.main_frame_statusbar.SetStatusText('Data exported', 1)

        dlg.Destroy()

    def eh_mb_export_table(self, event):
        '''
        Exports the table to a dat file given by a filedialog.
        '''
        dlg = wx.FileDialog(self, message="Export table", defaultFile="",
                            wildcard="Table File (*.tab)|*.tab",
                            style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            fname = dlg.GetPath()
            base, ext = os.path.splitext(fname)
            if ext=='':
                ext = '.tab'
            fname = base+ext
            result = True
            if os.path.exists(fname):
                filepath, filename = os.path.split(fname)
                result = ShowQuestionDialog(self,
                                            'The file %s already exists. Do you wish to overwrite it?'%filename
                                            , 'Overwrite?')
            if result:
                try:
                    # frame.model.export_table(dlg.GetPath())
                    self.model.export_table(fname)
                except GenxIOError as e:
                    ShowModelErrorDialog(self, str(e))
                    self.main_frame_statusbar.SetStatusText(
                        'Error when exporting table', 1)
                    return
                except Exception as e:
                    ShowErrorDialog(self, str(e),
                                    'export table - model.export_table')
                    self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
                    return
                else:
                    self.main_frame_statusbar.SetStatusText(
                        'Table exported to file', 1)

        dlg.Destroy()

    def eh_mb_export_script(self, event):
        '''
        Exports the script to a python file given by a filedialog.
        '''
        dlg = wx.FileDialog(self, message="Export script", defaultFile="",
                            wildcard="Python File (*.py)|*.py",
                            style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            fname = dlg.GetPath()
            base, ext = os.path.splitext(fname)
            if ext=='':
                ext = '.py'
            fname = base+ext
            result = True
            if os.path.exists(fname):
                filepath, filename = os.path.split(fname)
                result = ShowQuestionDialog(self,
                                            'The file %s already exists. Do you wish to overwrite it?'%filename
                                            , 'Overwrite?')
            if result:
                try:
                    # frame.model.export_script(dlg.GetPath())
                    self.model.export_script(fname)
                except GenxIOError as e:
                    ShowModelErrorDialog(self, str(e))
                    self.main_frame_statusbar.SetStatusText(
                        'Error when exporting script', 1)
                    return
                except Exception as e:
                    ShowErrorDialog(self, str(e),
                                    'export script - model.export_script')
                    self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
                    return
                else:
                    self.main_frame_statusbar.SetStatusText(
                        'Script exported to file', 1)

        dlg.Destroy()

    def eh_mb_quit(self, event):
        '''
        Quit the program
        '''
        # Check so the model is saved before quitting
        if not self.model.saved:
            ans = ShowQuestionDialog(self, 'If you continue any changes in'
                                           ' your model will not be saved.',
                                     'Model not saved')
            if not ans:
                return

        self.opt.hsize, self.opt.vsize = self.GetSize()
        self.opt.vsplit=self.ver_splitter.GetSashPosition()
        self.opt.hsplit=self.hor_splitter.GetSashPosition()
        self.opt.psplit=self.plot_splitter.GetSashPosition()
        self.opt.safe_config(default=True)

        config_path = appdirs.user_data_dir('GenX3', 'ArturGlavic')
        io.config.write_default(os.path.join(config_path, 'genx.conf'))

        self.findreplace_dlg.Destroy()
        self.findreplace_dlg = None

        event.Skip()
        self.Destroy()

    def eh_mb_copy_graph(self, event):
        '''
        Callback that copies the current graph in the plot notebook to
        the clipboard.
        '''
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel<len(pages):
            pages[sel].CopyToClipboard()

    def eh_mb_copy_sim(self, event):
        '''
        Copies the simulation and the data to the clipboard. Note that this
        copies ALL data.
        '''
        text_string = self.model.get_data_as_asciitable()
        text = wx.TextDataObject(text_string)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(text)
            wx.TheClipboard.Close()

    def eh_mb_copy_table(self, event):
        '''
        Copies the table as ascii text to the clipboard
        '''
        ascii_table = self.paramter_grid.table.pars.get_ascii_output()
        text_table = wx.TextDataObject(ascii_table)
        if wx.TheClipboard.Open():
            wx.TheClipboard.SetData(text_table)
            wx.TheClipboard.Close()

    def eh_mb_view_zoom(self, event):
        '''
        Takes care of clicks on the toolbar zoom button and the menu item zoom.
        '''
        if event.GetId()==ToolId.ZOOM:
            zoom_state = self.main_frame_toolbar.GetToolState(ToolId.ZOOM)
            self.main_frame_menubar.mb_view_zoom.Check(zoom_state)
        else:
            zoom_state = self.main_frame_menubar.mb_view_zoom.IsChecked()
            self.main_frame_toolbar.ToggleTool(ToolId.ZOOM, zoom_state)

        # Synchronize all plots with zoom state
        pages = self.get_pages()
        for page in pages:
            page.SetZoom(zoom_state)
        event.Skip()

    def eh_mb_view_grid_slider(self, event):
        """
        Change the state of the grid value input, either as slider or as a number.
        """
        self.paramter_grid.SetValueEditorSlider(self.main_frame_menubar.mb_view_grid_slider.IsChecked())
        self.paramter_grid.toggle_slider_tool(self.main_frame_menubar.mb_view_grid_slider.IsChecked())
        self.paramter_grid.Refresh()
        event.Skip()

    def eh_mb_fit_start(self, event):
        '''
        Event handler to start fitting
        '''
        if self.model.compiled:
            try:
                self.solver_control.StartFit()
            except GenxError as e:
                ShowModelErrorDialog(self, str(e))
                self.main_frame_statusbar.SetStatusText('Error in fitting', 1)
            except Exception as e:
                ShowErrorDialog(self, str(e))
                self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            else:
                self.main_frame_statusbar.SetStatusText('Fitting starting ...', 1)
        else:
            ShowNotificationDialog(self, 'The script is not compiled, do a'
                                         ' simulation before you start fitting.')

    def eh_mb_fit_stop(self, event):
        '''
        Event handler to stop the fitting routine
        '''
        self.solver_control.StopFit()

    def eh_mb_fit_resume(self, event):
        '''
        Event handler to resume the fitting routine. No initilization.
        '''
        if self.model.compiled:
            try:
                self.solver_control.ResumeFit()
            except GenxError as e:
                ShowModelErrorDialog(self, str(e))
                self.main_frame_statusbar.SetStatusText('Error in fitting', 1)
            except Exception as e:
                ShowErrorDialog(self, str(e))
                self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            else:
                self.main_frame_statusbar.SetStatusText('Fitting starting ...', 1)
        else:
            ShowNotificationDialog(self, 'The script is not compiled, do a'
                                         ' simulation before you start fitting.')

    def eh_mb_fit_analyze(self, event):
        warning("Event handler `eh_mb_fit_analyze' not implemented")
        event.Skip()

    def eh_mb_misc_showman(self, event):
        webbrowser.open_new(manual_url)

    def eh_mb_misc_about(self, event):
        '''
        Show an about box about GenX with some info...
        '''
        import numpy, scipy, matplotlib, platform
        useful = ''
        try:
            import numba
            useful += 'Numba: %s, '%numba.__version__
        except:
            pass
        try:
            import vtk
            useful += 'VTK: %s, '%vtk.vtkVersion.GetVTKVersion()
        except:
            pass
        try:
            import bumps
            useful += 'Bumps: %s, '%bumps.__version__
        except:
            pass

        info = wx.adv.AboutDialogInfo()
        info.Name = "GenX"
        info.Version = program_version
        info.Copyright = "(C) 2008 Matts Bjorck; 2020 Artur Glavic"
        info.Description = wordwrap(
            "GenX is a multipurpose refinement program using the differential "
            "evolution algorithm. It is developed  mainly for refining x-ray reflectivity "
            "and neutron reflectivity data."

            "\n\nThe versions of the mandatory libraries are:\n"
            "Python: %s, wxPython: %s, Numpy: %s, Scipy: %s, Matplotlib: %s"
            "\n\nThe non-mandatory but useful packages:\n%s"
            ""%(platform.python_version(), wx.__version__,
                numpy.__version__, scipy.__version__,
                matplotlib.__version__, useful),
            500, wx.ClientDC(self))
        info.WebSite = ("http:////genx.sourceforge.net", "GenX homepage")
        # No developers yet
        # info.Developers = []
        # head, tail = os.path.split(__file__)
        # license_text = file(head + '/LICENSE.txt','r').read()
        # license_text = file(_path + 'LICENSE.txt','r').read()
        # info.License = license_text#wordwrap(license_text, 500, wx.ClientDC(self))
        info.Licence = wordwrap('This program is free software: you can redistribute it and/or modify '
                                'it under the terms of the GNU General Public License as published by '
                                'the Free Software Foundation, either version 3 of the License, or '
                                '(at your option) any later version. '
                                '\n\nThis program is distributed in the hope that it will be useful, '
                                'but WITHOUT ANY WARRANTY; without even the implied warranty of '
                                'MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the '
                                'GNU General Public License for more details. '
                                '\n\nYou should have received a copy of the GNU General Public License '
                                'along with this program.  If not, see <http://www.gnu.org/licenses/>. '
                                , 400, wx.ClientDC(self))

        wx.adv.AboutBox(info)

    def eh_data_grid_choice(self, event):
        '''
        change the data displayed in the grid...
        '''
        dataset = self.model.data[event.GetSelection()]
        rows = self.data_grid.GetNumberRows()
        new_rows = max(len(dataset.x), len(dataset.y),
                       len(dataset.x_raw), len(dataset.y_raw))
        self.data_grid.DeleteRows(numRows=rows)
        self.data_grid.AppendRows(new_rows)
        [[self.data_grid.SetCellValue(row, col, '-') for col in range(6)] \
         for row in range(new_rows)]
        [self.data_grid.SetCellValue(row, 0, '%.3e'%dataset.x_raw[row]) \
         for row in range(len(dataset.x_raw))]
        [self.data_grid.SetCellValue(row, 1, '%.3e'%dataset.y_raw[row]) \
         for row in range(len(dataset.y_raw))]
        [self.data_grid.SetCellValue(row, 2, '%.3e'%dataset.error_raw[row]) \
         for row in range(len(dataset.error_raw))]
        [self.data_grid.SetCellValue(row, 3, '%.3e'%dataset.x[row]) \
         for row in range(len(dataset.x))]
        [self.data_grid.SetCellValue(row, 4, '%.3e'%dataset.y[row]) \
         for row in range(len(dataset.y))]
        [self.data_grid.SetCellValue(row, 5, '%.3e'%dataset.error[row]) \
         for row in range(len(dataset.error))]

    def eh_tb_new(self, event):
        self.eh_mb_new(event)

    def eh_tb_open(self, event):
        self.eh_mb_open(event)

    def eh_tb_save(self, event):
        self.eh_mb_save(event)

    def do_simulation(self, from_thread=False):
        if not from_thread: self.main_frame_statusbar.SetStatusText('Simulating...', 1)
        self.model.set_script(self.script_editor.GetText())
        try:
            # when updated from thread and was compiled before, do not compile again
            self.model.simulate(compile=not (from_thread and self.model.is_compiled()))
        except GenxError as e:
            wx.CallAfter(ShowModelErrorDialog, self, str(e))
            if not from_thread: self.main_frame_statusbar.SetStatusText('Error in simulation', 1)
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            wx.CallAfter(ShowErrorDialog, self, val)
            if not from_thread: self.main_frame_statusbar.SetStatusText('Fatal Error - simulate', 1)
        else:
            wx.CallAfter(_post_sim_plot_event, self, self.model, 'Simulation')
            wx.CallAfter(self.plugin_control.OnSimulate, None)
            if not from_thread: self.main_frame_statusbar.SetStatusText('Simulation Sucessful', 1)

    def set_possible_parameters_in_grid(self):
        # Now we should find the parameters that we can use to
        # in the grid
        try:
            pardict = self.model.get_possible_parameters()
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            # ShowErrorDialog(frame, val)
            ShowErrorDialog(self, val,
                            'simulate - model.get_possible_parameters')
            self.main_frame_statusbar.SetStatusText('Fatal Error', 0)
            return

        try:
            self.paramter_grid.SetParameterSelections(pardict)
        except Exception as e:
            ShowErrorDialog(self, str(e),
                            'simulate - parameter_grid.SetParameterSelection')
            self.main_frame_statusbar.SetStatusText('Fatal Error', 0)
            return
        # Set the function for which the parameter can be evaluated with
        self.paramter_grid.SetEvalFunc(self.model.eval_in_model)

    def eh_tb_simulate(self, event):
        '''
        Event handler for simulation.
        '''
        self.flag_simulating = True
        self.do_simulation()
        self.set_possible_parameters_in_grid()
        self.flag_simulating = False

    def eh_tb_start_fit(self, event):
        self.eh_mb_fit_start(event)

    def eh_tb_stop_fit(self, event):
        self.eh_mb_fit_stop(event)

    def eh_tb_restart_fit(self, event):
        self.eh_mb_fit_resume(event)

    def eh_tb_zoom(self, event):
        self.eh_mb_view_zoom(event)

    def eh_new_model(self, event):
        '''
        Callback for NEW_MODEL event. Used to update the script for
        a new model i.e. put the string to the correct value.
        '''
        # Set the string in the script_editor
        self.script_editor.SetText(event.GetModel().get_script())
        # Let the solvergui do its loading and updating:
        self.solver_control.ModelLoaded()
        # Lets update the mb_use_toggle_show Menu item
        self.main_frame_menubar.mb_use_toggle_show.Check(self.data_list.list_ctrl.opt.toggle_show)
        self.main_frame_menubar.mb_fit_autosim.Check(self.paramter_grid.opt.auto_sim)
        # Let other event handlers recieve the event as well
        event.Skip()

    def eh_mb_saveas(self, event):
        '''
        Event handler for save as ...
        '''
        dlg = wx.FileDialog(self, message="Save As", defaultFile="",
                            wildcard="HDF5 GenX File (*.hgx)|*.hgx|GenX File (*.gx)|*.gx",
                            style=wx.FD_SAVE  # | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            self.update_for_save()
            fname = dlg.GetPath()
            base, ext = os.path.splitext(fname)
            if ext=='':
                ext = '.hgx'
            fname = base+ext
            result = True
            if os.path.exists(fname):
                filepath, filename = os.path.split(fname)
                result = ShowQuestionDialog(self, 'The file %s already exists. Do you wish to overwrite it?'%filename,
                                            'Overwrite?')
            if result:
                try:
                    io.save_file(fname, self.model, self.solver_control.optimizer)
                except GenxIOError as e:
                    ShowModelErrorDialog(self, e.__str__())
                except Exception as e:
                    outp = StringIO()
                    traceback.print_exc(200, outp)
                    val = outp.getvalue()
                    outp.close()
                    ShowErrorDialog(self, 'Could not save the file. Python Error:\n%s'%(val,))
                self.set_title()
        dlg.Destroy()

    def eh_ex_status_text(self, event):
        self.main_frame_statusbar.SetStatusText(event.text, 1)

    def eh_ex_point_pick(self, event):
        self.main_frame_statusbar.SetStatusText(event.text, 2)

    def eh_ex_plot_settings_changed(self, event):
        '''
        Callback for the settings change event for the current plot
         - change the toggle for the zoom icon and change the menu items.
        '''
        self.main_frame_toolbar.ToggleTool(ToolId.ZOOM, event.zoomstate)
        self.main_frame_menubar.mb_view_zoom.Check(event.zoomstate)
        if event.yscale=='log':
            self.main_frame_menubar.mb_view_yscale_log.Check(True)
        elif event.yscale=='linear':
            self.main_frame_menubar.mb_view_yscale_lin.Check(True)
        if event.xscale=='log':
            self.main_frame_menubar.mb_view_xscale_log.Check(True)
        elif event.xscale=='linear':
            self.main_frame_menubar.mb_view_xscale_lin.Check(True)
        self.main_frame_menubar.mb_view_autoscale.Check(event.autoscale)
        event.Skip()

    def eh_tb_calc_error_bars(self, event):
        '''
        callback to calculate the error bars on the data.
        '''
        try:
            error_values = self.solver_control.CalcErrorBars()
        except ErrorBarError as e:
            ShowNotificationDialog(self, str(e))
        except Exception as e:
            ShowErrorDialog(self, str(e), 'solvergui - CalcErrorBars')
            self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            self.model.parameters.set_error_pars(error_values)
            self.paramter_grid.SetParameters(self.model.parameters)
            self.main_frame_statusbar.SetStatusText('Errorbars calculated', 1)

    def eh_tb_error_stats(self, event):
        try:
            from .bumps_interface import StatisticalAnalysisDialog
        except Exception as e:
            ShowErrorDialog(self, str(e), 'solvergui - CalcErrorBars')
            self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            dia = StatisticalAnalysisDialog(self, self.model)
            dia.ShowModal()

    def eh_plot_page_changed(self, event):
        '''plot_page_changed(frame, event) --> None

        Callback for page change in plot notebook. Changes the state of
        the zoom toggle button.
        '''
        sel = event.GetSelection()
        pages = self.get_pages()
        if sel<len(pages):
            zoom_state = pages[sel].GetZoom()
            # Set the zoom button to the correct value
            self.main_frame_toolbar.ToggleTool(ToolId.ZOOM, zoom_state)
            self.main_frame_menubar.mb_view_zoom.Check(zoom_state)

            yscale = pages[sel].GetYScale()
            if yscale=='log':
                self.main_frame_menubar.mb_view_yscale_log.Check(True)
            elif yscale=='linear':
                self.main_frame_menubar.mb_view_yscale_lin.Check(True)
            xscale = pages[sel].GetXScale()
            if xscale=='log':
                self.main_frame_menubar.mb_view_yscale_log.Check(True)
            elif xscale=='linear':
                self.main_frame_menubar.mb_view_yscale_lin.Check(True)
        event.Skip()

    def eh_mb_view_zoomall(self, event):
        '''zoomall(self, event) --> None

        Zoom out and show all data points
        '''
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel<len(pages):
            tmp = pages[sel].GetAutoScale()
            pages[sel].SetAutoScale(True)
            pages[sel].AutoScale()
            pages[sel].SetAutoScale(tmp)
            pages[sel].AutoScale()
        event.Skip()

    def view_yscale(self, value):
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel<len(pages):
            pages[sel].SetYScale(value)

    def view_xscale(self, value):
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel<len(pages):
            pages[sel].SetXScale(value)

    def eh_mb_view_yscale_log(self, event):
        self.view_yscale('log')

    def eh_mb_view_yscale_linear(self, event):
        self.view_yscale('linear')

    def eh_mb_view_xscale_log(self, event):
        '''
        Set the x-scale of the current plot. type should be linear or log, strings.
        '''
        self.view_xscale('log')

    def eh_mb_view_xscale_linear(self, event):
        self.view_xscale('linear')

    def eh_mb_view_autoscale(self, event):
        '''on_autoscale(frame, event) --> None

        Toggles the autoscale of the current plot.
        '''
        sel = self.sep_plot_notebook.GetSelection()
        pages = self.get_pages()
        if sel<len(pages):
            pages[sel].SetAutoScale(not pages[sel].GetAutoScale())

    def activate_cuda(self):
        dlg = wx.ProgressDialog(parent=self,
                                maximum=3,
                                message="Compiling CUDA GPU computing functions with Numba",
                                title='Activating CUDA...')
        dlg.Show()

        dlg.Update(1)
        from genx.models.lib import paratt_cuda
        dlg.Update(2)
        from genx.models.lib import neutron_cuda
        dlg.Update(3)
        from models.lib import paratt, neutron_refl
        paratt.Refl = paratt_cuda.Refl
        paratt.ReflQ = paratt_cuda.ReflQ
        paratt.Refl_nvary2 = paratt_cuda.Refl_nvary2
        neutron_refl.Refl = neutron_cuda.Refl
        from genx.models.lib import paratt, neutron_refl
        paratt.Refl = paratt_cuda.Refl
        paratt.ReflQ = paratt_cuda.ReflQ
        paratt.Refl_nvary2 = paratt_cuda.Refl_nvary2
        neutron_refl.Refl = neutron_cuda.Refl

        dlg.Destroy()

    def deactivate_cuda(self):
        from genx.models.lib import paratt_numba, neutron_numba
        from models.lib import paratt, neutron_refl
        paratt.Refl = paratt_numba.Refl
        paratt.ReflQ = paratt_numba.ReflQ
        paratt.Refl_nvary2 = paratt_numba.Refl_nvary2
        neutron_refl.Refl = neutron_numba.Refl
        from genx.models.lib import paratt, neutron_refl
        paratt.Refl = paratt_numba.Refl
        paratt.ReflQ = paratt_numba.ReflQ
        paratt.Refl_nvary2 = paratt_numba.Refl_nvary2
        neutron_refl.Refl = neutron_numba.Refl

    def eh_mb_use_cuda(self, event):
        if self.main_frame_menubar.mb_use_cuda.IsChecked():
            self.activate_cuda()
        else:
            self.deactivate_cuda()

    def eh_mb_set_opt(self, event):
        self.solver_control.ParametersDialog(self)

    def eh_mb_import_data(self, event):
        '''
        callback to import data into the program
        '''
        try:
            self.data_list.eh_tb_open(event)
        except Exception as e:
            ShowErrorDialog(self, str(e),
                            'import data - data_list.eh_tb_open')
            self.main_frame_statusbar.SetStatusText('Fatal Error', 1)

    def eh_mb_import_table(self, event):
        '''
        imports a table from the file given by a file dialog box
        '''
        dlg = wx.FileDialog(self, message="Import script", defaultFile="",
                            wildcard="Table File (*.tab)|*.tab|All files (*.*)|*.*",
                            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            try:
                self.model.import_table(dlg.GetPath())
            except GenxIOError as e:
                ShowModelErrorDialog(self, str(e))
                self.main_frame_statusbar.SetStatusText(
                    'Error when importing script', 1)
                dlg.Destroy()
                return
            except Exception as e:
                ShowErrorDialog(self, str(e),
                                'import script - model.import_script')
                self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
                dlg.Destroy()
                return

        dlg.Destroy()
        # Post event to tell that the model has cahnged
        _post_new_model_event(self, self.model)
        self.main_frame_statusbar.SetStatusText('Table imported from file', 1)

    def eh_mb_import_script(self, event):
        '''
        imports a script from the file given by a file dialog box
        '''
        dlg = wx.FileDialog(self, message="Import script", defaultFile="",
                            wildcard="Python files (*.py)|*.py|All files (*.*)|*.*",
                            style=wx.FD_OPEN | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            try:
                self.model.import_script(dlg.GetPath())
                # frame.model.import_script(fname)
            except GenxIOError as e:
                ShowModelErrorDialog(self, str(e))
                self.main_frame_statusbar.SetStatusText(
                    'Error when importing script', 1)
                return
            except Exception as e:
                ShowErrorDialog(self, str(e),
                                'import script - model.import_script')
                self.main_frame_statusbar.SetStatusText('Fatal Error', 1)
                return
            try:
                self.plugin_control.OnOpenModel(None)
            except Exception as e:
                outp = StringIO()
                traceback.print_exc(200, outp)
                val = outp.getvalue()
                outp.close()
                ShowErrorDialog(self, 'Problems when plugins processed model.' \
                                      ' Python Error:\n%s'%(val,))
            else:
                self.main_frame_statusbar.SetStatusText(
                    'Script imported from file', 1)
        dlg.Destroy()
        # Post event to tell that the model has changed
        _post_new_model_event(self, self.model)

    def eh_external_fom_value(self, event):
        '''
        Callback to update the fom_value displayed by the gui
        '''
        fom_value = event.model.fom
        fom_name = event.model.fom_func.__name__
        if fom_value:
            self.main_frame_fom_text.SetLabel('        FOM %s: %.4e'%(fom_name, fom_value))
        else:
            self.main_frame_fom_text.SetLabel('        FOM %s: None'%fom_name)
        event.Skip()

    def eh_mb_set_dal(self, event):
        self.data_list.DataLoaderSettingsDialog()

    def eh_external_update_data_grid_choice(self, event):
        '''
        Updates the choices of the grids to display from the data.
        '''
        data = event.GetData()
        names = [data_set.name for data_set in data]
        self.data_grid_choice.Clear()
        self.data_grid_choice.AppendItems(names)
        event.Skip()

    def eh_external_update_data(self, event):
        self.plugin_control.OnDataChanged(event)
        event.Skip()

    def eh_mb_fit_evaluate(self, event):
        '''
        Event handler for only evaluating the Sim function - no recompiling
        '''
        self.flag_simulating = True
        self.main_frame_statusbar.SetStatusText('Simulating...', 1)
        # Compile is not necessary when using simualate...
        # frame.model.compile_script()
        try:
            self.model.simulate(compile=False)
        except GenxError as e:
            ShowModelErrorDialog(self, str(e))
            self.main_frame_statusbar.SetStatusText('Error in simulation', 1)
        except Exception as e:
            outp = StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            ShowErrorDialog(self, val)
            self.main_frame_statusbar.SetStatusText('Fatal Error - simulate', 1)
        else:
            _post_sim_plot_event(self, self.model, 'Simulation')
            self.main_frame_statusbar.SetStatusText('Simulation Sucessful', 1)
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
        '''
        Show a help dialog for information about the different models.
        '''
        dlg = help.PluginHelpDialog(self, 'models', title="Models help")
        if self.model.is_compiled():
            current_model = self.model.script_module.model.__name__.rsplit('.', 1)[1]
            if current_model in dlg.choice.GetStrings():
                dlg.choice.SetStringSelection(current_model)
                dlg.on_choice(None)
        dlg.Show()

    def eh_external_model_changed(self, event):
        '''
        callback when something has changed in the model so that the
        user can be made aware that the model needs saving.
        '''
        try:
            self.model.saved = not event.permanent_change
        except AttributeError:
            self.model.saved = False
        else:
            self.plugin_control.OnGridChanged(event)
        self.set_title()
        event.Skip()

    def eh_mb_plugins_help(self, event):
        '''
        Show a help dialog for information about the different plugins.
        '''
        dlg = help.PluginHelpDialog(self, 'plugins.add_ons', title="Plugins help")
        dlg.Show()

    def eh_mb_data_loaders_help(self, event):
        '''
        Show a help dialog for information about the different data_loaders.
        '''
        dlg = help.PluginHelpDialog(self, 'plugins.data_loaders', title="Data loaders help")
        dlg.Show()

    def eh_mb_findreplace(self, event):
        self.findreplace_dlg.Show(True)

    def eh_external_find(self, event):
        '''callback for find events - coupled to the script
        '''
        evtype = event.GetEventType()

        def find():
            find_str = event.GetFindString()
            ##print frame.findreplace_data.GetFlags()
            flags = event.GetFlags()
            if flags & 1:
                ##print "Searching down"
                pos = self.script_editor.SearchNext(flags, find_str)
            else:
                ##print "Searching up"
                pos = self.script_editor.SearchPrev(flags, find_str)
            if pos==-1:
                self.main_frame_statusbar.SetStatusText(
                    'Could not find text %s'%find_str, 1)
            return pos

        def replace():
            replace_str = event.GetReplaceString()
            self.script_editor.ReplaceSelection(replace_str)

        # Deal with the different cases
        if evtype==wx.wxEVT_COMMAND_FIND:
            self.script_editor.SearchAnchor()
            find()

        elif evtype==wx.wxEVT_COMMAND_FIND_NEXT:
            pnew = self.script_editor.GetSelectionEnd()
            ##print pnew
            self.script_editor.GotoPos(pnew)
            self.script_editor.SetAnchor(pnew)
            self.script_editor.SearchAnchor()
            ##print 'Finding next'
            find()

        elif evtype==wx.wxEVT_COMMAND_FIND_REPLACE:
            # If we do not have found text already
            # or if we have marked other text by mistake...
            if self.script_editor.GetSelectedText()!= \
                    event.GetFindString():
                find()
            # We already have found and marked text that we should
            # replace
            else:
                self.script_editor.ReplaceSelection(
                    event.GetReplaceString())
                # Find a new text to replace
                find()
        elif evtype==wx.wxEVT_COMMAND_FIND_REPLACE_ALL:
            if self.script_editor.GetSelectedText()!= \
                    event.GetFindString():
                pos = find()
            else:
                pos = -1
            i = 0
            while pos!=-1:
                self.script_editor.ReplaceSelection(
                    event.GetReplaceString())
                i += 1
                pos = find()
            self.main_frame_statusbar.SetStatusText(
                'Replaces %d occurancies of  %s'%(i,
                                                  event.GetFindString()), 1)

        else:
           ShowErrorDialog(self, 'Faulty event supplied in find and'
                                 ' repalce functionallity', 'on_find_event')
        # This will scroll the editor to the right position so we can see
        # the text
        self.script_editor.EnsureCaretVisible()

    def eh_mb_fom_help(self, event):
        '''
        Show a help dialog for information about the different fom.
        '''
        dlg = help.PluginHelpDialog(self, 'fom_funcs', title="FOM functions help")
        dlg.Show()

    def eh_mb_view_use_toggle_show(self, event):
        new_val=self.main_frame_menubar.mb_use_toggle_show.IsChecked()
        self.data_list.list_ctrl.SetShowToggle(new_val)

    def eh_mb_misc_openhomepage(self, event):
        webbrowser.open_new(homepage_url)

    def eh_show_startup_dialog(self, event):
        pre_dia=self.wstartup.copy()
        self.startup_dialog(config_path, force_show=True)
        print(pre_dia==self.wstartup)

    def simulation_loop(self):
        """ Simulation loop for threading to increase the speed of the interactive simulations
        :param self:
        :return:
        """
        self.flag_simulating = True
        while self.simulation_queue_counter>0:
            self.do_simulation(from_thread=True)
            time.sleep(0.1)
            self.simulation_queue_counter = min(1, self.simulation_queue_counter-1)
        self.flag_simulating = False

    def eh_external_parameter_value_changed(self, event):
        """
        Event handler for when a value of a parameter in the grid has been updated.
        """
        self.simulation_queue_counter += 1
        if self.main_frame_menubar.mb_fit_autosim.IsChecked() and not self.flag_simulating:
            _thread.start_new_thread(self.simulation_loop, ())

    def eh_mb_fit_autosim(self, event):
        event.Skip()


class MyApp(wx.App):
    def __init__(self, show_startup, *args, **kwargs):
        debug('App init started')
        self.show_startup=show_startup
        wx.App.__init__(self, *args, **kwargs)
        debug('App init complete')

    def OnInit(self):
        debug('entering init phase')
        locale=wx.Locale(wx.LANGUAGE_ENGLISH)
        self.locale=locale

        main_frame=MainFrame(self)
        self.SetTopWindow(main_frame)

        # main_frame.Show()
        if self.show_startup:
            main_frame.startup_dialog(config_path)

        debug('init complete')
        wx.CallAfter(main_frame.plugin_control.LoadDefaultPlugins)
        wx.CallAfter(main_frame.Show)
        return 1


class StartUpConfigDialog(wx.Dialog):
    def __init__(self, parent, config_folder, show_cb=True, wide=False):
        wx.Dialog.__init__(self, parent, -1, 'Change Startup Configuration')

        self.config_folder=config_folder
        self.selected_config=None

        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add((-1, 10), 0, wx.EXPAND)

        sizer.Add(wx.StaticText(self, label='Choose the profile you want GenX to use:            '),
                  0, wx.ALIGN_LEFT, 5)
        self.profiles=self.get_possible_configs()
        self.config_list=wx.ListBox(self, size=(-1, 200), choices=self.profiles, style=wx.LB_SINGLE)
        self.config_list.SetSelection(self.profiles.index('SimpleReflectivity'))
        sizer.Add(self.config_list, 1, wx.GROW | wx.TOP, 5)

        startup_cb=wx.CheckBox(self, -1, "Show at startup", style=wx.ALIGN_LEFT)
        startup_cb.SetValue(show_cb)
        self.startup_cb=startup_cb
        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(startup_cb, 0, wx.EXPAND, 5)
        wide_cb=wx.CheckBox(self, -1, "Widescreen (need restart)", style=wx.ALIGN_LEFT)
        wide_cb.SetValue(wide)
        self.wide_cb=wide_cb
        sizer.Add(wide_cb, 0, wx.EXPAND, 5)

        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(wx.StaticText(self, label='These settings can be changed at the menu:\n Options/Startup Profile'),
                  0, wx.ALIGN_LEFT, 5)

        # Add the Dilaog buttons
        button_sizer=wx.StdDialogButtonSizer()
        okay_button=wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        # Add some eventhandlers
        self.Bind(wx.EVT_BUTTON, self.OnClickOkay, okay_button)

        line=wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.TOP, 20)

        sizer.Add((-1, 4), 0, wx.EXPAND)
        sizer.Add(button_sizer, 0,
                  flag=wx.ALIGN_RIGHT, border=20)
        sizer.Add((-1, 4), 0, wx.EXPAND)

        main_sizer=wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add((10, -1), 0, wx.EXPAND)
        main_sizer.Add(sizer, 1, wx.EXPAND)
        main_sizer.Add((10, -1), 0, wx.EXPAND)
        self.SetSizer(main_sizer)

        sizer.Fit(self)
        self.Layout()
        self.CentreOnScreen()

    def OnClickOkay(self, event):
        self.selected_config=self.profiles[self.config_list.GetSelection()]
        self.show_at_startup=self.startup_cb.GetValue()
        self.widescreen=self.wide_cb.GetValue()

    def GetConfigFile(self):
        if self.selected_config:
            return self.selected_config+'.conf'
        else:
            return None

    def GetShowAtStartup(self):
        return self.show_at_startup

    def GetWidescreen(self):
        return self.widescreen

    def get_possible_configs(self)->List[str]:
        '''
        search the plugin directory. 
        Checks the list for python scripts and returns a list of 
        module names that are loadable .
        '''
        plugins=[s[:-5] for s in os.listdir(self.config_folder) if '.conf'==s[-5:]
                 and s[:2]!='__']
        return plugins

def ShowQuestionDialog(frame, message, title='Question?'):
    dlg=wx.MessageDialog(frame, message,
                         title,
                         wx.OK | wx.CANCEL | wx.OK_DEFAULT | wx.ICON_QUESTION
                         )
    result=dlg.ShowModal()==wx.ID_OK
    dlg.Destroy()
    return result

def ShowModelErrorDialog(frame, message):
    dlg=wx.MessageDialog(frame, message,
                         'Warning',
                         wx.OK | wx.ICON_WARNING
                         )
    dlg.ShowModal()
    dlg.Destroy()

def ShowNotificationDialog(frame, message):
    dlg=wx.MessageDialog(frame, message,
                         'Information',
                         wx.OK | wx.ICON_INFORMATION
                         )
    dlg.ShowModal()
    dlg.Destroy()

def ShowErrorDialog(frame, message, position=''):
    if position!='':
        dlg=wx.MessageDialog(frame, message+'\n'+'Position: '+position,
                             'FATAL ERROR',
                             wx.OK | wx.ICON_ERROR
                             )
    else:
        dlg=wx.MessageDialog(frame, message,
                             'FATAL ERROR',
                             wx.OK | wx.ICON_ERROR
                             )
    dlg.ShowModal()
    dlg.Destroy()

# =============================================================================
# Custom events needed for updating and message parsing between the different
# modules.

class GenericModelEvent(wx.CommandEvent):
    '''
    Event class for a new model - for updating
    of the paramters, plots and script.
    '''

    def __init__(self, evt_type, id, model):
        wx.CommandEvent.__init__(self, evt_type, id)
        self.model=model
        self.description=''

    def GetModel(self):
        return self.model

    def SetModel(self, model):
        self.model=model

    def SetDescription(self, desc):
        '''
        Set a string that describes the event that has occurred
        '''
        self.description=desc

# Generating an event type:
myEVT_NEW_MODEL=wx.NewEventType()
# Creating an event binder object
EVT_NEW_MODEL=wx.PyEventBinder(myEVT_NEW_MODEL)

def _post_new_model_event(parent, model, desc=''):
    # Send an event that a new data set has been loaded
    evt=GenericModelEvent(myEVT_NEW_MODEL, parent.GetId(), model)
    evt.SetDescription(desc)
    # Process the event!
    parent.GetEventHandler().ProcessEvent(evt)

# Generating an event type:
myEVT_SIM_PLOT=wx.NewEventType()
# Creating an event binder object
EVT_SIM_PLOT=wx.PyEventBinder(myEVT_SIM_PLOT)

def _post_sim_plot_event(parent, model, desc=''):
    # Send an event that a new data set ahs been loaded
    evt=GenericModelEvent(myEVT_SIM_PLOT, parent.GetId(), model)
    evt.SetDescription(desc)
    # Process the event!
    parent.GetEventHandler().ProcessEvent(evt)


if __name__=="__main__":
    app=MyApp(True, 0)
    app.MainLoop()
