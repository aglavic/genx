"""
Custom IDs for GenX menus and toolbars.
"""

from enum import Enum

import wx


class ToolId(int, Enum):
    NEW_MODEL = wx.Window.NewControlId()
    NEW_FROM_FILE = wx.Window.NewControlId()
    OPEN_MODEL = wx.Window.NewControlId()
    SAVE_MODEL = wx.Window.NewControlId()
    SIM_MODEL = wx.Window.NewControlId()
    START_FIT = wx.Window.NewControlId()
    STOP_FIT = wx.Window.NewControlId()
    RESTART_FIT = wx.Window.NewControlId()
    CALC_ERROR = wx.Window.NewControlId()
    SOLVER_SELECT = wx.Window.NewControlId()
    ZOOM = wx.Window.NewControlId()
    ERROR_STATS = wx.Window.NewControlId()


class MenuId(int, Enum):
    NEW_MODEL = wx.Window.NewControlId()
    NEW_FROM_FILE = wx.Window.NewControlId()
    OPEN_MODEL = wx.Window.NewControlId()
    SAVE_MODEL = wx.Window.NewControlId()
    SAVE_MODEL_AS = wx.Window.NewControlId()
    MODEL_BATCH = wx.Window.NewControlId()

    IMPORT_DATA = wx.Window.NewControlId()
    IMPORT_TABLE = wx.Window.NewControlId()
    IMPORT_SCRIPT = wx.Window.NewControlId()

    EXPORT_ORSO = wx.Window.NewControlId()
    EXPORT_DATA = wx.Window.NewControlId()
    EXPORT_TABLE = wx.Window.NewControlId()
    EXPORT_SCRIPT = wx.Window.NewControlId()

    PUBLISH_PLOT = wx.Window.NewControlId()
    PRINT_PLOT = wx.Window.NewControlId()
    PRINT_GRID = wx.Window.NewControlId()
    PRINT_SCRIPT = wx.Window.NewControlId()

    QUIT = wx.Window.NewControlId()

    UNDO = wx.Window.NewControlId()
    REDO = wx.Window.NewControlId()
    HISTORY = wx.Window.NewControlId()
    COPY_GRAPH = wx.Window.NewControlId()
    COPY_SIM = wx.Window.NewControlId()
    COPY_TABLE = wx.Window.NewControlId()
    FIND_REPLACE = wx.Window.NewControlId()
    OPEN_IN_EDITOR = wx.Window.NewControlId()

    NEW_DATA = wx.Window.NewControlId()
    DELETE_DATA = wx.Window.NewControlId()
    LOWER_DATA = wx.Window.NewControlId()
    RAISE_DATA = wx.Window.NewControlId()
    TOGGLE_SHOW = wx.Window.NewControlId()
    TOGGLE_USE = wx.Window.NewControlId()
    TOGGLE_ERROR = wx.Window.NewControlId()
    CALCS_DATA = wx.Window.NewControlId()

    TOGGLE_SLIDER = wx.Window.NewControlId()
    ZOOM = wx.Window.NewControlId()
    ZOOM_ALL = wx.Window.NewControlId()
    Y_SCALE_LIN = wx.Window.NewControlId()
    Y_SCALE_LOG = wx.Window.NewControlId()
    X_SCALE_LIN = wx.Window.NewControlId()
    X_SCALE_LOG = wx.Window.NewControlId()
    AUTO_SCALE = wx.Window.NewControlId()
    USE_TOGGLE_SHOW = wx.Window.NewControlId()

    SIM_MODEL = wx.Window.NewControlId()
    EVAL_MODEL = wx.Window.NewControlId()
    TOGGLE_CUDA = wx.Window.NewControlId()
    START_FIT = wx.Window.NewControlId()
    STOP_FIT = wx.Window.NewControlId()
    RESTART_FIT = wx.Window.NewControlId()
    CALC_ERROR = wx.Window.NewControlId()
    ANALYZE = wx.Window.NewControlId()
    AUTO_SIM = wx.Window.NewControlId()

    SET_OPTIMIZER = wx.Window.NewControlId()
    SET_DATA_LOADER = wx.Window.NewControlId()
    SET_IMPORT = wx.Window.NewControlId()
    SET_PLOT = wx.Window.NewControlId()
    SET_PROFILE = wx.Window.NewControlId()
    SET_EDITOR = wx.Window.NewControlId()

    HELP_MODEL = wx.Window.NewControlId()
    HELP_FOM = wx.Window.NewControlId()
    HELP_PLUGINS = wx.Window.NewControlId()
    HELP_DATA_LOADERS = wx.Window.NewControlId()
    HELP_MANUAL = wx.Window.NewControlId()
    HELP_EXAMPLES = wx.Window.NewControlId()
    HELP_HOMEPAGE = wx.Window.NewControlId()
    HELP_ABOUT = wx.Window.NewControlId()
    HELP_DEBUG = wx.Window.NewControlId()


def rebuild_IDs():
    # re-generate above Enums with new IDs for e.g. window recreation
    global ToolId, MenuId
    new_vals = dict([(key, wx.Window.NewControlId()) for key in ToolId.__members__.keys()])
    ToolId = Enum("ToolId", new_vals, type=int, module=__name__)
    new_vals = dict([(key, wx.Window.NewControlId()) for key in MenuId.__members__.keys()])
    MenuId = Enum("MenuId", new_vals, type=int, module=__name__)
