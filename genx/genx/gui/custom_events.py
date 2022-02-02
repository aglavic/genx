'''
All custom events used in GenX to be directly imported.
'''
import wx
from wx.lib.newevent import NewEvent
from ..core.decorators import decorator

# a decorator to make sure events processed by a method are skiped first
@decorator
def skips_event(function, *args, **kwargs):
    event=args[1]
    event.Skip()
    return function(*args, **kwargs)

## main_window
# Generating an event type:
new_model_type=wx.NewEventType()
# Creating an event binder object
EVT_NEW_MODEL=wx.PyEventBinder(new_model_type)
# Generating an event type:
sim_plot_type=wx.NewEventType()
# Creating an event binder object
EVT_SIM_PLOT=wx.PyEventBinder(sim_plot_type)


## parametergrid
# Event for when the grid has new values
grid_change, EVT_PARAMETER_GRID_CHANGE=NewEvent()
# Event for then the value of a parameter has changed. Should be used to do
# simulations interactively.
value_change, EVT_PARAMETER_VALUE_CHANGE=NewEvent()
set_parameter_value, EVT_PARAMETER_SET_VALUE=NewEvent()
move_parameter, EVT_MOVE_PARAMETER=NewEvent()
inset_parameter, EVT_INSERT_PARAMETER=NewEvent()
delete_parameters, EVT_DELETE_PARAMETERS=NewEvent()
sort_and_group_parameters, EVT_SORT_AND_GROUP_PARAMETERS=NewEvent()

## datalist
# Generating an event type:
data_list_type=wx.NewEventType()
# Creating an event binder object
EVT_DATA_LIST=wx.PyEventBinder(data_list_type)
# Send when plot setting is to be changed
update_plotsettings, EVT_UPDATE_PLOTSETTINGS=NewEvent()

## plotpanel
# Event for a click inside an plot which yields a number
plot_position, EVT_PLOT_POSITION=NewEvent()
# Event to tell the main window that the zoom state has changed
state_changed, EVT_PLOT_SETTINGS_CHANGE=NewEvent()

## solvergui
# Custom events needed for updating and message parsing between the different
# modules.
update_script, EVT_UPDATE_SCRIPT=NewEvent()
update_plot, EVT_UPDATE_PLOT=NewEvent()
update_text, EVT_SOLVER_UPDATE_TEXT=NewEvent()
update_parameters, EVT_UPDATE_PARAMETERS=NewEvent()
fitting_ended, EVT_FITTING_ENDED=NewEvent()
autosave, EVT_AUTOSAVE=NewEvent()
batch_next, EVT_BATCH_NEXT=NewEvent()

## plugins
# new model is ready with a script as value.
update_model_event, EVT_UPDATE_MODEL=wx.lib.newevent.NewEvent()

## custom_logging
log_message_event, EVT_LOG_MESSAGE=wx.lib.newevent.NewEvent()