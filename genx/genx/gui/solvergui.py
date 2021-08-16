'''
Controller class for the differnetial evolution class diffev
Takes care of stopping and starting - output to the gui as well
as some input from dialog boxes.
'''
import numpy as np
import time
from typing import Union
from threading import Thread, Event

import wx
import wx.lib.newevent

from .exception_handling import CatchModelError
from .settings_dialog import SettingsDialog
from .. import diffev, fom_funcs, model_control, levenberg_marquardt
from ..core.custom_logging import iprint
from ..solver_basis import SolverParameterInfo, SolverResultInfo, SolverUpdateInfo, GenxOptimizerCallback


# Custom events needed for updating and message parsing between the different
# modules.
(update_plot, EVT_UPDATE_PLOT)=wx.lib.newevent.NewEvent()
(update_text, EVT_SOLVER_UPDATE_TEXT)=wx.lib.newevent.NewEvent()
(update_parameters, EVT_UPDATE_PARAMETERS)=wx.lib.newevent.NewEvent()
(fitting_ended, EVT_FITTING_ENDED)=wx.lib.newevent.NewEvent()
(autosave, EVT_AUTOSAVE)=wx.lib.newevent.NewEvent()

class GuiCallbacks(GenxOptimizerCallback):
    def __init__(self, parent: wx.Window):
        self.parent=parent

    def text_output(self, text):
        '''
        Function to present the output from the optimizer to the user.
        Takes a string as input.
        '''
        evt=update_text(text=text)
        wx.QueueEvent(self.parent, evt)

    def plot_output(self, update_data):
        '''
        Solver to present the graphical output from the optimizer to the
        user. Takes the solver as input argument and picks out the
        variables to show in the GUI.
        '''
        evt=update_plot(data=update_data.data, fom_value=update_data.fom_value,
                        fom_name=update_data.fom_name,
                        fom_log=update_data.fom_log, update_fit=update_data.new_best,
                        desc='Fitting update')
        wx.QueueEvent(self.parent, evt)

    def parameter_output(self, param_info):
        '''
        Function to send an update event to update windows that displays
        the parameters to update the values.
        Takes the solver as input argument and picks out the variables to
        show in the GUI.
        '''
        evt=update_parameters(values=param_info.values,
                              new_best=param_info.new_best,
                              population=param_info.population,
                              max_val=param_info.max_val,
                              min_val=param_info.min_val,
                              fitting=True,
                              desc='Parameter Update', update_errors=False,
                              permanent_change=False)
        wx.QueueEvent(self.parent, evt)

    def fitting_ended(self, result_data):
        '''
        function used to post an event when the fitting has ended.
        This must be done since it is not thread safe otherwise. Same GUI in
        two threads when dialogs are run. dangerous...
        '''
        evt=fitting_ended(start_guess=result_data.start_guess,
                          error_message=result_data.error_message,
                          values=result_data.values,
                          new_best=result_data.new_best,
                          population=result_data.population,
                          max_val=result_data.max_val,
                          min_val=result_data.min_val,
                          fitting=True, desc='Fitting Ended')
        wx.QueueEvent(self.parent, evt)

    def autosave(self):
        '''
        Function that conducts an autosave of the model.
        '''
        evt=autosave()
        wx.QueueEvent(self.parent, evt)

class DelayedCallbacks(Thread, GuiCallbacks):
    last_text: Union[str, None]=None
    last_param: Union[SolverParameterInfo, None]=None
    last_update: Union[SolverUpdateInfo, None]=None
    last_endet: Union[SolverResultInfo, None]=None
    min_time=0.5
    last_iter: float=0.0
    wait_lock: Event
    stop_thread: Event

    def __init__(self, parent: wx.Window):
        GuiCallbacks.__init__(self, parent)
        Thread.__init__(self, daemon=True, name="GenxDelayedCallbacks")
        self.wait_lock=Event()
        self.stop_thread=Event()

    def run(self):
        self.last_iter=time.time()
        self.stop_thread.clear()
        while not self.stop_thread.is_set():
            # main loop for checking for updates and sending GUI events
            time.sleep(max(0., (self.last_iter-time.time()+self.min_time)))
            if self.last_text:
                GuiCallbacks.text_output(self, self.last_text)
                self.last_text=None
            if self.last_param:
                GuiCallbacks.parameter_output(self, self.last_param)
                self.last_param=None
            if self.last_update:
                GuiCallbacks.plot_output(self, self.last_update)
                self.last_update=None
            if self.last_endet:
                GuiCallbacks.fitting_ended(self, self.last_endet)
                self.last_endet=None
            self.last_iter=time.time()
            self.wait_lock.clear()
            self.wait_lock.wait()

    def exit(self):
        self.stop_thread.set()
        self.wait_lock.set()
        self.join(timeout=1.0)

    def text_output(self, text):
        self.last_text=text
        self.wait_lock.set()

    def fitting_ended(self, result_data):
        self.last_endet=result_data
        self.wait_lock.set()

    def parameter_output(self, param_info):
        self.last_param=param_info
        self.wait_lock.set()

    def plot_output(self, update_data):
        self.last_update=update_data
        self.wait_lock.set()

class ModelControlGUI:
    '''
    Class to take care of the GUI - solver interaction.
    Implements dialogboxes for setting parameters and controls
    for the solver routine. This is where the application specific
    code are used i.e. interfacing the optimization code to the GUI.
    '''

    def __init__(self, parent):
        self.parent=parent
        self.solvers={
            'Differential Evolution': diffev.DiffEv(),
            'Levenberg-Marquardt': levenberg_marquardt.LMOptimizer()
            }
        try:
            from ..bumps_optimizer import BumpsOptimizer
        except ImportError:
            pass
        else:
            self.solvers['Bumps']=BumpsOptimizer()

        self.controller=model_control.ModelController(self.solvers['Differential Evolution'])
        self.callback_controller=DelayedCallbacks(parent)
        self.callback_controller.start()
        self.controller.set_callbacks(self.callback_controller)
        self.parent.Bind(EVT_FITTING_ENDED, self.OnFittingEnded)
        self.parent.Bind(EVT_AUTOSAVE, self.AutoSave)

        # Now load the default configuration
        self.ReadConfig()

    def get_solvers(self):
        return list(self.solvers.keys())

    def set_solver(self, name):
        self.controller.optimizer=self.solvers[name]
        self.controller.set_callbacks(self.callback_controller)

    def ReadConfig(self):
        '''
        Reads the parameter that should be read from the config file.
        And set the parameters in both the optimizer and this class.
        '''
        self.controller.ReadConfig()

    def WriteConfig(self):
        '''
        Writes the current configuration of the solver to file.
        '''
        self.controller.WriteConfig()

    def ParametersDialog(self, frame):
        '''
        Shows the Parameters dialog box to set the parameters for the solver.
        '''
        # Update the configuration if a model has been loaded after
        # the object have been created..
        self.ReadConfig()
        fom_func_name=self.controller.model.fom_func.__name__
        if not fom_func_name in fom_funcs.func_names:
            ShowWarningDialog(self.parent, 'The loaded fom function, ' \
                              +fom_func_name+', does not exist '+ \
                              'in the local fom_funcs file. The fom fucntion has been'+
                              ' temporary added to the list of availabe fom functions')
            fom_funcs.func_names.append(fom_func_name)
            exectext='fom_funcs.'+fom_func_name+ \
                     ' = self.parent.model.fom_func'
            exec(exectext, locals(), globals())

        combined_options=self.controller.model.solver_parameters|self.controller.optimizer.opt
        dlg=SettingsDialog(frame, combined_options, title='Optimizer Settings')

        res=dlg.ShowModal()
        if res==wx.ID_OK:
            self.controller.model.WriteConfig()
            self.controller.optimizer.WriteConfig()
        dlg.Destroy()

    def ModelLoaded(self):
        '''
        Function that takes care of resetting everything when a model has
        been loaded.
        '''
        evt=update_plot(model=self.controller.get_fitted_model(),
                        fom_log=self.controller.get_fom_log(), update_fit=False,
                        desc='Model loaded')
        wx.PostEvent(self.parent, evt)

        # Update the parameter plot ...
        if self.controller.is_configured():
            # remember to add a check
            res=self.controller.get_result_info()
            try:
                evt=update_parameters(values=res.values,
                                      new_best=False,
                                      population=res.population,
                                      max_val=res.par_max,
                                      min_val=res.par_min,
                                      fitting=True,
                                      desc='Parameter Update', update_errors=False,
                                      permanent_change=False)
            except AttributeError:
                iprint('Could not create data for parameters')
            else:
                wx.PostEvent(self.parent, evt)

    def OnFittingEnded(self, evt):
        '''
        Callback when fitting has ended. Takes care of cleaning up after
        the fit. Calculates errors on the parameters and updates the grid.
        '''
        if evt.error_message:
            ShowErrorDialog(self.parent, evt.error_message)
            return

        message='Do you want to keep the parameter values from the fit?'
        dlg=wx.MessageDialog(self.parent, message, 'Keep the fit?', wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal()==wx.ID_YES:
            evt = update_parameters(values=evt.values,
                                    new_best=True,
                                    population=evt.population,
                                    max_val=evt.max_val,
                                    min_val=evt.min_val,
                                    fitting=False,
                                    desc='Parameter Update', update_errors=False,
                                    permanent_change=True)
            wx.PostEvent(self.parent, evt)
        else:
            evt = update_parameters(values=evt.start_guess,
                                    new_best=True,
                                    population=evt.population,
                                    max_val=evt.max_val,
                                    min_val=evt.min_val,
                                    fitting=False,
                                    desc='Parameter Update', update_errors=False,
                                    permanent_change=False)
            wx.PostEvent(self.parent, evt)

    def CalcErrorBars(self):
        return self.controller.CalcErrorBars()

    def ProjectEvals(self, parameter):
        return self.controller.ProjectEvals(parameter)

    def ScanParameter(self, parameter, points):
        '''
        Scans one parameter and records its fom value as a function 
        of the parameter value.
        '''
        row=parameter
        model=self.controller.model
        (funcs, vals)=model.get_sim_pars()
        minval=model.parameters.get_data()[row][3]
        maxval=model.parameters.get_data()[row][4]
        parfunc=funcs[model.parameters.get_sim_pos_from_row(row)]
        par_def_val=vals[model.parameters.get_sim_pos_from_row(row)]
        step=(maxval-minval)/points
        par_vals=np.arange(minval, maxval+step, step)
        fom_vals=np.array([])

        par_name=model.parameters.get_data()[row][0]
        dlg=wx.ProgressDialog("Scan Parameter",
                              "Scanning parameter "+par_name,
                              maximum=len(par_vals),
                              parent=self.parent,
                              style=wx.PD_APP_MODAL | wx.PD_ELAPSED_TIME
                                    | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE)
        with CatchModelError(self.parent, 'ScanParameter', 'scan through values') as cme:
            # Start with setting all values
            [f(v) for (f, v) in zip(funcs, vals)]
            for par_val in par_vals:
                parfunc(par_val)
                fom_vals=np.append(fom_vals, model.evaluate_fit_func())
                dlg.Update(len(fom_vals))
        dlg.Destroy()
        # resetting the scanned parameter
        parfunc(par_def_val)
        if cme.successful:
            return par_vals, fom_vals

    def ResetOptimizer(self):
        pass

    def StartFit(self):
        self.controller.StartFit()

    def StopFit(self):
        self.controller.StopFit()

    def ResumeFit(self):
        self.controller.ResumeFit()

    def IsFitted(self):
        return self.controller.IsFitted()

    def AutoSave(self, _event):
        self.controller.save()

    def load_file(self, fname):
        self.controller.load_file(fname)
        solver_classes=[si.__class__ for si in self.solvers.values()]
        loaded_solver=self.controller.optimizer.__class__
        if loaded_solver in solver_classes:
            current_solver=list(self.solvers.keys())[solver_classes.index(loaded_solver)]
        else:
            self.solvers[loaded_solver.__name__]=self.controller.optimizer
            current_solver=loaded_solver.__name__
            self.parent.eh_ex_add_solver_selection(current_solver)
        self.parent.eh_ex_set_solver_selection(current_solver)

    def set_error_bars_level(self, value):
        '''
        Sets the value of increase of the fom used for errorbar calculations
        '''
        if value<1:
            raise ValueError('fom_error_bars_level has to be above 1')
        else:
            self.controller.optimizer.opt.errorbar_level=value

    def set_save_all_evals(self, value):
        '''
        Sets the boolean value to save all evals to file
        '''
        self.controller.optimizer.opt.save_all_evals=bool(value)


def ShowWarningDialog(frame, message):
    dlg=wx.MessageDialog(frame, message,
                         'Warning',
                         wx.OK | wx.ICON_WARNING
                         )
    dlg.ShowModal()
    dlg.Destroy()

def ShowErrorDialog(frame, message, position=''):
    if position!='':
        dlg=wx.MessageDialog(frame, message+'\n'+'Position: '+position,
                             'ERROR',
                             wx.OK | wx.ICON_ERROR
                             )
    else:
        dlg=wx.MessageDialog(frame, message,
                             'ERROR',
                             wx.OK | wx.ICON_ERROR
                             )
    dlg.ShowModal()
    dlg.Destroy()

