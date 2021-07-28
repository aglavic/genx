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
from wx.lib.masked import NumCtrl

from .exception_handling import CatchModelError
from . import diffev, fom_funcs, model_control
from genx.lib.custom_logging import iprint
from .solver_basis import SolverParameterInfo, SolverResultInfo, SolverUpdateInfo, GenxOptimizerCallback


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

        self.controller=model_control.ModelController(diffev.DiffEv())
        self.callback_controller=DelayedCallbacks(parent)
        self.callback_controller.start()
        self.controller.set_callbacks(self.callback_controller)
        self.parent.Bind(EVT_FITTING_ENDED, self.OnFittingEnded)
        self.parent.Bind(EVT_AUTOSAVE, self.AutoSave)

        # Now load the default configuration
        self.ReadConfig()

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

        dlg=SettingsDialog(frame, self.controller, fom_func_name)

        def applyfunc():
            self.WriteConfig()
            self.controller.model.set_fom_func(eval('fom_funcs.'+dlg.get_fom_string()))

        dlg.set_apply_change_func(applyfunc)

        dlg.ShowModal()
        # if dlg.ShowModal() == wx.ID_OK:
        #    pass
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

# ==============================================================================
class SettingsDialog(wx.Dialog):
    def __init__(self, parent, controller: model_control.ModelController, fom_string: str):
        '''
        Configuration optitons for a DiffEv solver.
        '''
        wx.Dialog.__init__(self, parent, -1, 'Optimizer settings')
        # noinspection PyTypeChecker
        self.solver: diffev.DiffEv=controller.optimizer
        self.model=controller.model
        self.apply_change=None

        col_sizer=wx.BoxSizer(wx.HORIZONTAL)
        row_sizer1=wx.BoxSizer(wx.VERTICAL)

        # FOM BOX SIZER
        fom_box=wx.StaticBox(self, -1, "FOM")
        fom_box_sizer=wx.StaticBoxSizer(fom_box, wx.VERTICAL)

        # FOM choice
        fom_sizer=wx.BoxSizer(wx.HORIZONTAL)
        fom_text=wx.StaticText(self, -1, 'Figure of merit ')
        self.fom_choice=wx.Choice(self, -1, choices=fom_funcs.func_names)
        self.fom_choice.SetSelection(fom_funcs.func_names.index(fom_string))
        fom_sizer.Add(fom_text, 0,
                      wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=10)
        fom_sizer.Add(self.fom_choice, 0, wx.EXPAND, border=10)
        fom_box_sizer.Add(fom_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

        cb_sizer=wx.BoxSizer()
        fom_box_sizer.Add(cb_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        # Check box for ignoring nans
        self.fom_ignore_nan_control=wx.CheckBox(self, -1, "Ignore Nan")
        cb_sizer.Add(self.fom_ignore_nan_control, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.fom_ignore_nan_control.SetValue(self.model.solver_parameters.ignore_fom_nan)
        # Check box for ignoring infinities
        self.fom_ignore_inf_control=wx.CheckBox(self, -1, "Ignore +/-Inf")
        cb_sizer.Add(self.fom_ignore_inf_control, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.fom_ignore_inf_control.SetValue(self.model.solver_parameters.ignore_fom_inf)

        # Errorbar level
        errorbar_sizer=wx.BoxSizer(wx.HORIZONTAL)
        errorbar_text=wx.StaticText(self, -1, 'Error bar level ')
        self.errorbar_control=NumCtrl(self, value=
        self.solver.opt.errorbar_level, fractionWidth=2, integerWidth=2)
        errorbar_sizer.Add(errorbar_text, 0, wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=10)
        errorbar_sizer.Add(self.errorbar_control, 1, wx.ALIGN_CENTER_VERTICAL, border=10)
        errorbar_sizer.Add((10, 20), 0, wx.EXPAND)
        fom_box_sizer.Add(errorbar_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

        # X-Range limiting
        cb_sizer=wx.BoxSizer(wx.HORIZONTAL)
        fom_box_sizer.Add(cb_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.limit_fit_range=wx.CheckBox(self, -1, "Limit x-range")
        cb_sizer.Add(self.limit_fit_range, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.limit_fit_range.SetValue(self.solver.opt.limit_fit_range)

        cb_sizer=wx.BoxSizer(wx.HORIZONTAL)
        fom_box_sizer.Add(cb_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        cb_sizer.Add(wx.StaticText(self, -1, 'x_min'))
        self.fit_xmin=wx.SpinCtrlDouble(self, -1, min=0., max=99.0,
                                        initial=self.solver.opt.fit_xmin, inc=0.01)
        cb_sizer.Add(self.fit_xmin)
        cb_sizer.Add(wx.StaticText(self, -1, 'x_max'))
        self.fit_xmax=wx.SpinCtrlDouble(self, -1, min=0., max=99.0,
                                        initial=self.solver.opt.fit_xmax, inc=0.01)
        cb_sizer.Add(self.fit_xmax)

        row_sizer1.Add(fom_box_sizer, 0, wx.EXPAND, 5)

        # Make the Fitting box
        fit_box=wx.StaticBox(self, -1, "Fitting")
        fit_box_sizer=wx.StaticBoxSizer(fit_box, wx.VERTICAL)

        # Make a sizer for the check boxes
        cb_sizer=wx.BoxSizer(wx.HORIZONTAL)
        fit_box_sizer.Add(cb_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        # Check box for start guess
        startguess_control=wx.CheckBox(self, -1, "Start guess")
        cb_sizer.Add(startguess_control, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        startguess_control.SetValue(self.solver.opt.use_start_guess)
        self.startguess_control=startguess_control

        # Check box for using boundaries
        bound_control=wx.CheckBox(self, -1, "Use (Max, Min)")
        cb_sizer.Add(bound_control, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        bound_control.SetValue(self.solver.opt.use_boundaries)
        self.bound_control=bound_control

        # Check box and integer input for autosave
        autosave_sizer=wx.BoxSizer(wx.HORIZONTAL)
        use_autosave_control=wx.CheckBox(self, -1, "Autosave, interval ")
        use_autosave_control.SetValue(self.solver.opt.use_autosave)
        autosave_sc=wx.SpinCtrl(self)
        autosave_sc.SetRange(1, 1000)
        autosave_sc.SetValue(self.solver.opt.autosave_interval)
        autosave_sc.Enable(True)
        autosave_sizer.Add(use_autosave_control, 0,
                           wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=5)
        autosave_sizer.Add(autosave_sc, 0, wx.ALIGN_CENTER_VERTICAL, border=5)
        self.autosave_sc=autosave_sc
        self.use_autosave_control=use_autosave_control
        fit_box_sizer.Add(autosave_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

        # Checkbox for saving all evals
        save_sizer=wx.BoxSizer(wx.HORIZONTAL)
        save_all_control=wx.CheckBox(self, -1, "Save evals, buffer ")
        save_all_control.SetValue(self.solver.opt.save_all_evals)
        buffer_sc=wx.SpinCtrl(self)
        buffer_sc.SetRange(1000, 100000000)
        buffer_sc.SetValue(self.solver.opt.max_log_elements)
        buffer_sc.Enable(True)
        save_sizer.Add(save_all_control, 0,
                       wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=5)
        save_sizer.Add(buffer_sc, 0, wx.ALIGN_CENTER_VERTICAL, border=5)
        self.buffer_sc=buffer_sc
        self.save_all_control=save_all_control
        fit_box_sizer.Add(save_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

        row_sizer1.Add(fit_box_sizer, 1, wx.EXPAND, 5)

        col_sizer.Add(row_sizer1, 1, wx.ALIGN_CENTRE | wx.ALL, 5)

        row_sizer2=wx.BoxSizer(wx.VERTICAL)

        # Make the Diff. Ev. box
        de_box=wx.StaticBox(self, -1, "Diff. Ev.")
        de_box_sizer=wx.StaticBoxSizer(de_box, wx.VERTICAL)
        de_grid=wx.GridBagSizer(2, 2)

        km_sizer=wx.BoxSizer(wx.HORIZONTAL)
        km_text=wx.StaticText(self, -1, 'k_m ')
        self.km_control=NumCtrl(self, value=self.solver.opt.km,
                                fractionWidth=2, integerWidth=2)
        km_sizer.Add(km_text, 0,
                     wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=10)
        km_sizer.Add(self.km_control, 1, wx.ALIGN_CENTER_VERTICAL, border=10)
        km_sizer.Add((10, 20), 0, wx.EXPAND)
        de_grid.Add(km_sizer, (0, 0),
                    flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL,
                    border=5)

        kr_sizer=wx.BoxSizer(wx.HORIZONTAL)
        kr_sizer.Add((10, 20), 0, wx.EXPAND)
        kr_text=wx.StaticText(self, -1, 'k_r ')
        self.kr_control=NumCtrl(self, value=self.solver.opt.kr,
                                fractionWidth=2, integerWidth=2)
        kr_sizer.Add(kr_text, 1,
                     wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=10)
        kr_sizer.Add(self.kr_control, 0, wx.ALIGN_CENTER_VERTICAL, border=10)
        de_grid.Add(kr_sizer, (0, 1),
                    flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL,
                    border=5)

        method_sizer=wx.BoxSizer(wx.HORIZONTAL)
        method_text=wx.StaticText(self, -1, 'Method ')
        mut_schemes=[f.__name__ for f in self.solver.mutation_schemes]
        self.method_choice=wx.Choice(self, -1,
                                     choices=mut_schemes)
        self.method_choice.SetSelection(self.solver.get_create_trial(True))
        method_sizer.Add(method_text, 0,
                         wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=10)
        method_sizer.Add(self.method_choice, 0, wx.ALIGN_CENTER_VERTICAL, border=10)
        de_grid.Add(method_sizer, (1, 0), (1, 2),
                    flag=wx.ALIGN_CENTER | wx.ALIGN_CENTER_VERTICAL | wx.EXPAND,
                    border=5)

        de_box_sizer.Add(de_grid, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        row_sizer2.Add(de_box_sizer, 0, wx.EXPAND, 5)

        # Make the Population box
        pop_box=wx.StaticBox(self, -1, "Population size")
        pop_box_sizer=wx.StaticBoxSizer(pop_box, wx.VERTICAL)
        pop_grid=wx.FlexGridSizer(0, 2, 0, 0)

        multsize_radio=wx.RadioButton(self, -1, " Relative size ",
                                      style=wx.RB_GROUP)
        fixedsize_radio=wx.RadioButton(self, -1, " Fixed size ")

        multsize_sc=wx.SpinCtrl(self)
        multsize_sc.SetRange(1, 1000)
        multsize_sc.SetValue(self.solver.opt.pop_mult)
        multsize_sc.Enable(self.solver.opt.use_pop_mult)
        fixedsize_sc=wx.SpinCtrl(self)
        fixedsize_sc.SetRange(1, 1000)
        fixedsize_sc.SetValue(self.solver.opt.pop_size)
        fixedsize_sc.Enable(not self.solver.opt.use_pop_mult)

        self.pop_multsize_radio=multsize_radio
        self.pop_fixedsize_radio=fixedsize_radio
        self.pop_multsize_sc=multsize_sc
        self.pop_fixedsize_sc=fixedsize_sc

        pop_grid.Add(multsize_radio, 0,
                     wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        pop_grid.Add(multsize_sc, 0,
                     wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        pop_grid.Add(fixedsize_radio, 0,
                     wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        pop_grid.Add(fixedsize_sc, 0,
                     wx.ALIGN_RIGHT | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        pop_box_sizer.Add(pop_grid, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        row_sizer2.Add(pop_box_sizer, 1, wx.EXPAND, 5)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_pop_select, multsize_radio)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_pop_select, fixedsize_radio)
        multsize_radio.SetValue(self.solver.opt.use_pop_mult)
        fixedsize_radio.SetValue(not self.solver.opt.use_pop_mult)

        # Make the Generation box
        gen_box=wx.StaticBox(self, -1, "Max Generations")
        gen_box_sizer=wx.StaticBoxSizer(gen_box, wx.VERTICAL)
        gen_grid=wx.FlexGridSizer(0, 2, 0, 0)

        gen_multsize_radio=wx.RadioButton(self, -1, " Relative size ",
                                          style=wx.RB_GROUP)
        gen_fixedsize_radio=wx.RadioButton(self, -1, " Fixed size ")

        gen_multsize_sc=wx.SpinCtrl(self)
        gen_multsize_sc.SetRange(1, 10000)
        gen_multsize_sc.SetValue(self.solver.opt.max_generation_mult)
        gen_multsize_sc.Enable(not self.solver.opt.use_max_generations)
        gen_fixedsize_sc=wx.SpinCtrl(self)
        gen_fixedsize_sc.SetRange(1, 10000)
        gen_fixedsize_sc.SetValue(self.solver.opt.max_generations)
        gen_fixedsize_sc.Enable(self.solver.opt.use_max_generations)

        self.gen_multsize_radio=gen_multsize_radio
        self.gen_fixedsize_radio=gen_fixedsize_radio
        self.gen_multsize_sc=gen_multsize_sc
        self.gen_fixedsize_sc=gen_fixedsize_sc

        gen_grid.Add(gen_multsize_radio, 0,
                     wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        gen_grid.Add(gen_multsize_sc, 0,
                     wx.ALIGN_CENTRE | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        gen_grid.Add(gen_fixedsize_radio, 0,
                     wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        gen_grid.Add(gen_fixedsize_sc, 0,
                     wx.ALIGN_CENTRE | wx.LEFT | wx.RIGHT | wx.TOP, 5)

        gen_box_sizer.Add(gen_grid, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        row_sizer2.Add(gen_box_sizer, 1, wx.EXPAND, 5)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_gen_select, gen_multsize_radio)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_gen_select, gen_fixedsize_radio)
        gen_fixedsize_radio.SetValue(self.solver.opt.use_max_generations)
        gen_multsize_radio.SetValue(not self.solver.opt.use_max_generations)

        ##
        # Make the parallel fitting box
        parallel_box=wx.StaticBox(self, -1, "Parallel processing")
        parallel_box_sizer=wx.StaticBoxSizer(parallel_box, wx.VERTICAL)

        use_parallel_control=wx.CheckBox(self, -1, "Parallel fitting")
        use_parallel_control.SetValue(self.solver.opt.use_parallel_processing)
        use_parallel_control.Enable(diffev.__parallel_loaded__)
        self.use_parallel_control=use_parallel_control
        parallel_box_sizer.Add(use_parallel_control, 1, wx.EXPAND, 5)

        processes_sc=wx.SpinCtrl(self, size=(80, -1))
        processes_sc.SetRange(1, 100)
        processes_sc.SetValue(self.solver.opt.parallel_processes)
        processes_sc.Enable(diffev.__parallel_loaded__)
        chunk_size_sc=wx.SpinCtrl(self, size=(80, -1))
        chunk_size_sc.SetRange(1, 100)
        chunk_size_sc.SetValue(self.solver.opt.parallel_chunksize)
        chunk_size_sc.Enable(diffev.__parallel_loaded__)
        self.processes_sc=processes_sc
        self.chunk_size_sc=chunk_size_sc
        parallel_sizer=wx.BoxSizer(wx.HORIZONTAL)
        p_text=wx.StaticText(self, -1, '# Processes')
        parallel_sizer.Add(p_text, 0,
                           wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=10)
        parallel_sizer.Add((10, 20), 1, wx.EXPAND)
        parallel_sizer.Add(processes_sc, 0, wx.ALIGN_CENTER_VERTICAL, border=10)
        parallel_box_sizer.Add(parallel_sizer, 1, wx.EXPAND, 10)
        parallel_sizer=wx.BoxSizer(wx.HORIZONTAL)
        p_text=wx.StaticText(self, -1, ' Chunk size ')
        parallel_sizer.Add(p_text, 0,
                           wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=10)
        parallel_sizer.Add((10, 20), 1, wx.EXPAND)
        parallel_sizer.Add(chunk_size_sc, 0, wx.ALIGN_CENTER_VERTICAL, border=10)

        parallel_box_sizer.Add(parallel_sizer, 1, wx.EXPAND, 10)
        row_sizer2.Add(parallel_box_sizer, 1, wx.EXPAND, 5)

        col_sizer.Add(row_sizer2, 1, wx.ALIGN_CENTRE | wx.ALL, 5)
        ##

        # Add the Dialog buttons
        button_sizer=wx.StdDialogButtonSizer()
        okay_button=wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        apply_button=wx.Button(self, wx.ID_APPLY)
        apply_button.SetDefault()
        button_sizer.AddButton(apply_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        # Add some event handlers
        self.Bind(wx.EVT_BUTTON, self.on_apply_change, okay_button)
        self.Bind(wx.EVT_BUTTON, self.on_apply_change, apply_button)

        sizer=wx.BoxSizer(wx.VERTICAL)
        sizer.Add(col_sizer, 1, wx.GROW, 20)
        # sizer.Add(col_sizer, 1, wx.GROW|wx.ALL|wx.EXPAND, 20)
        line=wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.TOP, 20)

        sizer.Add(button_sizer, 0,
                  flag=wx.ALIGN_RIGHT, border=20)
        self.SetSizer(sizer)

        sizer.Fit(self)
        self.Layout()

    def on_pop_select(self, event):
        '''
        callback for selction of a radio button in the population group
        '''
        radio_selected=event.GetEventObject()

        if radio_selected is self.pop_fixedsize_radio:
            self.pop_fixedsize_sc.Enable(True)
            self.pop_multsize_sc.Enable(False)
        else:
            self.pop_fixedsize_sc.Enable(False)
            self.pop_multsize_sc.Enable(True)

    def on_gen_select(self, event):
        '''
        callback for selection of a radio button in the Generation group
        '''
        radio_selected=event.GetEventObject()

        if radio_selected is self.gen_fixedsize_radio:
            self.gen_fixedsize_sc.Enable(True)
            self.gen_multsize_sc.Enable(False)
        else:
            self.gen_fixedsize_sc.Enable(False)
            self.gen_multsize_sc.Enable(True)

    def get_fom_string(self):
        return self.fom_choice.GetStringSelection()

    def on_apply_change(self, event):
        model=self.model

        self.solver.opt.kr=self.kr_control.GetValue()
        self.solver.opt.km=self.km_control.GetValue()
        self.solver.opt.max_generation_mult=self.gen_multsize_sc.GetValue()
        self.solver.opt.max_generations=self.gen_fixedsize_sc.GetValue()
        self.solver.opt.pop_mult=self.pop_multsize_sc.GetValue()
        self.solver.opt.pop_size=self.pop_fixedsize_sc.GetValue()
        self.solver.opt.use_max_generations=self.gen_fixedsize_radio.GetValue()
        self.solver.opt.use_pop_mult=self.pop_multsize_radio.GetValue()
        self.solver.opt.use_start_guess=self.startguess_control.GetValue()
        self.solver.opt.use_boundaries=self.bound_control.GetValue()
        self.solver.set_create_trial(self.method_choice.GetStringSelection())
        self.solver.opt.use_parallel_processing=self.use_parallel_control.GetValue()
        self.solver.opt.processes=self.processes_sc.GetValue()
        self.solver.opt.chunksize=self.chunk_size_sc.GetValue()
        model.set_fom_ignore_inf(self.fom_ignore_inf_control.GetValue())
        model.set_fom_ignore_nan(self.fom_ignore_nan_control.GetValue())
        self.solver.opt.errorbar_level=self.errorbar_control.GetValue()
        self.solver.opt.use_autosave=self.use_autosave_control.GetValue()
        self.solver.opt.autosave_interval=self.autosave_sc.GetValue()
        self.solver.opt.save_all_evals=self.save_all_control.GetValue()
        self.solver.opt.max_log=self.buffer_sc.GetValue()
        self.solver.opt.limit_fit_range=self.limit_fit_range.GetValue()
        self.solver.opt.fit_xmin=self.fit_xmin.GetValue()
        self.solver.opt.fit_xmax=self.fit_xmax.GetValue()

        model.opt.limit_fit_range, model.opt.fit_xmin, model.opt.fit_xmax=(
            self.solver.opt.limit_fit_range,
            self.solver.opt.fit_xmin,
            self.solver.opt.fit_xmax)

        if self.apply_change:
            self.apply_change()

        event.Skip()

    def set_apply_change_func(self, func):
        '''
        Set the apply_change function. Is executed when the apply or ok button
        is clicked.
        '''
        self.apply_change=func

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

