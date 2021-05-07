'''
Controller class for the differnetial evolution class diffev
Takes care of stopping and starting - output to the gui as well
as some input from dialog boxes.
Programmer Matts Bjorck
Last Changed 2009 05 12
'''
import wx, io, traceback
import wx.lib.newevent
from wx.lib.masked import NumCtrl

from . import diffev, fom_funcs
from . import filehandling as io
from .gui_logging import iprint
import numpy as np

# ==============================================================================
class SolverController:
    '''
    Class to take care of the GUI - solver interaction.
    Implements dialogboxes for setting parameters and controls
    for the solver routine. This is where the application specific
    code are used i.e. interfacing the optimization code to the GUI.
    '''

    def __init__(self, parent, config=None):
        # Create the optimizer we are using. In this case the standard
        # Differential evolution optimizer.
        self.optimizer=diffev.DiffEv()
        # Store the parent we need this to bind the different components to
        # the optimization algorithm.
        self.parent=parent
        self.config=config

        # Just storage of the starting values for the paramters before
        # the fit is started so the user can go back to the start values
        self.start_parameter_values=None
        # The level used for error bar calculations
        self.fom_error_bars_level=1.05

        # Setup the output functions.
        self.optimizer.set_text_output_func(self.TextOutput)
        self.optimizer.set_plot_output_func(self.PlotOutput)
        self.optimizer.set_parameter_output_func(self.ParameterOutput)
        self.optimizer.set_fitting_ended_func(self.FittingEnded)
        self.optimizer.set_autosave_func(self.AutoSave)

        self.parent.Bind(EVT_FITTING_ENDED, self.OnFittingEnded)

        # Now load the default configuration
        self.ReadConfig()

    def ReadConfig(self):
        '''ReadConfig(self) --> None
        
        Reads the parameter that should be read from the config file.
        And set the parameters in both the optimizer and this class.
        '''
        error_bars_level, save_all_evals=io.load_opt_config(self.optimizer, self.config)
        self.set_error_bars_level(error_bars_level)
        self.set_save_all_evals(save_all_evals)

    def WriteConfig(self):
        ''' WriteConfig(self) --> None
        
        Writes the current configuration of the solver to file.
        '''
        io.save_opt_config(self.optimizer, self.config, self.fom_error_bars_level, self.save_all_evals)

    def ParametersDialog(self, frame):
        '''ParametersDialog(self, frame) --> None
        
        Shows the Parameters dialog box to set the parameters for the solver.
        '''
        # Update the configuration if a model has been loaded after
        # the object have been created..
        self.ReadConfig()
        fom_func_name=self.parent.model.fom_func.__name__
        if not fom_func_name in fom_funcs.func_names:
            ShowWarningDialog(self.parent, 'The loaded fom function, ' \
                              +fom_func_name+', does not exist '+ \
                              'in the local fom_funcs file. The fom fucntion has been'+
                              ' temporary added to the list of availabe fom functions')
            fom_funcs.func_names.append(fom_func_name)
            exectext='fom_funcs.'+fom_func_name+ \
                     ' = self.parent.model.fom_func'
            exec(exectext, locals(), globals())

        dlg=SettingsDialog(frame, self.optimizer, self, fom_func_name)

        def applyfunc(object):
            self.WriteConfig()
            self.parent.model.set_fom_func(eval('fom_funcs.'+object.get_fom_string()))

        dlg.set_apply_change_func(applyfunc)

        dlg.ShowModal()
        # if dlg.ShowModal() == wx.ID_OK:
        #    pass
        dlg.Destroy()

    def TextOutput(self, text):
        '''TextOutput(self, text) --> None
        Function to present the output from the optimizer to the user. 
        Takes a string as input.
        '''
        # self.parent.main_frame_statusbar.SetStatusText(text, 0)
        evt=update_text(text=text)
        wx.PostEvent(self.parent, evt)
        # wx.CallAfter(wx.PostEvent, self.parent, evt)

    def PlotOutput(self, solver):
        ''' PlotOutput(self, solver) --> None
        Solver to present the graphical output from the optimizer to the 
        user. Takes the solver as input argument and picks out the 
        variables to show in the GUI.
        '''
        # print 'sending event plotting'
        # _post_solver_event(self.parent, solver, desc = 'Fitting update')
        evt=update_plot(model=solver.get_model(),
                        fom_log=solver.get_fom_log(), update_fit=solver.new_best,
                        desc='Fitting update')
        wx.PostEvent(self.parent, evt)
        # wx.CallAfter(wx.PostEvent, self.parent, evt)
        # Hard code the events for the plugins so that they can be run syncrously.
        # This is important since the Refelctevity model, for example, relies on the
        # current state of the model.
        try:
            self.parent.plugin_control.OnFittingUpdate(evt)
            # wx.CallAfter(self.parent.plugin_control.OnFittingUpdate, evt)
            # pass
        except Exception as e:
            iprint('Error in plot output:\n'+repr(e))

    def ParameterOutput(self, solver):
        '''ParameterOutput(self, solver) --> none
        
        Function to send an update event to update windows that displays
        the parameters to update the values. 
        Takes the solver as input argument and picks out the variables to 
        show in the GUI.
        '''
        evt=update_parameters(values=solver.best_vec.copy(),
                              new_best=solver.new_best,
                              population=solver.pop_vec,
                              max_val=solver.par_max,
                              min_val=solver.par_min,
                              fitting=True,
                              desc='Parameter Update', update_errors=False,
                              permanent_change=False)
        wx.PostEvent(self.parent, evt)
        # wx.CallAfter(wx.PostEvent, self.parent, evt)

    def ModelLoaded(self):
        '''ModelLoaded(self) --> None
        
        Function that takes care of resetting everything when a model has
        been loaded.
        '''
        evt=update_plot(model=self.optimizer.get_model(),
                        fom_log=self.optimizer.get_fom_log(), update_fit=False,
                        desc='Model loaded')
        wx.PostEvent(self.parent, evt)

        # Update the parameter plot ... 
        if self.optimizer.setup_ok:
            # remeber to add a check 
            solver=self.optimizer
            try:
                evt=update_parameters(values=solver.best_vec.copy(),
                                      new_best=False,
                                      population=solver.pop_vec,
                                      max_val=solver.par_max,
                                      min_val=solver.par_min,
                                      fitting=True,
                                      desc='Parameter Update', update_errors=False,
                                      permanent_change=False)
            except:
                iprint('Could not create data for paraemters')
            else:
                wx.PostEvent(self.parent, evt)

    def AutoSave(self):
        '''DoAutoSave(self) --> None
        
        Function that conducts an autosave of the model.
        '''
        io.save_file(self.parent.model.get_filename(), self.parent.model, self.optimizer, self.config)
        # print 'AutoSaved!'

    def FittingEnded(self, solver):
        '''FittingEnded(self, solver) --> None
        
        function used to post an event when the fitting has ended.
        This must be done since it is not htread safe otherwise. Same GUI in
        two threads when dialogs are run. dangerous...
        '''
        evt=fitting_ended(solver=solver, desc='Fitting Ended')
        wx.PostEvent(self.parent, evt)

    def OnFittingEnded(self, evt):
        '''OnFittingEnded(self, solver) --> None
        
        Callback when fitting has ended. Takes care of cleaning up after
        the fit. Calculates errors on the parameters and updates the grid.
        '''
        solver=evt.solver
        if solver.error:
            ShowErrorDialog(self.parent, solver.error)
            return

        message='Do you want to keep the parameter values from '+ \
                'the fit?'
        dlg=wx.MessageDialog(self.parent, message, 'Keep the fit?',
                             wx.YES_NO | wx.ICON_QUESTION)
        if dlg.ShowModal()==wx.ID_YES:
            evt=update_parameters(values=solver.best_vec.copy(),
                                  desc='Parameter Update', new_best=True,
                                  update_errors=False, fitting=False,
                                  permanent_change=True)
            wx.PostEvent(self.parent, evt)
        else:
            # print 'Resetting the values in the grid to ',\
            #    self.start_parameter_values
            evt=update_parameters(values=solver.start_guess,
                                  desc='Parameter Update', new_best=True,
                                  update_errors=False, fitting=False,
                                  permanent_change=False)
            wx.PostEvent(self.parent, evt)

    def CalcErrorBars(self):
        '''
        Method that calculates the errorbars for the fit that has been
        done. Note that the fit has to been conducted before this is run.
        '''
        if len(self.optimizer.fom_evals)==0:
            raise ErrorBarError('Can not find any stored evaluations of the model in the optimizer.\n'
                                'Run a fit before calculating the errorbars.')
        if self.optimizer.start_guess is not None and not self.optimizer.running:
            n_elements=len(self.optimizer.start_guess)
            # print 'Number of elemets to calc errobars for ', n_elements
            error_values=[]
            dlg=wx.ProgressDialog("Calculating", "Error bars are calculated ...",
                                  maximum=n_elements, parent=self.parent,
                                  style=wx.PD_AUTO_HIDE)
            for index in range(n_elements):
                # calculate the error
                # TODO: Check the error bar buisness again and how to treat 
                # Chi2 
                try:
                    (error_low, error_high)=self.optimizer.calc_error_bar(index, self.fom_error_bars_level)
                except diffev.ErrorBarError as e:
                    ShowWarningDialog(self.parent, str(e))
                    break
                error_str='(%.3e, %.3e)'%(error_low, error_high)
                error_values.append(error_str)
                dlg.Update(index+1)

            dlg.Destroy()
            return error_values
        else:
            raise ErrorBarError()

    def ProjectEvals(self, parameter):
        ''' ProjectEvals(self, parameter) --> prameter, fomvals
        
        Projects the parameter number parameter on one axis and returns
        the fomvals.
        '''
        model=self.parent.model
        row=model.parameters.get_pos_from_row(parameter)
        if self.optimizer.start_guess is not None and not self.optimizer.running:
            return self.optimizer.par_evals[:, row], \
                   self.optimizer.fom_evals[:]
        else:
            raise ErrorBarError()

    def ScanParameter(self, parameter, points):
        '''ScanParameter(self, parameter, points) 
            --> par_vals, fom_vals
        
        Scans one parameter and records its fom value as a function 
        of the parameter value.
        '''
        row=parameter
        model=self.parent.model
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
        try:
            # Start with setting all values
            [f(v) for (f, v) in zip(funcs, vals)]
            for par_val in par_vals:
                parfunc(par_val)
                fom_vals=np.append(fom_vals, model.evaluate_fit_func())
                dlg.Update(len(fom_vals))
        except Exception as e:
            dlg.Destroy()
            outp=io.StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            ShowWarningDialog(self.parent, 'Error while evaluatating the'+ \
                              ' simulation and fom. Please check so it is possible to simulate'+ \
                              ' your model. Detailed output below: \n\n'+val)
        else:
            dlg.Destroy()
        # resetting the scanned parameter
        parfunc(par_def_val)
        return par_vals, fom_vals

    def ResetOptimizer(self):
        '''ResetOptimizer(self) --> None
        
        Resets the optimizer - clears the memory and special flags.
        '''
        self.start_parameter_values=None

    def StartFit(self):
        ''' StartFit(self) --> None
        Function to start running the fit
        '''
        # Make sure that the config of the solver is updated..
        self.ReadConfig()
        model=self.parent.model
        # Reset all the errorbars
        model.parameters.clear_error_pars()
        # self.start_parameter_values = model.get_fit_values()
        self.optimizer.start_fit(model)
        # print 'Optimizer starting'

    def StopFit(self):
        ''' StopFit(self) --> None
        Function to stop a running fit
        '''
        self.optimizer.stop_fit()

    def ResumeFit(self):
        ''' ResumeFit(self) --> None
        
        Function to resume the fitting after it has been stopped
        '''
        # Make sure the settings are updated..
        self.ReadConfig()
        model=self.parent.model
        # Remove all previous erros ...

        self.optimizer.resume_fit(model)

    def IsFitted(self):
        '''IsFitted(self) --> bool
        
        Returns true if a fit has been started otherwise False
        '''
        return len(self.optimizer.start_guess)>0

    def set_error_bars_level(self, value):
        '''set_error_bars_level(value) --> None
        
        Sets the value of increase of the fom used for errorbar calculations
        '''
        if value<1:
            raise ValueError('fom_error_bars_level has to be above 1')
        else:
            self.fom_error_bars_level=value

    def set_save_all_evals(self, value):
        '''Sets the boolean value to save all evals to file
        '''
        self.save_all_evals=bool(value)

# ==============================================================================
class SettingsDialog(wx.Dialog):
    def __init__(self, parent, solver, solvergui, fom_string):
        '''__init__(self, parent, solver, fom_string, mut_schemes,\
                    current_mut_scheme)
                    
        parent - parent window, solver - the solver (Diffev alg.)
        fom_string - the fom function string
        '''
        wx.Dialog.__init__(self, parent, -1, 'Optimizer settings')
        # self.SetAutoLayout(True)
        self.solver=solver
        self.solvergui=solvergui
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
        self.fom_ignore_nan_control.SetValue(self.solvergui.parent.model.fom_ignore_nan)
        # Check box for ignoring infs
        self.fom_ignore_inf_control=wx.CheckBox(self, -1, "Ignore +/-Inf")
        cb_sizer.Add(self.fom_ignore_inf_control, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.fom_ignore_inf_control.SetValue(self.solvergui.parent.model.fom_ignore_inf)
        self.fom_ignore_inf_control.SetValue(self.solvergui.parent.model.fom_ignore_inf)

        # Errorbar level 
        errorbar_sizer=wx.BoxSizer(wx.HORIZONTAL)
        errorbar_text=wx.StaticText(self, -1, 'Error bar level ')
        self.errorbar_control=NumCtrl(self, value=
        self.solvergui.fom_error_bars_level,
                                      fractionWidth=2, integerWidth=2)
        errorbar_sizer.Add(errorbar_text, 0,
                           wx.ALIGN_LEFT | wx.ALIGN_CENTER_VERTICAL, border=10)
        errorbar_sizer.Add(self.errorbar_control, 1, wx.ALIGN_CENTER_VERTICAL, border=10)
        errorbar_sizer.Add((10, 20), 0, wx.EXPAND)
        fom_box_sizer.Add(errorbar_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)

        # X-Range limiting
        cb_sizer=wx.BoxSizer(wx.HORIZONTAL)
        fom_box_sizer.Add(cb_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.limit_fit_range=wx.CheckBox(self, -1, "Limit x-range")
        cb_sizer.Add(self.limit_fit_range, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.limit_fit_range.SetValue(self.solver.limit_fit_range)

        cb_sizer=wx.BoxSizer(wx.HORIZONTAL)
        fom_box_sizer.Add(cb_sizer, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        cb_sizer.Add(wx.StaticText(self, -1, 'x_min'))
        self.fit_xmin=wx.SpinCtrlDouble(self, -1, min=0., max=99.0,
                                        initial=self.solver.fit_xmin, inc=0.01)
        cb_sizer.Add(self.fit_xmin)
        cb_sizer.Add(wx.StaticText(self, -1, 'x_max'))
        self.fit_xmax=wx.SpinCtrlDouble(self, -1, min=0., max=99.0,
                                        initial=self.solver.fit_xmax, inc=0.01)
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
        startguess_control.SetValue(self.solver.use_start_guess)
        self.startguess_control=startguess_control

        # Check box for using boundaries
        bound_control=wx.CheckBox(self, -1, "Use (Max, Min)")
        cb_sizer.Add(bound_control, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        bound_control.SetValue(self.solver.use_boundaries)
        self.bound_control=bound_control

        # Check box and integer input for autosave
        autosave_sizer=wx.BoxSizer(wx.HORIZONTAL)
        use_autosave_control=wx.CheckBox(self, -1, "Autosave, interval ")
        use_autosave_control.SetValue(self.solver.use_autosave)
        autosave_sc=wx.SpinCtrl(self)
        autosave_sc.SetRange(1, 1000)
        autosave_sc.SetValue(self.solver.autosave_interval)
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
        save_all_control.SetValue(self.solvergui.save_all_evals)
        buffer_sc=wx.SpinCtrl(self)
        buffer_sc.SetRange(1000, 100000000)
        buffer_sc.SetValue(self.solver.max_log)
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
        self.km_control=NumCtrl(self, value=self.solver.km,
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
        self.kr_control=NumCtrl(self, value=self.solver.kr,
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
        multsize_sc.SetValue(self.solver.pop_mult)
        multsize_sc.Enable(self.solver.use_pop_mult)
        fixedsize_sc=wx.SpinCtrl(self)
        fixedsize_sc.SetRange(1, 1000)
        fixedsize_sc.SetValue(self.solver.pop_size)
        fixedsize_sc.Enable(not self.solver.use_pop_mult)

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
        multsize_radio.SetValue(self.solver.use_pop_mult)
        fixedsize_radio.SetValue(not self.solver.use_pop_mult)

        # Make the Generation box
        gen_box=wx.StaticBox(self, -1, "Max Generations")
        gen_box_sizer=wx.StaticBoxSizer(gen_box, wx.VERTICAL)
        gen_grid=wx.FlexGridSizer(0, 2, 0, 0)

        gen_multsize_radio=wx.RadioButton(self, -1, " Relative size ",
                                          style=wx.RB_GROUP)
        gen_fixedsize_radio=wx.RadioButton(self, -1, " Fixed size ")

        gen_multsize_sc=wx.SpinCtrl(self)
        gen_multsize_sc.SetRange(1, 10000)
        gen_multsize_sc.SetValue(self.solver.max_generation_mult)
        gen_multsize_sc.Enable(not self.solver.use_max_generations)
        gen_fixedsize_sc=wx.SpinCtrl(self)
        gen_fixedsize_sc.SetRange(1, 10000)
        gen_fixedsize_sc.SetValue(self.solver.max_generations)
        gen_fixedsize_sc.Enable(self.solver.use_max_generations)

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
        gen_fixedsize_radio.SetValue(self.solver.use_max_generations)
        gen_multsize_radio.SetValue(not self.solver.use_max_generations)

        ##
        # Make the parallel fitting box
        parallel_box=wx.StaticBox(self, -1, "Parallel processing")
        parallel_box_sizer=wx.StaticBoxSizer(parallel_box, wx.VERTICAL)

        use_parallel_control=wx.CheckBox(self, -1, "Parallel fitting")
        use_parallel_control.SetValue(self.solver.use_parallel_processing)
        use_parallel_control.Enable(diffev.__parallel_loaded__)
        self.use_parallel_control=use_parallel_control
        parallel_box_sizer.Add(use_parallel_control, 1, wx.EXPAND, 5)

        processes_sc=wx.SpinCtrl(self, size=(80, -1))
        processes_sc.SetRange(1, 100)
        processes_sc.SetValue(self.solver.processes)
        processes_sc.Enable(diffev.__parallel_loaded__)
        chunk_size_sc=wx.SpinCtrl(self, size=(80, -1))
        chunk_size_sc.SetRange(1, 100)
        chunk_size_sc.SetValue(self.solver.chunksize)
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
        # Add some eventhandlers
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
        '''on_pop_select(self, event) --> None
        
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
        '''on_pop_select(self, event) --> None
        
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
        self.solver.kr=self.kr_control.GetValue()
        self.solver.km=self.km_control.GetValue()
        self.solver.max_generation_mult=self.gen_multsize_sc.GetValue()
        self.solver.max_generations=self.gen_fixedsize_sc.GetValue()
        self.solver.pop_mult=self.pop_multsize_sc.GetValue()
        self.solver.pop_size=self.pop_fixedsize_sc.GetValue()
        self.solver.use_max_generations=self.gen_fixedsize_radio.GetValue()
        self.solver.use_pop_mult=self.pop_multsize_radio.GetValue()
        self.solver.use_start_guess=self.startguess_control.GetValue()
        self.solver.use_boundaries=self.bound_control.GetValue()
        self.solver.set_create_trial(self.method_choice.GetStringSelection())
        self.solver.use_parallel_processing=self.use_parallel_control.GetValue()
        self.solver.processes=self.processes_sc.GetValue()
        self.solver.chunksize=self.chunk_size_sc.GetValue()
        self.solvergui.parent.model.set_fom_ignore_inf(self.fom_ignore_inf_control.GetValue())
        self.solvergui.parent.model.set_fom_ignore_nan(self.fom_ignore_nan_control.GetValue())
        self.solvergui.fom_error_bars_level=self.errorbar_control.GetValue()
        self.solver.use_autosave=self.use_autosave_control.GetValue()
        self.solver.autosave_interval=self.autosave_sc.GetValue()
        self.solvergui.save_all_evals=self.save_all_control.GetValue()
        self.solver.max_log=self.buffer_sc.GetValue()
        self.solver.limit_fit_range=self.limit_fit_range.GetValue()
        self.solver.fit_xmin=self.fit_xmin.GetValue()
        self.solver.fit_xmax=self.fit_xmax.GetValue()

        model=self.solvergui.parent.model
        model.limit_fit_range, model.fit_xmin, model.fit_xmax=(
            self.solver.limit_fit_range,
            self.solver.fit_xmin,
            self.solver.fit_xmax)

        if self.apply_change:
            self.apply_change(self)

        event.Skip()

    def set_apply_change_func(self, func):
        '''set_apply_change_func(self, func) --> None
        
        Set the apply_change function. Is executed when the apply or ok button
        is clicked.
        '''
        self.apply_change=func

# ==============================================================================
# Custom events needed for updating and message parsing between the different
# modules.

(update_plot, EVT_UPDATE_PLOT)=wx.lib.newevent.NewEvent()
(update_text, EVT_SOLVER_UPDATE_TEXT)=wx.lib.newevent.NewEvent()
(update_parameters, EVT_UPDATE_PARAMETERS)=wx.lib.newevent.NewEvent()
(fitting_ended, EVT_FITTING_ENDED)=wx.lib.newevent.NewEvent()

# ==============================================================================
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

class GenericError(Exception):
    ''' Just a empty class used for inheritance. Only useful
    to check if the errors are originating from the model library.
    All these errors are controllable. If they not originate from
    this class something has passed trough and that should be impossible '''
    pass

class ErrorBarError(GenericError):
    def __init__(self, error_message=None):
        """Error class for the fom evaluation

        :param error_message: Error message that explains the error, string.
        :return:
        """
        if error_message is None:
            self.error_message='Could not evaluate the error bars. A fit has to be run before they can be calculated'
        else:
            self.error_message=error_message

    def __str__(self):
        return self.error_message
