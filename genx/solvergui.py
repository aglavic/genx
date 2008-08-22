'''
Controller class for the differnetial evolution class diffev
Takes care of stopping and starting - output to the gui as well
as some input from dialog boxes.
Programmer Matts Bjorck
Last Changed 2008 09 03
'''
import wx, StringIO, traceback
import  wx.lib.newevent
from wx.lib.masked import NumCtrl

import diffev, fom_funcs
import io
import numpy as np

# TODO: Include mutation schemes

#==============================================================================
class SolverController:
    '''
    Class to take care of the GUI - solver interaction.
    Implements dialogboxes for setting parameters and controls
    for the solver routine. This is where the application specific
    code are used i.e. interfacing the optimization code to the GUI.
    '''
    
    def __init__(self, parent, config = None):
        # Create the optimizer we are using. In this case the standard
        # Differential evolution optimizer.
        self.optimizer = diffev.DiffEv()
        # Store the parent we need this to bind the different components to
        # the optimization algorithm.
        self.parent = parent
        self.config = config
        
        # Just storage of the starting values for the paramters before
        # the fit is started so the user can go back to the start values
        self.start_parameter_values = None
        # The level used for error bar calculations
        self.fom_error_bars_level = 1.05
        
        # Setup the output functions.
        self.optimizer.set_text_output_func(self.TextOutput)
        self.optimizer.set_plot_output_func(self.PlotOutput)
        self.optimizer.set_parameter_output_func(self.ParameterOutput)
        self.optimizer.set_fitting_ended_func(self.FittingEnded)
        
        self.parent.Bind(EVT_FITTING_ENDED, self.OnFittingEnded)
        
        # Now load the default configuration
        self.ReadConfig()
        
        
    def ReadConfig(self):
        '''ReadConfig(self) --> None
        
        Reads the parameter that should be read from the config file.
        And set the parameters in both the optimizer and this class.
        '''
        # Define all the options we want to set
        options_float = ['km', 'kr', 'pop mult', 'pop size',\
                         'max generations', 'max generation mult',\
                         'sleep time']
        setfunctions_float = [self.optimizer.set_km, self.optimizer.set_kr,
                          self.optimizer.set_pop_mult,\
                          self.optimizer.set_pop_size,\
                         self.optimizer.set_max_generations,\
                         self.optimizer.set_max_generation_mult,\
                         self.optimizer.set_sleep_time]

        options_bool = ['use pop mult', 'use max generations',\
                        'use start guess']
        setfunctions_bool = [ self.optimizer.set_use_pop_mult,\
                            self.optimizer.set_use_max_generations,\
                            self.optimizer.set_use_start_guess]
        
        # Make sure that the config is set
        if self.config:
            # Start witht the float values
            for index in range(len(options_float)):
                try:
                    val = self.config.get_float('solver', options_float[index])
                except io.OptionError, e:
                    print 'Could not locate option solver.' +\
                            options_float[index]
                else:
                    setfunctions_float[index](val)
            
            # Then the bool flags
            for index in range(len(options_bool)):
                try:
                    val = self.config.get_boolean('solver',\
                            options_bool[index])
                except io.OptionError, e:
                    print 'Could not read option solver.' +\
                            options_bool[index]
                else:
                    setfunctions_bool[index](val)

    def WriteConfig(self):
        ''' WriteConfig(self) --> None
        
        Writes the current configuration of the solver to file.
        '''
        
        # Define all the options we want to set
        options_float = ['km', 'kr', 'pop mult', 'pop size',\
                         'max generations', 'max generation mult',\
                         'sleep time']
        set_float = [self.optimizer.km, self.optimizer.kr,
                          self.optimizer.pop_mult,\
                          self.optimizer.pop_size,\
                         self.optimizer.max_generations,\
                         self.optimizer.max_generation_mult,\
                         self.optimizer.sleep_time]

        options_bool = ['use pop mult', 'use max generations',\
                        'use start guess']
        set_bool = [ self.optimizer.use_pop_mult,\
                            self.optimizer.use_max_generations,\
                            self.optimizer.use_start_guess]
        
        # Make sure that the config is set
        if self.config:
            # Start witht the float values
            for index in range(len(options_float)):
                try:
                    val = self.config.set('solver', options_float[index],\
                            set_float[index])
                except io.OptionError, e:
                    print 'Could not locate save solver.' +\
                            options_float[index]
            
            # Then the bool flags
            for index in range(len(options_bool)):
                try:
                    val = self.config.set('solver',\
                            options_bool[index], set_bool[index])
                except io.OptionError, e:
                    print 'Could not read option solver.' +\
                            options_bool[index]
        
    def ParametersDialog(self, frame):
        '''ParametersDialog(self, frame) --> None
        
        Shows the Parameters dialog box to set the parameters for the solver.
        '''
        # Update the configuration if a model has been loaded after
        # the object have been created..
        self.ReadConfig()
        fom_func_name = self.parent.model.fom_func.__name__
        if not fom_func_name in fom_funcs.func_names:
            ShowWarningDialog(self.parent, 'The loaded fom function, '\
            + fom_func_name+ ', does not exist ' + \
            'in the local fom_funcs file. The fom fucntion has been' +
            ' temporary added to the list of availabe fom functions')
            fom_funcs.func_names.append(fom_func_name)
            exectext = 'fom_funcs.' + fom_func_name +\
                        ' = self.parent.model.fom_func'
            exec exectext in locals(), globals()
        dlg = SettingsDialog(frame, self.optimizer, fom_func_name)
        
        def applyfunc(object):
            self.WriteConfig()
            self.parent.model.set_fom_func(\
                    eval('fom_funcs.'+object.get_fom_string()))
            
        dlg.set_apply_change_func(applyfunc)
        
        dlg.ShowModal()
        #if dlg.ShowModal() == wx.ID_OK:
        #    pass
        dlg.Destroy()
        
        
        
    def TextOutput(self, text):
        '''TextOutput(self, text) --> None
        Function to present the output from the optimizer to the user. 
        Takes a string as input.
        '''
        #self.parent.main_frame_statusbar.SetStatusText(text, 0)
        evt = update_text(text = text)
        wx.PostEvent(self.parent, evt)
        
    def PlotOutput(self, solver):
        ''' PlotOutput(self, solver) --> None
        Solver to present the graphical output from the optimizer to the 
        user. Takes the solver as input argument and picks out the 
        variables to show in the GUI.
        '''
        #print 'sending event plotting'
        #_post_solver_event(self.parent, solver, desc = 'Fitting update')
        evt = update_plot(model = solver.get_model(), \
                fom_log = solver.get_fom_log(), update_fit = solver.new_best,\
                desc = 'Fitting update')
        wx.PostEvent(self.parent, evt)
        
    def ParameterOutput(self, solver):
        '''ParameterOutput(self, solver) --> none
        
        Function to send an update event to update windows that displays
        the parameters to update the values. 
        Takes the solver as input argument and picks out the variables to 
        show in the GUI.
        '''
        evt = update_parameters(values = solver.best_vec.copy(),\
                new_best = solver.new_best,\
                population = solver.pop_vec,\
                max_val = solver.par_max, \
                min_val = solver.par_min, \
                fitting = True,\
                desc = 'Parameter Update', update_errors = False)
        wx.PostEvent(self.parent, evt)
        
    def FittingEnded(self, solver):
        '''FittingEnded(self, solver) --> None
        
        function used to post an event when the fitting has ended.
        This must be done since it is not htread safe otherwise. Same GUI in
        two threads when dialogs are run. dangerous...
        '''
        evt = fitting_ended(solver = solver, desc = 'Fitting Ended')
        wx.PostEvent(self.parent, evt)
    
    def OnFittingEnded(self, evt):
        '''OnFittingEnded(self, solver) --> None
        
        Callback when fitting has ended. Takes care of cleaning up after
        the fit. Calculates errors on the parameters and updates the grid.
        '''
        solver = evt.solver
        
        message = 'Do you want to keep the parameter values from' +\
                'the fit?'
        dlg = wx.MessageDialog(self.parent, message,'Keep the fit?', 
            wx.YES_NO|wx.ICON_QUESTION)
        if dlg.ShowModal() == wx.ID_YES:
            pass
        else:
            print 'Resetting the values in the grid to ',\
                self.start_parameter_values
            evt = update_parameters(values = self.start_parameter_values,\
                desc = 'Parameter Update', new_best = True, \
                update_errors = False, fitting = False)
            wx.PostEvent(self.parent, evt)
            
    def CalcErrorBars(self):
        '''CalcErrorBars(self) -- None
        
        Method that calculates the errorbars for the fit that has been
        done. Note that the fit has to been conducted before this is runned.
        '''
        if self.start_parameter_values != None and not self.optimizer.running:
            n_elements = len(self.start_parameter_values)
            #print 'Number of elemets to calc errobars for ', n_elements
            error_values = []
            dlg = wx.ProgressDialog("Calculating", \
                               "Error bars are calculated ...", \
                               maximum = n_elements, parent=self.parent, \
                               style = wx.PD_AUTO_HIDE)
            for index in range(n_elements):
                # calculate the error
                # TODO: Check the error bar buisness again and how to treat 
                # Chi2 
                try:
                    (error_low, error_high) = self.optimizer.calc_error_bar(\
                                            index, self.fom_error_bars_level)
                except diffev.ErrorBarError, e:
                    ShowWarningDialog(self.parent, str(e))
                    break
                error_str = '(%.3e, %.3e,)'%(error_low, error_high)
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
        model  = self.parent.model
        row = model.parameters.get_pos_from_row(parameter)
        if self.start_parameter_values != None and not self.optimizer.running:
            return self.optimizer.par_evals[:,row],\
                self.optimizer.fom_evals
        else:
            raise ErrorBarError()

    def ScanParameter(self, parameter, points):
        '''ScanParameter(self, parameter, points) 
            --> par_vals, fom_vals
        
        Scans one parameter and records its fom value as a function 
        of the parameter value.
        '''
        row = parameter
        model = self.parent.model
        (funcs, vals) = model.get_sim_pars()
        minval = model.parameters.get_data()[row][3]
        maxval = model.parameters.get_data()[row][4]
        parfunc = funcs[row]
        step = (maxval - minval)/points
        par_vals = np.arange(minval, maxval + step, step)
        fom_vals = np.array([])
        
        par_name = model.parameters.get_data()[row][0]
        dlg = wx.ProgressDialog("Scan Parameter",
                               "Scanning parameter " + par_name,
                               maximum = len(par_vals),
                               parent=self.parent,
                               style = wx.PD_APP_MODAL| wx.PD_ELAPSED_TIME
                               | wx.PD_REMAINING_TIME | wx.PD_AUTO_HIDE)
        try:
            for par_val in par_vals:
                parfunc(par_val)
                fom_vals = np.append(fom_vals, model.evaluate_fit_func())
                dlg.Update(len(fom_vals))
        except Exception, e:
            dlg.Destroy()
            outp = StringIO.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            ShowWarningDialog(self.parent, 'Error while evaluation the' + \
            ' simulation and fom. Please check so it is possible to simulate'+\
            ' your model. Detailed output below: \n\n' + val)
        else:
            dlg.Destroy()
        
        return par_vals, fom_vals
        
    def ResetOptimizer(self):
        '''ResetOptimizer(self) --> None
        
        Resets the optimizer - clears the memory and special flags.
        '''
        self.start_parameter_values = None
        
    
    def StartFit(self):
        ''' StartFit(self) --> None
        Function to start running the fit
        '''
        # Make sure that the config of the solver is updated..
        self.ReadConfig()
        model = self.parent.model
        self.start_parameter_values = model.get_fit_values()
        self.optimizer.start_fit(model)
        print 'Optimizer starting'
        
    def StopFit(self):
        ''' StopFit(self) --> None
        Function to stop a running fit
        '''
        self.optimizer.stop_fit()
        
    def ResumeFit(self):
        ''' ResumeFit(self) --> None
        
        Function to resume the fitting after it has been stopped
        '''
        # Make sure teh settings are updated..
        self.ReadConfig()
        self.optimizer.resume_fit()
        
    def IsFitted(self):
        '''IsFitted(self) --> bool
        
        Returns true if a fit has been started otherwise False
        '''
        return self.start_parameter_values != None
        
        

#==============================================================================
class SettingsDialog(wx.Dialog):
    def __init__(self, parent, solver, fom_string):
        wx.Dialog.__init__(self, parent, -1, 'Optimizer settings')
        #self.SetAutoLayout(True)
        self.solver = solver
        self.apply_change = None
        
        col_sizer = wx.BoxSizer(wx.HORIZONTAL)
        row_sizer1 = wx.BoxSizer(wx.VERTICAL)
        
        # Make the Diff. Ev. box
        de_box = wx.StaticBox(self, -1, "Diff. Ev." )
        de_box_sizer = wx.StaticBoxSizer(de_box, wx.VERTICAL )
        de_grid = wx.GridBagSizer(2, 2)
        
        km_sizer = wx.BoxSizer(wx.HORIZONTAL)
        km_text = wx.StaticText(self, -1, 'k_m ')
        self.km_control = NumCtrl(self, value = self.solver.km,\
            fractionWidth = 2, integerWidth = 2)
        km_sizer.Add(km_text,0, \
                wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL, border = 10)
        km_sizer.Add(self.km_control,0, \
                wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border = 10)
        km_sizer.Add((10, 20), 0, wx.EXPAND)
        de_grid.Add(km_sizer, (0,0),\
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,\
                    border = 5)
        
        kr_sizer = wx.BoxSizer(wx.HORIZONTAL)
        kr_sizer.Add((10, 20), 0, wx.EXPAND)
        kr_text = wx.StaticText(self, -1, 'k_r ')
        self.kr_control = NumCtrl(self, value = self.solver.kr,\
            fractionWidth = 2, integerWidth = 2)
        kr_sizer.Add(kr_text, 0, \
                wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL, border = 10)
        kr_sizer.Add(self.kr_control, 0, \
                wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border = 10)
        de_grid.Add(kr_sizer, (0,1), \
                    flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,\
                    border = 5)
        
        method_sizer = wx.BoxSizer(wx.HORIZONTAL)
        method_text = wx.StaticText(self, -1, 'Method ')
        self.method_choice = wx.Choice(self, -1,choices = ['best1_bin'])
        method_sizer.Add(method_text,0, \
            wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL, border = 10)
        method_sizer.Add(self.method_choice,0,\
            wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border = 10)
        de_grid.Add(method_sizer, (1,0),(1,2), \
                    flag = wx.ALIGN_CENTER|wx.ALIGN_CENTER_VERTICAL,\
                    border = 5)
        
        de_box_sizer.Add(de_grid, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        row_sizer1.Add(de_box_sizer, 0, wx.ALIGN_CENTRE|wx.ALL|wx.EXPAND, 5)
        
        # Make the Fitting box
        fit_box = wx.StaticBox(self, -1, "Fitting" )
        fit_box_sizer = wx.StaticBoxSizer(fit_box, wx.VERTICAL )
        
        # Check box for start guess
        startguess_control = wx.CheckBox(self, -1, "Use start guess")
        fit_box_sizer.Add(startguess_control, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        startguess_control.SetValue(self.solver.use_start_guess)
        self.startguess_control = startguess_control
        
        # FOM choice
        fom_sizer = wx.BoxSizer(wx.HORIZONTAL)
        fom_text = wx.StaticText(self, -1, 'FOM ')
        self.fom_choice = wx.Choice(self, -1,choices = fom_funcs.func_names)
        self.fom_choice.SetSelection(fom_funcs.func_names.index(fom_string))
        fom_sizer.Add(fom_text,0, \
            wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL, border = 10)
        fom_sizer.Add(self.fom_choice,0,\
            wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border = 10)
        
        fit_box_sizer.Add(fom_sizer, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        row_sizer1.Add(fit_box_sizer, 0, wx.ALIGN_CENTRE|wx.ALL|wx.EXPAND, 5)
        
        col_sizer.Add(row_sizer1, 0, wx.ALIGN_CENTRE|wx.ALL, 5)
        
        row_sizer2 = wx.BoxSizer(wx.VERTICAL)

        # Make the Population box
        pop_box = wx.StaticBox(self, -1, "Population size" )
        pop_box_sizer = wx.StaticBoxSizer(pop_box, wx.VERTICAL )
        pop_grid = wx.FlexGridSizer(0, 2, 0, 0)
        
        multsize_radio = wx.RadioButton(self, -1,  " Relative size ",\
                            style = wx.RB_GROUP )
        fixedsize_radio = wx.RadioButton(self, -1, " Fixed size " )
        
        multsize_sc = wx.SpinCtrl(self)
        multsize_sc.SetRange(1,1000)
        multsize_sc.SetValue(self.solver.pop_mult)
        multsize_sc.Enable(self.solver.use_pop_mult)
        fixedsize_sc = wx.SpinCtrl(self)
        fixedsize_sc.SetRange(1,1000)
        fixedsize_sc.SetValue(self.solver.pop_size)
        fixedsize_sc.Enable(not self.solver.use_pop_mult)

        self.pop_multsize_radio = multsize_radio
        self.pop_fixedsize_radio = fixedsize_radio
        self.pop_multsize_sc = multsize_sc
        self.pop_fixedsize_sc = fixedsize_sc
        
        pop_grid.Add(multsize_radio, 0,\
            wx.ALIGN_LEFT|wx.LEFT|wx.RIGHT|wx.TOP, 5 )
        pop_grid.Add(multsize_sc, 0,\
            wx.ALIGN_RIGHT|wx.LEFT|wx.RIGHT|wx.TOP, 5 )
        pop_grid.Add( fixedsize_radio, 0,\
            wx.ALIGN_LEFT|wx.LEFT|wx.RIGHT|wx.TOP, 5 )
        pop_grid.Add(fixedsize_sc, 0,\
            wx.ALIGN_RIGHT|wx.LEFT|wx.RIGHT|wx.TOP, 5 )
        
        pop_box_sizer.Add(pop_grid, 0, wx.ALIGN_CENTRE|wx.ALL, 5 )
        row_sizer2.Add(pop_box_sizer, 0, wx.ALIGN_CENTRE|wx.ALL, 5 )
        self.Bind(wx.EVT_RADIOBUTTON, self.on_pop_select, multsize_radio)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_pop_select, fixedsize_radio)
        multsize_radio.SetValue(self.solver.use_pop_mult)
        fixedsize_radio.SetValue(not self.solver.use_pop_mult)
        
        # Make the Generation box
        gen_box = wx.StaticBox(self, -1, "Max Generations" )
        gen_box_sizer = wx.StaticBoxSizer(gen_box, wx.VERTICAL )
        gen_grid = wx.FlexGridSizer(0, 2, 0, 0)
        
        gen_multsize_radio = wx.RadioButton( self, -1, " Relative size ",\
                            style = wx.RB_GROUP )
        gen_fixedsize_radio = wx.RadioButton( self, -1, " Fixed size " )

        gen_multsize_sc = wx.SpinCtrl(self)
        gen_multsize_sc.SetRange(1,10000)
        gen_multsize_sc.SetValue(self.solver.max_generation_mult)
        gen_multsize_sc.Enable(not self.solver.use_max_generations)
        gen_fixedsize_sc = wx.SpinCtrl(self)
        gen_fixedsize_sc.SetRange(1,10000)
        gen_fixedsize_sc.SetValue(self.solver.max_generations)
        gen_fixedsize_sc.Enable(self.solver.use_max_generations)
        
        self.gen_multsize_radio = gen_multsize_radio
        self.gen_fixedsize_radio = gen_fixedsize_radio
        self.gen_multsize_sc = gen_multsize_sc
        self.gen_fixedsize_sc = gen_fixedsize_sc
        
        gen_grid.Add(gen_multsize_radio, 0,\
            wx.ALIGN_LEFT|wx.LEFT|wx.RIGHT|wx.TOP, 5 )
        gen_grid.Add(gen_multsize_sc, 0,\
            wx.ALIGN_CENTRE|wx.LEFT|wx.RIGHT|wx.TOP, 5 )
        gen_grid.Add(gen_fixedsize_radio, 0,\
            wx.ALIGN_LEFT|wx.LEFT|wx.RIGHT|wx.TOP, 5 )
        gen_grid.Add(gen_fixedsize_sc, 0,\
            wx.ALIGN_CENTRE|wx.LEFT|wx.RIGHT|wx.TOP, 5 )
        
        gen_box_sizer.Add(gen_grid, 0, wx.ALIGN_CENTRE|wx.ALL, 5 )
        row_sizer2.Add(gen_box_sizer, 0, wx.ALIGN_CENTRE|wx.ALL, 5 )
        self.Bind(wx.EVT_RADIOBUTTON, self.on_gen_select, gen_multsize_radio)
        self.Bind(wx.EVT_RADIOBUTTON, self.on_gen_select, gen_fixedsize_radio)
        gen_fixedsize_radio.SetValue(self.solver.use_max_generations)
        gen_multsize_radio.SetValue(not self.solver.use_max_generations)
        
        col_sizer.Add(row_sizer2, 1, wx.ALIGN_CENTRE|wx.ALL, 5)
        
        # Add the Dialog buttons
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
        self.Bind(wx.EVT_BUTTON, self.on_apply_change, okay_button)
        self.Bind(wx.EVT_BUTTON, self.on_apply_change, apply_button)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(col_sizer, 1, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL, 20)
        #sizer.Add(col_sizer, 1, wx.GROW|wx.ALL|wx.EXPAND, 20)
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL|wx.TOP, 20)
        
        sizer.Add(button_sizer,0,\
                flag = wx.ALIGN_RIGHT, border = 20)
        self.SetSizer(sizer)
        
        sizer.Fit(self)
        self.Layout()
        
    def on_pop_select(self, event):
        '''on_pop_select(self, event) --> None
        
        callback for selction of a radio button in the population group
        '''
        radio_selected = event.GetEventObject()
        
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
        radio_selected = event.GetEventObject()
        
        if radio_selected is self.gen_fixedsize_radio:
            self.gen_fixedsize_sc.Enable(True)
            self.gen_multsize_sc.Enable(False)
        else:
            self.gen_fixedsize_sc.Enable(False)
            self.gen_multsize_sc.Enable(True)
    
    def get_fom_string(self):
        return self.fom_choice.GetStringSelection()
    
    def on_apply_change(self, event):
        self.solver.kr = self.kr_control.GetValue()
        self.solver.km = self.km_control.GetValue()
        self.solver.max_generation_mult = self.gen_multsize_sc.GetValue()
        self.solver.max_generations = self.gen_fixedsize_sc.GetValue()
        self.solver.pop_mult = self.pop_multsize_sc.GetValue()
        self.solver.pop_size = self.pop_fixedsize_sc.GetValue() 
        self.solver.use_max_generations = self.gen_fixedsize_radio.GetValue()
        self.solver.use_pop_mult = self.pop_multsize_radio.GetValue()
        self.solver.use_start_guess = self.startguess_control.GetValue()
        if self.apply_change:
            self.apply_change(self)
            
        event.Skip()
    
    def set_apply_change_func(self, func):
        '''set_apply_change_func(self, func) --> None
        
        Set the apply_change function. Is executed when the apply or ok button
        is clicked.
        '''
        self.apply_change = func
    

#==============================================================================
# Custom events needed for updating and message parsing between the different
# modules.

(update_plot, EVT_UPDATE_PLOT) = wx.lib.newevent.NewEvent()
(update_text, EVT_SOLVER_UPDATE_TEXT) = wx.lib.newevent.NewEvent()
(update_parameters, EVT_UPDATE_PARAMETERS) = wx.lib.newevent.NewEvent()
(fitting_ended, EVT_FITTING_ENDED) = wx.lib.newevent.NewEvent()
#==============================================================================
def ShowWarningDialog(frame, message):
    dlg = wx.MessageDialog(frame, message,
                               'Warning',
                               wx.OK | wx.ICON_WARNING
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
    '''Error class for the fom evaluation'''
    def __init__(self):
        ''' __init__(self) --> None'''
        #self.error_message = error_message
    
    def __str__(self):
        text = 'Could not evaluate the error bars. A fit has to be run ' +\
                'before they can be calculated'
        return text
                
    