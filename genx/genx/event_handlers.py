'''
Just a library of the different eventhandler needed in the GUI
File started by: Matts Bjorck
'''
from . import version

__version__=version.__version__

import _thread, time
import wx, os, traceback
from io import StringIO
import wx.adv
from wx.lib.wordwrap import wordwrap
import webbrowser
import numpy as np
from logging import debug

from . import model as modellib
from . import solvergui, help
from . import filehandling as io
from .gui_logging import iprint

manual_url='https://aglavic.github.io/genx/doc/'
homepage_url='https://aglavic.github.io/genx/'

def get_pages(frame):
    # Get all plot panel objects in GUI
    pages=[]
    for page in frame.plot_notebook.GetChildren():
        pages+=page.GetChildren()
    if frame.sep_plot_notebook is not frame.plot_notebook:
        for page in frame.sep_plot_notebook.GetChildren():
            pages+=page.GetChildren()

    # pages = [frame.plot_data, frame.plot_fom, frame.plot_pars,\
    #             frame.plot_fomscan]
    return pages

def set_title(frame):
    filepath, filename=os.path.split(frame.model.filename)
    if filename!='':
        if frame.model.saved:
            frame.SetTitle(filename+' - '+filepath+' - GenX ' \
                           +__version__)
        else:
            frame.SetTitle(filename+'* - '+filepath+' - GenX ' \
                           +__version__)
    else:
        frame.SetTitle('GenX '+__version__)

def models_changed(frame, event):
    '''models_changed(frame, event) --> None
    
    callback when something has changed in the model so that the 
    user can be made aware that the model needs saving.
    '''
    try:
        frame.model.saved=not event.permanent_change
    except AttributeError:
        frame.model.saved=False
    else:
        frame.plugin_control.OnGridChanged(event)
    set_title(frame)

def new(frame, event):
    '''
    new(frame, event) --> None
    
    Event handler for creating a new model
    '''
    # print "Event handler: new"
    if not frame.model.saved:
        ans=ShowQuestionDialog(frame, 'If you continue any changes in'
                                      ' your model will not be saved.',
                               'Model not saved')
        if not ans:
            return

    # Reset the model - remove everything from the previous model
    frame.model.new_model()
    # Update all components so all the traces are gone.
    _post_new_model_event(frame, frame.model, desc='Fresh model')
    frame.plugin_control.OnNewModel(None)
    frame.main_frame_statusbar.SetStatusText('New model created', 1)
    set_title(frame)
    frame.model.saved=True

def open(frame, event):
    '''
    open(frame, event) --> None
    
    Event handler for opening a model file...
    '''
    # Check so the model is saved before quitting
    if not frame.model.saved:
        ans=ShowQuestionDialog(frame, 'If you continue any changes in'
                                      ' your model will not be saved.',
                               'Model not saved')
        if not ans:
            return

    dlg=wx.FileDialog(frame, message="Open", defaultFile="",
                      wildcard="GenX File (*.hgx;*.gx)|*.hgx;*.gx",
                      style=wx.FD_OPEN  # | wx.FD_CHANGE_DIR
                      )
    if dlg.ShowModal()==wx.ID_OK:
        path=dlg.GetPath()
        debug('open: path retrieved')
        open_model(frame, path)

    dlg.Destroy()

def open_model(frame, path):
    debug('open_model: clear model')
    frame.model.new_model()
    frame.paramter_grid.PrepareNewModel()
    # Update all components so all the traces are gone.
    # _post_new_model_event(frame, frame.model)
    debug('open_model: load_file')
    try:
        io.load_file(path, frame.model, frame.solver_control.optimizer, frame.config)
    except modellib.IOError as e:
        ShowModelErrorDialog(frame, e.__str__())
    except Exception as e:
        outp=StringIO()
        traceback.print_exc(200, outp)
        val=outp.getvalue()
        outp.close()
        iprint('Error in loading the file ', path, '. Pyton traceback:\n ', val)
        ShowErrorDialog(frame, 'Could not open the file. Python Error:\n%s'%(val,))
        return
    debug('open_model: read config')
    try:
        [p.ReadConfig() for p in get_pages(frame)]
    except Exception as e:
        outp=StringIO()
        traceback.print_exc(200, outp)
        val=outp.getvalue()
        outp.close()
        iprint('Error in loading config for the plots. Pyton traceback:\n ', val)
        ShowErrorDialog(frame, 'Could not read the config for the plots. Python Error:\n%s'%(val,))
    try:
        frame.paramter_grid.ReadConfig()
    except Exception as e:
        outp=StringIO()
        traceback.print_exc(200, outp)
        val=outp.getvalue()
        outp.close()
        iprint('Error in loading config for parameter grid. Pyton traceback:\n ', val)
        ShowErrorDialog(frame, 'Could not read the config for the parameter grid. Python Error:\n%s'%(val,))
    else:
        # Update the Menu choice
        frame.main_frame_menubar.mb_view_grid_slider.Check(frame.paramter_grid.GetValueEditorSlider())
    debug('open_model: update plugins')
    try:
        frame.plugin_control.OnOpenModel(None)
    except Exception as e:
        outp=StringIO()
        traceback.print_exc(200, outp)
        val=outp.getvalue()
        outp.close()
        ShowErrorDialog(frame, 'Problems when plugins processed model.' \
                               ' Python Error:\n%s'%(val,))
    frame.main_frame_statusbar.SetStatusText('Model loaded from file',
                                             1)
    # Post an event to update everything else
    debug('open_model: post new model event')
    _post_new_model_event(frame, frame.model)
    # Needs to put it to saved since all the widgets will have 
    # been updated
    frame.model.saved=True
    set_title(frame)

def on_new_model(frame, event):
    '''
    on_new_model(frame, event) --> None
    
    Callback for NEW_MODEL event. Used to update the script for
    a new model i.e. put the string to the correct value. 
    '''
    # print "Callback: on_new_model"
    # Set the string in the script_editor
    frame.script_editor.SetText(event.GetModel().get_script())
    # Let the solvergui do its loading and updating:
    frame.solver_control.ModelLoaded()
    # Lets update the mb_use_toggle_show Menu item
    frame.main_frame_menubar.mb_use_toggle_show.Check(
        frame.config.get_boolean('data handling', 'toggle show'))
    try:
        val=frame.config.get_boolean('parameter grid', 'auto sim')
    except io.OptionError:
        iprint('Could not locate option parameters.auto sim')
        frame.main_frame_menubar.mb_fit_autosim.Check(True)
    else:
        frame.main_frame_menubar.mb_fit_autosim.Check(val)

    # Let other event handlers recieve the event as well
    event.Skip()

def update_for_save(frame):
    """Updates the various objects for a save"""
    frame.model.set_script(frame.script_editor.GetText())
    # Save the current state of autosim to the config
    frame.config.set('parameter grid', 'auto sim', frame.main_frame_menubar.mb_fit_autosim.IsChecked())

def save(frame, event):
    '''
    save(frame, event) --> None
    
    Event handler for saving a model file ...
    '''
    update_for_save(frame)
    fname=frame.model.get_filename()
    # If model hasn't been saved
    if fname=='':
        # Proceed with calling save as
        save_as(frame, event)
    else:
        # If it has been saved just save it
        try:
            io.save_file(fname, frame.model, frame.solver_control.optimizer, frame.config)
        except modellib.IOError as e:
            ShowModelErrorDialog(frame, e.__str__())
        except Exception as e:
            outp=StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            ShowErrorDialog(frame, 'Could not save the file. Python Error: \n%s'%(val,))
        set_title(frame)

    frame.main_frame_statusbar.SetStatusText('Model saved to file', 1)

def save_as(frame, event):
    '''save_as(frame, event) --> None
    
    Event handler for save as ...
    '''
    dlg=wx.FileDialog(frame, message="Save As", defaultFile="",
                      wildcard="HDF5 GenX File (*.hgx)|*.hgx|GenX File (*.gx)|*.gx",
                      style=wx.FD_SAVE  # | wx.FD_CHANGE_DIR
                      )
    if dlg.ShowModal()==wx.ID_OK:
        update_for_save(frame)
        fname=dlg.GetPath()
        base, ext=os.path.splitext(fname)
        if ext=='':
            ext='.hgx'
        fname=base+ext
        result=True
        if os.path.exists(fname):
            filepath, filename=os.path.split(fname)
            result=ShowQuestionDialog(frame, 'The file %s already exists. Do you wish to overwrite it?'%filename,
                                      'Overwrite?')
        if result:
            try:
                io.save_file(fname, frame.model, frame.solver_control.optimizer,
                             frame.config)
            except modellib.IOError as e:
                ShowModelErrorDialog(frame, e.__str__())
            except Exception as e:
                outp=StringIO()
                traceback.print_exc(200, outp)
                val=outp.getvalue()
                outp.close()
                ShowErrorDialog(frame, 'Could not save the file. Python Error:\n%s'%(val,))
            set_title(frame)

    dlg.Destroy()

def export_orso(frame, event):
    '''export_data(frame, event) --> None

    exports the data to one file per data set with a basename with
    extention given by a save dialog.
    '''
    dlg=wx.FileDialog(frame, message="Export data and model", defaultFile="",
                      wildcard="ORSO Text File (*.ort)|*.ort",
                      style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                      )
    if dlg.ShowModal()==wx.ID_OK:
        try:
            frame.model.export_orso(dlg.GetPath())
        except modellib.IOError as e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText(
                'Error when exporting data', 1)
        except Exception as e:
            ShowErrorDialog(frame, str(e), 'export data - model.export_orso')
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            frame.main_frame_statusbar.SetStatusText('Data exported', 1)

    dlg.Destroy()

def export_data(frame, event):
    '''export_data(frame, event) --> None
    
    exports the data to one file per data set with a basename with
    extention given by a save dialog.
    '''
    dlg=wx.FileDialog(frame, message="Export data", defaultFile="",
                      wildcard="Dat File (*.dat)|*.dat",
                      style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                      )
    if dlg.ShowModal()==wx.ID_OK:
        try:
            frame.model.export_data(dlg.GetPath())
        except modellib.IOError as e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText(
                'Error when exporting data', 1)
        except Exception as e:
            ShowErrorDialog(frame, str(e), 'export data - model.export_data')
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            frame.main_frame_statusbar.SetStatusText('Data exported', 1)

    dlg.Destroy()

def export_script(frame, event):
    '''export_script(frame, event) --> None
    
    Exports the script to a python file given by a filedialog.
    '''
    dlg=wx.FileDialog(frame, message="Export data", defaultFile="",
                      wildcard="Python File (*.py)|*.py",
                      style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                      )
    if dlg.ShowModal()==wx.ID_OK:
        fname=dlg.GetPath()
        base, ext=os.path.splitext(fname)
        if ext=='':
            ext='.py'
        fname=base+ext
        result=True
        if os.path.exists(fname):
            filepath, filename=os.path.split(fname)
            result=ShowQuestionDialog(frame,
                                      'The file %s already exists. Do you wish to overwrite it?'%filename
                                      , 'Overwrite?')
        if result:
            try:
                # frame.model.export_script(dlg.GetPath())
                frame.model.export_script(fname)
            except modellib.IOError as e:
                ShowModelErrorDialog(frame, str(e))
                frame.main_frame_statusbar.SetStatusText(
                    'Error when exporting script', 1)
                return
            except Exception as e:
                ShowErrorDialog(frame, str(e),
                                'export script - model.export_script')
                frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
                return
            else:
                frame.main_frame_statusbar.SetStatusText(
                    'Script exported to file', 1)

    dlg.Destroy()

def export_table(frame, event):
    '''export_table(frame, event) --> None
    
    Exports the table to a dat file given by a filedialog.
    '''
    dlg=wx.FileDialog(frame, message="Export data", defaultFile="",
                      wildcard="Table File (*.tab)|*.tab",
                      style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                      )
    if dlg.ShowModal()==wx.ID_OK:
        fname=dlg.GetPath()
        base, ext=os.path.splitext(fname)
        if ext=='':
            ext='.tab'
        fname=base+ext
        result=True
        if os.path.exists(fname):
            filepath, filename=os.path.split(fname)
            result=ShowQuestionDialog(frame,
                                      'The file %s already exists. Do you wish to overwrite it?'%filename
                                      , 'Overwrite?')
        if result:
            try:
                # frame.model.export_table(dlg.GetPath())
                frame.model.export_table(fname)
            except modellib.IOError as e:
                ShowModelErrorDialog(frame, str(e))
                frame.main_frame_statusbar.SetStatusText(
                    'Error when exporting table', 1)
                return
            except Exception as e:
                ShowErrorDialog(frame, str(e),
                                'export table - model.export_table')
                frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
                return
            else:
                frame.main_frame_statusbar.SetStatusText(
                    'Table exported to file', 1)

    dlg.Destroy()

def import_script(frame, event):
    '''import_script(frame, event) --> None
    
    imports a script from the file given by a file dialog box
    '''
    dlg=wx.FileDialog(frame, message="Import script", defaultFile="",
                      wildcard="Python files (*.py)|*.py|All files (*.*)|*.*",
                      style=wx.FD_OPEN | wx.FD_CHANGE_DIR
                      )
    if dlg.ShowModal()==wx.ID_OK:
        try:
            frame.model.import_script(dlg.GetPath())
            # frame.model.import_script(fname)
        except modellib.IOError as e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText(
                'Error when importing script', 1)
            return
        except Exception as e:
            ShowErrorDialog(frame, str(e),
                            'import script - model.import_script')
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            return
        try:
            frame.plugin_control.OnOpenModel(None)
        except Exception as e:
            outp=StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            ShowErrorDialog(frame, 'Problems when plugins processed model.' \
                                   ' Python Error:\n%s'%(val,))
        else:
            frame.main_frame_statusbar.SetStatusText(
                'Script imported from file', 1)
    dlg.Destroy()
    # Post event to tell that the model has changed
    _post_new_model_event(frame, frame.model)

def import_data(frame, event):
    ''' import_data(frame, event) -->None
    
    callback to import data into the program
    '''
    # Reuse of the callback in the datalist.DataController
    try:
        frame.data_list.eh_tb_open(event)
    except Exception as e:
        ShowErrorDialog(frame, str(e),
                        'import data - data_list.eh_tb_open')
        frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        return

def import_table(frame, event):
    '''import_table(frame, event) --> None
    
    imports a table from the file given by a file dialog box
    '''
    dlg=wx.FileDialog(frame, message="Import script", defaultFile="",
                      wildcard="Table File (*.tab)|*.tab|All files (*.*)|*.*",
                      style=wx.FD_OPEN | wx.FD_CHANGE_DIR
                      )
    if dlg.ShowModal()==wx.ID_OK:
        try:
            frame.model.import_table(dlg.GetPath())
        except modellib.IOError as e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText(
                'Error when importing script', 1)
            dlg.Destroy()
            return
        except Exception as e:
            ShowErrorDialog(frame, str(e),
                            'import script - model.import_script')
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            dlg.Destroy()
            return

    dlg.Destroy()
    # Post event to tell that the model has cahnged
    _post_new_model_event(frame, frame.model)
    frame.main_frame_statusbar.SetStatusText('Table imported from file', 1)

def parameter_value_changed(frame, event):
    """ Event handler for when a value of a parameter in the grid has been updated.

    :param frame:
    :param event:
    :return:
    """
    frame.simulation_queue_counter+=1
    if frame.main_frame_menubar.mb_fit_autosim.IsChecked() and not frame.flag_simulating:
        _thread.start_new_thread(simulation_loop, (frame,))

def simulation_loop(frame):
    """ Simulation loop for threading to increase the speed of the interactive simulations
    :param frame:
    :return:
    """
    frame.flag_simulating=True
    while frame.simulation_queue_counter>0:
        do_simulation(frame, from_thread=True)
        time.sleep(0.1)
        frame.simulation_queue_counter=min(1, frame.simulation_queue_counter-1)
    frame.flag_simulating=False

def evaluate(frame, event):
    '''evaluate(frame, event) --> None
    
    Envent handler for only evaluating the Sim function - no recompiling
    '''
    frame.flag_simulating=True
    frame.main_frame_statusbar.SetStatusText('Simulating...', 1)
    # Compile is not necessary when using simualate...
    # frame.model.compile_script()
    try:
        frame.model.simulate(compile=False)
    except modellib.GenericError as e:
        ShowModelErrorDialog(frame, str(e))
        frame.main_frame_statusbar.SetStatusText('Error in simulation', 1)
    except Exception as e:
        outp=StringIO()
        traceback.print_exc(200, outp)
        val=outp.getvalue()
        outp.close()
        ShowErrorDialog(frame, val)
        frame.main_frame_statusbar.SetStatusText('Fatal Error - simulate', 1)
    else:
        _post_sim_plot_event(frame, frame.model, 'Simulation')
        frame.main_frame_statusbar.SetStatusText('Simulation Sucessful', 1)
        frame.plugin_control.OnSimulate(None)
    frame.flag_simulating=False

def simulate(frame, event):
    '''
    simulate(frame, event) --> None
    
    Event handler for simulation.
    '''
    frame.flag_simulating=True
    do_simulation(frame)
    set_possible_parameters_in_grid(frame)
    frame.flag_simulating=False

def do_simulation(frame, from_thread=False):
    if not from_thread: frame.main_frame_statusbar.SetStatusText('Simulating...', 1)
    frame.model.set_script(frame.script_editor.GetText())
    try:
        # when updated from thread and was compiled before, do not compile again
        frame.model.simulate(compile=not (from_thread and frame.model.is_compiled()))
    except modellib.GenericError as e:
        wx.CallAfter(ShowModelErrorDialog, frame, str(e))
        if not from_thread: frame.main_frame_statusbar.SetStatusText('Error in simulation', 1)
    except Exception as e:
        outp=StringIO()
        traceback.print_exc(200, outp)
        val=outp.getvalue()
        outp.close()
        wx.CallAfter(ShowErrorDialog, frame, val)
        if not from_thread: frame.main_frame_statusbar.SetStatusText('Fatal Error - simulate', 1)
    else:
        wx.CallAfter(_post_sim_plot_event, frame, frame.model, 'Simulation')
        wx.CallAfter(frame.plugin_control.OnSimulate, None)
        if not from_thread: frame.main_frame_statusbar.SetStatusText('Simulation Sucessful', 1)

def set_possible_parameters_in_grid(frame):
    # Now we should find the parameters that we can use to
    # in the grid
    try:
        pardict=frame.model.get_possible_parameters()
    except Exception as e:
        outp=StringIO()
        traceback.print_exc(200, outp)
        val=outp.getvalue()
        outp.close()
        # ShowErrorDialog(frame, val)
        ShowErrorDialog(frame, val,
                        'simulate - model.get_possible_parameters')
        frame.main_frame_statusbar.SetStatusText('Fatal Error', 0)
        return

    try:
        frame.paramter_grid.SetParameterSelections(pardict)
    except Exception as e:
        ShowErrorDialog(frame, str(e),
                        'simulate - parameter_grid.SetParameterSelection')
        frame.main_frame_statusbar.SetStatusText('Fatal Error', 0)
        return
    # Set the function for which the parameter can be evaluated with
    frame.paramter_grid.SetEvalFunc(frame.model.eval_in_model)

def start_fit(frame, event):
    '''start_fit(frame, event) --> None
    
    Event handler to start fitting
    '''
    if frame.model.compiled:
        try:
            frame.solver_control.StartFit()
        except modellib.GenericError as e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText('Error in fitting', 1)
        except Exception as e:
            ShowErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            frame.main_frame_statusbar.SetStatusText('Fitting starting ...', 1)
    else:
        ShowNotificationDialog(frame, 'The script is not compiled, do a'
                                      ' simulation before you start fitting.')

def stop_fit(frame, event):
    '''stop_fit(frame, event) --> None
    
    Event handler to stop the fitting routine
    '''
    frame.solver_control.StopFit()

def resume_fit(frame, event):
    '''resume_fit(frame, event) --> None
    
    Event handler to resume the fitting routine. No initilization.
    '''
    if frame.model.compiled:
        try:
            frame.solver_control.ResumeFit()
        except modellib.GenericError as e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText('Error in fitting', 1)
        except Exception as e:
            ShowErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            frame.main_frame_statusbar.SetStatusText('Fitting starting ...', 1)
    else:
        ShowNotificationDialog(frame, 'The script is not compiled, do a'
                                      ' simulation before you start fitting.')

def calculate_error_bars(frame, evt):
    '''calculate_error_bars(frame, evt) --> None
    
    callback to calculate the error bars on the data.
    '''
    try:
        error_values=frame.solver_control.CalcErrorBars()
    except solvergui.ErrorBarError as e:
        ShowNotificationDialog(frame, str(e))
    except Exception as e:
        ShowErrorDialog(frame, str(e), 'solvergui - CalcErrorBars')
        frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
    else:
        frame.model.parameters.set_error_pars(error_values)
        frame.paramter_grid.SetParameters(frame.model.parameters)
        frame.main_frame_statusbar.SetStatusText('Errorbars calculated', 1)

def error_stats(frame, evt):
    try:
        from .bumps_interface import StatisticalAnalysisDialog
    except Exception as e:
        ShowErrorDialog(frame, str(e), 'solvergui - CalcErrorBars')
        frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
    else:
        dia=StatisticalAnalysisDialog(frame, frame.model)
        dia.ShowModal()

def scan_parameter(frame, row):
    ''' scan_parameter(frame, row) --> None
    
    Scans the parameter in row row [int] from max to min in the number
    of steps given by dialog input.
    '''
    if not frame.model.is_compiled():
        ShowNotificationDialog(frame, 'Please conduct a simulation before'+ \
                               ' scanning a parameter. The script needs to be compiled.')
        return

    dlg=wx.NumberEntryDialog(frame,
                             'Input the number of evaluation points for the scan',
                             'Steps', '', 50, 2, 1000)
    if dlg.ShowModal()==wx.ID_OK:
        frame.main_frame_statusbar.SetStatusText('Scanning parameter', 1)
        try:
            x, y=frame.solver_control.ScanParameter(row, dlg.GetValue())
            fs, pars=frame.model.get_sim_pars()
            bestx=frame.model.parameters.get_data()[row][1]
            besty=frame.model.fom

            frame.plot_fomscan.SetPlottype('scan')
            frame.plot_fomscan.Plot((x, y, bestx, besty,
                                     frame.solver_control.fom_error_bars_level),
                                    frame.model.parameters.get_names()[row],
                                    'FOM')
            frame.sep_plot_notebook.SetSelection(3)
        except Exception as e:
            outp=StringIO()
            traceback.print_exc(200, outp)
            val=outp.getvalue()
            outp.close()
            ShowErrorDialog(frame, val)
            frame.main_frame_statusbar.SetStatusText('Fatal Error - scan fom', 1)
        else:
            frame.main_frame_statusbar.SetStatusText('Scanning finished', 1)

    dlg.Destroy()

def project_fom_parameter(frame, row):
    '''project_fom_parameter(frame, row) --> None
    
    Plots the project fom given by the row row [int]
    '''
    if not frame.solver_control.IsFitted():
        ShowNotificationDialog(frame, 'Please conduct a fit before'+
                               ' scanning a parameter. The script needs to be compiled and foms have'
                               +' to be collected.')
        return

    frame.main_frame_statusbar.SetStatusText('Trying to project fom', 1)
    try:
        x, y=frame.solver_control.ProjectEvals(row)
        if len(x)==0 or len(y)==0:
            ShowNotificationDialog(frame, 'Please conduct a fit before'+
                                   ' projecting a parameter. The script needs to be compiled and foms have'
                                   +' to be collected.')
            return
        elif frame.model.fom is None or np.isnan(frame.model.fom):
            ShowNotificationDialog(frame, 'The model must be simulated (FOM is not a valid number)')
            return
        fs, pars=frame.model.get_sim_pars()
        bestx=pars[row]
        besty=frame.model.fom
        frame.plot_fomscan.SetPlottype('project')
        frame.plot_fomscan.Plot((x, y, bestx, besty,
                                 frame.solver_control.fom_error_bars_level),
                                frame.model.parameters.get_names()[row],
                                'FOM')
        frame.sep_plot_notebook.SetSelection(3)
    except Exception as e:
        outp=StringIO()
        traceback.print_exc(200, outp)
        val=outp.getvalue()
        outp.close()
        ShowErrorDialog(frame, val)
        frame.main_frame_statusbar.SetStatusText('Fatal Error - project fom', 1)
    else:
        frame.main_frame_statusbar.SetStatusText('Projected fom plotted', 1)

def on_optimizer_settings(frame, event):
    '''on_optimizer_settings(self, event) --> None
    
    Show the settings dialog for the optimizer
    '''
    # try:
    frame.solver_control.ParametersDialog(frame)
    # except Exception, e:
    #    raise e

def on_data_loader_settings(frame, event):
    '''on_data_loader_settings(frame, event) --> None
    
    Show the data_loader settings dialog. Allow the user to change the 
    data loader.
    '''
    frame.data_list.DataLoaderSettingsDialog()

def quit(frame, event):
    '''quit(frame, event) --> None
    
    Quit the program 
    '''
    # Check so the model is saved before quitting
    if not frame.model.saved:
        ans=ShowQuestionDialog(frame, 'If you continue any changes in'
                                      ' your model will not be saved.',
                               'Model not saved')
        if not ans:
            return

    import appdirs
    config_path=appdirs.user_data_dir('GenX3', 'ArturGlavic')+'/'
    hsize, vsize=frame.GetSize()
    frame.config.default_set('gui', 'hsize', hsize)
    frame.config.default_set('gui', 'vsize', vsize)
    frame.config.default_set('gui', 'vsplit', frame.ver_splitter.GetSashPosition())
    frame.config.default_set('gui', 'hsplit', frame.hor_splitter.GetSashPosition())
    frame.config.default_set('gui', 'psplit', frame.plot_splitter.GetSashPosition())
    frame.config.write_default(config_path+'genx.conf')

    # app=frame.parent
    frame.findreplace_dlg.Destroy()
    frame.findreplace_dlg=None
    event.Skip()
    frame.Destroy()
    # app.ExitMainLoop()

def status_text(frame, event):
    '''status_text(frame, event) --> None
    Print a status text in the window. event should have a string
    member text. This will display the message in the status bar.
    '''
    frame.main_frame_statusbar.SetStatusText(event.text, 1)

def fom_value(frame, event):
    '''fom_value(frame, event) --> None
    
    Callback to update the fom_value displayed by the gui
    '''
    fom_value=event.model.fom
    fom_name=event.model.fom_func.__name__
    if fom_value:
        frame.main_frame_fom_text.SetLabel('        FOM %s: %.4e'%(fom_name, fom_value))
    else:
        frame.main_frame_fom_text.SetLabel('        FOM %s: None'%fom_name)

def point_pick(frame, event):
    '''point_pick(frame, event) --> None
    Callback for the picking of a data point in a plotting window.
    This will display the message in the status bar.
    '''
    frame.main_frame_statusbar.SetStatusText(event.text, 2)

def on_zoom_check(frame, event):
    '''on_zoom_toolbar(event) --> none
    
    Takes care of clicks on the toolbar zoom button and the menu item zoom.
    '''
    if event.GetId()==10009:
        zoom_state=frame.main_frame_toolbar.GetToolState(10009)
        frame.main_frame_menubar.mb_view_zoom.Check(zoom_state)
    else:
        zoom_state=frame.main_frame_menubar.mb_view_zoom.IsChecked()
        frame.main_frame_toolbar.ToggleTool(10009, zoom_state)

    # Synchronize all plots with zoom state
    pages=get_pages(frame)
    for page in pages:
        page.SetZoom(zoom_state)

def zoomall(frame, event):
    '''zoomall(self, event) --> None
    
    Zoom out and show all data points
    '''
    sel=frame.sep_plot_notebook.GetSelection()
    pages=get_pages(frame)
    if sel<len(pages):
        tmp=pages[sel].GetAutoScale()
        pages[sel].SetAutoScale(True)
        pages[sel].AutoScale()
        pages[sel].SetAutoScale(tmp)
        pages[sel].AutoScale()

def set_yscale(frame, type):
    '''set_yscale(frame, type) --> None
    
    Set the y-scale of the current plot. type should be linear or log, strings. 
    '''
    sel=frame.sep_plot_notebook.GetSelection()
    pages=get_pages(frame)
    if sel<len(pages):
        pages[sel].SetYScale(type)

def set_xscale(frame, type):
    '''set_xscale(frame, type) --> None

    Set the x-scale of the current plot. type should be linear or log, strings.
    '''
    sel=frame.sep_plot_notebook.GetSelection()
    pages=get_pages(frame)
    if sel<len(pages):
        pages[sel].SetXScale(type)

def on_grid_slider_check(frame, event):
    """Change the state of the grid value input, either as slider or as a number.
    :param frame:
    :param event:
    :return:
    """
    frame.paramter_grid.SetValueEditorSlider(frame.main_frame_menubar.mb_view_grid_slider.IsChecked())
    # print frame.paramter_grid.get_toggle_slider_tool_state()
    frame.paramter_grid.toggle_slider_tool(frame.main_frame_menubar.mb_view_grid_slider.IsChecked())
    frame.paramter_grid.Refresh()

def on_autoscale(frame, event):
    '''on_autoscale(frame, event) --> None
    
    Toggles the autoscale of the current plot.
    '''
    sel=frame.sep_plot_notebook.GetSelection()
    pages=get_pages(frame)
    if sel<len(pages):
        pages[sel].SetAutoScale(not pages[sel].GetAutoScale())

def plot_settings_changed(frame, event):
    '''zoom_changed(frame, event) --> None
    
    Callback for the settings change event for the current plot
     - change the toggle for the zoom icon and change the menu items.
    '''
    frame.main_frame_toolbar.ToggleTool(10009, event.zoomstate)
    frame.main_frame_menubar.mb_view_zoom.Check(event.zoomstate)
    if event.yscale=='log':
        frame.main_frame_menubar.mb_view_yscale_log.Check(True)
    elif event.yscale=='linear':
        frame.main_frame_menubar.mb_view_yscale_lin.Check(True)
    if event.xscale=='log':
        frame.main_frame_menubar.mb_view_xscale_log.Check(True)
    elif event.xscale=='linear':
        frame.main_frame_menubar.mb_view_xscale_lin.Check(True)
    frame.main_frame_menubar.mb_view_autoscale.Check(event.autoscale)

def plot_page_changed(frame, event):
    '''plot_page_changed(frame, event) --> None
        
    Callback for page change in plot notebook. Changes the state of
    the zoom toggle button.
    '''
    sel=event.GetSelection()
    pages=get_pages(frame)
    if sel<len(pages):
        zoom_state=pages[sel].GetZoom()
        # Set the zoom button to the correct value
        frame.main_frame_toolbar.ToggleTool(10009, zoom_state)
        frame.main_frame_menubar.mb_view_zoom.Check(zoom_state)

        yscale=pages[sel].GetYScale()
        if yscale=='log':
            frame.main_frame_menubar.mb_view_yscale_log.Check(True)
        elif yscale=='linear':
            frame.main_frame_menubar.mb_view_yscale_lin.Check(True)
        xscale=pages[sel].GetXScale()
        if xscale=='log':
            frame.main_frame_menubar.mb_view_yscale_log.Check(True)
        elif xscale=='linear':
            frame.main_frame_menubar.mb_view_yscale_lin.Check(True)

def print_plot(frame, event):
    '''print_plot(frame, event) --> None
    
    prints the current plot in the plot notebook.
    '''
    sel=frame.sep_plot_notebook.GetSelection()
    pages=get_pages(frame)
    if sel<len(pages):
        pages[sel].Print()

def print_preview_plot(frame, event):
    '''print_preview_plot(frame, event) --> None
    
    prints a preview of the current plot int the plot notebook.
    '''
    sel=frame.sep_plot_notebook.GetSelection()
    pages=get_pages(frame)
    if sel<len(pages):
        pages[sel].PrintPreview()

def print_parameter_grid(frame, event):
    ''' print_parameter_grid(frame, event) --> None
    
    Prints the table of parameters that have been fitted.
    '''
    frame.paramter_grid.Print()
    # genxprint.TablePanel(frame)

def print_preview_parameter_grid(frame, event):
    ''' print_parameter_grid(frame, event) --> None
    
    Prints the table of parameters that have been fitted.
    '''
    frame.paramter_grid.PrintPreview()

def copy_graph(frame, event):
    '''copy_graph(self, event) --> None
    
    Callback that copies the current graph in the plot notebook to
    the clipboard.
    '''
    sel=frame.sep_plot_notebook.GetSelection()
    pages=get_pages(frame)
    if sel<len(pages):
        pages[sel].CopyToClipboard()

def copy_table(frame, event):
    '''copy_table(frame, event) --> None
    
    Copies the table as ascii text to the clipboard
    '''
    ascii_table=frame.paramter_grid.table.pars.get_ascii_output()
    text_table=wx.TextDataObject(ascii_table)
    if wx.TheClipboard.Open():
        wx.TheClipboard.SetData(text_table)
        wx.TheClipboard.Close()

def copy_sim(frame, event):
    '''copy_sim(frame, event) --> None
    
    Copies the simulation and the data to the clipboard. Note that this
    copies ALL data.
    '''
    text_string=frame.model.get_data_as_asciitable()
    text=wx.TextDataObject(text_string)
    if wx.TheClipboard.Open():
        wx.TheClipboard.SetData(text)
        wx.TheClipboard.Close()

def on_findreplace(frame, event):
    '''Show the find and replace dialog box.
    '''
    frame.findreplace_dlg.Show(True)

def on_find_event(frame, event):
    '''callback for find events - coupled to the script
    '''
    evtype=event.GetEventType()

    def find():
        find_str=event.GetFindString()
        ##print frame.findreplace_data.GetFlags()
        flags=event.GetFlags()
        if flags & 1:
            ##print "Searching down"
            pos=frame.script_editor.SearchNext(flags, find_str)
        else:
            ##print "Searching up"
            pos=frame.script_editor.SearchPrev(flags, find_str)
        if pos==-1:
            frame.main_frame_statusbar.SetStatusText(
                'Could not find text %s'%find_str, 1)
        return pos

    def replace():
        replace_str=event.GetReplaceString()
        frame.script_editor.ReplaceSelection(replace_str)

    # Deal with the different cases
    if evtype==wx.wxEVT_COMMAND_FIND:
        frame.script_editor.SearchAnchor()
        find()

    elif evtype==wx.wxEVT_COMMAND_FIND_NEXT:
        pnew=frame.script_editor.GetSelectionEnd()
        ##print pnew
        frame.script_editor.GotoPos(pnew)
        frame.script_editor.SetAnchor(pnew)
        frame.script_editor.SearchAnchor()
        ##print 'Finding next'
        find()

    elif evtype==wx.wxEVT_COMMAND_FIND_REPLACE:
        # print 'find and replace'
        # If we do not have found text already
        # or if we have marked other text by mistake...
        if frame.script_editor.GetSelectedText()!= \
                event.GetFindString():
            find()
        # We already have found and marked text that we should
        # replace
        else:
            frame.script_editor.ReplaceSelection(
                event.GetReplaceString())
            # Find a new text to replace
            find()
    elif evtype==wx.wxEVT_COMMAND_FIND_REPLACE_ALL:
        # print 'find and replace all'
        if frame.script_editor.GetSelectedText()!= \
                event.GetFindString():
            pos=find()
        i=0
        while pos!=-1:
            frame.script_editor.ReplaceSelection(
                event.GetReplaceString())
            i+=1
            pos=find()
        frame.main_frame_statusbar.SetStatusText(
            'Replaces %d occurancies of  %s'%(i,
                                              event.GetFindString()), 1)

    ##else:
    ##    ShowErrorDialog(frame, 'Faulty event supplied in find and'\
    ##                    ' repalce functionallity', 'on_find_event')
    # This will scroll the editor to the right position so we can see
    # the text
    frame.script_editor.EnsureCaretVisible()

def change_data_grid_view(frame, event):
    '''change_data_grid_view(frame, event) --> None
    
    change the data displayed in the grid...
    '''
    # print event.GetSelection()
    dataset=frame.model.data[event.GetSelection()]
    rows=frame.data_grid.GetNumberRows()
    new_rows=max(len(dataset.x), len(dataset.y),
                 len(dataset.x_raw), len(dataset.y_raw))
    frame.data_grid.DeleteRows(numRows=rows)
    frame.data_grid.AppendRows(new_rows)
    [[frame.data_grid.SetCellValue(row, col, '-') for col in range(6)] \
     for row in range(new_rows)]
    [frame.data_grid.SetCellValue(row, 0, '%.3e'%dataset.x_raw[row]) \
     for row in range(len(dataset.x_raw))]
    [frame.data_grid.SetCellValue(row, 1, '%.3e'%dataset.y_raw[row]) \
     for row in range(len(dataset.y_raw))]
    [frame.data_grid.SetCellValue(row, 2, '%.3e'%dataset.error_raw[row]) \
     for row in range(len(dataset.error_raw))]
    [frame.data_grid.SetCellValue(row, 3, '%.3e'%dataset.x[row]) \
     for row in range(len(dataset.x))]
    [frame.data_grid.SetCellValue(row, 4, '%.3e'%dataset.y[row]) \
     for row in range(len(dataset.y))]
    [frame.data_grid.SetCellValue(row, 5, '%.3e'%dataset.error[row]) \
     for row in range(len(dataset.error))]

def update_data_grid_choice(frame, event):
    '''update_data_grid_choice(frame, event) --> None
    
    Updates the choices of the grids to display from the data.
    '''
    data=event.GetData()
    names=[data_set.name for data_set in data]
    # frame.data_grid_choice.SetItems(names)
    frame.data_grid_choice.Clear()
    frame.data_grid_choice.AppendItems(names)
    event.Skip()

def update_data(frame, event):
    '''update_data(frame, event) --> None
    
    callback for updating data, right now in the plugins
    '''
    frame.plugin_control.OnDataChanged(event)

def activate_cuda(frame, event):
    dlg=wx.ProgressDialog(parent=frame,
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
    paratt.Refl=paratt_cuda.Refl
    paratt.ReflQ=paratt_cuda.ReflQ
    paratt.Refl_nvary2=paratt_cuda.Refl_nvary2
    neutron_refl.Refl=neutron_cuda.Refl
    from genx.models.lib import paratt, neutron_refl
    paratt.Refl=paratt_cuda.Refl
    paratt.ReflQ=paratt_cuda.ReflQ
    paratt.Refl_nvary2=paratt_cuda.Refl_nvary2
    neutron_refl.Refl=neutron_cuda.Refl

    dlg.Destroy()

def deactivate_cuda(frame, event):
    from genx.models.lib import paratt_numba, neutron_numba
    from models.lib import paratt, neutron_refl
    paratt.Refl=paratt_numba.Refl
    paratt.ReflQ=paratt_numba.ReflQ
    paratt.Refl_nvary2=paratt_numba.Refl_nvary2
    neutron_refl.Refl=neutron_numba.Refl
    from genx.models.lib import paratt, neutron_refl
    paratt.Refl=paratt_numba.Refl
    paratt.ReflQ=paratt_numba.ReflQ
    paratt.Refl_nvary2=paratt_numba.Refl_nvary2
    neutron_refl.Refl=neutron_numba.Refl

def fom_help(frame, event):
    '''Show a help dialog for information about the different fom.
    '''
    dlg=help.PluginHelpDialog(frame, 'fom_funcs', title="FOM functions help")
    dlg.Show()

def models_help(frame, event):
    '''models_help(frame, event) --> None
    
    Show a help dialog for information about the different models.
    '''
    dlg=help.PluginHelpDialog(frame, 'models', title="Models help")
    if frame.model.is_compiled():
        current_model=frame.model.script_module.model.__name__.rsplit('.', 1)[1]
        if current_model in dlg.choice.GetStrings():
            dlg.choice.SetStringSelection(current_model)
            dlg.on_choice(None)
    dlg.Show()

def plugins_help(frame, event):
    '''plugins_help(frame, event) --> None
    
    Show a help dialog for information about the different plugins.
    '''
    dlg=help.PluginHelpDialog(frame, 'plugins.add_ons', title="Plugins help")
    dlg.Show()

def data_loaders_help(frame, event):
    '''data_loaders_help(frame, event) --> None
    
    Show a help dialog for information about the different data_loaders.
    '''
    dlg=help.PluginHelpDialog(frame, 'plugins.data_loaders', title="Data loaders help")
    dlg.Show()

def show_manual(frame, event):
    '''show_manual(frame, event) --> None
    
    Callback to show the manual
    '''
    # ShowNotificationDialog(frame, 'There is no manual yet!')
    webbrowser.open_new(manual_url)

def show_homepage(frame, event):
    '''show_homepage(frame, event) --> None
    
    Callback to show the homepage
    '''
    webbrowser.open_new(homepage_url)

def show_about_box(frame, event):
    '''show_about_box(frame, event) --> None
    
    Show an about box about GenX with some info...
    '''
    import numpy, scipy, matplotlib, platform
    useful=''
    try:
        import numba
        useful+='Numba: %s, '%numba.__version__
    except:
        pass
    try:
        import vtk
        useful+='VTK: %s, '%vtk.vtkVersion.GetVTKVersion()
    except:
        pass
    try:
        import dumps
        useful+='Dumps: %s, '%dumps.__version__
    except:
        pass

    info=wx.adv.AboutDialogInfo()
    info.Name="GenX"
    info.Version=__version__
    info.Copyright="(C) 2008 Matts Bjorck; 2020 Artur Glavic"
    info.Description=wordwrap(
        "GenX is a multipurpose refinement program using the differential "
        "evolution algorithm. It is developed  mainly for refining x-ray reflectivity "
        "and neutron reflectivity data."

        "\n\nThe versions of the mandatory libraries are:\n"
        "Python: %s, wxPython: %s, Numpy: %s, Scipy: %s, Matplotlib: %s"
        "\n\nThe non-mandatory but useful packages:\n%s"
        ""%(platform.python_version(), wx.__version__,
            numpy.__version__, scipy.__version__,
            matplotlib.__version__, useful),
        500, wx.ClientDC(frame))
    info.WebSite=("http:////genx.sourceforge.net", "GenX homepage")
    # No developers yet
    # info.Developers = []
    # head, tail = os.path.split(__file__)
    # license_text = file(head + '/LICENSE.txt','r').read()
    # license_text = file(_path + 'LICENSE.txt','r').read()
    # info.License = license_text#wordwrap(license_text, 500, wx.ClientDC(self))
    info.Licence=wordwrap('This program is free software: you can redistribute it and/or modify '
                          'it under the terms of the GNU General Public License as published by '
                          'the Free Software Foundation, either version 3 of the License, or '
                          '(at your option) any later version. '
                          '\n\nThis program is distributed in the hope that it will be useful, '
                          'but WITHOUT ANY WARRANTY; without even the implied warranty of '
                          'MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the '
                          'GNU General Public License for more details. '
                          '\n\nYou should have received a copy of the GNU General Public License '
                          'along with this program.  If not, see <http://www.gnu.org/licenses/>. '
                          , 400, wx.ClientDC(frame))

    wx.adv.AboutBox(info)

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

# ==============================================================================
## Functions for showing error dialogs

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
# ==============================================================================
