'''
Just a library of the different eventhandler needed in the GUI
Programmer: Matts Bjorck
Last changed: 2008 06 24
'''

__version__ = '2.0b trunk'

import wx, os, StringIO, traceback
from wx.lib.wordwrap import wordwrap

import model as modellib
import solvergui, help, io

def get_pages(frame):
    pages = [frame.plot_data, frame.plot_fom, frame.plot_pars,\
                frame.plot_fomscan]
    return pages

def set_title(frame):
    filepath, filename = os.path.split(frame.model.filename)
    if filename != '':
        if frame.model.saved:
            frame.SetTitle(filename + ' - ' + filepath + ' - GenX '\
                + __version__)
        else:
            frame.SetTitle(filename + '* - ' + filepath + ' - GenX '\
                + __version__)
    else:
        frame.SetTitle('GenX ' + __version__)
        
def models_changed(frame, event):
    '''models_changed(frame, event) --> None
    
    callback when something has changed in the model so that the 
    user can be made aware that the model needs saving.
    '''
    frame.model.saved = False
    set_title(frame)


def new(frame, event):
    '''
    new(frame, event) --> None
    
    Event handler for creating a new model
    '''
    # Reset the model - remove everything from the previous model
    frame.model.new_model()
    # Update all components so all the traces are gone.
    _post_new_model_event(frame, frame.model, desc = 'Fresh model')
    frame.plugin_control.OnNewModel(None)
    frame.main_frame_statusbar.SetStatusText('New model created', 1)
    set_title(frame)
    
def open(frame, event):
    '''
    open(frame, event) --> None
    
    Event handler for opening a model file...
    '''
    dlg = wx.FileDialog(frame, message="Open", defaultFile="",\
                        wildcard="GenX File (*.gx)|*.gx",\
                         style=wx.OPEN | wx.CHANGE_DIR 
                       )
    if dlg.ShowModal() == wx.ID_OK:
        path = dlg.GetPath()
        frame.model.load(dlg.GetPath())
        try:
            frame.solver_control.optimizer.pickle_load(\
                                    frame.model.load_addition('optimizer'))
        except Exception, e:
            ShowNotificationDialog(frame, 'The optimizer could not be loaded'\
                'from the saved file')
        frame.config.load_model(frame.model.load_addition('config'))
        [p.ReadConfig() for p in get_pages(frame)]
        # Letting the plugin do their stuff...
        frame.plugin_control.OnOpenModel(None)
        frame.main_frame_statusbar.SetStatusText('Model loaded from file', 1)
        # Post an event to update everything else
        _post_new_model_event(frame, frame.model)
        # Needs to put it to saved since all the widgets will have been updated
        frame.model.saved = True
        set_title(frame)
        
    dlg.Destroy()
    
    
    
    
def on_new_model(frame, event):
    '''
    on_new_model(frame, event) --> None
    
    Callback for NEW_MODEL event. Used to update the script for
    a new model i.e. put the string to the correct value. 
    '''
    # Set the string in the script_editor
    frame.script_editor.SetText(event.GetModel().get_script())
    # Let other event handlers recieve the event as well
    event.Skip()

def save(frame, event):
    '''
    save(frame, event) --> None
    
    Event handler for saving a model file ...
    '''
    frame.model.set_script(frame.script_editor.GetText())
    fname = frame.model.get_filename()
    # If model hasn't been saved
    if  fname == '':
        # Proceed with calling save as
        save_as(frame, event)
    else:
        # If it has been saved just save it
        #frame.model.save(fname)
        io.save_gx(fname, frame.model, frame.solver_control.optimizer,\
                        frame.config)
        set_title(frame)
        
        #frame.model.save_addition('optimizer',\
        #                        frame.solver_control.optimizer.pickle_string())
        #frame.model.save_addition('config', frame.config.model_dump())
        
    frame.main_frame_statusbar.SetStatusText('Model saved to file', 1)
    

def save_as(frame, event):
    '''save_as(frame, event) --> None
    
    Event handler for save as ...
    '''
    dlg = wx.FileDialog(frame, message="Save As", defaultFile="",\
                        wildcard="GenX File (*.gx)|*.gx",\
                         style=wx.SAVE | wx.CHANGE_DIR 
                       )
    if dlg.ShowModal() == wx.ID_OK:
        frame.model.set_script(frame.script_editor.GetText())
        fname = dlg.GetPath()
        result = True
        if os.path.exists(fname):
            filepath, filename = os.path.split(frame.model.filename)
            result = ShowQuestionDialog(frame, \
            'The file %s already exists. Do you wish to overwrite it?'%filename\
            , 'Overwrite?')
        if result:
            #frame.model.save(fname)
            io.save_gx(fname, frame.model, frame.solver_control.optimizer,\
                        frame.config)
            set_title(frame)
            #frame.model.save_addition('optimizer',\
            #                    frame.solver_control.optimizer.pickle_string())
            #frame.model.save_addition('config', frame.config.model_dump())
            #frame.main_frame_statusbar.SetStatusText('Model Saved to file', 1)
    dlg.Destroy()
    
def export_data(frame, event):
    '''export_data(frame, event) --> None
    
    exports the data to one file per data set with a basename with
    extention given by a save dialog.
    '''
    dlg = wx.FileDialog(frame, message="Export data", defaultFile="",\
                        wildcard="Dat File (*.dat)|*.dat",\
                         style=wx.SAVE | wx.CHANGE_DIR 
                       )
    if dlg.ShowModal() == wx.ID_OK:
        try:
            frame.model.export_data(dlg.GetPath())
        except modellib.IOError, e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText(\
                    'Error when exporting data', 1)
        except Exception, e:
            ShowErrorDialog(frame, str(e), 'export data - model.export_data')
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            frame.main_frame_statusbar.SetStatusText('Data exported', 1)
        
    dlg.Destroy()
    
def export_script(frame, event):
    '''export_script(frame, event) --> None
    
    Exports the script to a python file given by a filedialog.
    '''
    dlg = wx.FileDialog(frame, message="Export data", defaultFile="",\
                        wildcard="Python File (*.py)|*.py",\
                         style=wx.SAVE | wx.CHANGE_DIR 
                       )
    if dlg.ShowModal() == wx.ID_OK:
        fname = dlg.GetPath()
        result = True
        if os.path.exists(fname):
            filepath, filename = os.path.split(fname)
            result = ShowQuestionDialog(frame, \
            'The file %s already exists. Do you wish to overwrite it?'%filename\
            , 'Overwrite?')
        if result:
            try:
                frame.model.export_script(dlg.GetPath())
            except modellib.IOError, e:
                ShowModelErrorDialog(frame, str(e))
                frame.main_frame_statusbar.SetStatusText(\
                        'Error when exporting script', 1)
                return
            except Exception, e:
                ShowErrorDialog(frame, str(e),\
                                    'export script - model.export_script')
                frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
                return
            else:
                frame.main_frame_statusbar.SetStatusText(\
                                                'Script exported to file', 1)
            
    dlg.Destroy()
    
def export_table(frame, event):
    '''export_table(frame, event) --> None
    
    Exports the table to a dat file given by a filedialog.
    '''
    dlg = wx.FileDialog(frame, message="Export data", defaultFile="",\
                        wildcard="Table File (*.tab)|*.tab",\
                         style=wx.SAVE | wx.CHANGE_DIR 
                       )
    if dlg.ShowModal() == wx.ID_OK:
        fname = dlg.GetPath()
        result = True
        if os.path.exists(fname):
            filepath, filename = os.path.split(fname)
            result = ShowQuestionDialog(frame, \
            'The file %s already exists. Do you wish to overwrite it?'%filename\
            , 'Overwrite?')
        if result:
            try:
                frame.model.export_table(dlg.GetPath())
            except modellib.IOError, e:
                ShowModelErrorDialog(frame, str(e))
                frame.main_frame_statusbar.SetStatusText(\
                        'Error when exporting table', 1)
                return
            except Exception, e:
                ShowErrorDialog(frame, str(e),\
                                    'export table - model.export_table')
                frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
                return
            else:
                frame.main_frame_statusbar.SetStatusText(\
                                                'Table exported to file', 1)
            
    dlg.Destroy()
    
def import_script(frame, event):
    '''import_script(frame, event) --> None
    
    imports a script from the file given by a file dialog box
    '''
    dlg = wx.FileDialog(frame, message="Import script", defaultFile="",\
                    wildcard="Python files (*.py)|*.py|All files (*.*)|*.*",\
                         style=wx.OPEN | wx.CHANGE_DIR 
                       )
    if dlg.ShowModal() == wx.ID_OK:
        try:
            frame.model.import_script(dlg.GetPath())
        except modellib.IOError, e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText(\
                    'Error when importing script', 1)
            return
        except Exception, e:
            ShowErrorDialog(frame, str(e),\
                                'import script - model.import_script')
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            return
        else:
            frame.main_frame_statusbar.SetStatusText(\
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
    except Exception, e:
            ShowErrorDialog(frame, str(e),\
                                'import data - data_list.eh_tb_open')
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            return
    

def import_table(frame, event):
    '''import_table(frame, event) --> None
    
    imports a table from the file given by a file dialog box
    '''
    dlg = wx.FileDialog(frame, message="Import script", defaultFile="",\
                    wildcard="Table File (*.tab)|*.tab|All files (*.*)|*.*",\
                         style=wx.OPEN | wx.CHANGE_DIR 
                       )
    if dlg.ShowModal() == wx.ID_OK:
        try:
            frame.model.import_table(dlg.GetPath())
        except modellib.IOError, e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText(\
                    'Error when importing script', 1)
            dlg.Destroy()
            return
        except Exception, e:
            ShowErrorDialog(frame, str(e),\
                                'import script - model.import_script')
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
            dlg.Destroy()
            return
            
    dlg.Destroy()
    # Post event to tell that the model has cahnged
    _post_new_model_event(frame, frame.model)
    frame.main_frame_statusbar.SetStatusText('Table imported from file', 1)
    
def evaluate(frame, event):
    '''evaluate(frame, event) --> None
    
    Envent handler for only evaluating the Sim function - no recompiling
    '''
    frame.main_frame_statusbar.SetStatusText('Simulating...', 1)
    # Compile is not necessary when using simualate...
    #frame.model.compile_script()
    try:
        frame.model.simulate(compile = False)
    except modellib.GenericError, e:
        ShowModelErrorDialog(frame, str(e))
        frame.main_frame_statusbar.SetStatusText('Error in simulation', 1)
        return
    except Exception, e:
        outp = StringIO.StringIO()
        traceback.print_exc(200, outp)
        val = outp.getvalue()
        outp.close()
        ShowErrorDialog(frame, val)
        frame.main_frame_statusbar.SetStatusText('Fatal Error - simulate', 1)
        return
    else:
        _post_sim_plot_event(frame, frame.model, 'Simulation')
        frame.main_frame_statusbar.SetStatusText('Simulation Sucessful', 1)
        frame.plugin_control.OnSimulate(None)
        
    
def simulate(frame, event):
    '''
    simulate(frame, event) --> None
    
    Event handler for simulation.
    '''
    # Just a debugging output...
    # print frame.script_editor.GetText()
    frame.main_frame_statusbar.SetStatusText('Simulating...', 1)
    frame.model.set_script(frame.script_editor.GetText())
    # Compile is not necessary when using simualate...
    #frame.model.compile_script()
    try:
        frame.model.simulate()
    except modellib.GenericError, e:
        ShowModelErrorDialog(frame, str(e))
        frame.main_frame_statusbar.SetStatusText('Error in simulation', 1)
        return
    except Exception, e:
        outp = StringIO.StringIO()
        traceback.print_exc(200, outp)
        val = outp.getvalue()
        outp.close()
        ShowErrorDialog(frame, val)
        frame.main_frame_statusbar.SetStatusText('Fatal Error - simulate', 1)
        return
    else:
        _post_sim_plot_event(frame, frame.model, 'Simulation')
        frame.plugin_control.OnSimulate(None)
        frame.main_frame_statusbar.SetStatusText('Simulation Sucessful', 1)
    
    
    # Now we should find the parameters that we can use to
    # in the grid
    try:
        objlist, funclist = frame.model.get_possible_parameters()
    except Exception, e:
        ShowErrorDialog(frame, str(e),\
            'simulate - model.get_possible_parameters')
        frame.main_frame_statusbar.SetStatusText('Fatal Error', 0)
        return
    
    try:
        frame.paramter_grid.SetParameterSelections(objlist, funclist)
    except Exception, e:
        ShowErrorDialog(frame, str(e),\
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
        except modellib.GenericError, e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText('Error in fitting', 1)
        except Exception, e:
            ShowErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            frame.main_frame_statusbar.SetStatusText('Fitting starting ...', 1)
    else:
        ShowNotificationDialog(frame, 'The script is not compiled, do a'\
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
        except modellib.GenericError, e:
            ShowModelErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText('Error in fitting', 1)
        except Exception, e:
            ShowErrorDialog(frame, str(e))
            frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
        else:
            frame.main_frame_statusbar.SetStatusText('Fitting starting ...',1)
    else:
        ShowNotificationDialog(frame, 'The script is not compiled, do a'\
        ' simulation before you start fitting.')
    
    
def calculate_error_bars(frame, evt):
    '''calculate_error_bars(frame, evt) --> None
    
    callback to calculate the error bars on the data.
    '''
    try:
        error_values = frame.solver_control.CalcErrorBars()
    except solvergui.ErrorBarError, e:
        ShowNotificationDialog(frame, str(e))
    except Exception, e:
        ShowErrorDialog(frame, str(e), 'solvergui - CalcErrorBars')
        frame.main_frame_statusbar.SetStatusText('Fatal Error', 1)
    else:
        frame.model.parameters.set_error_pars(error_values)
        frame.paramter_grid.SetParameters(frame.model.parameters)
        frame.main_frame_statusbar.SetStatusText('Errorbars calculated', 1)

def scan_parameter(frame, row):
    ''' scan_parameter(frame, row) --> None
    
    Scans the parameter in row row [int] from max to min in the number
    of steps given by dialog input.
    '''
    if not frame.model.is_compiled():
        ShowNotificationDialog(frame, 'Please conduct a simulation before' +\
        ' scanning a parameter. The script needs to be compiled.')
        return
    
    dlg = wx.NumberEntryDialog(frame,\
                'Input the number of evaluation points for the scan',\
                'Steps', '', 50, 2, 1000)
    if dlg.ShowModal() ==wx.ID_OK:        
        frame.main_frame_statusbar.SetStatusText('Scanning parameter', 1)
        try:
            x, y = frame.solver_control.ScanParameter(row, dlg.GetValue())
            fs, pars = frame.model.get_sim_pars()
            bestx = pars[row]
            besty = frame.model.fom
            
            frame.plot_fomscan.SetPlottype('scan')
            frame.plot_fomscan.Plot((x, y, bestx, besty,\
                        frame.solver_control.fom_error_bars_level))
        except Exception, e:
            outp = StringIO.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
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
        ShowNotificationDialog(frame, 'Please conduct a fit before' +\
        ' scanning a parameter. The script needs to be compiled and foms have'\
         + ' to be collected.')
        return
    frame.main_frame_statusbar.SetStatusText('Trying to project fom', 1)
    try:
        x, y = frame.solver_control.ProjectEvals(row)
        fs, pars = frame.model.get_sim_pars()
        bestx = pars[row]
        besty = frame.model.fom
        frame.plot_fomscan.SetPlottype('project')
        frame.plot_fomscan.Plot((x, y, bestx, besty,\
                        frame.solver_control.fom_error_bars_level))
    except Exception, e:
        outp = StringIO.StringIO()
        traceback.print_exc(200, outp)
        val = outp.getvalue()
        outp.close()
        ShowErrorDialog(frame, val)
        frame.main_frame_statusbar.SetStatusText('Fatal Error - project fom', 1)
    else:    
        frame.main_frame_statusbar.SetStatusText('Projected fom plotted', 1)
    
def on_optimizer_settings(frame, event):
    '''on_optimizer_settings(self, event) --> None
    
    Show the settings dialog for the optimizer
    '''
    #try:
    frame.solver_control.ParametersDialog(frame)
    #except Exception, e:
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
        ans = ShowQuestionDialog(frame, 'The current model is not saved! '\
        'Do you want to abort quitting and save your data?', 'Abort close?')
        if not ans:
            frame.Destroy()
    else:
        frame.Destroy()

    
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
    fom_value = event.model.fom
    frame.main_frame_fom_text.SetLabel('        FOM: %.4e'%fom_value)
    
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
    sel = frame.plot_notebook.GetSelection()
    pages = get_pages(frame)
    if sel < len(pages):
        zoom_state = not pages[sel].GetZoom()
        pages[sel].SetZoom(zoom_state)
            
        frame.main_frame_toolbar.ToggleTool(10009, zoom_state)
        frame.mb_view_zoom.Check(zoom_state)

def zoomall(frame, event):
    '''zoomall(self, event) --> None
    
    Zoom out and show all data points
    '''
    sel = frame.plot_notebook.GetSelection()
    pages = get_pages(frame)
    if sel < len(pages):
        tmp = pages[sel].GetAutoScale()
        pages[sel].SetAutoScale(True)
        pages[sel].AutoScale()
        pages[sel].SetAutoScale(tmp)
        pages[sel].AutoScale()
        
def set_yscale(frame, type):
    '''set_yscale(frame, type) --> None
    
    Set the y-scale of the current plot. type should be linear or log, strings. 
    '''
    sel = frame.plot_notebook.GetSelection()
    pages = get_pages(frame)
    if sel < len(pages):
        pages[sel].SetYScale(type)
        
def on_autoscale(frame, event):
    '''on_autoscale(frame, event) --> None
    
    Toggles the autoscale of the current plot.
    '''
    sel = frame.plot_notebook.GetSelection()
    pages = get_pages(frame)
    if sel < len(pages):
        pages[sel].SetAutoScale(not pages[sel].GetAutoScale())
    
                
def plot_settings_changed(frame, event):
    '''zoom_changed(frame, event) --> None
    
    Callback for the settings change event for the current plot
     - change the toggle for the zoom icon and change the menu items.
    '''
    frame.main_frame_toolbar.ToggleTool(10009,event.zoomstate)
    frame.mb_view_zoom.Check(event.zoomstate)
    if event.yscale == 'log':
        frame.mb_view_yscale_log.Check(True)
    elif event.yscale == 'linear':
        frame.mb_view_yscale_lin.Check(True)
    frame.mb_view_autoscale.Check(event.autoscale)
    
def plot_page_changed(frame, event):
    '''plot_page_changed(frame, event) --> None
        
    Callback for page change in plot notebook. Changes the state of
    the zoom toggle button.
    '''
    sel = event.GetSelection()
    pages = get_pages(frame)
    if sel < len(pages):
        zoom_state = pages[sel].GetZoom()        
        # Set the zoom button to the correct value
        frame.main_frame_toolbar.ToggleTool(10009, zoom_state)
        frame.mb_view_zoom.Check(zoom_state)
        
        yscale = pages[sel].GetYScale()
        if yscale == 'log':
            frame.mb_view_yscale_log.Check(True)
        elif yscale == 'linear':
            frame.mb_view_yscale_lin.Check(True)

def print_plot(frame, event):
    '''print_plot(frame, event) --> None
    
    prints the current plot in the plot notebook.
    '''
    sel = frame.plot_notebook.GetSelection()
    pages = get_pages(frame)
    if sel < len(pages):
        pages[sel].Print()
        
def print_preview_plot(frame, event):
    '''print_preview_plot(frame, event) --> None
    
    prints a preview of the current plot int the plot notebook.
    '''
    sel = frame.plot_notebook.GetSelection()
    pages = get_pages(frame)
    if sel < len(pages):
        pages[sel].PrintPreview()
        
def print_parameter_grid(frame, event):
    ''' print_parameter_grid(frame, event) --> None
    
    Prints the table of parameters that have been fitted.
    '''
    frame.paramter_grid.Print()
    
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
    sel = frame.plot_notebook.GetSelection()
    pages = get_pages(frame)
    if sel < len(pages):
        pages[sel].CopyToClipboard()
        
def copy_table(frame, event):
    '''copy_table(frame, event) --> None
    
    Copies the table as ascii text to the clipboard
    '''
    ascii_table = frame.paramter_grid.table.pars.get_ascii_output()
    text_table=wx.TextDataObject(ascii_table)
    if wx.TheClipboard.Open():
        wx.TheClipboard.SetData(text_table)
        wx.TheClipboard.Close()
        
def copy_sim(frame, event):
    '''copy_sim(frame, event) --> None
    
    Copies the simulation and the data to the clipboard. Note that this
    copies ALL data.
    '''
    text_string = frame.model.get_data_as_asciitable()
    text = wx.TextDataObject(text_string)
    if wx.TheClipboard.Open():
        wx.TheClipboard.SetData(text)
        wx.TheClipboard.Close()
        
def change_data_grid_view(frame, event):
    '''change_data_grid_view(frame, event) --> None
    
    change the data displayed in the grid...
    '''
    #print event.GetSelection()
    dataset = frame.model.data[event.GetSelection()]
    rows = frame.data_grid.GetNumberRows()
    new_rows = max(len(dataset.x), len(dataset.y),\
                    len(dataset.x_raw), len(dataset.y_raw))
    frame.data_grid.DeleteRows(numRows = rows)
    frame.data_grid.AppendRows(new_rows)
    [[frame.data_grid.SetCellValue(row, col, '-') for col in range(6)]\
        for row in range(new_rows)]
    [frame.data_grid.SetCellValue(row, 0, '%.3e'%dataset.x_raw[row])\
        for row in range(len(dataset.x_raw))]
    [frame.data_grid.SetCellValue(row, 1, '%.3e'%dataset.y_raw[row])\
        for row in range(len(dataset.y_raw))]
    [frame.data_grid.SetCellValue(row, 2, '%.3e'%dataset.error_raw[row])\
        for row in range(len(dataset.error_raw))]
    [frame.data_grid.SetCellValue(row, 3, '%.3e'%dataset.x[row])\
        for row in range(len(dataset.x))]
    [frame.data_grid.SetCellValue(row, 4, '%.3e'%dataset.y[row])\
        for row in range(len(dataset.y))]
    [frame.data_grid.SetCellValue(row, 5, '%.3e'%dataset.error[row])\
        for row in range(len(dataset.error))]
        
def update_data_grid_choice(frame, event):
    '''update_data_grid_choice(frame, event) --> None
    
    Updates the choices of the grids to display from the data.
    '''
    data = event.GetData()
    names = [data_set.name for data_set in data]
    frame.data_grid_choice.SetItems(names)
    event.Skip()
    
def update_data(frame, event):
    '''update_data(frame, event) --> None
    
    callback for updating data, right now in the plugins
    '''
    frame.plugin_control.OnDataChanged(event)

def models_help(frame, event):
    '''models_help(frame, event) --> None
    
    Show a help dialog for information about the different models.
    '''
    dlg = help.PluginHelpDialog(frame,'models')
    dlg.Show()
    
def plugins_help(frame, event):
    '''plugins_help(frame, event) --> None
    
    Show a help dialog for information about the different plugins.
    '''
    dlg = help.PluginHelpDialog(frame,'plugins.add_ons')
    dlg.Show()

def data_loaders_help(frame, event):
    '''data_loaders_help(frame, event) --> None
    
    Show a help dialog for information about the different data_loaders.
    '''
    dlg = help.PluginHelpDialog(frame,'plugins.data_loaders')
    dlg.Show()    
    
def show_manual(frame, event):
    '''show_manual(frame, event) --> None
    
    Callback to show the manual
    '''
    ShowNotificationDialog(frame, 'There is no manual yet!')

def show_about_box(frame, event):
    '''show_about_box(frame, event) --> None
    
    Show an about box about GenX with some info...
    '''
    import numpy, scipy, matplotlib, platform
    try:
        import weave
    except:
        weave_version = 'Not installed'
    else:
        weave_version = weave.version.version
    try:
        import processing
    except:
        processing_version = 'Not installed'
    else:
        processing_version = processing.__version__
        
    info = wx.AboutDialogInfo()
    info.Name = "GenX"
    info.Version = __version__
    info.Copyright = "(C) 2008 Matts Bjorck"
    info.Description = wordwrap(
        "GenX is a multipurpose refinement program using the differential"
        "evolution algorithm. It is developed  mainly for refining x-ray reflectivity"
        "and neutron reflectivity data."
        
        "\n\nThe versions of the mandatory libraries are:\n"
        "Python: %s, wxPython: %s, Numpy: %s, Scipy: %s, Matplotlib: %s"
        "\nThe non-mandatory but useful packages:\n"
        "weave: %s, processing: %s"%(platform.python_version(), wx.__version__,\
            numpy.version.version, scipy.version.version,\
             matplotlib.__version__, weave_version, processing_version),
        350, wx.ClientDC(frame))
    info.WebSite = ("http:////genx.sourceforge.net", "GenX homepage")
    # No developers yet
    #info.Developers = []
    head, tail = os.path.split(__file__)
    license_text = file(head + '/LICENSE.txt','r').read()
    info.License = license_text#wordwrap(license_text, 500, wx.ClientDC(self))

    
    wx.AboutBox(info)
        
#=============================================================================
# Custom events needed for updating and message parsing between the different
# modules.

class GenericModelEvent(wx.PyCommandEvent):
    '''
    Event class for a new model - for updating
    of the paramters, plots and script.
    '''
    
    def __init__(self,evt_type, id, model):
        wx.PyCommandEvent.__init__(self, evt_type, id)
        self.model = model
        self.description = ''
        
    def GetModel(self):
        return self.model
        
    def SetModel(self, model):
        self.model = model
        
    def SetDescription(self, desc):
        '''
        Set a string that describes the event that has occurred
        '''
        self.description = desc
        
# Generating an event type:
myEVT_NEW_MODEL = wx.NewEventType()
# Creating an event binder object
EVT_NEW_MODEL = wx.PyEventBinder(myEVT_NEW_MODEL)

def _post_new_model_event(parent, model, desc = ''):
    # Send an event that a new data set has been loaded
        evt = GenericModelEvent(myEVT_NEW_MODEL, parent.GetId(), model)
        evt.SetDescription(desc)
        # Process the event!
        parent.GetEventHandler().ProcessEvent(evt)


# Generating an event type:
myEVT_SIM_PLOT = wx.NewEventType()
# Creating an event binder object
EVT_SIM_PLOT = wx.PyEventBinder(myEVT_SIM_PLOT)

def _post_sim_plot_event(parent, model, desc = ''):
    # Send an event that a new data set ahs been loaded
        evt = GenericModelEvent(myEVT_SIM_PLOT, parent.GetId(), model)
        evt.SetDescription(desc)
        # Process the event!
        parent.GetEventHandler().ProcessEvent(evt)
        

#==============================================================================
## Functions for showing error dialogs

def ShowQuestionDialog(frame, message, title = 'Question?'):
    dlg = wx.MessageDialog(frame, message,
                               title,
                               wx.YES_NO | wx.ICON_QUESTION
                               )
    result = dlg.ShowModal() == wx.ID_YES
    dlg.Destroy()
    return result

def ShowModelErrorDialog(frame, message):
    dlg = wx.MessageDialog(frame, message,
                               'Warning',
                               wx.OK | wx.ICON_WARNING
                               )
    dlg.ShowModal()
    dlg.Destroy()
    
def ShowNotificationDialog(frame, message):
    dlg = wx.MessageDialog(frame, message,
                               'Information',
                               wx.OK | wx.ICON_INFORMATION
                               )
    dlg.ShowModal()
    dlg.Destroy()
    
def ShowErrorDialog(frame, message, position = ''):
    if position != '':
        dlg = wx.MessageDialog(frame, message + '\n' + 'Position: ' + position,
                               'FATAL ERROR',
                               wx.OK | wx.ICON_ERROR
                               )
    else:
        dlg = wx.MessageDialog(frame, message,
                               'FATAL ERROR',
                               wx.OK | wx.ICON_ERROR
                               )
    dlg.ShowModal()
    dlg.Destroy()
#==============================================================================
