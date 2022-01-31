''' <h1>Reflectivity plugin </h1>
Reflectivity is a plugin for providing a graphical user
interface to define multilayer structures in GenX. It works
on quite general principels with dynamic generation of the
graphical user interface depending on the model. It also
dynamically generates the script for the model. Thus, it is 
always possible to go in and edit the script manually. <p>

The plugin consists of the following components:
<h2>Sample tab</h2>
This tab has the definitons for the layers and stacks. 
remember that a layer has to be inside a stack. 
Also, the name of the layers must be uniqe and can not be change
after the layer has been created. The functions of the buttons 
from left to right are:
<dl>
    <dt><b>Add Layer</b></dt>
    <dd>Add a new layer to the current position</dd>
    <dt><b>Add Stack</b></dt>
    <dd>Add a new stack to the current position</dd>
    <dt><b>Remove item</b></dt>
    <dd>Removes the current item. Note that the substrate, Sub, and the 
    ambient material can not be removed.</dd>
    <dt><b>Move item up</b></dt>
    <dd>Move item up</dd>
    <dt><b>Move item down</b></dt>
    <dd>Move item down</dd>
    <dt><b>Sample parameters</b></dt>
    <dd>Edit global parameters for the entire sample</dd>
    <dt><b>Instrument</b></dt>
    <dd>Edit parameters such as resolution and incident intesnity that
    is defines the instrument.</dd>
</dl>
<h2>Simulation tab</h2>
Here it is possible to add commands that are conducted before a data 
set is calculated. This done by adding a new command by pressing the green
plus sign. This brings up a dialog where the object and paraemter can
be chosen and the expression typed in. Note that the list one can choose from 
is <b>only</b> updated when the simulation button is pressed.<p>

The blue nut button to the right brings up a menu that allows the definition
of custom variables. These can be used to define problem specific parameters
 such as, for example, compostion of layers. One can also use this for parameter
coupling that yields a speedup in fitting. For example, fitting the repetition
length for a multilayer. 

<h2>SLD tab</h2>
This shows the real and imaginary part of the scattering length as a function
of depth for the sample. The substrate is to the left and the ambient material
is to the right. This is updated when the simulation button is pressed.
'''
import io, traceback
import wx.html

from .help_modules.reflectivity_misc import ReflectivityModule
from .help_modules.reflectivity_sample_plot import SamplePlotPanel
from .help_modules.custom_dialog import *
from .help_modules.reflectivity_utils import SampleHandler, SampleBuilder, avail_models
from .help_modules.reflectivity_gui import SamplePanel, DataParameterPanel
from .. import add_on_framework as framework
from genx.exceptions import GenxError
from genx.core.custom_logging import iprint
from genx.gui.custom_events import EVT_UPDATE_SCRIPT
from genx.model import Model


class Plugin(framework.Template, SampleBuilder, wx.EvtHandler):
    previous_xaxis = None
    _last_script = None
    model: ReflectivityModule
    sample_widget: SamplePanel
    sampleh: SampleHandler

    def __init__(self, parent):
        if 'SimpleReflectivity' in parent.plugin_control.plugin_handler.get_loaded_plugins():
            parent.plugin_control.UnLoadPlugin_by_Name('SimpleReflectivity')
        framework.Template.__init__(self, parent)
        wx.EvtHandler.__init__(self)
        # self.parent = parent
        self.model_obj = self.GetModel()
        sample_panel = self.NewInputFolder('Sample')
        sample_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sample_panel.SetSizer(sample_sizer)
        debug("Reflectivity: build sample widget")
        self.sample_widget = SamplePanel(sample_panel, self)
        sample_sizer.Add(self.sample_widget, 1, wx.EXPAND)
        sample_panel.Layout()

        simulation_panel = self.NewInputFolder('Simulations')
        simulation_sizer = wx.BoxSizer(wx.HORIZONTAL)
        simulation_panel.SetSizer(simulation_sizer)
        debug("Reflectivity: build data parameter widget")
        self.simulation_widget = DataParameterPanel(simulation_panel, self)
        simulation_sizer.Add(self.simulation_widget, 1, wx.EXPAND)
        simulation_panel.Layout()

        self.sample_widget.SetUpdateCallback(self.UpdateScript)
        self.simulation_widget.SetUpdateScriptFunc(self.UpdateScript)

        # Create the SLD plot
        sld_plot_panel = self.NewPlotFolder('SLD')
        sld_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sld_plot_panel.SetSizer(sld_sizer)
        debug("Reflectivity: build SLD plot")
        self.sld_plot = SamplePlotPanel(sld_plot_panel, self)
        sld_sizer.Add(self.sld_plot, 1, wx.EXPAND)
        sld_plot_panel.Layout()

        if self.model_obj.script!='':
            if self.model_obj.filename!='':
                iprint("Reflectivity plugin: Reading loaded model")
                try:
                    self.ReadModel()
                except Exception as e:
                    iprint("Reflectivity plugin model could not be read.")
                    self.Remove()
                    raise e
            else:
                try:
                    self.ReadModel()
                except:
                    iprint("Reflectivity plugin: Creating new model")
                    self.CreateNewModel()
        else:
            iprint("Reflectivity plugin: Creating new model")
            self.CreateNewModel()

        # Create a menu for handling the plugin
        menu = self.NewMenu('Reflec')
        self.mb_export_sld = wx.MenuItem(menu, wx.NewId(),
                                         "Export SLD...",
                                         "Export the SLD to a ASCII file",
                                         wx.ITEM_NORMAL)
        menu.Append(self.mb_export_sld)
        self.mb_show_imag_sld = wx.MenuItem(menu, wx.NewId(),
                                            "Show Im SLD",
                                            "Toggles showing the imaginary part of the SLD",
                                            wx.ITEM_CHECK)
        menu.Append(self.mb_show_imag_sld)
        self.mb_generate_uncertainty = wx.MenuItem(menu, wx.NewId(),
                                                   "Uncertainty Profile...",
                                                   "Generate a plot showing SLD uncertainty after a fit",
                                                   wx.ITEM_NORMAL)
        menu.Append(self.mb_generate_uncertainty)
        self.mb_export_uncertainty = wx.MenuItem(menu, wx.NewId(),
                                                 "Export Uncertainty...",
                                                 "Export SLD uncertainty after a fit",
                                                 wx.ITEM_NORMAL)
        menu.Append(self.mb_export_uncertainty)
        menu.AppendSeparator()
        self.mb_show_imag_sld.Check(self.sld_plot.opt.show_imag)
        self.mb_autoupdate_sld = wx.MenuItem(menu, wx.NewId(),
                                             "Autoupdate SLD",
                                             "Toggles autoupdating the SLD during fitting",
                                             wx.ITEM_CHECK)
        menu.Append(self.mb_autoupdate_sld)
        self.mb_autoupdate_sld.Check(False)
        # self.mb_autoupdate_sld.SetCheckable(True)
        self.parent.Bind(wx.EVT_MENU, self.OnExportSLD, self.mb_export_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnGenerateUncertainty, self.mb_generate_uncertainty)
        self.parent.Bind(wx.EVT_MENU, self.OnExportUncertainty, self.mb_export_uncertainty)
        self.parent.Bind(wx.EVT_MENU, self.OnAutoUpdateSLD, self.mb_autoupdate_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnShowImagSLD, self.mb_show_imag_sld)
        self.parent.model_control.Bind(EVT_UPDATE_SCRIPT, self.ReadUpdateModel)
        self.StatusMessage('Reflectivity plugin loaded')

    def SetModelScript(self, script):
        framework.Template.SetModelScript(self, script)
        self._last_script = script

    def UpdateScript(self, event):
        self.WriteModel()

    def OnAutoUpdateSLD(self, evt):
        # self.mb_autoupdate_sld.Check(not self.mb_autoupdate_sld.IsChecked())
        pass

    def OnShowImagSLD(self, evt):
        self.sld_plot.opt.show_imag = self.mb_show_imag_sld.IsChecked()
        self.sld_plot.WriteConfig()
        self.sld_plot.Plot()

    @property
    def show_imag_sld(self):
        return self.sld_plot.opt.show_imag

    def OnExportSLD(self, evt):
        dlg = wx.FileDialog(self.parent, message="Export SLD to ...",
                            defaultFile="",
                            wildcard="Dat File (*.dat)|*.dat",
                            style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
            fname = dlg.GetPath()
            result = True
            if os.path.exists(fname):
                filepath, filename = os.path.split(fname)
                result = self.ShowQuestionDialog('The file %s already exists.'
                                                 ' Do'
                                                 ' you wish to overwrite it?'
                                                 %filename)
            if result:
                try:
                    self.sld_plot.SavePlotData(fname)
                except IOError as e:
                    self.ShowErrorDialog(e.__str__())
                except Exception as e:
                    outp = io.StringIO()
                    traceback.print_exc(200, outp)
                    val = outp.getvalue()
                    outp.close()
                    self.ShowErrorDialog('Could not save the file.'
                                         ' Python Error:\n%s'%(val,))
        dlg.Destroy()

    def OnGenerateUncertainty(self, evt):
        validators = {
            'Reference Surface': ValueValidator(int),
            'Number of Samples': ValueValidator(int)
            }
        vals = {
            'Reference Surface': 0,
            'Number of Samples': 1000
            }
        pars = ['Reference Surface', 'Number of Samples']
        dlg = ValidateDialog(self.parent, pars, vals, validators,
                             title='Parameters for uncertainty graph')

        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
            vals = dlg.GetValues()
            dlg.Destroy()
        else:
            dlg.Destroy()
            return

        model: Model = self.GetModel()
        model.simulate()
        self.sld_plot.PlotConfidence(int(vals['Reference Surface']), int(vals['Number of Samples']))

    def OnExportUncertainty(self, evt):
        dlg = wx.FileDialog(self.parent, message="Export SLD to ...",
                            defaultFile="",
                            wildcard="Dat File (*.dat)|*.dat",
                            style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()!=wx.ID_OK:
            dlg.Destroy()
            return

        fname = dlg.GetPath()
        dlg.Destroy()

        result = True
        if os.path.exists(fname):
            filepath, filename = os.path.split(fname)
            result = self.ShowQuestionDialog('The file %s already exists.'
                                             ' Do'
                                             ' you wish to overwrite it?'
                                             %filename)
        if not result:
            return

        validators = {
            'Reference Surface': ValueValidator(int),
            'Number of Samples': ValueValidator(int)
            }
        vals = {
            'Reference Surface': 0,
            'Number of Samples': 1000
            }
        pars = ['Reference Surface', 'Number of Samples']
        dlg = ValidateDialog(self.parent, pars, vals, validators,
                             title='Parameters for uncertainty graph')

        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
            vals = dlg.GetValues()
            dlg.Destroy()
        else:
            dlg.Destroy()
            return

        model: Model = self.GetModel()
        model.simulate()
        self.sld_plot.SaveConfidenceData(fname, int(vals['Reference Surface']), int(vals['Number of Samples']),
                                         do_plot=True)

    def OnNewModel(self, event):
        ''' Create a new model
        '''
        dlg = wx.SingleChoiceDialog(self.parent, 'Choose a model type to use',
                                    'Models', avail_models,
                                    wx.CHOICEDLG_STYLE
                                    )

        if dlg.ShowModal()==wx.ID_OK:
            self.CreateNewModel('models.%s'%dlg.GetStringSelection())
        dlg.Destroy()

    def OnDataChanged(self, event):
        ''' Take into account changes in data..
        '''
        if event.new_model:
            return

        if event.data_moved or event.deleted or event.new_data or event.name_change:
            names = [data_set.name for data_set in self.GetModel().get_data()]
            self.simulation_widget.SetDataList(names)

            expl = self.simulation_widget.GetExpressionList()

            if len(names)-len(expl)==1:
                # Data set has been added:
                expl.append([])
                self.insert_new_data_segment(len(expl)-1)

            sims, insts, args = self.simulation_widget.GetSimArgs()

            if event.deleted:
                pos = list(range(len(expl)))
                [self.remove_data_segment(pos[-index-1]) for index in \
                 range(len(event.position))]
                [expl.pop(index) for index in event.position]
                [sims.pop(index) for index in event.position]
                [insts.pop(index) for index in event.position]
                [args.pop(index) for index in event.position]
            elif event.data_moved:
                if event.up:
                    # Moving up
                    for pos in event.position:
                        tmp = expl.pop(pos)
                        expl.insert(pos-1, tmp)
                        tmp = sims.pop(pos)
                        sims.insert(pos-1, tmp)
                        tmp = insts.pop(pos)
                        insts.insert(pos-1, tmp)
                        tmp = args.pop(pos)
                        args.insert(pos-1, tmp)
                else:
                    # Moving down...
                    for pos in event.position:
                        tmp = expl.pop(pos)
                        expl.insert(pos+1, tmp)
                        tmp = sims.pop(pos)
                        sims.insert(pos+1, tmp)
                        tmp = insts.pop(pos)
                        insts.insert(pos+1, tmp)
                        tmp = args.pop(pos)
                        args.insert(pos+1, tmp)

            self.simulation_widget.SetSimArgs(sims, insts, args)
            self.simulation_widget.SetExpressionList(expl)

            # Check so we have not clicked on new model button
            if self.GetModel().script!='':
                self.WriteModel()
                self.simulation_widget.UpdateListbox()
                if event.name_change:
                    self.sld_plot.Plot()
            else:
                self.simulation_widget.UpdateListbox(update_script=True)
        else:
            if event.data_changed:
                self.sld_plot.Plot()

    def OnOpenModel(self, event):
        '''
        Loads the sample into the plugin...
        '''
        self.mb_show_imag_sld.Check(self.sld_plot.opt.show_imag)
        self.ReadModel()

    def OnSimulate(self, event):
        '''
        Updates stuff after simulation
        '''
        if not self.mb_autoupdate_sld.IsChecked():
            # Calculate and update the sld plot, don't repeat when OnFittingUpdate does the call already
            wx.CallAfter(self.sld_plot.Plot)

    def OnFittingUpdate(self, event):
        '''
        Updates stuff during fitting
        '''
        # Calculate and update the sld plot
        if self.mb_autoupdate_sld.IsChecked():
            wx.CallAfter(self.sld_plot.Plot)

    def OnGridChange(self, event):
        """ Updates the simualtion panel when the grid changes

        :param event:
        :return:
        """
        self.sample_widget.Update(update_script=False)

    def InstrumentNameChange(self, old_name, new_name):
        '''OnInstrumentNameChange --> None
        
        Exchanges old_name to new name in the simulaitons.
        '''
        self.simulation_widget.InstrumentNameChange(old_name, new_name)

    def CreateNewModel(self, modelname='models.spec_nx'):
        '''Init the script in the model to yield the 
        correct script for initilization
        '''
        model_data = self.GetModel().get_data()
        nb_data_sets = len(model_data)

        script = self.GetNewModelScript(modelname, nb_data_sets)
        self.BuildNewModel(script)

        self.sample_widget.set_sampleh(self.sampleh)
        self.sample_widget.set_model(self.model)
        instrument = self.model.Instrument()
        self.sample_widget.SetInstrument({'inst': instrument})

        names = [data_set.name for data_set in model_data]
        self.simulation_widget.SetDataList(names)
        self.simulation_widget.SetParameterList([])
        # An empty list to the expression widget...
        self.simulation_widget.SetExpressionList([[] for item in names])
        self.simulation_widget.SetSimArgs(['Specular']*nb_data_sets,
                                          ['inst']*nb_data_sets,
                                          [['d.x'] for i in range(nb_data_sets)])
        self.simulation_widget.UpdateListbox(update_script=True)

        self.sample_widget.Update(update_script=True)
        # self.WriteModel()

    def WriteModel(self):
        parameter_list = self.simulation_widget.GetParameterList()
        sim_funcs, sim_insts, sim_args = self.simulation_widget.GetSimArgs()
        expression_list = self.simulation_widget.GetExpressionList()
        instruments = self.sample_widget.instruments

        self.write_model_script(sim_funcs, sim_insts, sim_args,
                                expression_list, parameter_list, instruments)
        try:
            self.SetXAxis(instruments[sim_insts[0]])
        except AttributeError:
            pass

    def SetXAxis(self, instrument):
        if self.previous_xaxis==instrument.coords:
            return
        coords = instrument.coords
        from genx import data
        if coords=='q':
            data.DataSet.simulation_params[0] = 0.001
            data.DataSet.simulation_params[1] = 0.601
        else:
            data.DataSet.simulation_params[0] = 0.01
            data.DataSet.simulation_params[1] = 6.01
        for ds in self.parent.model.data:
            ds.run_command()

    def AppendSim(self, sim_func, inst, args):
        self.simulation_widget.AppendSim(sim_func, inst, args)

    def ReadUpdateModel(self, evt):
        try:
            self.ReadModel(verbose=False)
        except GenxError:
            pass
        except Exception as e:
            self.StatusMessage(f'could not analyze script: {e}')

    def ReadModel(self, reevaluate=False, verbose=True):
        '''
        Reads in the current model and locates layers and stacks
        and sample defined inside BEGIN Sample section.
        '''
        if verbose: self.StatusMessage('Compiling the script...')
        self.CompileScript()
        if verbose: self.StatusMessage('Script compiled!')

        if verbose: self.StatusMessage('Trying to interpret the script...')

        instrument_names = self.find_instrument_names()

        if len(instrument_names)==0:
            self.ShowErrorDialog('Could not find any Instruments in the'+ \
                                 ' model script. Check the script.')
            self.StatusMessage('ERROR No Instruments in script')
            return

        if not 'inst' in instrument_names:
            self.ShowErrorDialog('Could not find the default'+
                                 ' Instrument, inst, in the'+
                                 ' model script. Check the script.')
            self.StatusMessage('ERROR No Instrument called inst in script')
            return

        sample_text = self.find_sample_section()

        if sample_text is None:
            self.ShowErrorDialog('Could not find the sample section'+ \
                                 ' in the model script.\n Can not load the sample in the editor.')
            self.StatusMessage('ERROR No sample section in script')
            return

        all_names, layers, stacks = self.find_layers_stacks(sample_text)

        if len(layers)==0:
            self.ShowErrorDialog('Could not find any Layers in the'+ \
                                 ' model script. Check the script.')
            self.StatusMessage('ERROR No Layers in script')
            return

        # Now its time to set all the parameters so that we have the strings
        # instead of the evaluated value - looks better
        for lay in layers:
            for par in lay[1].split(','):
                vars = par.split('=')
                exec('%s.%s = "%s"'%(lay[0], vars[0].strip(), vars[1].strip()), self.GetModel().script_module.__dict__)
        try:
            data_names, insts, sim_args, sim_exp, sim_funcs = self.find_sim_function_parameters()
        except LookupError:
            self.ShowErrorDialog('Could not locate all data sets in the'
                                 ' script. There should be %i datasets'%len(self.GetModel().get_data()))
            self.StatusMessage('ERROR No Layers in script')
            return

        uservars_lines = self.find_user_parameters()

        self.model = self.GetModel().script_module.model
        sample = self.GetModel().script_module.sample

        self.sampleh = SampleHandler(sample, all_names)
        self.sampleh.set_model(self.model)
        self.sample_widget.set_sampleh(self.sampleh)
        self.sample_widget.set_model(self.model)
        instruments = {}
        for name in instrument_names:
            instruments[name] = getattr(self.GetModel().script_module, name)
        self.sample_widget.SetInstrument(instruments)

        self.simulation_widget.SetDataList(data_names)
        self.simulation_widget.SetExpressionList(sim_exp)
        self.simulation_widget.SetParameterList(uservars_lines)

        self.simulation_widget.SetSimArgs(sim_funcs, insts, sim_args)

        self.sample_widget.Update(update_script=False)
        self.simulation_widget.UpdateListbox(update_script=False)
        # The code have a tendency to screw up the model slightly when compiling it - the sample will be connected
        # to the module therefore reset the compiled flag so that the model has to be recompiled before fitting.
        self.GetModel().compiled = False
        if reevaluate:
            if verbose: self.StatusMessage('Model analyzed and plugin updated!')
        else:
            if verbose: self.StatusMessage('New sample loaded to plugin!')
        self._last_script = self.model_obj.script

        # Setup the plot x-axis and simulation standard
        try:
            self.SetXAxis(self.sample_widget.instruments[instrument_names[0]])
        except AttributeError:
            pass
