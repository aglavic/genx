""" <h1>Reflectivity plugin </h1>
Reflectivity is a plugin for providing a graphical user
interface to define multilayer structures in GenX.

Qt port.
"""

import io
import os
import traceback

from logging import debug
from PySide6 import QtCore, QtWidgets

from genx.core.custom_logging import iprint
from genx.exceptions import GenxError
from genx.model import Model
from genx.plugins import add_on_framework as framework

from .help_modules.custom_dialog import ValidateDialog, ValueValidator
from .help_modules.reflectivity_gui import DataParameterPanel, SamplePanel
from .help_modules.reflectivity_misc import ReflectivityModule
from .help_modules.reflectivity_sample_plot import SamplePlotPanel
from .help_modules.reflectivity_utils import SampleBuilder, SampleHandler, avail_models


class Plugin(framework.Template, SampleBuilder):
    previous_xaxis = None
    _last_script = None
    model: ReflectivityModule
    sample_widget: SamplePanel
    sampleh: SampleHandler

    def __init__(self, parent):
        if "SimpleReflectivity" in parent.plugin_control.plugin_handler.get_loaded_plugins():
            parent.plugin_control.UnLoadPlugin_by_Name("SimpleReflectivity")
        framework.Template.__init__(self, parent)

        self.model_obj = self.GetModel()

        sample_panel = self.NewInputFolder("Sample")
        sample_layout = QtWidgets.QHBoxLayout(sample_panel)
        debug("Reflectivity: build sample widget")
        self.sample_widget = SamplePanel(sample_panel, self)
        sample_layout.addWidget(self.sample_widget, 1)

        simulation_panel = self.NewInputFolder("Simulations")
        simulation_layout = QtWidgets.QHBoxLayout(simulation_panel)
        debug("Reflectivity: build data parameter widget")
        self.simulation_widget = DataParameterPanel(simulation_panel, self)
        simulation_layout.addWidget(self.simulation_widget, 1)

        self.sample_widget.SetUpdateCallback(self.UpdateScript)
        self.simulation_widget.SetUpdateScriptFunc(self.UpdateScript)

        sld_plot_panel = self.NewPlotFolder("SLD")
        sld_layout = QtWidgets.QHBoxLayout(sld_plot_panel)
        debug("Reflectivity: build SLD plot")
        self.sld_plot = SamplePlotPanel(sld_plot_panel, self)
        sld_layout.addWidget(self.sld_plot, 1)

        if self.model_obj.script != "":
            if self.model_obj.filename != "":
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
                except Exception:
                    iprint("Reflectivity plugin: Creating new model")
                    self.CreateNewModel()
        else:
            iprint("Reflectivity plugin: Creating new model")
            self.CreateNewModel()

        menu = self.NewMenu("Reflec")
        self.action_export_sld = menu.addAction("Export SLD...")
        self.action_export_sld.setToolTip("Export the SLD to a ASCII file")

        self.action_show_imag = menu.addAction("Show Im SLD")
        self.action_show_imag.setCheckable(True)
        self.action_show_imag.setChecked(self.sld_plot.opt.show_imag)

        self.action_mass_density = menu.addAction("Mass Density")
        self.action_mass_density.setCheckable(True)
        self.action_mass_density.setChecked(self.sld_plot.opt.use_mass_density)

        self.action_generate_uncertainty = menu.addAction("Uncertainty Profile...")
        self.action_generate_uncertainty.setToolTip("Generate a plot showing SLD uncertainty after a fit")

        self.action_export_uncertainty = menu.addAction("Export Uncertainty...")
        self.action_export_uncertainty.setToolTip("Export SLD uncertainty after a fit")

        menu.addSeparator()
        self.action_autoupdate_sld = menu.addAction("Autoupdate SLD")
        self.action_autoupdate_sld.setCheckable(True)
        self.action_autoupdate_sld.setChecked(False)

        self.action_export_sld.triggered.connect(self.OnExportSLD)
        self.action_generate_uncertainty.triggered.connect(self.OnGenerateUncertainty)
        self.action_export_uncertainty.triggered.connect(self.OnExportUncertainty)
        self.action_autoupdate_sld.triggered.connect(self.OnAutoUpdateSLD)
        self.action_show_imag.triggered.connect(self.OnShowImagSLD)
        self.action_mass_density.triggered.connect(self.OnShowMassDensity)

        self.parent.model_control.update_script.connect(self.ReadUpdateModel)
        self.StatusMessage("Reflectivity plugin loaded")

    def SetModelScript(self, script):
        framework.Template.SetModelScript(self, script)
        self._last_script = script

    def UpdateScript(self, _event):
        self.WriteModel()

    def OnAutoUpdateSLD(self, _evt):
        pass

    def OnShowImagSLD(self, _evt):
        self.sld_plot.opt.show_imag = self.action_show_imag.isChecked()
        self.sld_plot.WriteConfig()
        self.sld_plot.Plot()

    def OnShowMassDensity(self, _evt):
        self.sld_plot.opt.use_mass_density = self.action_mass_density.isChecked()
        self.sld_plot.WriteConfig()
        self.sld_plot.Plot()

    @property
    def show_imag_sld(self):
        return self.sld_plot.opt.show_imag

    def OnExportSLD(self, _evt):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent,
            "Export SLD to ...",
            "",
            "Dat File (*.dat)",
        )
        if not fname:
            return
        result = True
        if os.path.exists(fname):
            filename = os.path.split(fname)[1]
            result = self.ShowQuestionDialog(
                "The file %s already exists. Do you wish to overwrite it?" % filename
            )
        if result:
            try:
                self.sld_plot.SavePlotData(fname)
            except IOError as exc:
                self.ShowErrorDialog(str(exc))
            except Exception:
                outp = io.StringIO()
                traceback.print_exc(200, outp)
                val = outp.getvalue()
                outp.close()
                self.ShowErrorDialog("Could not save the file. Python Error:\n%s" % (val,))

    def OnGenerateUncertainty(self, _evt):
        validators = {"Reference Surface": ValueValidator(int), "Number of Samples": ValueValidator(int)}
        vals = {"Reference Surface": 0, "Number of Samples": 1000}
        pars = ["Reference Surface", "Number of Samples"]
        dlg = ValidateDialog(self.parent, pars, vals, validators, title="Parameters for uncertainty graph")

        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            vals = dlg.GetValues()
            dlg.close()
        else:
            dlg.close()
            return

        model: Model = self.GetModel()
        model.simulate()
        self.sld_plot.PlotConfidence(int(vals["Reference Surface"]), int(vals["Number of Samples"]))

    def OnExportUncertainty(self, _evt):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.parent,
            "Export SLD to ...",
            "",
            "Dat File (*.dat)",
        )
        if not fname:
            return

        result = True
        if os.path.exists(fname):
            filename = os.path.split(fname)[1]
            result = self.ShowQuestionDialog(
                "The file %s already exists. Do you wish to overwrite it?" % filename
            )
        if not result:
            return

        validators = {"Reference Surface": ValueValidator(int), "Number of Samples": ValueValidator(int)}
        vals = {"Reference Surface": 0, "Number of Samples": 1000}
        pars = ["Reference Surface", "Number of Samples"]
        dlg = ValidateDialog(self.parent, pars, vals, validators, title="Parameters for uncertainty graph")

        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            vals = dlg.GetValues()
            dlg.close()
        else:
            dlg.close()
            return

        model: Model = self.GetModel()
        model.simulate()
        self.sld_plot.SaveConfidenceData(
            fname, int(vals["Reference Surface"]), int(vals["Number of Samples"]), do_plot=True
        )

    def OnNewModel(self, _event):
        selection, ok = QtWidgets.QInputDialog.getItem(
            self.parent,
            "Models",
            "Choose a model type to use",
            avail_models,
            0,
            False,
        )
        if ok and selection:
            self.CreateNewModel("models.%s" % selection)

    def OnDataChanged(self, event):
        if event.new_model:
            return

        if event.data_moved or event.deleted or event.new_data or event.name_change:
            names = [data_set.name for data_set in self.GetModel().get_data()]
            self.simulation_widget.SetDataList(names)

            expl = self.simulation_widget.GetExpressionList()

            if len(names) - len(expl) == 1:
                expl.append([])
                self.insert_new_data_segment(len(expl) - 1)

            sims, insts, args = self.simulation_widget.GetSimArgs()

            if event.deleted:
                pos = list(range(len(expl)))
                [self.remove_data_segment(pos[-index - 1]) for index in range(len(event.position))]
                [expl.pop(index) for index in event.position]
                [sims.pop(index) for index in event.position]
                [insts.pop(index) for index in event.position]
                [args.pop(index) for index in event.position]
            elif event.data_moved:
                if event.up:
                    for pos in event.position:
                        tmp = expl.pop(pos)
                        expl.insert(pos - 1, tmp)
                        tmp = sims.pop(pos)
                        sims.insert(pos - 1, tmp)
                        tmp = insts.pop(pos)
                        insts.insert(pos - 1, tmp)
                        tmp = args.pop(pos)
                        args.insert(pos - 1, tmp)
                else:
                    for pos in event.position:
                        tmp = expl.pop(pos)
                        expl.insert(pos + 1, tmp)
                        tmp = sims.pop(pos)
                        sims.insert(pos + 1, tmp)
                        tmp = insts.pop(pos)
                        insts.insert(pos + 1, tmp)
                        tmp = args.pop(pos)
                        args.insert(pos + 1, tmp)

            self.simulation_widget.SetSimArgs(sims, insts, args)
            self.simulation_widget.SetExpressionList(expl)

            if self.GetModel().script != "":
                self.WriteModel()
                self.simulation_widget.UpdateListbox()
                if event.name_change:
                    self.sld_plot.Plot()
            else:
                self.simulation_widget.UpdateListbox(update_script=True)
        else:
            if event.data_changed:
                self.sld_plot.Plot()

    def OnOpenModel(self, _event):
        self.action_show_imag.setChecked(self.sld_plot.opt.show_imag)
        self.ReadModel()

    def OnSimulate(self, _event):
        if not self.action_autoupdate_sld.isChecked():
            QtCore.QTimer.singleShot(0, self.sld_plot.Plot)

    def OnFittingUpdate(self, _event):
        if self.action_autoupdate_sld.isChecked():
            QtCore.QTimer.singleShot(0, self.sld_plot.Plot)

    def OnGridChange(self, _event):
        self.sample_widget.Update(update_script=False)

    def InstrumentNameChange(self, old_name, new_name):
        self.simulation_widget.InstrumentNameChange(old_name, new_name)

    def CreateNewModel(self, modelname="models.spec_nx"):
        model_data = self.GetModel().get_data()
        nb_data_sets = len(model_data)

        script = self.GetNewModelScript(modelname, nb_data_sets)
        self.BuildNewModel(script)

        self.sample_widget.set_sampleh(self.sampleh)
        self.sample_widget.set_model(self.model)
        instrument = self.model.Instrument()
        self.sample_widget.SetInstrument({"inst": instrument})

        names = [data_set.name for data_set in model_data]
        self.simulation_widget.SetDataList(names)
        self.simulation_widget.SetParameterList([])
        self.simulation_widget.SetExpressionList([[] for _item in names])
        self.simulation_widget.SetSimArgs(
            ["Specular"] * nb_data_sets, ["inst"] * nb_data_sets, [["d.x"] for _i in range(nb_data_sets)]
        )
        self.simulation_widget.UpdateListbox(update_script=True)

        self.sample_widget.Update(update_script=True)

    def WriteModel(self):
        parameter_list = self.simulation_widget.GetParameterList()
        sim_funcs, sim_insts, sim_args = self.simulation_widget.GetSimArgs()
        expression_list = self.simulation_widget.GetExpressionList()
        instruments = self.sample_widget.instruments

        self.write_model_script(sim_funcs, sim_insts, sim_args, expression_list, parameter_list, instruments)
        try:
            self.SetXAxis(instruments[sim_insts[0]])
        except AttributeError:
            pass

    def SetXAxis(self, instrument):
        if self.previous_xaxis == instrument.coords:
            return
        coords = instrument.coords
        from genx import data

        if coords == "q":
            data.DataSet.simulation_params[0] = 0.001
            data.DataSet.simulation_params[1] = 0.601
        else:
            data.DataSet.simulation_params[0] = 0.01
            data.DataSet.simulation_params[1] = 6.01
        for ds in self.GetModel().data:
            ds.run_command()

    def AppendSim(self, sim_func, inst, args):
        self.simulation_widget.AppendSim(sim_func, inst, args)

    def ReadUpdateModel(self, *_args):
        try:
            self.ReadModel(verbose=False)
        except GenxError:
            pass
        except Exception as exc:
            self.StatusMessage(f"could not analyze script: {exc}")

    def ReadModel(self, reevaluate=False, verbose=True):
        if verbose:
            self.StatusMessage("Compiling the script...")
        self.CompileScript()
        if verbose:
            self.StatusMessage("Script compiled!")

        if verbose:
            self.StatusMessage("Trying to interpret the script...")

        instrument_names = self.find_instrument_names()

        if len(instrument_names) == 0:
            self.ShowErrorDialog("Could not find any Instruments in the model script. Check the script.")
            self.StatusMessage("ERROR No Instruments in script")
            return

        if "inst" not in instrument_names:
            self.ShowErrorDialog(
                "Could not find the default Instrument, inst, in the model script. Check the script."
            )
            self.StatusMessage("ERROR No Instrument called inst in script")
            return

        sample_text = self.find_sample_section()

        if sample_text is None:
            self.ShowErrorDialog(
                "Could not find the sample section in the model script.\nCan not load the sample in the editor."
            )
            self.StatusMessage("ERROR No sample section in script")
            return

        all_names, layers, stacks = self.find_layers_stacks(sample_text)

        if len(layers) == 0:
            self.ShowErrorDialog("Could not find any Layers in the model script. Check the script.")
            self.StatusMessage("ERROR No Layers in script")
            return

        for lay in layers:
            for par in lay[1].split(","):
                vars = par.split("=")
                exec(
                    '%s.%s = "%s"' % (lay[0], vars[0].strip(), vars[1].strip()),
                    self.GetModel().script_module.__dict__,
                )
        try:
            data_names, insts, sim_args, sim_exp, sim_funcs = self.find_sim_function_parameters()
        except LookupError:
            self.ShowErrorDialog(
                "Could not locate all data sets in the script. There should be %i datasets"
                % len(self.GetModel().get_data())
            )
            self.StatusMessage("ERROR No Layers in script")
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
        self.GetModel().compiled = False
        if verbose:
            if reevaluate:
                self.StatusMessage("Model analyzed and plugin updated!")
            else:
                self.StatusMessage("New sample loaded to plugin!")
        self._last_script = self.model_obj.script

        try:
            self.SetXAxis(self.sample_widget.instruments[instrument_names[0]])
        except AttributeError:
            pass
