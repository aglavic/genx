'''<h1> SXRD plugin </h1>
This plugin auto generates the sample definition and simulations of a surface x-ray diffraction experiment.

'''

__author__ = 'Matts Bjorck'

import importlib

import wx

import plugins.add_on_framework as framework
import help_modules.model_interactors as mi

code = """
        # BEGIN Instruments
        inst = model.Instrument(wavel=0.64, alpha=0.5, geom="alpha_in fixed")
        # END Instruments

        # BEGIN UnitCells
        uc = model.UnitCell(a=6, b=6, c=6, alpha=45, beta=45, gamma=45)
        # END UnitCells

        # BEGIN Slabs
        slab = model.Slab(c=1.0)
        slab.add_atom(id='al', el='Al', x=0.5, y=0.5, z=0.5, u=1.0, m=1, oc=1.0)
        slab.add_atom(id='la', el='La', x=0.0, y=0.0, z=0.0, u=0.05, m=4, oc=0.5)

        slab2 = model.Slab(c=1.0)
        slab2.add_atom(id='Y', el='Y', x=0.5, y=0.5, z=0.5, u=1.0, m=1, oc=1.0)
        slab2.add_atom(id='Cu1+', el='Cu', x=0.0, y=0.0, z=0.0, u=0.05, m=4, oc=0.5)
        # END Slabs

        # BEGIN Domains
        domain1 = model.Domain(slab, [slab, slab2], uc, occ=1.0)
        domain2 = model.Domain(slab, [slab, slab, slab, slab2, slab2], uc, occ=1.0)
        # END Domains

        # BEGIN Samples
        sample = model.Sample([domain1, domain2], cohf=0.0)
        # END Samples

        def Sim(data):
            I = []
            # BEGIN DataSet 0
            d = data[0]
            I.append(sample.calc_i(inst, d.h, d.k, d.l))
            # END DataSet 0
            return I
        """

class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.setup_script_interactor()

        self.layout_sample_edit()
        self.layout_simulation_edit()

        self.update_script()

    def setup_script_interactor(self, model_name='sxrd'):
        """Setup the script interactor"""
        model = __import__('models.%s' % model_name, globals(), locals(), [model_name], -1)
        print dir(model)
        script_interactor = mi.ModelScriptInteractor(preamble='import models.%s as model\n' % model_name)

        script_interactor.add_section('Instruments', mi.ObjectScriptInteractor, class_name='model.Instrument',
                                      class_impl=model.Instrument)
        script_interactor.add_section('UnitCells', mi.ObjectScriptInteractor, class_name='model.UnitCell',
                                      class_impl=model.UnitCell)
        script_interactor.add_section('Slabs', mi.SlabInteractor, class_name='model.Slab', class_impl=model.Slab)
        script_interactor.add_section('Domains', mi.DomainInteractor, class_name='model.Domain', class_impl=model.Domain)
        script_interactor.add_section('Samples', mi.SampleInteractor, class_name='model.Sample', class_impl=model.Sample)

        script_interactor.parse_code(code)
        self.script_interactor = script_interactor

    def layout_sample_view(self):
        """Layouts the sample_view_panel"""
        panel = self.NewPlotFolder('Test')


    def layout_sample_edit(self):
        """Layouts the sample_edit_panel"""
        panel = self.NewInputFolder('Sample')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel.SetSizer(sizer)
        self.sample_edit_widget = mi.DomainListCtrl(panel, domain_list=self.script_interactor.domains,
                                                    slab_list=self.script_interactor.slabs,
                                                    unitcell_list=self.script_interactor.unitcells)
        sizer.Add(self.sample_edit_widget, 1, wx.EXPAND)
        panel.Layout()

    def layout_simulation_edit(self):
        """Layouts the simulation_edit_panel"""
        panel = self.NewInputFolder('Simulations')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel.SetSizer(sizer)
        self.simulation_edit_widget = mi.SimulationListCtrl(panel, self, self.script_interactor)
        sizer.Add(self.simulation_edit_widget, 1, wx.EXPAND)
        panel.Layout()

    def layout_misc_edit(self):
        """Layouts the misc_edit_panel"""
        self.misc_object_edit = self.NewDataFolder('UnitCell')

    def create_main_window_menu(self):
        """Creates the window menu"""
        self.menu = self.NewMenu('SXRD')

    def update_script(self):
        """Updates the script with new data"""
        old_script = self.GetModelScript()
        if old_script == "":
            self.SetModelScript(self.script_interactor.get_code())
        else:
            try:
                self.SetModelScript(self.script_interactor.update_code(old_script))
            except Exception, e:
                if self.ShowQuestionDialog('Could not update the script due to syntax issues. Python error: %s\n\n'
                                            'Do you wish to reset the model to the one defined in the user interface?'):
                    self.SetModelScript(self.script_interactor.get_code())


    def OnNewModel(self, event):
        """Callback for creating a new model"""
        self.update_script()

    def OnOpenModel(self, event):
        """Callback for opening a model"""
        pass

    def OnDataChanged(self, event):
        """Callback for changing of the data sets (dataset added or removed)"""
        pass

    def OnSimulate(self, event):
        """Callback called after simulation"""
        pass



