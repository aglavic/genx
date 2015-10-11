'''<h1> SXRD plugin </h1>
This plugin auto generates the sample definition and simulations of a surface x-ray diffraction experiment.

'''

__author__ = 'Matts Bjorck'

import StringIO
import traceback

import wx

import plugins.add_on_framework as framework
import help_modules.model_interactors as mi
import help_modules.atom_viewer as atom_viewer


code = """
        # BEGIN Instruments
        inst = model.Instrument(wavel=0.64, alpha=0.5, geom="alpha_in fixed")
        # END Instruments

        # BEGIN UnitCells
        uc = model.UnitCell(a=6, b=6, c=6, alpha=90, beta=90, gamma=90)
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
        domain1 = model.Domain(slab, [slab, slab2], uc, occ=1.0, surface_sym=p1, bulk_sym=p1)
        domain2 = model.Domain(slab, [slab, slab, slab, slab2, slab2], uc, occ=1.0, surface_sym=p1, bulk_sym=p1)
        # END Domains

        # BEGIN Samples
        sample = model.Sample([domain1, domain2], cohf=0.0)
        # END Samples

        # BEGIN Parameters
        cp = UserVars()
        cp.new_var('test', 0.0)
        # END Parameters

        def Sim(data):
            I = []
            # BEGIN DataSet 0
            d = data[0]
            I.append(sample.calc_i(inst, 0, 0, d.x))
            # END DataSet 0
            return I
        """

class Plugin(framework.Template):
    def __init__(self, parent):
        #print "__init__ SXRD Plugin"
        framework.Template.__init__(self, parent)
        self.setup_script_interactor()
        if self.GetModelScript() == '':
            self.SetModelScript(self.script_interactor.get_code())
        else:
            self.script_interactor.parse_code(self.GetModelScript())

        self.layout_sample_edit()
        self.layout_simulation_edit()
        self.layout_misc_edit()
        self.layout_domain_viewer()
        self.create_main_window_menu()


        self.OnInteractorChanged(None)
        self.update_data_names()
        self.simulation_edit_widget.Update()
        self.update_widgets()

    def setup_script_interactor(self, model_name='sxrd2'):
        """Setup the script interactor"""
        model = __import__('models.%s' % model_name, globals(), locals(), [model_name], -1)
        preamble = "import models.%s as model\nfrom models.utils import UserVars\nfrom models.symmetries import *\n" % model_name
        script_interactor = mi.ModelScriptInteractor(preamble=preamble)

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
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        self.sample_edit_widget = mi.DomainListCtrl(panel, domain_list=self.script_interactor.domains,
                                                    slab_list=self.script_interactor.slabs,
                                                    unitcell_list=self.script_interactor.unitcells)
        panel.Bind(mi.EVT_INTERACTOR_CHANGED, self.OnInteractorChanged, self.sample_edit_widget)
        sizer.Add(self.sample_edit_widget, 1, wx.EXPAND)
        panel.Layout()

    def layout_sample_edit(self):
        """Layouts the sample_edit_panel"""
        panel = self.NewInputFolder('Sample')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel.SetSizer(sizer)
        self.sample_edit_widget = mi.DomainListCtrl(panel, domain_list=self.script_interactor.domains,
                                                    slab_list=self.script_interactor.slabs,
                                                    unitcell_list=self.script_interactor.unitcells,
                                                    sample_list=self.script_interactor.samples)
        panel.Bind(mi.EVT_INTERACTOR_CHANGED, self.OnInteractorChanged, self.sample_edit_widget)
        panel.Bind(mi.EVT_SELECTION_CHANGED, self.OnSelectionChanged, self.sample_edit_widget)

        sizer.Add(self.sample_edit_widget, 1, wx.EXPAND)
        panel.Layout()

    def layout_simulation_edit(self):
        """Layouts the simulation_edit_panel"""
        panel = self.NewInputFolder('Simulations')
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel.SetSizer(sizer)
        self.simulation_edit_widget = mi.SimulationListCtrl(panel, self, self.script_interactor)
        panel.Bind(mi.EVT_INTERACTOR_CHANGED, self.OnInteractorChanged, self.simulation_edit_widget)
        sizer.Add(self.simulation_edit_widget, 1, wx.EXPAND)
        panel.Layout()

    def layout_misc_edit(self):
        """Layouts the misc_edit_panel"""
        panel = self.NewDataFolder('Misc')
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        # Make the box for putting in the columns
        col_box = wx.StaticBox(panel, -1, "Unit Cells")
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL)
        sizer.Add(col_box_sizer, 1, wx.EXPAND)
        self.unitcell_edit_widget = mi.EditList(panel, object_list=self.script_interactor.unitcells,
                                                default_name='Unitcells',
                                                edit_dialog=mi.ObjectDialog, edit_dialog_name='Unitcell Editor')
        panel.Bind(mi.EVT_INTERACTOR_CHANGED, self.OnInteractorChanged, self.unitcell_edit_widget)
        col_box_sizer.Add(self.unitcell_edit_widget, 1, wx.EXPAND)
        # Make the box for putting in the columns
        col_box = wx.StaticBox(panel, -1, "Instruments")
        col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL)
        sizer.Add(col_box_sizer, 1, wx.EXPAND)
        self.instrument_edit_widget = mi.EditList(panel, object_list=self.script_interactor.instruments,
                                                  default_name='Instruments',
                                                  edit_dialog=mi.ObjectDialog, edit_dialog_name='Instrument Editor')
        panel.Bind(mi.EVT_INTERACTOR_CHANGED, self.OnInteractorChanged, self.instrument_edit_widget)
        col_box_sizer.Add(self.instrument_edit_widget, 1, wx.EXPAND)
        panel.Layout()

    def layout_domain_viewer(self):
        """Creates a 3D view of the sample."""
        panel = self.NewPlotFolder('Sample view')
        sample_view_sizer = wx.BoxSizer(wx.HORIZONTAL)
        panel.SetSizer(sample_view_sizer)
        self.sample_view = atom_viewer.VTKview(panel)

        sample_view_sizer.Add(self.sample_view,
                              1, wx.EXPAND|wx.GROW|wx.ALL)
        panel.Layout()

        # Just to init the view properly
        cur_page = self.parent.plot_notebook.Selection
        self.parent.plot_notebook.SetSelection(self.parent.plot_notebook.GetPageCount() - 1)
        self.parent.plot_notebook.SetSelection(cur_page)
        self.sample_view.show()

    def create_main_window_menu(self):
        """Creates the window menu"""
        self.menu = self.NewMenu('SXRD')
        menu_item = wx.MenuItem(self.menu, wx.NewId(), "Domain Viewer Settings...", "Edit Viewer settings",
                                wx.ITEM_NORMAL)
        self.menu.AppendItem(menu_item)
        self.parent.Bind(wx.EVT_MENU, self.OnDomainViewerSettings, menu_item)

    def update_script(self):
        """Updates the script with new data"""
        old_script = self.GetModelScript()
        if old_script == "":
            self.SetModelScript(self.script_interactor.get_code())
        else:
            try:
                self.SetModelScript(self.script_interactor.update_code(old_script))
            except Exception, e:
                outp = StringIO.StringIO()
                traceback.print_exc(200, outp)
                tbtext = outp.getvalue()
                outp.close()
                print "Error updating the script: "
                print tbtext
                if self.ShowQuestionDialog('Could not update the script due to syntax issues. Python error: %s\n\n'
                                            'Do you wish to reset the model to the one defined in the user interface?'):
                    self.SetModelScript(self.script_interactor.get_code())

    def update_data_names(self):
        """Updates the DataSetInteractors names from the DataSet names in the model"""
        data_set_list = self.GetModel().data

        assert(len(data_set_list) == len(self.script_interactor.data_sections_interactors))

        for interactor, data_set in zip(self.script_interactor.data_sections_interactors, data_set_list):
            interactor.set_name(data_set.name)

    def update_widgets(self):
        """Collects the already defined names and sets the already taken names in the different controls.
        Updates the symmetries from the script.
        """
        self.CompileScript()
        names = dir(self.GetScriptModule())
        self.instrument_edit_widget.set_taken_names(names[:])
        self.unitcell_edit_widget.set_taken_names(names[:])
        self.sample_edit_widget.set_taken_names(names[:])

        model = self.GetModel()
        symmetries = [obj for obj in names if model.eval_in_model('isinstance(%s, Sym)' % obj)]
        self.sample_edit_widget.set_symmetries(symmetries)

    def set_constant_names(self):
        """Sets the name that needs to constant (used in other defs)"""
        self.unitcell_edit_widget.set_undeletable_names([d.unitcell for d in self.script_interactor.domains])
        self.instrument_edit_widget.set_undeletable_names([ds.instrument for ds in
                                                           self.script_interactor.data_sections_interactors])

    def update_domain_view(self):
        domain = self.sample_edit_widget.get_selected_domain_name()
        if domain:
            try:
                domain = self.GetModel().eval_in_model(domain)
            except Exception:
                print "Could not load domain ", domain
            else:
                self.sample_view.build_sample(domain, use_opacity=False)

    def OnDomainViewerSettings(self, event):
        """Callback for showing the Domain Viewer settings dialog"""
        self.sample_view.ShowSettingDialog()
        self.update_domain_view()

    def OnInteractorChanged(self, event):
        """Callback when an Interactor has been changed by the GUI"""
        self.update_script()
        self.set_constant_names()
        self.update_widgets()
        self.update_domain_view()

    def OnSelectionChanged(self, evnet):
        """Callback when the selection in the sample widget has changed"""
        self.update_domain_view()


    def OnNewModel(self, event):
        """Callback for creating a new model"""
        #print "OnNewModel"
        self.update_script()
        self.update_data_names()
        self.simulation_edit_widget.Update()

    def OnOpenModel(self, event):
        """Callback for opening a model"""
        #print 'OnOpenModel'
        #print self.GetModelScript()
        self.script_interactor.parse_code(self.GetModelScript())
        self.update_data_names()
        self.simulation_edit_widget.Update()
        self.set_constant_names()

    def OnDataChanged(self, event):
        """Callback for changing of the data sets (dataset added or removed)"""
        # We have the following possible events:
        # event.new_model, event.data_moved, event.deleted, event.new_data, event.name_change
        #print "OnDataChanged"
        if event.new_model:
            # If a new model is created bail out
            return

        if event.new_data:
            # New data has been added:
            self.script_interactor.append_dataset()
        elif event.deleted:
            for pos in event.position:
                self.script_interactor.data_sections_interactors.pop(pos)
        elif event.data_moved:
            if event.up:
                for pos in event.position:
                    tmp = self.script_interactor.data_sections_interactors.pop(pos)
                    self.script_interactor.data_sections_interactors.insert(pos - 1, tmp)
            else:
                for pos in event.position:
                    tmp = self.script_interactor.data_sections_interactors.pop(pos)
                    self.script_interactor.data_sections_interactors.insert(pos + 1, tmp)

        self.update_data_names()
        self.simulation_edit_widget.Update()
        self.update_script()


    def OnSimulate(self, event):
        """Callback called after simulation"""
        pass



