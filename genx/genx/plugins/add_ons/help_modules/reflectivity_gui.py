"""
GUI support classes for the Reflectivity plug-in.
"""
import wx.html

from genx.core.custom_logging import iprint
from genx.model import Model
from genx.plugins.utils import ShowQuestionDialog, ShowWarningDialog
from . import reflectivity_images as images
from .custom_dialog import ComplexObjectValidator, FloatObjectValidator, NoMatchValidTextObjectValidator, \
    ParameterExpressionCombo, TextObjectValidator, ValidateDialog, ValidateFitDialog, \
    ValidateFitNotebookDialog
from .reflectivity_misc import ReflectivityModule
from .reflectivity_utils import find_code_segment, SampleHandler

_set_func_prefix='set'

class MyHtmlListBox(wx.html.HtmlListBox):

    def __init__(self, parent, id, size=(-1, -1), style=wx.BORDER_SUNKEN):
        wx.html.HtmlListBox.__init__(self, parent, id, size=size,
                                     style=style)
        self.SetItemList(['Starting up...'])

    def SetItemList(self, list):
        self.html_items = list
        self.SetItemCount(len(list))
        # self.RefreshAll()
        self.Refresh()

    def OnGetItem(self, n):
        return self.html_items[n]


class SamplePanel(wx.Panel):
    sampleh: SampleHandler
    model: ReflectivityModule

    def __init__(self, parent, plugin, refindexlist=None):
        wx.Panel.__init__(self, parent)
        if refindexlist is None:
            refindexlist = []
        self.refindexlist = refindexlist
        self.plugin = plugin
        self.variable_span = 0.25

        # Colours indicating different states
        # Green wx.Colour(138, 226, 52), ORANGE wx.Colour(245, 121, 0)
        self.fit_colour = (245, 121, 0)
        # Tango Sky blue wx.Colour(52, 101, 164), wx.Colour(114, 159, 207)
        self.const_fit_colour = (114, 159, 207)

        boxver = wx.BoxSizer(wx.HORIZONTAL)
        boxhor = wx.BoxSizer(wx.VERTICAL)
        self.toolbar = wx.ToolBar(self, style=wx.TB_FLAT | wx.TB_HORIZONTAL)
        self.do_toolbar()
        boxhor.Add(self.toolbar, proportion=0, flag=wx.EXPAND)
        boxhor.AddSpacer(2)
        self.listbox = MyHtmlListBox(self, -1, style=wx.BORDER_SUNKEN)
        # self.listbox.SetItemList(self.sampleh.getStringList())
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.lbDoubleClick, self.listbox)
        boxhor.Add(self.listbox, 1, wx.EXPAND)

        boxver.Add(boxhor, 1, wx.EXPAND)

        self.SetSizer(boxver)
        self.toolbar.Realize()
        self.update_callback = lambda event: ''

    def do_toolbar(self):
        dpi_scale_factor = wx.GetApp().dpi_scale_factor
        tb_bmp_size = int(dpi_scale_factor*20)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Insert Layer',
                             bitmap=wx.Bitmap(images.insert_layer.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Insert a Layer')
        self.Bind(wx.EVT_TOOL, self.InsertLay, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Insert Stack',
                             bitmap=wx.Bitmap(images.insert_stack.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Insert a Stack')
        self.Bind(wx.EVT_TOOL, self.InsertStack, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Delete',
                             bitmap=wx.Bitmap(images.delete.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Delete item')
        self.Bind(wx.EVT_TOOL, self.DeleteSample, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Rename',
                             bitmap=wx.Bitmap(images.change_name.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Rename')
        self.Bind(wx.EVT_TOOL, self.ChangeName, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Move up',
                             bitmap=wx.Bitmap(images.move_up.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Move item up')
        self.Bind(wx.EVT_TOOL, self.MoveUp, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Move down',
                             bitmap=wx.Bitmap(images.move_down.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Move item down')
        self.Bind(wx.EVT_TOOL, self.MoveDown, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Edit Sample',
                             bitmap=wx.Bitmap(images.sample.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Edit Sample parameters')
        self.Bind(wx.EVT_TOOL, self.EditSampleParameters, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Edit Instrument',
                             bitmap=wx.Bitmap(images.instrument.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Edit Instruments')
        self.Bind(wx.EVT_TOOL, self.EditInstrument, id=newid)

    def SetUpdateCallback(self, func):
        ''' SetUpdateCallback(self, func) --> None

        Sets the update callback will be called when the sample is updated.
        The call is on the form func(event)
        '''
        self.update_callback = func

    def set_sampleh(self, sampleh: SampleHandler):
        self.sampleh = sampleh

    def set_model(self, model: ReflectivityModule):
        self.model = model

    def create_html_decorator(self):
        """
        creates a html decorator function
        :return:
        """
        grid_parameters = self.plugin.GetModel().get_parameters()
        dic_lookup = {}
        for par in grid_parameters.get_names():
            l = par.split('.')
            if len(l)==2:
                name = l[0]
                par_name = l[1][3:].lower()
                dic_lookup[(name, par_name)] = (grid_parameters.get_value_by_name(par),
                                                grid_parameters.get_fit_state_by_name(par)
                                                )
        fit_color_str = "rgb(%d,%d,%d)"%self.fit_colour
        const_fit_color_str = "rgb(%d,%d,%d)"%self.const_fit_colour

        def decorator(name, str):
            """ Decorator to indicate the parameters that are fitted"""
            try:
                start_index = str.index('(')+1
            except ValueError:
                start_index = 0
            ret_str = str[:start_index]
            for par_str in str[start_index:].split(','):
                par_name = par_str.split('=')[0].strip()
                # par_name normal paramter (real number)
                if (name, par_name) in dic_lookup:
                    val, state = dic_lookup[(name, par_name)]
                    if state==1:
                        par_str = ' <font color=%s><b>%s=%.2e</b></font>,'%(fit_color_str, par_name, val)
                    elif state==2:
                        par_str = ' <font color=%s><b>%s=%.2e</b></font>,'%(const_fit_color_str, par_name, val)
                # par_name is a complex parameter...
                elif (name, par_name+'real') in dic_lookup or (name, par_name+'imag') in dic_lookup:
                    if (name, par_name+'real') in dic_lookup:
                        val, state = dic_lookup[(name, par_name+'real')]
                        if state==1:
                            par_str = ' <font color=%s><b>%s=(%.2e,</b></font>'%(fit_color_str, par_name, val)
                        elif state==2:
                            par_str = ' <font color=%s><b>%s=(%.2e,</b></font>'%(const_fit_color_str, par_name, val)
                    else:
                        par_str = ' <b>%s=??+</b>'%par_name
                    if (name, par_name+'imag') in dic_lookup:
                        val, state = dic_lookup[(name, par_name+'imag')]
                        if state==1:
                            par_str += ' <font color=%s><b>%.2e)</b></font>,'%(fit_color_str, val)
                        elif state==2:
                            par_str += ' <font color=%s><b>%.2e)</b></font>,'%(const_fit_color_str, val)
                    else:
                        par_str += ' <b>??)</b>,'

                else:
                    par_str += ','
                ret_str += par_str
            # Remove trailing ,
            if ret_str[-1]==',':
                ret_str = ret_str[:-1]
            if str[-1]==')' and ret_str[-1]!=')':
                ret_str += ')'
            return ret_str

        return decorator

    def Update(self, update_script=True):
        deco = self.create_html_decorator()
        sl = self.sampleh.getStringList(html_encoding=True, html_decorator=deco)
        self.listbox.SetItemList(sl)
        if update_script:
            self.update_callback(None)

    def SetSample(self, sample, names):
        self.sampleh.sample = sample
        self.sampleh.names = names
        self.Update()

    def EditSampleParameters(self, evt):
        """ Event handler that creates a dialog box to edit the sample parameters.

        :param evt:
        :return: Nothing
        """
        obj_name = 'sample'
        eval_func = self.plugin.GetModel().eval_in_model
        grid_parameters = self.plugin.GetModel().get_parameters()

        validators = {}
        vals = {}
        pars = []
        items = []
        editable = {}
        try:
            string_choices = self.model.sample_string_choices
        except Exception as e:
            string_choices = {}
        for item in self.model.SampleParameters:
            if item!='Stacks' and item!='Substrate' and item!='Ambient':
                if item in string_choices:
                    validators[item] = string_choices[item]
                else:
                    validators[item] = FloatObjectValidator()
                val = getattr(self.sampleh.sample, item)
                vals[item] = val
                pars.append(item)
                items.append((item, val))
                # Check if the parameter is in the grid and in that case set it as uneditable
                func_name = obj_name+'.'+_set_func_prefix+item.capitalize()
                grid_value = grid_parameters.get_value_by_name(func_name)
                editable[item] = grid_parameters.get_fit_state_by_name(func_name)
                if grid_value is not None:
                    vals[item] = grid_value
        try:
            groups = self.model.SampleGroups
        except Exception:
            groups = False
        try:
            units = self.model.SampleUnits
        except Exception:
            units = False

        dlg = ValidateFitDialog(self, pars, vals, validators,
                                title='Sample Editor', groups=groups,
                                units=units, editable_pars=editable)

        if dlg.ShowModal()==wx.ID_OK:
            old_vals = vals
            vals = dlg.GetValues()
            # print vals
            states = dlg.GetStates()
            for par in pars:
                if not states[par]:
                    old_type = type(old_vals[par])
                    setattr(self.sampleh.sample, par, old_type(vals[par]))
                if editable[par]!=states[par]:
                    value = eval_func(vals[par])
                    minval = min(value*(1-self.variable_span), value*(1+self.variable_span))
                    maxval = max(value*(1-self.variable_span), value*(1+self.variable_span))
                    func_name = obj_name+'.'+_set_func_prefix+par.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value, states[par], minval, maxval)
                    # Tell the grid to reload the parameters
                    self.plugin.parent.paramter_grid.SetParameters(grid_parameters)

            self.Update()

        dlg.Destroy()

    def SetInstrument(self, instruments):
        '''SetInstrument(self, instrument) --> None

        Sets the instruments should be a dictionary of instruments with the key being the
        name of the instrument
        '''
        self.instruments = instruments

    def EditInstrument(self, evt):
        """Event handler that creates an dialog box to edit the instruments.

        :param evt:
        :return: Nothing
        """
        eval_func = self.plugin.GetModel().eval_in_model
        validators = {}
        vals = {}
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()
        for inst_name in self.instruments:
            vals[inst_name] = {}
            editable[inst_name] = {}

        pars = []
        for item in self.model.InstrumentParameters:
            if item in self.model.instrument_string_choices:
                # validators.append(self.model.instrument_string_choices[item])
                validators[item] = self.model.instrument_string_choices[item]
            else:
                # validators.append(FloatObjectValidator())
                validators[item] = FloatObjectValidator()
            for inst_name in self.instruments:
                val = getattr(self.instruments[inst_name], item)
                vals[inst_name][item] = val
                # Check if the parameter is in the grid and in that case set it as uneditable
                func_name = inst_name+'.'+_set_func_prefix+item.capitalize()
                grid_value = grid_parameters.get_value_by_name(func_name)
                editable[inst_name][item] = grid_parameters.get_fit_state_by_name(func_name)
                if grid_value is not None:
                    vals[inst_name][item] = grid_value
            pars.append(item)

        old_insts = []
        for inst_name in self.instruments:
            old_insts.append(inst_name)

        try:
            groups = self.model.InstrumentGroups
        except Exception:
            groups = False
        try:
            units = self.model.InstrumentUnits
        except Exception:
            units = False
        dlg = ValidateFitNotebookDialog(self, pars, vals, validators,
                                        title='Instrument Editor', groups=groups,
                                        units=units, fixed_pages=['inst'], editable_pars=editable)

        if dlg.ShowModal()==wx.ID_OK:
            old_vals = vals
            vals = dlg.GetValues()
            # print vals
            states = dlg.GetStates()
            self.instruments = {}
            for inst_name in vals:
                new_instrument = False
                if inst_name not in self.instruments:
                    # A new instrument must be created:
                    self.instruments[inst_name] = self.model.Instrument()
                    new_instrument = True
                for par in self.model.InstrumentParameters:
                    if not states[inst_name][par]:
                        old_type = type(old_vals[inst_name][par])
                        if old_type is str:
                            e_value = vals[inst_name][par]
                        else:
                            e_value = eval_func(vals[inst_name][par])
                        setattr(self.instruments[inst_name], par, old_type(e_value))
                    else:
                        setattr(self.instruments[inst_name], par, old_vals[inst_name][par])
                    if new_instrument and states[inst_name][par]>0:
                        value = eval_func(vals[inst_name][par])
                        minval = min(value*(1-self.variable_span), value*(1+self.variable_span))
                        maxval = max(value*(1-self.variable_span), value*(1+self.variable_span))
                        func_name = inst_name+'.'+_set_func_prefix+par.capitalize()
                        grid_parameters.set_fit_state_by_name(func_name, value, states[inst_name][par], minval, maxval)
                    elif not new_instrument:
                        if editable[inst_name][par]!=states[inst_name][par]:
                            value = eval_func(vals[inst_name][par])
                            minval = min(value*(1-self.variable_span), value*(1+self.variable_span))
                            maxval = max(value*(1-self.variable_span), value*(1+self.variable_span))
                            func_name = inst_name+'.'+_set_func_prefix+par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name, value, states[inst_name][par], minval,
                                                                  maxval)

            # Loop to remove instrument from grid if not returned from Dialog
            for inst_name in old_insts:
                if inst_name not in list(vals.keys()):
                    for par in self.model.InstrumentParameters:
                        if editable[inst_name][par]>0:
                            func_name = inst_name+'.'+_set_func_prefix+par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name, 0, 0, 0, 0)

            # Tell the grid to reload the parameters
            self.plugin.parent.paramter_grid.SetParameters(grid_parameters)

            for change in dlg.GetChanges():
                if change[0]!='' and change[1]!='':
                    self.plugin.InstrumentNameChange(change[0], change[1])
                elif change[1]=='':
                    self.plugin.InstrumentNameChange(change[0], 'inst')

            self.Update()
        else:
            pass
        dlg.Destroy()

    def MoveUp(self, evt):
        sl = self.sampleh.moveUp(self.listbox.GetSelection())
        if sl:
            self.Update()
            self.listbox.SetSelection(self.listbox.GetSelection()-1)

    def MoveDown(self, evt):
        sl = self.sampleh.moveDown(self.listbox.GetSelection())
        if sl:
            self.Update()
            self.listbox.SetSelection(self.listbox.GetSelection()+1)

    def InsertStack(self, evt):
        # Create Dialog box
        validators = {'Name': NoMatchValidTextObjectValidator(self.sampleh.names)}
        vals = {}
        pars = ['Name']
        vals['Name'] = 'name'
        dlg = ValidateDialog(self, pars, vals, validators,
                             title='Give Stack Name')

        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
            vals = dlg.GetValues()
        dlg.Destroy()
        # if not a value is selected operate on first
        pos = max(self.listbox.GetSelection(), 0)
        sl = self.sampleh.insertItem(pos, 'Stack', vals['Name'])
        if sl:
            self.Update()
        else:
            self.plugin.ShowWarningDialog('Can not insert a stack at the'
                                          ' current position.')

    def InsertLay(self, evt):
        # Create Dialog box
        # items = [('Name', 'name')]
        # validators = [NoMatchValidTextObjectValidator(self.sampleh.names)]
        dlg = ValidateDialog(self, ['Name'], {'Name': 'name'},
                             {'Name': NoMatchValidTextObjectValidator(self.sampleh.names)},
                             title='Give Layer Name')
        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
            vals = dlg.GetValues()
        else:
            vals = {'Name': 'name'}
        dlg.Destroy()
        # if not a value is selected operate on first
        pos = max(self.listbox.GetSelection(), 0)
        # Create the Layer
        sl = self.sampleh.insertItem(pos, 'Layer', vals['Name'])
        if sl:
            self.Update()
        else:
            self.plugin.ShowWarningDialog('Can not insert a layer at the'
                                          ' current position. Layers has to be part of a stack.')

    def DeleteSample(self, evt):
        slold = self.sampleh.getStringList()
        sl = self.sampleh.deleteItem(self.listbox.GetSelection())
        if sl:
            self.Update()

    def ChangeName(self, evt):
        '''Change the name of the current selected item.
        '''
        pos = self.listbox.GetSelection()
        if pos==0 or pos==len(self.sampleh.names)-1:
            self.plugin.ShowInfoDialog('It is forbidden to change the'
                                       'name of the substrate (Sub) and the Ambient (Amb) layers.')
        else:
            unallowed_names = self.sampleh.names[:pos]+ \
                              self.sampleh.names[max(0, pos-1):]
            dlg = ValidateDialog(self, ['Name'], {'Name': self.sampleh.names[pos]},
                                 {'Name': NoMatchValidTextObjectValidator(unallowed_names)},
                                 title='Give New Name')

            if dlg.ShowModal()==wx.ID_OK:
                vals = dlg.GetValues()
                result = self.sampleh.changeName(pos, vals['Name'])
                if result:
                    self.Update()
                else:
                    iprint('Unexpected problems when changing name...')
            dlg.Destroy()

    def lbDoubleClick(self, evt):
        sel = self.sampleh.getItem(self.listbox.GetSelection())
        obj_name = self.sampleh.getName(self.listbox.GetSelection())
        eval_func = self.plugin.GetModel().eval_in_model
        sl = None
        items = []
        validators = {}
        vals = {}
        pars = []
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()
        if isinstance(sel, self.model.Layer):
            # The selected item is a Layer
            for item in list(self.model.LayerParameters.keys()):
                value = getattr(sel, item)
                vals[item] = value
                # if item!='n' and item!='fb':
                if type(self.model.LayerParameters[item])!=type(1+1.0J):
                    # Handle real parameters
                    validators[item] = FloatObjectValidator(eval_func, alt_types=[self.model.Layer])
                    func_name = obj_name+'.'+_set_func_prefix+item.capitalize()
                    grid_value = grid_parameters.get_value_by_name(func_name)
                    if grid_value is not None:
                        vals[item] = grid_value
                    editable[item] = grid_parameters.get_fit_state_by_name(func_name)

                else:
                    # Handle complex parameters
                    validators[item] = ComplexObjectValidator(eval_func, alt_types=[self.model.Layer])
                    func_name = obj_name+'.'+_set_func_prefix+item.capitalize()
                    grid_value_real = grid_parameters.get_value_by_name(func_name+'real')
                    grid_value_imag = grid_parameters.get_value_by_name(func_name+'imag')
                    if grid_value_real is not None:
                        v = eval_func(vals[item]) if type(vals[item]) is str else vals[item]
                        vals[item] = grid_value_real+v.imag*1.0J
                    if grid_value_imag is not None:
                        v = eval_func(vals[item]) if type(vals[item]) is str else vals[item]
                        vals[item] = v.real+grid_value_imag*1.0J
                    editable[item] = max(grid_parameters.get_fit_state_by_name(func_name+'real'),
                                         grid_parameters.get_fit_state_by_name(func_name+'imag'))

                items.append((item, value))
                pars.append(item)

                # Check if the parameter is in the grid and in that case set it as uneditable
                # func_name = obj_name + '.' + _set_func_prefix + item.capitalize()
                # grid_value = grid_parameters.get_value_by_name(func_name)
                # editable[item] = grid_parameters.get_fit_state_by_name(func_name)

            try:
                groups = self.model.LayerGroups
            except Exception:
                groups = False
            try:
                units = self.model.LayerUnits
            except Exception:
                units = False

            dlg = ValidateFitDialog(self, pars, vals, validators,
                                    title='Layer Editor', groups=groups,
                                    units=units, editable_pars=editable)

            if dlg.ShowModal()==wx.ID_OK:
                vals = dlg.GetValues()
                states = dlg.GetStates()
                for par in list(self.model.LayerParameters.keys()):
                    if not states[par]:
                        setattr(sel, par, vals[par])
                    if editable[par]!=states[par]:
                        value = eval_func(vals[par])

                        if type(self.model.LayerParameters[par]) is complex:
                            # print type(value)
                            func_name = obj_name+'.'+_set_func_prefix+par.capitalize()+'real'
                            val = value.real
                            minval = min(val*(1-self.variable_span), val*(1+self.variable_span))
                            maxval = max(val*(1-self.variable_span), val*(1+self.variable_span))
                            grid_parameters.set_fit_state_by_name(func_name, val, states[par], minval, maxval)
                            val = value.imag
                            minval = min(val*(1-self.variable_span), val*(1+self.variable_span))
                            maxval = max(val*(1-self.variable_span), val*(1+self.variable_span))
                            func_name = obj_name+'.'+_set_func_prefix+par.capitalize()+'imag'
                            grid_parameters.set_fit_state_by_name(func_name, val, states[par], minval, maxval)
                        else:
                            val = value
                            minval = min(val*(1-self.variable_span), val*(1+self.variable_span))
                            maxval = max(val*(1-self.variable_span), val*(1+self.variable_span))
                            func_name = obj_name+'.'+_set_func_prefix+par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name, value, states[par], minval, maxval)

                        # Does not seem to be necessary
                        self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
                sl = self.sampleh.getStringList()
            dlg.Destroy()

        else:
            # The selected item is a Stack
            for item in list(self.model.StackParameters.keys()):
                if item!='Layers':
                    value = getattr(sel, item)
                    if isinstance(value, float):
                        validators[item] = FloatObjectValidator(eval_func, alt_types=[self.model.Stack])
                    else:
                        validators[item] = TextObjectValidator()
                    items.append((item, value))
                    pars.append(item)
                    vals[item] = value

                    # Check if the parameter is in the grid and in that case set it as uneditable
                    func_name = obj_name+'.'+_set_func_prefix+item.capitalize()
                    grid_value = grid_parameters.get_value_by_name(func_name)
                    editable[item] = grid_parameters.get_fit_state_by_name(func_name)
                    if grid_value is not None:
                        vals[item] = grid_value

            try:
                groups = self.model.StackGroups
            except Exception:
                groups = False
            try:
                units = self.model.StackUnits
            except Exception:
                units = False

            dlg = ValidateFitDialog(self, pars, vals, validators,
                                    title='Layer Editor', groups=groups,
                                    units=units, editable_pars=editable)
            if dlg.ShowModal()==wx.ID_OK:
                vals = dlg.GetValues()
                states = dlg.GetStates()
                for par in pars:
                    if not states[par]:
                        setattr(sel, par, vals[par])
                    if editable[par]!=states[par]:
                        value = eval_func(vals[par])
                        minval = min(value*(1-self.variable_span), value*(1+self.variable_span))
                        maxval = max(value*(1-self.variable_span), value*(1+self.variable_span))
                        func_name = obj_name+'.'+_set_func_prefix+par.capitalize()
                        grid_parameters.set_fit_state_by_name(func_name, value, states[par], minval, maxval)
                        # Does not seem to be necessary
                        self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
                sl = self.sampleh.getStringList()

            dlg.Destroy()

        if sl:
            self.Update()


class DataParameterPanel(wx.Panel):
    ''' Widget that defines parameters coupling and different parameters
    for different data sets.
    '''

    def __init__(self, parent, plugin):
        wx.Panel.__init__(self, parent)
        self.plugin = plugin
        boxver = wx.BoxSizer(wx.VERTICAL)
        # Indention for a command - used to seperate commands and data
        self.command_indent = '<pre>   '
        self.script_update_func = None
        self.parameterlist = []

        self.toolbar = wx.ToolBar(self, style=wx.TB_FLAT | wx.TB_HORIZONTAL)
        self.do_toolbar()
        boxver.Add(self.toolbar, proportion=0, flag=wx.EXPAND, border=1)
        boxver.AddSpacer(4)

        self.listbox = MyHtmlListBox(self, -1, style=wx.BORDER_SUNKEN)
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.Edit, self.listbox)
        boxver.Add(self.listbox, 1, wx.EXPAND)

        self.SetSizer(boxver)

    def do_toolbar(self):
        dpi_scale_factor = wx.GetApp().dpi_scale_factor
        tb_bmp_size = int(dpi_scale_factor*20)

        button_names = ['Insert', 'Delete', 'User Variables']
        button_images = [wx.Bitmap(images.add.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                         wx.Bitmap(images.delete.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                         wx.Bitmap(images.custom_parameters.GetImage().Scale(tb_bmp_size, tb_bmp_size))]
        callbacks = [self.Insert, self.Delete, self.EditPars]
        tooltips = ['Insert a command', 'Delete command', 'Edit user variables']

        for i in range(len(button_names)):
            newid = wx.NewId()
            self.toolbar.AddTool(newid, label=button_names[i],
                                 bitmap=button_images[i],
                                 shortHelp=tooltips[i])
            self.Bind(wx.EVT_TOOL, callbacks[i], id=newid)
        self.toolbar.Realize()

    def onsimulate(self, event):
        """ Function to simulate the model.

        :return:
        """
        self.plugin.parent.eh_tb_simulate(event)

    def SetDataList(self, datalist):
        '''SetDataList(self, datalist) --> None

        Sets the name of the available data sets
        '''
        self.datalist = datalist

    def GetDataList(self):
        '''SetDataList(self) --> list

        Retrives the data list
        '''
        return self.datalist

    def SetParameterList(self, parameterlist):
        '''SetParameterList(self, parameterlist) --> None

        Sets the code list for all definition of custom parameters
        '''
        self.parameterlist = parameterlist

    def GetParameterList(self):
        '''SetParameterList(self) --> list

        Retrives the parameter list
        '''
        return self.parameterlist

    def SetExpressionList(self, expressionlist):
        '''SetExpressionList(expressionlist) --> None

        Sets the expression list, should be a 2D list with the
        one list for each item in datalist
        '''
        if len(expressionlist)!=len(self.datalist):
            raise ValueError('The list of expression has to have the'+ \
                             ' same length as the data list')
        self.expressionlist = expressionlist

    def GetExpressionList(self):
        '''GetExpressionList(self) --> expression list

        Returns the expressionlist
        '''
        return self.expressionlist

    def SetSimArgs(self, sim_funcs, insts, args):
        '''SetSimArgs(self, sim_func, inst, args) --> None

        Sets the current simulation function for each data set
        their instruments and arguments
        sim_func: A list of simulation names
        inst: A list of instrument names
        args: A list of argument list for the sim_func's each argument should be a string.
        '''
        if len(sim_funcs)!=len(self.datalist):
            raise ValueError('The list of sim_funcs has to have the'+ \
                             ' same length as the data list')
        if len(insts)!=len(self.datalist):
            raise ValueError('The list of insts has to have the'+ \
                             ' same length as the data list')
        if len(args)!=len(self.datalist):
            raise ValueError('The list of args has to have the'+ \
                             ' same length as the data list')
        self.sim_funcs = sim_funcs[:]
        self.insts = insts[:]
        self.args = args[:]

    def GetSimArgs(self):
        '''GetSimArgs(self) --> (sim_funcs, insts, args)

        See SetSimArgs for a description of the parameters
        '''
        return self.sim_funcs, self.insts, self.args

    def AppendSim(self, sim_func, inst, args):
        '''AppendSim(self, sim_func, inst, args) --> None

        Appends a simultion to the Panel lists
        '''
        self.sim_funcs.append(sim_func)
        self.insts.append(inst)
        self.args.append(args)

    def InstrumentNameChange(self, old_name, new_name):
        '''OnInstrumentNameChange --> None

        Exchanges old_name to new name in the simulations.'''

        for i in range(len(self.insts)):
            if self.insts[i]==old_name:
                self.insts[i] = new_name
        self.update_listbox()

    def SetUpdateScriptFunc(self, func):
        '''SetUpdateScriptFunc(self, func) --> None

        Sets the function to be called when the script needs to be updated.
        will only be called as func(event)
        '''
        self.script_update_func = func

    def UpdateListbox(self, update_script=True):
        '''Update(self) --> None

        Update the listbox and runs the callback script_update_func
        '''
        self.update_listbox()
        if self.script_update_func and update_script:
            self.script_update_func(None)
        self.Refresh()

    def update_listbox(self):
        '''update_listbox(self) --> None

         updates the listbox.
        '''
        list_strings = []
        for i in range(len(self.datalist)):
            str_arg = ', '.join(self.args[i])
            list_strings.append('<code><b>%s</b>: %s(%s, %s)</code>'
                                ' \n'%(self.datalist[i],
                                       self.sim_funcs[i],
                                       str_arg, self.insts[i]))
            for item in self.expressionlist[i]:
                list_strings.append(self.command_indent+'%s</pre>'%item)

        self.listbox.SetItemList(list_strings)

    def get_expression_position(self):
        '''get_expression_position(self) --> (dataitem, expression)

        Finds the position in the expression list for a certain item.
        return -1 if it can not be found.
        '''
        index = self.listbox.GetSelection()

        if index==wx.NOT_FOUND:
            return -1, -1

        dataindex = -1
        itemindex = -1
        listindex = -1
        for i in range(len(self.datalist)):
            dataindex += 1
            listindex += 1
            itemindex = -1
            if listindex>=index:
                return dataindex, itemindex
            for item in self.expressionlist[i]:
                listindex += 1
                itemindex += 1
                if listindex>=index:
                    return dataindex, itemindex

        # If all other things fail...
        return -1, -1

    def Edit(self, event):
        '''Edit(self, event) --> None

        Edits an entry in the list
        '''
        data_pos, exp_pos = self.get_expression_position()
        if exp_pos!=-1 and data_pos!=-1:
            # Editing the expressions for variables
            list_item = self.expressionlist[data_pos][exp_pos]
            dlg = ParameterExpressionDialog(self, self.plugin.GetModel(),
                                            list_item, sim_func=self.onsimulate)
            if dlg.ShowModal()==wx.ID_OK:
                exp = dlg.GetExpression()
                self.expressionlist[data_pos][exp_pos] = exp
                self.UpdateListbox()
        if exp_pos==-1 and data_pos!=-1:
            # Editing the simulation function and its arguments
            dlg = SimulationExpressionDialog(self, self.plugin.GetModel(),
                                             self.plugin.sample_widget.instruments,
                                             self.sim_funcs[data_pos],
                                             self.args[data_pos],
                                             self.insts[data_pos], data_pos)
            if dlg.ShowModal()==wx.ID_OK:
                self.args[data_pos] = dlg.GetExpressions()
                self.insts[data_pos] = dlg.GetInstrument()
                self.sim_funcs[data_pos] = dlg.GetSim()
                self.UpdateListbox()

    def Insert(self, event):
        ''' Insert(self, event) --> None

        Inserts a new operations
        '''
        data_pos, exp_pos = self.get_expression_position()
        if data_pos!=-1:
            dlg = ParameterExpressionDialog(self, self.plugin.GetModel(), sim_func=self.onsimulate)
            if dlg.ShowModal()==wx.ID_OK:
                exp = dlg.GetExpression()
                if exp_pos==-1:
                    self.expressionlist[data_pos].insert(0, exp)
                else:
                    self.expressionlist[data_pos].insert(exp_pos, exp)
                self.UpdateListbox()

    def Delete(self, event):
        '''Delete(self, event) --> None

        Deletes an operation
        '''
        data_pos, exp_pos = self.get_expression_position()
        if exp_pos!=-1 and data_pos!=-1:
            self.expressionlist[data_pos].pop(exp_pos)
            self.UpdateListbox()

    def MoveUp(self, event):
        '''MoveUp(self, event) --> None

        Move an operation up
        '''
        pass

    def MoveDown(self, event):
        '''MoveDown(self, event) --> None

        Moves an operation down
        '''
        pass

    def EditPars(self, event):
        ''' EditPars(self, event) --> None

        Creates a new parameter
        '''
        dlg = EditCustomParameters(self, self.plugin.GetModel(),
                                   self.parameterlist)
        if dlg.ShowModal()==wx.ID_OK:
            self.parameterlist = dlg.GetLines()
            self.UpdateListbox()
        dlg.Destroy()

    def OnDataChanged(self, event):
        '''OnDataChanged(self, event) --> None

        Updated the data list
        '''
        self.UpdateListbox(update_script=False)


class EditCustomParameters(wx.Dialog):
    model: Model

    def __init__(self, parent, model, lines):
        wx.Dialog.__init__(self, parent, -1, 'Custom parameter editor')
        self.SetAutoLayout(True)
        self.model = model
        self.lines = lines
        self.var_name = 'cp'

        sizer = wx.BoxSizer(wx.VERTICAL)
        name_ctrl_sizer = wx.GridBagSizer(2, 3)

        col_labels = ['Name', 'Value', 'Sigma (for systematic error)']

        for item, index in zip(col_labels, list(range(len(col_labels)))):
            label = wx.StaticText(self, -1, item)
            name_ctrl_sizer.Add(label, (0, index), flag=wx.ALIGN_LEFT, border=5)

        self.name_ctrl = wx.TextCtrl(self, -1, size=(120, -1))
        name_ctrl_sizer.Add(self.name_ctrl, (1, 0),
                            flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        self.value_ctrl = wx.TextCtrl(self, -1, size=(120, -1))
        name_ctrl_sizer.Add(self.value_ctrl, (1, 1),
                            flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        self.error_ctrl = wx.TextCtrl(self, -1, size=(120, -1))
        name_ctrl_sizer.Add(self.error_ctrl, (1, 2),
                            flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        self.add_button = wx.Button(self, id=wx.ID_ANY, label='Add')
        name_ctrl_sizer.Add(self.add_button, (1, 3),
                            flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        sizer.Add(name_ctrl_sizer)
        self.Bind(wx.EVT_BUTTON, self.OnAdd, self.add_button)

        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.RIGHT | wx.TOP, 5)

        self.listbox = MyHtmlListBox(self, -1, size=(-1, 150),
                                     style=wx.BORDER_SUNKEN)
        self.listbox.SetItemList(self.lines)
        sizer.Add(self.listbox, 1, wx.GROW | wx.ALL, 10)

        self.delete_button = wx.Button(self, id=wx.ID_ANY, label='Delete')
        sizer.Add(self.delete_button, 0, wx.CENTRE, 0)
        self.Bind(wx.EVT_BUTTON, self.OnDelete, self.delete_button)

        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, id=wx.ID_OK)
        # okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, id=wx.ID_CANCEL))
        button_sizer.Realize()
        self.Bind(wx.EVT_BUTTON, self.OnApply, okay_button)

        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.RIGHT | wx.TOP, 5)

        sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT, 5)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()

    def OnApply(self, event):
        '''OnApply(self, event) --> None

        Callback for ok button click or apply button
        '''
        event.Skip()

    def OnAdd(self, event):
        '''OnAdd(self, event) --> None

        Callback for adding an entry
        '''
        sigma = self.error_ctrl.GetValue()
        if sigma.strip()=='':
            line = '%s.new_var(\'%s\', %s)'%(self.var_name,
                                             self.name_ctrl.GetValue(), self.value_ctrl.GetValue())
        else:
            line = '%s.new_sys_err(\'%s\', %s, %s)'%(self.var_name, self.name_ctrl.GetValue(),
                                                     self.value_ctrl.GetValue(), sigma)
        try:
            self.model.eval_in_model(line)
        except Exception as e:
            result = 'Could not evaluate the expression. The python error'+ \
                     'is: \n'+e.__repr__()
            ShowWarningDialog(self, result, 'Error in expression')
        else:
            self.lines.append(line)
            self.listbox.SetItemList(self.lines)

    def OnDelete(self, event):
        '''OnDelete(self, event) --> None

        Callback for deleting an entry
        '''
        result = 'Do you want to delete the expression?\n'+ \
                 'Remember to check if parameter is used elsewhere!'
        result = ShowQuestionDialog(self, result, 'Delete expression?')
        if result:
            self.lines.pop(self.listbox.GetSelection())
            self.listbox.SetItemList(self.lines)

    def GetLines(self):
        '''GetLines(self) --> uservars lines [list]

        Returns the list user variables.
        '''
        return self.lines


class SimulationExpressionDialog(wx.Dialog):
    '''A dialog to edit the Simulation expression
    '''
    model: Model

    def __init__(self, parent, model, instruments, sim_func, arguments, inst_name,
                 data_index):
        '''Creates a SimualtionExpressionDialog.

        model - a Model object.
        instruments - a dictionary of possible instruments
        arguments - the arguments to the simulation function, a list of strings.
        sim_func - a string of the simulation function name.
        inst_name - the name of the current instrument.
        data_index - an integer for the current data index.
        '''
        if not model.compiled:
            model.compile_script()

        self.model = model
        self.instruments = instruments
        self.available_sim_funcs = list(self.model.eval_in_model('model.SimulationFunctions.keys()'))
        self.data_index = data_index

        # Do the layout of the dialog
        wx.Dialog.__init__(self, parent, -1, 'Simulation editor')
        self.SetAutoLayout(True)

        # Find out the maximum number of arguments to the available sim_funcs
        max_val = -1
        self.sim_args = {}
        self.sim_defaults = {}
        for func in self.available_sim_funcs:
            doc = self.model.eval_in_model('model.SimulationFunctions'
                                           '["%s"].__doc__'%func)
            doc_lines = find_code_segment(doc, 'Parameters').splitlines()
            max_val = max(len(doc_lines), max_val)
            args = []
            defaults = []
            for line in doc_lines:
                items = line.lstrip().rstrip().split(' ')
                args.append(items[0])
                defaults.append(items[1].replace('data', 'd'))
            self.sim_args[func] = args
            self.sim_defaults[func] = defaults

        expressions = {'Instrument': inst_name}
        for arg_name, arg in zip(self.sim_args[sim_func], arguments):
            expressions[arg_name] = arg

        if max_val<0:
            raise ValueError('Wrongly formatted function docs for the simulation functions')

        gbs = wx.GridBagSizer(2, max_val)

        # Creating the column labels
        col_labels = ['Simulation', 'Instrument']
        [col_labels.append(arg) for arg in self.sim_args[sim_func] if not arg in col_labels]
        self.labels = []
        self.arg_controls = []
        for index in range(2+max_val):
            label = wx.StaticText(self, -1, '')
            gbs.Add(label, (0, index), flag=wx.ALIGN_LEFT, border=5)
            self.labels.append(label)
            # If the expression is not an instrument or simulation function
            if index>1:
                exp_ctrl = wx.TextCtrl(self, -1, size=(100, -1))
                gbs.Add(exp_ctrl, (1, index),
                        flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
                self.arg_controls.append(exp_ctrl)

        for item, label in zip(col_labels[:2], self.labels[:2]):
            label.SetLabel(item)
        # Creating the text boxes for the arguments
        # Setting the text in the column labels and text controls
        for item, label, arg_ctrl in zip(col_labels[2:],
                                         self.labels[2:],
                                         self.arg_controls):
            label.SetLabel(item)
            arg_ctrl.SetValue(expressions[item])
            arg_ctrl.SetEditable(True)

        for i in range(len(col_labels)-2, len(self.arg_controls)):
            self.arg_controls[i].SetEditable(False)
            # self.arg_controls[i].Show(False)
            # self.arg_controls[i].SetValue('NA')

        # Creating the controls
        # Simulation choice control
        self.sim_choice = wx.Choice(self, -1,
                                    choices=self.available_sim_funcs)
        self.Bind(wx.EVT_CHOICE, self.on_sim_change, self.sim_choice)
        self.sim_choice.SetSelection(self.available_sim_funcs.index(sim_func))
        gbs.Add(self.sim_choice, (1, 0),
                flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)

        # Instrument choice control
        self.inst_choice = wx.Choice(self, -1,
                                     choices=list(self.instruments.keys()))
        # self.Bind(wx.EVT_CHOICE, self.on_inst_change, self.inst_choice)
        self.inst_choice.SetSelection(list(self.instruments.keys()).index(expressions['Instrument']))
        gbs.Add(self.inst_choice, (1, 1),
                flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)

        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))

        button_sizer.Realize()
        self.Bind(wx.EVT_BUTTON, self.on_ok_button, okay_button)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(gbs, 1, wx.GROW | wx.ALL, 10)
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.RIGHT | wx.TOP, 5)
        sizer.AddSpacer(5)

        sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT, 5)
        sizer.AddSpacer(5)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()

    def on_sim_change(self, evt):
        '''Callback for changing the choice widget for the different simulations.
        '''
        new_sim = self.sim_choice.GetStringSelection()
        # Update the column labels
        new_labels = []
        for label, arg_name in zip(self.labels[2:], self.sim_args[new_sim]):
            new_labels.append(label.GetLabel()!=arg_name)
            label.SetLabel(arg_name)
        # Clear the remaining column labels
        for label in self.labels[len(self.sim_args[new_sim])+2:]:
            label.SetLabel('')

        # Update the text controls - if needed
        for i in range(len(self.sim_args[new_sim])):
            # if new_labels[i]:
            if True:
                self.arg_controls[i].SetValue(self.sim_defaults[new_sim][i])
            self.arg_controls[i].SetEditable(True)
            # self.arg_controls[i].Show(True)
        # Hide and clear the remaining text controls
        for ctrl in self.arg_controls[len(self.sim_args[new_sim]):]:
            ctrl.SetEditable(False)
            ctrl.SetValue('')

            # ctrl.Show(False)

    def on_ok_button(self, event):
        '''Callback for pressing the ok button in the dialog'''
        expressions = self.GetExpressions()
        # Hack to get it working with d = data[0]
        exec('d = data[%d]'%self.data_index, self.model.script_module.__dict__)
        for exp in expressions:
            try:
                self.model.eval_in_model(exp)
            except Exception as e:
                result = ('Could not evaluate expression:\n%s.\n'%exp+
                          ' The python error is: \n'+e.__repr__())
                ShowWarningDialog(self, result, 'Error in expression')
            else:
                event.Skip()

    def GetExpressions(self):
        ''' Returns the current expressions in the dialog box '''
        return [ctrl.GetValue() for ctrl in self.arg_controls
                if ctrl.IsEditable()]

    def GetInstrument(self):
        ''' Returns the selected instrument, a string'''
        return self.inst_choice.GetStringSelection()

    def GetSim(self):
        ''' Returns the selected simulation, a string'''
        return self.sim_choice.GetStringSelection()


class ParameterExpressionDialog(wx.Dialog):
    ''' A dialog for setting parameters for fitting
    '''
    model: Model

    def __init__(self, parent, model, expression=None, sim_func=None):
        wx.Dialog.__init__(self, parent, -1, 'Parameter editor')
        self.SetAutoLayout(True)
        self.model = model
        self.sim_func = sim_func
        if not model.compiled:
            model.compile_script()

        gbs = wx.GridBagSizer(2, 3)

        col_labels = ['Object', 'Parameter', 'Expression']

        for item, index in zip(col_labels, list(range(len(col_labels)))):
            label = wx.StaticText(self, -1, item)
            gbs.Add(label, (0, index), flag=wx.ALIGN_LEFT, border=5)

        # Get the objects that should be in the choiceboxes
        par_dict = model.get_possible_parameters()
        objlist = []
        funclist = []
        for cl in par_dict:
            obj_dict = par_dict[cl]
            for obj in obj_dict:
                objlist.append(obj)
                funclist.append(obj_dict[obj])

        self.objlist = objlist
        self.funclist = funclist
        self.obj_choice = wx.Choice(self, -1, choices=objlist)
        self.Bind(wx.EVT_CHOICE, self.on_obj_change, self.obj_choice)

        self.func_choice = wx.Choice(self, -1)
        # This will init func_choice
        self.obj_choice.SetSelection(0)

        gbs.Add(self.obj_choice, (1, 0),
                flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)
        gbs.Add(self.func_choice, (1, 1),
                flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)

        exp_right = ''
        if expression:
            p = expression.find('(')
            exp_left = expression[:p]
            obj = exp_left.split('.')[0]
            func = exp_left.split('.')[1]
            exp_right = expression[p+1:-1]
            obj_pos = [i for i in range(len(objlist)) if objlist[i]==obj]
            if len(obj_pos)>0:
                self.obj_choice.SetSelection(obj_pos[0])
                self.on_obj_change(None)
                func_pos = [i for i in range(len(funclist[obj_pos[0]])) \
                            if funclist[obj_pos[0]][i]==func]
                if len(func_pos)>0:
                    self.func_choice.SetSelection(func_pos[0])
                else:
                    raise ValueError('The function %s for object %s does not exist'%(func, obj))
            else:
                raise ValueError('The object %s does not exist'%obj)

        # self.expression_ctrl = wx.TextCtrl(self, -1, exp_right,\
        #                       size=(300, -1))

        self.expression_ctrl = ParameterExpressionCombo(par_dict, sim_func, self, -1, exp_right,
                                                        size=(300, -1))
        gbs.Add(self.expression_ctrl, (1, 2),
                flag=wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL, border=5)

        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        # apply_button = wx.Button(self, wx.ID_APPLY)
        # apply_button.SetDefault()
        # button_sizer.AddButton(apply_button)
        button_sizer.Realize()
        self.Bind(wx.EVT_BUTTON, self.OnApply, okay_button)
        # self.Bind(wx.EVT_BUTTON, self.OnApply, apply_button)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(gbs, 1, wx.GROW | wx.ALL, 10)
        line = wx.StaticLine(self, -1, size=(20, -1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW | wx.RIGHT | wx.TOP, 5)
        sizer.AddSpacer(5)

        sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT, 5)
        sizer.AddSpacer(5)
        self.SetSizer(sizer)
        sizer.Fit(self)
        self.Layout()

    def on_obj_change(self, event):
        '''on_obj_change(self, event) --> None

        On changing the object the funclist should be updated
        '''
        index = self.obj_choice.GetSelection()
        self.func_choice.SetItems(self.funclist[index])

    def OnApply(self, event):
        '''OnApply(self, event) --> None
        '''
        evalstring = self.GetExpression()
        try:
            self.model.eval_in_model(evalstring)
        except Exception as e:
            result = 'Could not evaluate the expression. The python'+ \
                     'is: \n'+e.__repr__()
            ShowWarningDialog(self, result, 'Error in expression')
        else:
            event.Skip()

    def GetExpression(self):
        '''GetExpression(self) --> expression

        Yields the string that has been edited in the dialog
        '''
        objstr = self.obj_choice.GetStringSelection()
        funcstr = self.func_choice.GetStringSelection()
        set_expression = self.expression_ctrl.GetValue()
        evalstring = '%s.%s(%s)'%(objstr, funcstr, set_expression)

        return evalstring

