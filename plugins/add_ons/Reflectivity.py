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

import plugins.add_on_framework as framework
from plotpanel import PlotPanel
import model as modellib
import wx

import numpy as np
import sys, os, re, time, StringIO, traceback

from help_modules.custom_dialog import *
import help_modules.reflectivity_images as images

_avail_models = ['spec_nx', 'interdiff', 'xmag', 'mag_refl', 'soft_nx']
_set_func_prefix = 'set'


def default_html_decorator(name, str):
        return str


class SampleHandler:
    def __init__(self, sample, names):
        self.sample = sample
        self.names = names
        self.getStringList()

    def getStringList(self, html_encoding = False, html_decorator=default_html_decorator):
        '''
        Function to generate a list of strings that gives
        a visual representation of the sample.
        '''
        slist=[self.sample.Substrate.__repr__()]
        poslist=[(None,None)]
        i=0;j=0
        for stack in self.sample.Stacks:
            j=0
            for layer in stack.Layers:
                slist.append(layer.__repr__())
                poslist.append((i,j))
                j+=1
            slist.append('Stack: Repetitions = %s'%str(stack.Repetitions))
            poslist.append((i,None))
            i+=1
        slist.append(self.sample.Ambient.__repr__())
        for item in range(len(slist)):
            name = self.names[-item-1]
            par_str = slist[item]
            if slist[item][0] == 'L' and item != 0 and item != len(slist) - 1:
                if html_encoding:
                    slist[item] = ('<code>&nbsp;&nbsp;&nbsp;<b>' + name + '</b> = '
                                   + html_decorator(name, par_str) + '</code>')
                else:
                    slist[item] = self.names[-item-1] + ' = model.' + slist[item]
            else:
                if item == 0 or item == len(slist) - 1:
                    # This is then the ambient or substrates
                    if html_encoding:
                        slist[item] = ('<code><b>' + name + '</b> = '
                                       + html_decorator(name, par_str) + '</code>')
                    else:
                        slist[item] = self.names[-item-1] + ' = model.' + slist[item]
                else:
                    # This is a stack!
                    if html_encoding:
                        slist[item] = ('<font color = "BLUE"><code><b>' + name +'</b> = '
                                       + html_decorator(name, par_str) + '</code></font>')
                    else:
                        slist[item] = self.names[-item-1] + ' = model.' + slist[item]
        poslist.append((None, None))
        slist.reverse()
        poslist.reverse()
        self.poslist = poslist
        return slist
    
    def htmlize(self, code): 
        '''htmlize(self, code) --> code
        
        htmlize the code for display
        '''
        p = code.index('=')
        name = '<code><b>%s</b></code>'%code[:p]
        items = code[p:].split(',')
        return name + ''.join(['<code>%s,</code>'%item for item in items])

    def getCode(self):
        '''
        Generate the python code for the current sample structure.
        '''
        slist = self.getStringList()
        layer_code=''
        
        # Create code for the layers:
        for item in slist:
            if item.find('Layer')>-1:
                itemp=item.lstrip()
                layer_code=layer_code+itemp+'\n'
        #Create code for the Stacks:
        i=0
        stack_code=''
        item=slist[i]
        maxi=len(slist)-1
        while(i<maxi):
            if item.find('Stack')>-1:
                stack_strings=item.split(':')
                stack_code = stack_code + stack_strings[0] + '(Layers=['
                i += 1
                item = slist[i]
                stack_layers = []
                while(item.find('Stack') < 0 and i < maxi):
                    itemp = item.split('=')[0]
                    itemp = itemp.lstrip()
                    stack_layers.append(itemp)
                    #stack_code = stack_code + itemp+','
                    i += 1
                    item = slist[i]
                stack_layers.reverse()
                stack_code += ', '.join(stack_layers)
                i-=1
                if stack_code[-1] != '[':
                    stack_code = stack_code[:-1]+'],'+stack_strings[1]+')\n'
                else:
                    stack_code = stack_code[:]+'],'+stack_strings[1]+')\n'
            i+=1
            item = slist[i]
        # Create the code for the sample
        sample_code = 'sample = model.Sample(Stacks = ['
        stack_strings = stack_code.split('\n')
        rest_sample_rep = '], '
        sample_string_pars = self.sample.__repr__().split(':')[1].split('\n')[0].lstrip()
        if len(sample_string_pars) != 0:
            sample_string_pars += ', '
        rest_sample_rep += sample_string_pars + 'Ambient = Amb, Substrate = Sub)\n'
        if stack_strings != ['']:
            # Added 20080831 MB bugfix
            stack_strings.reverse()
            for item in stack_strings[1:]:
                itemp=item.split('=')[0]
                sample_code = sample_code + itemp + ','
            sample_code = sample_code[:-2] + rest_sample_rep
        else:
            sample_code += rest_sample_rep
            
        return layer_code,stack_code, sample_code

    def getItem(self, pos):
        '''
        Returns the item (Stack or Layer) at position pos
        '''
        if pos==0:
            return self.sample.Ambient
        if pos==len(self.poslist)-1:
            return self.sample.Substrate
        stack=self.sample.Stacks[self.poslist[pos][0]]
        if self.poslist[pos][1] == None:
            return stack
        return stack.Layers[self.poslist[pos][1]]

    def deleteItem(self,pos):
        '''
        Delete item pos in the lsit if the item is a stack all the Layers
        are deleted as well.
        '''
        if pos==0:
            return None
        if pos==len(self.poslist)-1:
            return None
        stack=self.sample.Stacks[self.poslist[pos][0]]
        if self.poslist[pos][1]==None:
            self.sample.Stacks.pop(self.poslist[pos][0])
            p = self.poslist[pos][0]
            pt = pos
            while self.poslist[pt][0] == p:
                pt += 1
            pt -=1
            while self.poslist[pt][0] == p:
               self.names.pop(pt)
               pt -=1
                
        else:
            stack.Layers.pop(self.poslist[pos][1])
            self.names.pop(pos)
        return self.getStringList()

    def insertItem(self,pos,type, name = 'test'):
        '''
        Insert an item into the sample at position pos in the list
        and of type. type is a string of either Stack or Layer
        '''
        spos=self.poslist[pos]
        added=False
        last=False
        if pos==0:
            spos=(self.poslist[1][0],self.poslist[1][1])#+1
            #spos=(None,None)
        if pos==len(self.poslist)-1:
            spos=self.poslist[-2]
            last=True
        stackp=False
        if spos[1]==None:
            spos=(spos[0],0)
            stackp=True
        if spos[0]==None:
            spos=(0,spos[1])
        
        # If it not the first item i.e. can't insert anything before the 
        # ambient layer
        if pos !=0 :
            if type=='Stack':
                stack=self.model.Stack(Layers=[])
                if last:
                    self.names.insert(pos,name)
                else:
                    if stackp:
                        self.names.insert(pos+len(self.sample.Stacks[spos[0]].Layers)+1,name)
                    else:
                        self.names.insert(pos+spos[1]+1,name)
                self.sample.Stacks.insert(spos[0],stack)
                added=True
                
            if type=='Layer' and len(self.poslist)>2:
                layer=self.model.Layer()
                if last:
                    self.names.insert(pos, name)
                else:
                    if spos[1] >= 0:
                        self.names.insert(pos+1,name)
                    else:
                        self.names.insert(pos+len(self.sample.Stacks[spos[0]].Layers)+1,name)
                if last:
                    self.sample.Stacks[spos[0]].Layers.insert(0,layer)
                else:
                    if self.poslist[pos][1] == None:
                        self.sample.Stacks[spos[0]].Layers.append(layer)
                    else:
                        self.sample.Stacks[spos[0]].Layers.insert(spos[1], layer)
                added=True
                
        else:
            if type=='Stack':
                stack=self.model.Stack(Layers=[])
                self.sample.Stacks.append(stack)
                added=True
                self.names.insert(pos+1,name)
            if type=='Layer' and len(self.poslist)>2:
                layer=self.model.Layer()
                self.sample.Stacks[spos[0]].Layers.append(layer)
                added=True
                self.names.insert(pos+2,name)
        if added:
            
            return self.getStringList()
        else:
            return None

        

    def canInsertLayer():
        return self.poslist>2

    def checkName(self, name):
        return self.names.__contains__(name)

    def getName(self, pos):
        """ Returns the name for the object at pos
        :param pos: list position for the name
        :return: the name (string)
        """
        return self.names[pos]
    
    def changeName(self, pos, name):
        if name in self.names and name != self.names[pos]:
            return False
        elif pos == len(self.names)-1 or pos == 0:
            return False
        else:
            self.names[pos] = name
            return True
            
    
    def moveUp(self,pos):
        '''
        Move the item up - with stacks move the entire stack up one step.
        Moves layer only if it is possible.
        '''
        if pos > 1 and pos!=len(self.poslist)-1:
            if self.poslist[pos][1]==None:
                temp=self.sample.Stacks.pop(self.poslist[pos][0])
                temps=[]
                for index in range(len(temp.Layers)+1):
                    temps.append(self.names.pop(pos))
                for index in range(len(temp.Layers)+1):
                    self.names.insert(pos-len(self.sample.Stacks[self.poslist[pos][0]].Layers)-1,temps[-index-1])
                self.sample.Stacks.insert(self.poslist[pos][0]+1,temp)
                return self.getStringList()
            else: #i.e. it is a layer we move
                if pos > 2:
                    temp=self.sample.Stacks[self.poslist[pos][0]].Layers.pop(self.poslist[pos][1])
                    temps=self.names.pop(pos)
                    if self.poslist[pos-1][1]==None: # Next item a Stack i.e. jump up
                        self.sample.Stacks[self.poslist[pos-2][0]].Layers.insert(0,temp)
                        self.names.insert(pos-1,temps)
                    else: #Moving inside a stack
                        self.sample.Stacks[self.poslist[pos][0]].Layers.insert(self.poslist[pos][1]+1,temp)
                        self.names.insert(pos-1,temps)
                    return self.getStringList()
                else:
                    return None
        else:
            return None

    def moveDown(self,pos):
        '''
        Move the item down - with stacks move the entire stack up one step.
        Moves layer only if it is possible.
        '''

        if pos != 0 and pos < len(self.poslist)-2:

            if self.poslist[pos][1]==None: #Moving a stack
                if self.poslist[pos][0]!=0:
                    temp=self.sample.Stacks.pop(self.poslist[pos][0])
                    temps=[]
                    for index in range(len(temp.Layers)+1):
                        temps.append(self.names.pop(pos))
                    for index in range(len(temp.Layers)+1):
                        self.names.insert(pos+len(self.sample.Stacks[self.poslist[pos][0]-1].Layers)+1,temps[-index-1])
                    self.sample.Stacks.insert(self.poslist[pos][0]-1,temp)
                    return self.getStringList()
                else:
                    return None
                    
            else: #i.e. it is a layer we move
                if pos < len(self.poslist)-2:
                    temp=self.sample.Stacks[self.poslist[pos][0]].Layers.pop(self.poslist[pos][1])
                    temps=self.names.pop(pos)
                    if self.poslist[pos+1][1]==None: # Next item a Stack i.e. jump down
                        self.sample.Stacks[self.poslist[pos+1][0]].Layers.insert(len(self.sample.Stacks[self.poslist[pos+1][0]].Layers),temp)
                        self.names.insert(pos+1,temps)
                    else: #Moving inside a stack
                        self.sample.Stacks[self.poslist[pos][0]].Layers.insert(self.poslist[pos][1]-1,temp)#-2
                        self.names.insert(pos+1,temps)
                    return self.getStringList()
        else:
            return None

class MyHtmlListBox(wx.HtmlListBox):

    def __init__(self, parent, id, size=(-1, -1), style = wx.BORDER_SUNKEN):
        wx.HtmlListBox.__init__(self, parent, id, size=size,
                                style=style)
        self.SetItemList(['Starting up...'])

    def SetItemList(self, list):
        self.html_items = list
        self.SetItemCount(len(list))
        #self.RefreshAll()
        self.Refresh()
        
    def OnGetItem(self, n):
        return self.html_items[n]
   

class SamplePanel(wx.Panel):
    def __init__(self, parent, plugin, refindexlist=[]):
        wx.Panel.__init__(self, parent)
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
        self.toolbar = wx.ToolBar(self, style=wx.TB_FLAT|wx.TB_HORIZONTAL)
        boxhor.Add((-1, 2))
        self.do_toolbar()
        boxhor.Add(self.toolbar, proportion=0, flag=wx.EXPAND, border=1)
        boxhor.Add((-1, 2))
        self.listbox = MyHtmlListBox(self, -1, style=wx.BORDER_SUNKEN)
        #self.listbox.SetItemList(self.sampleh.getStringList())
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.lbDoubleClick, self.listbox)
        boxhor.Add(self.listbox, 1, wx.EXPAND)

        boxver.Add(boxhor, 1,  wx.EXPAND)

        self.SetSizer(boxver)
        self.toolbar.Realize()
        self.update_callback = lambda event:''

    def do_toolbar(self):

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, 'Insert Layer', bitmap=images.insert_layer.getBitmap(),
                                  shortHelp='Insert a Layer')
        self.Bind(wx.EVT_TOOL, self.InsertLay, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, 'Insert Stack', bitmap=images.insert_stack.getBitmap(),
                                  shortHelp='Insert a Stack')
        self.Bind(wx.EVT_TOOL, self.InsertStack, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, 'Delete', bitmap=images.delete.getBitmap(), shortHelp='Delete item')
        self.Bind(wx.EVT_TOOL, self.DeleteSample, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, 'Rename', bitmap=images.change_name.getBitmap(),
                                  shortHelp='Rename')
        self.Bind(wx.EVT_TOOL, self.ChangeName, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, 'Move up', bitmap=images.move_up.getBitmap(),
                                  shortHelp='Move item up')
        self.Bind(wx.EVT_TOOL, self.MoveUp, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, 'Move down', bitmap=images.move_down.getBitmap(),
                                  shortHelp='Move item down')
        self.Bind(wx.EVT_TOOL, self.MoveDown, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, 'Edit Sample', bitmap=images.sample.getBitmap(),
                                  shortHelp='Edit Sample parameters')
        self.Bind(wx.EVT_TOOL, self.EditSampleParameters, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, 'Edit Instrument', bitmap=images.instrument.getBitmap(),
                                  shortHelp='Edit Instruments')
        self.Bind(wx.EVT_TOOL, self.EditInstrument, id=newid)


    def SetUpdateCallback(self, func):
        ''' SetUpdateCallback(self, func) --> None
        
        Sets the update callback will be called when the sample is updated.
        The call is on the form func(event)
        '''
        self.update_callback = func

    def create_html_decorator(self):
        """
        creates a html decorator function
        :return:
        """
        grid_parameters = self.plugin.GetModel().get_parameters()
        dic_lookup = {}
        for par in grid_parameters.get_names():
            l = par.split('.')
            if len(l) == 2:
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
                start_index = str.index('(') + 1
            except ValueError:
                start_index = 0
            ret_str = str[:start_index]
            for par_str in str[start_index:].split(','):
                par_name = par_str.split('=')[0].strip()
                # par_name normal paramter (real number)
                if (name, par_name) in dic_lookup:
                    val, state = dic_lookup[(name, par_name)]
                    if state == 1:
                        par_str = ' <font color=%s><b>%s=%.2e</b></font>,'%(fit_color_str, par_name, val)
                    elif state == 2:
                        par_str = ' <font color=%s><b>%s=%.2e</b></font>,'%(const_fit_color_str, par_name, val)
                # par_name is a complex parameter...
                elif (name, par_name + 'real') in dic_lookup or (name, par_name + 'imag') in dic_lookup:
                    if (name, par_name + 'real') in dic_lookup:
                        val, state = dic_lookup[(name, par_name + 'real')]
                        if state == 1:
                            par_str = ' <font color=%s><b>%s=(%.2e,</b></font>'%(fit_color_str, par_name, val)
                        elif state == 2:
                            par_str = ' <font color=%s><b>%s=(%.2e,</b></font>'%(const_fit_color_str, par_name, val)
                    else:
                        par_str = ' <b>%s=??+</b>'%(par_name)
                    if (name, par_name + 'imag') in dic_lookup:
                        val, state = dic_lookup[(name, par_name + 'imag')]
                        if state == 1:
                            par_str += ' <font color=%s><b>%.2e)</b></font>,'%(fit_color_str, val)
                        elif state == 2:
                            par_str += ' <font color=%s><b>%.2e)</b></font>,'%(const_fit_color_str, val)
                    else:
                        par_str += ' <b>??)</b>,'

                else:
                    par_str += ','
                ret_str += par_str
            # Remove trailing ,
            if ret_str[-1] == ',':
                ret_str = ret_str[:-1]
            if str[-1] == ')' and ret_str[-1] != ')':
                ret_str += ')'
            return ret_str
        return decorator


        
    def Update(self, update_script = True):
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
        except Exception, e:
            string_choices = {}
        for item in self.model.SampleParameters:
            if item != 'Stacks' and item != 'Substrate' and item != 'Ambient':
                if string_choices.has_key(item):
                    validators[item] = string_choices[item]
                else:
                    validators[item] = FloatObjectValidator()
                val = getattr(self.sampleh.sample, item)
                vals[item] = val
                pars.append(item)
                items.append((item, val))
                # Check if the parameter is in the grid and in that case set it as uneditable
                func_name = obj_name + '.' + _set_func_prefix + item.capitalize()
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
            vals=dlg.GetValues()
            #print vals
            states = dlg.GetStates()
            for par in pars:
                if not states[par]:
                    setattr(self.sampleh.sample, par, vals[par])
                if editable[par] != states[par]:
                    value = eval_func(vals[par])
                    minval = min(value*(1 - self.variable_span), value*(1 + self.variable_span))
                    maxval = max(value*(1 - self.variable_span), value*(1 + self.variable_span))
                    func_name = obj_name + '.' + _set_func_prefix + par.capitalize()
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
            if self.model.instrument_string_choices.has_key(item):
                #validators.append(self.model.instrument_string_choices[item])
                validators[item] = self.model.instrument_string_choices[item]
            else:
                #validators.append(FloatObjectValidator())
                validators[item] = FloatObjectValidator()
            for inst_name in self.instruments:
                val = getattr(self.instruments[inst_name], item)
                vals[inst_name][item] = val
                # Check if the parameter is in the grid and in that case set it as uneditable
                func_name = inst_name + '.' + _set_func_prefix + item.capitalize()
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
            #print vals
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
                        setattr(self.instruments[inst_name], par, vals[inst_name][par])
                    else:
                        setattr(self.instruments[inst_name], par, old_vals[inst_name][par])
                    if new_instrument and states[inst_name][par] > 0:
                        value = eval_func(vals[inst_name][par])
                        minval = min(value*(1 - self.variable_span), value*(1 + self.variable_span))
                        maxval = max(value*(1 - self.variable_span), value*(1 + self.variable_span))
                        func_name = inst_name + '.' + _set_func_prefix + par.capitalize()
                        grid_parameters.set_fit_state_by_name(func_name, value, states[inst_name][par], minval, maxval)
                    elif not new_instrument:
                        if editable[inst_name][par] != states[inst_name][par]:
                            value = eval_func(vals[inst_name][par])
                            minval = min(value*(1 - self.variable_span), value*(1 + self.variable_span))
                            maxval = max(value*(1 - self.variable_span), value*(1 + self.variable_span))
                            func_name = inst_name + '.' + _set_func_prefix + par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name, value, states[inst_name][par], minval, maxval)

            # Loop to remove instrument from grid if not returned from Dialog
            for inst_name in old_insts:
                if inst_name not in vals.keys():
                    for par in self.model.InstrumentParameters:
                        if editable[inst_name][par] > 0:
                            func_name = inst_name + '.' + _set_func_prefix + par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name, 0, 0, 0, 0)


            # Tell the grid to reload the parameters
            self.plugin.parent.paramter_grid.SetParameters(grid_parameters)

            for change in dlg.GetChanges():
                if change[0] != '' and change[1] != '':
                    self.plugin.InstrumentNameChange(change[0], change[1])
                elif change[1] == '':
                    self.plugin.InstrumentNameChange(change[0], 'inst')
                     
            self.Update()
        else:
            pass
        dlg.Destroy()
        
        
    def MoveUp(self,evt):
        sl=self.sampleh.moveUp(self.listbox.GetSelection())
        if sl:
            self.Update()
            self.listbox.SetSelection(self.listbox.GetSelection()-1)

    def MoveDown(self,evt):
        sl=self.sampleh.moveDown(self.listbox.GetSelection())
        if sl:
            self.Update()
            self.listbox.SetSelection(self.listbox.GetSelection()+1)

    def InsertStack(self,evt):
        # Create Dialog box
        items = [('Name', 'name')]
        validators = {}
        vals = {}
        validators['Name'] = NoMatchValidTextObjectValidator(self.sampleh.names)
        pars = ['Name']
        vals['Name'] = 'name'
        dlg = ValidateDialog(self, pars, vals, validators,
                             title = 'Give Stack Name')
        
        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
                vals = dlg.GetValues()
        dlg.Destroy()
        # if not a value is selected operate on first
        pos = max(self.listbox.GetSelection(),0)
        sl = self.sampleh.insertItem(pos, 'Stack',  vals['Name'])
        if sl:
            self.Update()
        else:
            self.plugin.ShowWarningDialog('Can not insert a stack at the'
            ' current position.')
            
    def InsertLay(self,evt):
        # Create Dialog box
        #items = [('Name', 'name')]
        #validators = [NoMatchValidTextObjectValidator(self.sampleh.names)]
        dlg = ValidateDialog(self, ['Name'], {'Name': 'name'},
                             {'Name': NoMatchValidTextObjectValidator(self.sampleh.names)}, 
                             title='Give Layer Name')
        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
                vals=dlg.GetValues()
        dlg.Destroy()
        # if not a value is selected operate on first
        pos = max(self.listbox.GetSelection(),0)
        #Create the Layer
        sl = self.sampleh.insertItem(pos, 'Layer', vals['Name'])
        if sl:
            self.Update()
        else:
            self.plugin.ShowWarningDialog('Can not insert a layer at the'
            ' current position. Layers has to be part of a stack.')
        
    def DeleteSample(self,evt):
        slold=self.sampleh.getStringList()
        sl=self.sampleh.deleteItem(self.listbox.GetSelection())
        if sl:
            self.Update()
            
    def ChangeName(self, evt):
        '''Change the name of the current selected item.
        '''
        pos = self.listbox.GetSelection()
        if pos == 0 or pos == len(self.sampleh.names) - 1:
            self.plugin.ShowInfoDialog('It is forbidden to change the'\
                'name of the substrate (Sub) and the Ambient (Amb) layers.')
        else:
            unallowed_names = self.sampleh.names[:pos] +\
                                self.sampleh.names[max(0,pos - 1):]
            dlg = ValidateDialog(self,['Name'], {'Name':self.sampleh.names[pos]},
                                 {'Name':NoMatchValidTextObjectValidator(unallowed_names)},
                                 title='Give New Name')
            
            if dlg.ShowModal()==wx.ID_OK:
                    vals=dlg.GetValues()
                    result = self.sampleh.changeName(pos, vals['Name'])
                    if result:
                        self.Update()
                    else:
                        print 'Unexpected problems when changing name...'
            dlg.Destroy()
        
        

    def lbDoubleClick(self,evt):
        sel = self.sampleh.getItem(self.listbox.GetSelection())
        obj_name = self.sampleh.getName(self.listbox.GetSelection())
        eval_func = self.plugin.GetModel().eval_in_model
        sl=None
        items=[]
        validators = {}
        vals = {}
        pars = []
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()
        if isinstance(sel, self.model.Layer):
            # The selected item is a Layer
            for item in self.model.LayerParameters.keys():
                value = getattr(sel, item)
                vals[item] = value
                #if item!='n' and item!='fb':
                if type(self.model.LayerParameters[item]) != type(1+1.0J):
                    # Handle real parameters
                    validators[item] = FloatObjectValidator(eval_func, alt_types=[self.model.Layer])
                    func_name = obj_name + '.' + _set_func_prefix + item.capitalize()
                    grid_value = grid_parameters.get_value_by_name(func_name)
                    if grid_value is not None:
                        vals[item] = grid_value
                    editable[item] = grid_parameters.get_fit_state_by_name(func_name)

                else:
                    # Handle complex parameters
                    validators[item] = ComplexObjectValidator(eval_func, alt_types=[self.model.Layer])
                    func_name = obj_name + '.' + _set_func_prefix + item.capitalize()
                    grid_value_real = grid_parameters.get_value_by_name(func_name + 'real')
                    grid_value_imag = grid_parameters.get_value_by_name(func_name + 'imag')
                    if grid_value_real is not None:
                        v = eval_func(vals[item]) if type(vals[item]) is str else vals[item]
                        vals[item] = grid_value_real + v.imag*1.0J
                    if grid_value_imag is not None:
                        v = eval_func(vals[item]) if type(vals[item]) is str else vals[item]
                        vals[item] = v.real + grid_value_imag*1.0J
                    editable[item] = max(grid_parameters.get_fit_state_by_name(func_name + 'real'),
                                         grid_parameters.get_fit_state_by_name(func_name + 'imag'))

                items.append((item, value))
                pars.append(item)

                # Check if the parameter is in the grid and in that case set it as uneditable
                #func_name = obj_name + '.' + _set_func_prefix + item.capitalize()
                #grid_value = grid_parameters.get_value_by_name(func_name)
                #editable[item] = grid_parameters.get_fit_state_by_name(func_name)

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
                for par in self.model.LayerParameters.keys():
                    if not states[par]:
                        setattr(sel, par, vals[par])
                    if editable[par] != states[par]:
                        value = eval_func(vals[par])

                        if type(self.model.LayerParameters[par]) is complex:
                            #print type(value)
                            func_name = obj_name + '.' + _set_func_prefix + par.capitalize() + 'real'
                            val = value.real
                            minval = min(val*(1 - self.variable_span), val*(1 + self.variable_span))
                            maxval = max(val*(1 - self.variable_span), val*(1 + self.variable_span))
                            grid_parameters.set_fit_state_by_name(func_name, val, states[par], minval, maxval)
                            val = value.imag
                            minval = min(val*(1 - self.variable_span), val*(1 + self.variable_span))
                            maxval = max(val*(1 - self.variable_span), val*(1 + self.variable_span))
                            func_name = obj_name + '.' + _set_func_prefix + par.capitalize() + 'imag'
                            grid_parameters.set_fit_state_by_name(func_name, val, states[par], minval, maxval)
                        else:
                            val = value
                            minval = min(val*(1 - self.variable_span), val*(1 + self.variable_span))
                            maxval = max(val*(1 - self.variable_span), val*(1 + self.variable_span))
                            func_name = obj_name + '.' + _set_func_prefix + par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name, value, states[par], minval, maxval)

                        # Does not seem to be necessary
                        self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
                sl = self.sampleh.getStringList()
            dlg.Destroy()

        else: 
            # The selected item is a Stack
            for item in self.model.StackParameters.keys():
                if item!='Layers':
                    value=getattr(sel, item)
                    if isinstance(value,float):
                        validators[item] = FloatObjectValidator(eval_func, alt_types=[self.model.Stack])
                    else:
                        validators[item] = TextObjectValidator()
                    items.append((item,value))
                    pars.append(item)
                    vals[item] = value

                    # Check if the parameter is in the grid and in that case set it as uneditable
                    func_name = obj_name + '.' + _set_func_prefix + item.capitalize()
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
                    if editable[par] != states[par]:
                        value = eval_func(vals[par])
                        minval = min(value*(1 - self.variable_span), value*(1 + self.variable_span))
                        maxval = max(value*(1 - self.variable_span), value*(1 + self.variable_span))
                        func_name = obj_name + '.' + _set_func_prefix + par.capitalize()
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
        wx.Panel.__init__(self,parent)
        self.plugin = plugin
        boxver = wx.BoxSizer(wx.VERTICAL)
        # Indention for a command - used to seperate commands and data
        self.command_indent = '<pre>   '
        self.script_update_func = None
        self.parameterlist = []

        self.toolbar = wx.ToolBar(self, style=wx.TB_FLAT|wx.TB_HORIZONTAL)
        self.do_toolbar()
        boxver.Add((-1, 2))
        boxver.Add(self.toolbar, proportion=0, flag=wx.EXPAND, border=1)
        boxver.Add((-1, 2))
        
        self.listbox = MyHtmlListBox(self, -1, style=wx.BORDER_SUNKEN)
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.Edit, self.listbox)
        boxver.Add(self.listbox, 1, wx.EXPAND)
        
        self.SetSizer(boxver)
        self.toolbar.Realize()

    def do_toolbar(self):

        button_names = ['Insert', 'Delete', 'User Variables']
        button_images = [images.getaddBitmap(), images.getdeleteBitmap(),\
            images.getcustom_parameterBitmap()]
        callbacks = [self.Insert, self.Delete, self.EditPars]
        tooltips = ['Insert a command', 'Delete command', 'Edit user variables']

        for i in range(len(button_names)):
            newid = wx.NewId()
            self.toolbar.AddLabelTool(newid, label=button_names[i], bitmap=button_images[i], shortHelp=tooltips[i])
            self.Bind(wx.EVT_TOOL, callbacks[i], id=newid)


    def onsimulate(self):
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
        if len(expressionlist) != len(self.datalist):
            raise ValueError('The list of expression has to have the' +\
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
        if len(sim_funcs) != len(self.datalist):
            raise ValueError('The list of sim_funcs has to have the' +\
                ' same length as the data list')
        if len(insts) != len(self.datalist):
            raise ValueError('The list of insts has to have the' +\
                ' same length as the data list')
        if len(args) != len(self.datalist):
            raise ValueError('The list of args has to have the' +\
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
            if self.insts[i] == old_name:
                self.insts[i] = new_name
        self.update_listbox()
                 
    def SetUpdateScriptFunc(self, func):
        '''SetUpdateScriptFunc(self, func) --> None
        
        Sets the function to be called when the script needs to be updated.
        will only be called as func(event)
        '''
        self.script_update_func = func
    
    def Update(self, update_script = True):
        '''Update(self) --> None
        
        Update the listbox and runs the callback script_update_func
        '''
        self.update_listbox()
        
        if self.script_update_func and update_script:
            self.script_update_func(None)
            
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
                list_strings.append(self.command_indent + '%s</pre>'%item)
        
        self.listbox.SetItemList(list_strings)
        
    def get_expression_position(self):
        '''get_expression_position(self) --> (dataitem, expression)
        
        Finds the position in the expression list for a certain item.
        return -1 if it can not be found.
        '''
        index = self.listbox.GetSelection()
        
        if index == wx.NOT_FOUND:
            return (-1, -1)

        dataindex = -1
        itemindex = -1
        listindex = -1
        for i in range(len(self.datalist)):
            dataindex += 1
            listindex +=1
            itemindex = -1
            if listindex >= index:
                return (dataindex, itemindex)
            for item in self.expressionlist[i]:
                listindex += 1
                itemindex += 1
                if listindex >= index:
                    return (dataindex, itemindex)
                
        # If all other things fail...
        return (-1, -1)
            
            
    
    def Edit(self, event):
        '''Edit(self, event) --> None
        
        Edits an entry in the list
        '''
        data_pos, exp_pos = self.get_expression_position()
        if exp_pos != -1 and data_pos != -1:
            # Editing the expressions for variables
            list_item = self.expressionlist[data_pos][exp_pos]
            dlg = ParameterExpressionDialog(self, self.plugin.GetModel(),\
                list_item, sim_func=self.onsimulate)
            if dlg.ShowModal() == wx.ID_OK:
                exp = dlg.GetExpression()
                self.expressionlist[data_pos][exp_pos] = exp
                self.Update()
        if exp_pos == -1 and data_pos != -1:
            # Editing the simulation function and its arguments
            dlg = SimulationExpressionDialog(self, self.plugin.GetModel(),
                                              self.plugin.sample_widget.instruments,
                                              self.sim_funcs[data_pos], 
                                              self.args[data_pos], 
                                              self.insts[data_pos], data_pos)
            if dlg.ShowModal() == wx.ID_OK:
                self.args[data_pos] = dlg.GetExpressions()
                self.insts[data_pos] = dlg.GetInstrument()
                self.sim_funcs[data_pos] = dlg.GetSim()
                self.Update()
        
    def Insert(self, event):
        ''' Insert(self, event) --> None
        
        Inserts a new operations
        '''
        data_pos, exp_pos = self.get_expression_position()
        if data_pos != -1:
            dlg = ParameterExpressionDialog(self, self.plugin.GetModel(), sim_func=self.onsimulate)
            if dlg.ShowModal() == wx.ID_OK:
                exp = dlg.GetExpression()
                if exp_pos == -1:
                    self.expressionlist[data_pos].insert(0, exp)
                else:
                    self.expressionlist[data_pos].insert(exp_pos, exp)
                self.Update()
        
    def Delete(self, event):
        '''Delete(self, event) --> None
        
        Deletes an operation
        '''
        data_pos, exp_pos = self.get_expression_position()
        if exp_pos != -1 and data_pos != -1:
            self.expressionlist[data_pos].pop(exp_pos)
            self.Update()
    
        
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
        dlg = EditCustomParameters(self, self.plugin.GetModel(),\
            self.parameterlist)
        if dlg.ShowModal() == wx.ID_OK:
            self.parameterlist = dlg.GetLines()
            self.Update()
        dlg.Destroy()
        
    def OnDataChanged(self, event):
        '''OnDataChanged(self, event) --> None
        
        Updated the data list
        '''
        self.Update(update_script = False)
        
class EditCustomParameters(wx.Dialog):
    def __init__(self, parent, model, lines):
        wx.Dialog.__init__(self, parent, -1, 'Custom parameter editor')
        self.SetAutoLayout(True)
        self.model = model
        self.lines = lines
        self.var_name = 'cp'
        
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        name_ctrl_sizer = wx.GridBagSizer(2,3)
        
        col_labels = ['Name', 'Value']
        
        for item, index in zip(col_labels, range(len(col_labels))):
            label = wx.StaticText(self, -1, item)
            name_ctrl_sizer.Add(label,(0, index),flag=wx.ALIGN_LEFT,border=5)
        
        self.name_ctrl = wx.TextCtrl(self, -1, size = (120, -1))
        name_ctrl_sizer.Add(self.name_ctrl, (1,0),\
            flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
        self.value_ctrl = wx.TextCtrl(self, -1, size = (120, -1))
        name_ctrl_sizer.Add(self.value_ctrl, (1,1),\
            flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
        self.add_button = wx.Button(self, -1, 'Add')
        name_ctrl_sizer.Add(self.add_button, (1,2), \
            flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
        sizer.Add(name_ctrl_sizer)
        self.Bind(wx.EVT_BUTTON, self.OnAdd, self.add_button)

        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)
        
        self.listbox = MyHtmlListBox(self, -1, size = (-1,150),\
            style =  wx.BORDER_SUNKEN)
        self.listbox.SetItemList(self.lines)
        sizer.Add(self.listbox, 1, wx.GROW|wx.ALL, 10)
        
        self.delete_button = wx.Button(self, -1, 'Delete')
        sizer.Add(self.delete_button, 0, wx.CENTRE, 0)
        self.Bind(wx.EVT_BUTTON, self.OnDelete, self.delete_button)
        
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        #okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        button_sizer.Realize()
        self.Bind(wx.EVT_BUTTON, self.OnApply, okay_button)
        
        
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)
        
        sizer.Add(button_sizer,0, wx.ALIGN_RIGHT, 5)
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
        line = '%s.new_var(\'%s\', %s)'%(self.var_name,\
            self.name_ctrl.GetValue(), self.value_ctrl.GetValue())
        try:
            self.model.eval_in_model(line)
        except Exception, e:
            result = 'Could not evaluate the expression. The python error' +\
            'is: \n' + e.__repr__()
            dlg = wx.MessageDialog(self, result, 'Error in expression',
                               wx.OK | wx.ICON_WARNING)
            dlg.ShowModal()
            dlg.Destroy()
        else:
            self.lines.append(line)
            self.listbox.SetItemList(self.lines)
        
        
    def OnDelete(self, event):
        '''OnDelete(self, event) --> None
        
        Callback for deleting an entry
        '''
        result = 'Do you want to delete the expression?\n' + \
            'Remember to check if parameter is used elsewhere!'
        dlg = wx.MessageDialog(self, result, 'Delete expression?',
                               wx.YES_NO | wx.NO_DEFAULT| wx.ICON_INFORMATION)
        if dlg.ShowModal() == wx.ID_YES:
            self.lines.pop(self.listbox.GetSelection())
            self.listbox.SetItemList(self.lines)
        dlg.Destroy()
        
    def GetLines(self):
        '''GetLines(self) --> uservars lines [list]
        
        Returns the list user variables.
        '''
        return self.lines

class SimulationExpressionDialog(wx.Dialog):
    '''A dialog to edit the Simulation expression
    '''
    
    def __init__(self, parent, model, instruments,  sim_func, arguments, inst_name,
                 data_index):
        '''Creates a SimualtionExpressionDialog. 
        
        model - a Model object.
        instruments - a dictionary of possible instruments
        arguments - the arguments to the simulation function, a list of strings.
        sim_func - a string of the simulation function name.
        inst_name - the name of the current instrument.
        data_index - an integer for the current data index.
        '''
        
        self.model = model
        self.instruments = instruments
        self.available_sim_funcs = self.model.eval_in_model('model.SimulationFunctions.keys()')
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
            
                
        if max_val < 0:
            raise ValueError('Wrongly formatted function docs for the simulation functions')    
        
        gbs = wx.GridBagSizer(2, max_val)
        
        # Creating the column labels
        col_labels = ['Simulation', 'Instrument']
        [col_labels.append(arg) for arg in self.sim_args[sim_func] if not arg in col_labels]
        self.labels = []
        self.arg_controls = []
        for index in range(2 + max_val):
            label = wx.StaticText(self, -1, '')
            gbs.Add(label,(0, index),flag=wx.ALIGN_LEFT,border=5)
            self.labels.append(label)
            # If the expression is not an instrument or simulation function
            if index > 1:
                exp_ctrl = wx.TextCtrl(self, -1, size=(100, -1))               
                gbs.Add(exp_ctrl, (1,index),
                        flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border=5)
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
            
        for i in range(len(col_labels) - 2, len(self.arg_controls)):
            self.arg_controls[i].SetEditable(False)
            #self.arg_controls[i].Show(False)
            #self.arg_controls[i].SetValue('NA')
            
        # Creating the controls
        # Simulation choice control
        self.sim_choice = wx.Choice(self, -1, 
                                    choices = self.available_sim_funcs)
        self.Bind(wx.EVT_CHOICE, self.on_sim_change, self.sim_choice)
        self.sim_choice.SetSelection(self.available_sim_funcs.index(sim_func))
        gbs.Add(self.sim_choice, (1,0),\
            flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
       
        # Instrument choice control
        self.inst_choice = wx.Choice(self, -1, 
                                    choices=self.instruments.keys())
        #self.Bind(wx.EVT_CHOICE, self.on_inst_change, self.inst_choice)
        self.inst_choice.SetSelection(self.instruments.keys().index(expressions['Instrument']))
        gbs.Add(self.inst_choice, (1,1),\
            flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
        
            
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
       
        button_sizer.Realize()
        self.Bind(wx.EVT_BUTTON, self.on_ok_button, okay_button)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(gbs, 1, wx.GROW|wx.ALL, 10)
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)
        sizer.Add((-1,5))
        
        sizer.Add(button_sizer,0, wx.ALIGN_RIGHT, 5)
        sizer.Add((-1,5))
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
            new_labels.append(label.GetLabel() != arg_name)
            label.SetLabel(arg_name)
        # Clear the remaining column labels
        for label in self.labels[len(self.sim_args[new_sim]) + 2:]:
            label.SetLabel('')
                
        # Update the text controls - if needed
        for i in range(len(self.sim_args[new_sim])):
            #if new_labels[i]:
            if True:
                self.arg_controls[i].SetValue(self.sim_defaults[new_sim][i])
            self.arg_controls[i].SetEditable(True)
            #self.arg_controls[i].Show(True)
        # Hide and clear the remaining text controls
        for ctrl in self.arg_controls[len(self.sim_args[new_sim]):]:
            ctrl.SetEditable(False)
            ctrl.SetValue('')
            
            #ctrl.Show(False)
    
    def on_ok_button(self, event):
        '''Callback for pressing the ok button in the dialog'''
        expressions = self.GetExpressions()
        # Hack to get it working with d = data[0]
        exec 'd = data[%d]'%self.data_index in self.model.script_module.__dict__
        for exp in expressions:
            try:
                self.model.eval_in_model(exp)
            except Exception, e:
                result = ('Could not evaluate expression:\n%s.\n'%exp +
                ' The python error is: \n' + e.__repr__())
                dlg = wx.MessageDialog(self, result, 'Error in expression',
                               wx.OK | wx.ICON_WARNING)
                dlg.ShowModal()
                dlg.Destroy()
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
    def __init__(self, parent, model, expression=None, sim_func=None):
        wx.Dialog.__init__(self, parent, -1, 'Parameter editor')
        self.SetAutoLayout(True)
        self.model = model
        self.sim_func = sim_func
        
        gbs = wx.GridBagSizer(2, 3)
        
        col_labels = ['Object', 'Parameter', 'Expression']
        
        for item, index in zip(col_labels, range(len(col_labels))):
            label = wx.StaticText(self, -1, item)
            gbs.Add(label,(0, index),flag=wx.ALIGN_LEFT,border=5)
            
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
        self.obj_choice = wx.Choice(self, -1, choices = objlist)
        self.Bind(wx.EVT_CHOICE, self.on_obj_change, self.obj_choice)
        
        self.func_choice = wx.Choice(self, -1)
        # This will init func_choice
        self.obj_choice.SetSelection(0)
        
        gbs.Add(self.obj_choice, (1,0),\
            flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
        gbs.Add(self.func_choice, (1,1),\
            flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
            
        exp_right = ''
        if expression:
            p = expression.find('(')
            exp_left = expression[:p]
            obj = exp_left.split('.')[0]
            func = exp_left.split('.')[1]
            exp_right = expression[p+1:-1]
            obj_pos = [i for i in range(len(objlist)) if objlist[i] == obj]
            if len(obj_pos) > 0:
                self.obj_choice.SetSelection(obj_pos[0])
                self.on_obj_change(None)
                func_pos = [i for i in range(len(funclist[obj_pos[0]]))\
                                if funclist[obj_pos[0]][i] == func]
                if len(func_pos) > 0:
                    self.func_choice.SetSelection(func_pos[0])
                else:
                    raise ValueError('The function %s for object %s does not exist'%(func, obj))
            else:
                raise ValueError('The object %s does not exist'%obj)
    
        #self.expression_ctrl = wx.TextCtrl(self, -1, exp_right,\
        #                       size=(300, -1))

        self.expression_ctrl = ParameterExpressionCombo(par_dict, sim_func, self, -1, exp_right,
                                                        size=(300, -1))
        gbs.Add(self.expression_ctrl, (1,2),\
            flag=wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border=5)
        
        button_sizer = wx.StdDialogButtonSizer()
        okay_button = wx.Button(self, wx.ID_OK)
        okay_button.SetDefault()
        button_sizer.AddButton(okay_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_CANCEL))
        #apply_button = wx.Button(self, wx.ID_APPLY)
        #apply_button.SetDefault()
        #button_sizer.AddButton(apply_button)
        button_sizer.Realize()
        self.Bind(wx.EVT_BUTTON, self.OnApply, okay_button)
        #self.Bind(wx.EVT_BUTTON, self.OnApply, apply_button)
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(gbs, 1, wx.GROW|wx.ALL, 10)
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)
        sizer.Add((-1,5))
        
        sizer.Add(button_sizer,0, wx.ALIGN_RIGHT, 5)
        sizer.Add((-1,5))
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
        except Exception, e:
            result = 'Could not evaluate the expression. The python' +\
            'is: \n' + e.__repr__()
            dlg = wx.MessageDialog(self, result, 'Error in expression',
                               wx.OK | wx.ICON_WARNING)
            dlg.ShowModal()
            dlg.Destroy()
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
        

class SamplePlotPanel(wx.Panel):
    ''' Widget for plotting the scattering length density of 
    a sample.
    '''
    def __init__(self, parent, plugin, id = -1, color = None, dpi = None
    , style = wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        ''' Inits the plotpanel
        '''
        wx.Panel.__init__(self, parent)
        self.plot = PlotPanel(self, -1, color, dpi, style, **kwargs)
        self.plugin = plugin
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.plot, 1, wx.EXPAND|wx.GROW|wx.ALL)
        
        self.plot.update(None)
        self.plot.ax = self.plot.figure.add_subplot(111)
        box = self.plot.ax.get_position()
        self.plot.ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
        self.plot.ax.set_autoscale_on(True)
        self.plot.update = self.Plot
        self.SetSizer(sizer)
        self.plot.ax.set_autoscale_on(False)
        self.plot_dict = {}
        
    def Plot(self):
        ''' Plot(self) --> None
        
        Plotting the sample Sample.
        '''
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        model = self.plugin.GetModel().script_module
        #self.plot_dict = model.sample.SimSLD(None, model.inst)
        self.plot_dicts = []
        self.plot.ax.lines = []
        self.plot.ax.clear()
        i = 0
        data = self.plugin.GetModel().get_data()
        sld_units = []

        if self.plugin.sim_returns_sld and model._sim:
            # New style sim function with one sld for each simulation
            self.plot_dicts = model.SLD
            for sim in range(len(self.plot_dicts)):
                if data[sim].show:
                    for key in self.plot_dicts[sim]:
                        is_imag = key[:2] == 'Im' or key[:4] == 'imag'
                        if (is_imag and self.plugin.show_imag_sld) or not is_imag:
                            if key != 'z' and key != 'SLD unit':
                                label = data[sim].name + '\n' + key
                                self.plot.ax.plot(self.plot_dicts[sim]['z'], self.plot_dicts[sim][key],\
                                                  colors[i%len(colors)], label = label)

                                if self.plot_dicts[sim].has_key('SLD unit'):
                                    if not self.plot_dicts[sim]['SLD unit'] in sld_units:
                                        sld_units.append(self.plot_dicts[sim]['SLD unit'])
                                i += 1
        else:
            # Old style plotting just one sld
            if self.plugin.GetModel().compiled:
                plot_dict = model.sample.SimSLD(None, None, model.inst)
                self.plot_dicts = [plot_dict]
                for key in self.plot_dicts[0]:
                    is_imag = key[:2] == 'Im' or key[:4] == 'imag'
                    if (is_imag and self.plugin.show_imag_sld) or not is_imag:
                        if key != 'z' and key != 'SLD unit':
                            label = key
                            self.plot.ax.plot(self.plot_dicts[0]['z'], self.plot_dicts[0][key],\
                                              colors[i%len(colors)], label = label)

                            if self.plot_dicts[0].has_key('SLD unit'):
                                if not self.plot_dicts[0]['SLD unit'] in sld_units:
                                    sld_units.append(self.plot_dicts[0]['SLD unit'])
                            i += 1
                    
        if i > 0:
            self.plot.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                                prop = {'size': 10}, ncol = 1)
        
            sld_unit = ', '.join(sld_units)
            self.plot.ax.yaxis.label.set_text('$\mathrm{\mathsf{SLD\,[%s]}}$'%(sld_unit))
            #if self.plot_dict.has_key('SLD unit'):
            #    self.plot.ax.yaxis.label.set_text('$\mathrm{\mathsf{SLD\,[%s]}}$'%(sld_unit))
            self.plot.ax.xaxis.label.set_text('$\mathrm{\mathsf{ z\,[\AA]}}$')
            wx.CallAfter(self.plot.flush_plot)
            self.plot.AutoScale()
    
    def SavePlotData(self, filename):
        ''' Save all the SLD profiles to file with filename.'''
        # Check so that there are a simulation to save
        try:
            self.plot_dicts
        except:
            self.plugin.ShowWarningDialog('No SLD data to save.'
                                          ' Simulate the model first and then save.')
            return
        base, ext = os.path.splitext(filename)
        if ext == '':
            ext = '.dat'
        data = self.plugin.GetModel().get_data()
        for sim in range(len(self.plot_dicts)):
            new_filename = (base + '%03d'%sim + ext)
            save_array = np.array([self.plot_dicts[sim]['z']])
            header = 'z\t' 
            for key in self.plot_dicts[sim]:
                if key != 'z' and key != 'SLD unit':
                    save_array = np.r_[save_array, [self.plot_dicts[sim][key]]]
                    header += key + '\t'
            f = open(new_filename, 'w')
            f.write("# File exported from GenX's Reflectivity plugin\n")
            f.write("# File created: %s\n"%time.ctime())
            f.write("# Simulated SLD for data set: %s\n"%data[sim].name)
            f.write("# Headers: \n")
            f.write('#' + header + '\n')
            np.savetxt(f, save_array.transpose())
            f.close()
        
        
        
class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        #self.parent = parent
        self.model_obj = self.GetModel()
        sample_panel = self.NewInputFolder('Sample')
        sample_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sample_panel.SetSizer(sample_sizer)
        self.defs = ['Instrument', 'Sample']
        self.sample_widget = SamplePanel(sample_panel, self)
        sample_sizer.Add(self.sample_widget, 1, wx.EXPAND|wx.GROW|wx.ALL)
        sample_panel.Layout()
        
        simulation_panel = self.NewInputFolder('Simulations')
        simulation_sizer = wx.BoxSizer(wx.HORIZONTAL)
        simulation_panel.SetSizer(simulation_sizer)
        self.simulation_widget = DataParameterPanel(simulation_panel, self)
        simulation_sizer.Add(self.simulation_widget, 1, wx.EXPAND|wx.GROW|wx.ALL)
        simulation_panel.Layout()
        
        self.sample_widget.SetUpdateCallback(self.UpdateScript)
        self.simulation_widget.SetUpdateScriptFunc(self.UpdateScript)
        
        # Create the SLD plot
        sld_plot_panel = self.NewPlotFolder('SLD')
        sld_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sld_plot_panel.SetSizer(sld_sizer)
        self.sld_plot = SamplePlotPanel(sld_plot_panel, self)
        sld_sizer.Add(self.sld_plot, 1, wx.EXPAND|wx.GROW|wx.ALL)
        sld_plot_panel.Layout()
        
        if self.model_obj.filename != '' and self.model_obj.script != '':
            self.ReadModel()
        else:
            self.CreateNewModel()
        
        # Create a menu for handling the plugin
        menu = self.NewMenu('Reflec')
        self.mb_export_sld = wx.MenuItem(menu, wx.NewId(), 
                                         "Export SLD...", 
                                         "Export the SLD to a ASCII file", 
                                         wx.ITEM_NORMAL)
        menu.AppendItem(self.mb_export_sld)
        self.mb_show_imag_sld = wx.MenuItem(menu, wx.NewId(), 
                                         "Show Im SLD", 
                                         "Toggles showing the imaginary part of the SLD", 
                                         wx.ITEM_CHECK)
        menu.AppendItem(self.mb_show_imag_sld)
        self.mb_show_imag_sld.Check(False)
        self.show_imag_sld = self.mb_show_imag_sld.IsChecked()
        self.mb_autoupdate_sld = wx.MenuItem(menu, wx.NewId(), 
                                         "Autoupdate SLD", 
                                         "Toggles autoupdating the SLD during fitting", 
                                         wx.ITEM_CHECK)
        menu.AppendItem(self.mb_autoupdate_sld)
        self.mb_autoupdate_sld.Check(False)
        #self.mb_autoupdate_sld.SetCheckable(True)
        self.parent.Bind(wx.EVT_MENU, self.OnExportSLD, self.mb_export_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnAutoUpdateSLD, self.mb_autoupdate_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnShowImagSLD, self.mb_show_imag_sld)
        
        self.StatusMessage('Reflectivity plugin loaded')
        
    def UpdateScript(self, event):
        self.WriteModel()
        
    def OnAutoUpdateSLD(self, evt):
        #self.mb_autoupdate_sld.Check(not self.mb_autoupdate_sld.IsChecked())
        pass
    
    def OnShowImagSLD(self, evt):
        self.show_imag_sld = self.mb_show_imag_sld.IsChecked()
        self.sld_plot.Plot()
    
    def OnExportSLD(self, evt):
        dlg = wx.FileDialog(self.parent, message="Export SLD to ...", 
                            defaultFile="",
                            wildcard="Dat File (*.dat)|*.dat",
                            style=wx.SAVE | wx.CHANGE_DIR 
                            )
        if dlg.ShowModal() == wx.ID_OK:
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
                except IOError, e:
                    self.ShowErrorDialog(e.__str__())
                except Exception, e:
                    outp = StringIO.StringIO()
                    traceback.print_exc(200, outp)
                    val = outp.getvalue()
                    outp.close()
                    self.ShowErrorDialog('Could not save the file.'
                                         ' Python Error:\n%s'%(val,))
        dlg.Destroy()
        
    def OnNewModel(self, event):
        ''' Create a new model
        '''
        dlg = wx.SingleChoiceDialog(self.parent, 'Choose a model type to use',\
                'Models', _avail_models, 
                wx.CHOICEDLG_STYLE
                )

        if dlg.ShowModal() == wx.ID_OK:
            self.CreateNewModel('models.%s'%dlg.GetStringSelection())
        dlg.Destroy()
    
    def OnDataChanged(self, event):
        ''' Take into account changes in data..
        '''
        if event.new_model:
            return

        if event.data_moved or event.deleted or event.new_data\
            or event.name_change:
            names = [data_set.name for data_set in self.GetModel().get_data()]
            self.simulation_widget.SetDataList(names)

            expl = self.simulation_widget.GetExpressionList()

            if len(names)-len(expl) == 1:
                # Data set has been added:
                expl.append([])
                self.insert_new_data_segment(len(expl)-1)

            sims, insts, args = self.simulation_widget.GetSimArgs()

            if event.deleted:
                pos = range(len(expl))
                [self.remove_data_segment(pos[-index-1]) for index in\
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
                    #Moving down...
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
            if self.GetModel().script != '':
                self.WriteModel()
                self.simulation_widget.Update()
                if event.name_change:
                    self.sld_plot.Plot()
            else:
                self.simulation_widget.Update(update_script=False)
        else:
            if event.data_changed:
                self.sld_plot.Plot()
        
    def OnOpenModel(self, event):
        '''OnOpenModel(self, event) --> None
        
        Loads the sample into the plugin...
        '''
        
        self.ReadModel()

    def OnSimulate(self, event):
        '''OnSimulate(self, event) --> None
        
        Updates stuff after simulation
        '''
        # Calculate and update the sld plot
        self.sld_plot.Plot()
        
    def OnFittingUpdate(self, event):
        '''OnSimulate(self, event) --> None
        
        Updates stuff during fitting
        '''
        # Calculate and update the sld plot
        if self.mb_autoupdate_sld.IsChecked():
            self.sld_plot.Plot()
        #self.sample_widget.Update(update_script=False)

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
        
    def CreateNewModel(self, modelname = 'models.spec_nx'):
        '''Init the script in the model to yield the 
        correct script for initilization
        '''
        script = 'from numpy import *\n'
        script += 'import %s as model\n'%modelname
        script += 'from models.utils import UserVars, fp, fw, bc, bw\n\n'
        
        for item in self.defs:
            script += '# BEGIN %s DO NOT CHANGE\n'%item
            script += '# END %s\n\n'%item
        
        script += '# BEGIN Parameters DO NOT CHANGE\n'
        script += 'cp = UserVars()\n'
        script += '# END Parameters\n\n'
        script += 'SLD = []\n'
        script += 'def Sim(data):\n'
        script += '    I = []\n'
        script += '    SLD[:] = []\n'
        nb_data_sets = len(self.GetModel().get_data())
        for i in range(nb_data_sets):
            script += '    # BEGIN Dataset %i DO NOT CHANGE\n'%i
            script += '    d = data[%i]\n'%i
            script += '    I.append(sample.SimSpecular(d.x, inst))\n'
            script += '    if _sim: SLD.append(sample.SimSLD(None, None, inst))\n'
            script += '    # END Dataset %i\n'%i
        script += '    return I\n'
        
        self.sim_returns_sld = True
        
        self.SetModelScript(script)
        self.CompileScript()
        self.model = self.GetModel().script_module.model
        
        names = ['Amb','Sub']
        Amb = self.model.Layer()
        Sub = self.model.Layer()
        sample = self.model.Sample(Stacks = [], Ambient = Amb, Substrate = Sub)
        instrument = self.model.Instrument()
        #self.sample_widget.SetSample(sample, names)
        self.sampleh = SampleHandler(sample, names)
        self.sampleh.model = self.model
        self.sample_widget.sampleh = self.sampleh
        self.sample_widget.model = self.model
        self.sample_widget.SetInstrument({'inst':instrument})
        
        names = [data_set.name for data_set in self.GetModel().get_data()]
        self.simulation_widget.SetDataList(names)
        self.simulation_widget.SetParameterList([])
        # An empty list to the expression widget...
        self.simulation_widget.SetExpressionList([[] for item in names])
        self.simulation_widget.SetSimArgs(['Specular']*nb_data_sets,
                                          ['inst']*nb_data_sets, 
                                          [['d.x'] for i in range(nb_data_sets)])
        self.simulation_widget.Update(update_script = True)
        
        self.sample_widget.Update(update_script = True)
        #self.WriteModel()
    
    def WriteModel(self):
        script = self.GetModel().get_script()
        
        # Instrument script creation
        code = 'from models.utils import create_fp, create_fw\n'
        instruments = self.sample_widget.instruments
        for inst_name in instruments:
            code += ('%s = model.'%inst_name + 
                     instruments[inst_name].__repr__() + '\n')
            code += '%s_fp = create_fp(%s.wavelength);'%(inst_name, inst_name)
            code += ' %s_fw = create_fw(%s.wavelength)\n\n'%(inst_name, inst_name)
        code += ('fp.set_wavelength(inst.wavelength); '
                 + 'fw.set_wavelength(inst.wavelength)\n')
        script = self.insert_code_segment(script, 'Instrument', code)
        
        # Sample script creation
        layer_code, stack_code, sample_code = self.sampleh.getCode()
        code = layer_code + '\n' + stack_code + '\n' + sample_code
        script = self.insert_code_segment(script, 'Sample', code)
        
        # User Vars (Parameters) script creation
        code = 'cp = UserVars()\n'
        code += ''.join([line + '\n' for line in\
            self.simulation_widget.GetParameterList()])
        script = self.insert_code_segment(script, 'Parameters', code)
        
        # Expressions evaluted during simulations (parameter couplings) script creation
        sim_funcs, insts, args = self.simulation_widget.GetSimArgs()
        for (i,exps) in enumerate(self.simulation_widget.GetExpressionList()):
            exp = [ex + '\n' for ex in exps]
            exp.append('d = data[%i]\n'%i)
            str_arg = ', '.join(args[i])
            exp.append('I.append(sample.'
                       'Sim%s(%s, %s))\n'%(sim_funcs[i], str_arg, 
                                           insts[i]))
            if self.sim_returns_sld:
                exp.append('if _sim: SLD.append(sample.'
                           'SimSLD(None, None, %s))\n'%insts[i])
            code = ''.join(exp)
            script = self.insert_code_segment(script, 'Dataset %i'%i, code)
        
        self.SetModelScript(script)
        
    def insert_new_data_segment(self, number):
        '''insert_new_data_segment(self, number) --> None
        
        Inserts a new data segment into the script
        '''
        code = self.GetModel().get_script()
        script_lines = code.splitlines(True)
        line_index = 0
        found = 0
        for line in script_lines[line_index:]:
            line_index += 1
            if line.find('    return I') != -1:
                found = 1
                break
            
        if found < 1:
            raise LookupError('Could not find "return I" in the script')

        self.simulation_widget.AppendSim('Specular', 'inst', ['d.x'])
        
        script = ''.join(script_lines[:line_index-1])
        script += '    # BEGIN Dataset %i DO NOT CHANGE\n'%number
        script += '    d = data[%i]\n'%number
        script += '    I.append(sample.SimSpecular(d.x, inst))\n'
        script += '    if _sim: SLD.append(sample.SimSLD(None, None, inst))\n'
        script += '    # END Dataset %i\n'%number
        script += ''.join(script_lines[line_index-1:])
        self.SetModelScript(script)
        
    def remove_data_segment(self, number):
        '''remove_data_segment(self, number) --> None
        
        Removes data segment number
        '''
        code = self.GetModel().get_script()
        found = 0
        script_lines = code.splitlines(True)
        start_index = -1
        stop_index = -1
        for line in range(len(script_lines)):
            if script_lines[line].find('# BEGIN Dataset %i'%number) != -1:
                start_index = line+1
            if script_lines[line].find('# END Dataset %i'%number) != -1:
                stop_index = line-1
                break
            
        # Check so everything have preceeded well
        if stop_index < 0 and start_index < 0:
            raise LookupError('Code segement: %s could not be found'%descriptor)
        
        script = ''.join(script_lines[:start_index-1])
        script += ''.join(script_lines[stop_index+2:])
        self.SetModelScript(script)
        
    def find_code_segment(self, code, descriptor):
        '''find_code_segment(self, code, descriptor) --> string
        
        Finds a segment of code between BEGIN descriptor and END descriptor
        returns a LookupError if the segement can not be found
        '''
        
        return find_code_segment(code, descriptor)
    
    def insert_code_segment(self, code, descriptor, insert_code):
        '''insert_code_segment(self, code, descriptor, insert_code) --> None
        
        Inserts code segment into the file. See find_code segment.
        '''
        found = 0
        script_lines = code.splitlines(True)
        start_index = -1
        stop_index = -1
        for line in range(len(script_lines)):
            if script_lines[line].find('# BEGIN %s'%descriptor) != -1:
                start_index = line+1
            if script_lines[line].find('# END %s'%descriptor) != -1:
                stop_index = line-1
                break
            
        # Check so everything have preceeded well
        if stop_index < 0 and start_index < 0:
            raise LookupError('Code segement: %s could not be found'%descriptor)
        
        # Find the tablevel
        #tablevel = len([' ' for char in script_lines[stop_index+1]\
        #    if char == ' '])
        tablevel = len(script_lines[stop_index+1])\
                    - len(script_lines[stop_index+1].lstrip())

        # Make the new code tabbed
        tabbed_code = [' '*tablevel + line for line in\
            insert_code.splitlines(True)]
        # Replace the new code segment with the new
        new_code = ''.join(script_lines[:start_index] + tabbed_code\
            + script_lines[stop_index+1:])
            
        return new_code
        
    def ReadModel(self):
        '''ReadModel(self)  --> None
        
        Reads in the current model and locates layers and stacks
        and sample defined inside BEGIN Sample section.
        '''
        self.StatusMessage('Compiling the script...')
        try:
            self.CompileScript()
        except modellib.GenericError, e:
            self.ShowErrorDialog(str(e))
            self.StatusMessage('Error when compiling the script')
            return
        except Exception, e:
            outp = StringIO.StringIO()
            traceback.print_exc(200, outp)
            val = outp.getvalue()
            outp.close()
            self.ShowErrorDialog(val)
            self.Statusmessage('Fatal Error - compling, Reflectivity')
            return
        self.StatusMessage('Script compiled!')
        
        self.StatusMessage('Trying to interpret the script...')
        
        script = self.GetModel().script
        code = self.find_code_segment(script, 'Instrument')
        re_layer = re.compile('([A-Za-z]\w*)\s*=\s*model\.Instrument\s*\((.*)\)\n')
        instrument_strings = re_layer.findall(code)
        instrument_names = [t[0] for t in instrument_strings]
        
        if len(instrument_names) == 0:
            self.ShowErrorDialog('Could not find any Instruments in the' +\
                ' model script. Check the script.')
            self.StatusMessage('ERROR No Instruments in script')
            return
        
        if not 'inst' in instrument_names:
            self.ShowErrorDialog('Could not find the default' +
                                 ' Instrument, inst, in the' +
                                 ' model script. Check the script.')
            self.StatusMessage('ERROR No Instrument called inst in script')
            return
        
        # Get the current script and split the lines into list items
        script_lines = self.GetModel().get_script().splitlines(True)
        # Try to find out if the script works with multiple SLDs
        for line in script_lines:
            if line.find('SLD[:]') != -1:
                self.sim_returns_sld = True
                break
            else:
                self.sim_returns_sld = False
        script = ''
        # Locate the Sample definition
        line_index = 0
        # Start by finding the right section
        found = 0
        for line in script_lines[line_index:]:
            line_index += 1
            if line.find('# BEGIN Sample') != -1:
                found += 1
                break
            
        sample_text = ''
        for line in script_lines[line_index:]:
            line_index += 1
            sample_text += line
            if line.find('# END Sample') != -1:
                found += 1
                break
        
        if found != 2:
            self.ShowErrorDialog('Could not find the sample section' + \
                ' in the model script.\n Can not load the sample in the editor.')
            self.StatusMessage('ERROR No sample section in script')
            return
        
        re_layer = re.compile('([A-Za-z]\w*)\s*=\s*model\.Layer\s*\((.*)\)\n')
        re_stack = re.compile('([A-Za-z]\w*)\s*=\s*model\.Stack\s*\(\s*Layers=\[(.*)\].*\n')
        
        layers = re_layer.findall(sample_text)
        layer_names = [t[0] for t in layers]
        stacks = re_stack.findall(sample_text)

        if len(layer_names) == 0:
            self.ShowErrorDialog('Could not find any Layers in the' +\
                ' model script. Check the script.')
            self.StatusMessage('ERROR No Layers in script')
            return
        
        # Now its time to set all the parameters so that we have the strings
        # instead of the evaluated value - looks better
        for lay in layers:
            for par in lay[1].split(','):
                vars = par.split('=')
                exec '%s.%s = "%s"'%(lay[0], vars[0].strip(), vars[1].strip())\
                                    in self.GetModel().script_module.__dict__
        
        all_names = [layer_names.pop(0)]
        for stack in stacks:
            all_names.append(stack[0])
            first_name = stack[1].split(',')[0].strip()
            # check so stack is non-empty
            if first_name != '':
                # Find all items above the first name in the stack
                while(layer_names[0] != first_name):
                    all_names.append(layer_names.pop(0))
                all_names.append(layer_names.pop(0))
        all_names += layer_names
        
        # Load the simulation parameters
        script = self.GetModel().script
        sim_exp = []
        data_names = []
        data = self.GetModel().get_data()
        # Lists holding the simulation function arguments
        sim_funcs = []
        sim_args = []
        insts = []
        try:
            for i in range(len(data)):
                code = self.find_code_segment(script, 'Dataset %i'%i)
                sim_exp.append([])
                data_names.append(data[i].name)
                #for line in code.splitlines()[:-1]:
                #    sim_exp[-1].append(line.strip())
                for line in code.splitlines():
                    if (line.find('I.append') == -1 and line.find('SLD.append') == -1
                        and line.find('d = data') == -1):
                        # The current line is a command for a parameter
                        sim_exp[-1].append(line.strip()) 
                    elif line.find('I.append') > -1:
                        # The current line is a simulations
                        (tmp, sim_func, args) = line.split('(',2)
                        sim_funcs.append(sim_func[10:])
                        sim_args.append([arg.strip() for arg in args.split(',')[:-1]])
                        insts.append(args.split(',')[-1][:-2].strip())
        except LookupError:
            self.ShowErrorDialog('Could not locate all data sets in the'
            ' script. There should be %i datasets'%len(data))
            self.StatusMessage('ERROR No Layers in script')
            return
        # Load the custom parameters:
        code = self.find_code_segment(script, 'Parameters')
        uservars_lines = code.splitlines()[1:]
        
        self.model = self.GetModel().script_module.model
        sample = self.GetModel().script_module.sample
        
        self.sampleh = SampleHandler(sample, all_names)
        self.sampleh.model = self.model
        self.sample_widget.sampleh = self.sampleh
        self.sample_widget.model = self.model
        instruments = {}
        for name in instrument_names:
            instruments[name] = getattr(self.GetModel().script_module, name)
        self.sample_widget.SetInstrument(instruments)
        
        self.simulation_widget.SetDataList(data_names)
        self.simulation_widget.SetExpressionList(sim_exp)
        self.simulation_widget.SetParameterList(uservars_lines)
        
        self.simulation_widget.SetSimArgs(sim_funcs, insts, sim_args)
        
        self.sample_widget.Update(update_script = False)
        self.simulation_widget.Update(update_script = False)
        # The code have a tendency to screw up the model slightly when compiling it - the sample will be connected to
        # to the module therefore reset the compiled flag so that the model has to be recompiled before fitting.
        self.GetModel().compiled = False
        self.StatusMessage('New sample loaded to plugin!')

        
def find_code_segment(code, descriptor):
        '''find_code_segment(code, descriptor) --> string
        
        Finds a segment of code between BEGIN descriptor and END descriptor
        returns a LookupError if the segement can not be found
        '''
        found = 0
        script_lines = code.splitlines(True)
        line_index = 0
        for line in script_lines[line_index:]:
            line_index += 1
            if line.find('# BEGIN %s'%descriptor) != -1:
                found += 1
                break
            
        text = ''
        for line in script_lines[line_index:]:
            line_index += 1
            if line.find('# END %s'%descriptor) != -1:
                found += 1
                break
            text += line
        
        if found != 2:
            raise LookupError('Code segement: %s could not be found'%descriptor)
        
        return text
        
if __name__ == '__main__':
    import models.interdiff as Model
    
    nSi=3.0
    Fe=Model.Layer(d=10,sigmar=3.0,n=1-2.247e-5+2.891e-6j)
    Si=Model.Layer(d=15,sigmar=3.0,n='nSi')
    sub=Model.Layer(sigmar=3.0,n=1-7.577e-6+1.756e-7j)
    amb=Model.Layer(n=1.0)
    stack=Model.Stack(Layers=[Fe,Si],Repetitions=20)
    stack2=Model.Stack(Layers=[Fe,Si])
    sample=Model.Sample(Stacks=[stack,stack2],Ambient=amb,Substrate=sub,eta_z=500.0,eta_x=100.0)
    print sample
    inst=Model.Instrument(Wavelength=1.54,Coordinates=1)
    s = ['Amb','stack1','Fe1','Si1','s2','Fe2','Si2','Sub']
    sh=SampleHandler(sample,s)
    sh.getStringList()

    class MyApp(wx.App):
        def OnInit(self):
            #wx.InitAllImageHandlers()
            frame = SampleFrame(None, -1, "Sample",sh)
            frame.Show(True)
            self.SetTopWindow(frame)
            return True


    print Si.getN().__repr__()
    app = MyApp(0)
    app.MainLoop()
