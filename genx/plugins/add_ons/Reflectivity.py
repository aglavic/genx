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
import  wx

import numpy as np
import sys, os, re, time, StringIO, traceback


# Make Modules a search path for python..
#sys.path.insert(1,os.getcwd()+'/Models')

from help_modules.custom_dialog import *
import help_modules.reflectivity_images as images

_avail_models = ['spec_nx', 'interdiff', 'xmag', 'mag_refl']

class SampleHandler:
    def __init__(self,sample,names):
        self.sample=sample
        self.names=names
        self.getStringList()
        
    def getStringList(self, html_encoding = False):
        '''
        Function to generate a lsit of strings that gives
        a visual representation of the sample.
        '''
        #print 'getStringList sample:'
        #print self.sample
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
            if slist[item][0]=='L' and item != 0 and item != len(slist) - 1:
                if html_encoding:
                    slist[item]='<code>&nbsp;&nbsp;&nbsp;<b>'+self.names[-item-1]+'</b> = '+slist[item] + '</code>'
                    #slist[item] = self.htmlize(self.names[-item-1] + ' = model.' + slist[item])
                else:
                    slist[item] = self.names[-item-1] + ' = model.' + slist[item]
            else:
                if item == 0 or item == len(slist) - 1:
                    # This is then the ambient or substrates
                    if html_encoding:
                        slist[item]='<code><b>' + self.names[-item-1]+'</b> = '+slist[item] + '</code>'
                        #slist[item] = self.htmlize(self.names[-item-1] + ' = model.' + slist[item])
                    else:
                        slist[item] = self.names[-item-1] + ' = model.' + slist[item]
                else:
                    # This is a stack!
                    if html_encoding:
                        slist[item]='<font color = "BLUE"><code><b>' + self.names[-item-1]+'</b> = '+slist[item] + '</code></font>'
                        #slist[item] = self.htmlize(self.names[-item-1] + ' = model.' + slist[item])
                    else:
                        slist[item] = self.names[-item-1] + ' = model.' + slist[item]
        poslist.append((None,None))
        slist.reverse()
        poslist.reverse()
        self.poslist=poslist
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
            
        #print layer_code,stack_code,sample_code
        return layer_code,stack_code, sample_code

    def getItem(self,pos):
        '''
        Returns the item (Stack or Layer) at position pos
        '''
        if pos==0:
            return self.sample.Ambient
        if pos==len(self.poslist)-1:
            return self.sample.Substrate
        stack=self.sample.Stacks[self.poslist[pos][0]]
        if self.poslist[pos][1]==None:
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
            #print self.poslist
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
            #print 'Hej'
            spos=(self.poslist[1][0],self.poslist[1][1])#+1
            #spos=(None,None)
        if pos==len(self.poslist)-1:
            #print 'daa'
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
                #print spos[0]
                if last:
                    self.names.insert(pos, name)
                else:
                    #print 'spos : ', spos
                    if spos[1] >= 0:
                        self.names.insert(pos+1,name)
                    else:
                        self.names.insert(pos+len(self.sample.Stacks[spos[0]].Layers)+1,name)
                #print 'spos ', spos
                #print 'poslist ', self.poslist
                if last:
                    self.sample.Stacks[spos[0]].Layers.insert(0,layer)
                else:
                    if self.poslist[pos][1] == None:
                        #print spos
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
                #print spos[0]
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
                #print temps
                for index in range(len(temp.Layers)+1):
                    self.names.insert(pos-len(self.sample.Stacks[self.poslist[pos][0]].Layers)-1,temps[-index-1])
                self.sample.Stacks.insert(self.poslist[pos][0]+1,temp)
                return self.getStringList()
            else: #i.e. it is a layer we move
                if pos > 2:
                    temp=self.sample.Stacks[self.poslist[pos][0]].Layers.pop(self.poslist[pos][1])
                    temps=self.names.pop(pos)
                    if self.poslist[pos-1][1]==None: # Next item a Stack i.e. jump up
                        #print self.sample.Stacks[self.poslist[pos-2][0]]
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
                    #print temps
                    #print pos
                    for index in range(len(temp.Layers)+1):
                        self.names.insert(pos+len(self.sample.Stacks[self.poslist[pos][0]-1].Layers)+1,temps[-index-1])
                    self.sample.Stacks.insert(self.poslist[pos][0]-1,temp)
                    return self.getStringList()
                else:
                    return None
                    
            else: #i.e. it is a layer we move
                if pos < len(self.poslist)-2:
                    #print 'Moving a layer'
                    temp=self.sample.Stacks[self.poslist[pos][0]].Layers.pop(self.poslist[pos][1])
                    temps=self.names.pop(pos)
                    if self.poslist[pos+1][1]==None: # Next item a Stack i.e. jump down
                        #print self.sample.Stacks[self.poslist[pos+1][0]]
                        self.sample.Stacks[self.poslist[pos+1][0]].Layers.insert(len(self.sample.Stacks[self.poslist[pos+1][0]].Layers),temp)
                        self.names.insert(pos+1,temps)
                    else: #Moving inside a stack
                        self.sample.Stacks[self.poslist[pos][0]].Layers.insert(self.poslist[pos][1]-1,temp)#-2
                        self.names.insert(pos+1,temps)
                    return self.getStringList()
        else:
            return None

class MyHtmlListBox(wx.HtmlListBox):

    def __init__(self, parent, id, size = (-1, -1), style = wx.BORDER_SUNKEN):
        wx.HtmlListBox.__init__(self, parent, id, size = size, 
                                style = wx.BORDER_SUNKEN)
        self.SetItemList(['Starting up...'])

    def SetItemList(self, list):
        self.html_items = list
        self.SetItemCount(len(list))
        #self.RefreshAll()
        self.Refresh()
        
    def OnGetItem(self, n):
        return self.html_items[n]
   

class SamplePanel(wx.Panel):
    def __init__(self, parent, plugin, refindexlist = []):
        wx.Panel.__init__(self, parent)
        self.refindexlist = refindexlist
        self.plugin = plugin
        
        boxver = wx.BoxSizer(wx.HORIZONTAL)
        boxhor = wx.BoxSizer(wx.VERTICAL)
        boxbuttons=wx.BoxSizer(wx.HORIZONTAL)
        boxhor.Add((-1,2))
        boxhor.Add(boxbuttons, 0)
        boxhor.Add((-1,2))
        self.listbox = MyHtmlListBox(self, -1, style =  wx.BORDER_SUNKEN)
        #self.listbox.SetItemList(self.sampleh.getStringList())
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.lbDoubleClick , self.listbox)
        boxhor.Add(self.listbox, 1, wx.EXPAND)

        if os.name == 'nt':
            size = (24, 24)
        else:
            size = (-1, -1)
        space = (2, -1)
        #InsertLayButton = wx.Button(self,-1, "Insert Layer")
        InsertLayButton =  wx.BitmapButton(self, -1
        , images.getinsert_layerBitmap(), size = size, style=wx.NO_BORDER)
        boxbuttons.Add(InsertLayButton,0)
        boxbuttons.Add(space)
        InsertLayButton.SetToolTipString('Insert a new layer')
        self.Bind(wx.EVT_BUTTON, self.InsertLay, InsertLayButton)
        #InsertStackButton=wx.Button(self,-1, "Insert Stack")
        InsertStackButton = wx.BitmapButton(self, -1
        , images.getinsert_stackBitmap(), size = size, style=wx.NO_BORDER)
        boxbuttons.Add(InsertStackButton, 0)
        boxbuttons.Add(space)
        InsertStackButton.SetToolTipString('Insert a new stack')
        self.Bind(wx.EVT_BUTTON, self.InsertStack, InsertStackButton)
        #DeleteButton=wx.Button(self,-1, "Delete")
        DeleteButton = wx.BitmapButton(self, -1
        , images.getdeleteBitmap(), size = size, style=wx.NO_BORDER)
        boxbuttons.Add(DeleteButton, 0)
        boxbuttons.Add(space)
        DeleteButton.SetToolTipString('Delete')
        self.Bind(wx.EVT_BUTTON, self.DeleteSample, DeleteButton)
        CNameButton = wx.BitmapButton(self, -1
        , images.getchange_nameBitmap(), size = size, style=wx.NO_BORDER)
        boxbuttons.Add(CNameButton, 0)
        boxbuttons.Add(space)
        DeleteButton.SetToolTipString('Change Name')
        self.Bind(wx.EVT_BUTTON, self.ChangeName, CNameButton)
        #MUpButton=wx.Button(self,-1, "MoveUp")
        MUpButton = wx.BitmapButton(self, -1
        , images.getmove_upBitmap(), size = size, style=wx.NO_BORDER)
        boxbuttons.Add(MUpButton, 0)
        boxbuttons.Add(space)
        MUpButton.SetToolTipString('Move item up')
        self.Bind(wx.EVT_BUTTON, self.MoveUp, MUpButton)
        #MDownButton=wx.Button(self,-1, "MoveDown")
        MDownButton = wx.BitmapButton(self, -1
        , images.getmove_downBitmap(), size = size, style=wx.NO_BORDER)
        boxbuttons.Add(MDownButton, 0)
        boxbuttons.Add(space)
        MDownButton.SetToolTipString('Move item down')
        self.Bind(wx.EVT_BUTTON, self.MoveDown, MDownButton)
        #SampleButton = wx.Button(self,-1, "Sample")
        SampleButton = wx.BitmapButton(self, -1
        , images.getsampleBitmap(), size = size, style=wx.NO_BORDER)
        boxbuttons.Add(SampleButton, 0)
        boxbuttons.Add(space)
        SampleButton.SetToolTipString('Edit sample parameters')
        self.Bind(wx.EVT_BUTTON, self.EditSampleParameters, SampleButton)
        #InstrumentButton = wx.Button(self,-1, "Instrument")
        InstrumentButton = wx.BitmapButton(self, -1
        , images.getinstrumentBitmap(), size = size, style=wx.NO_BORDER)
        boxbuttons.Add(InstrumentButton, 0)
        boxbuttons.Add(space)
        InstrumentButton.SetToolTipString('Edit instrument')
        self.Bind(wx.EVT_BUTTON, self.EditInstrument, InstrumentButton)
        
        #boxhor.Add(boxbuttons)
        boxver.Add(boxhor, 1 ,  wx.EXPAND)
        
        boxhorpar=wx.BoxSizer(wx.HORIZONTAL)

        #self.tc=[]
        #for item in self.model.SampleParameters.keys():
        #    if item != 'Stacks' and item != 'Substrate' and item != 'Ambient':
        #        boxhorpar.Add(wx.StaticText(self,-1,item+': '),0)
        #        self.tc.append(wx.TextCtrl(self, -1,\
        #         str(self.sampleh.sample.__getattribute__(item)),\
        #         validator = FloatObjectValidator()))
        #        boxhorpar.Add(self.tc[-1],0)
        #boxver.Add(boxhorpar,0)
        self.SetSizer(boxver)
        
        self.update_callback = lambda event:''
        
    def SetUpdateCallback(self, func):
        ''' SetUpdateCallback(self, func) --> None
        
        Sets the update callback will be called when the sample is updated.
        The call is on the form func(event)
        '''
        self.update_callback = func
        
    def Update(self):
        sl = self.sampleh.getStringList(html_encoding = True)
        self.listbox.SetItemList(sl)
        self.update_callback(None)
                
    def SetSample(self, sample, names):
        #print 'SetSample sample:'
        #print sample, '\n'
        self.sampleh.sample = sample
        self.sampleh.names = names
        self.Update()
        
    def EditSampleParameters(self, evt):
        validators = {}
        vals = {}
        pars = []
        items = []
        try:
            #print self.model.sample_string_choices
            string_choices = self.model.sample_string_choices
        except Exception, e:
            string_choices = {}
        for item in self.model.SampleParameters:
            if item != 'Stacks' and item != 'Substrate' and item != 'Ambient':
                if string_choices.has_key(item):
                    validators[item] = string_choices[item]
                else:
                    validators[item] = FloatObjectValidator()
                val = self.sampleh.sample.__getattribute__(item)
                vals[item] = val
                pars.append(item)
                items.append((item, val))
        #print items, validators
        try:
            groups = self.model.SampleGroups
        except Exception:
            groups = False
        try:
            units = self.model.SampleUnits
        except Exception:
            units = False
            
        #dlg = ValidateDialog(self,items,validators,title='Layer Editor')
        dlg = ValidateDialog(self, pars, vals, validators,
                             title = 'Sample Editor', groups = groups,
                             units = units)
        
        if dlg.ShowModal()==wx.ID_OK:
            #print 'Pressed OK'
            vals=dlg.GetValues()
            #print vals
            for par in pars:
                if string_choices.has_key(par):
                    self.sampleh.sample.__setattr__(par, vals[par])
                else:
                    self.sampleh.sample.__setattr__(par, float(vals[par]))
            self.Update()
        else:
            #print 'Pressed Cancel'
            pass
        dlg.Destroy()
    
    def SetInstrument(self, instrument):
        '''SetInstrument(self, instrument) --> None
        
        Sets the intrument
        '''
        self.instrument = instrument
    
    def EditInstrument(self, evt):
        #validators = []
        validators = {}
        vals = {}
        items = []
        pars = []
        for item in self.model.InstrumentParameters:
            if self.model.instrument_string_choices.has_key(item):
                #validators.append(self.model.instrument_string_choices[item])
                validators[item] = self.model.instrument_string_choices[item]
            else:
                #validators.append(FloatObjectValidator())
                validators[item] = FloatObjectValidator()
            val = self.instrument.__getattribute__(item)
            vals[item] = val
            pars.append(item)
            items.append((item, val))
            
        try:
            groups = self.model.InstrumentGroups
        except Exception:
            groups = False
        try:
            units = self.model.InstrumentUnits
        except Exception:
            units = False
        
        dlg = ValidateDialog(self, pars, vals, validators,
                             title = 'Instrument Editor', groups = groups,
                             units = units)
        if dlg.ShowModal()==wx.ID_OK:
            #print 'Pressed OK'
            vals = dlg.GetValues()
            #print vals
            for par in self.model.InstrumentParameters:
                self.instrument.__setattr__(par, vals[par])
            self.Update()
        else:
            #print 'Pressed Cancel'
            pass
        dlg.Destroy()
        
        
    def MoveUp(self,evt):
        #print dir(self.listbox)
        sl=self.sampleh.moveUp(self.listbox.GetSelection())
        if sl:
            self.Update()
            self.listbox.SetSelection(self.listbox.GetSelection()-1)

    def MoveDown(self,evt):
        #print dir(self.listbox)
        #print self.listbox.GetSelection()
        sl=self.sampleh.moveDown(self.listbox.GetSelection())
        if sl:
            self.Update()
            self.listbox.SetSelection(self.listbox.GetSelection()+1)

    def InsertStack(self,evt):
        # Create Dialog box
        items = [('Name', 'name')]
        #validators = [NoMatchValidTextObjectValidator(self.sampleh.names)]
        validators = {}
        vals = {}
        validators['Name'] = NoMatchValidTextObjectValidator(self.sampleh.names)
        pars = ['Name']
        vals['Name'] = 'name'
        #dlg = ValidateDialog(self, items, validators, title='Give Stack Name')
        dlg = ValidateDialog(self, pars, vals, validators,\
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
                        print 'Unexpected problems...'
            dlg.Destroy()
        
        

    def lbDoubleClick(self,evt):
        sel=self.sampleh.getItem(self.listbox.GetSelection())
        eval_func = self.plugin.GetModel().eval_in_model
        sl=None
        items=[]
        validators = {}
        vals = {}
        pars = []
        if isinstance(sel,self.model.Layer): 
            # The selected item is a Layer
            for item in self.model.LayerParameters.keys():
                value=sel.__getattribute__(item)
                #if item!='n' and item!='fb':
                if type(self.model.LayerParameters[item]) != type(1+1.0J):
                    #validators.append(FloatObjectValidator(eval_func))
                    validators[item] = FloatObjectValidator(eval_func)
                else:
                    #print 'n exists'
                    #validators.append(MatchTextObjectValidator(self.refindexlist))
                    #validators.append(ComplexObjectValidator(eval_func))
                    validators[item] = ComplexObjectValidator(eval_func)
                items.append((item, value))
                pars.append(item)
                vals[item] = value
            try:
                groups = self.model.LayerGroups
            except Exception:
                groups = False
            try:
                units = self.model.LayerUnits
            except Exception:
                units = False
            
            #dlg = ValidateDialog(self,items,validators,title='Layer Editor')
            dlg = ValidateDialog(self, pars, vals, validators,
                                 title = 'Layer Editor', groups = groups,
                                 units = units)
            
            if dlg.ShowModal()==wx.ID_OK:
                #print 'Pressed OK'
                vals=dlg.GetValues()
                for par in self.model.LayerParameters.keys():
                    sel.__setattr__(par, vals[par])
                sl=self.sampleh.getStringList()
            else:
                pass
                #print 'Pressed Cancel'
            dlg.Destroy()

        else: 
            # The selected item is a Stack
            for item in self.model.StackParameters.keys():
                if item!='Layers':
                    value=sel.__getattribute__(item)
                    if isinstance(value,float):
                        #validators.append(FloatObjectValidator(eval_func))
                        validators[item] = FloatObjectValidator(eval_func)
                    else:
                        #validators.append(TextObjectValidator())
                        validators[item] = TextObjectValidator()
                    items.append((item,value))
                    pars.append(item)
                    vals[item] = value
            
            try:
                groups = self.model.StackGroups
            except Exception:
                groups = False
            try:
                units = self.model.StackUnits
            except Exception:
                units = False
            
            #dlg = ValidateDialog(self,items,validators,title='Layer Editor')
            dlg = ValidateDialog(self, pars, vals, validators,
                                 title = 'Stack Editor', groups = groups,
                                 units = units)
            if dlg.ShowModal()==wx.ID_OK:
                #print 'Pressed OK'
                vals=dlg.GetValues()
                for par in pars:
                    sel.__setattr__(par, vals[par])
                sl=self.sampleh.getStringList()
            else:
                #print 'Pressed Cancel'
                pass
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
        
        # BEGIN BUTTON SECTION
        boxbuttons = wx.BoxSizer(wx.HORIZONTAL)
        #button_names = ['Insert', 'Delete', 'Move up', 'Move down',\
        #    'Edit Parameters']
        #callbacks = [self.Insert, self.Delete, self.MoveUp, self.MoveDown,\
        #    self.EditPars]
        if os.name == 'nt':
            size = (24, 24)
        else:
            size = (-1, -1)
        space = (2, -1)
        button_names = ['Insert', 'Delete', 'User Variables']
        button_images = [images.getaddBitmap(), images.getdeleteBitmap(),\
            images.getcustom_parameterBitmap()]
        callbacks = [self.Insert, self.Delete, self.EditPars]
        tooltips = ['Insert a command', 'Delete command', 'Edit user variables']
        for i in range(len(button_names)):
            #button = wx.Button(self,-1, button_names[i])
            button = wx.BitmapButton(self, -1, button_images[i],\
                    style=wx.NO_BORDER, size = size)
            boxbuttons.Add(button, 1, wx.EXPAND)
            boxbuttons.Add(space)
            button.SetToolTipString(tooltips[i])
            self.Bind(wx.EVT_BUTTON, callbacks[i], button)
        # END BUTTON SECTION
        boxver.Add((-1,2))
        boxver.Add(boxbuttons)
        boxver.Add((-1,2))
        
        self.listbox = MyHtmlListBox(self, -1, style =  wx.BORDER_SUNKEN)
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.Edit , self.listbox)
        boxver.Add(self.listbox, 1, wx.EXPAND)
        
        self.SetSizer(boxver)
        
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
    
    def SetUpdateScriptFunc(self, func):
        '''SetUpdateScriptFunc(self, func) --> None
        
        Sets the function to be called when the script needs to be updated.
        will only be called as func(event)
        '''
        self.script_update_func = func
    
    def Update(self):
        '''Update(self) --> None
        
        Update the listbox.
        '''
        list_strings = []
        for i in range(len(self.datalist)):
            list_strings.append('%s\'s commands:'%self.datalist[i])
            for item in self.expressionlist[i]:
                list_strings.append(self.command_indent + '%s</pre>'%item)
        
        self.listbox.SetItemList(list_strings)
        if self.script_update_func:
            self.script_update_func(None)
        
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
        #print self.datalist
        #print self.expressionlist
        for i in range(len(self.datalist)):
            dataindex += 1
            listindex +=1
            #print 'test'
            if listindex >= index:
                return (dataindex, itemindex)
            itemindex = -1
            for item in self.expressionlist[i]:
                itemindex += 1
                listindex += 1
                if listindex >= index:
                    return (dataindex, itemindex)
        # If all other things fail...
        return (-1, -1)
            
            
    
    def Edit(self, event):
        '''Edit(self, event) --> None
        
        Edits an entry.
        '''
        data_pos, exp_pos = self.get_expression_position()
        if exp_pos != -1 and data_pos != -1:
            list_item = self.expressionlist[data_pos][exp_pos]
            dlg = ParameterExpressionDialog(self, self.plugin.GetModel(),\
                list_item)
            if dlg.ShowModal() == wx.ID_OK:
                exp = dlg.GetExpression()
                self.expressionlist[data_pos][exp_pos] = exp
                self.Update()
        
    def Insert(self, event):
        ''' Insert(self, event) --> None
        
        Inserts a new operations
        '''
        data_pos, exp_pos = self.get_expression_position()
        #print data_pos, exp_pos
        if data_pos != -1:
            dlg = ParameterExpressionDialog(self, self.plugin.GetModel())
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
        self.Update()
        
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
    
class ParameterExpressionDialog(wx.Dialog):
    ''' A dialog for setting parameters for fitting
    '''
    def __init__(self, parent, model, expression = None):
        wx.Dialog.__init__(self, parent, -1, 'Parameter expression editor')
        self.SetAutoLayout(True)
        self.model = model
        
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
        #print model.compiled
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
            #print expression
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
                raise Valueerror('The object %s does not exist'%obj)
    
        self.expression_ctrl = wx.TextCtrl(self, -1, exp_right,\
                                size=(300, -1))
                                
        gbs.Add(self.expression_ctrl, (1,2),\
            flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL,border = 5)
        
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
        
        sizer.Add(button_sizer,0, wx.ALIGN_RIGHT, 5)
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
        #print 'Trying to evaluate evalstring'
        #print evalstring
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
        
        if self.plugin.sim_returns_sld and model._sim:
            self.plot_dicts = model.SLD
            #self.plot_dict = self.plot_dicts[0]
        else:
            if (not self.plugin.sim_returns_sld) and self.plugin.GetModel().compiled:
                plot_dict = model.sample.SimSLD(None, model.inst)
                self.plot_dicts = [plot_dict]
        
        self.plot.ax.lines = []
        self.plot.ax.clear()
        i = 0
        data = self.plugin.GetModel().get_data()
        sld_units = []
        for sim in range(len(self.plot_dicts)):
            if data[sim].show:
                for key in self.plot_dicts[sim]:
                    is_imag = key[:2] == 'Im' or key[:4] == 'imag'
                    if (is_imag and self.plugin.show_imag_sld) or not is_imag:
                        if key != 'z' and key != 'SLD unit': 
                            label = data[sim].name + '\n' + key
                            self.plot.ax.plot(self.plot_dicts[sim]['z'], self.plot_dicts[sim][key],\
                                          colors[i%len(colors)], label = label)
                            i += 1
                            if self.plot_dicts[sim].has_key('SLD unit'):
                                if not self.plot_dicts[sim]['SLD unit'] in sld_units:
                                    sld_units.append(self.plot_dicts[sim]['SLD unit'])
                    
        if i > 0:
            self.plot.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                                prop = {'size': 10}, ncol = 1)
        
            sld_unit = ', '.join(sld_units)
            self.plot.ax.yaxis.label.set_text('$\mathrm{\mathsf{SLD\,[%s]}}$'%(sld_unit))
            #if self.plot_dict.has_key('SLD unit'):
            #    self.plot.ax.yaxis.label.set_text('$\mathrm{\mathsf{SLD\,[%s]}}$'%(sld_unit))
            self.plot.ax.xaxis.label.set_text('$\mathrm{\mathsf{ z\,[\AA]}}$')
            self.plot.flush_plot()
            self.plot.AutoScale()
    
    def SavePlotData(self, filename):
        ''' Save the data that is plotted to file with filename.'''
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
            print save_array.shape
            header = 'z\t' 
            for key in self.plot_dicts[sim]:
                if key != 'z' and key != 'SLD unit':
                    save_array = np.r_[save_array, [self.plot_dicts[sim][key]]]
                    header += key + '\t'
            print save_array.shape
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
        #print 'Updating the script'
        #print dir(event)
        self.WriteModel()
        
    def OnAutoUpdateSLD(self, evt):
        #self.mb_autoupdate_sld.Check(not self.mb_autoupdate_sld.IsChecked())
        #print self.mb_autoupdate_sld.IsChecked()
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
        #print 'New model'
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
        if event.data_moved or event.deleted or event.new_data\
            or event.name_change:
            names = [data_set.name for data_set in self.GetModel().get_data()]
            self.simulation_widget.SetDataList(names)
            
            expl = self.simulation_widget.GetExpressionList()
            if event.deleted:
                pos = range(len(expl))
                [self.remove_data_segment(pos[-index-1]) for index in\
                    range(len(event.position))]
                [expl.pop(index) for index in event.position]
            if event.data_moved:
                if event.up:
                    # Moving up
                    for pos in event.position:
                        tmp = self.items.pop(pos)
                        expl.insert(pos-1, tmp)
                else:
                    #Moving down...
                    for pos in event.position:
                        tmp = self.items.pop(pos)
                        expl.insert(pos+1, tmp)
                        
            if len(names)-len(expl) == 1:
                # Data set has been added:
                expl.append([])
                self.insert_new_data_segment(len(expl)-1)
            
            self.simulation_widget.SetExpressionList(expl)
            # Check so we have not clicked on new model button
            if self.GetModel().script != '':
                self.simulation_widget.Update()
                self.WriteModel()
                if event.name_change:
                    self.sld_plot.Plot()
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
        
        Updates stuff after simulation
        '''
        # Calculate and update the sld plot
        if self.mb_autoupdate_sld.IsChecked():
            self.sld_plot.Plot()
        
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
        for i in range(len(self.GetModel().get_data())):
            script += '    # BEGIN Dataset %i DO NOT CHANGE\n'%i
            script += '    I.append(sample.SimSpecular(data[%i].x, inst))\n'%i
            script += '    if _sim: SLD.append(sample.SimSLD(None, inst))\n'
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
        self.sample_widget.SetInstrument(instrument)
        
        names = [data_set.name for data_set in self.GetModel().get_data()]
        self.simulation_widget.SetDataList(names)
        # An empty list to the expression widget...
        self.simulation_widget.SetExpressionList([[] for item in names])
        self.simulation_widget.Update()
        
        self.sample_widget.Update()
        #self.WriteModel()
    
    def WriteModel(self):
        script = self.GetModel().get_script()
        
        code = 'inst = model.' + self.sample_widget.instrument.__repr__() + '\n'
        code += ('fp.set_wavelength(inst.wavelength)\n'
                 '#Compability issues for pre-fw created gx files\n'
                 'try:\n\t fw\nexcept:\n\tpass\nelse:\n'
                '\tfw.set_wavelength(inst.wavelength)\n')
        script = self.insert_code_segment(script, 'Instrument', code)
        
        layer_code, stack_code, sample_code = self.sampleh.getCode()
        code = layer_code + '\n' + stack_code + '\n' + sample_code
        script = self.insert_code_segment(script, 'Sample', code)
        
        code = 'cp = UserVars()\n'
        code += ''.join([line + '\n' for line in\
            self.simulation_widget.GetParameterList()])
        #print self.simulation_widget.GetParameterList()
        script = self.insert_code_segment(script, 'Parameters', code)
        
        for (i,exps) in enumerate(self.simulation_widget.GetExpressionList()):
            exp = [ex + '\n' for ex in exps]
            exp.append('I.append(sample.SimSpecular(data[%i].x, inst))\n'%i)
            if self.sim_returns_sld:
                exp.append('if _sim: SLD.append(sample.SimSLD(None, inst))\n')
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
            raise LookupError('Could not fing return I in the script')
        
        script = ''.join(script_lines[:line_index-1])
        script += '    # BEGIN Dataset %i DO NOT CHANGE\n'%number
        script += '    I.append(sample.SimSpecular(data[%i].x, inst))\n'%number
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
        #print insert_code
        #print tablevel
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
        #print 'Reflectivity'
        #print self.parent.model.script
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
        #print stacks
        #print layers
        #print layer_names
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
                #print first_name
                # Find all items above the first name in the stack
                while(layer_names[0] != first_name):
                    all_names.append(layer_names.pop(0))
                    #print 'all names ',all_names[-1]
                all_names.append(layer_names.pop(0))
        all_names += layer_names
            
        #print all_names
        
        # Load the simulation parameters
        script = self.GetModel().script
        #print script
        sim_exp = []
        data_names = []
        data = self.GetModel().get_data()
        try:
            for i in range(len(data)):
                code = self.find_code_segment(script, 'Dataset %i'%i)
                sim_exp.append([])
                data_names.append(data[i].name)
                #for line in code.splitlines()[:-1]:
                #    sim_exp[-1].append(line.strip())
                for line in code.splitlines():
                    if line.find('I.append') == -1 and line.find('SLD.append') == -1:
                        sim_exp[-1].append(line.strip()) 
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
        self.sample_widget.SetInstrument(self.GetModel().script_module.inst)
        
        self.simulation_widget.SetDataList(data_names)
        self.simulation_widget.SetExpressionList(sim_exp)
        self.simulation_widget.SetParameterList(uservars_lines)
        self.sample_widget.Update()
        self.simulation_widget.Update()
        self.StatusMessage('New sample loaded to plugin!')
        
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
