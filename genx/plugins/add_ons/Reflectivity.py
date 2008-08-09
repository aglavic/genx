# Reflectivity.py written by Matts Bjorck
# A GUI for defineition of Reflectivity models
# LAst changed 20080523
# Ported from old GenX to cerate a GUI interface
# for sample definitions

import plugins.add_on_framework as framework

import  wx

import sys
import os

# Make Modules a search path for python..
#sys.path.insert(1,os.getcwd()+'/Models')

import models.interdiff as Model

from help_modules.custom_dialog import *

class SampleHandler:
    def __init__(self,sample,names):
        self.sample=sample
        self.names=names
        
    def getStringList(self, html_encoding = False):
        '''
        Function to generate a lsit of strings that gives
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
            slist.append('Stack: Reptetitions= %s'%str(stack.Repetitions))
            poslist.append((i,None))
            i+=1
        slist.append(self.sample.Ambient.__repr__())
        for item in range(len(slist)):
            if slist[item][0]=='L' and item!=0 and item!=len(slist)-1:
                if html_encoding:
                    slist[item]='<pre>   <b>'+self.names[-item-1]+'</b>='+slist[item] + '</pre>'
                else:
                    slist[item] = self.names[-item-1] + ' = ' + slist[item]
            else:
                if html_encoding:
                    slist[item]='<pre><b>' + self.names[-item-1]+'</b>='+slist[item] + '</pre>'
                else:
                    slist[item] = self.names[-item-1] + '=' + slist[item]
        poslist.append((None,None))
        slist.reverse()
        poslist.reverse()
        self.poslist=poslist
        return slist

    def getCode(self):
        '''
        Generate the python code for the current sample structure.
        '''
        slist=self.getStringList()
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
                stack_code=stack_code+stack_strings[0]+'(Layers=['
                i+=1
                item=slist[i]
                while(item.find('Stack')<0 and i<maxi):
                    itemp=item.split('=')[0]
                    itemp=itemp.lstrip()
                    stack_code=stack_code+itemp+','
                    i+=1
                    item=slist[i]
                i-=1
                if stack_code[-1] != '[':
                    stack_code=stack_code[:-1]+'],'+stack_strings[1]+')\n'
                else:
                    stack_code=stack_code[:]+'],'+stack_strings[1]+')\n'
            i+=1
            item=slist[i]
        # Create the code for the sample
        sample_code='sample=Sample(Stacks=['
        stack_strings=stack_code.split('\n')
        if stack_strings != ['']:
            for item in stack_strings:
                itemp=item.split('=')[0]
                sample_code = sample_code + itemp + ','
            sample_code = sample_code[:-2] + '],Ambient=Amb,Substrate=Sub)\n'
        else:
            sample_code += '],Ambient=Amb,Substrate=Sub)\n'
            
        print layer_code,stack_code,sample_code
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
            print self.poslist
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
        print spos
        # If it not the first item i.e. can't insert anything before the 
        # ambient layer
        if pos !=0 :
            if type=='Stack':
                stack=Model.Stack(Layers=[])
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
                layer=Model.Layer()
                print spos[0]
                if last:
                    self.names.insert(pos,name)
                else:
                    if spos[1] >= 0:
                        self.names.insert(pos+1,name)
                    else:
                        self.names.insert(pos+len(self.sample.Stacks[spos[0]].Layers)+1,name)
                self.sample.Stacks[spos[0]].Layers.insert(spos[1],layer)
                added=True
                
        else:
            if type=='Stack':
                stack=Model.Stack(Layers=[])
                self.sample.Stacks.append(stack)
                added=True
                self.names.insert(pos+1,name)
            if type=='Layer' and len(self.poslist)>2:
                layer=Model.Layer()
                print spos[0]
                self.sample.Stacks[spos[0]].Layers.append(layer)
                added=True
                self.names.insert(pos+2,name)
        if added:
            
            return self.getStringList()
        else:
            return None

        

    def canInsertLayer():
        return self.poslist>2

    def checkName(name):
        return self.names.__contains__(name)
    
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
                print temps
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
                    print temps
                    print pos
                    for index in range(len(temp.Layers)+1):
                        self.names.insert(pos+len(self.sample.Stacks[self.poslist[pos][0]-1].Layers)+1,temps[-index-1])
                    self.sample.Stacks.insert(self.poslist[pos][0]-1,temp)
                    return self.getStringList()
                else:
                    return None
                    
            else: #i.e. it is a layer we move
                if pos < len(self.poslist)-2:
                    print 'Moving a layer'
                    temp=self.sample.Stacks[self.poslist[pos][0]].Layers.pop(self.poslist[pos][1])
                    temps=self.names.pop(pos)
                    if self.poslist[pos+1][1]==None: # Next item a Stack i.e. jump down
                        print self.sample.Stacks[self.poslist[pos+1][0]]
                        self.sample.Stacks[self.poslist[pos+1][0]].Layers.insert(len(self.sample.Stacks[self.poslist[pos+1][0]].Layers),temp)
                        self.names.insert(pos+1,temps)
                    else: #Moving inside a stack
                        self.sample.Stacks[self.poslist[pos][0]].Layers.insert(self.poslist[pos][1]-1,temp)#-2
                        self.names.insert(pos+1,temps)
                    return self.getStringList()
        else:
            return None

class MyHtmlListBox(wx.HtmlListBox):
    def SetItemList(self, list):
        self.html_items = list
        self.SetItemCount(len(list))
        self.RefreshAll()
        
    def OnGetItem(self, n):
        return self.html_items[n]
   

class SamplePanel(wx.Panel):
    def __init__(self,parent,sampleh,refindexlist = []):
        wx.Panel.__init__(self,parent)
        self.sampleh=sampleh
        self.refindexlist=refindexlist
        self.instrument = Model.Instrument()
        
        boxver = wx.BoxSizer(wx.VERTICAL)
        boxhor = wx.BoxSizer(wx.HORIZONTAL)
        
        self.listbox = MyHtmlListBox(self, -1, style =  wx.BORDER_SUNKEN)
        self.listbox.SetItemList(self.sampleh.getStringList())
        self.Bind(wx.EVT_LISTBOX_DCLICK, self.lbDoubleClick , self.listbox)
        boxhor.Add(self.listbox, 1, wx.EXPAND)
        boxbuttons=wx.BoxSizer(wx.VERTICAL)
        MUpButton=wx.Button(self,-1, "MoveUp")
        boxbuttons.Add(MUpButton,1,wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.MoveUp, MUpButton)
        MDownButton=wx.Button(self,-1, "MoveDown")
        boxbuttons.Add(MDownButton,1,wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.MoveDown, MDownButton)
        InsertLayButton=wx.Button(self,-1, "Insert Layer")
        boxbuttons.Add(InsertLayButton,1,wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.InsertLay, InsertLayButton)
        InsertStackButton=wx.Button(self,-1, "Insert Stack")
        boxbuttons.Add(InsertStackButton,1,wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.InsertStack, InsertStackButton)
        DeleteButton=wx.Button(self,-1, "Delete")
        boxbuttons.Add(DeleteButton,1,wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.DeleteSample, DeleteButton)
        InstrumentButton = wx.Button(self,-1, "Instrument")
        boxbuttons.Add(InstrumentButton,1,wx.EXPAND)
        self.Bind(wx.EVT_BUTTON, self.EditInstrument, InstrumentButton)
        boxhor.Add(boxbuttons)
        boxver.Add(boxhor,1,wx.EXPAND)
        boxhorpar=wx.BoxSizer(wx.HORIZONTAL)

        self.tc=[]
        for item in Model.SampleParameters.keys():
            if item != 'Stacks' and item != 'Substrate' and item != 'Ambient':
                boxhorpar.Add(wx.StaticText(self,-1,item+': '),0)
                self.tc.append(wx.TextCtrl(self, -1,\
                 str(self.sampleh.sample.__getattribute__(item)),\
                 validator = FloatObjectValidator()))
                boxhorpar.Add(self.tc[-1],0)
        boxver.Add(boxhorpar,0)
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
        self.sampleh.sample = sample
        self.sampleh.names = names
        self.Update()
        
    def EditInstrument(self, evt):
        validators = []
        items = []
        for item in Model.InstrumentParameters:
            validators.append(FloatObjectValidator())
            val = self.instrument.__getattribute__(item)
            items.append((item, val))
            
        dlg = ValidateDialog(self, items, validators,\
            title = 'Instrument Editor')
        if dlg.ShowModal()==wx.ID_OK:
            print 'Pressed OK'
            vals=dlg.GetValues()
            for index in range(len(vals)):
                self.instrument.__setattr__(items[index][0],vals[index])
            else:
                print 'Pressed Cancel'
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
        items = [('Name', 'new_name')]
        validators = [NoMatchTextObjectValidator(self.sampleh.names)]
        dlg = ValidateDialog(self,items,validators,title='Give Stack Name')
        
        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
                print 'Pressed OK'
                vals=dlg.GetValues()
    
        sl=self.sampleh.insertItem(self.listbox.GetSelection(),'Stack', vals[0])
        if sl:
            self.Update()

    def InsertLay(self,evt):
        # Create Dialog box
        items = [('Name', 'new_name')]
        validators = [NoMatchTextObjectValidator(self.sampleh.names)]
        dlg = ValidateDialog(self,items,validators,title='Give Layer Name')
        
        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
                print 'Pressed OK'
                vals=dlg.GetValues()
        
        #Create the Layer
        sl=self.sampleh.insertItem(self.listbox.GetSelection(),'Layer', vals[0])
        if sl:
            self.Update()
        dlg.Destroy()
        
    def DeleteSample(self,evt):
        slold=self.sampleh.getStringList()
        sl=self.sampleh.deleteItem(self.listbox.GetSelection())
        if sl:
            self.Update()

    def lbDoubleClick(self,evt):
        sel=self.sampleh.getItem(self.listbox.GetSelection())
        sl=None
        if isinstance(sel,Model.Layer): # Check if the selceted item is a Layer
            items=[]
            validators=[]
            for item in Model.LayerParameters.keys():
                value=sel.__getattribute__(item)
                #if item!='n' and item!='fb':
                if type(Model.LayerParameters[item]) != type(1+1.0J):
                    validators.append(FloatObjectValidator())
                else:
                    print 'n exists'
                    #validators.append(MatchTextObjectValidator(self.refindexlist))
                    validators.append(ComplexObjectValidator())
                items.append((item,value))
            
            dlg = ValidateDialog(self,items,validators,title='Layer Editor')
            if dlg.ShowModal()==wx.ID_OK:
                print 'Pressed OK'
                vals=dlg.GetValues()
                for index in range(len(vals)):
                    sel.__setattr__(items[index][0],vals[index])
                sl=self.sampleh.getStringList()
            else:
                print 'Pressed Cancel'
            dlg.Destroy()

        else: # The selected item is a Stack
            items=[]
            validators=[]
            for item in Model.StackParameters.keys():
                if item!='Layers':
                    value=sel.__getattribute__(item)
                    if isinstance(value,float):
                        validators.append(FloatObjectValidator())
                    else:
                        validators.append(TextObjectValidator())
                    items.append((item,value))
            
            dlg = ValidateDialog(self,items,validators,title='Stack Editor')
            if dlg.ShowModal()==wx.ID_OK:
                print 'Pressed OK'
                vals=dlg.GetValues()
                for index in range(len(vals)):
                    sel.__setattr__(items[index][0],vals[index])
                sl=self.sampleh.getStringList()
            else:
                print 'Pressed Cancel'
            dlg.Destroy()
        if sl:
            self.Update()
            #for index in range(len(sl)):
            #    self.listbox.SetString(index,sl[index])
        
        
class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.model = self.GetModel()
        sample_panel = self.NewInputFolder('Sample')
        sample_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sample_panel.SetSizer(sample_sizer)
        
        self.defs = ['Sample', 'Instrument']
        
        sub = Model.Layer(sigmar=3.0,n=1-7.577e-6+1.756e-7j)
        amb = Model.Layer(n = 1.0)
        sample = Model.Sample(Stacks=[], Ambient = amb,\
            Substrate = sub, eta_z = 500.0, eta_x = 100.0)
        print sample
        
        inst = Model.Instrument(Wavelength = 1.54, Coordinates = 1)
        s = ['Amb', 'Sub']
        self.sampleh=SampleHandler(sample, s)
        self.sample_widget=SamplePanel(sample_panel, self.sampleh)
        sample_sizer.Add(self.sample_widget, 1, wx.EXPAND)
        
        self.sample_widget.SetUpdateCallback(self.UpdateScript)
        
        self.CreateNewModel()
        
    def UpdateScript(self, event):
        self.WriteModel()
        
    def CreateNewModel(self, modelname = 'models.interdiff'):
        '''Init the script in the model to yield the 
        correct script for initilization
        '''
        script = 'from %s import *\n\n'%modelname 
        
        for item in self.defs:
            script += '# BEGIN %s DO NOT CHANGE\n'%item
            script += '# END %s\n\n'%item
            
        script += 'def Sim(data):\n'
        script += '\tI = []\n'
        script += '\tfor data_set in data:\n'
        script += '\t\tI.append(sample.SimSpecular(data_set.x, inst))\n'
        script += '\treturn I\n'
        
        self.SetModelScript(script)
        
        names = ['Amb','Sub']
        Amb = Model.Layer()
        Sub = Model.Layer()
        sample = Model.Sample(Ambient = Amb, Substrate = Sub)
        self.sample_widget.SetSample(sample, names)
        self.WriteModel()
    
    def WriteModel(self):
        layer_code, stack_code, sample_code = self.sampleh.getCode()
        script_lines = self.GetModel().get_script().splitlines(True)
        script = ''
        # Locate the Sample definition
        line_index = 0
        for line in script_lines[line_index:]:
            script += line
            line_index += 1
            if line.find('# BEGIN Sample') != -1:
                break
        script += layer_code + '\n' + stack_code + '\n' + sample_code
        
        for line in script_lines[line_index:]:
            if line.find('# END Sample') != -1:
                break
            line_index += 1
        
        for line in script_lines[line_index:]:
            script += line
            line_index += 1
            if line.find('# BEGIN Instrument') != -1:
                break
        
        script += 'inst = ' + self.sample_widget.instrument.__repr__() + '\n'
        
        for line in script_lines[line_index:]:
            if line.find('# END Instrument') != -1:
                #line_index -= 1
                break
            line_index += 1
            
        for line in script_lines[line_index:]:
            script += line
            line_index += 1
        
        self.SetModelScript(script)
        
        
if __name__ == '__main__':

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
