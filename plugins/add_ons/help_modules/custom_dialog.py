import wx, math, os

import string

import reflectivity_images as images

def is_reflfunction(obj):
    ''' Convenience function to determine whether obj belongs to the ReflFunction class.
    Return boolean.
    '''
    return obj.__class__.__name__ == 'ReflFunction'

class TextObjectValidator(wx.PyValidator):
    """ This validator is used to ensure that the user has entered something
        into the text object editor dialog's text field.
    """
    def __init__(self):
        """ Standard constructor.
        """
        wx.PyValidator.__init__(self)


    def Clone(self):
        """ Standard cloner.

           Note that every validator must implement the Clone() method.
        """
        return TextObjectValidator()


    def Validate(self, win):
        """ Validate the contents of the given text control.
        """
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()

        if len(text) == 0:
            wx.MessageBox("A text object must contain some text!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            textCtrl.SetBackgroundColour(
                 wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True


    def TransferToWindow(self):
        """ Transfer data from validator to window.

            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True # Prevent wxDialog from complaining.


    def TransferFromWindow(self):
        """ Transfer data from window to validator.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
         """
        return True # Prevent wxDialog from complaining.

class MatchTextObjectValidator(wx.PyValidator):
    """ This validator is used to ensure that the user has entered something
        into the text object editor dialog's text field.
    """
    def __init__(self,stringlist):
        """ Standard constructor.
        """
        wx.PyValidator.__init__(self)
        self.stringlist=stringlist

    def Clone(self):
        """ Standard cloner.

            Note that every validator must implement the Clone() method.
        """
        return MatchTextObjectValidator(self.stringlist)


    def Validate(self, win):
        """ Validate the contents of the given text control.
        """
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()

        if self.stringlist.__contains__(text):
            textCtrl.SetBackgroundColour(
                wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True
        else:
            wx.MessageBox("The name is not defined!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False


    def TransferToWindow(self):
        """ Transfer data from validator to window.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True # Prevent wxDialog from complaining.


    def TransferFromWindow(self):
        """ Transfer data from window to validator.

            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True # Prevent wxDialog from complaining.

class NoMatchTextObjectValidator(wx.PyValidator):
    """ This validator is used to ensure that the user has entered something
        into the text object editor dialog's text field.
    """
    def __init__(self,stringlist):
        """ Standard constructor.
        """
        wx.PyValidator.__init__(self)
        self.stringlist=stringlist

    def Clone(self):
        """ Standard cloner.
            Note that every validator must implement the Clone() method.
        """
        return NoMatchTextObjectValidator(self.stringlist)

    def Validate(self, win):
        """ Validate the contents of the given text control.
        """
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        if len(text) == 0:
            wx.MessageBox("A text object must contain some text!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif self.stringlist.__contains__(text):
            wx.MessageBox("Duplicates are not allowed!", "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            textCtrl.SetBackgroundColour(
                wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """ Transfer data from validator to window.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True # Prevent wxDialog from complaining.


    def TransferFromWindow(self):
        """ Transfer data from window to validator.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True # Prevent wxDialog from complaining.

class NoMatchValidTextObjectValidator(wx.PyValidator):
    """ This validator is used to ensure that the user has entered something
        into the text object editor dialog's text field. It should not match
        a name in stringlist and it should be a valid varaible name. I.e 
        is should start with a letter and can only contains ordinary
        letters as in string.letters as well as numbers as in string.digits
        or _
    """
    def __init__(self,stringlist):
        """ Standard constructor.
        """
        wx.PyValidator.__init__(self)
        self.stringlist = stringlist
        self.reserved_words = ['and', 'del', 'from', 'not', 'while', 'as',\
                    'elif', 'global', 'or', 'with', 'assert', 'else', 'if',\
                    'pass', 'yield', 'break', 'except', 'import', 'print',\
                    'class', 'exec', 'in', 'raise', 'continue', 'finally', \
                    'is', 'return', 'def', 'for', 'lambda', 'try']

        self.allowed_chars = string.digits+string.letters + '_'

    def Clone(self):
        """ Standard cloner.
            Note that every validator must implement the Clone() method.
        """
        return NoMatchValidTextObjectValidator(self.stringlist)

    def Validate(self, win):
        """ Validate the contents of the given text control.
        """
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        #print text, len(text)
        #print sum([char in self.allowed_chars for char in text])
        if len(text) == 0:
            wx.MessageBox("A text object must contain some text!", "Bad Input")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif self.stringlist.__contains__(text):
            wx.MessageBox("Duplicates are not allowed!", "Bad Input")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif text in self.reserved_words:
            wx.MessageBox("Python keywords are not allowed!", "Bad Input")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        elif sum([char in self.allowed_chars for char in text]) != len(text)\
            or text[0] in string.digits:
            wx.MessageBox("Not a vaild name. Names can only contain letters"\
            ", digits and underscores(_) and not start with a digit.",\
             "Bad Input")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            textCtrl.SetBackgroundColour(
                wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """ Transfer data from validator to window.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True # Prevent wxDialog from complaining.


    def TransferFromWindow(self):
        """ Transfer data from window to validator.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True # Prevent wxDialog from complaining.

class FloatObjectValidator(wx.PyValidator):
    """ This validator is used to ensure that the user has entered something
        into the text object editor dialog's text field.
    """
    def __init__(self, eval_func = eval):
        """ Standard constructor.
        """
        wx.PyValidator.__init__(self)
        self.value=None
        self.eval_func = eval_func

    def Clone(self):
        """ Standard cloner.

            Note that every validator must implement the Clone() method.
        """
        return FloatObjectValidator(self.eval_func)


    def Validate(self, win):
        """ Validate the contents of the given text control.
        """
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        self.value=None
        try:
            val = self.eval_func(text)
            if is_reflfunction(val):
                val = val.validate()
            self.value=float(val)
        except StandardError,S:
            wx.MessageBox("Can't evaluate the expression!!\nERROR:\n%s"%S.__str__(), "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            #print 'OK'
            textCtrl.SetBackgroundColour(
                wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True

    def TransferToWindow(self):
        """ Transfer data from validator to window.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True


    def TransferFromWindow(self):
        """ Transfer data from window to validator.

            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True
         
class ComplexObjectValidator(wx.PyValidator):
    """ This validator is used to ensure that the user has entered something
       into the text object editor dialog's text field.
    """
    def __init__(self, eval_func = eval):
        """ Standard constructor.
        """
        wx.PyValidator.__init__(self)
        self.value=None
        self.eval_func = eval_func
     
    def Clone(self):
        """ Standard cloner.
            Note that every validator must implement the Clone() method.
        """
        return ComplexObjectValidator(self.eval_func)


    def Validate(self, win):
        """ Validate the contents of the given text control.
        """
        textCtrl = self.GetWindow()
        text = textCtrl.GetValue()
        self.value=None
        try:
            # Have to do it differentily to work with proxys
            # self.value=complex(self.eval_func(text))
            val = self.eval_func(text)
            if is_reflfunction(val):
                val = val.validate()
            self.value = complex(val.real, val.imag)
        except AttributeError, S:
            try:
               self.value = complex(val)
            except StandardError,S:
                wx.MessageBox("Can't evaluate the expression!!\nERROR:\n%s"%S.__str__(), "Error")
                textCtrl.SetBackgroundColour("pink")
                textCtrl.SetFocus()
                textCtrl.Refresh()
                return False
        except StandardError, S:
            wx.MessageBox("Can't evaluate the expression!!\nERROR:\n%s"%S.__str__(), "Error")
            textCtrl.SetBackgroundColour("pink")
            textCtrl.SetFocus()
            textCtrl.Refresh()
            return False
        else:
            #print 'OK'
            textCtrl.SetBackgroundColour(
                wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW))
            textCtrl.Refresh()
            return True


    def TransferToWindow(self):
        """ Transfer data from validator to window.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True


    def TransferFromWindow(self):
        """ Transfer data from window to validator.
            The default implementation returns False, indicating that an error
            occurred.  We simply return True, as we don't do any data transfer.
        """
        return True

#----------------------------------------------------------------------

class ValidateDialog(wx.Dialog):
    def __init__(self, parent, pars, vals, validators, title="Validated Dialog", 
                 units = False, groups = False, cols = 2):
        ''' A dialog that validates the input when OK are pressed. The validation is done
        through the validators given as input.
        
        Pars should contain a list of names.
        Vals should be a dictionary of names and values for the different parameters one wish to set. 
        Groups should either be false or a consist of a list of lists which
        in turn consist of the (name, value) tuples where the first item
        in the list should be a string describing the group. This will be layed out 
        with subboxes. An example: 
        groups = [['Standard', ('f', 25), ('sigma', 7)], ['Neutron', ('b', 3.0)]]
        
        Note validators, values and units should be dictonaries of values!
        '''
        wx.Dialog.__init__(self, parent, -1, title)
        self.pars = pars
        self.validators = validators
        self.cols = cols
        self.vals = vals
        self.units = units
        self.groups = groups
        self.tc = {}
        self.SetAutoLayout(True)
        
        if self.groups:
            self.main_sizer, self.tc = self.grid_layout(self, self.vals)
        else:
            self.main_sizer, self.tc = self.layout_group(self, self.pars, self.vals)
        
        self.main_layout()
        self.Layout()
          
    def main_layout(self):
        border = wx.BoxSizer(wx.VERTICAL)
        border.Add(self.main_sizer, 1, wx.GROW|wx.ALL, 5)
        
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        border.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)
        
        buttons = self.create_buttons()
        border.Add(buttons, flag = wx.ALIGN_RIGHT|wx.ALL, border = 5)
        self.SetSizer(border)
        border.Fit(self)
        self.border_sizer = border
        
    def create_buttons(self):
        buttons = wx.StdDialogButtonSizer() #wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self, wx.ID_OK, "OK")
        b.SetDefault()
        buttons.AddButton(b)
        buttons.AddButton(wx.Button(self, wx.ID_CANCEL, "Cancel"))
        buttons.Realize()
        return buttons
        
              
    def grid_layout(self, parent, vals):
        '''Do an more advanced layout with subboxes
        '''
        rows = math.ceil(len(self.pars)/(self.cols*1.0))
        sizer = wx.FlexGridSizer(rows = rows, cols = self.cols, 
                                           vgap = 10, hgap = 10)
        tc = {}
        for group in self.groups:
            if type(group[0]) != str:
                raise TypeError('First item in a group has to be a string')
            # Make the box for putting in the columns
            col_box = wx.StaticBox(parent, -1, group[0])
            col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL )
            group, group_tc = self.layout_group(parent, group[1], vals)
            col_box_sizer.Add(group, 
                              flag = wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND, 
                              border = 5)
            for item in group_tc:
                tc[item] = group_tc[item]
            sizer.Add(col_box_sizer, flag = wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND)
        return sizer, tc
                      
            
    def layout_group(self, parent, pars, vals):
        if self.units:
            layout_cols = 3
        else:
            layout_cols = 2
        sizer = wx.FlexGridSizer(len(pars) + 1, layout_cols,
                                  vgap = 10, hgap = 5)
        tc = {}
        for par in pars:
            label = wx.StaticText(parent, -1, par + ': ')
            validator = self.validators[par]#.Clone()
            val = vals[par]
            if type(validator) == type([]):
                # There should be a list of choices
                validator = validator[:]
                tc[par] = wx.Choice(parent, -1,
                                    choices = validator)
                # Since we work with strings we have to find the right
                # strings positons to initilize the choice box.
                pos = 0
                if type(val) == type(''):
                    pos = validator.index(val)
                elif type(par) == type(1):
                    pos = par
                tc[par].SetSelection(pos)
            # Otherwise it should be a validator ...
            else:
                validator = validator.Clone()
                tc[par] = wx.TextCtrl(parent, -1, str(val),
                                           validator = validator,
                                           style = wx.TE_RIGHT)
            sizer.Add(label, \
                flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border = 5)
            sizer.Add(tc[par], 
                    flag = wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, border = 5)
            # If we include units as well:
            if self.units:
                unit_label = wx.StaticText(parent, -1, ' ' + self.units[par])
                sizer.Add(unit_label, \
                          flag = wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL, 
                          border = 5)
        return sizer, tc          
  

    def GetValues(self):
        #print dir(self.tc[0])
        #print self.tc[0].GetValue()
        p = {}
        for par in self.validators.keys():
            if type(self.validators[par]) == type([]):
                # have to pad teh text to make it a string inside a string...
                #text = '\'' 
                text = self.validators[par][self.tc[par].GetSelection()]
                #text += '\''
                p[par] = text
            else:
                p[par] = self.tc[par].GetValue()
        return p
    
class ValidateNotebookDialog(ValidateDialog):
    def __init__(self, parent, pars, vals, validators, title="Validated Dialog", 
                 units = False, groups = False, cols = 2, fixed_pages = []):
        ''' A dialog that validates the input when OK are pressed. The validation is done
        through the validators given as input.
        
        Pars should contain a list of names.
        Vals should be a dictionary of dictionaries of names and values for the different parameters one wish to set. 
        The upper dictionary level should contain the object names.
        Groups should either be false or a consist of a list of lists which
        in turn consist of the (name, value) tuples where the first item
        in the list should be a string describing the group. This will be layed out 
        with subboxes. An example: 
        groups = [['Standard', ('f', 25), ('sigma', 7)], ['Neutron', ('b', 3.0)]]
        
        Note validators, values and units should be dictonaries of values!
        '''
        wx.Dialog.__init__(self, parent, -1, title)
        self.pars = pars
        self.validators = validators
        self.cols = cols
        self.vals = vals
        self.units = units
        self.groups = groups
        self.fixed_pages = fixed_pages
        self.changes = []
        self.text_controls = {}
        self.SetAutoLayout(True)
        
        self.layout_notebook()
        self.do_toolbar()
        
        self.main_layout()
        self.Layout()
        
    def main_layout(self):
        border = wx.BoxSizer(wx.VERTICAL)
        border.Add((-1,5))
        border.Add(self.toolbar)
        #border.Add((-1,2))
        
        border.Add(self.main_sizer, 1, wx.GROW|wx.ALL, 5)
        
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        border.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)
        
        buttons = self.create_buttons()
        border.Add(buttons, flag = wx.ALIGN_RIGHT|wx.ALL, border = 5)
        self.SetSizer(border)
        border.Fit(self)
        self.border_sizer = border
        
    def do_toolbar(self):
        if os.name == 'nt':
            size = (24, 24)
        else:
            size = (-1, -1)
        space = (5, -1)
        button_names = ['Insert', 'Delete', 'Rename']
        button_images = [images.getaddBitmap(), images.getdeleteBitmap(),\
            images.getchange_nameBitmap()]
        callbacks = [self.eh_insert, self.eh_delete, self.eh_rename]
        tooltips = ['Insert', 'Delete', 'Rename']
        
        boxbuttons = wx.BoxSizer(wx.HORIZONTAL)
        boxbuttons.Add((5,-1))
        for i in range(len(button_names)):
            #button = wx.Button(self,-1, button_names[i])
            button = wx.BitmapButton(self, -1, button_images[i],\
                    style=wx.NO_BORDER, size = size)
            boxbuttons.Add(button, 1, wx.EXPAND,5)
            boxbuttons.Add(space)
            button.SetToolTipString(tooltips[i])
            self.Bind(wx.EVT_BUTTON, callbacks[i], button)
        self.toolbar = boxbuttons
            
        
    def layout_notebook(self):
        '''Do the notebook layout in the dialog
        '''
        self.main_sizer = wx.Notebook(self, -1, style=wx.NB_TOP)
        for item in self.vals:
            self.AddPage(item, self.vals[item])
    
    def eh_insert(self, evt):
        '''Eventhandler that creates a new panel
        '''
        pos = self.main_sizer.GetSelection()
        current_name = self.main_sizer.GetPageText(pos)
        index = 1
        while '%s_%d'%(current_name, index) in self.vals.keys():
            index += 1
        new_name = '%s_%d'%(current_name, index)
            
        self.AddPage(new_name, self.vals[current_name], select = True)
        self.changes.append(('', new_name))
    
    def eh_delete(self, evt):
        '''Eventhandler that deletes the current page and its data
        '''
        pos = self.main_sizer.GetSelection()
        current_name = self.main_sizer.GetPageText(pos)
        self.RemovePage(current_name)
        self.changes.append((current_name, ''))
    
    def eh_rename(self, evt):
        '''Eventhandler that renames the current tab
        '''
        pos = self.main_sizer.GetSelection()
        current_name = self.main_sizer.GetPageText(pos)
        if current_name in self.fixed_pages:
            wx.MessageBox('It is forbidden to change the'\
                'name of %s.'%current_name)
        else:
            unallowed_names = []
            for name in self.vals:
                if name != current_name:
                    unallowed_names.append(name)
            dlg = ValidateDialog(self,['Name'], {'Name':current_name},
                                 {'Name':NoMatchValidTextObjectValidator(unallowed_names)},
                                 title='Give New Name')
            
            if dlg.ShowModal()==wx.ID_OK:
                    vals=dlg.GetValues()
                    self.vals[vals['Name']] = self.vals.pop(current_name)
                    self.text_controls[vals['Name']] = self.text_controls.pop(current_name)
                    self.main_sizer.SetPageText(pos, vals['Name'])
                    self.changes.append((current_name, vals['Name']))
            dlg.Destroy()
            
    def AddPage(self, name, vals, select = False):
        '''Adds a page to the notebook 
        
        the flag select activates a page change to the current page
        '''
        panel = wx.Panel(self.main_sizer, -1)
        self.main_sizer.AddPage(panel, name, select = select)
        if self.groups:
            sizer, page_text_controls = self.grid_layout(panel, vals)
        else:
            sizer, page_text_controls = self.layout_group(panel, self.pars, vals)
        self.text_controls[name] = page_text_controls
        panel.SetSizerAndFit(sizer)
        panel.Layout()
        self.Fit()
        # Make sure that the data is added if we add a new page after init has
        # been executed
        if not name in self.vals:
            self.vals[name] = vals.copy()
            
    def RemovePage(self, name):
        '''Remove a page from the notebook
        '''
        sucess = False
        index = -1
        if name in self.fixed_pages:
            return sucess
        for i in range(self.main_sizer.GetPageCount()):
            if self.main_sizer.GetPageText(i) == name:
                sucess = True
                index = i
        if sucess:
            self.main_sizer.DeletePage(index)
            self.vals.pop(name)
        return sucess
    
    def GetValues(self):
        '''Extracts values from the text controls and returns is as a 
        two level nested dictionary.
        '''
        p = {}
        for page in self.vals:
            p[page] = {}
            for par in self.validators.keys():
                if type(self.validators[par]) == type([]):
                    # have to pad teh text to make it a string inside a string...
                    #text = '\'' 
                    text = self.validators[par][self.text_controls[page][par].GetSelection()]
                    #text += '\''
                    p[page][par] = text
                else:
                    p[page][par] = self.text_controls[page][par].GetValue()
        return p
    
    def GetChanges(self):
        '''GetChanges(self) --> changes
        Returns the changes as a list consisting of tuples (old_name, new_name).
        if the page has been deleted new_name is an empty string.
        ''' 
        return self.changes

class ZoomFrame(wx.MiniFrame):
    def __init__(self, parent):
        wx.MiniFrame.__init__(self, parent, -1, "X-Y Scales")
        
        #self.SetAutoLayout(True)

        VSPACE = 10

        self.panel=wx.Panel(self,-1,style = wx.TAB_TRAVERSAL
                     | wx.CLIP_CHILDREN
                     | wx.FULL_REPAINT_ON_RESIZE
                     )

        gbs=wx.GridBagSizer(3, 3)
        label = wx.StaticText(self.panel, -1, 'Min')
        gbs.Add(label,(0,1),flag=wx.ALIGN_CENTER,border=2)
        label = wx.StaticText(self.panel, -1, 'Max')
        gbs.Add(label,(0,2),flag=wx.ALIGN_CENTER,border=2)
        label = wx.StaticText(self.panel, -1, ' X'+': ')
        gbs.Add(label,(1,0),flag=wx.ALIGN_RIGHT,border=2)
        self.xmin=wx.TextCtrl(self.panel, -1, '0', validator = FloatObjectValidator())
        gbs.Add(self.xmin,(1,1),flag=wx.ALIGN_CENTER|wx.EXPAND,border=2)
        self.xmax=wx.TextCtrl(self.panel, -1, '0', validator = FloatObjectValidator())
        gbs.Add(self.xmax,(1,2),flag=wx.ALIGN_CENTER|wx.EXPAND,border=2)
        label = wx.StaticText(self.panel, -1, ' Y'+': ')
        gbs.Add(label,(2,0),flag=wx.ALIGN_RIGHT,border=2)
        self.ymin=wx.TextCtrl(self.panel, -1, '0', validator = FloatObjectValidator())
        gbs.Add(self.ymin,(2,1),flag=wx.ALIGN_CENTER|wx.EXPAND,border=2)
        self.ymax=wx.TextCtrl(self.panel, -1, '0', validator = FloatObjectValidator())
        gbs.Add(self.ymax,(2,2),flag=wx.ALIGN_CENTER|wx.EXPAND,border=2)


        buttons = wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self.panel, wx.ID_OK, "Apply")
        #b.SetDefault()
        #buttons.AddButton(wx.Button(self, wx.ID_CANCEL, "Cancel"))
        #buttons.Realize()
        buttons.Add((0,0),2,wx.EXPAND)
        buttons.Add(b,1,flag=wx.ALIGN_RIGHT)
        border = wx.BoxSizer(wx.VERTICAL)
        
        border.Add(gbs, 0, wx.GROW|wx.ALL, 2)
        border.Add(buttons)
        
        self.panel.SetSizerAndFit(border)
        self.SetClientSize(self.panel.GetSize())
        #self.Layout()

    
if __name__=='__main__':
    
    class MyApp(wx.App):
        def OnInit(self):
            #wx.InitAllImageHandlers()
            frame = ZoomFrame(None)
            #frame.Show(True)
            #self.SetTopWindow(frame)
            pars = ['test1', 'test2']
            vals = {'inst':{'test1':3, 'test2':5.0J},'x-ray':{'test1':3, 'test2':5.0+9J}}
            vals_old = {'test1':3, 'test2':7.0J}
            validators = {'test1':FloatObjectValidator(), 'test2':ComplexObjectValidator()}
            units = {'test1':'barn', 'test2':'m/s'}
            dlg = ValidateNotebookDialog(frame, pars, vals, validators,
                             title = 'Instrument Editor', units = units,
                             fixed_pages = ['inst'])
            dlg.AddPage('test', vals_old)
            dlg.RemovePage('inst')
            #dlg = ValidateDialog(frame, pars, vals_old, validators,
            #                 title = 'Instrument Editor', units = units)
            if dlg.ShowModal() == wx.ID_OK:
                #print 'Pressed OK'
                vals = dlg.GetValues()
                print vals
            return True


    app = MyApp(0)
    #app.MainLoop()