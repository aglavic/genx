import wx, math

import string

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
           self.value=float(self.eval_func(text))
        
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
           self.value = complex(val.real, val.imag)
        except AttributeError,S:
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
        ''' Pars should contain a list of names 
        for the different parameters one wish to set. 
        If use_subboxes is True pars should consist of a list of lists which
        in turn consist of the (name, value) tuples where the first item
        in the list should be a string describing the group. This will be layed out 
        with subboxes. Note validators and values should be dictonaries of values!
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
            self.grid_layout()
        else:
            #self.simple_layout()
            self.main_sizer = self.layout_group(self.pars)
        
        buttons = wx.StdDialogButtonSizer() #wx.BoxSizer(wx.HORIZONTAL)
        b = wx.Button(self, wx.ID_OK, "OK")
        b.SetDefault()
        buttons.AddButton(b)
        buttons.AddButton(wx.Button(self, wx.ID_CANCEL, "Cancel"))
        buttons.Realize()

        border = wx.BoxSizer(wx.VERTICAL)
        border.Add(self.main_sizer, 1, wx.GROW|wx.ALL, 5)
        
        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        border.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.TOP, 5)
        
        border.Add(buttons, flag = wx.ALIGN_RIGHT|wx.ALL, border = 5)
        self.SetSizer(border)
        border.Fit(self)
        self.Layout()
        
    def grid_layout(self):
        '''Do an more advanced layout with subboxes
        '''
        rows = math.ceil(len(self.pars)/(self.cols*1.0))
        self.main_sizer = wx.FlexGridSizer(rows = rows, cols = self.cols, 
                                           vgap = 10, hgap = 10)
        for group in self.groups:
            if type(group[0]) != str:
                raise TypeError('First item in a group has to be a string')
            # Make the box for putting in the columns
            col_box = wx.StaticBox(self, -1, group[0])
            col_box_sizer = wx.StaticBoxSizer(col_box, wx.VERTICAL )
            col_box_sizer.Add(self.layout_group(group[1]), 
                              flag = wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND, 
                              border = 5)
            self.main_sizer.Add(col_box_sizer, flag = wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND)
            
            
            
            
    def layout_group(self, pars):
        if self.units:
            layout_cols = 3
        else:
            layout_cols = 2
        sizer = wx.FlexGridSizer(len(pars) + 1, layout_cols,
                                  vgap = 10, hgap = 5)
        for par in pars:
            label = wx.StaticText(self, -1, par + ': ')
            validator = self.validators[par]
            val = self.vals[par]
            if type(validator) == type([]):
                # There should be a list of choices
                self.tc[par] = wx.Choice(self, -1,
                                    choices = validator)
                # Since we work with strings we have to find the right
                # strings positons to initilize the choice box.
                pos = 0
                if type(val) == type(''):
                    pos = validator.index(val)
                elif type(par) == type(1):
                    pos = par
                self.tc[par].SetSelection(pos)
            # Otherwise it should be a validator ...
            else:
                self.tc[par] = wx.TextCtrl(self, -1, str(val),
                                           validator = validator,
                                           style = wx.TE_RIGHT)
            sizer.Add(label, \
                flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border = 5)
            sizer.Add(self.tc[par], 
                    flag = wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, border = 5)
            # If we include units as well:
            if self.units:
                unit_label = wx.StaticText(self, -1, ' ' + self.units[par])
                sizer.Add(unit_label, \
                          flag = wx.ALIGN_LEFT|wx.ALIGN_CENTER_VERTICAL, 
                          border = 5)
        return sizer
            
            
            

    def simple_layout(self):
        '''Implements the ordinary "simple" layout of the boxes
        '''
        pars = self.pars
        validators = self.validators
        gbs=wx.GridBagSizer(len(pars)+1, 2)
        VSPACE = 10
        
        self.tc=[]
        for index in range(len(pars)):
            label = wx.StaticText(self, -1, pars[index][0]+': ')
            gbs.Add(label,(index,0), \
                flag = wx.ALIGN_RIGHT|wx.ALIGN_CENTER_VERTICAL, border = 5)
            # If the current validator is a string, we should give the
            # user only choices...
            if type(validators[index]) == type([]):
                self.tc.append(wx.Choice(self, -1,\
                                    choices = validators[index]))
                # Since we want to work with strings we have to find the right
                # strings positons
                #print validators[index], pars[index]
                pos = 0
                if type(pars[index][1]) == type(''):
                    for i in range(len(validators[index])):
                        if validators[index][i] == pars[index][1]:
                            pos = i
                            break
                elif type(pars[index][1]) == type(1):
                    pos = pars[index][1]
                self.tc[-1].SetSelection(pos)
            # Otherwise it should be a validator ...
            else:
                self.tc.append(wx.TextCtrl(self, -1, str(pars[index][1]),\
                    validator = validators[index]))
            gbs.Add(self.tc[index], (index, 1),\
                    flag = wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, border = 5)
            
        self.main_sizer = gbs


        

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
            frame.Show(True)
            self.SetTopWindow(frame)
            return True


    app = MyApp(0)
    app.MainLoop()
