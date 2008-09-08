''' help.py - A module that includes various help functionality 
for genx. 

First is the ModelsHelpDialog which displays the doc strings of each module
in the models directory.
'''

import wx, os
import  wx.html as  html

class ModelsHelpDialog(wx.Frame):
    def __init__(self, parent, module):
        wx.Frame.__init__(self, parent, -1, 'Models help')
        #self.SetAutoLayout(True)
        self.module = module
        
        sizer = wx.BoxSizer(wx.VERTICAL)
        choice_sizer = wx.BoxSizer(wx.HORIZONTAL)
        choice_sizer.Add(wx.StaticText(self, -1, 'Model: '),0 , wx.CENTER)
        self.choice = wx.Choice(self, -1, choices = self.find_modules(module))
        choice_sizer.Add(self.choice, 0, flag = wx.EXPAND|wx.CENTER, border = 20)
        self.Bind(wx.EVT_CHOICE, self.on_choice, self.choice)
        sizer.Add(choice_sizer, 0, wx.EXPAND, border = 20)
        
        
        self.html_win = html.HtmlWindow(self, -1,\
                        style = wx.NO_FULL_REPAINT_ON_RESIZE)
        sizer.Add(self.html_win, 1, flag = wx.EXPAND, border = 20)
        
        # Add the Dialog buttons
        #button_sizer = wx.StdDialogButtonSizer()
        #okay_button = wx.Button(self, wx.ID_OK)
        #okay_button.SetDefault()
        #button_sizer.AddButton(okay_button)
        #button_sizer.Realize()
        
        #line = wx.StaticLine(self, -1, size=(40,-1), style=wx.LI_HORIZONTAL)
        #sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_HORIZONTAL|wx.TOP, 40)
        
        #sizer.Add(button_sizer, 0, flag = wx.ALIGN_RIGHT, border = 20)
        self.SetSizer(sizer)
        
        sizer.Fit(self)
        self.Layout()
        x,y = parent.GetSizeTuple()
        x = max(y/2, x/3)
        self.SetSize((x,y))
        
        self.choice.SetSelection(0)
        self.on_choice(None)
        
    def on_choice(self, event):
        '''on_choice(self, event) --> None
        
        Callback for a choice selction in the choice window
        '''
        sub_module = self.choice.GetStringSelection()
        doc = self.load_doc(self.module, sub_module)
        self.html_win.SetPage(doc)
        
    def find_modules(self, module):
        '''find_modules(self, directory) --> string_list
        
        Finds all modules in a directory. Only finds .py files and not files
        beginning _
        '''
        mod = __import__(module)
        return [s[:-3] for s in os.listdir(mod.__path__[0])\
                    if s[0] != '_' and s[-3:] == '.py']
                    
    def load_doc(self, module, sub_module):
        '''load_doc(self, module, sub_moduls) --> doc string
        
        Returns the doc string for module.sub_module list
        '''
        docs = ''
        try:
            mod = __import__('%s.%s'%(module, sub_module), fromlist = [None])
            docs = mod.__doc__
        except:
            docs = 'Could not load docstring for %s'%sub_module
        if type(docs) != type(''):
            docs = 'The doc string is of the wrong type in module %s'%sub_module
        return docs