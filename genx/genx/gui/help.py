'''
A module that includes various help functionality for genx.

First is the ModelsHelpDialog which displays the doc strings of each module
in the models directory.
'''

import wx, os
import wx.html as html

try:
    from docutils.core import publish_doctree, publish_from_doctree
    from docutils.parsers.rst import roles
except ImportError:
    def rst_html(text):
        return "For proper display install docutils.<br>\n"+text.replace('\n', '<br>\n')
else:
    def _role_fn(name, rawtext, text, lineno, inliner, options=None, content=None):
        if options is None:
            options={}
        if content is None:
            content=[]
        return [], []

    roles.register_canonical_role('mod', _role_fn)

    def rst_html(text):
        return publish_from_doctree(publish_doctree(text), writer_name='html').decode()

class ExampleHandler:
    '''A class to handle the examples bundled with GenX
    '''

    def __init__(self, parent, example_menu, path):
        ''' Inits the Handler,
        
        parent is the main window
        path is the path to the example folder
        example_menu is the menu item that should contain the examples
        '''
        self.parent=parent
        self.path=path
        self.menu=example_menu

        self.update_menu()

    def update_menu(self):
        ''' Updates the list of examples'''
        # Remove all items present in the submenu
        items=self.menu.GetMenuItems()
        [self.menu.DeleteItem(item) for item in items]

        examples=self.get_examples()
        examples.sort()

        # Add new menu items
        for ex in examples:
            menu=self.menu.Append(-1, ex)
            self.parent.Bind(wx.EVT_MENU, self.LoadExample, menu)

    def get_examples(self):
        examples=[s[:-3] for s in os.listdir(self.path) if '.gx'==s[-3:]]
        return examples

    def LoadExample(self, event):
        menuitem=self.menu.FindItemById(event.GetId())
        example=menuitem.GetLabel()
        path=self.path+example+'.gx'
        # eh.open_model(self.parent, path)

class PluginHelpDialog(wx.Frame):
    def __init__(self, parent, module, title='Models help'):
        wx.Frame.__init__(self, parent, -1, title)
        # self.SetAutoLayout(True)
        self.module=module
        self.sub_modules=True

        sizer=wx.BoxSizer(wx.VERTICAL)
        choice_sizer=wx.BoxSizer(wx.HORIZONTAL)
        choice_sizer.Add(wx.StaticText(self, -1, 'Module: '), 0, wx.CENTER)
        mod_list=self.find_modules(module)

        self.choice=wx.Choice(self, -1, choices=mod_list)
        choice_sizer.Add(self.choice, 0, flag=wx.EXPAND | wx.CENTER,
                         border=20)
        self.Bind(wx.EVT_CHOICE, self.on_choice, self.choice)
        sizer.Add(choice_sizer, 0, wx.EXPAND, border=20)

        self.html_win=html.HtmlWindow(self, -1,
                                      style=wx.NO_FULL_REPAINT_ON_RESIZE)
        sizer.Add(self.html_win, 1, flag=wx.EXPAND, border=20)

        self.SetSizer(sizer)

        sizer.Fit(self)
        self.Layout()
        size=parent.GetSize()
        x=max(size.y/2, size.x/3)
        self.SetSize((x, size.y))

        self.choice.SetSelection(0)
        self.on_choice(None)

    def on_choice(self, event):
        '''on_choice(self, event) --> None
        
        Callback for a choice selction in the choice window
        '''
        sub_module=self.choice.GetStringSelection()
        if self.sub_modules:
            doc=self.load_doc(self.module, sub_module)
        else:
            doc=self.load_doc(self.module)
        self.html_win.SetPage(doc)

    def find_modules(self, module):
        '''find_modules(self, directory) --> string_list
        
        Finds all modules in a directory. Only finds .py files and not files
        beginning _
        '''
        # Load the package, note the non-empty fromlist that 
        # makes subpackages being loaded
        mod=__import__('genx.'+module, globals(), locals(), [''])
        try:
            modules=[s[:-3] for s in os.listdir(mod.__path__[0]) \
                     if s[0]!='_' and s[-3:]=='.py']
        except AttributeError:
            modules=[module]
            self.sub_modules=False
        return modules

    def load_doc(self, module, sub_module=None):
        '''load_doc(self, module, sub_moduls) --> doc string
        
        Returns the doc string for module.sub_module list
        '''
        docs=''
        try:
            if sub_module is not None:
                mod=__import__('genx.%s.%s'%(module, sub_module),
                               globals(), locals(), [''])
            else:
                mod=__import__('genx.%s'%module,
                               globals(), locals(), [''])
        except Exception as e:
            docs='Could not load docstring for %s.'%sub_module
            docs+='\n The following exception occured: %s'%str(e)
        else:
            if mod.__doc__ is None:
                docs="No documentation available for module."
            elif '<h1>' in mod.__doc__:
                docs=mod.__doc__
            else:
                docs=rst_html(mod.__doc__)
        if type(docs)!=type(''):
            docs='The doc string is of the wrong type in module %s'%sub_module
        return docs
