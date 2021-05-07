'''<h1>UserFuncs</h1>
A plugin so that users can include their own user function into
the model script and execute them inside the GUI of GenX.<p>

The following design criteria exists:
The function has to be defined inside the script.
The function has to take zero input arguments.
<p>
Thus the following example can serve as an template:
<pre>def myuserfunc():
    # Do something
    print 'It works!'
</pre>
This should be added somewhere in the script. This provides a great way
to, for example, dump some internal data into a file or for checking the status
of some variables.
'''
from .. import add_on_framework as framework
import types, wx, io, traceback

class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.menu=self.NewMenu('User funcs')
        self.parent=parent
        self.StatusMessage('Sucessfully loaded UserFuncs...')

    def OnSimulate(self, event):
        '''OnSimulate(self, event) --> None
        
        Updates stuff after simulation
        '''
        model=self.GetModel()
        # locate all functions in the model.script_module
        funcs=[eval('model.script_module.%s'%name, globals(), {'model': model}) \
               for name in dir(model.script_module) \
               if type(eval('model.script_module.%s'%name, globals(), {'model': model})
                       )==types.FunctionType]
        # Find the functions that are defined in the script and takes zero
        # input parameters
        user_funcs=[f.__name__ for f in funcs if \
                    f.__module__=='genx_script_module' \
                    and f.__code__.co_argcount==0]
        # Remove all the previous functions
        self.clear_menu()
        # Lets add our user functions to our custo menu
        for name in user_funcs:
            menuitem=wx.MenuItem(self.menu, wx.NewId(), name,
                                 'Evaluate %s'%name, wx.ITEM_NORMAL)
            self.menu.AppendItem(menuitem)
            self.parent.Bind(wx.EVT_MENU, self.eh_menu_choice, menuitem)

    def clear_menu(self):
        '''clear_menu(self) --> None
        
        Clears the menu from all items present in it
        '''
        [self.menu.RemoveItem(item) for item in self.menu.GetMenuItems()]

    def eh_menu_choice(self, event):
        '''eh_menu_choice(self, event)
        
        event handler for the choice of an menuitem in the User Functions 
        menu. Executes the function as defined in the script. 
        With error catching.
        '''
        menu=event.GetEventObject()
        menuitem=menu.FindItemById(event.Id)
        fname=menuitem.GetLabel()
        model=self.GetModel()
        # Now try to evaluate the function
        self.StatusMessage('Trying to evaluate %s'%fname)
        try:
            exec('%s()'%fname, model.script_module.__dict__)
        except Exception as e:
            # abit of traceback
            outp=io.StringIO()
            traceback.print_exc(200, outp)
            tb_string=outp.getvalue()
            outp.close()
            self.ShowWarningDialog('Error in evaluating the' \
                                   ' function: %s.\n The error gave the following traceback:' \
                                   '\n%s'%(fname, tb_string))
            self.StatusMessage('error in function %s'%fname)
        else:
            self.StatusMessage('%s sucessfully evaluated'%fname)
