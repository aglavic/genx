'''
Library for the GUI components of the Parameter grid which is used to 
define which parameters to fit. The library parameters contains the class
definition of the parameters. 
Programmer: Matts Bjorck
Last Changed: 2008 09 03
'''

import  wx
import  wx.grid as gridlib
import wx.lib.printout as printout
from numpy import *
import parameters

#=============================================================================
#class ParameterDataTable

class ParameterDataTable(gridlib.PyGridTableBase):
    '''
    Class for the datatable which is used by the grid. 
    Mostly a compatability layer between the Parameters class
    and the grid.
    '''
    def __init__(self, parent):
        gridlib.PyGridTableBase.__init__(self)
        self.parent = parent
        self.pars = parameters.Parameters()

        self.data_types = [gridlib.GRID_VALUE_STRING,
                          gridlib.GRID_VALUE_FLOAT,
                          gridlib.GRID_VALUE_BOOL,
                          gridlib.GRID_VALUE_FLOAT,
                          gridlib.GRID_VALUE_FLOAT,
                          gridlib.GRID_VALUE_STRING,
                          ]

    # required methods for the wxPyGridTableBase interface
    
    def GetNumberRows(self):
        return self.pars.get_len_rows()

    def GetNumberCols(self):
        return self.pars.get_len_cols()
        
    def GetRowLabelValue(self, row):
        
        if row < self.pars.get_len_rows() and self.pars.get_value(row, 2):
            number=sum([self.pars.get_value(i, 2) for i in range(row)])
            return "%d" % int(number)
        else:
            return '-'

    def IsEmptyCell(self, row, col):
        try:
            return not self.pars.get_value(row, col)
        except IndexError:
            return True

    # Get/Set values in the table.  The Python version of these
    # methods can handle any data-type, (as long as the Editor and
    # Renderer understands the type too,) not just strings as in the
    # C++ version.
    def GetValue(self, row, col):
        try:
            return self.pars.get_value(row, col)
        except IndexError:
            return ''

    def SetValue(self, row, col, value):
        try:
            self.pars.set_value(row, col, value)
        except IndexError, e:
            # add a new row
            self.pars.append()
            #print 'Value:', value
            
            #self.SetValue(row, col, value)
            # tell the grid we've added a row
            msg = gridlib.GridTableMessage(self,            # The table
                    gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED, # what we did to it
                    1                                       # how many
                    )

            self.GetView().ProcessTableMessage(msg)
            self.pars.set_value(row, col, value)
        # For updating the column labels according to the number of fitted parameters
        if col == 2 or col == 3 or col == 4:
             self.GetView().ForceRefresh()
        self.parent._grid_changed()

    def DeleteRows(self,rows):
        delete_count = self.pars.delete_rows(rows)
        
        msg = gridlib.GridTableMessage(self,\
            gridlib.GRIDTABLE_NOTIFY_ROWS_DELETED, self.GetNumberRows(),\
             delete_count)
        self.GetView().ProcessTableMessage(msg)
        msg = gridlib.GridTableMessage(self,\
                    gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.parent._grid_changed()
        #print self.data
        
    def InsertRow(self,row):
        self.pars.insert_row(row)
        
        msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1)
        self.GetView().ProcessTableMessage(msg)
        msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.parent._grid_changed()
        return True
    
    def AppendRows(self, num_rows = 1):
        #print num_rows
        [self.pars.append() for i in range(rows)]
        
        msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED, rows)
        self.GetView().ProcessTableMessage(msg)
        msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.parent._grid_changed()
        return True
    
    def GetAttr(self, row, col, kind):
        '''Called by the grid to find the attributes of the cell,
        bkg color, text colour, font and so on.
        '''
        attr = gridlib.GridCellAttr()
        if col == 1 and row < self.pars.get_len_rows():
            val = self.pars.get_value(row,1)
            max_val = self.pars.get_value(row,4)
            min_val = self.pars.get_value(row,3)
            if  val > max_val or val < min_val:
                attr.SetBackgroundColour(wx.Colour(204, 0, 0))
                attr.SetTextColour(wx.Colour(255, 255, 255))
                
        return attr
        
    def GetColLabelValue(self, col):
        '''Called when the grid needs to display labels
        '''
        return self.pars.get_col_headers()[col]

    def GetTypeName(self, row, col):
        '''Called to determine the kind of editor/renderer to use by
        default, doesn't necessarily have to be the same type used
        natively by the editor/renderer if they know how to convert.
        '''
        return self.data_types[col]

    def CanGetValueAs(self, row, col, type_name):
        '''Called to determine how the data can be fetched and stored by the
        editor and renderer.  This allows you to enforce some type-safety
        in the grid.
        '''
        col_type = self.data_types[col].split(':')[0]
        if type_name == col_type:
            return True
        else:
            return False

    def CanSetValueAs(self, row, col, type_name):
        return self.CanGetValueAs(row, col, type_name)

    def SetParameters(self, pars, clear = True, permanent_change = True):
        '''
        SetParameters(self, pars) --> None
        
        Set the parameters in the table to pars. 
        pars has to an instance of Parameters.
        '''
        if clear:
            # Start by deleting all rows:
            msg=gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_NOTIFY_ROWS_DELETED,\
                self.parent.GetNumberRows(), self.parent.GetNumberRows())
            self.pars = parameters.Parameters()
            self.GetView().ProcessTableMessage(msg)
            msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)            
            self.GetView().ProcessTableMessage(msg)
            
            self.GetView().ForceRefresh()
            
            self.pars = pars
            msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED, self.pars.get_len_rows()+1)
            self.GetView().ProcessTableMessage(msg)
            
            msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)            
            self.GetView().ProcessTableMessage(msg)
            
            self.GetView().ForceRefresh()
            #print 'In parametergrid ', self.pars.data
        else:
            self.pars = pars
            msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)            
            self.GetView().ProcessTableMessage(msg)
        if permanent_change:
            self.parent._grid_changed()
        


#Class ParameterDataTable ends here
###############################################################################

class ParameterGrid(gridlib.Grid):
    '''
    The GUI component itself. This is the thing to use in a GUI.
    '''
    def __init__(self, parent, frame):
        gridlib.Grid.__init__(self, parent, -1)
        self.parent = frame
        self.prt = printout.PrintTable(parent)
        
        
        self.prt.left_margin = 0.5
        self.prt.right_margin = 0.5
        self.prt.text_font_size = 8
        self.prt.cell_left_margin = 0
        
        self.project_func = None
        self.scan_func = None
        
        self.table = ParameterDataTable(self)
        
        self.variable_span = 0.25
        #The functions has to begin with the following letters:
        self.set_func = 'set'
        self.get_func = 'get'
        # The second parameter means that the grid is to take ownership of the
        # table and will destroy it when done.  Otherwise you would need to keep
        # a reference to it and call it's Destroy method later.
        self.SetTable(self.table, True)

        self.SetRowLabelSize(50)
        self.SetMargins(0,0)
        # This is the my original column True means set as min...
        #self.AutoSizeColumns(True)
        # The new
        self.AutoSizeColumn(0, False)
        self.AutoSizeColumn(1, False)
        self.AutoSizeColumn(2, True)
        self.AutoSizeColumn(3, False)
        self.AutoSizeColumn(4, False)
        self.AutoSizeColumn(5, False)

        #Test cases for the parameter choosing function of the grid
        #self.objlist=['t','e']
        #self.funclist=[['e','w'],['x','y']]
        self.par_dict = {}

        self.Bind(gridlib.EVT_GRID_CELL_LEFT_DCLICK, self.OnLeftDClick)
        self.Bind(gridlib.EVT_GRID_CMD_CELL_LEFT_CLICK, self.OnLeftClick)
        self.Bind(gridlib.EVT_GRID_CMD_CELL_RIGHT_CLICK, self.OnRightClick)
        self.Bind(gridlib.EVT_GRID_LABEL_RIGHT_CLICK,self.OnLabelRightClick)
        self.Bind(wx.EVT_SIZE,self.OnResize)
    
    def _grid_changed(self):
        '''_grid_changed(self) --> None
        
        internal function to yield a EVT_PARAMETER_GRID_CHANGE
        '''
        #print 'Posting'
        evt = grid_change()
        wx.PostEvent(self.parent, evt)
    
    def _update_printer(self):
        '''_update_printer(self) --> None
        
        Update the printer to have the same values as in the grid.
        '''
        data = []
        for row in self.GetParameters().get_data():
            data.append([' '+row[0],' %.5g'%row[1],\
                ' %d'%row[2],' %.5g'%row[3],' %.5g'%row[4],' '+row[5]])
        self.prt.data=data
        self.prt.label = self.GetParameters().get_col_headers()
        
    def Print(self):
        '''Print(self) --> None
        
        Prints the values to the printer
        '''
        self._update_printer()
        self.prt.Print()
        
    def PrintPreview(self):
        '''PrintPreview(self) --> None
        
        Prints a preview of the grid
        '''
        self.prt.Preview()
        
        
    def OnNewModel(self, evt):
        '''
        OnNewModel(self, evt) --> None
        
        callback for when a new model has been loaded or created 
        - reload all the parameters
        '''
        # Set the parameters in the table 
        # this also updates the grid. 
        self.table.SetParameters(evt.GetModel().get_parameters(),\
                permanent_change = False)
        # Let the event proceed to other fucntions that have signed up.
        evt.Skip()
    
    def SetParameters(self, pars):
        self.table.SetParameters(pars)
    
    def OnSolverUpdateEvent(self, evt):
        '''OnSolverUpdateEvent(self, evt) --> None
        
        Callback to update the values in the grid from the optimizer
        Assumes that evt holds the following members:
        values: An array of the appropriate length (same as the number of 
                checked parameters to fit)
        new_best: A boolean indicating if there are a new best.
        '''
        if evt.new_best:
            #print evt.fitting
            self.table.pars.set_value_pars(evt.values)
            self.table.SetParameters(self.table.pars, clear = False,\
                permanent_change = evt.permanent_change)
        
        evt.Skip()
        
        

    # Start editing the cells on double click not just a second click...
    def OnLeftDClick(self, evt):
        if self.CanEnableCellControl() and evt.GetCol()!=2:
            self.EnableCellEditControl()

    # This is to make the checkboxes to react on a single click intstead
    # of a double click as it is by standard.
    def OnLeftClick(self,evt):      
        col,row=evt.GetCol(),evt.GetRow()
        if col==2 and row>-1:
            self.table.SetValue(row,col,not self.table.GetValue(row,col))
        else:
            evt.Skip()
    
    
    def OnLabelRightClick(self,evt):
        '''
        Callback fucntion that opens a popupmenu for appending or
        deleting rows in the grid.
        '''
        #check so a row is selected
        col,row=evt.GetCol(),evt.GetRow()
        if col==-1:
            if not self.GetSelectedRows():
                self.SelectRow(row)
            #making the menus
            insertID = wx.NewId()
            deleteID = wx.NewId()
            projectID = wx.NewId()
            scanID = wx.NewId()
            menu = wx.Menu()
            menu.Append(insertID, "Insert Row")
            menu.Append(deleteID, "Delete Row(s)")
            # Item is checked for fitting
            if self.table.GetValue(row, 2):
                menu.Append(projectID, "Project FOM")
            if self.table.GetValue(row, 0) != '':
                menu.Append(scanID, "Scan FOM")
            
            def insert(event, self=self, row=row):
                self.table.InsertRow(row)

            def delete(event, self=self, row=row):
                rows = self.GetSelectedRows()
                self.table.DeleteRows(rows)
                
            def projectfom(event, self = self, row = row):
                if self.project_func:
                    self.project_func(row)
                
            def scanfom(event, self = self, row = row):
                if self.scan_func:
                    self.scan_func(row)
            
            
            self.Bind(wx.EVT_MENU, insert, id = insertID)
            self.Bind(wx.EVT_MENU, delete, id = deleteID)
            # Item is checked for fitting
            if self.table.GetValue(row, 2):
                self.Bind(wx.EVT_MENU, projectfom, id = projectID)
            if self.table.GetValue(row, 0) != '':
                self.Bind(wx.EVT_MENU, scanfom, id = scanID)
            self.PopupMenu(menu)
            menu.Destroy()   
    
    def SetFOMFunctions(self, projectfunc, scanfunc):
        '''SetFOMFunctions(self, projectfunc, scanfunc) --> None
        
        Set the functions for executing the projection and scan function.
        '''
        self.project_func = projectfunc
        self.scan_func = scanfunc
    
    def OnRightClick(self,evt):
        '''
        Callback function that creates a popupmenu when a row in the 
        first column is clicked. The menu contains all parameters that are 
        selectable and fitable. 
        '''
        #print dir(evt)
        col=evt.GetCol()
        row=evt.GetRow()
        #print row,col
        if col==0:
            self.CurSelection=(row,col)
            self.SetGridCursor(row,col)
            pos=evt.GetPosition()
            self.pmenu=wx.Menu()
            par_dict = self.par_dict
            classes = par_dict.keys()
            classes.sort(lambda x, y: cmp(x.lower(), y.lower()))
            for cl in classes:
                # Create a submenu for each class
                clmenu = wx.Menu()
                obj_dict = par_dict[cl]
                objs = obj_dict.keys()
                objs.sort(lambda x, y: cmp(x.lower(), y.lower()))
                # Create a submenu for each object
                for obj in objs:
                    obj_menu = wx.Menu()
                    funcs = obj_dict[obj]
                    funcs.sort(lambda x, y: cmp(x.lower(), y.lower()))
                    # Create an item for each method
                    for func in funcs:
                        item = obj_menu.Append( -1, obj + '.' + func)
                        self.Bind(wx.EVT_MENU, self.OnPopUpItemSelected, item)
                    clmenu.AppendMenu(-1, obj, obj_menu)
                self.pmenu.AppendMenu(-1, cl, clmenu)
            self.PopupMenu(self.pmenu, evt.GetPosition())
            self.pmenu.Destroy()
            
    
    def OnPopUpItemSelected(self,event):
        '''
        When a item from the OnRightClick is selcted this fucntion takes care
        of setting:
        1. the parametername
        2. The current value of the parameter in Value
        3. The min and max values
        '''
        item = self.pmenu.FindItemById(event.GetId())
        # GetText seems to screw up underscores a bit replacing the with a 
        # double one - this fixes it
        text = item.GetText().replace('__','_')
        #print text, self.objlist, self.funclist
        self.SetCellValue(self.CurSelection[0],self.CurSelection[1],text)
        # Try to find out if the values also has an get function so that
        # we can init the value to the default!
        lis=text.split('.'+self.set_func)
        if len(lis)==2:
            try:
                value = self.evalf(lis[0]+'.'+self.get_func+lis[1])()
                #print value
                self.table.SetValue(self.CurSelection[0],1,value)
                self.table.SetValue(self.CurSelection[0],3,value*(1-
                self.variable_span))
                self.table.SetValue(self.CurSelection[0],4,value*(1
                +self.variable_span))
            except StandardError,S:
                print "Not possible to init the variable automatically"
                #print S
                                
            
    def OnResize(self, evt):
        self.SetColWidths()
    
    def SetParameterSelections(self, par_dict): #objlist,funclist):
        '''SetParameterSelections(self,objlist,funclist) --> None
        
        Function to set which parameter function that are chooseable 
        in the popup menu. objlist and funclist are lists of strings which
        has to be of the same size
        '''
        self.par_dict = par_dict
            
    def SetEvalFunc(self,func):
        '''
        Sets the fucntion that evaluates the expression that is exexuted in the
        model. The fucntion should take a string as input.
        '''
        self.evalf=func
        
    def SetColWidths(self):
        '''
        Function automatically set the cells width in the Grid to 
        reasonable values.
        '''
        width=(self.GetSize().GetWidth()-self.GetColSize(2)-self.GetRowLabelSize())/5-1
        self.SetColSize(0,width)
        self.SetColSize(1,width)
        self.SetColSize(3,width)
        self.SetColSize(4,width)
        self.SetColSize(5,width)
        
    def GetParameters(self):
        '''
        Function that returns the parameters - Is this needed anymore?
        '''
        return self.table.pars
    
#==============================================================================
# Custom events needed for updating and message parsing between the different
# modules.

(grid_change, EVT_PARAMETER_GRID_CHANGE) = wx.lib.newevent.NewEvent()
