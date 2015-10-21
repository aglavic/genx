'''
Library for the GUI components of the Parameter grid which is used to 
define which parameters to fit. The library parameters contains the class
definition of the parameters. 
Programmer: Matts Bjorck
Last Changed: 2014 07 17
'''

import os
import wx
import wx.grid as gridlib
import wx.lib.printout as printout
import wx.lib.agw.floatspin as fs
from wx.lib.masked import NumCtrl
import string

from numpy import *

import parameters
import images as img
import lib.controls as ctrls
import filehandling


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

    def DeleteRows(self, rows):
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
        
    def InsertRow(self, row):
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

    def MoveRowUp(self, row):
        """ Move row up one row.

        :param row: Integer row number to move up
        :return: Boolean
        """

        success = self.pars.move_row_up(row)

        msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.parent._grid_changed()
        return success

    def MoveRowDown(self, row):
        """ Move row down one row.

        :param row: Integer row number to move down
        :return: Boolean
        """

        success = self.pars.move_row_down(row)

        msg = gridlib.GridTableMessage(self, gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.parent._grid_changed()
        return success

    def SortRows(self):
        """ Sort the rows in the table

        :return: Boolean to indicate success
        """
        success = self.pars.sort_rows()

        msg = gridlib.GridTableMessage(self, gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.parent._grid_changed()
        return success


    
    def AppendRows(self, num_rows = 1):
        #print num_rows
        [self.pars.append() for i in range(num_rows)]
        
        msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED, num_rows)
        self.GetView().ProcessTableMessage(msg)
        msg = gridlib.GridTableMessage(self,\
                gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.parent._grid_changed()
        return True
    
    #def GetAttr(self, row, col, kind):
    #    '''Called by the grid to find the attributes of the cell,
    #    bkg color, text colour, font and so on.
    #    '''
    #    attr = super(ParameterDataTable, self).GetAttr(row, col, kind)

    #    if col == 1 and row < self.pars.get_len_rows():
    #        if attr is None:
    #            attr = gridlib.GridCellAttr()
    #        attr = attr.Clone()
    #        val = self.pars.get_value(row,1)
    #        max_val = self.pars.get_value(row,4)
    #        min_val = self.pars.get_value(row,3)
    #        if val > max_val or val < min_val:
    #            attr.SetBackgroundColour(wx.Colour(204, 0, 0))
    #            attr.SetTextColour(wx.Colour(255, 255, 255))


    #    return attr
        
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
        self.parent._grid_changed(permanent_change=permanent_change)

    def ChangeValueInteractively(self, row, value):
        """ Callback for a change of the value. Used to interactively set the value and notify other parts
        of GenX by posting a EVT_PARAMETER_VALUE_CHANGE event.

        """
        self.SetValue(row, 1, value)
        self.parent.PostValueChangedEvent()


#Class ParameterDataTable ends here
#------------------------------------------------------------------------------
class SliderCellEditor(gridlib.PyGridCellEditor):

    def __init__(self, value=0.0, min_value=0.0, max_value=100.0):
        gridlib.PyGridCellEditor.__init__(self)
        self.value = value
        self.startValue = value
        self.min_value = min_value
        self.max_value = max_value

    def Create(self, parent, id, evtHandler):
        self._tc = ctrls.SliderControl(parent, id, value=self.value, max_value=self.max_value,
                                           min_value=self.min_value, font=parent.GetFont())
        self.SetControl(self._tc)

        if evtHandler:
            self._tc.PushEventHandler(evtHandler)

    def GetValue(self):
        return self._tc.GetValue()


    def SetSize(self, rect):
        """
        Called to position/size the edit control within the cell rectangle.
        """
        self._tc.SetDimensions(rect.x-1, rect.y-1, rect.width+2, rect.height+2,
                               wx.SIZE_ALLOW_MINUS_ONE)

    def StartingClick(self):
        """
        Make a correct action if editor activated by mouse click
        """
        #print "Staring click"
        pass


    def BeginEdit(self, row, col, grid):
        """
        Fetch the value from the table and prepare the edit control
        to begin editing.  Set the focus to the edit control.
        """

        self.startValue = grid.GetTable().GetValue(row, col)
        self.max_value = grid.GetTable().GetValue(row, col + 3)
        self.min_value = grid.GetTable().GetValue(row, col + 2)
        if self.startValue != '':
            self.startValue = float(self.startValue)
        else:
            self.startValue = 0.0
            grid.GetTable().SetValue(row, col, self.startValue)

        if self.max_value != '':
            self.max_value = float(self.max_value)
        else:
            self.max_value = 0.0
            grid.GetTable().SetValue(row, col, self.max_value)
        if self.min_value != '':
            self.min_value = float(self.min_value)
        else:
            self.min_value = 0.0
            grid.GetTable().SetValue(row, col, self.min_value)

        self._tc.SetValue(float(self.startValue))
        self._tc.SetMaxValue(self.max_value)
        self._tc.SetMinValue(self.min_value)
        self._tc.bind_handlers()
        self._tc.SetScrollCallback(lambda val: grid.GetTable().ChangeValueInteractively(row, val))
        self._tc.SetFocus()
        #print "begin edit finished"

    def EndEdit(self, row, col, grid, oldVal):
        """
        End editing the cell.  This function must check if the current
        value of the editing control is valid and different from the
        original value (available as oldval in its string form.)  If
        it has not changed then simply return None, otherwise return
        the value in its string form.
        """
        #print "EndEdit"
        self._tc.unbind_handlers()
        val = self._tc.GetValue()
        self._tc.SetScrollCallback(None)
        if val != oldVal:
            return val
        else:
            return None


    def ApplyEdit(self, row, col, grid):
        """
        This function  saves the value of the control into the
        grid or grid table.
        """
        val = self._tc.GetValue()
        grid.GetTable().SetValue(row, col, val) # update the table

        self.startValue = float(val)
        self._tc.SetValue(self.startValue)


    def Reset(self):
        """
        Reset the value in the control back to its starting value.
        """
        self._tc.SetValue(self.startValue)

    def Clone(self):
        """
        Create a new object which is the copy of this one
        """
        return SliderCellEditor(value=self.value, min_value=self.min_value, max_value=self.max_value)

    def Destroy(self):
        """final cleanup"""
        self._tc.Destroy()
        super(SliderCellEditor, self).Destroy()

#---------------------------------------------------------------------------
class SliderCellRenderer(gridlib.PyGridCellRenderer):
    """ Renderer for the Slider Editor. Yields the same representation as the Editor.
    """
    def __init__(self, value=0, max_value=100.0, min_value=100):
        gridlib.PyGridCellRenderer.__init__(self)
        self.slider_drawer = ctrls.SliderDrawer(value, max=max_value, min=min_value)
        self.slider_drawer.ShowGuide(False)

    def Draw(self, grid, attr, dc, rect, row, col, isSelected):
        if grid.GetCellValue(row,col) != '':
            val = float(grid.GetCellValue(row,col))
            min_val = float(grid.GetCellValue(row,col+2))
            max_val = float(grid.GetCellValue(row,col+3))
            val = max(min(val, max_val), min_val)
            grid.GetTable().SetValue(row, col, (val))
            self.slider_drawer.SetValue(val)
            self.slider_drawer.SetMaxValue(max_val)
            self.slider_drawer.SetMinValue(min_val)
            self.slider_drawer.SetTextHeight(attr.GetFont().GetPixelSize().GetHeight())
            if isSelected:
                self.slider_drawer.SetBackgroundColour(grid.GetSelectionBackground())
            else:
                self.slider_drawer.SetBackgroundColour(attr.GetBackgroundColour())
            self.slider_drawer.Draw(dc, rect.width, rect.height, rect.x, rect.y)
        else:
            dc.SetBackgroundMode(wx.SOLID)
            if isSelected:
                dc.SetBrush(wx.Brush(grid.GetSelectionBackground(), wx.SOLID))
            else:
                dc.SetBrush(wx.Brush(attr.GetBackgroundColour(), wx.SOLID))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.DrawRectangle(rect.x, rect.y, rect.width, rect.height)

    def Clone(self):
        return SliderCellRenderer(self.slider_drawer.value, max=self.slider_drawer.max_value,
                                  min=self.slider_drawer.min_value)


class ValueLimitCellEditor(gridlib.PyGridCellEditor):
    """Editor for the parameter values with a spin control"""
    def __init__(self, value=0.0, min_value=0.0, max_value=100.0, ticks=20, digits=5):
        gridlib.PyGridCellEditor.__init__(self)
        self.value = value
        self.startValue = value
        self.min_value = min_value
        self.max_value = max_value
        self.ticks = float(ticks)
        self.digits = digits

    def Create(self, parent, id, evtHandler):
        #self._tc = wx.TextCtrl(parent, id, style=wx.ALIGN_RIGHT, validator=NumberValidator())
        self._tc = ctrls.SpinCtrl(parent, id, value=self.value)
        self.SetControl(self._tc)

        if evtHandler:
            self._tc.PushEventHandler(evtHandler)

    def Show(self, show, attr):
        super(ValueLimitCellEditor, self).Show(show, attr)

    def GetValue(self):
        return self._tc.GetValue()

    def SetSize(self, rect):
        """Called to position/size the edit control within the cell rectangle."""
        self._tc.SetDimensions(rect.x-1, rect.y-1, rect.width+2, rect.height+2,
                               wx.SIZE_ALLOW_MINUS_ONE)

    def BeginEdit(self, row, col, grid):
        """ Fetch the value from the table and prepare the edit control to begin editing.  Set the focus to the
        edit control.
        """
        self.startValue = grid.GetTable().GetValue(row, col)
        self.max_value = grid.GetTable().GetValue(row, col + 3)
        self.min_value = grid.GetTable().GetValue(row, col + 2)
        if self.startValue != '':
            self.startValue = float(self.startValue)
        else:
            self.startValue = 0.0
            grid.GetTable().SetValue(row, col, self.startValue)

        if self.max_value != '':
            self.max_value = float(self.max_value)
        else:
            self.max_value = 0.0
            grid.GetTable().SetValue(row, col, self.max_value)
        if self.min_value != '':
            self.min_value = float(self.min_value)
        else:
            self.min_value = 0.0
            grid.GetTable().SetValue(row, col, self.min_value)

        #self._tc.SetValue('%.7g'%(self.startValue))
        self._tc.SetValue(self.startValue)
        self._tc.SetRange(self.min_value, self.max_value)
        #self._tc.SetIncrement((self.max_value - self.min_value)/self.ticks)
        self._tc.SetValueChangeCallback(lambda val: grid.GetTable().ChangeValueInteractively(row, val))
        self._tc.SetFocus()


    def EndEdit(self, row, col, grid, oldVal):
        """
        End editing the cell.  This function must check if the current
        value of the editing control is valid and different from the
        original value (available as oldval in its string form.)  If
        it has not changed then simply return None, otherwise return
        the value in its string form.
        """
        val = float(self._tc.GetValue())
        #val = max(self.min_value, val)
        #val = min(self.max_value, val)
        #self._tc.SetValue('%.5g'%(val))
        #self._tc.SetValue(val)
        #self._tc.value_change_func(val)
        self._tc.SetValueChangeCallback(None)
        return float(val)


    def ApplyEdit(self, row, col, grid):
        """ This function  saves the value of the control into the grid or grid table."""
        val = self._tc.GetValue()
        grid.GetTable().SetValue(row, col, float(val)) # update the table

        self.startValue = float(val)
        #self._tc.SetValue('%.5g' % self.startValue)
        self._tc.SetValue(self.startValue)

    def IsAcceptedKey(self, evt):
        """
        Return True to allow the given key to start editing: the base class
        version only checks that the event has no modifiers.  F2 is special
        and will always start the editor.
        """
        key = evt.GetKeyCode()
        return chr(key) in (string.digits + '.-')

    def StartingKey(self, evt):
        """
        If the editor is enabled by pressing keys on the grid, this will be
        called to let the editor do something about that first key if desired.
        """
        key = evt.GetKeyCode()
        ch = None
        if key in [ wx.WXK_NUMPAD0, wx.WXK_NUMPAD1, wx.WXK_NUMPAD2, wx.WXK_NUMPAD3,
                    wx.WXK_NUMPAD4, wx.WXK_NUMPAD5, wx.WXK_NUMPAD6, wx.WXK_NUMPAD7,
                    wx.WXK_NUMPAD8, wx.WXK_NUMPAD9
                    ]:

            ch = chr(ord('0') + key - wx.WXK_NUMPAD0)

        elif key < 256 and key >= 0 and chr(key) in string.printable:
            ch = chr(key)

        if ch in (string.digits + '.-'):
            # For this example, replace the text.  Normally we would append it.
            #self._tc.AppendText(ch)
            self._tc.SetValue(ch)
            self._tc.SetInsertionPointEnd()
        #else:
        #    evt.Skip()



    def Reset(self):
        """Reset the value in the control back to its starting value."""
        self._tc.SetValue('%.5g' % self.startValue)

    def Clone(self):
        """
        Create a new object which is the copy of this one
        """
        return ValueLimitCellEditor(value=self.value, min_value=self.min_value, max_value=self.max_value)


class ValueLimitCellRenderer(gridlib.PyGridCellRenderer):
    """ Renderer for the Parameter Values. Colours the Cell if the value is out of bounds.
    """
    def __init__(self, value=0, max_value=100.0, min_value=100):
        gridlib.PyGridCellRenderer.__init__(self)

    def Draw(self, grid, attr, dc, rect, row, col, isSelected):
        if grid.GetCellValue(row,col) != '':
            val = float(grid.GetCellValue(row,col))
            min_val = float(grid.GetCellValue(row,col+2))
            max_val = float(grid.GetCellValue(row,col+3))
            if val > max_val or val < min_val:
                bkg_colour = wx.Colour(204, 0, 0)
                txt_colour = wx.Colour(255, 255, 255)
            else:
                if not isSelected:
                    bkg_colour = attr.GetBackgroundColour()
                else:
                    bkg_colour = grid.GetSelectionBackground()
                txt_colour = wx.Colour(0, 0, 0)

            dc.SetBackgroundMode(wx.SOLID)
            dc.SetBrush(wx.Brush(bkg_colour, wx.SOLID))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.SetClippingRect(rect)
            dc.DrawRectangle(rect.x, rect.y, rect.width, rect.height)

            text = '%.7g'%val

            dc.SetTextForeground(txt_colour)
            dc.SetTextBackground(bkg_colour)
            dc.SetFont(attr.GetFont())
            width, height = dc.GetTextExtent(text)
            dc.DrawText(text, rect.x+rect.width - width - 1, rect.y + 1)

            dc.DestroyClippingRegion()
        else:
            dc.SetBackgroundMode(wx.SOLID)
            if isSelected:
                dc.SetBrush(wx.Brush(grid.GetSelectionBackground(), wx.SOLID))
            else:
                dc.SetBrush(wx.Brush(attr.GetBackgroundColour(), wx.SOLID))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.DrawRectangle(rect.x, rect.y, rect.width, rect.height)


    def Clone(self):
        return SliderCellRenderer(self.slider_drawer.value, max=self.slider_drawer.max_value,
                                  min=self.slider_drawer.min_value)

class ValueCellEditor(gridlib.PyGridCellEditor):
    """Editor for the parameter values with a spin control"""
    def __init__(self, value=0.0, digits=5):
        gridlib.PyGridCellEditor.__init__(self)
        self.value = value
        self.startValue = value
        #self.min_value = min_value
        #self.max_value = max_value
        #self.ticks = float(ticks)
        self.digits = digits

    def Create(self, parent, id, evtHandler):
        self._tc = wx.TextCtrl(parent, id, style=wx.ALIGN_RIGHT, validator=ctrls.NumberValidator())
        #self._tc = ctrls.SpinCtrl(parent, id, value=self.value)
        self.SetControl(self._tc)

        if evtHandler:
            self._tc.PushEventHandler(evtHandler)

    def Show(self, show, attr):
        super(ValueCellEditor, self).Show(show, attr)

    def GetValue(self):
        return float(self._tc.GetValue())

    def SetSize(self, rect):
        """Called to position/size the edit control within the cell rectangle."""
        self._tc.SetDimensions(rect.x-1, rect.y-1, rect.width+2, rect.height+2,
                               wx.SIZE_ALLOW_MINUS_ONE)

    def BeginEdit(self, row, col, grid):
        """ Fetch the value from the table and prepare the edit control to begin editing.  Set the focus to the
        edit control.
        """
        self.startValue = grid.GetTable().GetValue(row, col)
        if self.startValue != '':
            self.startValue = float(self.startValue)
        else:
            self.startValue = 0.0
            grid.GetTable().SetValue(row, col, self.startValue)

        self._tc.SetValue('%.7g'%(self.startValue))
        #self._tc.SetValue(self.startValue)
        #self._tc.SetValueChangeCallback(lambda val: grid.GetTable().ChangeValueInteractively(row, val))
        self._tc.SetFocus()


    def EndEdit(self, row, col, grid, oldVal):
        """
        End editing the cell.  This function must check if the current
        value of the editing control is valid and different from the
        original value (available as oldval in its string form.)  If
        it has not changed then simply return None, otherwise return
        the value in its string form.
        """
        val = float(self._tc.GetValue())
        #val = max(self.min_value, val)
        #val = min(self.max_value, val)
        #self._tc.SetValue('%.5g'%(val))
        #self._tc.SetValue(val)
        #self._tc.value_change_func(val)
        #self._tc.SetValueChangeCallback(None)
        return float(val)


    def ApplyEdit(self, row, col, grid):
        """ This function  saves the value of the control into the grid or grid table."""
        val = self._tc.GetValue()
        grid.GetTable().SetValue(row, col, float(val)) # update the table

        self.startValue = float(val)
        self._tc.SetValue('%.7g' % self.startValue)
        #self._tc.SetValue(self.startValue)

    def IsAcceptedKey(self, evt):
        """
        Return True to allow the given key to start editing: the base class
        version only checks that the event has no modifiers.  F2 is special
        and will always start the editor.
        """
        key = evt.GetKeyCode()
        return chr(key) in (string.digits + '.-')

    def StartingKey(self, evt):
        """
        If the editor is enabled by pressing keys on the grid, this will be
        called to let the editor do something about that first key if desired.
        """
        key = evt.GetKeyCode()
        ch = None
        if key in [ wx.WXK_NUMPAD0, wx.WXK_NUMPAD1, wx.WXK_NUMPAD2, wx.WXK_NUMPAD3,
                    wx.WXK_NUMPAD4, wx.WXK_NUMPAD5, wx.WXK_NUMPAD6, wx.WXK_NUMPAD7,
                    wx.WXK_NUMPAD8, wx.WXK_NUMPAD9
                    ]:

            ch = chr(ord('0') + key - wx.WXK_NUMPAD0)

        elif key < 256 and key >= 0 and chr(key) in string.printable:
            ch = chr(key)

        if ch in (string.digits + '.-'):
            # For this example, replace the text.  Normally we would append it.
            #self._tc.AppendText(ch)
            self._tc.SetValue(ch)
            self._tc.SetInsertionPointEnd()
        #else:
        #    evt.Skip()



    def Reset(self):
        """Reset the value in the control back to its starting value."""
        self._tc.SetValue('%.5g' % self.startValue)

    def Clone(self):
        """
        Create a new object which is the copy of this one
        """
        return ValueCellEditor(value=self.value, min_value=self.min_value, max_value=self.max_value)


class ValueCellRenderer(gridlib.PyGridCellRenderer):
    """ Renderer for the Parameter Values. Colours the Cell if the value is out of bounds.
    """
    def __init__(self, value=0, max_value=100.0, min_value=100):
        gridlib.PyGridCellRenderer.__init__(self)

    def Draw(self, grid, attr, dc, rect, row, col, isSelected):
        if grid.GetCellValue(row,col) != '':
            val = float(grid.GetCellValue(row,col))
            if not isSelected:
                bkg_colour = attr.GetBackgroundColour()
            else:
                bkg_colour = grid.GetSelectionBackground()
            txt_colour = wx.Colour(0, 0, 0)

            dc.SetBackgroundMode(wx.SOLID)
            dc.SetBrush(wx.Brush(bkg_colour, wx.SOLID))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.SetClippingRect(rect)
            dc.DrawRectangle(rect.x, rect.y, rect.width, rect.height)

            text = '%.7g'%val

            dc.SetTextForeground(txt_colour)
            dc.SetTextBackground(bkg_colour)
            dc.SetFont(attr.GetFont())
            width, height = dc.GetTextExtent(text)
            dc.DrawText(text, rect.x+rect.width - width - 1, rect.y + 1)

            dc.DestroyClippingRegion()
        else:
            dc.SetBackgroundMode(wx.SOLID)
            if isSelected:
                dc.SetBrush(wx.Brush(grid.GetSelectionBackground(), wx.SOLID))
            else:
                dc.SetBrush(wx.Brush(attr.GetBackgroundColour(), wx.SOLID))
            dc.SetPen(wx.TRANSPARENT_PEN)
            dc.DrawRectangle(rect.x, rect.y, rect.width, rect.height)


    def Clone(self):
        return SliderCellRenderer(self.slider_drawer.value, max=self.slider_drawer.max_value,
                                  min=self.slider_drawer.min_value)


#------------------------------------------------------------------------------
class ParameterGrid(wx.Panel):
    '''
    The GUI component itself. This is the thing to use in a GUI.
    '''
    def __init__(self, parent, frame, config=None):
        wx.Panel.__init__(self, parent)

        self.config = config
        self.config_name = 'parameter grid'

        # The two main widgets
        self.toolbar = wx.ToolBar(self,  style=wx.TB_FLAT|wx.TB_VERTICAL)
        self.grid = gridlib.Grid(self, -1, style=wx.NO_BORDER)
        self.grid._grid_changed = self._grid_changed
        self.grid.PostValueChangedEvent = self.PostValueChangedEvent
        #self.grid.SetForegroundColour('BLUE')

        self.do_toolbar()

        self.sizer_hor=wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(self.sizer_hor)
        self.sizer_hor.Add(self.toolbar, proportion=0, flag=wx.EXPAND, border=0)
        self.sizer_hor.Add(self.grid, proportion=1, flag=wx.EXPAND, border=0)

        self.parent = frame
        self.prt = printout.PrintTable(parent)
        
        self.project_func = None
        self.scan_func = None
        
        self.table = ParameterDataTable(self.grid)
        
        self.variable_span = 0.25
        #The functions has to begin with the following letters:
        self.set_func = 'set'
        self.get_func = 'get'
        # The second parameter means that the grid is to take ownership of the
        # table and will destroy it when done.  Otherwise you would need to keep
        # a reference to it and call it's Destroy method later.
        self.grid.SetTable(self.table, True)
        self.grid.SetSelectionMode(gridlib.Grid.SelectRows)
        #self.grid.SetSelectionBackground()

        self.grid.SetRowLabelSize(50)
        self.grid.SetMargins(0, 0)
        # This is the my original column True means set as min...
        #self.AutoSizeColumns(True)
        # The new
        self.grid.AutoSizeColumn(0, False)
        self.grid.AutoSizeColumn(1, False)
        self.grid.AutoSizeColumn(2, True)
        self.grid.AutoSizeColumn(3, False)
        self.grid.AutoSizeColumn(4, False)
        self.grid.AutoSizeColumn(5, False)

        self.par_dict = {}

        self.grid.GetGridWindow().Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.grid.Bind(gridlib.EVT_GRID_CELL_LEFT_DCLICK, self.OnLeftDClick)
        self.grid.Bind(gridlib.EVT_GRID_CMD_CELL_LEFT_CLICK, self.OnLeftClick)
        self.grid.Bind(gridlib.EVT_GRID_CMD_CELL_RIGHT_CLICK, self.OnRightClick)
        self.grid.Bind(gridlib.EVT_GRID_LABEL_RIGHT_CLICK,self.OnLabelRightClick)
        self.grid.Bind(wx.EVT_SIZE, self.OnResize)
        self.grid.Bind(gridlib.EVT_GRID_SELECT_CELL, self.OnSelectCell)

        self.toolbar.Realize()
        self.show_slider = False
        self.SetValueEditorSlider(slider=False)
        attr = gridlib.GridCellAttr()
        attr.SetEditor(ValueCellEditor())
        attr.SetRenderer(ValueCellRenderer())
        self.grid.SetColAttr(3, attr)
        self.grid.SetColAttr(4, attr)

    def PrepareNewModel(self):
        """ Hack to prepare the grid for a new model.
        :return:
        """
        # This hack is needed to deselect any current cell. Have not found a better way to solve it.
        # If not called the program can cause an segmentation fault and crash.
        self.grid.SetGridCursor(0, 3)
        self.grid.SetGridCursor(0, 4)

    def ReadConfig(self):
        """ Reads the variables stored in the config file."""
        try:
            val = self.config.get_boolean(self.config_name, 'value slider')
        except filehandling.OptionError:
            print 'Could not locate option %s.%s'%(self.config_name, 'y scale')
            self.SetValueEditorSlider(False)
        else:
            self.SetValueEditorSlider(val)

    def WriteConfig(self, show_slider=None):
        """Writes the varaibles to be stored to the config"""
        if show_slider is not None:
            self.config.set(self.config_name, 'value slider', show_slider)


    def SetValueEditorSlider(self, slider=True):
        """ Set the Editor and Renderer as slider instead of text.

        :param slider: Flag determining if the Editor (and Renderer) should be a slider.
        :return:
        """
        #print "SetValueEditorSlider"
        row = self.grid.GetGridCursorRow()
        col = self.grid.GetGridCursorCol()
        #print row, col
        #print self.grid.GetSelectedCells()
        # This will disable the editor in the cell that currently is under editing.
        # If the editor is active the program will crash when changing the cell editor.
        self.grid.SetGridCursor(0, 3)
        self.grid.SetGridCursor(0, 4)
        self.grid.EnableCellEditControl(False)
        attr = gridlib.GridCellAttr()
        if slider:
            attr.SetEditor(SliderCellEditor())
            attr.SetRenderer(SliderCellRenderer())
        else:
            attr.SetEditor(ValueLimitCellEditor())
            attr.SetRenderer(ValueLimitCellRenderer())
        self.grid.SetColAttr(1, attr)
        self.grid.SetGridCursor(row, col)
        self.show_slider = slider
        self.WriteConfig(show_slider=slider)

    def GetValueEditorSlider(self):
        """Returns True if the slider editor is active"""
        return self.show_slider

    def PostValueChangedEvent(self):
        evt = value_change()
        wx.PostEvent(self.parent, evt)

    def do_toolbar(self):
        #if os.name == 'nt':
        #    size = (24, 24)
        #else:
        #    size = (-1, -1)
        #self.toolbar.SetToolBitmapSize((21,21))
        #self.toolbar.SetToolSeparation(5)
        #self.toolbar.SetBackgroundStyle(wx.BG_STYLE_COLOUR)
        #self.toolbar.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENUBAR))
        #self.toolbar.SetBackgroundColour('BLUE')

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, label='Add a new row', bitmap=img.getaddBitmap(), shortHelp='Insert new row')
        self.Bind(wx.EVT_TOOL, self.eh_add_row, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, label='Delete row', bitmap=img.getdeleteBitmap(), shortHelp='Delete row')
        self.Bind(wx.EVT_TOOL, self.eh_delete_row, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, label='Move row up', bitmap=img.getmove_upBitmap(), shortHelp='Move row up')
        self.Bind(wx.EVT_TOOL, self.eh_move_row_up, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, label='Move row down', bitmap=img.getmove_downBitmap(),
                                  shortHelp='Move row down')
        self.Bind(wx.EVT_TOOL, self.eh_move_row_down, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, label='Sort parameters', bitmap=img.sort.getBitmap(),
                                  shortHelp='Sort the rows by class, object and name')
        self.Bind(wx.EVT_TOOL, self.eh_sort, id=newid)

        newid = wx.NewId()
        self.slider_tool_id = newid
        self.slider_tool = self.toolbar.AddCheckLabelTool(newid, label='Show sliders', bitmap=img.slider.getBitmap(),
                                                          shortHelp='Show the parameter values as sliders')
        self.Bind(wx.EVT_TOOL, self.eh_slider_toggle, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, label='Project FOM evals', bitmap=img.par_proj.getBitmap(),
                                  shortHelp='Project FOM on parameter axis')
        self.Bind(wx.EVT_TOOL, self.eh_project_fom, id=newid)

        newid = wx.NewId()
        self.toolbar.AddLabelTool(newid, label='Scan parameter', bitmap=img.par_scan.getBitmap(), shortHelp='Scan FOM')
        self.Bind(wx.EVT_TOOL, self.eh_scan_fom, id=newid)


    def eh_add_row(self, event):
        """ Event handler for adding a row

        :param event:
        :return:
        """
        row = self.grid.GetGridCursorRow()
        self.table.InsertRow(row)

    def eh_delete_row(self, event):
        """Event handler for deleteing a row

        :param event:
        :return:
        """
        row = self.grid.GetGridCursorRow()
        self.table.DeleteRows([row,])

    def eh_move_row_up(self, event):
        """ Event handler for moving a row up.

        :param event:
        :return:
        """
        row = self.grid.GetGridCursorRow()
        if self.table.MoveRowUp(row):
            self.grid.SetGridCursor(row - 1, self.grid.GetGridCursorCol())

    def eh_move_row_down(self, event):
        """ Event handler for moving a row down.

        :param event:
        :return:
        """
        row = self.grid.GetGridCursorRow()
        if self.table.MoveRowDown(row):
            self.grid.SetGridCursor(row + 1, self.grid.GetGridCursorCol())

    def eh_slider_toggle(self, event):
        """ Event handler for showing the sliders instead of values

        :param event:
        :return:
        """
        new_state = not self.GetValueEditorSlider()
        self.SetValueEditorSlider(new_state)
        self.parent.mb_view_grid_slider.Check(new_state)
        self.Refresh()

    def toggle_slider_tool(self, state):
        self.toolbar.ToggleTool(self.slider_tool_id, state)

    def get_toggle_slider_tool_state(self):
        self.toolbar.GetToolState(self.slider_tool_id)

    def eh_project_fom(self, event):
        """ Event handler for toolbar project fom
        :param event:
        :return:
        """
        row = self.grid.GetGridCursorRow()
        if self.project_func:
            self.project_func(row)

    def eh_scan_fom(self, event):
        """ Event handler for scanning fom on selected parameter

        :param event:
        :return:
        """
        row = self.grid.GetGridCursorRow()
        if self.scan_func:
            self.scan_func(row)

    def eh_sort(self, event):
        """Event handler for the sorting

        :param event:
        :return:
        """
        self.table.SortRows()


    def OnSelectCell(self, evt):
         self.grid.SelectRow(evt.GetRow())
         evt.Skip()

    
    def _grid_changed(self, permanent_change = True):
        '''_grid_changed(self) --> None
        
        internal function to yield a EVT_PARAMETER_GRID_CHANGE
        '''
        #print 'Posting'
        evt = grid_change()
        evt.permanent_change = permanent_change
        wx.PostEvent(self.parent, evt)
    
    def _update_printer(self):
        '''_update_printer(self) --> None
        
        Update the printer to have the same values as in the grid.
        '''
        data = []
        for row in self.GetParameters().get_data():
            #data.append([' %.30s'%row[0],' %.5f'%row[1],\
               #' %d'%row[2],' %.5g'%row[3],' %.5g'%row[4],' '+row[5]])
            data.append([row[0],'%.5f'%row[1],\
                         '%d'%row[2],'%.5f'%row[3],'%.5f'%row[4], row[5]])
        self.prt.data = data
        self.prt.label = self.GetParameters().get_col_headers()
        self.prt.SetPaperId(wx.PAPER_A4)
        self.prt.SetLandscape()
        self.prt.page_width = 11.69
        self.prt.page_height = 8.26
        self.prt.cell_right_margin = 0
        self.prt.cell_left_margin = 0.05
        self.prt.text_font = {"Name":'Arial',"Size":14,"Colour":[0, 0, 0], "Attr":[0,0,0]}
        self.prt.label_font = {"Name":'Arial',"Size":24,"Colour":[0,0,0], "Attr":[0,0,0]}
        self.prt.set_column = [ 3, 1.8, 0.5, 1.5, 1.5, 1.5]
        self.prt.vertical_offset = 0
        self.prt.horizontal_offset = 0
        #self.prt.SetRowSpacing(0,0)
        self.prt.SetCellText(4, 2, wx.NamedColour('RED'))
        self.prt.SetHeader("Fittting parameters",align=wx.ALIGN_CENTRE, colour = wx.NamedColour('RED'), size=14)
        self.prt.SetFooter("Print Date/Time: ", type = "Date & Time", align=wx.ALIGN_CENTRE, indent = 1, colour = wx.NamedColour('RED'), size=14)
        
    def Print(self):
        '''Print(self) --> None
        
        Prints the values to the printer
        '''
        pd = wx.PrintData()
        pd.SetPrinterName('')
        pd.SetOrientation(wx.LANDSCAPE) 
        pd.SetPaperId(wx.PAPER_A4)
        pd.SetQuality(wx.PRINT_QUALITY_DRAFT)
        pd.SetColour(True)
        pd.SetNoCopies(1)
        pd.SetCollate(True)
        
        pdd = wx.PrintDialogData()
        pdd.SetPrintData(pd)
        pdd.SetMinPage(1)
        pdd.SetMaxPage(1)
        pdd.SetFromPage(1)
        pdd.SetToPage(2)
        pdd.SetPrintToFile(True)

        #printer = wx.Printer(pdd)
        self._update_printer()
        self.prt.Print()
        #prtout = printout.SetPrintout(self.prt)
        #printer.Print(self.parent, prtout, True)
        
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
                permanent_change=False)
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
        

    def OnLeftDClick(self, evt):
        """ Event handler that starts editing the cells on a double click and not
        just a second click as is default.

        :param evt: The Event passed to the event handler
        :return: Nothing
        """
        #if self.grid.CanEnableCellControl() and evt.GetCol() != 2:
        #    self.grid.EnableCellEditControl()
        ##self.grid.SelectRow(evt.GetRow())
        col, row = evt.GetCol(), evt.GetRow()
        if col == 0 and row > -1:
            if self.grid.GetGridCursorRow() == row:
                self.CurSelection = (row, col)
                self.grid.SetGridCursor(row, col)
                pos = evt.GetPosition()
                self.show_parameter_menu(pos)
            else:
                evt.Skip()
        else:
            #if self.grid.CanEnableCellControl() and evt.GetCol() != 2:
            #    self.grid.EnableCellEditControl()
            evt.Skip()


    def OnLeftClick(self,evt):
        """Callback to handle left clicks in the cells.

        :param evt: A Event from a Grid
        :return: Nothing
        """
        col, row = evt.GetCol(), evt.GetRow()
        #self.grid.SelectRow(row)
        if col == 2 and row > -1:
            self.table.SetValue(row, col, not self.table.GetValue(row, col))

        elif col == 0 and row > -1:
            if self.grid.GetGridCursorRow() == row:
                self.CurSelection = (row, col)
                self.grid.SetGridCursor(row, col)
                pos = evt.GetPosition()
                self.show_parameter_menu(pos)
            else:
                evt.Skip()
        elif col == 1 and row > -1:
            if self.show_slider:
                self.grid.SetGridCursor(row, col)
                self.grid.EnableCellEditControl()
            evt.Skip()
            #return
        else:
            evt.Skip()

    def OnLeftDown(self, event):
        """ Callback for left down - handling slider activation

        :param evt:
        :return:
        """
        #print "Left Down"
        x, y = self.grid.CalcUnscrolledPosition(event.GetX(), event.GetY())
        row, col = self.grid.XYToCell(x, y)
        if col == 1 and row > -1:
            #print "Activating editor"
            #self.grid.SetGridCursor(row, col)
            #wx.CallAfter(self.grid.EnableCellEditControl)#self.grid.EnableCellEditControl()
            #self.grid.ShowCellEditControl()
            event.Skip()
        else:
            event.Skip()

    def show_label_menu(self, row):
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
            rows = self.grid.GetSelectedRows()
            self.table.DeleteRows(rows)

        def projectfom(event, self=self, row=row):
            if self.project_func:
                self.project_func(row)

        def scanfom(event, self=self, row=row):
            if self.scan_func:
                self.scan_func(row)

        self.Bind(wx.EVT_MENU, insert, id=insertID)
        self.Bind(wx.EVT_MENU, delete, id=deleteID)
        # Item is checked for fitting
        if self.table.GetValue(row, 2):
            self.Bind(wx.EVT_MENU, projectfom, id=projectID)
        if self.table.GetValue(row, 0) != '':
            self.Bind(wx.EVT_MENU, scanfom, id=scanID)
        self.PopupMenu(menu)
        menu.Destroy()

    def OnLabelRightClick(self,evt):
        """ Callback function that opens a popupmenu for appending or
        deleting rows in the grid.
        """
        #check so a row is selected
        col, row = evt.GetCol(), evt.GetRow()
        if col == -1:
            if not self.grid.GetSelectedRows():
                self.grid.SelectRow(row)
            #making the menus
            self.show_label_menu(row)
    
    def SetFOMFunctions(self, projectfunc, scanfunc):
        '''SetFOMFunctions(self, projectfunc, scanfunc) --> None
        
        Set the functions for executing the projection and scan function.
        '''
        self.project_func = projectfunc
        self.scan_func = scanfunc

    def show_parameter_menu(self, pos):
        self.pmenu = wx.Menu()
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
                    item = obj_menu.Append(-1, obj + '.' + func)
                    self.Bind(wx.EVT_MENU, self.OnPopUpItemSelected, item)
                clmenu.AppendMenu(-1, obj, obj_menu)
            self.pmenu.AppendMenu(-1, cl, clmenu)
        # Check if there are no available classes
        if len(classes) == 0:
            # Add an item to compile the model
            item = self.pmenu.Append(-1, 'Simulate to see parameters')
            self.Bind(wx.EVT_MENU, self.OnPopUpItemSelected, item)
        # Add an item for edit the cell manually
        item = self.pmenu.Append(-1, 'Manual Edit')
        self.Bind(wx.EVT_MENU, self.OnPopUpItemSelected, item)

        self.PopupMenu(self.pmenu, pos)
        self.pmenu.Destroy()

    def OnRightClick(self, evt):
        ''' Callback function that creates a popupmenu when a row in the
        first column is clicked. The menu contains all parameters that are 
        selectable and fitable. 
        '''
        #print dir(evt)
        col=evt.GetCol()
        row=evt.GetRow()
        #print row,col
        if col==0:
            self.CurSelection = (row, col)
            self.grid.SetGridCursor(row, col)
            pos = evt.GetPosition()
            self.show_parameter_menu(pos)
            
    def OnPopUpItemSelected(self, event):
        """ Callback for the popmenu to select parameter to fit.
        When a item from the OnRightClick is selected this function takes care
        of setting:
        1. the parametername
        2. The current value of the parameter in Value
        3. The min and max values

        :param event: event from the menu
        :return: Nothing
        """
        item = self.pmenu.FindItemById(event.GetId())
        # Check if the item should be edit manually
        if item.GetText() == 'Simulate to see parameters':
            self.parent.eh_tb_simulate(event)
            return
        if item.GetText() == 'Manual Edit':
            if self.grid.CanEnableCellControl():
                self.grid.EnableCellEditControl()
            return
        # GetText seems to screw up underscores a bit replacing the with a 
        # double one - this fixes it
        text = item.GetText().replace('__', '_')
        self.grid.SetCellValue(self.CurSelection[0], self.CurSelection[1], text)
        # Try to find out if the values also has an get function so that
        # we can init the value to the default!
        lis = text.split('.' + self.set_func)
        if len(lis) == 2:
            try:
                value = self.evalf(lis[0] + '.' + self.get_func + lis[1])().real
                self.table.SetValue(self.CurSelection[0], 1, value)
                # Takes care so that also negative numbers give the
                # correct values in the min and max cells
                minval = value*(1 - self.variable_span)
                maxval = value*(1 + self.variable_span)
                self.table.SetValue(self.CurSelection[0], 3,
                                    min(minval, maxval))
                self.table.SetValue(self.CurSelection[0], 4,
                                    max(minval, maxval))
            except StandardError, S:
                print "Not possible to init the variable automatically"
                #print S

    def OnResize(self, evt):
        self.SetColWidths()
        evt.Skip()
    
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
        width=(self.grid.GetSize().GetWidth()-self.grid.GetColSize(2)-self.grid.GetRowLabelSize())/5-1
        # To avoid warnings relating to a width < 0. This can occur during startup
        if width <= 0:
            width = 1
        self.grid.SetColSize(0,width)
        self.grid.SetColSize(1,width)
        self.grid.SetColSize(3,width)
        self.grid.SetColSize(4,width)
        self.grid.SetColSize(5,width)
        
    def GetParameters(self):
        '''
        Function that returns the parameters - Is this needed anymore?
        '''
        return self.table.pars
    
#==============================================================================
# Custom events needed for updating and message parsing between the different
# modules.

# Event for when the grid has new values
(grid_change, EVT_PARAMETER_GRID_CHANGE) = wx.lib.newevent.NewEvent()
# Event for then the value of a parameter has changed. Should be used to do
# simulations interactively.
(value_change, EVT_PARAMETER_VALUE_CHANGE) = wx.lib.newevent.NewEvent()

