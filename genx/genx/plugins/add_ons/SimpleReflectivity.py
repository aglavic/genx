''' <h1>Simple Reflectivity plugin </h1>
Reflectivity is a plugin for beginners just showing one single tab for
the sample and hiding all complex GenX functionality<p>

The plugin consists of the following components:
<h2>Sample tab</h2>
This tab has the definiton for the layers in a table.

<h2>SLD tab</h2>
This shows the real and imaginary part of the scattering length as a function
of depth for the sample. The substrate is to the left and the ambient material
is to the right. This is updated when the simulation button is pressed.
'''
import io, traceback

import wx.grid as gridlib
from wx.adv import Wizard, WizardPageSimple
from orsopy.slddb import api

from .. import add_on_framework as framework
from genx.model import Model
from genx.exceptions import GenxError
from genx.gui.custom_events import EVT_UPDATE_PARAMETERS, EVT_PARAMETER_GRID_CHANGE, update_model_event, \
    EVT_UPDATE_MODEL
from .help_modules.custom_dialog import *
from .help_modules.reflectivity_utils import find_code_segment
from .help_modules.reflectivity_sample_plot import SamplePlotPanel
from .help_modules import reflectivity_images as images
from genx.gui.images import getopenBitmap, getplottingBitmap
from .help_modules.materials_db import mdb, Formula, MASS_DENSITY_CONVERSION
from genx.core.custom_logging import iprint
from genx.gui.parametergrid import ValueCellRenderer
from genx.gui.custom_events import EVT_UPDATE_SCRIPT, skips_event
from ...gui.custom_ids import MenuId


_set_func_prefix = 'set'

try:
    # set locale to prevent some issues with data format
    import locale


    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    # initialize and potentially update local version of ORSO SLD db
    api.check()
except:
    api = None


def get_mat_api(frm):
    # Use ORSO SLD db to query for a material by formula
    if not api:
        return None
    try:
        res = api.localquery(dict(formula=frm.estr()))
        if len(res):
            mat = api.localmaterial(res[0]['ID'])
            return mat.dens, res[0]['ID'], res[0]['validated']
    except Exception:
        debug("Error in SLD DB query", exc_info=True)
        return None


class SampleGrid(gridlib.Grid):
    info_text: wx.StaticText

    def __init__(self, parent, *args, **kw):
        gridlib.Grid.__init__(self, parent, *args, **kw)
        self.parent = parent

        self.cb = None
        self.Bind(gridlib.EVT_GRID_CELL_LEFT_CLICK, self.onCellSelected)
        self.Bind(gridlib.EVT_GRID_EDITOR_CREATED, self.onEditorCreated)
        self.Bind(gridlib.EVT_GRID_EDITOR_SHOWN, self.onEditorShown)
        self.Bind(gridlib.EVT_GRID_EDITOR_HIDDEN, self.onEditorHidden)
        self._activated_ctrl = False

    def onCellSelected(self, evt):
        if evt.Col in [1]:
            self._activated_ctrl = True
            wx.CallAfter(self.EnableCellEditControl)
        if evt.Col in [3, 5, 7, 9]:
            if not self.parent.sample_table.GetAttr(evt.Row, evt.Col, None).IsReadOnly():
                self.parent.sample_table.SetValue(evt.Row, evt.Col,
                                                  not self.parent.sample_table.GetValue(evt.Row, evt.Col))
                self.ForceRefresh()
        else:
            evt.Skip()

    def onEditorCreated(self, evt):
        # Show tooltip on formula entry to give feedback on input
        inp = evt.Control
        inp.Bind(wx.EVT_TEXT, self.onFormula)
        if evt.Col in [3, 5, 7, 9] and self._activated_ctrl:
            self._activated_ctrl = False
        if evt.Col==2 and self.GetTable().GetValue(evt.Row, 1)=='Formula':
            self.info_text.Show()
            self.info_text.SetLabel('Enter Chemical Formula:')
            self.parent.Layout()
        evt.Skip()

    def onEditorHidden(self, evt):
        if self.info_text.IsShown():
            self.info_text.Hide()
            self.parent.Layout()
        evt.Skip()

    def onEditorShown(self, evt):
        self.dens = None
        if evt.Col==2 and self.GetTable().GetValue(evt.Row, 1)=='Formula':
            # Show tooltip on formula entry to give feedback on input
            self.info_text.Show()
            self.info_text.SetLabel('Enter Chemical Formula:')
            self.parent.Layout()
        evt.Skip()

    def onFormula(self, evt):
        self.dens = None
        if not self.info_text.IsShown():
            evt.Skip()
            return
        txt = evt.GetString()
        if txt.strip()=='':
            self.info_text.SetLabel('')
            return
        try:
            frm = Formula.from_str(txt)
        except Exception as e:
            self.info_text.SetLabel('Error in Formula:\n'+str(e))
        else:
            txt = 'Analyzed Formula:\n'+frm.describe()
            if frm in mdb:
                self.dens = mdb.dens_mass(frm)
                txt += '\n\nFound in Materials:\n%g g/cm³'%self.dens
            else:
                res = get_mat_api(frm)
                if res:
                    txt += '\n\nFound in ORSO DB:\n%g g/cm³'%res[0]
                    if res[2]:
                        txt += f'\nORSO validated\nID: {res[1]}'
                    else:
                        txt += f'\n!!NOT validated!!\nID: {res[1]}\n'
            self.info_text.SetLabel(txt)


TOP_LAYER = 0
ML_LAYER = 1
BOT_LAYER = 2


class SampleTable(gridlib.GridTableBase):
    _columns = [
        ('Layer', gridlib.GRID_VALUE_STRING),
        ('Formula Params:\n'
         '---------------'
         '\nMixure Params:', gridlib.GRID_VALUE_CHOICE+':Formula,Mixure'),
        ('Chem. Formula\n'
         '-------------'
         '\nSLD-1 [10⁻⁶Å⁻²]', gridlib.GRID_VALUE_STRING),
        ('', gridlib.GRID_VALUE_BOOL, False),
        ('Density [g/cm³]\n'
         '---------------'
         '\nSLD-2 [10⁻⁶Å⁻²]', gridlib.GRID_VALUE_STRING),
        ('', gridlib.GRID_VALUE_BOOL, False),
        ('Moment [µB/FU]\n'
         '--------------'
         '\nFraction [% SLD-1]', gridlib.GRID_VALUE_STRING),
        ('', gridlib.GRID_VALUE_BOOL, True),
        ('d [Å]', gridlib.GRID_VALUE_STRING),
        ('', gridlib.GRID_VALUE_BOOL, False),
        ('σ [Å]', gridlib.GRID_VALUE_STRING),
        ]

    _last_layer_data = []

    defaults = {
        'Formula': ['Layer', 'Formula', 'SLD',
                    False, '2.0', False, '0.0',
                    True, '10.0', False, '5.0', ML_LAYER],
        'Mixure': ['MixLayer', 'Mixure', '6.0',
                   False, '2.0', False, '100',
                   True, '10.0', False, '5.0', ML_LAYER],
        }

    repetitions = 1

    def __init__(self, parent, grid):
        gridlib.GridTableBase.__init__(self)
        self.parent = parent
        self.grid = grid
        self.layers = []

        self.ResetModel(first=True)

        self.grid.SetTable(self, True)

        dpi_scale_factor = wx.GetApp().dpi_scale_factor

        self.grid.SetRowLabelSize(30*dpi_scale_factor)
        self.grid.SetColLabelSize(50*dpi_scale_factor)
        for i, colinfo in enumerate(self._columns):
            # self.parent.SetColSize(i, 50)
            self.grid.AutoSizeColumn(i, True)
        wx.CallAfter(self.updateModel)

    def ResetModel(self, first=False):
        old_len = len(self.layers)

        self.ambient = [None, 'Formula', 'SLD',
                        False, '0.0', False, '0.0',
                        False, '0', False, '0']
        self.substrate = [None, 'Formula', Formula([['Si', 1.0]]),
                          False, '2.32998', False, '0.0',
                          False, '0', False, '5.0']
        self.layers = [['Surface_Layer', 'Formula', Formula([['Fe', 2.0], ['O', 2.0]]),
                        False, '5.25568', False, '0.0',
                        False, '20.0', False, '5.0', TOP_LAYER],
                       ['Layer_1', 'Formula', Formula([['Fe', 1.0]]),
                        False, '7.87422', False, '0.0',
                        False, '100.0', False, '5.0', ML_LAYER],
                       ['Interface_Layer', 'Formula', Formula([['Si', 1.0], ['O', 2.0]]),
                        False, '4.87479', False, '0.0',
                        False, '20.0', False, '5.0', BOT_LAYER]
                       ]

        if not first:
            diff = old_len-len(self.layers)
            for i in range(abs(diff)):
                if diff<0:
                    msg = gridlib.GridTableMessage(self,
                                                   gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1, 1)
                    self.GetView().ProcessTableMessage(msg)
                elif diff>0:
                    msg = gridlib.GridTableMessage(self,
                                                   gridlib.GRIDTABLE_NOTIFY_ROWS_DELETED, 1, 1)
                    self.GetView().ProcessTableMessage(msg)
            msg = gridlib.GridTableMessage(self,
                                           gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
            self.GetView().ProcessTableMessage(msg)
            self.GetView().ForceRefresh()
            self.updateModel()

    def GetNumberRows(self):
        return len(self.layers)+3

    def GetNumberCols(self):
        return len(self._columns)

    def GetRowLabelValue(self, row):
        if row==self.repeatedInfo():
            return ''
        row = self.realRow(row)
        if row in [0, self.GetNumberRows()-2]:
            return '-'
        else:
            return '% 2i'%row

    def IsEmptyCell(self, row, col):
        try:
            return not self.GetValue(row, col)
        except IndexError:
            return True

    def GetValue(self, row, col):
        if row==self.repeatedInfo():
            if col==0:
                return 'Repeated layer structure (white background)'
            if col==8:
                return 'Repetitions:'
            if col==10:
                return str(self.repetitions)
            return None
        row = self.realRow(row)
        if col==0:
            if row==0:
                return 'Ambient'
            elif row==(self.GetNumberRows()-2):
                return 'Substrate'
            else:
                return self.layers[row-1][col].replace('_', ' ')
        if row==0:
            if col in [7, 8, 9, 10]:
                return None
            return self.ambient[col]
        elif row==self.GetNumberRows()-2:
            if col in [7, 8]:
                return None
            return self.substrate[col]

        return self.layers[row-1][col]

    def get_valid_name(self, name):
        # generate a valid identifier string from name
        identifyier = ''
        for char in name.replace(' ', '_'):
            if (identifyier+char).isidentifier():
                identifyier += char
        if identifyier in self.invalid_identifiers:
            identifyier = '_'+identifyier

        existing = [li[0] for li in self.layers]
        if not identifyier in existing:
            return identifyier
        if identifyier.split('_')[-1].isdigit():
            identifyier = identifyier.rsplit('_', 1)[0]
        i = 1
        while '%s_%i'%(identifyier, i) in existing:
            i += 1
        return '%s_%i'%(identifyier, i)

    def SetValue(self, row, col, value):
        # ignore unchanged values
        if value==self.GetValue(row, col):
            return

        if row==self.repeatedInfo():
            if col==10:
                self.repetitions = int(value)
            self.updateModel()
            return
        row = self.realRow(row)

        if row==0:
            to_edit = self.ambient
        elif row==(self.GetNumberRows()-2):
            to_edit = self.substrate
        else:
            to_edit = self.layers[row-1]
        if col==0:
            # name change
            old_name = to_edit[0]
            to_edit[0] = 'AboutToChangeValue'
            to_edit[0] = self.get_valid_name(value)
            self.delete_grid_items(old_name)
        elif col==2:
            # check formula
            if to_edit[1]=='Formula':
                if value=='SLD':
                    to_edit[2] = value
                else:
                    try:
                        formula = Formula.from_str(value)
                    except:
                        pass
                    else:
                        to_edit[2] = formula
                        # a new formula was set, if in DB, set its density
                        if formula in mdb:
                            to_edit[4] = '%g'%mdb.dens_mass(formula)
                        else:
                            res = get_mat_api(formula)
                            if res:
                                to_edit[4] = '%g'%res[0]
            else:
                try:
                    val = float(eval('%s'%value))
                    if 0<=val<=100:
                        to_edit[2] = value
                except:
                    pass
                else:
                    to_edit[col] = value
        elif col==1:
            # change of layer type resets material data columns
            to_edit[1] = value
            for i in [2, 3, 4, 5, 6]:
                to_edit[i] = self.defaults[value][i]
        elif col in [3, 5, 7, 9]:
            # boolean columns are always correct
            to_edit[col] = value
        elif col in [4, 6, 8, 10]:
            # evaluate float values, can be written as formla
            try:
                float(eval('%s'%value))
            except:
                pass
            else:
                to_edit[col] = value
        self.updateModel()

    def updateModel(self, evt=None):
        model_code = self.getModelCode()
        evt = update_model_event()
        evt.script = model_code
        wx.PostEvent(self.parent, evt)

    def repeatedInfo(self):
        # get the first row of the repeated layer structure
        info_row = 1
        for li in self.layers:
            if li[11]!=TOP_LAYER:
                break
            info_row += 1
        return info_row

    def realRow(self, row):
        info_row = self.repeatedInfo()
        if row<info_row:
            return row
        else:
            return row-1

    def GetAttr(self, row, col, kind):
        '''Called by the grid to find the attributes of the cell,
        bkg color, text colour, font and so on.
        '''
        attr = gridlib.GridCellAttr()
        if row==self.repeatedInfo():
            attr.SetAlignment(wx.ALIGN_LEFT, wx.ALIGN_BOTTOM)
            if col!=10:
                attr.SetReadOnly()
            if col==0:
                attr.SetSize(1, 8)
                attr.SetAlignment(wx.ALIGN_CENTER, wx.ALIGN_BOTTOM)
                return attr
            if col==8:
                attr.SetSize(1, 2)
                return attr
            if col==10:
                return attr
            return attr
        row = self.realRow(row)

        attr.SetAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTER)
        if row==0 and col==6:
            # ambiance has no thickness or roughness
            attr.SetSize(1, 5)
        if row==(self.GetRowsCount()-2) and col==6:
            # ambiance has no thickness
            attr.SetSize(1, 3)
        if row in [0, (self.GetRowsCount()-2)]:
            if row==0:
                if col==1:
                    attr.SetAlignment(wx.ALIGN_CENTER, wx.ALIGN_CENTER)
                elif col in [3, 5, 7, 9]:
                    attr.SetAlignment(wx.ALIGN_RIGHT, wx.ALIGN_CENTER)
                    # If layer is defined as fraction, only allow fitting of either
                    # density 2 or fraction.
                    if self.ambient[1]=='Mixure' and col==3 and self.ambient[5]:
                        attr.SetReadOnly()
                    elif self.ambient[1]=='Mixure' and col==5 and self.ambient[3]:
                        attr.SetReadOnly()
                else:
                    attr.SetAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTER)
                attr.SetBackgroundColour('#dddddd')
                if col in [9, 10]:
                    attr.SetReadOnly()
            else:
                if col==1:
                    attr.SetAlignment(wx.ALIGN_CENTER, wx.ALIGN_BOTTOM)
                elif col in [3, 5, 7, 9]:
                    attr.SetAlignment(wx.ALIGN_RIGHT, wx.ALIGN_BOTTOM)
                    # If layer is defined as fraction, only allow fitting of either
                    # density 2 or fraction.
                    if self.substrate[1]=='Mixure' and col==3 and self.substrate[5]:
                        attr.SetReadOnly()
                    elif self.substrate[1]=='Mixure' and col==5 and self.substrate[3]:
                        attr.SetReadOnly()
                else:
                    attr.SetAlignment(wx.ALIGN_LEFT, wx.ALIGN_BOTTOM)
                attr.SetBackgroundColour('#aaaaff')
            if col in [0, 7, 8]:
                attr.SetReadOnly()
        else:
            if col==0:
                attr.SetAlignment(wx.ALIGN_RIGHT, wx.ALIGN_CENTER)
            if col==1:
                attr.SetAlignment(wx.ALIGN_CENTER, wx.ALIGN_CENTER)
            if col in [3, 5, 7, 9]:
                attr.SetAlignment(wx.ALIGN_RIGHT, wx.ALIGN_CENTER)
                # If layer is defined as fraction, only allow fitting of either
                # density 2 or fraction.
                if self.layers[row-1][1]=='Mixure' and col==3 and self.layers[row-1][5]:
                    attr.SetReadOnly()
                elif self.layers[row-1][1]=='Mixure' and col==5 and self.layers[row-1][3]:
                    attr.SetReadOnly()
            if self.layers[row-1][11]==TOP_LAYER:
                attr.SetBackgroundColour('#ccffcc')
            elif self.layers[row-1][11]==BOT_LAYER:
                attr.SetBackgroundColour('#ffaaff')
        if col in [4, 6, 8, 10]:
            attr.SetRenderer(ValueCellRenderer())
        return attr

    def GetColLabelValue(self, col):
        '''Called when the grid needs to display labels
        '''
        return self._columns[col][0]

    def GetTypeName(self, row, col):
        '''Called to determine the kind of editor/renderer to use by
        default, doesn't necessarily have to be the same type used
        natively by the editor/renderer if they know how to convert.
        '''
        return self._columns[col][1]

    def CanGetValueAs(self, row, col, type_name):
        '''Called to determine how the data can be fetched and stored by the
        editor and renderer.  This allows you to enforce some type-safety
        in the grid.
        '''
        col_type = self._columns[col][1].split(':')[0]
        if type_name==col_type:
            return True
        else:
            return False

    def CanSetValueAs(self, row, col, type_name):
        return self.CanGetValueAs(row, col, type_name)

    def SetParameters(self, pars, clear=True, permanent_change=True):
        '''
        SetParameters(self, pars) --> None

        Set the parameters in the table to pars.
        pars has to an instance of Parameters.
        '''
        pass

    def InsertRow(self, row):
        model_row = self.realRow(row)
        if model_row==(self.GetNumberRows()-2):
            layer_type = self.substrate[1]
            layer_stack = BOT_LAYER
            model_row -= 1
        elif row==self.repeatedInfo():
            layer_type = self.layers[row-1][1]
            layer_stack = self.layers[row-1][11]
        elif model_row>0:
            layer_type = self.layers[model_row-1][1]
            layer_stack = self.layers[model_row-1][11]
        else:
            layer_type = self.ambient[1]
            layer_stack = TOP_LAYER
        newlayer = list(self.defaults[layer_type])
        newlayer[11] = layer_stack
        newlayer[0] = self.get_valid_name(newlayer[0])
        self.layers.insert(model_row, newlayer)

        msg = gridlib.GridTableMessage(self,
                                       gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1, 1)
        self.GetView().ProcessTableMessage(msg)
        msg = gridlib.GridTableMessage(self,
                                       gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.updateModel()
        return True

    def RebuildTable(self, data):
        diff = len(self.layers)-len(data)
        self.layers = list(data)

        for i in range(abs(diff)):
            if diff<0:
                # list of layers is longer than previous table data
                msg = gridlib.GridTableMessage(self,
                                               gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED,
                                               1, 1)
                self.GetView().ProcessTableMessage(msg)
            elif diff>0:
                # list of layers is shorter than previous table data
                msg = gridlib.GridTableMessage(self,
                                               gridlib.GRIDTABLE_NOTIFY_ROWS_DELETED,
                                               1, 1)
                self.GetView().ProcessTableMessage(msg)
        msg = gridlib.GridTableMessage(self,
                                       gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.updateModel()

    def DeleteRow(self, row):
        if row==self.repeatedInfo():
            return False
        row = self.realRow(row)
        if row in [0, self.GetNumberRows()-1]:
            return False
        # make sure we don't delete the last ML layer
        if self.layers[row-1][11]==ML_LAYER and len([li for li in self.layers if li[11]==ML_LAYER])==1:
            return False
        del_layer = self.layers.pop(row-1)

        # remove grid fit parameters from model, if any
        grid_parameters = self.parent.plugin.GetModel().get_parameters()
        for fi in ['dens', 'magn', 'd', 'sigma']:
            func_name = del_layer[0]+'.'+_set_func_prefix+fi.capitalize()
            grid_parameters.set_fit_state_by_name(func_name, 0., 0, 0., 0.)
        self.parent.UpdateGrid(grid_parameters)

        msg = gridlib.GridTableMessage(self,
                                       gridlib.GRIDTABLE_NOTIFY_ROWS_DELETED, 1, 1)
        self.GetView().ProcessTableMessage(msg)
        msg = gridlib.GridTableMessage(self,
                                       gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.updateModel()
        return True

    def MoveRow(self, row_from, row_to):
        ignore_rows = [0, self.repeatedInfo(), self.GetNumberRows()-1]
        if row_from in ignore_rows or row_to in ignore_rows or row_to<0:
            return False
        row_from = self.realRow(row_from)
        row_to = self.realRow(row_to)
        if self.layers[row_from-1][11]!=self.layers[row_to-1][11]:
            return False
        moved_row = self.layers.pop(row_from-1)
        self.layers.insert(row_to-1, moved_row)

        msg = gridlib.GridTableMessage(self,
                                       gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.updateModel()
        return True

    def getLayerCode(self, layer):
        out_param = {}
        output = "model.Layer("
        if layer[1]=='Formula':
            formula = layer[2]
            if formula=='SLD':
                nSLD = float(eval(layer[4]))
                mSLD = float(eval(layer[6]))
                output += "f=%s, "%(10*nSLD-10j*mSLD)
                output += "b=%s, "%nSLD
                output += "dens=0.1, magn=%s, "%mSLD
                out_param['dens'] = 0.1
                out_param['magn'] = mSLD
            else:
                dens = (eval(layer[4])*MASS_DENSITY_CONVERSION/formula.mFU())
                output += "f=%s, "%formula.f()
                output += "b=%s, "%formula.b()
                output += "dens=%s, "%dens
                output += "magn=%s, "%layer[6]
                out_param['dens'] = dens
                out_param['magn'] = float(eval(layer[6]))
        else:
            SLD1 = float(eval(layer[2]))
            SLD2 = float(eval(layer[4]))
            frac = float(eval(layer[6]))/100.
            output += "f=(%s*%s+(1-%s)*%s), "%(frac, SLD1, frac, SLD2)
            output += "b=(%s*%s+(1-%s)*%s), "%(frac, SLD1, frac, SLD2)
            output += "dens=0.1, magn=0.0, "
            out_param['dens'] = 0.1
            out_param['magn'] = 0.0
        output += "d=%s, "%layer[8]
        output += "sigma=%s, "%layer[10]
        output += "xs_ai=0.0, magn_ang=0.0)"
        out_param['d'] = float(eval(layer[8]))
        out_param['sigma'] = float(eval(layer[10]))
        return output, out_param

    invalid_identifiers = ['sample', 'Sim', 'model',
                           'fw', 'fp', 'bc', 'bw',
                           'Amb', 'Sub', 'inst']

    def getModelCode(self):
        '''
        Generate the python code for the current sample structure.
        '''
        grid_parameters = self.parent.plugin.GetModel().get_parameters()

        script = "# BEGIN Sample DO NOT CHANGE\n"
        li, oi = self.getLayerCode(self.ambient)
        script += "Amb = %s\n"%li
        for pi, fi in [(3, 'dens'), (5, 'magn')]:
            if pi==5 and self.ambient[1]=='Mixure':
                if not self.ambient[5]:
                    continue
                fi = 'dens'
            value = oi[fi]
            minval = value*0.5
            maxval = value*2.0
            func_name = 'Amb.'+_set_func_prefix+fi.capitalize()
            grid_parameters.set_fit_state_by_name(func_name, 0., 0, 0., 0.)
            grid_parameters.set_fit_state_by_name(func_name, value,
                                                  int(self.ambient[pi]), minval, maxval)

        for layer in self.layers:
            li, oi = self.getLayerCode(layer)
            script += "%s = %s\n"%(layer[0], li)
            for pi, fi in [(3, 'dens'), (5, 'magn'), (7, 'd'), (9, 'sigma')]:
                if pi==5 and layer[1]=='Mixure':
                    if not layer[5]:
                        continue
                    fi = 'dens'
                value = oi[fi]
                minval = value*0.5
                maxval = value*2.0
                func_name = layer[0]+'.'+_set_func_prefix+fi.capitalize()
                grid_parameters.set_fit_state_by_name(func_name, 0., 0, 0., 0.)
                grid_parameters.set_fit_state_by_name(func_name, value,
                                                      int(layer[pi]), minval, maxval)

        li, oi = self.getLayerCode(self.substrate)
        script += "\nSub = %s\n"%li
        for pi, fi in [(3, 'dens'), (5, 'magn'), (9, 'sigma')]:
            if pi==5 and self.substrate[1]=='Mixure':
                if not self.substrate[5]:
                    continue
                fi = 'dens'
            value = oi[fi]
            minval = value*0.5
            maxval = value*2.0
            func_name = 'Sub.'+_set_func_prefix+fi.capitalize()
            grid_parameters.set_fit_state_by_name(func_name, 0., 0, 0., 0.)
            grid_parameters.set_fit_state_by_name(func_name, value,
                                                  int(self.substrate[pi]), minval, maxval)

        self.parent.UpdateGrid(grid_parameters)

        top = [li[0] for li in self.layers if li[11]==TOP_LAYER]
        ml = [li[0] for li in self.layers if li[11]==ML_LAYER]
        bot = [li[0] for li in self.layers if li[11]==BOT_LAYER]
        script += "\nTop = model.Stack(Layers=[%s ], Repetitions = 1)\n"%str(
            ", ".join(reversed(top))
            )
        script += "\nML = model.Stack(Layers=[%s ], Repetitions = %i)\n"%(str(
            ", ".join(reversed(ml))), self.repetitions)
        script += "\nBot = model.Stack(Layers=[%s ], Repetitions = 1)\n"%str(
            ", ".join(reversed(bot))
            )
        script += "\nsample = model.Sample(Stacks = [Bot, ML, Top], Ambient = Amb, Substrate = Sub)\n" \
                  "# END Sample\n\n" \
                  "# BEGIN Parameters DO NOT CHANGE\n"
        # store data used for the last script for reuse on update
        self._last_layer_data = [list(self.ambient)]
        for li in self.layers:
            self._last_layer_data.append(list(li))
        self._last_layer_data.append(list(self.substrate))
        return script

    def delete_grid_items(self, name):
        # remove fit grid entries corresponding to a renamed layer
        grid_parameters = self.parent.plugin.GetModel().get_parameters()
        for fi in ['dens', 'magn', 'd', 'sigma']:
            func_name = name+'.'+_set_func_prefix+fi.capitalize()
            grid_parameters.set_fit_state_by_name(func_name, 0., 0, 0., 0.)
        self.parent.UpdateGrid(grid_parameters)

    def get_name_list(self):
        out = ['Amb']
        for li in self.layers:
            out.append(li[0])
        out.append('Sub')
        return out

    def update_layer_parameters(self, layer, dens=None, magn=None,
                                d=None, sigma=None):
        # update the table during/after a fit, layer can be index or name
        if type(layer) is not int:
            layer = self.get_name_list().index(layer)
        if layer==0:
            data = self.ambient
        elif layer==(len(self.layers)+1):
            data = self.substrate
        else:
            data = self.layers[layer-1]
        ref_data = self._last_layer_data[layer]

        if data[1]=='Formula':
            formula = data[2]
            if formula=='SLD':
                if dens is not None and data[3]:
                    data[4] = str(eval(ref_data[4])*dens/0.1)
                if magn is not None and data[5]:
                    data[6] = str(eval(ref_data[6])*dens/0.1)
            else:
                if dens is not None and data[3]:
                    new_dens = float(dens)*formula.mFU()/MASS_DENSITY_CONVERSION
                    data[4] = str(new_dens)
                if magn is not None and data[5]:
                    data[6] = str(float(magn))
        elif dens is not None:
            # FIXIT: now yet working as calculation compares with current value
            SLD1 = float(eval(ref_data[2]))
            SLD2 = float(eval(ref_data[4]))
            frac = float(eval(ref_data[6]))/100.
            new_dens = (frac*SLD1+(1-frac)*SLD2)*dens/0.1
            if data[3]:
                # SLD-2 was fitted
                sld2_fraction = new_dens-frac*SLD1
                data[4] = str(sld2_fraction/(1.-frac))
            if data[5]:
                # percentage was fitted
                new_frac = (new_dens-SLD2)/(SLD1-SLD2)
                data[6] = str(new_frac*100.)
        if d is not None and data[7]:
            data[8] = str(float(d))
        if sigma is not None and data[9]:
            data[10] = str(float(sigma))


class SamplePanel(wx.Panel):
    model: Model
    last_sample_script = ''

    inst_params = dict(probe='neutron pol',
                       I0=1.0,
                       res=0.0,
                       wavelength=1.54,
                       pol='uu',
                       Ibkg=0.,
                       samplelen=10.,
                       beamw=0.1,
                       footype='no corr',
                       name='inst',
                       coords='q')

    def __init__(self, parent, plugin, refindexlist=None):
        wx.Panel.__init__(self, parent)
        if refindexlist is None:
            refindexlist = []
        self.refindexlist = refindexlist
        self.plugin = plugin
        self.variable_span = 0.25
        self.inst_params = dict(SamplePanel.inst_params)

        # Colours indicating different states
        # Green wx.Colour(138, 226, 52), ORANGE wx.Colour(245, 121, 0)
        self.fit_colour = (245, 121, 0)
        # Tango Sky blue wx.Colour(52, 101, 164), wx.Colour(114, 159, 207)
        self.const_fit_colour = (114, 159, 207)

        boxver = wx.BoxSizer(wx.HORIZONTAL)
        boxhor = wx.BoxSizer(wx.VERTICAL)
        self.toolbar = wx.ToolBar(self, style=wx.TB_FLAT | wx.TB_HORIZONTAL)
        boxhor.Add((-1, 2))
        self.do_toolbar()
        boxhor.Add(self.toolbar, proportion=0, flag=wx.EXPAND, border=1)
        boxhor.Add((-1, 2))
        self.grid = SampleGrid(self, -1, style=wx.NO_BORDER)
        self.sample_table = SampleTable(self, self.grid)

        self.last_sample_script = ''  # self.sample_table.getModelCode()
        self.Bind(EVT_UPDATE_MODEL, self.UpdateModel)
        boxhor.Add(self.grid, 1, wx.EXPAND)
        boxver.Add(boxhor, 1, wx.EXPAND)

        self.grid.info_text = wx.StaticText(self, -1, 'Bier')
        boxver.Add(self.grid.info_text, 0)
        self.grid.info_text.Hide()

        self.SetSizer(boxver)
        self.toolbar.Realize()
        self.update_callback = lambda event: ''
        self._last_grid_data = []

    def do_toolbar(self):
        dpi_scale_factor = wx.GetApp().dpi_scale_factor
        tb_bmp_size = int(dpi_scale_factor*20)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Insert Layer',
                             bitmap=wx.Bitmap(images.insert_layer.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Insert a Layer')
        self.Bind(wx.EVT_TOOL, self.OnLayerAdd, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Delete',
                             bitmap=wx.Bitmap(images.delete.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Delete item')
        self.Bind(wx.EVT_TOOL, self.OnLayerDelete, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Move up',
                             bitmap=wx.Bitmap(images.move_up.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Move item up')
        self.Bind(wx.EVT_TOOL, self.MoveUp, id=newid)

        newid = wx.NewId()
        self.toolbar.AddTool(newid, 'Move down',
                             bitmap=wx.Bitmap(images.move_down.GetImage().Scale(tb_bmp_size, tb_bmp_size)),
                             shortHelp='Move item down')
        self.Bind(wx.EVT_TOOL, self.MoveDown, id=newid)
        self.toolbar.AddSeparator()

        newid = wx.NewId()
        button = wx.Button(self.toolbar, newid, label='Instrument Settings',
                           size=(132*dpi_scale_factor, 22*dpi_scale_factor))
        button.SetBitmap(wx.Bitmap(images.instrument.GetImage().Scale(tb_bmp_size, tb_bmp_size)), dir=wx.LEFT)
        self.toolbar.AddControl(button)
        self.Bind(wx.EVT_BUTTON, self.EditInstrument, id=newid)
        self.toolbar.AddSeparator()

        self.toolbar.AddStretchableSpace()

        newid = wx.NewId()
        button = wx.Button(self.toolbar, newid, label='to Advanced Modelling',
                           size=(150*dpi_scale_factor, 22*dpi_scale_factor))
        button.SetBitmap(wx.Bitmap(images.custom_parameters.GetImage().Scale(tb_bmp_size, tb_bmp_size)), dir=wx.LEFT)
        button.SetToolTip("Switch to Reflectivity plugin for advanced modeling options.\n"
                          "This converts the model and can't be undone.")
        self.toolbar.AddControl(button)
        self.Bind(wx.EVT_BUTTON, self.SwitchAdvancedReflectivity, id=newid)

    def set_model(self, model: Model):
        self.model = model

    def ChangeRepetitions(self, evt):
        self.UpdateModel()

    def instrumentCode(self):
        pars = dict(self.inst_params)
        if pars['res']==0:
            pars['restype'] = 'no conv'
        else:
            pars['restype'] = 'full conv and varying res.'
        template = "%(name)s = model.Instrument(probe='%(probe)s', wavelength=%(wavelength)g, " \
                   "coords='%(coords)s', I0=%(I0)g, res=%(res)g, " \
                   "restype='%(restype)s', respoints=11, " \
                   "resintrange=3, beamw=%(beamw)g, footype='%(footype)s', " \
                   "samplelen=%(samplelen)g, incangle=0.0, pol='%(pol)s', " \
                   "Ibkg=%(Ibkg)g, tthoff=0.0,)\n"
        if pars['probe'] in ['x-ray', 'neutron']:
            return ['inst'], template%pars
        output = template%pars
        if pars['probe']=='neutron pol':
            pars['pol'] = 'dd'
            pars['name'] = 'inst_down'
            output += template%pars
            insts = ['inst', 'inst_down']
        else:
            insts = ['inst']
        return insts, output

    def UpdateModel(self, evt=None, first=False, re_color=False):
        coords = self.inst_params['coords']
        if evt in [None, 'inst']:
            sample_script = self.last_sample_script
        else:
            sample_script = evt.script
            self.last_sample_script = sample_script
        script = 'from numpy import *\n' \
                 'import models.spec_nx as model\n' \
                 'from models.utils import UserVars, fp, fw, bc, bw\n\n' \
                 '# BEGIN Instrument DO NOT CHANGE\n' \
                 'from models.utils import create_fp, create_fw\n'
        insts, inst_str = self.instrumentCode()
        script += inst_str

        script += "inst_fp = create_fp(inst.wavelength); inst_fw = create_fw(inst.wavelength)\n" \
                  "fp.set_wavelength(inst.wavelength); fw.set_wavelength(inst.wavelength)\n" \
                  "# END Instrument\n\n"
        # add sample description code
        script += sample_script
        script += "cp = UserVars()\n" \
                  "# END Parameters\n\n" \
                  "SLD = []\n" \
                  "def Sim(data):\n" \
                  "    I = []\n" \
                  "    SLD[:] = []\n"
        datasets = self.model.data
        from genx import data
        if coords=='q':
            data.DataSet.simulation_params[0] = 0.001
            data.DataSet.simulation_params[1] = 0.601
        else:
            data.DataSet.simulation_params[0] = 0.01
            data.DataSet.simulation_params[1] = 6.01
        res_set = False
        for i, di in enumerate(datasets):
            di.run_command()
            script += "    # BEGIN Dataset %i DO NOT CHANGE\n" \
                      "    d = data[%i]\n"%(i, i)
            if hasattr(di, 'res') and di.res is not None:
                script += "    inst.setRes(data[%i].res)\n"%i
            elif res_set:
                script += "    inst.setRes(0.001)\n"
            inst_id = i%len(insts)
            try:
                # extract polarization channel from ORSO metadata
                pol = di.meta['data_source']['measurement']['instrument_settings']['polarization']
                if pol=='unpolarized':
                    pass
                elif pol in ['m', 'mm']:
                    inst_id = 1%len(insts)
                else:
                    inst_id = 0
            except KeyError:
                pass
            script += "    I.append(sample.SimSpecular(d.x, %s))\n" \
                      "    if _sim: SLD.append(sample.SimSLD(None, None, inst))\n" \
                      "    # END Dataset %i\n"%(insts[inst_id], i)
            if re_color and len(insts)>1:
                if inst_id==0:
                    di.data_color = (0.7, 0.0, 0.0)
                    di.sim_color = (1.0, 0.0, 0.0)
                else:
                    di.data_color = (0.0, 0.0, 0.7)
                    di.sim_color = (0.0, 0.0, 1.0)
            elif di.name.startswith('Data') and len(insts)>1:
                prefix = ['Spin Up', 'Spin Down'][inst_id]
                di.name = prefix+' %i'%(i//2+1)
                if inst_id==0:
                    di.data_color = (0.7, 0.0, 0.0)
                    di.sim_color = (1.0, 0.0, 0.0)
                else:
                    di.data_color = (0.0, 0.0, 0.7)
                    di.sim_color = (0.0, 0.0, 1.0)
            elif (di.name.startswith('Spin') and len(insts)==1) or first:
                di.name = 'Data %i'%i
                di.data_color = (0.0, 0.7, 0.0)
                di.sim_color = (0.0, 1.0, 0.0)
        datasets.update_data()
        # TODO: this is a very ugly unstable solution, try to find alternative.
        self.plugin.parent.data_list.list_ctrl._UpdateImageList()

        script += "    return I"
        self.plugin.SetModelScript(script)

        if evt is not None and self.plugin.mb_autoupdate_sim.IsChecked():
            self.plugin.parent.eh_tb_simulate(None)

    def OnLayerAdd(self, evt):
        row = self.grid.GetGridCursorRow()
        self.sample_table.InsertRow(row)

    def OnLayerDelete(self, evt):
        row = self.grid.GetGridCursorRow()
        self.sample_table.DeleteRow(row)

    def SetUpdateCallback(self, func):
        ''' SetUpdateCallback(self, func) --> None

        Sets the update callback will be called when the sample is updated.
        The call is on the form func(event)
        '''
        self.update_callback = func

    def Update(self, update_script=True):
        if update_script:
            self.update_callback(None)

    def EditInstrument(self, evt):
        """Event handler that creates an dialog box to edit the instruments.

        :param evt:
        :return: Nothing
        """
        validators = {
            'probe': ['x-ray', 'neutron', 'neutron pol'],
            'coords': ['q', '2θ'], 'I0': FloatObjectValidator(),
            'res': FloatObjectValidator(), 'wavelength': FloatObjectValidator(),
            'Ibkg': FloatObjectValidator(), 'samplelen': FloatObjectValidator(),
            'beamw': FloatObjectValidator(),
            'footype': ['no corr', 'gauss beam', 'square beam'],
            }
        inst_name = 'inst'
        vals = {}
        editable = {}
        grid_parameters = self.plugin.GetModel().get_parameters()

        pars = ['probe', 'coords', 'wavelength', 'I0', 'Ibkg',
                'res', 'footype', 'samplelen', 'beamw']
        units = {
            'probe': '', 'wavelength': 'Å', 'coords': '',
            'I0': 'arb.', 'res': '[coord]', 'beamw': 'mm',
            'footype': '', 'samplelen': 'mm',
            'Ibkg': 'arb.', '2θ': '°', 'q': 'Å$^-1$'
            }
        groups = [('Radiation', ('probe', 'wavelength', 'I0')),
                  ('Data', ('coords', 'Ibkg', 'res')),
                  ('Footprint Correction', ('footype', 'samplelen', 'beamw'))]
        for item in validators.keys():
            func_name = inst_name+'.'+_set_func_prefix+item.capitalize()
            editable[item] = grid_parameters.get_fit_state_by_name(func_name)
            val = self.inst_params[item]
            vals[item] = val

        dlg = ValidateFitDialog(self, pars, vals, validators,
                                title='Instrument Editor', groups=groups,
                                units=units,
                                editable_pars=editable)

        if dlg.ShowModal()==wx.ID_OK:
            vals = dlg.GetValues()
            states = dlg.GetStates()
            for key, value in vals.items():
                if type(self.inst_params[key]) is not str:
                    value = float(vals[key])
                    minval = value*0.5
                    maxval = value*2.0
                    func_name = inst_name+'.'+_set_func_prefix+key.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value, states[key], minval, maxval)
                    self.inst_params[key] = value
                else:
                    self.inst_params[key] = value
            self.UpdateGrid(grid_parameters)
            self.UpdateModel(evt='inst')
        else:
            pass
        dlg.Destroy()

    def SwitchAdvancedReflectivity(self, evt):
        # Load Reflectivity plugin for more advanced options, this will unload current plugin
        plugin_control = self.plugin.parent.plugin_control

        try:
            plugin_control.plugin_handler.load_plugin('Reflectivity')
        except:
            outp = io.StringIO()
            traceback.print_exc(200, outp)
            tbtext = outp.getvalue()
            outp.close()
        else:
            plugin_control.RegisterPlugin('Reflectivity')

    def CheckGridUpdate(self, parameters=None):
        if parameters is None:
            new_grid = self.plugin.GetModel().get_parameters()
            new_data = [list(di) for di in new_grid.data]
        else:
            new_data = parameters
        if self._last_grid_data!=new_data:
            layers = self.sample_table.get_name_list()
            for (pi, val, _, _, _, _) in new_data:
                try:
                    name, param = pi.split('.', 1)
                except ValueError:
                    continue

                if name in layers:
                    if param==_set_func_prefix+'Dens':
                        self.sample_table.update_layer_parameters(name, dens=val)
                    if param==_set_func_prefix+'Magn':
                        self.sample_table.update_layer_parameters(name, magn=val)
                    if param==_set_func_prefix+'D':
                        self.sample_table.update_layer_parameters(name, d=val)
                    if param==_set_func_prefix+'Sigma':
                        self.sample_table.update_layer_parameters(name, sigma=val)
            msg = gridlib.GridTableMessage(self.sample_table,
                                           gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
            self.sample_table.GetView().ProcessTableMessage(msg)
            self.sample_table.GetView().ForceRefresh()
            self._last_grid_data = new_data

    def UpdateGrid(self, grid_parameters):
        self._last_grid_data = [list(di) for di in grid_parameters.data]
        self.plugin.parent.paramter_grid.SetParameters(grid_parameters)

    def MoveUp(self, evt):
        row = self.grid.GetGridCursorRow()
        res = self.sample_table.MoveRow(row, row-1)
        if res:
            self.grid.MoveCursorUp(False)

    def MoveDown(self, evt):
        row = self.grid.GetGridCursorRow()
        res = self.sample_table.MoveRow(row, row+1)
        if res:
            self.grid.MoveCursorDown(False)


class WizarSelectionPage(WizardPageSimple):

    def __init__(self, parent, choices, intro_title='', intro_text='',
                 choice_label='', choices_help=None,
                 prev=None, next=None, bitmap=wx.NullBitmap):
        WizardPageSimple.__init__(self, parent, prev=prev, next=next, bitmap=bitmap)
        dpi_scale_factor = wx.GetApp().dpi_scale_factor

        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        text = wx.StaticText(self, style=wx.ALIGN_LEFT, label=intro_title)
        text.SetFont(text.GetFont().Scale(1.5))
        vbox.Add(text, 0, wx.EXPAND)
        text = wx.StaticText(self, style=wx.ALIGN_LEFT, label=intro_text)
        text.Wrap(text.GetSize().width)
        vbox.Add(text, 0, wx.EXPAND)
        vbox.AddStretchSpacer()

        self.ctrl = {}

        box = wx.Window(self)
        out_layout = wx.BoxSizer(wx.VERTICAL)
        box.SetSizer(out_layout)
        box_layout = wx.GridSizer(min(4, len(choices)//4+1), 2, 2)
        out_layout.Add(box_layout, 1, wx.EXPAND | wx.TOP | wx.LEFT, 12*dpi_scale_factor)
        for choice in choices:
            self.ctrl[choice] = wx.RadioButton(box, label=choice,
                                               size=wx.Size(-1, 16*dpi_scale_factor))
            box_layout.Add(self.ctrl[choice], 0, wx.FIXED_MINSIZE, 2)
        self.ctrl[choices[0]].SetValue(True)

        vbox.Add(box, 0, 0)
        if choices_help:
            for i, ti in enumerate(choices_help):
                # self.ctrl.SetItemToolTip(i, ti)
                self.ctrl[choices[i]].SetToolTip(ti)
            text = wx.StaticText(self, style=wx.ALIGN_LEFT, label='Hover items for info')
            text.SetFont(text.GetFont().Scale(0.75))
            vbox.Add(text, 0, 0)

    def selection(self):
        # return self.ctrl.GetString(self.ctrl.GetSelection())
        for key, ctrl in self.ctrl.items():
            if ctrl.GetValue():
                return key


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
        if line.find('# BEGIN %s'%descriptor)!=-1:
            found += 1
            break

    text = ''
    for line in script_lines[line_index:]:
        line_index += 1
        if line.find('# END %s'%descriptor)!=-1:
            found += 1
            break
        text += line

    if found!=2:
        raise LookupError('Code segement: %s could not be found'%descriptor)

    return text


class Plugin(framework.Template):
    previous_xaxis = None
    sim_returns_sld = True
    model_obj: Model

    def __init__(self, parent):
        if 'Reflectivity' in parent.plugin_control.plugin_handler.get_loaded_plugins():
            parent.plugin_control.UnLoadPlugin_by_Name('Reflectivity')
        framework.Template.__init__(self, parent)

        self.model_obj = self.GetModel()
        prev_script = self.model_obj.script
        sample_panel = self.NewInputFolder('Model')
        sample_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sample_panel.SetSizer(sample_sizer)
        self.defs = ['Instrument', 'Sample']
        self.sample_widget = SamplePanel(sample_panel, self)
        self.sample_widget.set_model(self.model_obj)
        sample_sizer.Add(self.sample_widget, 1, wx.EXPAND | wx.GROW | wx.ALL)
        sample_panel.Layout()

        self.sample_widget.SetUpdateCallback(self.UpdateScript)

        # Create the SLD plot
        sld_plot_panel = self.NewPlotFolder('SLD')
        sld_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sld_plot_panel.SetSizer(sld_sizer)
        self.sld_plot = SamplePlotPanel(sld_plot_panel, self)
        sld_sizer.Add(self.sld_plot, 1, wx.EXPAND | wx.GROW | wx.ALL)
        sld_plot_panel.Layout()

        # Create a menu for handling the plugin
        menu = self.NewMenu('Reflec')
        self.mb_export_sld = wx.MenuItem(menu, wx.NewId(),
                                         "Export SLD...",
                                         "Export the SLD to a ASCII file",
                                         wx.ITEM_NORMAL)
        menu.Append(self.mb_export_sld)
        self.mb_show_imag_sld = wx.MenuItem(menu, wx.NewId(),
                                            "Show Im SLD",
                                            "Toggles showing the imaginary part of the SLD",
                                            wx.ITEM_CHECK)
        menu.Append(self.mb_show_imag_sld)
        self.mb_show_imag_sld.Check(self.sld_plot.opt.show_imag)
        self.mb_autoupdate_sim = parent.mb_checkables[MenuId.AUTO_SIM]
        self.mb_autoupdate_sim.Check(True)
        self.mb_autoupdate_sld = wx.MenuItem(menu, wx.NewId(),
                                             "Autoupdate SLD",
                                             "Toggles autoupdating the SLD during fitting",
                                             wx.ITEM_CHECK)
        menu.Append(self.mb_autoupdate_sld)
        self.mb_autoupdate_sld.Check(True)

        self.mb_hide_advanced = wx.MenuItem(menu, wx.NewId(),
                                            "Hide Advanced",
                                            "Toggles hiding of advanced model tabs",
                                            wx.ITEM_CHECK)
        menu.Append(self.mb_hide_advanced)
        self.mb_hide_advanced.Check(True)

        if prev_script!='':
            if self.model_obj.filename!='':
                iprint("SimpleReflectivity plugin: Reading loaded model")
                self.ReadModel()
            else:
                try:
                    self.ReadModel()
                except:
                    iprint("SimpleReflectivity plugin: Creating new model")
                    self.OnNewModel(None)
        else:
            iprint("SimpleReflectivity plugin: Creating new model")
            self.OnNewModel(None)

        # self.mb_autoupdate_sld.SetCheckable(True)
        self.parent.Bind(wx.EVT_MENU, self.OnExportSLD, self.mb_export_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnAutoUpdateSLD,
                         self.mb_autoupdate_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnShowImagSLD,
                         self.mb_show_imag_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnHideAdvanced,
                         self.mb_hide_advanced)

        self.DisableGrid()
        self.HideUIElements()
        self.sample_widget.UpdateModel()
        self.StatusMessage('Simple Reflectivity plugin loaded')

        self.parent.Bind(EVT_UPDATE_PARAMETERS, self.OnFitParametersUpdated)
        self.parent.Bind(EVT_PARAMETER_GRID_CHANGE, self.OnGridMayHaveErrors)
        self.parent.model_control.Bind(EVT_UPDATE_SCRIPT, self.ReadUpdateModel)

    def OnHideAdvanced(self, evt):
        if self.mb_hide_advanced.IsChecked():
            self.HideUIElements()
        else:
            self.ShowUIElements()

    def HideUIElements(self):
        self._hidden_pages = []
        # hide all standard tabs
        nb = self.parent.input_notebook
        for i, page_i in reversed(list(enumerate(nb.Children))):
            title = nb.GetPageText(i)
            if title!='Model':
                self._hidden_pages.append([title, page_i, i])
                nb.RemovePage(i)
        self._hidden_pages.reverse()

    def ShowUIElements(self):
        nb = self.parent.input_notebook
        for title, page_i, i in self._hidden_pages:
            nb.InsertPage(i, page_i, title)
        self._hidden_pages = None

    def DisableGrid(self):
        nb = self.parent.input_notebook
        for i, page_i in enumerate(nb.Children):
            title = nb.GetPageText(i)
            if title=='Grid':
                page_i.Disable()

    def EnableGrid(self):
        nb = self.parent.input_notebook
        for i, page_i in enumerate(nb.Children):
            title = nb.GetPageText(i)
            if title=='Grid':
                page_i.Enable()

    def Remove(self):
        if self.mb_hide_advanced.IsChecked():
            self.ShowUIElements()
        self.EnableGrid()

        self.parent.Unbind(EVT_UPDATE_PARAMETERS, handler=self.OnFitParametersUpdated)
        self.parent.Unbind(EVT_PARAMETER_GRID_CHANGE, handler=self.OnGridMayHaveErrors)
        framework.Template.Remove(self)

    def UpdateScript(self, event):
        self.WriteModel()

    def OnAutoUpdateSLD(self, evt):
        # self.mb_autoupdate_sld.Check(not self.mb_autoupdate_sld.IsChecked())
        pass

    def OnShowImagSLD(self, evt):
        self.sld_plot.opt.show_imag = self.mb_show_imag_sld.IsChecked()
        self.sld_plot.WriteConfig()
        self.sld_plot.Plot()

    def OnExportSLD(self, evt):
        dlg = wx.FileDialog(self.parent, message="Export SLD to ...",
                            defaultFile="",
                            wildcard="Dat File (*.dat)|*.dat",
                            style=wx.FD_SAVE | wx.FD_CHANGE_DIR
                            )
        if dlg.ShowModal()==wx.ID_OK:
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
                except IOError as e:
                    self.ShowErrorDialog(e.__str__())
                except Exception as e:
                    outp = io.StringIO()
                    traceback.print_exc(200, outp)
                    val = outp.getvalue()
                    outp.close()
                    self.ShowErrorDialog('Could not save the file.'
                                         ' Python Error:\n%s'%(val,))
        dlg.Destroy()

    def OnNewModel(self, event):
        ''' Create a new model
        '''
        dlg = self.parent.data_list.list_ctrl.data_loader_cont
        dls = list(sorted(dlg.plugin_handler.get_possible_plugins()))
        dls.remove('auto')

        wiz = Wizard(self.parent, title='Create New Model...',
                     bitmap=images.getinstrumentBitmap(), style=wx.DEFAULT_DIALOG_STYLE | wx.STAY_ON_TOP)

        # def show_dl_help(event):
        #     dlg=PluginHelpDialog(wiz, 'plugins.data_loaders')
        #     dlg.Show()
        p1 = WizarSelectionPage(wiz, ['x-ray', 'neutron', 'neutron pol'],
                                'Select Probe\n',
                                'Please choose the experiment you want to model'
                                '/fit. This option can be changed later '
                                'from the Instrument Settings dialog.')
        p2 = WizarSelectionPage(wiz, ['auto']+dls,
                                'Selecte Data Loader\n',
                                'How to import data into GenX. '
                                '\nIf your instrument is not listed '
                                'use the default/resolution loader. '
                                'This reads  3/4 columns from an ASCII file. '
                                'If the file does not '
                                'have x,y,dy(,res) column order, use Setting->'
                                'Import to select which columns to read.',
                                bitmap=getopenBitmap())
        p3 = WizarSelectionPage(wiz, ['q', '2θ'],
                                'Select Data Coordinates\n',
                                'Set the x-coordinates used for simulation. '
                                '\nYou can define more experimental parameters like '
                                'wavelength, resolution, footprint correction or '
                                'background in the Instrument Settings dialog.',
                                choices_help=['Reciprocal lattice vector out-of-plane component qz in Å⁻¹',
                                              'Detector angle in degrees'],
                                bitmap=getplottingBitmap())
        WizardPageSimple.Chain(p1, p2)
        WizardPageSimple.Chain(p2, p3)

        dpi_scale_factor = wx.GetApp().dpi_scale_factor
        wiz.SetPageSize(wx.Size(int(400*dpi_scale_factor), int(240*dpi_scale_factor)))

        if wiz.RunWizard(p1):
            self.sample_widget.inst_params = dict(SamplePanel.inst_params)
            params = self.sample_widget.inst_params
            params['probe'] = p1.selection()
            params['coords'] = p3.selection()
            dlg.LoadPlugin(p2.selection())
            # dlg.plugin_handler.load_plugin(p2.selection())
            self.sample_widget.sample_table.ResetModel()
            self.sample_widget.UpdateModel(first=True)
        wiz.Destroy()

    def OnDataChanged(self, event):
        ''' Take into account changes in data..
        '''
        self.sample_widget.UpdateModel()

    def OnOpenModel(self, event):
        '''
        Loads the sample into the plugin...
        '''
        self.mb_show_imag_sld.Check(self.sld_plot.opt.show_imag)
        # self.ReadModel()

    def OnSimulate(self, event):
        '''
        Updates stuff after simulation
        '''
        if not self.mb_autoupdate_sld.IsChecked():
            # Calculate and update the sld plot, don't repeat when OnFittingUpdate does the call already
            wx.CallAfter(self.sld_plot.Plot)

    def OnFittingUpdate(self, event):
        '''
        Updates stuff during fitting
        '''
        # Calculate and update the sld plot
        if self.mb_autoupdate_sld.IsChecked():
            wx.CallAfter(self.sld_plot.Plot)

    def OnGridChange(self, event):
        pass
        # self.sample_widget.CheckGridUpdate()
        # self.sample_widget.Update(update_script=False)

    @skips_event
    def OnGridMayHaveErrors(self, event):
        errors = [pi.error for pi in self.model_obj.parameters if pi.fit]
        if len(errors)>0 and not '-' in errors:
            #  summarize error information on fit parameters
            error_data = []
            ST = self.sample_widget.sample_table
            layers = ST.get_name_list()
            for pi in self.model_obj.parameters:
                try:
                    name, param = pi.name.split('.', 1)
                except ValueError:
                    continue
                show_data = ['', '', '', '']
                if name in layers:
                    lidx = ST.get_name_list().index(name)
                    if lidx==0:
                        st_data = ST.ambient
                        name = 'Ambient'
                    elif lidx==(len(ST.layers)+1):
                        name = 'Substrate'
                        st_data = ST.substrate
                    else:
                        st_data = ST.layers[lidx-1]
                    if param==_set_func_prefix+'Dens':
                        if st_data[1]=='Formula':
                            formula = st_data[2]
                            if formula=='SLD':
                                # result is SLD
                                show_data = [name, 'SLD [10^6 Å^-1]', '%.6g'%pi.value, pi.error]
                            else:
                                emin, emax = map(float, pi.error.strip('(').strip(')').strip().split(','))
                                scale = formula.mFU()/MASS_DENSITY_CONVERSION
                                show_data = [name, 'density [g/cm³]', '%.6g'%(pi.value*scale),
                                             '(%.4g, %.4g)'%(emin*scale, emax*scale)]
                        else:
                            show_data = [name, 'density', '%.6g'%pi.value, pi.error]
                    elif param==_set_func_prefix+'Magn':
                        show_data = [name, 'magnetization', '%.6g'%pi.value, pi.error]
                    elif param==_set_func_prefix+'D':
                        show_data = [name, 'thickness [Å]', '%.6g'%pi.value, pi.error]
                    elif param==_set_func_prefix+'Sigma':
                        show_data = [name, 'roughness [Å]', '%.6g'%pi.value, pi.error]
                    else:
                        show_data = [name, param, '%.6g'%pi.value, pi.error]
                else:
                    show_data = [pi.name, '', '%.6g'%pi.value, pi.error]
                error_data.append(show_data)
            dia = wx.Dialog(self.parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
            dia.SetTitle('Parameter Error Estimation')
            box = wx.BoxSizer(wx.VERTICAL)
            dia.SetSizer(box)
            grid = wx.grid.Grid(dia)
            grid.CreateGrid(len(error_data), 4)
            grid.DisableCellEditControl()
            grid.SetColLabelValue(0, 'Item')
            grid.SetColLabelValue(1, 'Parameter')
            grid.SetColLabelValue(2, 'Value')
            grid.SetColLabelValue(3, 'Error min/max')
            for i, row in enumerate(error_data):
                for j, cell_value in enumerate(row):
                    grid.SetCellValue(i, j, cell_value)
            for i in range(4):
                grid.AutoSizeColumn(i)
            box.Add(grid, proportion=1, flag=wx.EXPAND)

            w = min(grid.GetBestSize().width+80, self.parent.GetSize().width//1.5)
            h = min(grid.GetBestSize().height+80, self.parent.GetSize().height//1.3)
            dia.SetSize(wx.Size(w, h))

            dia.ShowModal()

    @skips_event
    def OnFitParametersUpdated(self, event):
        grid_parameters = self.GetModel().get_parameters()
        keys = grid_parameters.get_fit_pars()[1]
        values = event.values
        parameters = [(key, value, None, None, None, None) for key, value in zip(keys, values)]
        self.sample_widget.CheckGridUpdate(parameters=parameters)
        self.sample_widget.Update(update_script=False)
        if event.permanent_change:
            for pi, val in zip(keys, values):
                try:
                    name, param = pi.split('.', 1)
                except ValueError:
                    continue
                if name=='inst':
                    for key in self.sample_widget.inst_params.keys():
                        if param==_set_func_prefix+key.capitalize():
                            value = float(val)
                            self.sample_widget.inst_params[key] = value
                            minval = value*0.5
                            maxval = value*2.0
                            grid_parameters.set_fit_state_by_name(pi, value, 0, 0, 0)
                            grid_parameters.set_fit_state_by_name(pi, value, 1, minval, maxval)
            self.sample_widget.sample_table.updateModel()

    def WriteModel(self):
        return

    def ReadUpdateModel(self, evt):
        try:
            self.ReadModel(verbose=False)
        except GenxError:
            pass
        except Exception as e:
            self.StatusMessage(f'could not analyze script: {e}')

    def ReadModel(self, verbose=True):
        '''
        Reads in the current model and locates layers and stacks
        and sample defined inside BEGIN Sample section.
        '''
        if verbose: self.StatusMessage('Compiling the script...')
        self.CompileScript()

        if verbose: self.StatusMessage('Trying to interpret the script...')

        txt = self.GetModel().script
        grid_parameters = self.GetModel().get_parameters()
        # extract instrument parameters
        insttxt = find_code_segment(txt, 'Instrument')
        items = {}
        for li in insttxt.splitlines():
            if not li.split('=', 1)[0].strip()=='inst':
                continue
            instoptions = li.split('model.Instrument(')[1].strip()[:-1]
            instoptions = instoptions.strip(',').split(',')
            items = dict([tuple(map(str.strip, i.split('=', 1))) for i in instoptions])
        for key, value in self.sample_widget.inst_params.items():
            if key in items:
                newval = type(value)(eval(items[key]))
                self.sample_widget.inst_params[key] = newval

        # extract layer parameters
        sampletxt = find_code_segment(txt, 'Sample')
        layers = {}
        layer_order = []
        stacks = {}
        repetitions = 1
        for li in sampletxt.splitlines():
            if 'model.Layer' in li:
                name, ltxt = map(str.strip, li.split('=', 1))
                try:
                    layers[name] = analyze_layer_txt(name, ltxt)
                except Exception as e:
                    self.StatusMessage('Script read failed, SimpleReflectivity')
                    return
                layer_order.append(name)
            if 'model.Stack' in li:
                name, stxt = map(str.strip, li.split('=', 1))
                stacks[name] = analyze_stack_txt(stxt)
                if name=='ML':
                    repetitions = int(stxt.split('Repetitions', 1)[1].split('=', 1)[1].split(',', 1)[0].rstrip(')'))
        if 'Top' in stacks:
            Top = stacks['Top']
        else:
            Top = []
        if 'Bot' in stacks:
            Bottom = stacks['Bot']
        else:
            Bottom = []
        for ni, di in layers.items():
            if ni in Top:
                di.append(TOP_LAYER)
            elif ni in Bottom:
                di.append(BOT_LAYER)
            else:
                di.append(ML_LAYER)

        # get parameters to be fitted
        for li in layers.values():
            prefix = li[0]+'.set'
            fit_items = [(3, 'dens'), (5, 'magn'), (7, 'd'), (9, 'sigma')]
            for index, si in fit_items:
                if grid_parameters.get_fit_state_by_name(prefix+si.capitalize()):
                    li[index] = True

        table = self.sample_widget.sample_table
        table.ambient = layers['Amb']
        table.ambient[0] = None
        table.substrate = layers['Sub']
        table.substrate[0] = None
        table.repetitions = repetitions
        layers = [layers[key] for key in layer_order if not key in ['Amb', 'Sub']]

        table.RebuildTable(layers)

        if verbose: self.StatusMessage('New sample loaded to plugin!')


def analyze_layer_txt(name, txt):
    # ['Iron', 'Formula', Formula([['Fe', 1.0]]), False, '7.87422', False, '3.0', True, '100.0', False, '5.0']
    output = [name]
    layeroptions = txt.split('model.Layer(')[1].strip()[:-1]
    items = layeroptions.strip(',').split(',')
    items = dict([tuple(map(str.strip, i.split('=', 1))) for i in items])

    if 'bc.' in items['b'] or 'bw.' in items['b']:
        output.append('Formula')
        output.append(Formula.from_bstr(items['b']))
        # density
        output.append(False)
        if 'bc.' in items['b']:
            dens = float(items['dens'])*output[2].mFU()/MASS_DENSITY_CONVERSION
        else:
            dens = float(items['dens'])
        output.append(str(dens))
        # magnetization
        output.append(False)
        output.append(str(eval(items['magn'])))
    elif '+' in items['b'] and '*' in items['b']:
        output.append('Mixure')
        part1, part2 = items['b'].strip('()').split('+', 1)
        frac = float(part1.split('*')[0])*100
        SLD1 = float(part1.split('*')[1])
        SLD2 = float(part2.split('*')[1])
        output.append(str(SLD1))
        output.append(False)
        output.append(str(SLD2))
        output.append(False)
        output.append(str(frac))
    else:
        output.append('Formula')
        output.append('SLD')
        # density
        output.append(False)
        output.append(str(eval(items['b'])))
        # magnetization
        output.append(False)
        output.append(str(eval(items['magn'])))

    # thickness
    output.append(False)
    output.append(str(eval(items['d'])))
    # roughness
    output.append(False)
    output.append(str(eval(items['sigma'])))
    return output


def analyze_stack_txt(txt):
    txt = txt.split('Layers', 1)[1].split(']', 1)[0].strip('=[ ')
    layers = map(str.strip, txt.split(','))
    return list(layers)
