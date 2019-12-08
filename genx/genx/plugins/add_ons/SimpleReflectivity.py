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

from .. import add_on_framework as framework
from genx.plotpanel import PlotPanel
import genx.model as modellib
import wx.grid as gridlib

import numpy as np
import sys, os, re, time, io, traceback

from .Reflectivity import SamplePlotPanel, find_code_segment
from .help_modules.custom_dialog import *
from .help_modules import reflectivity_images as images
from .help_modules.materials_db import mdb, Formula, MASS_DENSITY_CONVERSION
from genx.gui_logging import iprint

_avail_models=['spec_nx', 'interdiff', 'xmag', 'mag_refl', 'soft_nx',
               'spec_inhom', 'spec_adaptive']
_set_func_prefix='set'

class SampleGrid(gridlib.Grid):
    def __init__(self, parent, *args, **kw):
        gridlib.Grid.__init__(self, parent, *args, **kw)
        self.parent=parent

        self.cb=None
        self.Bind(gridlib.EVT_GRID_CELL_LEFT_CLICK, self.onCellSelected)
        self.Bind(gridlib.EVT_GRID_EDITOR_CREATED, self.onEditorCreated)
        self.Bind(gridlib.EVT_GRID_EDITOR_SHOWN, self.onEditorShown)
        self.Bind(gridlib.EVT_GRID_EDITOR_HIDDEN, self.onEditorHidden)
        self._activated_ctrl=False
        
    def onCellSelected(self, evt):
        if evt.Col in [1,3,5,7,9]:
            self._activated_ctrl=True
            wx.CallAfter(self.EnableCellEditControl)
        evt.Skip()

    def onEditorCreated(self, evt):
        if evt.Col in [3, 5, 7, 9] and self._activated_ctrl:
            self.cb=evt.Control
            self.cb.WindowStyle|=wx.WANTS_CHARS
            wx.CallLater(100, self.toggleCheckbox)
            self._activated_ctrl=False
        if evt.Col==2 and self.GetTable().GetValue(evt.Row, 1)=='Formula':
            # Show tooltip on formula entry to give feedback on input
            inp=evt.Control
            inp.Bind(wx.EVT_TEXT, self.onFormula)
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
        if evt.Col in [3, 5, 7, 9] and self.cb is not None and self._activated_ctrl:
            wx.CallLater(100, self.toggleCheckbox)
            self._activated_ctrl=False
        if evt.Col==2 and self.GetTable().GetValue(evt.Row, 1)=='Formula':
            # Show tooltip on formula entry to give feedback on input
            self.info_text.Show()
            self.info_text.SetLabel('Enter Chemical Formula:')
            self.parent.Layout()
        evt.Skip()

    def toggleCheckbox(self):
        self.cb.SetValue(not self.cb.IsChecked())
        wx.CallAfter(self.DisableCellEditControl)
    
    def onFormula(self, evt):
        txt=evt.GetString()
        try:
            frm=Formula.from_str(txt)
        except Exception as e:
            self.info_text.SetLabel('Error in Formula:\n'+str(e))
        else:
            txt='Analyzed Formula:\n'+frm.describe()
            if frm in mdb:
                txt+='\n\nFound in DB:\n%g g/cm³'%mdb.dens_mass(frm)
            self.info_text.SetLabel(txt)

# new model is ready with a script as value.
(update_model_event, EVT_UPDATE_MODEL) = wx.lib.newevent.NewEvent()
TOP_LAYER=0
ML_LAYER=1
BOT_LAYER=2

class SampleTable(gridlib.GridTableBase):
    _columns=[
        ('Layer', gridlib.GRID_VALUE_STRING),
        ('Formula Params:\nMixure Params:', gridlib.GRID_VALUE_CHOICE+':Formula,Mixure'),
        ('Chem. Formula\nSLD-1 [10⁻⁶Å⁻²]', gridlib.GRID_VALUE_STRING),
        ('', gridlib.GRID_VALUE_BOOL, False),
        ('Density [g/cm³]\nSLD-2 [10⁻⁶Å⁻²]', gridlib.GRID_VALUE_STRING),
        ('', gridlib.GRID_VALUE_BOOL, False),
        ('Moment [µB/FU]\nFraction [%]', gridlib.GRID_VALUE_STRING),
        ('', gridlib.GRID_VALUE_BOOL, True),
        ('d [Å]', gridlib.GRID_VALUE_STRING),
        ('', gridlib.GRID_VALUE_BOOL, False),
        ('σ [Å]', gridlib.GRID_VALUE_STRING),
        ]

    defaults={
        'Formula': ['Layer', 'Formula', Formula([]),
                    False, '2.0', False, '0.0',
                    True, '10.0', False, '5.0', ML_LAYER],
        'Mixure':  ['MixLayer', 'Mixure', '6.0e-6',
                    False, '2.0e-6', False, '100',
                    True, '10.0', False, '5.0', ML_LAYER],
        }
    
    def __init__(self, parent, grid):
        gridlib.GridTableBase.__init__(self)
        self.parent=parent
        self.grid=grid
        
        self.ambient=[None, 'Formula', 'SLD',
                      False, '0.0', False, '0.0',
                      False, '0', False, '0']
        self.substrate=[None, 'Formula', Formula([['Si',1.0]]),
                        False, '2.32998', False, '0.0',
                        False, '0', True, '5.0']
        self.layers=[['Surface_Oxide', 'Formula', Formula([['Fe',2.0],['O', 2.0]]),
                      False, '5.25568', False, '0.0',
                      True, '20.0', False, '5.0', TOP_LAYER],
                     ['Iron', 'Formula', Formula([['Fe', 1.0]]),
                      False, '7.87422', False, '3.0',
                      True, '100.0', False, '5.0', ML_LAYER],
                     ['Natural_Oxide', 'Formula', Formula([['Si', 1.0], ['O', 2.0]]),
                      False, '4.87479', False, '0.0',
                      True, '20.0', False, '5.0', BOT_LAYER]
                     ]
        
        self.grid.SetTable(self, True)

        self.grid.SetRowLabelSize(40)
        self.grid.SetColLabelSize(60)
        for i, colinfo in enumerate(self._columns):
            # self.parent.SetColSize(i, 50)
            self.grid.AutoSizeColumn(i, True)
        
        wx.CallAfter(self.updateModel)

    def GetNumberRows(self):
        return len(self.layers)+2

    def GetNumberCols(self):
        return len(self._columns)

    def GetRowLabelValue(self, row):
        if row in [0, self.GetNumberRows()-1]:
            return '-'
        else:
            return '% 2i'%row

    def IsEmptyCell(self, row, col):
        try:
            return not self.GetValue(row, col)
        except IndexError:
            return True

    def GetValue(self, row, col):
        if col==0:
            if row==0:
                return 'Ambient'
            elif row==(self.GetNumberRows()-1):
                return 'Substrate'
            else:
                return self.layers[row-1][col].replace('_', ' ')
        if row==0:
            return self.ambient[col]
        elif row==self.GetNumberRows()-1:
            return self.substrate[col]
        
        return self.layers[row-1][col]

    def get_valid_name(self, name):
        # generate a valid identifier string from name
        identifyier=''
        for char in name.replace(' ', '_'):
            if (identifyier+char).isidentifier():
                identifyier+=char
        if identifyier in self.invalid_identifiers:
            identifyier='_'+identifyier
                
        existing=[li[0] for li in self.layers]
        if not identifyier in existing:
            return identifyier
        if identifyier.split('_')[-1].isdigit():
            identifyier=identifyier.rsplit('_',1)[0]
        i=1
        while '%s_%i'%(identifyier, i) in existing:
            i+=1
        return '%s_%i'%(identifyier, i)

    def SetValue(self, row, col, value):
        # ignore unchanged values
        if value==self.GetValue(row,col):
            return
        
        if row==0:
            to_edit=self.ambient
        elif row==(self.GetNumberRows()-1):
            to_edit=self.substrate
        else:
            to_edit=self.layers[row-1]
        if col==0:
            # name change
            to_edit[0]='AboutToChangeValue'
            to_edit[0]=self.get_valid_name(value)
        elif col==2:
            # check formula
            if to_edit[1]=='Formula':
                if value=='SLD':
                    to_edit[2]=value
                else:
                    try:
                        formula=Formula.from_str(value)
                    except:
                        pass
                    else:
                        to_edit[2]=formula
                        # a new formula was set, if in DB, set its density
                        if formula in mdb:
                            to_edit[4]='%g'%mdb.dens_mass(formula)
            else:
                try:
                    val=float(eval('%s'%value))
                    if val>=0 and val<=100:
                        to_edit[2]=value
                except:
                    pass
                else:
                    to_edit[col]=value
        elif col==1:
            # change of layer type resets material data columns
            to_edit[1]=value
            for i in [2,4,6]:
                to_edit[i]=self.defaults[value][i]
        elif col in [3, 5, 7, 9]:
            # boolean columns are always correct
            to_edit[col]=value
        elif col in [4,6,8,10]:
            # evaluate float values, can be written as formla
            try:
                float(eval('%s'%value))
            except:
                pass
            else:
                to_edit[col]=value
        self.updateModel()
    
    def updateModel(self, evt=None):
        model_code=self.getModelCode()
        evt=update_model_event()
        evt.script=model_code
        wx.PostEvent(self.parent, evt)

    def GetAttr(self, row, col, kind):
        '''Called by the grid to find the attributes of the cell,
        bkg color, text colour, font and so on.
        '''
        attr = gridlib.GridCellAttr()
        attr.SetAlignment(wx.ALIGN_LEFT, wx.ALIGN_CENTER)
        if row in [0, (self.GetRowsCount()-1)]:
            if row==0:
                if col==1:
                    attr.SetAlignment(wx.ALIGN_CENTER, wx.ALIGN_TOP)
                elif col in [3, 5, 7, 9]:
                    attr.SetAlignment(wx.ALIGN_RIGHT, wx.ALIGN_TOP)
                else:
                    attr.SetAlignment(wx.ALIGN_LEFT, wx.ALIGN_TOP)
                attr.SetBackgroundColour('#dddddd')
                if col in [9, 10]:
                    attr.SetReadOnly()
            else:
                if col==1:
                    attr.SetAlignment(wx.ALIGN_CENTER, wx.ALIGN_BOTTOM)
                elif col in [3, 5, 7, 9]:
                    attr.SetAlignment(wx.ALIGN_RIGHT, wx.ALIGN_BOTTOM)
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
            if col in [3,5,7,9]:
                attr.SetAlignment(wx.ALIGN_RIGHT, wx.ALIGN_CENTER)
            if self.layers[row-1][11]==TOP_LAYER:
                attr.SetBackgroundColour('#ccffcc')
            elif self.layers[row-1][11]==BOT_LAYER:
                attr.SetBackgroundColour('#ffaaff')
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
        col_type=self._columns[col][1].split(':')[0]
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
        if row==(self.GetNumberRows()-1):
            layer_type=self.substrate[1]
            layer_stack=BOT_LAYER
            row-=1
        elif row>0:
            layer_type=self.layers[row-1][1]
            layer_stack=self.layers[row-1][11]
        else:
            layer_type=self.ambient[1]
            layer_stack=TOP_LAYER
        newlayer=list(self.defaults[layer_type])
        newlayer[11]=layer_stack
        newlayer[0]=self.get_valid_name(newlayer[0])
        self.layers.insert(row, newlayer)
    
        msg=gridlib.GridTableMessage(self,
                                     gridlib.GRIDTABLE_NOTIFY_ROWS_APPENDED, 1)
        self.GetView().ProcessTableMessage(msg)
        msg=gridlib.GridTableMessage(self,
                                     gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.updateModel()
        return True

    def DeleteRow(self, row):
        if row in [0, self.GetNumberRows()-1]:
            return False
        # make sure we don't delete the last ML layer
        if self.layers[row-1][11]==ML_LAYER and len([li for li in self.layers if li[11]==ML_LAYER])==1:
            return False
        self.layers.pop(row-1)
    
        msg=gridlib.GridTableMessage(self,
                                     gridlib.GRIDTABLE_NOTIFY_ROWS_INSERTED, 1)
        self.GetView().ProcessTableMessage(msg)
        msg=gridlib.GridTableMessage(self,
                                     gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.updateModel()
        return True

    def MoveRow(self, row_from, row_to):
        if row_from in [0, self.GetNumberRows()-1] or row_to<0\
                or row_to in [0, (self.GetNumberRows()-1)]:
            return False
        if self.layers[row_from-1][11]!=self.layers[row_to-1][11]:
            return False
        moved_row=self.layers.pop(row_from-1)
        self.layers.insert(row_to-1, moved_row)
    
        msg=gridlib.GridTableMessage(self,
                                     gridlib.GRIDTABLE_REQUEST_VIEW_GET_VALUES)
        self.GetView().ProcessTableMessage(msg)
        self.GetView().ForceRefresh()
        self.updateModel()
        return True
    
    def getLayerCode(self, layer):
        output="model.Layer("
        if layer[1]=='Formula':
            formula=layer[2]
            if formula=='SLD':
                nSLD=float(eval(layer[4]))
                mSLD=float(eval(layer[6]))
                output+="f=%s, "%(10*nSLD-10j*mSLD)
                output+="b=%g, "%nSLD
                output+="dens=0.1, magn=%g, "%mSLD
            else:
                output+="f=%s, "%formula.f()
                output+="b=%s, "%formula.b()
                output+="dens=%g, "%(eval(layer[4])*MASS_DENSITY_CONVERSION/formula.mFU())
                output+="magn='%s', "%layer[6]
        else:
            SLD1=float(eval(layer[2]))
            SLD2=float(eval(layer[4]))
            frac=float(eval(layer[6]))/100.
            output+="f=%g, "%(frac*SLD1+(1-frac)*SLD2)
            output+="b=%g, "%(frac*SLD1+(1-frac)*SLD2)
            output+="dens=0.1, magn=0.0, "
        output+="d='%s', "%layer[8]
        output+="sigma='%s', "%layer[10]
        output+="xs_ai=0.0, magn_ang=0.0)"
        return output

    invalid_identifiers=['sample', 'Sim', 'model',
                         'fw', 'fp', 'bc', 'bw',
                         'Amb', 'Sub', 'inst']
    def getModelCode(self):
        '''
        Generate the python code for the current sample structure.
        '''
        script="# BEGIN Sample DO NOT CHANGE\n"
        script+="Amb = %s\n"%self.getLayerCode(self.ambient)
        for layer in self.layers:
            script+="%s = %s\n"%(layer[0], self.getLayerCode(layer))

        script+="\nSub = %s\n"%self.getLayerCode(self.substrate)
        
        top=[li[0] for li in self.layers if li[11]==TOP_LAYER]
        ml=[li[0] for li in self.layers if li[11]==ML_LAYER]
        bot=[li[0] for li in self.layers if li[11]==BOT_LAYER]
        script+="\nTop = model.Stack(Layers=[%s ], Repetitions = 1)\n"%str(
            ", ".join(reversed(top))
            )
        script+="\nML = model.Stack(Layers=[%s ], Repetitions = %%i)\n"%str(
            ", ".join(reversed(ml))
            )
        script+="\nBot = model.Stack(Layers=[%s ], Repetitions = 1)\n"%str(
            ", ".join(reversed(bot))
            )
        script+="\nsample = model.Sample(Stacks = [Bot, ML, Top], Ambient = Amb, Substrate = Sub)\n" \
                "# END Sample\n\n" \
                "# BEGIN Parameters DO NOT CHANGE\n"
        return script


class SamplePanel(wx.Panel):
    last_sample_script=''
    
    inst_params=dict(probe='neutron pol',
                     I0=1.0,
                     res=0.0001,
                     wl=1.54,
                     pol='uu',
                     Ibkg=0.,
                     samplelen=10.,
                     beamw=0.1,
                     footype='no corr',
                     name='inst',
                     coords='q')
    
    def __init__(self, parent, plugin, refindexlist=[]):
        wx.Panel.__init__(self, parent)
        self.refindexlist=refindexlist
        self.plugin=plugin
        self.variable_span=0.25
        
        # Colours indicating different states
        # Green wx.Colour(138, 226, 52), ORANGE wx.Colour(245, 121, 0)
        self.fit_colour=(245, 121, 0)
        # Tango Sky blue wx.Colour(52, 101, 164), wx.Colour(114, 159, 207)
        self.const_fit_colour=(114, 159, 207)
        
        boxver=wx.BoxSizer(wx.HORIZONTAL)
        boxhor=wx.BoxSizer(wx.VERTICAL)
        self.toolbar=wx.ToolBar(self, style=wx.TB_FLAT|wx.TB_HORIZONTAL)
        boxhor.Add((-1, 2))
        self.do_toolbar()
        boxhor.Add(self.toolbar, proportion=0, flag=wx.EXPAND, border=1)
        boxhor.Add((-1, 2))
        self.grid = SampleGrid(self, -1, style=wx.NO_BORDER)
        self.sample_table=SampleTable(self, self.grid)


        self.last_sample_script=self.sample_table.getModelCode()
        self.Bind(EVT_UPDATE_MODEL, self.UpdateModel)
        boxhor.Add(self.grid, 1, wx.EXPAND)
        
        boxver.Add(boxhor, 1, wx.EXPAND)
        
        self.grid.info_text=wx.StaticText(self, -1, 'Bier')
        boxver.Add(self.grid.info_text, 0)
        self.grid.info_text.Hide()
        
        self.SetSizer(boxver)
        self.toolbar.Realize()
        self.update_callback=lambda event: ''
    
    def do_toolbar(self):
        dpi_scale_factor=wx.GetDisplayPPI()[0]/96.
        tb_bmp_size=int(dpi_scale_factor*20)

        newid=wx.NewId()
        self.toolbar.AddTool(newid, 'Insert Layer',
                             bitmap=wx.Bitmap(images.insert_layer.GetImage().Scale(tb_bmp_size,tb_bmp_size)),
                             shortHelp='Insert a Layer')
        self.Bind(wx.EVT_TOOL, self.OnLayerAdd, id=newid)
        
        newid=wx.NewId()
        self.toolbar.AddTool(newid, 'Delete', bitmap=wx.Bitmap(images.delete.GetImage().Scale(tb_bmp_size,tb_bmp_size)),
                             shortHelp='Delete item')
        self.Bind(wx.EVT_TOOL, self.OnLayerDelete, id=newid)
        
        newid=wx.NewId()
        self.toolbar.AddTool(newid, 'Move up',
                             bitmap=wx.Bitmap(images.move_up.GetImage().Scale(tb_bmp_size,tb_bmp_size)),
                             shortHelp='Move item up')
        self.Bind(wx.EVT_TOOL, self.MoveUp, id=newid)
        
        newid=wx.NewId()
        self.toolbar.AddTool(newid, 'Move down',
                             bitmap=wx.Bitmap(images.move_down.GetImage().Scale(tb_bmp_size,tb_bmp_size)),
                             shortHelp='Move item down')
        self.Bind(wx.EVT_TOOL, self.MoveDown, id=newid)
        self.toolbar.AddSeparator()

        newid=wx.NewId()
        button=wx.Button(self.toolbar, newid, label='Instrument Settings')
        button.SetBitmap(wx.Bitmap(images.instrument.GetImage().Scale(tb_bmp_size,tb_bmp_size)), dir=wx.LEFT)
        self.toolbar.AddControl(button)
        self.Bind(wx.EVT_BUTTON, self.EditInstrument, id=newid)
        self.toolbar.AddSeparator()
        
        
        self.toolbar.AddStretchableSpace()
        text=wx.StaticText(self.toolbar, -1, 'Repitions:')
        text.SetToolTipString(
            'Number N of repetitions for a multilayer structure.\n'
            'The model structure is build as a set of bottom\n'
            'layers (purple) repeated layer structure (white)\n'
            'and a set of top layers (green):\n'
            'Model=Substrat/[purple]/Nx[white]/[green]/Ambient\n\n'
            'If top or bottom is missind, select Ambient or\n'
            'Substrate when adding a new layer.'
                                          )
        self.toolbar.AddControl(text)

        newid=wx.NewId()
        self.repetitions=wx.SpinCtrl(self.toolbar, newid,
                                     min=1, max=1000, initial=1,)
        self.toolbar.AddControl(self.repetitions)
        self.Bind(wx.EVT_SPINCTRL, self.ChangeRepetitions, id=newid)

    def ChangeRepetitions(self, evt):
        self.UpdateModel()
    
    def instrumentCode(self):
        pars=dict(self.inst_params)
        template="%(name)s = model.Instrument(probe='%(probe)s', wavelength=%(wl)g, " \
                 "coords='%(coords)s', I0=%(I0)g, res=%(res)g, " \
                 "restype='full conv and varying res.', respoints=9, " \
                 "resintrange=2, beamw=%(beamw)g, footype='%(footype)s', " \
                 "samplelen=%(samplelen)g, incangle=0.0, pol='%(pol)s', " \
                 "Ibkg=%(Ibkg)g, tthoff=0.0,)\n"
        if pars['probe'] in ['x-ray', 'neutron']:
            return ['inst'], template%pars
        output=template%pars
        if pars['probe']=='neutron pol':
            pars['pol']='dd'
            pars['name']='inst_down'
            output+=template%pars
            insts=['inst', 'inst_down']
        return insts, output

    def UpdateModel(self, evt=None):
        probe='neutron pol'
        if evt is None:
            sample_script=self.last_sample_script
        else:
            sample_script=evt.script
            self.last_sample_script=sample_script
        script='from numpy import *\n' \
               'import models.spec_nx as model\n' \
               'from models.utils import UserVars, fp, fw, bc, bw\n\n' \
               '# BEGIN Instrument DO NOT CHANGE\n' \
               'from models.utils import create_fp, create_fw\n'
        insts, inst_str=self.instrumentCode()
        script+=inst_str

        script+="inst_fp = create_fp(inst.wavelength); inst_fw = create_fw(inst.wavelength)\n" \
                "fp.set_wavelength(inst.wavelength); fw.set_wavelength(inst.wavelength)\n" \
                "# END Instrument\n\n"
        # add sample description code
        script+=sample_script%self.repetitions.GetValue()
        script+="cp = UserVars()\n" \
                "# END Parameters\n\n" \
                "SLD = []\n" \
                "def Sim(data):\n" \
                "    I = []\n" \
                "    SLD[:] = []\n"
        datasets=self.model.data
        for i, di in enumerate(datasets):
            di.run_command()
            if probe=='x-ray':
                self.plugin.parent.plot_data.update_labels('2θ [°]')
                from genx import data
                data.DataSet.simulation_params[0]=0.01
                data.DataSet.simulation_params[1]=6.01
            else:
                self.plugin.parent.plot_data.update_labels('q [Å$^{-1}$]')
                from genx import data
                data.DataSet.simulation_params[0]=0.001
                data.DataSet.simulation_params[1]=0.601
            script+="    # BEGIN Dataset %i DO NOT CHANGE\n" \
                    "    d = data[%i]\n" \
                    "    I.append(sample.SimSpecular(d.x, %s))\n" \
                    "    if _sim: SLD.append(sample.SimSLD(None, None, inst))\n" \
                    "    # END Dataset %i\n"%(i, i, insts[i%len(insts)], i)

        script+="    return I"
        # print(script)
        self.plugin.SetModelScript(script)
        
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
        self.update_callback=func
    
    def create_html_decorator(self):
        """
        creates a html decorator function
        :return:
        """
        grid_parameters=self.plugin.GetModel().get_parameters()
        dic_lookup={}
        for par in grid_parameters.get_names():
            l=par.split('.')
            if len(l)==2:
                name=l[0]
                par_name=l[1][3:].lower()
                dic_lookup[(name, par_name)]=(
                grid_parameters.get_value_by_name(par),
                grid_parameters.get_fit_state_by_name(par)
                )
        fit_color_str="rgb(%d,%d,%d)"%self.fit_colour
        const_fit_color_str="rgb(%d,%d,%d)"%self.const_fit_colour
        
        def decorator(name, str):
            """ Decorator to indicate the parameters that are fitted"""
            try:
                start_index=str.index('(')+1
            except ValueError:
                start_index=0
            ret_str=str[:start_index]
            for par_str in str[start_index:].split(','):
                par_name=par_str.split('=')[0].strip()
                # par_name normal paramter (real number)
                if (name, par_name) in dic_lookup:
                    val, state=dic_lookup[(name, par_name)]
                    if state==1:
                        par_str=' <font color=%s><b>%s=%.2e</b></font>,'%(
                        fit_color_str, par_name, val)
                    elif state==2:
                        par_str=' <font color=%s><b>%s=%.2e</b></font>,'%(
                        const_fit_color_str, par_name, val)
                # par_name is a complex parameter...
                elif (name, par_name+'real') in dic_lookup or (
                name, par_name+'imag') in dic_lookup:
                    if (name, par_name+'real') in dic_lookup:
                        val, state=dic_lookup[(name, par_name+'real')]
                        if state==1:
                            par_str=' <font color=%s><b>%s=(%.2e,</b></font>'%(
                            fit_color_str, par_name, val)
                        elif state==2:
                            par_str=' <font color=%s><b>%s=(%.2e,</b></font>'%(
                            const_fit_color_str, par_name, val)
                    else:
                        par_str=' <b>%s=??+</b>'%(par_name)
                    if (name, par_name+'imag') in dic_lookup:
                        val, state=dic_lookup[(name, par_name+'imag')]
                        if state==1:
                            par_str+=' <font color=%s><b>%.2e)</b></font>,'%(
                            fit_color_str, val)
                        elif state==2:
                            par_str+=' <font color=%s><b>%.2e)</b></font>,'%(
                            const_fit_color_str, val)
                    else:
                        par_str+=' <b>??)</b>,'
                
                else:
                    par_str+=','
                ret_str+=par_str
            # Remove trailing ,
            if ret_str[-1]==',':
                ret_str=ret_str[:-1]
            if str[-1]==')' and ret_str[-1]!=')':
                ret_str+=')'
            return ret_str
        
        return decorator
    
    def Update(self, update_script=True):
        if update_script:
            self.update_callback(None)
    
    def SetSample(self, sample, names):
        # self.sampleh.sample=sample
        # self.sampleh.names=names
        self.Update()
    
    def EditSampleParameters(self, evt):
        """ Event handler that creates a dialog box to edit the sample parameters.

        :param evt:
        :return: Nothing
        """
        obj_name='sample'
        eval_func=self.plugin.GetModel().eval_in_model
        grid_parameters=self.plugin.GetModel().get_parameters()
        
        validators={}
        vals={}
        pars=[]
        items=[]
        editable={}
        try:
            string_choices=self.model.sample_string_choices
        except Exception as e:
            string_choices={}
        for item in self.model.SampleParameters:
            if item!='Stacks' and item!='Substrate' and item!='Ambient':
                if item in string_choices:
                    validators[item]=string_choices[item]
                else:
                    validators[item]=FloatObjectValidator()
                val=getattr(self.sampleh.sample, item)
                vals[item]=val
                pars.append(item)
                items.append((item, val))
                # Check if the parameter is in the grid and in that case set it as uneditable
                func_name=obj_name+'.'+_set_func_prefix+item.capitalize()
                grid_value=grid_parameters.get_value_by_name(func_name)
                editable[item]=grid_parameters.get_fit_state_by_name(func_name)
                if grid_value is not None:
                    vals[item]=grid_value
        try:
            groups=self.model.SampleGroups
        except Exception:
            groups=False
        try:
            units=self.model.SampleUnits
        except Exception:
            units=False
        
        dlg=ValidateFitDialog(self, pars, vals, validators,
                              title='Sample Editor', groups=groups,
                              units=units, editable_pars=editable)
        
        if dlg.ShowModal()==wx.ID_OK:
            vals=dlg.GetValues()
            # print vals
            states=dlg.GetStates()
            for par in pars:
                if not states[par]:
                    setattr(self.sampleh.sample, par, vals[par])
                if editable[par]!=states[par]:
                    value=eval_func(vals[par])
                    minval=min(value*(1-self.variable_span),
                               value*(1+self.variable_span))
                    maxval=max(value*(1-self.variable_span),
                               value*(1+self.variable_span))
                    func_name=obj_name+'.'+_set_func_prefix+par.capitalize()
                    grid_parameters.set_fit_state_by_name(func_name, value,
                                                          states[par], minval,
                                                          maxval)
                    # Tell the grid to reload the parameters
                    self.plugin.parent.paramter_grid.SetParameters(
                        grid_parameters)
            
            self.Update()
        
        dlg.Destroy()
    
    def SetInstrument(self, instruments):
        '''SetInstrument(self, instrument) --> None

        Sets the instruments should be a dictionary of instruments with the key being the
        name of the instrument
        '''
        self.instruments=instruments
    
    def EditInstrument(self, evt):
        """Event handler that creates an dialog box to edit the instruments.

        :param evt:
        :return: Nothing
        """
        eval_func=self.plugin.GetModel().eval_in_model
        validators={}
        vals={}
        editable={}
        grid_parameters=self.plugin.GetModel().get_parameters()
        for inst_name in self.instruments:
            vals[inst_name]={}
            editable[inst_name]={}
        
        pars=[]
        for item in self.model.InstrumentParameters:
            if item in self.model.instrument_string_choices:
                # validators.append(self.model.instrument_string_choices[item])
                validators[item]=self.model.instrument_string_choices[item]
            else:
                # validators.append(FloatObjectValidator())
                validators[item]=FloatObjectValidator()
            for inst_name in self.instruments:
                val=getattr(self.instruments[inst_name], item)
                vals[inst_name][item]=val
                # Check if the parameter is in the grid and in that case set it as uneditable
                func_name=inst_name+'.'+_set_func_prefix+item.capitalize()
                grid_value=grid_parameters.get_value_by_name(func_name)
                editable[inst_name][
                    item]=grid_parameters.get_fit_state_by_name(func_name)
                if grid_value is not None:
                    vals[inst_name][item]=grid_value
            pars.append(item)
        
        old_insts=[]
        for inst_name in self.instruments:
            old_insts.append(inst_name)
        
        try:
            groups=self.model.InstrumentGroups
        except Exception:
            groups=False
        try:
            units=self.model.InstrumentUnits
        except Exception:
            units=False
        dlg=ValidateFitNotebookDialog(self, pars, vals, validators,
                                      title='Instrument Editor', groups=groups,
                                      units=units, fixed_pages=['inst'],
                                      editable_pars=editable)
        
        if dlg.ShowModal()==wx.ID_OK:
            old_vals=vals
            vals=dlg.GetValues()
            # print vals
            states=dlg.GetStates()
            self.instruments={}
            for inst_name in vals:
                new_instrument=False
                if inst_name not in self.instruments:
                    # A new instrument must be created:
                    self.instruments[inst_name]=self.model.Instrument()
                    new_instrument=True
                for par in self.model.InstrumentParameters:
                    if not states[inst_name][par]:
                        old_type=type(old_vals[inst_name][par])
                        setattr(self.instruments[inst_name], par,
                                old_type(vals[inst_name][par]))
                    else:
                        setattr(self.instruments[inst_name], par,
                                old_vals[inst_name][par])
                    if new_instrument and states[inst_name][par]>0:
                        value=eval_func(vals[inst_name][par])
                        minval=min(value*(1-self.variable_span),
                                   value*(1+self.variable_span))
                        maxval=max(value*(1-self.variable_span),
                                   value*(1+self.variable_span))
                        func_name=inst_name+'.'+_set_func_prefix+par.capitalize()
                        grid_parameters.set_fit_state_by_name(func_name, value,
                                                              states[
                                                                  inst_name][
                                                                  par], minval,
                                                              maxval)
                    elif not new_instrument:
                        if editable[inst_name][par]!=states[inst_name][par]:
                            value=eval_func(vals[inst_name][par])
                            minval=min(value*(1-self.variable_span),
                                       value*(1+self.variable_span))
                            maxval=max(value*(1-self.variable_span),
                                       value*(1+self.variable_span))
                            func_name=inst_name+'.'+_set_func_prefix+par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name,
                                                                  value,
                                                                  states[
                                                                      inst_name][
                                                                      par],
                                                                  minval,
                                                                  maxval)
            
            # Loop to remove instrument from grid if not returned from Dialog
            for inst_name in old_insts:
                if inst_name not in list(vals.keys()):
                    for par in self.model.InstrumentParameters:
                        if editable[inst_name][par]>0:
                            func_name=inst_name+'.'+_set_func_prefix+par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name, 0,
                                                                  0, 0, 0)
            
            # Tell the grid to reload the parameters
            self.plugin.parent.paramter_grid.SetParameters(grid_parameters)
            
            for change in dlg.GetChanges():
                if change[0]!='' and change[1]!='':
                    self.plugin.InstrumentNameChange(change[0], change[1])
                elif change[1]=='':
                    self.plugin.InstrumentNameChange(change[0], 'inst')
            
            self.Update()
        else:
            pass
        dlg.Destroy()
    
    def MoveUp(self, evt):
        row = self.grid.GetGridCursorRow()
        res=self.sample_table.MoveRow(row, row-1)
        if res:
            self.grid.MoveCursorUp(False)

    def MoveDown(self, evt):
        row = self.grid.GetGridCursorRow()
        res=self.sample_table.MoveRow(row, row+1)
        if res:
            self.grid.MoveCursorDown(False)
    
    def InsertStack(self, evt):
        # Create Dialog box
        items=[('Name', 'name')]
        validators={}
        vals={}
        validators['Name']=NoMatchValidTextObjectValidator(self.sampleh.names)
        pars=['Name']
        vals['Name']='name'
        dlg=ValidateDialog(self, pars, vals, validators,
                           title='Give Stack Name')
        
        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
            vals=dlg.GetValues()
        dlg.Destroy()
        # if not a value is selected operate on first
        pos=max(self.listbox.GetSelection(), 0)
        sl=self.sampleh.insertItem(pos, 'Stack', vals['Name'])
        if sl:
            self.Update()
        else:
            self.plugin.ShowWarningDialog('Can not insert a stack at the'
                                          ' current position.')
    
    def InsertLay(self, evt):
        # Create Dialog box
        # items = [('Name', 'name')]
        # validators = [NoMatchValidTextObjectValidator(self.sampleh.names)]
        dlg=ValidateDialog(self, ['Name'], {'Name': 'name'},
                           {
                               'Name': NoMatchValidTextObjectValidator(
                                   self.sampleh.names)
                               },
                           title='Give Layer Name')
        # Show the dialog
        if dlg.ShowModal()==wx.ID_OK:
            vals=dlg.GetValues()
        dlg.Destroy()
        # if not a value is selected operate on first
        pos=max(self.listbox.GetSelection(), 0)
        # Create the Layer
        sl=self.sampleh.insertItem(pos, 'Layer', vals['Name'])
        if sl:
            self.Update()
        else:
            self.plugin.ShowWarningDialog('Can not insert a layer at the'
                                          ' current position. Layers has to be part of a stack.')
    
    def DeleteSample(self, evt):
        slold=self.sampleh.getStringList()
        sl=self.sampleh.deleteItem(self.listbox.GetSelection())
        if sl:
            self.Update()
    
    def ChangeName(self, evt):
        '''Change the name of the current selected item.
        '''
        pos=self.listbox.GetSelection()
        if pos==0 or pos==len(self.sampleh.names)-1:
            self.plugin.ShowInfoDialog('It is forbidden to change the' \
                                       'name of the substrate (Sub) and the Ambient (Amb) layers.')
        else:
            unallowed_names=self.sampleh.names[:pos]+ \
                            self.sampleh.names[max(0, pos-1):]
            dlg=ValidateDialog(self, ['Name'],
                               {'Name': self.sampleh.names[pos]},
                               {
                                   'Name': NoMatchValidTextObjectValidator(
                                       unallowed_names)
                                   },
                               title='Give New Name')
            
            if dlg.ShowModal()==wx.ID_OK:
                vals=dlg.GetValues()
                result=self.sampleh.changeName(pos, vals['Name'])
                if result:
                    self.Update()
                else:
                    iprint('Unexpected problems when changing name...')
            dlg.Destroy()
    
    def lbDoubleClick(self, evt):
        sel=self.sampleh.getItem(self.listbox.GetSelection())
        obj_name=self.sampleh.getName(self.listbox.GetSelection())
        eval_func=self.plugin.GetModel().eval_in_model
        sl=None
        items=[]
        validators={}
        vals={}
        pars=[]
        editable={}
        grid_parameters=self.plugin.GetModel().get_parameters()
        if isinstance(sel, self.model.Layer):
            # The selected item is a Layer
            for item in list(self.model.LayerParameters.keys()):
                value=getattr(sel, item)
                vals[item]=value
                # if item!='n' and item!='fb':
                if type(self.model.LayerParameters[item])!=type(1+1.0J):
                    # Handle real parameters
                    validators[item]=FloatObjectValidator(eval_func,
                                                          alt_types=[
                                                              self.model.Layer])
                    func_name=obj_name+'.'+_set_func_prefix+item.capitalize()
                    grid_value=grid_parameters.get_value_by_name(func_name)
                    if grid_value is not None:
                        vals[item]=grid_value
                    editable[item]=grid_parameters.get_fit_state_by_name(
                        func_name)
                
                else:
                    # Handle complex parameters
                    validators[item]=ComplexObjectValidator(eval_func,
                                                            alt_types=[
                                                                self.model.Layer])
                    func_name=obj_name+'.'+_set_func_prefix+item.capitalize()
                    grid_value_real=grid_parameters.get_value_by_name(
                        func_name+'real')
                    grid_value_imag=grid_parameters.get_value_by_name(
                        func_name+'imag')
                    if grid_value_real is not None:
                        v=eval_func(vals[item]) if type(vals[item]) is str else \
                        vals[item]
                        vals[item]=grid_value_real+v.imag*1.0J
                    if grid_value_imag is not None:
                        v=eval_func(vals[item]) if type(vals[item]) is str else \
                        vals[item]
                        vals[item]=v.real+grid_value_imag*1.0J
                    editable[item]=max(grid_parameters.get_fit_state_by_name(
                        func_name+'real'),
                                       grid_parameters.get_fit_state_by_name(
                                           func_name+'imag'))
                
                items.append((item, value))
                pars.append(item)
                
                # Check if the parameter is in the grid and in that case set it as uneditable
                # func_name = obj_name + '.' + _set_func_prefix + item.capitalize()
                # grid_value = grid_parameters.get_value_by_name(func_name)
                # editable[item] = grid_parameters.get_fit_state_by_name(func_name)
            
            try:
                groups=self.model.LayerGroups
            except Exception:
                groups=False
            try:
                units=self.model.LayerUnits
            except Exception:
                units=False
            
            dlg=ValidateFitDialog(self, pars, vals, validators,
                                  title='Layer Editor', groups=groups,
                                  units=units, editable_pars=editable)
            
            if dlg.ShowModal()==wx.ID_OK:
                vals=dlg.GetValues()
                states=dlg.GetStates()
                for par in list(self.model.LayerParameters.keys()):
                    if not states[par]:
                        setattr(sel, par, vals[par])
                    if editable[par]!=states[par]:
                        value=eval_func(vals[par])
                        
                        if type(self.model.LayerParameters[par]) is complex:
                            # print type(value)
                            func_name=obj_name+'.'+_set_func_prefix+par.capitalize()+'real'
                            val=value.real
                            minval=min(val*(1-self.variable_span),
                                       val*(1+self.variable_span))
                            maxval=max(val*(1-self.variable_span),
                                       val*(1+self.variable_span))
                            grid_parameters.set_fit_state_by_name(func_name,
                                                                  val,
                                                                  states[par],
                                                                  minval,
                                                                  maxval)
                            val=value.imag
                            minval=min(val*(1-self.variable_span),
                                       val*(1+self.variable_span))
                            maxval=max(val*(1-self.variable_span),
                                       val*(1+self.variable_span))
                            func_name=obj_name+'.'+_set_func_prefix+par.capitalize()+'imag'
                            grid_parameters.set_fit_state_by_name(func_name,
                                                                  val,
                                                                  states[par],
                                                                  minval,
                                                                  maxval)
                        else:
                            val=value
                            minval=min(val*(1-self.variable_span),
                                       val*(1+self.variable_span))
                            maxval=max(val*(1-self.variable_span),
                                       val*(1+self.variable_span))
                            func_name=obj_name+'.'+_set_func_prefix+par.capitalize()
                            grid_parameters.set_fit_state_by_name(func_name,
                                                                  value,
                                                                  states[par],
                                                                  minval,
                                                                  maxval)
                        
                        # Does not seem to be necessary
                        self.plugin.parent.paramter_grid.SetParameters(
                            grid_parameters)
                sl=self.sampleh.getStringList()
            dlg.Destroy()
        
        else:
            # The selected item is a Stack
            for item in list(self.model.StackParameters.keys()):
                if item!='Layers':
                    value=getattr(sel, item)
                    if isinstance(value, float):
                        validators[item]=FloatObjectValidator(eval_func,
                                                              alt_types=[
                                                                  self.model.Stack])
                    else:
                        validators[item]=TextObjectValidator()
                    items.append((item, value))
                    pars.append(item)
                    vals[item]=value
                    
                    # Check if the parameter is in the grid and in that case set it as uneditable
                    func_name=obj_name+'.'+_set_func_prefix+item.capitalize()
                    grid_value=grid_parameters.get_value_by_name(func_name)
                    editable[item]=grid_parameters.get_fit_state_by_name(
                        func_name)
                    if grid_value is not None:
                        vals[item]=grid_value
            
            try:
                groups=self.model.StackGroups
            except Exception:
                groups=False
            try:
                units=self.model.StackUnits
            except Exception:
                units=False
            
            dlg=ValidateFitDialog(self, pars, vals, validators,
                                  title='Layer Editor', groups=groups,
                                  units=units, editable_pars=editable)
            if dlg.ShowModal()==wx.ID_OK:
                vals=dlg.GetValues()
                states=dlg.GetStates()
                for par in pars:
                    if not states[par]:
                        setattr(sel, par, vals[par])
                    if editable[par]!=states[par]:
                        value=eval_func(vals[par])
                        minval=min(value*(1-self.variable_span),
                                   value*(1+self.variable_span))
                        maxval=max(value*(1-self.variable_span),
                                   value*(1+self.variable_span))
                        func_name=obj_name+'.'+_set_func_prefix+par.capitalize()
                        grid_parameters.set_fit_state_by_name(func_name, value,
                                                              states[par],
                                                              minval, maxval)
                        # Does not seem to be necessary
                        self.plugin.parent.paramter_grid.SetParameters(
                            grid_parameters)
                sl=self.sampleh.getStringList()
            
            dlg.Destroy()
        
        if sl:
            self.Update()


class Plugin(framework.Template):
    previous_xaxis=None
    sim_returns_sld=True
    
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        # self.parent = parent
        self.model_obj=self.GetModel()
        sample_panel=self.NewInputFolder('Model')
        sample_sizer=wx.BoxSizer(wx.HORIZONTAL)
        sample_panel.SetSizer(sample_sizer)
        self.defs=['Instrument', 'Sample']
        self.sample_widget=SamplePanel(sample_panel, self)
        self.sample_widget.model=self.model_obj
        sample_sizer.Add(self.sample_widget, 1, wx.EXPAND|wx.GROW|wx.ALL)
        sample_panel.Layout()
        
        self.sample_widget.SetUpdateCallback(self.UpdateScript)
        
        # Create the SLD plot
        sld_plot_panel=self.NewPlotFolder('SLD')
        sld_sizer=wx.BoxSizer(wx.HORIZONTAL)
        sld_plot_panel.SetSizer(sld_sizer)
        self.sld_plot=SamplePlotPanel(sld_plot_panel, self)
        sld_sizer.Add(self.sld_plot, 1, wx.EXPAND|wx.GROW|wx.ALL)
        sld_plot_panel.Layout()
        
        self.sample_widget.UpdateModel()
        # if self.model_obj.script!='':
        #     if self.model_obj.filename!='':
        #         iprint("Reflectivity plugin: Reading loaded model")
        #         self.ReadModel()
        #     else:
        #         try:
        #             self.ReadModel()
        #         except:
        #             iprint("Reflectivity plugin: Creating new model")
        #             self.CreateNewModel()
        # else:
        #     iprint("Reflectivity plugin: Creating new model")
        #     self.CreateNewModel()
        
        # Create a menu for handling the plugin
        menu=self.NewMenu('Reflec')
        self.mb_export_sld=wx.MenuItem(menu, wx.NewId(),
                                       "Export SLD...",
                                       "Export the SLD to a ASCII file",
                                       wx.ITEM_NORMAL)
        menu.Append(self.mb_export_sld)
        self.mb_show_imag_sld=wx.MenuItem(menu, wx.NewId(),
                                          "Show Im SLD",
                                          "Toggles showing the imaginary part of the SLD",
                                          wx.ITEM_CHECK)
        menu.Append(self.mb_show_imag_sld)
        self.mb_show_imag_sld.Check(False)
        self.show_imag_sld=self.mb_show_imag_sld.IsChecked()
        self.mb_autoupdate_sld=wx.MenuItem(menu, wx.NewId(),
                                           "Autoupdate SLD",
                                           "Toggles autoupdating the SLD during fitting",
                                           wx.ITEM_CHECK)
        menu.Append(self.mb_autoupdate_sld)
        self.mb_autoupdate_sld.Check(False)
        
        self.mb_hide_advanced=wx.MenuItem(menu, wx.NewId(),
                                           "Hide Advanced",
                                           "Toggles hiding of advanced model tabs",
                                           wx.ITEM_CHECK)
        menu.Append(self.mb_hide_advanced)
        self.mb_hide_advanced.Check(True)
        # self.mb_autoupdate_sld.SetCheckable(True)
        self.parent.Bind(wx.EVT_MENU, self.OnExportSLD, self.mb_export_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnAutoUpdateSLD,
                         self.mb_autoupdate_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnShowImagSLD,
                         self.mb_show_imag_sld)
        self.parent.Bind(wx.EVT_MENU, self.OnHideAdvanced,
                         self.mb_hide_advanced)

        self.HideUIElements()
        self.StatusMessage('Simple Reflectivity plugin loaded')

    def OnHideAdvanced(self, evt):
        if self.mb_hide_advanced.IsChecked():
            self.HideUIElements()
        else:
            self.ShowUIElements()

    def HideUIElements(self):
        self._hidden_pages=[]
        # hide all standard tabs
        nb=self.parent.input_notebook
        for i, page_i in reversed(list(enumerate(nb.Children))):
            title=nb.GetPageText(i)
            if title!='Model':
                self._hidden_pages.append([title, page_i, i])
                nb.RemovePage(i)
        self._hidden_pages.reverse()

    def ShowUIElements(self):
        nb=self.parent.input_notebook
        for title, page_i, i in self._hidden_pages:
            nb.InsertPage(i, page_i, title)
        self._hidden_pages=None

    def Remove(self):
        self.ShowUIElements()
        framework.Template.Remove(self)

    
    def UpdateScript(self, event):
        self.WriteModel()
    
    def OnAutoUpdateSLD(self, evt):
        # self.mb_autoupdate_sld.Check(not self.mb_autoupdate_sld.IsChecked())
        pass
    
    def OnShowImagSLD(self, evt):
        self.show_imag_sld=self.mb_show_imag_sld.IsChecked()
        self.sld_plot.Plot()
    
    def OnExportSLD(self, evt):
        dlg=wx.FileDialog(self.parent, message="Export SLD to ...",
                          defaultFile="",
                          wildcard="Dat File (*.dat)|*.dat",
                          style=wx.FD_SAVE|wx.FD_CHANGE_DIR
                          )
        if dlg.ShowModal()==wx.ID_OK:
            fname=dlg.GetPath()
            result=True
            if os.path.exists(fname):
                filepath, filename=os.path.split(fname)
                result=self.ShowQuestionDialog('The file %s already exists.'
                                               ' Do'
                                               ' you wish to overwrite it?'
                                               %filename)
            if result:
                try:
                    self.sld_plot.SavePlotData(fname)
                except IOError as e:
                    self.ShowErrorDialog(e.__str__())
                except Exception as e:
                    outp=io.StringIO()
                    traceback.print_exc(200, outp)
                    val=outp.getvalue()
                    outp.close()
                    self.ShowErrorDialog('Could not save the file.'
                                         ' Python Error:\n%s'%(val,))
        dlg.Destroy()
    
    def OnNewModel(self, event):
        ''' Create a new model
        '''
        dlg=wx.SingleChoiceDialog(self.parent, 'Choose a model type to use', \
                                  'Models', _avail_models,
                                  wx.CHOICEDLG_STYLE
                                  )
        
        if dlg.ShowModal()==wx.ID_OK:
            self.CreateNewModel('models.%s'%dlg.GetStringSelection())
        dlg.Destroy()
    
    def OnDataChanged(self, event):
        ''' Take into account changes in data..
        '''
        self.sample_widget.UpdateModel()
    
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

        Updates stuff during fitting
        '''
        # Calculate and update the sld plot
        if self.mb_autoupdate_sld.IsChecked():
            self.sld_plot.Plot()
        # self.sample_widget.Update(update_script=False)
    
    def OnGridChange(self, event):
        """ Updates the simualtion panel when the grid changes

        :param event:
        :return:
        """
        self.sample_widget.Update(update_script=False)
    
    def CreateNewModel(self, modelname='models.spec_nx'):
        '''Init the script in the model to yield the
        correct script for initilization
        '''
        pass
    
    def WriteModel(self):
        return
    
    
    def ReadModel(self):
        '''ReadModel(self)  --> None

        Reads in the current model and locates layers and stacks
        and sample defined inside BEGIN Sample section.
        '''
        pass