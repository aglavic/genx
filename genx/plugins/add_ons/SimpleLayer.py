#-*- coding: utf8 -*-
'''
===========
SimpleLayer
===========

A plugin to allow a quick and simple creation of layers from materials created
from a crystallographic data file (.cif) available at e.g. ICSD
or the chemical formula and mass-density or crystal structure.
Materials are stored for later use.

To use the materials select any layer in the Sample tab and one material in the
Materials tab and than click the blue arrow button. If you select a Stack instead
a new layer will be created with the given material. The plugin will insert
the x-ray and neutron values ( *f* , *b* ) as well as the density and rename the layer
with the given formula.

Written by Artur Glavic
Last Changes 10/11/16
'''

import os, sys
import json
import wx
from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin
from math import cos, pi, sqrt
from models.utils import UserVars, fp, fw, bc, bw #@UnusedImport
import images as img
from plugins import add_on_framework as framework

# configuration file to store the known materials
try:
  import appdirs
except ImportError:
  config_path=os.path.expanduser(os.path.join('~', '.genx'))
else:
  config_path=appdirs.user_data_dir('GenX', 'MattsBjorck')
if not os.path.exists(config_path):
    os.makedirs(config_path)
config_file=os.path.join(config_path, 'materials.cfg')

mg=None

class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent
        # on the right side, add a list of materials with their density to selct from
        materials_panel = self.NewDataFolder('Materials')
        materials_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.materials_panel = wx.Panel(materials_panel)
        self.create_materials_list()
        materials_sizer.Add(self.materials_panel, 1, wx.EXPAND | wx.GROW | wx.ALL)
        materials_panel.SetSizer(materials_sizer)
        materials_panel.Layout()
        wx.CallAfter(self._init_refplugin)

    def _init_refplugin(self):
        # connect to the reflectivity plugin for layer creation
        self.refplugin = self.parent.plugin_control.plugin_handler.loaded_plugins['Reflectivity']

    def create_materials_list(self):
        '''
          Create a list of materials and it's graphical representation as
          well as a toolbar.
        '''
        # A list containing chemical formula and atomic density for different
        # materials. Each created material is stored and can be reused.
        if os.path.exists(config_file):
            self.known_materials=json.loads(open(config_file, 'rb').read())
        else:
            self.known_materials=[]
        self.tool_panel=wx.Panel(self.materials_panel)
        self.materials_list=MaterialsList(self.materials_panel, self.known_materials)
        self.sizer_vert=wx.BoxSizer(wx.VERTICAL)
        self.materials_panel.SetSizer(self.sizer_vert)

        self.create_toolbar()

        self.sizer_vert.Add(self.tool_panel, proportion=0, flag=wx.EXPAND, border=5)
        self.sizer_vert.Add((-1, 2))
        self.sizer_vert.Add(self.materials_list, proportion=1, flag=wx.EXPAND , border=5)
        self.tool_panel.SetSizer(self.sizer_hor)

    def create_toolbar(self):
        if os.name=='nt':
            size=(24, 24)
        else:
            size=(-1,-1)
        self.bitmap_button_add = wx.BitmapButton(self.tool_panel, -1, img.getaddBitmap(), size=size, style=wx.NO_BORDER)
        self.bitmap_button_add.SetToolTipString('Add material')
        self.bitmap_button_delete=wx.BitmapButton(self.tool_panel, -1, img.getdeleteBitmap(), size=size,
                                                  style=wx.NO_BORDER)
        self.bitmap_button_delete.SetToolTipString('Delete selected materials')
        self.bitmap_button_apply = wx.BitmapButton(self.tool_panel, -1, img.getmove_downBitmap(), size=size,
                                                   style=wx.NO_BORDER)
        self.bitmap_button_apply.SetToolTipString('New Layer/Apply to Layer')

        space = (2, -1)
        self.sizer_hor = wx.BoxSizer(wx.HORIZONTAL)
        self.sizer_hor.Add(self.bitmap_button_add, proportion=0, border=2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_delete, proportion=0, border=2)
        self.sizer_hor.Add(space)
        self.sizer_hor.Add(self.bitmap_button_apply, proportion=0, border=2)


        self.materials_panel.Bind(wx.EVT_BUTTON, self.material_add, self.bitmap_button_add)
        self.materials_panel.Bind(wx.EVT_BUTTON, self.material_delete, self.bitmap_button_delete)
        self.materials_panel.Bind(wx.EVT_BUTTON, self.material_apply, self.bitmap_button_apply)

    def material_add(self, event):
        dialog=MaterialDialog(self.parent)
        if dialog.ShowModal()==wx.ID_OK:
            self.materials_list.AddItem(dialog.GetResult())
        open(config_file, 'wb').write(json.dumps(self.known_materials))
        dialog.Destroy()

    def material_delete(self, event):
        self.materials_list.DeleteItem()
        open(config_file, 'wb').write(json.dumps(self.known_materials))

    def material_apply(self, event):
        index=self.materials_list.GetFirstSelected()
        formula, density=self.known_materials[index]
        f=self.get_f(formula)
        b=self.get_b(formula)
        layer=self.get_selected_layer()
        if layer:
            layer.f=f
            layer.b=b
            layer.dens=density
        name=u''
        for element, count in formula:
            if count==1:
                name+="%s"%(element)
            elif float(count)==int(count):
                name+="%s%i"%(element, count)
            else:
                name+=("%s%s"%(element, count)).replace('.', '_')
        self.set_layer_name(name)
        try:
          self.refplugin.sample_widget.UpdateListbox()
        except AttributeError:
          self.refplugin.sample_widget.Update()

    def get_selected_layer(self):
        layer_idx=self.refplugin.sample_widget.listbox.GetSelection()
        active_layer=self.refplugin.sampleh.getItem(layer_idx)
        if active_layer.__class__.__name__=="Stack":
                # create a new layer to return
            self.refplugin.sampleh.insertItem(layer_idx, 'Layer', 'WillChange')
            active_layer=self.refplugin.sampleh.getItem(layer_idx+1)
        return active_layer

    def set_layer_name(self, name):
        layer_idx = self.refplugin.sample_widget.listbox.GetSelection()
        if self.refplugin.sampleh.names[layer_idx] in ['Amb', 'Sub']:
            return
        active_layer = self.refplugin.sampleh.getItem(layer_idx)
        if active_layer.__class__.__name__ == "Stack":
            # create a new layer to return
            layer_idx += 1
        tmpname = name
        i = 1
        while tmpname in self.refplugin.sampleh.names:
            tmpname = u'%s_%i'%(name, i)
            i += 1
        self.refplugin.sampleh.names[layer_idx] = tmpname

    def get_f(self, formula):
        return self.get_forms('fp', formula)

    def get_b(self, formula):
        return self.get_forms('bc', formula)

    def get_forms(self, pre, formula):
        output = ''
        for item in formula:
            if item[0] in isotopes:
                if pre == 'bc':
                    name = isotopes[item[0]][0]
                else:
                    name = isotopes[item[0]][1]
            else:
                name = item[0]
            output += u'%s.%s*%g+'%(pre, name, item[1])
        return output[:-1] # remove last +



class MaterialsList(wx.ListCtrl, ListCtrlAutoWidthMixin):
    '''
    The ListCtrl for the materials data.
    '''
    def __init__(self, parent, materials_list):
        wx.ListCtrl.__init__(self, parent, -1, style=wx.LC_REPORT|wx.LC_VIRTUAL|wx.LC_EDIT_LABELS)
        ListCtrlAutoWidthMixin.__init__(self)
        self.materials_list = materials_list
        self.parent = parent
        #if sys.platform.startswith('win'):
        #    font = wx.Font(9, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, face="Lucida Sans Unicode")
        #else:
        #    font = wx.Font(9, wx.FONTFAMILY_ROMAN, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL,
        #                   encoding=wx.FONTENCODING_UNICODE)
        font = self.GetFont()
        font.SetPointSize(9)
        self.SetFont(font)

        # Set list length
        self.SetItemCount(len(materials_list))

        # Set the column headers
        for col, (text, width) in enumerate([
                                             (u"Formula", 80),
                                             (u"SLD-n [10⁻⁶Å⁻²]", 60),
                                             (u"SLD-kα [10⁻⁶Å⁻²]", 60),
                                             (u"Density [FU/Å³]", 60),
                                             (u"Density [g/cm³]", 60)
                                             ]):
            self.InsertColumn(col, text, width=width)

    def OnSelectionChanged(self, evt):
        if not self.toggleshow:
            indices = self._GetSelectedItems()
            indices.sort()
            if not indices == self.show_indices:
                self.data_cont.show_data(indices)
                self._UpdateData('Show data set flag toggled',
                                 data_changed=True)
                # Forces update of list control
                self.SetItemCount(self.data_cont.get_count())
        evt.Skip()

    def OnGetItemText(self, item, col):
        formula = self.materials_list[item][0]
        density = self.materials_list[item][1]
        try:
            density = eval('float(%s)'%density)
        except:
            density = 0.
        if col == 4:
            fu_mass = 0.
            for element, number in formula:
                fu_mass += number*atomic_data[element][2]
            mass_density = density*fu_mass/MaterialDialog.MASS_DENSITY_CONVERSION
            return "%.3f" % mass_density
        elif col == 3:
            return "%.4f" % density
        elif col == 2:
            fw.set_wavelength(1.54)
            elements=''
            for element, count in formula:
                elements += '+fp.%s*%f' % (element, count)
            sld = density*eval(elements)
            return "%.3f" % (sld.real*10.*2.82)
        elif col == 1:
            dens = eval('float(%s)' % density)
            elements = ''
            for element, count in formula:
                elements += '+bc.%s*%f' % (element, count)
            sld = dens*eval(elements)
            return "%.3f" % (sld.real*10.)
        else:
            formula = self.materials_list[item][0]
            output = u''
            for element, count in formula:
                if count == 1:
                    output += element
                else:
                    output += element + self.get_subscript(count)
            return output

    def get_subscript(self, count):
        '''
          Return a subscript unicode string that equals the given number.
        '''
        scount = '%g'%count
        result = u''
        for char in scount:
            if char == '.':
                result += u'﹒'
            else:
                # a subscript digit in unicode
                result += ('\\u208' + char).decode('unicode-escape')
        if '.' in scount:
            return result
        else:
            return result

    def _GetSelectedItems(self):
        ''' _GetSelectedItems(self) --> indices [list of integers]
        Function that yields a list of the currently selected items
        position in the list. In order of selction, i.e. no order.
        '''
        indices = [self.GetFirstSelected()]
        while indices[-1] != -1:
            indices.append(self.GetNextSelected(indices[-1]))

        # Remove the last will be -1
        indices.pop(-1)
        return indices

    def _CheckSelected(self, indices):
        '''_CheckSelected(self, indices) --> bool
        Checks so at least data sets are selcted, otherwise show a dialog box
        and return False
        '''
        # Check so that one dataset is selected
        if len(indices) == 0:
            dlg = wx.MessageDialog(self, 'At least one data set has to be selected', caption='Information',
                                   style=wx.OK | wx.ICON_INFORMATION)
            dlg.ShowModal()
            dlg.Destroy()
            return False
        return True

    def DeleteItem(self):
        index = self.GetFirstSelected()
        item = self.materials_list[index]
        item_formula = ''
        for element, count in item[0]:
            if count == 1:
                item_formula += "%s" % (element)
            elif float(count) == int(count):
                item_formula += "%s%i" % (element, count)
            else:
                item_formula += "%s(%f)" % (element, count)


        # Create the dialog box
        dlg = wx.MessageDialog(self, 'Remove material %s?' % (item_formula),
        caption = 'Remove?', style=wx.YES_NO | wx.ICON_QUESTION)

        # Show the dialog box
        if dlg.ShowModal() == wx.ID_YES:
            self.materials_list.pop(index)
            # Update the list
            self.SetItemCount(len(self.materials_list))

        dlg.Destroy()


    def AddItem(self, item):
        i = 0
        while i < len(self.materials_list) and self.materials_list[i][0] < item[0]:
            i+=1
        self.materials_list.insert(i, item)
        self.SetItemCount(len(self.materials_list))

class MaterialDialog(wx.Dialog):
    """
      Dialog to get material information from chemical formula and atomic density.
      Atomic density can ither be entered manually, by using unit cell parameter,
      by massdensity or by loading a .cif crystallographical file.
    """
    extracted_elements = []
    MASS_DENSITY_CONVERSION = 0.60221415  #g/cm³-> u/Å³ : 1e-24 (1/cm³->1/Å³) * 6.0221415e23 (Na)

    def __init__(self, parent):
        wx.Dialog.__init__(self, parent, name='New Material')
        self._create_entries()

    def _create_entries(self):
        base_layout = wx.BoxSizer(wx.VERTICAL)

        table = wx.GridBagSizer(5, 2)

        self.formula_entry = wx.TextCtrl(self, size=(100, 25))
        self.formula_entry.Bind(wx.EVT_TEXT, self.OnFormulaChanged)
        table.Add(wx.StaticText(self, label="Formula:"), (0, 0), flag=wx.ALIGN_CENTER)
        table.Add(self.formula_entry, (0, 1), span=(1, 2), flag=wx.EXPAND)

        table.Add(wx.StaticText(self, label="Extracted Elements:"), (1, 0),
                  span=(1, 3), flag=wx.ALIGN_CENTER)
        self.formula_display=wx.TextCtrl(self, size=(150, 100),
                                         style=wx.TE_MULTILINE|wx.TE_READONLY)
#        self.formula_display.Enable(False)
        table.Add(self.formula_display, (2, 0), span=(1, 3), flag=wx.EXPAND)

        table.Add(wx.StaticText(self, label="1. Unit Cell Parameters:"), (3, 0),
                  span=(1, 3), flag=wx.ALIGN_CENTER)
        table.Add(wx.StaticText(self, label=u"a [Å]"), (4, 0), flag=wx.ALIGN_CENTER)
        self.a_entry=wx.TextCtrl(self, size=(50, 25))
        table.Add(self.a_entry, (5, 0), flag=wx.EXPAND)
        table.Add(wx.StaticText(self, label=u"b [Å]"), (4, 1), flag=wx.ALIGN_CENTER)
        self.b_entry=wx.TextCtrl(self, size=(50, 25))
        table.Add(self.b_entry, (5, 1), flag=wx.EXPAND)
        table.Add(wx.StaticText(self, label=u"c [Å]"), (4, 2), flag=wx.ALIGN_CENTER)
        self.c_entry=wx.TextCtrl(self, size=(50, 25))
        table.Add(self.c_entry, (5, 2), flag=wx.EXPAND)

        table.Add(wx.StaticText(self, label=u"α"), (6, 0), flag=wx.ALIGN_CENTER)
        self.alpha_entry=wx.TextCtrl(self, size=(50, 25))
        self.alpha_entry.SetValue('90')
        table.Add(self.alpha_entry, (7, 0), flag=wx.EXPAND)
        table.Add(wx.StaticText(self, label=u"β"), (6, 1), flag=wx.ALIGN_CENTER)
        self.beta_entry=wx.TextCtrl(self, size=(50, 25))
        self.beta_entry.SetValue('90')
        table.Add(self.beta_entry, (7, 1), flag=wx.EXPAND)
        table.Add(wx.StaticText(self, label=u"γ"), (6, 2), flag=wx.ALIGN_CENTER)
        self.gamma_entry=wx.TextCtrl(self, size=(50, 25))
        self.gamma_entry.SetValue('90')
        table.Add(self.gamma_entry, (7, 2), flag=wx.EXPAND)

        table.Add(wx.StaticText(self, label=u"FUs"), (8, 0), flag=wx.ALIGN_CENTER)
        self.FUs_entry=wx.TextCtrl(self, size=(50, 25))
        self.FUs_entry.SetValue('1')
        table.Add(self.FUs_entry, (9, 0), flag=wx.ALIGN_CENTER)
        cif_button=wx.Button(self, label="From .cif File...")
        cif_button.Bind(wx.EVT_BUTTON, self.OnLoadCif)
        table.Add(cif_button, (8, 1), span=(2, 2), flag=wx.ALIGN_CENTER)

        if mg is None:
          try:
            global mg, MPRester
            import pymatgen as mg
            from pymatgen.matproj.rest import MPRester
          except ImportError:
            pass
        if mg is None:
          mg_txt=wx.StaticText(self, label="Install PyMatGen for Online Query")
          table.Add(mg_txt, (10, 0), span=(1, 3), flag=wx.ALIGN_CENTER)
        else:
          mg_button=wx.Button(self, label="Query Online (PyMatGen)")
          mg_button.Bind(wx.EVT_BUTTON, self.OnQuery)
          table.Add(mg_button, (10, 0), span=(1, 3), flag=wx.ALIGN_CENTER)

        for entry in [self.a_entry, self.b_entry, self.c_entry,
                      self.alpha_entry, self.beta_entry, self.gamma_entry,
                      self.FUs_entry]:
            entry.Bind(wx.EVT_TEXT, self.OnUnitCellChanged)


        table.Add(wx.StaticText(self, label="2. Physical Parameter:"), (11, 0),
                  span=(1, 3), flag=wx.ALIGN_CENTER)
        self.mass_density=wx.TextCtrl(self, size=(70, 25))
        self.mass_density.Bind(wx.EVT_TEXT, self.OnMassDensityChange)
        table.Add(wx.StaticText(self, label=u"Mass Density [g/cm³]:"), (12, 0))
        table.Add(self.mass_density, (12, 1), span=(1, 1))

        table.Add(wx.StaticText(self, label="Result from 1. or 2.:"), (13, 0), span=(1, 3))
        self.result_density=wx.TextCtrl(self, size=(150, 25))
        table.Add(wx.StaticText(self, label=u"Density [FU/Å³]:"), (14, 0))
        table.Add(self.result_density, (14, 1), span=(1, 2))

        buttons=self.CreateButtonSizer(wx.OK|wx.CANCEL)

        base_layout.Add(table, 1, wx.ALIGN_CENTER|wx.TOP)
        base_layout.Add(buttons, 0, wx.ALIGN_RIGHT)
        self.SetSizerAndFit(base_layout)

    def OnFormulaChanged(self, event):
        text=self.formula_entry.GetValue()
        for ign_char in [" ", "\t", "_", "-"]:
            text=text.replace(ign_char, "")
        if text=="":
            self.extracted_elements=[]
            self.formula_display.SetValue('')
            return
        extracted_elements=[]
        i=0
        mlen=len(text)
        while i<mlen:
            char1=text[i].upper()
            char2=''
            if i<mlen-1:
                char2=text[i+1].lower()
            if char1+char2 in atomic_data:
                element=char1+char2
            elif char1 in atomic_data:
                element=char1
            else:
                i+=1
                continue
            i=i+len(element)
            j=0
            while i+j<mlen and not text[i+j].isalpha():
                j+=1
            count_txt=text[i:i+j]
            i+=j
            if count_txt=='':
                count=1.
            else:
                try:
                    count=float(count_txt)
                except ValueError:
                    continue
            extracted_elements.append([element, count])
        self.extracted_elements=extracted_elements
        output=''
        for element, number in extracted_elements:
            output+="%g x %s\n"%(number, atomic_data[element][0])
        self.formula_display.SetValue(output[:-1])
        self.OnMassDensityChange(None)

    def OnUnitCellChanged(self, event):
        params=[]
        for entry in [self.a_entry, self.b_entry, self.c_entry,
                        self.alpha_entry, self.beta_entry, self.gamma_entry,
                        self.FUs_entry]:
            try:
                params.append(float(entry.GetValue()))
            except ValueError:
                return
        if params[3]==90 and params[4]==90 and params[5]==90:
            if params[0]==params[1] and params[0]==params[2]:
                self.density='%s/(%g**3)'%(params[6], params[0])
            else:
                self.density='%s/(%g*%g*%g)'%(params[6], params[0], params[1], params[2])
        else:
            # calculate general unit cell volume (triclinic formula applicable to all structures)
            alpha=params[3]*pi/180.
            beta=params[4]*pi/180.
            gamma=params[5]*pi/180.
            V=params[0]*params[1]*params[2]
            V*=sqrt(1-cos(alpha)**2-cos(beta)**2-cos(gamma)**2+2*cos(alpha)*cos(beta)*cos(gamma))
            self.density='%s/%g'%(params[6], V)
        self.result_density.SetValue(self.density)

    def OnMassDensityChange(self, event):
        fu_mass=0.
        for element, number in self.extracted_elements:
            fu_mass+=number*atomic_data[element][2]
        try:
            mass_density=float(self.mass_density.GetValue())
        except ValueError:
            return
        self.density="%g*%g/%g"%(mass_density, self.MASS_DENSITY_CONVERSION, fu_mass)
        self.result_density.SetValue(self.density)

    def OnLoadCif(self, event):
        fd=wx.FileDialog(self,
                         message="Open a (.cif) File...",
                         wildcard='Crystallographic Information File|*.cif;*.CIF|All Files|*',
                         defaultFile='crystal.cif',
                         style=wx.FD_OPEN|wx.FD_CHANGE_DIR)
        if fd.ShowModal()==wx.ID_OK:
            filename=fd.GetPath()
            self.extract_cif(filename)
        fd.Destroy()

    def OnQuery(self, event):
      if os.path.exists(os.path.join(config_path, 'materials.key')):
        key=open(os.path.join(config_path, 'materials.key'), 'r').read().strip()
      else:
        dia=wx.TextEntryDialog(self,
            'Enter your Materials Project API key,\n (https://www.materialsproject.org/dashboard)',
            'Enter Key')
        if not dia.ShowModal()==wx.ID_OK:
          return None
        key=dia.GetValue()
        open(os.path.join(config_path, 'materials.key'), 'w').write(key+'\n')
      a=MPRester(key)
      res=a.get_data(self.formula_entry.GetValue())
      if type(res) is not list:
        return
      if len(res)>1:
        # more then one structure available, ask for user input to select appropriate
        items=[]
        for i, ri in enumerate(res):
          cs=ri['spacegroup']['crystal_system']
          sgs=ri['spacegroup']['symbol']
          frm=ri['full_formula']
          v=ri['volume']
          dens=ri['density']
          items.append(u'%i: %s (%s) | UC Formula: %s\n     Density: %s g/cm³ | UC Volume: %s'%
                       (i+1, sgs, cs, frm, dens, v))
          if ri['tags'] is not None:
            items[-1]+='\n     '+';'.join(ri['tags'][:3])
        dia=wx.SingleChoiceDialog(self,
                                  'Several entries have been found, please select appropriate:',
                                  'Select correct database entry',
                                  items)
        if not dia.ShowModal()==wx.ID_OK:
          return None
        res=res[dia.GetSelection()]
      else:
        res=res[0]
      return self.analyze_cif(res['cif'])

    def GetResult(self):
        return (self.extracted_elements, self.result_density.GetValue())

    def extract_cif(self, filename):
        """
          Try to get unit cell and formula unit information from a .cif file.
        """
        if not os.path.exists(filename):
            return
        txt=open(filename).read()
        return self.analyze_cif(txt)

    def analyze_cif(self, txt):
        cell_params=[1., 1., 1., 90., 90., 90., 1.]
        composition=''
        file_lines=txt.splitlines()
        for line in file_lines:
            if line.startswith('_cell_length_a'):
                cell_params[0]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_length_b'):
                cell_params[1]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_length_c'):
                cell_params[2]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_angle_alpha'):
                cell_params[3]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_angle_beta'):
                cell_params[4]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_angle_gamma'):
                cell_params[5]=round(float(line.split()[1].split('(')[0]), 3)
            if line.startswith('_cell_formula_units_Z'):
                cell_params[6]=int(float(line.split()[1]))
            if line.startswith('_chemical_formula_structural'):
                composition=line.strip().split(None, 1)[1].replace(")", "").replace("(", "").replace("'", "").replace('"', '')
            if line.startswith('_chemical_formula_sum') and composition=='':
                composition=line.strip().split(None, 1)[1].replace("'", "").replace('"', '')
        self.formula_entry.SetValue(composition)
        self.OnFormulaChanged(None)
        for value, entry in zip(cell_params,
                                [self.a_entry,
                                 self.b_entry,
                                 self.c_entry,
                                 self.alpha_entry,
                                 self.beta_entry,
                                 self.gamma_entry,
                                 self.FUs_entry]):
            entry.SetValue(str(value))
        self.OnUnitCellChanged(None)

# list of elements with their name, atomic number and atomic mass values (+GenX name)
# mostly to calculate atomic density from mass density
atomic_data={
              "D": ("Deuterium", 1, 2.01410178),
              "H": ("Hydrogen", 1, 1.0079),
              "He": ("Helium", 2, 4.0026),
              "Li": ("Lithium", 3, 6.941),
              "Be": ("Beryllium", 4, 9.0122),
              "B": ("Boron", 5, 10.811),
              "C": ("Carbon", 6, 12.0107),
              "N": ("Nitrogen", 7, 14.0067),
              "O": ("Oxygen", 8, 15.9994),
              "F": ("Fluorine", 9, 18.9984),
              "Ne": ("Neon", 10, 20.1797),
              "Na": ("Sodium", 11, 22.9897),
              "Mg": ("Magnesium", 12, 24.305),
              "Al": ("Aluminum", 13, 26.9815),
              "Si": ("Silicon", 14, 28.0855),
              "P": ("Phosphorus", 15, 30.9738),
              "S": ("Sulfur", 16, 32.065),
              "Cl": ("Chlorine", 17, 35.453),
              "Ar": ("Argon", 18, 39.948),
              "K": ("Potassium", 19, 39.0983),
              "Ca": ("Calcium", 20, 40.078),
              "Sc": ("Scandium", 21, 44.9559),
              "Ti": ("Titanium", 22, 47.867),
              "V": ("Vanadium", 23, 50.9415),
              "Cr": ("Chromium", 24, 51.9961),
              "Mn": ("Manganese", 25, 54.938),
              "Fe": ("Iron", 26, 55.845),
              "Co": ("Cobalt", 27, 58.9332),
              "Ni": ("Nickel", 28, 58.6934),
              "Cu": ("Copper", 29, 63.546),
              "Zn": ("Zinc", 30, 65.39),
              "Ga": ("Gallium", 31, 69.723),
              "Ge": ("Germanium", 32, 72.64),
              "As": ("Arsenic", 33, 74.9216),
              "Se": ("Selenium", 34, 78.96),
              "Br": ("Bromine", 35, 79.904),
              "Kr": ("Krypton", 36, 83.8),
              "Rb": ("Rubidium", 37, 85.4678),
              "Sr": ("Strontium", 38, 87.62),
              "Y": ("Yttrium", 39, 88.9059),
              "Zr": ("Zirconium", 40, 91.224),
              "Nb": ("Niobium", 41, 92.9064),
              "Mo": ("Molybdenum", 42, 95.94),
              "Tc": ("Technetium", 43, 98),
              "Ru": ("Ruthenium", 44, 101.07),
              "Rh": ("Rhodium", 45, 102.906),
              "Pd": ("Palladium", 46, 106.42),
              "Ag": ("Silver", 47, 107.868),
              "Cd": ("Cadmium", 48, 112.411),
              "In": ("Indium", 49, 114.818),
              "Sn": ("Tin", 50, 118.71),
              "Sb": ("Antimony", 51, 121.76),
              "Te": ("Tellurium", 52, 127.6),
              "I": ("Iodine", 53, 126.904),
              "Xe": ("Xenon", 54, 131.293),
              "Cs": ("Cesium", 55, 132.905),
              "Ba": ("Barium", 56, 137.327),
              "La": ("Lanthanum", 57, 138.905),
              "Ce": ("Cerium", 58, 140.116),
              "Pr": ("Praseodymium", 59, 140.908),
              "Nd": ("Neodymium", 60, 144.24),
              "Pm": ("Promethium", 61, 145),
              "Sm": ("Samarium", 62, 150.36),
              "Eu": ("Europium", 63, 151.964),
              "Gd": ("Gadolinium", 64, 157.25),
              "Tb": ("Terbium", 65, 158.925),
              "Dy": ("Dysprosium", 66, 162.5),
              "Ho": ("Holmium", 67, 164.93),
              "Er": ("Erbium", 68, 167.259),
              "Tm": ("Thulium", 69, 168.934),
              "Yb": ("Ytterbium", 70, 173.04),
              "Lu": ("Lutetium", 71, 174.967),
              "Hf": ("Hafnium", 72, 178.49),
              "Ta": ("Tantalum", 73, 180.948),
              "W": ("Tungsten", 74, 183.84),
              "Re": ("Rhenium", 75, 186.207),
              "Os": ("Osmium", 76, 190.23),
              "Ir": ("Iridium", 77, 192.217),
              "Pt": ("Platinum", 78, 195.078),
              "Au": ("Gold", 79, 196.966),
              "Hg": ("Mercury", 80, 200.59),
              "Tl": ("Thallium", 81, 204.383),
              "Pb": ("Lead", 82, 207.2),
              "Bi": ("Bismuth", 83, 208.98),
              "Po": ("Polonium", 84, 209),
              "At": ("Astatine", 85, 210),
              "Rn": ("Radon", 86, 222),
              "Fr": ("Francium", 87, 223),
              "Ra": ("Radium", 88, 226),
              "Ac": ("Actinium", 89, 227),
              "Th": ("Thorium", 90, 232.038),
              "Pa": ("Protactinium", 91, 231.036),
              "U": ("Uranium", 92, 238.029),
              "Np": ("Neptunium", 93, 237),
              "Pu": ("Plutonium", 94, 244),
              "Am": ("Americium", 95, 243),
              "Cm": ("Curium", 96, 247),
              "Bk": ("Berkelium", 97, 247),
              "Cf": ("Californium", 98, 251),
              "Es": ("Einsteinium", 99, 252),
              "Fm": ("Fermium", 100, 257),
              "Md": ("Mendelevium", 101, 258),
              "No": ("Nobelium", 102, 259),
              "Lr": ("Lawrencium", 103, 262),
              "Rf": ("Rutherfordium", 104, 261),
              "Db": ("Dubnium", 105, 262),
              "Sg": ("Seaborgium", 106, 266),
              "Bh": ("Bohrium", 107, 264),
              "Hs": ("Hassium", 108, 277),
              "Mt": ("Meitnerium", 109, 268),
              }

isotopes={
          'D': ('i2H', 'H'),
          }
