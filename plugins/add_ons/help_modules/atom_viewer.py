__author__ = 'GenX'

import string

import wx

import vtk
import vtk.util.colors as vtkc

from wxVTKRenderWindow import wxVTKRenderWindow
import atom_colors as atom_colors
import custom_dialog
import sxrd_images


class VTKview(wxVTKRenderWindow):
    def __init__(self, parent):
        wxVTKRenderWindow.__init__(self, parent, wx.NewId(), stereo=0)
        self.ren = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.ren)
        self.ren.SetBackground(0, 0, 0)
        self.sphereActor = vtk.vtkActor()

        # Settings for creation of sample
        self.x_uc = 1
        self.y_uc = 1
        self.use_sym = True
        self.fold_sym = True

        # Some defualts
        self.radius = 1.0
        self.theta_res = 15
        self.phi_res = 15
        self.amb_col = (1., 1., 1.)
        self.diffuse = 1.0
        self.ambient = 0.2
        self.specular = 0.5
        self.specular_power = 50.
        self.specular_col = (1., 1., 1.)
        self.element_col = atom_colors.jmol
        self.default_col = vtkc.yellow

        self.search_radius = 0.1

        self.toolbar = None
        self.cursor_mode = 'orbit'

        self.textMapper = vtk.vtkTextMapper()
        tprop = self.textMapper.GetTextProperty()
        tprop.SetFontFamilyToArial()
        tprop.SetFontSize(10)
        tprop.BoldOn()
        tprop.ShadowOn()
        tprop.SetColor(1, 1, 1)
        self.textActor = vtk.vtkActor2D()
        self.textActor.VisibilityOff()
        self.textActor.SetMapper(self.textMapper)

        self.ren.AddActor(self.textActor)

    def do_toolbar(self, parent):
        """Create and return a toolbar that can be used to control the widget"""
        toolbar = wx.ToolBar(parent, style=wx.TB_FLAT | wx.TB_VERTICAL)

        button_names = ['View X', 'View Y', 'View Z', 'Isometric']
        button_images = [sxrd_images.x, sxrd_images.y, sxrd_images.z, sxrd_images.isometric]
        callbacks = [self.OnViewX, self.OnViewY, self.OnViewZ,self.OnViewIsometric]
        tooltips = ['View along X', 'View along Y', 'View along Z', 'Isometric view']

        for i in range(len(button_names)):
            new_id = wx.NewId()
            toolbar.AddLabelTool(new_id, label=button_names[i], bitmap=button_images[i].getBitmap(),
                                 shortHelp=tooltips[i])
            parent.Bind(wx.EVT_TOOL, callbacks[i], id=new_id)

        button_names = ['Select', 'Orbit', 'Zoom', 'Pan']
        button_images = [sxrd_images.selection, sxrd_images.orbit, sxrd_images.zoom_small, sxrd_images.pan]
        callbacks = [self.OnChangeCursorState, self.OnChangeCursorState, self.OnChangeCursorState,
                     self.OnChangeCursorState]
        tooltips = ['Object selection', 'Orbit', 'Zoom', 'Pan']

        self.cursor_ids = []
        for i in range(len(button_names)):
            new_id = wx.NewId()
            self.cursor_ids.append(new_id)
            print button_images[i].getBitmap()
            toolbar.AddCheckLabelTool(new_id, button_names[i], button_images[i].getBitmap(),
                                 shortHelp=tooltips[i])
            parent.Bind(wx.EVT_TOOL, callbacks[i], id=new_id)

        toolbar.ToggleTool(self.cursor_ids[1], True)
        self.toolbar = toolbar

        return toolbar

    def OnViewX(self, event):
        """Aligns the view with the x axis"""
        if self._CurrentRenderer:
            self._CurrentCamera.SetFocalPoint(0, 0, 0)
            self._CurrentCamera.SetPosition(1.0, 0.0, 0.0)
            self._CurrentCamera.SetViewUp(0.0, 0.0, 1.0)

            #self._CurrentCamera.OrthogonalizeViewUp()
            self._CurrentRenderer.ResetCamera()
            self.Render()

    def OnViewY(self, event):
        """Aligns the view with the y axis"""
        if self._CurrentRenderer:
            self._CurrentCamera.SetFocalPoint(0, 0, 0)
            self._CurrentCamera.SetPosition(0.0, 1.0, 0.0)
            self._CurrentCamera.SetViewUp(0.0, 0.0, 1.0)
            self._CurrentRenderer.ResetCamera()
            self.Render()

    def OnViewZ(self, event):
        """Alings the view with the z axis"""
        if self._CurrentRenderer:
            self._CurrentCamera.SetFocalPoint(0, 0, 0)
            self._CurrentCamera.SetPosition(0.0, 0.0, 1.0)
            self._CurrentCamera.SetViewUp(0.0, 1.0, 0.0)
            self._CurrentRenderer.ResetCamera()
            self.Render()

    def OnViewIsometric(self, event):
        """Creates an isometric view"""
        if self._CurrentRenderer:
            self._CurrentCamera.SetFocalPoint(0, 0, 0)
            self._CurrentCamera.SetPosition(1.0, 1.0, 1.0)
            self._CurrentCamera.SetViewUp(0.0, 0.0, 1.0)
            self._CurrentRenderer.ResetCamera()
            self.Render()

    def OnChangeCursorState(self, event):
        """Callback when changing the cursor state between select, orbit, zoom and pan"""
        print self.toolbar
        if self.toolbar:
            [self.toolbar.ToggleTool(cid, False) for cid in self.cursor_ids]
            self.toolbar.ToggleTool(event.GetId(), True)
            name = self.toolbar.FindById(event.GetId()).GetLabel()
            if self.cursor_mode == 'select':
                self.textActor.VisibilityOff()
                self.sphereActor.VisibilityOff()
                self.Render()
            if name == 'Select':
                self.cursor_mode = 'select'
            elif name == 'Orbit':
                self.cursor_mode = 'orbit'
            elif name == 'Zoom':
                self.cursor_mode = 'zoom'
            elif name == 'Pan':
                self.cursor_mode = 'pan'
            else:
                print 'VTKView.OnChangeCursorState: Button name ', name, 'is not a known button'
            print name, self.cursor_mode

    def highlight(self, actor):
        #outline = vtk.vtkOutlineFilter()
        #print dir(actor)
        #print dir(actor.GetProperty())

        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(self.radius*1.1)
        sphere.SetCenter(actor.GetCenter())
        sphere.SetThetaResolution(self.theta_res)
        sphere.SetPhiResolution(self.phi_res)

        spheremapper = vtk.vtkPolyDataMapper()
        spheremapper.SetInputConnection(sphere.GetOutputPort())

        sphereActor = vtk.vtkActor()
        sphereActor.SetMapper(spheremapper)
        sphereActor.GetProperty().SetColor((0.0, 0.0, 0.0))
        sphereActor.GetProperty().SetOpacity(0.5)

        #sphereActor.GetProperty().SetRepresentationToWireframe()
        try:
            self.ren.RemoveViewProp(self.sphereActor)
        except Exception, e:
            print e
        self.ren.AddActor(sphereActor)
        self.sphereActor = sphereActor
        self.Render()

    def OnLeftDown(self, event):
        """Overriding the defualt on left down"""
        if self.cursor_mode == 'zoom':
            self._Mode = "Zoom"
        elif self.cursor_mode == 'pan':
            self._Mode = "Pan"
        elif self.cursor_mode == 'orbit':
            self._Mode = "Rotate"
        elif self.cursor_mode == 'select':
            self.OnSelectAtom(event)
            event.Skip()

    def OnSelectAtom(self, event):
        """ Callback for  the selection of an atom"""
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            renderer = self._CurrentRenderer
            picker = self._Picker

            windowX, windowY = self._RenderWindow.GetSize()

            picker.Pick(x, (windowY - y - 1), 0.0, renderer)
            actor = picker.GetActor()
            if not picker.GetActor():
                self.textActor.VisibilityOff()
                self.sphereActor.VisibilityOff()
                self.Render()
                #print 'Could not locate an atom'
            else:
                selPt = picker.GetSelectionPoint()
                pickPos = picker.GetPickPosition()

                x, y, z = actor.GetCenter()
                self.textMapper.SetInput("%s" % reduce(lambda x, y : x + ',' + y, self.locate_ids(x, y, z)))
                self.textActor.SetPosition(picker.GetSelectionPoint()[:2])
                self.textActor.VisibilityOn()
                self.highlight(actor)

    def OnRightDown(self, event):
        ''' Overriding defualt Pick Actor '''
        if not self._Mode:
            if event.ControlDown():
                self._Mode = "Rotate"
            elif event.ShiftDown():
                self._Mode = "Pan"
            else:
                self._Mode = "Zoom"

    def locate_ids(self, x, y, z):
        ''' Given the picked positions x,y,z locate the strings ids that
            are within a distance of self.search_radius.
            '''
        within_dist = (((self.atom_x-x)**2 + (self.atom_y-y)**2 + (self.atom_z - z)**2)
                       < self.search_radius**2)
        ids = []
        for i, pick in enumerate(within_dist):
            if pick and not self.atom_ids[i] in ids:
                ids.append(self.atom_ids[i])
        return ids

    def _get_col(self, element):
        '''Translate element to color according to element dict.
        '''
        # We need to take care if the eleemnt is an ion.
        # the two last chars are on the form Xp or Xm where X is a digit
        element = element.lower()
        if len(element) > 2:
            if element[-2] in string.digits:
                element = element[:-2]
        if self.element_col.has_key(element):
            #print 'Found: ', element
            col = self.element_col[element]
        else:
            #print 'Missing: ', element
            col = self.default_col
        return col

    def clear_view(self):
        '''remove all actors so a new sample can be built.
        '''
        self.ren.RemoveAllViewProps()
        self.ren.AddActor2D(self.textActor)
        self.textActor.VisibilityOff()

    def create_axes(self, a=2.0, b=2.0, c=2.0):
        axesActor=vtk.vtkAxesActor()
        #axesActor.SetShaftTypeToCylinder()
        #axesActor.SetXAxisLabelText('x')
        #axesActor.SetYAxisLabelText('y')
        #axesActor.SetZAxisLabelText('z')
        axesActor.SetTotalLength(a, b, c)
        self.ren.AddActor(axesActor)

    def _create_sphere(self, x, y, z, col, opacity = 1.0):
        '''Create a sphere actor at (x,y,z) with color c and return it'''
        sphere=vtk.vtkSphereSource()
        sphere.SetRadius(self.radius)
        sphere.SetCenter(x,y,z)
        sphere.SetThetaResolution(self.theta_res)
        sphere.SetPhiResolution(self.phi_res)

        spheremapper = vtk.vtkPolyDataMapper()
        spheremapper.SetInputConnection(sphere.GetOutputPort())

        sphereActor = vtk.vtkActor()
        sphereActor.SetMapper(spheremapper)
        sphereActor.GetProperty().SetDiffuseColor(col)
        sphereActor.GetProperty().SetAmbientColor(self.amb_col)
        sphereActor.GetProperty().SetDiffuse(self.diffuse)
        sphereActor.GetProperty().SetAmbient(self.ambient)
        sphereActor.GetProperty().SetSpecular(self.specular)
        sphereActor.GetProperty().SetSpecularPower(self.specular_power)
        sphereActor.GetProperty().SetSpecularColor(self.specular_col)
        sphereActor.GetProperty().SetOpacity(opacity)
        return sphereActor

    def add_atom(self, x, y, z, id, element, opacity = 1.0):
        col = self._get_col(element)
        sphere_actor = self._create_sphere(x, y, z, col, opacity)
        self.ren.AddActor(sphere_actor)

    def build_sample(self, sample, use_opacity = False):
        '''Builds an sample view from the object sample that has to
        have the method _surf_pars that yields the x, y, z, u, oc, el
        '''
        self.clear_view()
        x, y, z, u, oc, el, ids = sample.create_uc_output(x_uc=self.x_uc, y_uc=self.y_uc, use_sym=self.use_sym,
                                                          fold_sym=self.fold_sym)
        self.atom_x = x
        self.atom_y = y
        self.atom_z = z
        self.atom_ids = ids
        if use_opacity:
            [self.add_atom(x[i], y[i], z[i], 'None', el[i], oc[i]) for i in range(len(x))]
        else:
            [self.add_atom(x[i], y[i], z[i], 'None', el[i], 1.0) for i in range(len(x))]
        self.create_axes()

        self.ren.ResetCamera()
        self.Render()

    def Show(self, *args):
        self.Render()

    def show(self):
        self.Reset()
        self.Render()

    def ShowSettingDialog(self):
        """Shows a settings dialog to change the settings"""
        parameters = ['Use symmetry', 'Fold symmetry op', 'a unit cell rep.', 'b unit cell rep.', 'Atom radius']
        values = {'Use symmetry': self.use_sym, 'Fold symmetry op': self.fold_sym, 'a unit cell rep.': self.x_uc,
                  'b unit cell rep.': self.y_uc, 'Atom radius': self.radius}
        units = {'Use symmetry': '', 'Fold symmetry op': '', 'a unit cell rep.': 'uc',
                 'b unit cell rep.': 'uc', 'Atom radius': 'AA'}
        validators = {'Use symmetry': True, 'Fold symmetry op': True, 'a unit cell rep.': 1,
                      'b unit cell rep.': 1, 'Atom radius': custom_dialog.FloatObjectValidator()}
        groups = [['Unit cell', ('Use symmetry', 'Fold symmetry op', 'a unit cell rep.',
                   'b unit cell rep.')], ['Rendering', ('Atom radius', )]]

        dlg = custom_dialog.ValidateDialog(self, parameters, values, validators, units=units, groups=groups,
                                           title="Domain Viewer Settings")
        if dlg.ShowModal() == wx.ID_OK:
            new_values = dlg.GetValues()
            self.use_sym = bool(new_values['Use symmetry'])
            self.fold_sym = bool(new_values['Fold symmetry op'])
            self.x_uc = int(new_values['a unit cell rep.'])
            self.y_uc = int(new_values['b unit cell rep.'])
            self.radius = float(new_values['Atom radius'])
