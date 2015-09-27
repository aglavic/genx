__author__ = 'GenX'

import string

import wx

import vtk
import vtk.util.colors as vtkc

from wxVTKRenderWindow import wxVTKRenderWindow
import atom_colors as atom_colors


class VTKview(wxVTKRenderWindow):
    def __init__(self, parent):
        wxVTKRenderWindow.__init__(self, parent, wx.NewId(), stereo=0)
        self.ren = vtk.vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.ren)
        self.ren.SetBackground(0, 0, 0)
        self.sphereActor = vtk.vtkActor()

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

    def highlight(self, actor):
        #outline = vtk.vtkOutlineFilter()
        #print dir(actor)
        #print dir(actor.GetProperty())

        sphere=vtk.vtkSphereSource()
        sphere.SetRadius(self.radius*1.1)
        sphere.SetCenter(actor.GetCenter())
        sphere.SetThetaResolution(self.theta_res)
        sphere.SetPhiResolution(self.phi_res)

        spheremapper=vtk.vtkPolyDataMapper()
        spheremapper.SetInputConnection(sphere.GetOutputPort())

        sphereActor=vtk.vtkActor()
        sphereActor.SetMapper(spheremapper)
        sphereActor.GetProperty().SetColor((0.0,0.0,0.0))
        sphereActor.GetProperty().SetOpacity(0.5)

        #sphereActor.GetProperty().SetRepresentationToWireframe()
        try:
            self.ren.RemoveViewProp(self.sphereActor)
        except Exception, e:
            print e
        self.ren.AddActor(sphereActor)
        self.sphereActor = sphereActor
        self.Render()

    def OnRightDown(self, event):
        ''' Overriding defualt Pick Actor '''
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
                #print pickPos
                #print actor.GetCenter()

                x, y, z = actor.GetCenter()
                self.textMapper.SetInput("%s" % reduce(lambda x, y : x + ',' + y, self.locate_ids(x, y, z)))
                self.textActor.SetPosition(picker.GetSelectionPoint()[:2])
                self.textActor.VisibilityOn()
                self.highlight(actor)

    def locate_ids(self, x, y, z):
        ''' Given the picked positions x,y,z locate the strings ids that
            are within a distance of self.search_radius.
            '''
        #print len(self.atom_x)
        #print len(self.atom_ids)
        #print x
        within_dist = (((self.atom_x-x)**2 + (self.atom_y-y)**2 + (self.atom_z - z)**2)
                       < self.search_radius**2)
        ids = []
        for i, pick in enumerate(within_dist):
            if pick and not self.atom_ids[i] in ids:
                ids.append(self.atom_ids[i])
        #print 'IDs:', ids
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

        spheremapper=vtk.vtkPolyDataMapper()
        spheremapper.SetInputConnection(sphere.GetOutputPort())

        sphereActor=vtk.vtkActor()
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
        x, y, z, u, oc, el, ids = sample.create_uc_output(x_uc=2, y_uc=2)
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
        #self.parent.Show(1)
        self.Reset()
        self.Render()