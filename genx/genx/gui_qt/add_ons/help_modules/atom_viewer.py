__author__ = "GenX"

import string
from functools import reduce

import vtk.util.colors as vtkc
from PySide6 import QtGui, QtWidgets
from numpy import arctan2, pi, sign, sqrt
from vtk import (
    vtkActor,
    vtkActor2D,
    vtkArrowSource,
    vtkAxesActor,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkSphereSource,
    vtkTextMapper,
    vtkTransform,
)

from genx.core.custom_logging import iprint
from genx.tools import atom_colors as atom_colors

from . import custom_dialog
from .wxVTKRenderWindow import wxVTKRenderWindow


class VTKview(wxVTKRenderWindow):
    def __init__(self, parent):
        super().__init__(parent, None, stereo=0)
        self.ren = vtkRenderer()
        self.GetRenderWindow().AddRenderer(self.ren)
        self.ren.SetBackground(0, 0, 0)
        self.sphereActor = vtkActor()

        self.x_uc = 1
        self.y_uc = 1
        self.use_sym = True
        self.fold_sym = True
        self.show_bulk = True
        self.arrows_for_atoms = 0.0

        self.radius = 1.0
        self.theta_res = 15
        self.phi_res = 15
        self.amb_col = (1.0, 1.0, 1.0)
        self.diffuse = 1.0
        self.ambient = 0.2
        self.specular = 0.5
        self.specular_power = 50.0
        self.specular_col = (1.0, 1.0, 1.0)
        self.element_col = atom_colors.jmol
        self.default_col = vtkc.yellow

        self.search_radius = 0.1

        self.toolbar = None
        self.cursor_mode = "orbit"

        self.textMapper = vtkTextMapper()
        tprop = self.textMapper.GetTextProperty()
        tprop.SetFontFamilyToArial()
        tprop.SetFontSize(10)
        tprop.BoldOn()
        tprop.ShadowOn()
        tprop.SetColor(1, 1, 1)
        self.textActor = vtkActor2D()
        self.textActor.VisibilityOff()
        self.textActor.SetMapper(self.textMapper)

        self.ren.AddActor(self.textActor)

    def do_toolbar(self, parent):
        toolbar = QtWidgets.QToolBar(parent)
        self.toolbar = toolbar
        self._cursor_actions = {}

        def add_action(label, callback, checkable=False):
            action = toolbar.addAction(label)
            action.setCheckable(checkable)
            action.triggered.connect(callback)
            return action

        add_action("View X", self.OnViewX)
        add_action("View Y", self.OnViewY)
        add_action("View Z", self.OnViewZ)
        add_action("Isometric", self.OnViewIsometric)
        toolbar.addSeparator()

        self._cursor_actions["select"] = add_action("Select", self.OnChangeCursorState, True)
        self._cursor_actions["orbit"] = add_action("Orbit", self.OnChangeCursorState, True)
        self._cursor_actions["zoom"] = add_action("Zoom", self.OnChangeCursorState, True)
        self._cursor_actions["pan"] = add_action("Pan", self.OnChangeCursorState, True)
        self._cursor_actions["orbit"].setChecked(True)

        return toolbar

    def OnViewX(self, _event=None):
        if self._CurrentRenderer:
            self._CurrentCamera.SetFocalPoint(0, 0, 0)
            self._CurrentCamera.SetPosition(1.0, 0.0, 0.0)
            self._CurrentCamera.SetViewUp(0.0, 0.0, 1.0)
            self._CurrentRenderer.ResetCamera()
            self.Render()

    def OnViewY(self, _event=None):
        if self._CurrentRenderer:
            self._CurrentCamera.SetFocalPoint(0, 0, 0)
            self._CurrentCamera.SetPosition(0.0, 1.0, 0.0)
            self._CurrentCamera.SetViewUp(0.0, 0.0, 1.0)
            self._CurrentRenderer.ResetCamera()
            self.Render()

    def OnViewZ(self, _event=None):
        if self._CurrentRenderer:
            self._CurrentCamera.SetFocalPoint(0, 0, 0)
            self._CurrentCamera.SetPosition(0.0, 0.0, 1.0)
            self._CurrentCamera.SetViewUp(0.0, 1.0, 0.0)
            self._CurrentRenderer.ResetCamera()
            self.Render()

    def OnViewIsometric(self, _event=None):
        if self._CurrentRenderer:
            self._CurrentCamera.SetFocalPoint(0, 0, 0)
            self._CurrentCamera.SetPosition(1.0, 1.0, 1.0)
            self._CurrentCamera.SetViewUp(0.0, 0.0, 1.0)
            self._CurrentRenderer.ResetCamera()
            self.Render()

    def OnChangeCursorState(self):
        if self.toolbar:
            for action in self._cursor_actions.values():
                action.setChecked(False)
            sender = self.sender()
            if sender:
                sender.setChecked(True)
                name = sender.text()
            else:
                name = "Orbit"
            if self.cursor_mode == "select":
                self.textActor.VisibilityOff()
                self.sphereActor.VisibilityOff()
                self.Render()
            if name == "Select":
                self.cursor_mode = "select"
            elif name == "Orbit":
                self.cursor_mode = "orbit"
            elif name == "Zoom":
                self.cursor_mode = "zoom"
            elif name == "Pan":
                self.cursor_mode = "pan"
            else:
                iprint("VTKView.OnChangeCursorState: Button name ", name, "is not a known button")

    def highlight(self, actor):
        sphere = vtkSphereSource()
        sphere.SetRadius(self.radius * 1.1)
        sphere.SetCenter(actor.GetCenter())
        sphere.SetThetaResolution(self.theta_res)
        sphere.SetPhiResolution(self.phi_res)

        spheremapper = vtkPolyDataMapper()
        spheremapper.SetInputConnection(sphere.GetOutputPort())

        sphereActor = vtkActor()
        sphereActor.SetMapper(spheremapper)
        sphereActor.GetProperty().SetColor((0.0, 0.0, 0.0))
        sphereActor.GetProperty().SetOpacity(0.5)

        try:
            self.ren.RemoveViewProp(self.sphereActor)
        except Exception as exc:
            iprint(exc)
        self.ren.AddActor(sphereActor)
        self.sphereActor = sphereActor
        self.Render()

    def OnLeftDown(self, event):
        if self.cursor_mode == "zoom":
            self._Mode = "Zoom"
        elif self.cursor_mode == "pan":
            self._Mode = "Pan"
        elif self.cursor_mode == "orbit":
            self._Mode = "Rotate"
        elif self.cursor_mode == "select":
            self.OnSelectAtom(event)
            event.Skip()

    def OnSelectAtom(self, event):
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
            else:
                x, y, z = actor.GetCenter()
                self.textMapper.SetInput("%s" % reduce(lambda x, y: x + "," + y, self.locate_ids(x, y, z)))
                self.textActor.SetPosition(picker.GetSelectionPoint()[:2])
                self.textActor.VisibilityOn()
                self.highlight(actor)

    def OnRightDown(self, event):
        if not self._Mode:
            if event.ControlDown():
                self._Mode = "Rotate"
            elif event.ShiftDown():
                self._Mode = "Pan"
            else:
                self._Mode = "Zoom"

    def locate_ids(self, x, y, z):
        within_dist = ((self.atom_x - x) ** 2 + (self.atom_y - y) ** 2 + (self.atom_z - z) ** 2) < self.search_radius**2
        ids = []
        for i, pick in enumerate(within_dist):
            if pick and self.atom_ids[i] not in ids:
                ids.append(self.atom_ids[i])
        return ids

    def _get_col(self, element):
        element = element.lower()
        if len(element) > 2:
            if element[-2] in string.digits:
                element = element[:-2]
        if element in self.element_col:
            col = self.element_col[element]
        else:
            col = self.default_col
        return col

    def clear_view(self):
        self.ren.RemoveAllViewProps()
        self.ren.AddActor2D(self.textActor)
        self.textActor.VisibilityOff()

    def create_axes(self, a=2.0, b=2.0, c=2.0):
        axesActor = vtkAxesActor()
        axesActor.SetShaftTypeToCylinder()
        axesActor.SetTotalLength(a, b, c)
        self.ren.AddActor(axesActor)

    def _create_sphere(self, x, y, z, col, opacity=1.0):
        sphere = vtkSphereSource()
        sphere.SetRadius(self.radius)
        sphere.SetCenter(x, y, z)
        sphere.SetThetaResolution(self.theta_res)
        sphere.SetPhiResolution(self.phi_res)

        spheremapper = vtkPolyDataMapper()
        spheremapper.SetInputConnection(sphere.GetOutputPort())

        sphereActor = vtkActor()
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

    def add_atom(self, x, y, z, _id, element, opacity=1.0):
        col = self._get_col(element)
        sphere_actor = self._create_sphere(x, y, z, col, opacity)
        self.ren.AddActor(sphere_actor)

    def add_atom_shifts(self, x, y, z, dx, dy, dz, element):
        col = self._get_col(element)
        dx *= self.arrows_for_atoms
        dy *= self.arrows_for_atoms
        dz *= self.arrows_for_atoms
        length = sqrt(dx**2 + dy**2 + dz**2)
        if length == 0:
            return

        arrowSource = vtkArrowSource()
        arrowSource.SetShaftRadius(0.15 / length)
        arrowSource.SetTipRadius(0.3 / length)
        arrowSource.SetShaftResolution(16)
        arrowSource.SetTipResolution(16)

        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(arrowSource.GetOutputPort())
        actor = vtkActor()
        actor.SetMapper(mapper)

        transform = vtkTransform()
        transform.Translate(x, y, z)
        transform.RotateY(180.0 / pi * arctan2(dz, sqrt(dx**2 + dy**2)))
        transform.RotateZ(180.0 / pi * arctan2(dy, dx))
        transform.Scale(length, length, length)
        actor.SetUserTransform(transform)
        actor.GetProperty().SetColor(col)
        self.ren.AddActor(actor)

    def build_sample(self, sample, use_opacity=False):
        self.clear_view()
        x, y, z, _u, oc, el, ids = sample.create_uc_output(
            x_uc=self.x_uc, y_uc=self.y_uc, use_sym=self.use_sym, fold_sym=self.fold_sym, use_bulk=self.show_bulk
        )
        self.atom_x = x
        self.atom_y = y
        self.atom_z = z
        self.atom_ids = ids
        if use_opacity:
            [self.add_atom(x[i], y[i], z[i], "None", el[i], oc[i]) for i in range(len(x))]
        else:
            [self.add_atom(x[i], y[i], z[i], "None", el[i], 1.0) for i in range(len(x))]
        if self.arrows_for_atoms > 0:
            dx, dy, dz = sample.create_shift_output(
                x_uc=self.x_uc, y_uc=self.y_uc, use_sym=self.use_sym, fold_sym=self.fold_sym, use_bulk=self.show_bulk
            )
            [self.add_atom_shifts(x[i], y[i], z[i], dx[i], dy[i], dz[i], el[i]) for i in range(len(x))]
        self.create_axes()
        self.ren.ResetCamera()
        self.Render()

    def Show(self, *args):
        self.Render()

    def show(self):
        self.Reset()
        self.Render()

    def ShowSettingDialog(self):
        parameters = [
            "Use symmetry",
            "Fold symmetry op",
            "Show Bulk",
            "a unit cell rep.",
            "b unit cell rep.",
            "Atom radius",
            "Delta Arrow Scale",
        ]
        values = {
            "Use symmetry": self.use_sym,
            "Fold symmetry op": self.fold_sym,
            "Show Bulk": self.show_bulk,
            "a unit cell rep.": self.x_uc,
            "b unit cell rep.": self.y_uc,
            "Atom radius": self.radius,
            "Delta Arrow Scale": self.arrows_for_atoms,
        }
        units = {
            "Use symmetry": "",
            "Fold symmetry op": "",
            "Show Bulk": "",
            "a unit cell rep.": "uc",
            "b unit cell rep.": "uc",
            "Atom radius": "AA",
            "Delta Arrow Scale": "0=off",
        }
        validators = {
            "Use symmetry": True,
            "Fold symmetry op": True,
            "Show Bulk": True,
            "a unit cell rep.": 1,
            "b unit cell rep.": 1,
            "Atom radius": custom_dialog.FloatObjectValidator(),
            "Delta Arrow Scale": custom_dialog.FloatObjectValidator(),
        }
        groups = [
            ["Unit cell", ("Use symmetry", "Fold symmetry op", "Show Bulk", "a unit cell rep.", "b unit cell rep.")],
            ["Rendering", ("Atom radius", "Delta Arrow Scale")],
        ]

        dlg = custom_dialog.ValidateDialog(
            self, parameters, values, validators, units=units, groups=groups, title="Domain Viewer Settings"
        )
        if dlg.ShowModal() == QtWidgets.QDialog.DialogCode.Accepted:
            new_values = dlg.GetValues()
            self.use_sym = bool(new_values["Use symmetry"])
            self.fold_sym = bool(new_values["Fold symmetry op"])
            self.show_bulk = bool(new_values["Show Bulk"])
            self.x_uc = int(new_values["a unit cell rep."])
            self.y_uc = int(new_values["b unit cell rep."])
            self.radius = float(new_values["Atom radius"])
            self.arrows_for_atoms = float(new_values["Delta Arrow Scale"])
