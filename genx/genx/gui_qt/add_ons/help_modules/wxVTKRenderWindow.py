"""
Qt VTK render window widget with wxVTKRenderWindow-compatible API.
"""

from PySide6 import QtCore, QtGui, QtWidgets

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except Exception:  # pragma: no cover - fallback for older VTK
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkProperty


class _EventProxy:
    def __init__(self, event, button=None, is_press=True):
        self._event = event
        self._button = button
        self._is_press = is_press

    def GetX(self):
        return int(self._event.position().x())

    def GetY(self):
        return int(self._event.position().y())

    def LeftDown(self):
        return self._is_press and self._button == QtCore.Qt.MouseButton.LeftButton

    def RightDown(self):
        return self._is_press and self._button == QtCore.Qt.MouseButton.RightButton

    def MiddleDown(self):
        return self._is_press and self._button == QtCore.Qt.MouseButton.MiddleButton

    def LeftUp(self):
        return (not self._is_press) and self._button == QtCore.Qt.MouseButton.LeftButton

    def RightUp(self):
        return (not self._is_press) and self._button == QtCore.Qt.MouseButton.RightButton

    def MiddleUp(self):
        return (not self._is_press) and self._button == QtCore.Qt.MouseButton.MiddleButton

    def ControlDown(self):
        return bool(self._event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier)

    def ShiftDown(self):
        return bool(self._event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier)

    def AltDown(self):
        return bool(self._event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier)

    def GetKeyCode(self):
        return self._event.key()

    def Skip(self):
        return None


class wxVTKRenderWindow(QVTKRenderWindowInteractor):
    """
    Qt equivalent of wxVTKRenderWindow with the same interaction API.
    """

    def __init__(self, parent, _id=None, stereo=0, **kwargs):
        super().__init__(parent, **kwargs)

        self._CurrentRenderer = None
        self._CurrentCamera = None
        self._CurrentZoom = 1.0
        self._CurrentLight = None

        self._ViewportCenterX = 0
        self._ViewportCenterY = 0

        self._Picker = vtkCellPicker()
        self._PickedActor = None
        self._PickedProperty = vtkProperty()
        self._PickedProperty.SetColor(1, 0, 0)
        self._PrePickedProperty = None

        self._LastX = 0
        self._LastY = 0

        self._Mode = None
        self._ActiveButton = None

        self._DesiredUpdateRate = 15
        self._StillUpdateRate = 0.0001

        self._RenderWindow = self.GetRenderWindow()
        if stereo:
            self._RenderWindow.StereoCapableWindowOn()
            self._RenderWindow.SetStereoTypeToCrystalEyes()

        self.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

    def SetDesiredUpdateRate(self, rate):
        self._DesiredUpdateRate = rate

    def GetDesiredUpdateRate(self):
        return self._DesiredUpdateRate

    def SetStillUpdateRate(self, rate):
        self._StillUpdateRate = rate

    def GetStillUpdateRate(self):
        return self._StillUpdateRate

    def GetRenderWindow(self):
        return super().GetRenderWindow()

    def GetPicker(self):
        return self._Picker

    def Render(self):
        if self._CurrentLight:
            light = self._CurrentLight
            light.SetPosition(self._CurrentCamera.GetPosition())
            light.SetFocalPoint(self._CurrentCamera.GetFocalPoint())
        self._RenderWindow.Render()

    def UpdateRenderer(self, event):
        x = event.GetX()
        y = event.GetY()
        windowX, windowY = self._RenderWindow.GetSize()

        renderers = self._RenderWindow.GetRenderers()
        numRenderers = renderers.GetNumberOfItems()

        self._CurrentRenderer = None
        renderers.InitTraversal()
        for _i in range(0, numRenderers):
            renderer = renderers.GetNextItem()
            vx, vy = (0, 0)
            if windowX > 1:
                vx = float(x) / (windowX - 1)
            if windowY > 1:
                vy = (windowY - float(y) - 1) / (windowY - 1)
            (vpxmin, vpymin, vpxmax, vpymax) = renderer.GetViewport()

            if vpxmin <= vx <= vpxmax and vpymin <= vy <= vpymax:
                self._CurrentRenderer = renderer
                self._ViewportCenterX = float(windowX) * (vpxmax - vpxmin) / 2.0 + vpxmin
                self._ViewportCenterY = float(windowY) * (vpymax - vpymin) / 2.0 + vpymin
                self._CurrentCamera = self._CurrentRenderer.GetActiveCamera()
                lights = self._CurrentRenderer.GetLights()
                lights.InitTraversal()
                self._CurrentLight = lights.GetNextItem()
                break

        self._LastX = x
        self._LastY = y

    def OnLeftDown(self, event):
        if not self._Mode:
            if event.ControlDown():
                self._Mode = "Zoom"
            elif event.ShiftDown():
                self._Mode = "Pan"
            else:
                self._Mode = "Rotate"

    def OnRightDown(self, event):
        if not self._Mode:
            self._Mode = "Zoom"

    def OnMiddleDown(self, event):
        if not self._Mode:
            self._Mode = "Pan"

    def OnLeftUp(self, _event):
        pass

    def OnRightUp(self, _event):
        pass

    def OnMiddleUp(self, _event):
        pass

    def OnButtonDown(self, event):
        if not self._Mode:
            self.UpdateRenderer(event)

        if event.LeftDown():
            self.OnLeftDown(event)
        elif event.RightDown():
            self.OnRightDown(event)
        elif event.MiddleDown():
            self.OnMiddleDown(event)

    def OnButtonUp(self, event):
        if event.LeftUp():
            self.OnLeftUp(event)
        elif event.RightUp():
            self.OnRightUp(event)
        elif event.MiddleUp():
            self.OnMiddleUp(event)

        if self._Mode and self._CurrentRenderer:
            self.Render()
        self._Mode = None

    def OnMotion(self, event):
        if self._Mode == "Pan":
            self.Pan(event)
        elif self._Mode == "Rotate":
            self.Rotate(event)
        elif self._Mode == "Zoom":
            self.Zoom(event)

    def OnChar(self, _event):
        pass

    def OnKeyDown(self, event):
        key = event.GetKeyCode()
        if key == ord("R"):
            self.Reset(event)
        if key == ord("W"):
            self.Wireframe()
        if key == ord("S"):
            self.Surface()
        if key == ord("P"):
            self.PickActor(event)
        if key < 256:
            self.OnChar(event)

    def OnKeyUp(self, _event):
        pass

    def GetZoomFactor(self):
        return self._CurrentZoom

    def Rotate(self, event):
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            self._CurrentCamera.Azimuth(self._LastX - x)
            self._CurrentCamera.Elevation(y - self._LastY)
            self._CurrentCamera.OrthogonalizeViewUp()

            self._LastX = x
            self._LastY = y

            self._CurrentRenderer.ResetCameraClippingRange()
            self.Render()

    def Pan(self, event):
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            renderer = self._CurrentRenderer
            camera = self._CurrentCamera
            (pPoint0, pPoint1, pPoint2) = camera.GetPosition()
            (fPoint0, fPoint1, fPoint2) = camera.GetFocalPoint()

            if camera.GetParallelProjection():
                renderer.SetWorldPoint(fPoint0, fPoint1, fPoint2, 1.0)
                renderer.WorldToDisplay()
                fx, fy, fz = renderer.GetDisplayPoint()
                renderer.SetDisplayPoint(fx - x + self._LastX, fy + y - self._LastY, fz)
                renderer.DisplayToWorld()
                fx, fy, fz, _fw = renderer.GetWorldPoint()
                camera.SetFocalPoint(fx, fy, fz)

                renderer.SetWorldPoint(pPoint0, pPoint1, pPoint2, 1.0)
                renderer.WorldToDisplay()
                fx, fy, fz = renderer.GetDisplayPoint()
                renderer.SetDisplayPoint(fx - x + self._LastX, fy + y - self._LastY, fz)
                renderer.DisplayToWorld()
                fx, fy, fz, _fw = renderer.GetWorldPoint()
                camera.SetPosition(fx, fy, fz)
            else:
                renderer.SetWorldPoint(fPoint0, fPoint1, fPoint2, 1.0)
                renderer.WorldToDisplay()
                dPoint = renderer.GetDisplayPoint()
                focalDepth = dPoint[2]

                aPoint0 = self._ViewportCenterX + (x - self._LastX)
                aPoint1 = self._ViewportCenterY - (y - self._LastY)

                renderer.SetDisplayPoint(aPoint0, aPoint1, focalDepth)
                renderer.DisplayToWorld()

                (rPoint0, rPoint1, rPoint2, rPoint3) = renderer.GetWorldPoint()
                if rPoint3 != 0.0:
                    rPoint0 = rPoint0 / rPoint3
                    rPoint1 = rPoint1 / rPoint3
                    rPoint2 = rPoint2 / rPoint3

                camera.SetFocalPoint(
                    (fPoint0 - rPoint0) + fPoint0, (fPoint1 - rPoint1) + fPoint1, (fPoint2 - rPoint2) + fPoint2
                )
                camera.SetPosition(
                    (fPoint0 - rPoint0) + pPoint0, (fPoint1 - rPoint1) + pPoint1, (fPoint2 - rPoint2) + pPoint2
                )

            self._LastX = x
            self._LastY = y
            self.Render()

    def Zoom(self, event):
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            renderer = self._CurrentRenderer
            camera = self._CurrentCamera

            zoomFactor = pow(1.02, (0.5 * (self._LastY - y)))
            self._CurrentZoom = self._CurrentZoom * zoomFactor

            if camera.GetParallelProjection():
                parallelScale = camera.GetParallelScale() / zoomFactor
                camera.SetParallelScale(parallelScale)
            else:
                camera.Dolly(zoomFactor)
                renderer.ResetCameraClippingRange()

            self._LastX = x
            self._LastY = y
            self.Render()

    def Reset(self, _event=None):
        if self._CurrentRenderer:
            self._CurrentRenderer.ResetCamera()
        self.Render()

    def Wireframe(self):
        actors = self._CurrentRenderer.GetActors()
        numActors = actors.GetNumberOfItems()
        actors.InitTraversal()
        for _i in range(0, numActors):
            actor = actors.GetNextItem()
            actor.GetProperty().SetRepresentationToWireframe()
        self.Render()

    def Surface(self):
        actors = self._CurrentRenderer.GetActors()
        numActors = actors.GetNumberOfItems()
        actors.InitTraversal()
        for _i in range(0, numActors):
            actor = actors.GetNextItem()
            actor.GetProperty().SetRepresentationToSurface()
        self.Render()

    def PickActor(self, event):
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()
            renderer = self._CurrentRenderer
            picker = self._Picker
            windowX, windowY = self._RenderWindow.GetSize()
            picker.Pick(x, (windowY - y - 1), 0.0, renderer)
            actor = picker.GetActor()

            if self._PickedActor is not None and self._PrePickedProperty is not None:
                self._PickedActor.SetProperty(self._PrePickedProperty)
                self._PrePickedProperty.UnRegister(self._PrePickedProperty)
                self._PrePickedProperty = None

            if actor is not None:
                self._PickedActor = actor
                self._PrePickedProperty = self._PickedActor.GetProperty()
                self._PrePickedProperty.Register(self._PrePickedProperty)
                self._PickedActor.SetProperty(self._PickedProperty)
            self.Render()

    def mousePressEvent(self, event):
        proxy = _EventProxy(event, event.button(), True)
        self._RenderWindow.SetDesiredUpdateRate(self._DesiredUpdateRate)
        if not self._ActiveButton:
            self._ActiveButton = event.button()
        self.OnButtonDown(proxy)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        proxy = _EventProxy(event, event.button(), False)
        self._RenderWindow.SetDesiredUpdateRate(self._StillUpdateRate)
        if self._ActiveButton == event.button():
            self._ActiveButton = None
        self.OnButtonUp(proxy)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        proxy = _EventProxy(event)
        self.OnMotion(proxy)
        super().mouseMoveEvent(event)

    def keyPressEvent(self, event):
        proxy = _EventProxy(event)
        self.OnKeyDown(proxy)
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        proxy = _EventProxy(event)
        self.OnKeyUp(proxy)
        super().keyReleaseEvent(event)

    def resizeEvent(self, event):
        size = event.size()
        self._RenderWindow.SetSize(size.width(), size.height())
        self.Render()
        super().resizeEvent(event)

    def paintEvent(self, event):
        self.Render()
        super().paintEvent(event)
