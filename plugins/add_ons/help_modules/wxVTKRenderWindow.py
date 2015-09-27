"""
A simple VTK widget for wxPython.  Note that wxPython comes
with its own wxVTKRenderWindow in wxPython.lib.vtk.  Try both
and see which one works better for you.

Find wxPython info at http://wxPython.org

Created by David Gobbi, December 2001
Modified by Matts Bjorck, September 2015

Based on vtkTkRenderWindget.py
"""

"""
Please see the example at the end of this file.

----------------------------------------
Creation:

wxVTKRenderWindow(parent, ID, stereo=0, [wx keywords]):

You should create a wxPySimpleApp() or some other wx**App
before creating the window.

----------------------------------------
Methods:

Render()
AddRenderer(ren)
GetRenderers()
GetRenderWindow()

----------------------------------------
Methods to override (all take a wxEvent):

OnButtonDown(event)  default: propagate event to Left, Right, Middle
OnLeftDown(event)    default: set _Mode to 'Rotate'
OnRightDown(event)   default: set _Mode to 'Zoom'
OnMiddleDown(event)  default: set _Mode to 'Pan'

OnButtonUp(event)    default: propagate event to L, R, M and unset _Mode
OnLeftUp(event)
OnRightUp(event)
OnMiddleUp(event)

OnMotion(event)      default: call appropriate handler for _Mode

OnEnterWindow(event) default: set focus to this window
OnLeaveWindow(event) default: release focus

OnKeyDown(event)     default: [R]eset, [W]irefreme, [S]olid, [P]ick
OnKeyUp(event)
OnChar(event)

OnSetFocus(event)
OnKillFocus(event)

OnSize(event)
OnMove(event)

OnPaint(event)       default: Render()

----------------------------------------
Protected Members:

_Mode:                 Current mode: 'Rotate', 'Zoom', 'Pan'
_LastX, _LastY:        The (x,y) coordinates of the previous event
_CurrentRenderer:      The renderer that was most recently clicked in
_CurrentCamera:        The camera for the current renderer

"""

# import usual libraries
import math, os, sys
import wx
from vtk import *

# a few configuration items, see what works best on your system

# Use wxGLCanvas as base class instead of wxWindow.
# This is sometimes necessary under wxGTK or the image is blank.
# (in wxWindows 2.3.1 and earlier, the GLCanvas had scroll bars)
#try:
#    wx.WX_USE_GL_CANVAS
#except NameError:
if True:
    if wx.Platform == '__WXMSW__':
        WX_USE_GLCANVAS = 0
    else:
        WX_USE_GLCANVAS = 1
WX_USE_GLCANVAS = 0
# Keep capturing mouse after mouse is dragged out of window
# (in wxGTK 2.3.2 there is a bug that keeps this from working,
# but it is only relevant in wxGTK if there are multiple windows)
#try:
#    wx.WX_USE_X_CAPTURE
#except NameError:
if True:
    if wx.Platform == '__WXMSW__':
        WX_USE_X_CAPTURE = 1
    else:
        WX_USE_X_CAPTURE = 0

# end of configuration items


if WX_USE_GLCANVAS:
    from wx.glcanvas import GLCanvas
    baseClass = GLCanvas
else:
    baseClass = wx.Window


class wxVTKRenderWindow(baseClass):
    """
    A wxRenderWindow for wxPython.
    Use GetRenderWindow() to get the vtkRenderWindow.
    Create with the keyword stereo=1 in order to
    generate a stereo-capable window.
    """
    def __init__(self, parent, ID, *args, **kw):

        # miscellaneous protected variables
        self._CurrentRenderer = None
        self._CurrentCamera = None
        self._CurrentZoom = 1.0
        self._CurrentLight = None

        self._ViewportCenterX = 0
        self._ViewportCenterY = 0
        
        self._Picker = vtkCellPicker()
        self._PickedActor = None
        self._PickedProperty = vtkProperty()
        self._PickedProperty.SetColor(1,0,0)
        self._PrePickedProperty = None
        
        # these record the previous mouse position
        self._LastX = 0
        self._LastY = 0

        # the current interaction mode (Rotate, Pan, Zoom, etc)
        self._Mode = None
        self._ActiveButton = None

        # private attributes
        self.__OldFocus = None

        # used by the LOD actors
        self._DesiredUpdateRate = 15
        self._StillUpdateRate = 0.0001

        # First do special handling of some keywords:
        # stereo, position, size, width, height, style
        
        stereo = 0
        
        if kw.has_key('stereo'):
            if kw['stereo']:
                stereo = 1
            del kw['stereo']

        position = wx.DefaultPosition

        if kw.has_key('position'):
            position = kw['position']
            del kw['position']

        size = wx.DefaultSize

        if kw.has_key('size'):
            size = kw['size']
            del kw['size']
        
        if kw.has_key('width') and kw.has_key('height'):
            size = (kw['width'], kw['height'])
            del kw['width']
            del kw['height']

        # wxWANTS_CHARS says to give us e.g. TAB
        # wxNO_FULL_REPAINT_ON_RESIZE cuts down resize flicker under GTK
        style = wx.WANTS_CHARS | wx.NO_FULL_REPAINT_ON_RESIZE

        if kw.has_key('style'):
            style = style | kw['style']
            del kw['style']

        # the enclosing frame must be shown under GTK or the windows
        #  don't connect together properly
        l = []
        p = parent
        while p: # make a list of all parents
            l.append(p)
            p = p.GetParent()
        l.reverse() # sort list into descending order
        for p in l:
            p.Show()

        # initialize the wxWindow
        baseClass.__init__(self, parent, ID, position, size, style)

        # create the RenderWindow and initialize it
        self._RenderWindow = vtkRenderWindow()
        try:
            self._RenderWindow.SetSize(size.width, size.height)
        except AttributeError:
            self._RenderWindow.SetSize(size[0], size[1])
        if stereo:
            self._RenderWindow.StereoCapableWindowOn()
            self._RenderWindow.SetStereoTypeToCrystalEyes()

        self.__Created = 0
        # Tell the RenderWindow to render inside the wxWindow.
        if self.GetHandle():
            self.__Created = 1
            self._RenderWindow.SetWindowInfo(str(self.GetHandle()))

        # refresh window by doing a Render
        wx.EVT_PAINT(self, self.OnPaint)
        # turn off background erase to reduce flicker
        wx.EVT_ERASE_BACKGROUND(self, lambda e: None)
        
        # Bind the events to the event converters
        wx.EVT_RIGHT_DOWN(self, self._OnButtonDown)
        wx.EVT_LEFT_DOWN(self, self._OnButtonDown)
        wx.EVT_MIDDLE_DOWN(self, self._OnButtonDown)
        wx.EVT_RIGHT_UP(self, self._OnButtonUp)
        wx.EVT_LEFT_UP(self, self._OnButtonUp)
        wx.EVT_MIDDLE_UP(self, self._OnButtonUp)
        wx.EVT_MOTION(self, self.OnMotion)

        wx.EVT_ENTER_WINDOW(self, self._OnEnterWindow)
        wx.EVT_LEAVE_WINDOW(self, self._OnLeaveWindow)

        wx.EVT_CHAR(self, self.OnChar)
        wx.EVT_KEY_DOWN(self, self.OnKeyDown)
        wx.EVT_KEY_UP(self, self.OnKeyUp)
        
        wx.EVT_SIZE(self, self._OnSize)
        wx.EVT_MOVE(self, self.OnMove)
        
        wx.EVT_SET_FOCUS(self, self.OnSetFocus)
        wx.EVT_KILL_FOCUS(self, self.OnKillFocus)

    def SetDesiredUpdateRate(self, rate):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor."""
        self._DesiredUpdateRate = rate

    def GetDesiredUpdateRate(self):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor."""
        return self._DesiredUpdateRate 
        
    def SetStillUpdateRate(self, rate):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor."""
        self._StillUpdateRate = rate

    def GetStillUpdateRate(self):
        """Mirrors the method with the same name in
        vtkRenderWindowInteractor."""
        return self._StillUpdateRate 

    def OnPaint(self,event):
        dc = wx.PaintDC(self)
        self.Render()

    def _OnSize(self,event):
        if wx.Platform != '__WXMSW__':
            try:
                width, height = event.GetSize()
            except:
                width = event.GetSize().width
                height = event.GetSize().height
            self._RenderWindow.SetSize(width, height)
        self.OnSize(event)
        if self.__Created:
            self._RenderWindow.Render()

    def OnSize(self, event):
        pass

    def OnMove(self,event):
        pass

    def _OnEnterWindow(self,event):
        self.UpdateRenderer(event)
        self.OnEnterWindow(event)

    def OnEnterWindow(self,event):
        if self.__OldFocus == None:
            self.__OldFocus = wx.Window_FindFocus()
            self.SetFocus()

    def _OnLeaveWindow(self,event):
        self.OnLeaveWindow(event)

    def OnLeaveWindow(self,event):
        if self.__OldFocus:
            self.__OldFocus.SetFocus()
            self.__OldFocus = None

    def OnSetFocus(self,event):
        pass

    def OnKillFocus(self,event):
        pass

    def _OnButtonDown(self,event):
        # helper function for capturing mouse until button released
        self._RenderWindow.SetDesiredUpdateRate(self._DesiredUpdateRate)
        
        if event.RightDown():
            button = "Right"
        elif event.LeftDown():
            button = "Left"
        elif event.MiddleDown():
            button = "Middle"
        else:
            button = None

        # save the button and capture mouse until the button is released
        if button and not self._ActiveButton:
            self._ActiveButton = button
            if WX_USE_X_CAPTURE:
                self.CaptureMouse()

        self.OnButtonDown(event)

    def OnButtonDown(self,event):
        if not self._Mode:
            # figure out what renderer the mouse is over
            self.UpdateRenderer(event)

        if event.LeftDown():
            self.OnLeftDown(event)
        elif event.RightDown():
            self.OnRightDown(event)
        elif event.MiddleDown():
            self.OnMiddleDown(event)

    def OnLeftDown(self,event):
        if not self._Mode:
            if event.ControlDown():
                self._Mode = "Zoom"
            elif event.ShiftDown():
                self._Mode = "Pan"
            else:
                self._Mode = "Rotate"

    def OnRightDown(self,event):
        if not self._Mode:
            self._Mode = "Zoom"

    def OnMiddleDown(self,event):
        if not self._Mode:
            self._Mode = "Pan"

    def _OnButtonUp(self,event):
        # helper function for releasing mouse capture
        self._RenderWindow.SetDesiredUpdateRate(self._StillUpdateRate)

        if event.RightUp():
            button = "Right"
        elif event.LeftUp():
            button = "Left"
        elif event.MiddleUp():
            button = "Middle"
        else:
            button = None

        # if the ActiveButton is realeased, then release mouse capture
        if self._ActiveButton and button == self._ActiveButton:
            if WX_USE_X_CAPTURE:
                self.ReleaseMouse()
            self._ActiveButton = None

        self.OnButtonUp(event)

    def OnButtonUp(self,event):
        if event.LeftUp():
            self.OnLeftUp(event)
        elif event.RightUp():
            self.OnRightUp(event)
        elif event.MiddleUp():
            self.OnMiddleUp(event)

        # if not interacting, then do nothing more
        if self._Mode:
            if self._CurrentRenderer:
                self.Render()

        self._Mode = None

    def OnLeftUp(self,event):
        pass

    def OnRightUp(self,event):
        pass

    def OnMiddleUp(self,event):
        pass

    def OnMotion(self,event):
        if self._Mode == "Pan":
            self.Pan(event)
        elif self._Mode == "Rotate":
            self.Rotate(event)
        elif self._Mode == "Zoom":
            self.Zoom(event)

    def OnChar(self,event):
        pass

    def OnKeyDown(self,event):
        if event.GetKeyCode() == ord('R'):
            self.Reset(event)
        if event.GetKeyCode() == ord('W'):
            self.Wireframe()
        if event.GetKeyCode() == ord('S'):
            self.Surface()
        if event.GetKeyCode() == ord('P'):
            self.PickActor(event)

        if event.GetKeyCode() < 256:
            self.OnChar(event)

    def OnKeyUp(self,event):
        pass

    def GetZoomFactor(self):
        return self._CurrentZoom

    def GetRenderWindow(self):
        return self._RenderWindow

    def GetPicker(self):
        return self._Picker

    def Render(self):
        if self._CurrentLight:
            light = self._CurrentLight
            light.SetPosition(self._CurrentCamera.GetPosition())
            light.SetFocalPoint(self._CurrentCamera.GetFocalPoint())

        if self.__Created:
            self._RenderWindow.Render()
        elif self.GetHandle():
            self._RenderWindow.SetWindowInfo(str(self.GetHandle()))
            self.__Created = 1
            self._RenderWindow.Render()

    def UpdateRenderer(self, event):
        """
        UpdateRenderer will identify the renderer under the mouse and set
        up _CurrentRenderer, _CurrentCamera, and _CurrentLight.
        """
        x = event.GetX()
        y = event.GetY()
        windowX, windowY = self._RenderWindow.GetSize()

        renderers = self._RenderWindow.GetRenderers()
        numRenderers = renderers.GetNumberOfItems()

        self._CurrentRenderer = None
        renderers.InitTraversal()
        for i in range(0,numRenderers):
            renderer = renderers.GetNextItem()
            vx,vy = (0,0)
            if (windowX > 1):
                vx = float(x)/(windowX-1)
            if (windowY > 1):
                vy = (windowY-float(y)-1)/(windowY-1)
            (vpxmin,vpymin,vpxmax,vpymax) = renderer.GetViewport()
            
            if (vx >= vpxmin and vx <= vpxmax and vy >= vpymin and vy <= vpymax):
                self._CurrentRenderer = renderer
                self._ViewportCenterX = float(windowX)*(vpxmax-vpxmin)/2.0\
                                        +vpxmin
                self._ViewportCenterY = float(windowY)*(vpymax-vpymin)/2.0\
                                        +vpymin
                self._CurrentCamera = self._CurrentRenderer.GetActiveCamera()
                lights = self._CurrentRenderer.GetLights()
                lights.InitTraversal()
                self._CurrentLight = lights.GetNextItem()
                break

        self._LastX = x
        self._LastY = y

    def GetCurrentRenderer(self):
        return self._CurrentRenderer
                
    def Rotate(self,event):
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

    def Pan(self,event):
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()
            
            renderer = self._CurrentRenderer
            camera = self._CurrentCamera
            (pPoint0,pPoint1,pPoint2) = camera.GetPosition()
            (fPoint0,fPoint1,fPoint2) = camera.GetFocalPoint()

            if (camera.GetParallelProjection()):
                renderer.SetWorldPoint(fPoint0,fPoint1,fPoint2,1.0)
                renderer.WorldToDisplay()
                fx,fy,fz = renderer.GetDisplayPoint()
                renderer.SetDisplayPoint(fx-x+self._LastX,
                                         fy+y-self._LastY,
                                         fz)
                renderer.DisplayToWorld()
                fx,fy,fz,fw = renderer.GetWorldPoint()
                camera.SetFocalPoint(fx,fy,fz)

                renderer.SetWorldPoint(pPoint0,pPoint1,pPoint2,1.0)
                renderer.WorldToDisplay()
                fx,fy,fz = renderer.GetDisplayPoint()
                renderer.SetDisplayPoint(fx-x+self._LastX,
                                         fy+y-self._LastY,
                                         fz)
                renderer.DisplayToWorld()
                fx,fy,fz,fw = renderer.GetWorldPoint()
                camera.SetPosition(fx,fy,fz)
                
            else:
                (fPoint0,fPoint1,fPoint2) = camera.GetFocalPoint()
                # Specify a point location in world coordinates
                renderer.SetWorldPoint(fPoint0,fPoint1,fPoint2,1.0)
                renderer.WorldToDisplay()
                # Convert world point coordinates to display coordinates
                dPoint = renderer.GetDisplayPoint()
                focalDepth = dPoint[2]
                
                aPoint0 = self._ViewportCenterX + (x - self._LastX)
                aPoint1 = self._ViewportCenterY - (y - self._LastY)
                
                renderer.SetDisplayPoint(aPoint0,aPoint1,focalDepth)
                renderer.DisplayToWorld()
                
                (rPoint0,rPoint1,rPoint2,rPoint3) = renderer.GetWorldPoint()
                if (rPoint3 != 0.0):
                    rPoint0 = rPoint0/rPoint3
                    rPoint1 = rPoint1/rPoint3
                    rPoint2 = rPoint2/rPoint3

                camera.SetFocalPoint((fPoint0 - rPoint0) + fPoint0, 
                                     (fPoint1 - rPoint1) + fPoint1,
                                     (fPoint2 - rPoint2) + fPoint2) 
                
                camera.SetPosition((fPoint0 - rPoint0) + pPoint0, 
                                   (fPoint1 - rPoint1) + pPoint1,
                                   (fPoint2 - rPoint2) + pPoint2)

            self._LastX = x
            self._LastY = y

            self.Render()

    def Zoom(self,event):
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            renderer = self._CurrentRenderer
            camera = self._CurrentCamera

            zoomFactor = math.pow(1.02,(0.5*(self._LastY - y)))
            self._CurrentZoom = self._CurrentZoom * zoomFactor

            if camera.GetParallelProjection():
                parallelScale = camera.GetParallelScale()/zoomFactor
                camera.SetParallelScale(parallelScale)
            else:
                camera.Dolly(zoomFactor)
                renderer.ResetCameraClippingRange()

            self._LastX = x
            self._LastY = y

            self.Render()

    def Reset(self,event=None):
        if self._CurrentRenderer:
            self._CurrentRenderer.ResetCamera()
            
        self.Render()

    def Wireframe(self):
        actors = self._CurrentRenderer.GetActors()
        numActors = actors.GetNumberOfItems()
        actors.InitTraversal()
        for i in range(0,numActors):
            actor = actors.GetNextItem()
            actor.GetProperty().SetRepresentationToWireframe()

        self.Render()
        
    def Surface(self):
        actors = self._CurrentRenderer.GetActors()
        numActors = actors.GetNumberOfItems()
        actors.InitTraversal()
        for i in range(0,numActors):
            actor = actors.GetNextItem()
            actor.GetProperty().SetRepresentationToSurface()

        self.Render()

    def PickActor(self,event):
        if self._CurrentRenderer:
            x = event.GetX()
            y = event.GetY()

            renderer = self._CurrentRenderer
            picker = self._Picker
            
            windowX, windowY = self._RenderWindow.GetSize()
            picker.Pick(x,(windowY - y - 1),0.0,renderer)
            actor = picker.GetActor()

            if (self._PickedActor != None and
                self._PrePickedProperty != None):
                self._PickedActor.SetProperty(self._PrePickedProperty)
                # release hold of the property
                self._PrePickedProperty.UnRegister(self._PrePickedProperty)
                self._PrePickedProperty = None

            if (actor != None):
                self._PickedActor = actor
                self._PrePickedProperty = self._PickedActor.GetProperty()
                # hold onto the property
                self._PrePickedProperty.Register(self._PrePickedProperty)
                self._PickedActor.SetProperty(self._PickedProperty)

            self.Render()

#----------------------------------------------------------------------------  
def wxVTKRenderWindowConeExample():

    """Like it says, just a simple example
    """
    # every wx app needs an app
    app = wx.PySimpleApp()

    # create the widget
    frame = wx.Frame(None, -1, "wxRenderWindow", size=wx.Size(400,400))
    widget = wxVTKRenderWindow(frame, -1)

    ren = vtkRenderer()
    widget.GetRenderWindow().AddRenderer(ren)

    cone = vtkConeSource()
    cone.SetResolution(8)
    
    coneMapper = vtkPolyDataMapper()
    coneMapper.SetInputConnection(cone.GetOutputPort())
    
    coneActor = vtkActor()
    coneActor.SetMapper(coneMapper)

    ren.AddActor(coneActor)

    # show the window
    
    frame.Show(1)

    app.MainLoop()

if __name__ == "__main__":
    from vtk import *
    wxVTKRenderWindowConeExample()

