import wx
from standard_colours import colours

class SliderDrawer:
    """" Class that holds the values and makes the drawing and calcualtions of the slider."""
    radius_scale = 0.92
    def __init__(self, value, min=0, max=100, text_height=10.0):
        self.value = value
        self.min_value = min
        self.max_value = max
        self.snap_levels = 20
        self.border_thickness = 1
        self.SetTextHeight(text_height)
        self.background_colour = wx.WHITE
        self.slider_colour = colours.get_colour('aluminum')#wx.Colour(0xcc, 0, 0)

    def SetValue(self, value):
        self.value = value

    def GetValue(self):
        return float(self.value)

    def SetTextHeight(self, text_height):
        self.radius = text_height/2.0*self.radius_scale

    def SetMaxValue(self, value):
        self.max_value = float(value)

    def SetMinValue(self, value):
        self.min_value = float(value)

    def SetSnapLevels(self, value):
        self.snap_levels = int(value)

    def SetBackgroundColour(self, value):
        self.background_colour = value

    def Draw(self, dc, width, height, x=0, y=0, isSelected=False, show_border=False):
        dc.SetBackgroundMode(wx.SOLID)
        dc.SetBrush(wx.Brush(self.background_colour, wx.SOLID))
        #dc.SetBrush(wx.BRUSHSTYLE_TRANSPARENT)
        if not show_border:
            dc.SetPen(wx.TRANSPARENT_PEN)
        else:
            dc.SetPen(wx.Pen(wx.BLACK, width=self.border_thickness, style=wx.PENSTYLE_SOLID))
        dc.DrawRectangle(x, y, width, height)

        dc.SetPen(wx.TRANSPARENT_PEN)
        dc.SetBrush(wx.Brush("GREY", wx.SOLID))
        dc.DrawRectangle(x + self.radius + self.border_thickness,
                         y + height/2.0 - 1, width - 2*(self.radius+self.border_thickness),
                         2.0)

        dc.SetBackgroundMode(wx.SOLID)
        dc.SetBrush(wx.Brush(self.slider_colour, wx.SOLID))
        dc.SetPen(wx.Pen(wx.BLACK, width=1, style=wx.PENSTYLE_SOLID))
        xc, yc = self._calc_circle_center(width, height)
        dc.DrawCircle(x + xc,
                      y + yc,
                      self.radius)

    def _calc_circle_center(self, width, height):
        try:
            x = (self.radius + self.border_thickness +
                 (width - 2*(self.radius + self.border_thickness))*(self.value - self.min_value)/
                 (self.max_value - self.min_value))
        except ZeroDivisionError, e:
            x = self.radius + self.border_thickness
        y = height/2.
        return x, y

    def IsPosInSlider(self, x, y, width, height, x_offset=0, y_offset=0):
        "Test if the position given by x,y is within the slider."
        xc, yc = self._calc_circle_center(width, height)
        return (x_offset + xc - x)**2 + (y_offset + yc - y)**2 < self.radius**2

    def PositionToValue(self, x, w, x_offset=0, update_value=True):
        """Transform a Position inside the the Slider to a value.
            Function also updates the value"""
        val = (self.max_value - self.min_value)*(x - x_offset - self.radius)/(w - 2*self.radius) + self.min_value
        val = min(val, self.max_value)
        val = max(val, self.min_value)
        if update_value:
            self.value = val
        return val

    def JumpToNextSnap(self, up=True):
        step_width = (self.max_value - self.min_value)/self.snap_levels
        tol = 1e-6*step_width
        try:
            steps = int((self.value - self.min_value)/step_width)
            # Flag to indicate that we are on a snap step
            on_step = (abs((self.value - self.min_value)%step_width) < tol or
                        abs(abs((self.value - self.min_value)%step_width) - step_width) < tol)
        except ZeroDivisionError:
            return
        if up:
            # Move up
            old_val = self.value
            self.value = min((steps + 1)*step_width + self.min_value, self.max_value)
            if abs(old_val - self.value) < tol:
                self.value = min((steps + 2)*step_width + self.min_value, self.max_value)
        else:
            #Move down
            if not on_step:
                self.value = max(steps*step_width + self.min_value, self.min_value)
            else:
                self.value = max((steps - 1)*step_width + self.min_value, self.min_value)


class SliderControl(wx.Control):
    """ A custom slider control."""
    def __init__(self, parent, id, value=0.0, min_value=0.0, max_value=100.0, font=None, border_frame=True):
        wx.Control.__init__(self, parent, id)

        #self.parent = parent
        self.border_frame = border_frame
        if font is not None:
            th = font.GetPixelSize().GetHeight()
        else:
            th = 12
        self.slider_drawer = SliderDrawer(value, min=min_value, max=max_value, text_height=th)

        self.slider_moving = False
        self.mouse_offset = (0,0)

        self.scroll_callback = None

    def bind_handlers(self):
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_MOUSE_EVENTS, self.OnMouse)

    def unbind_handlers(self):
        self.Unbind(wx.EVT_PAINT)
        self.Unbind(wx.EVT_SIZE)
        self.Unbind(wx.EVT_MOUSE_EVENTS)


    def Destroy(self):
        self.unbind_handlers()
        super(SliderControl, self).Destroy()


    def SetValue(self, value):
        self.slider_drawer.SetValue(float(value))

    def GetValue(self):
        return self.slider_drawer.GetValue()

    def SetMaxValue(self, value):
        self.slider_drawer.SetMaxValue(value)

    def SetMinValue(self, value):
        self.slider_drawer.SetMinValue(value)

    def SetScrollCallback(self, func):
        """ Set a callback that is called on a scroll event.

        func should only take one parameter which is the value, i.e. func(value).
        """
        self.scroll_callback = func

    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        w, h = self.GetSize()
        if wx.Platform == "__WXMSW__":
            self.slider_drawer.Draw(dc, w-1, h-1, x=-1, y =-1, show_border=self.border_frame)
        else:
            self.slider_drawer.Draw(dc, w-1, h-1, x=1, y=1, show_border=self.border_frame)

    def OnSize(self, event):
        self.Refresh()

    def OnMouse(self, event):
        x, y = event.GetPositionTuple()
        w, h = self.GetSize()
        if event.LeftDown() and self.slider_drawer.IsPosInSlider(x,y, w, h):
            # Slider drag starts
            xc, yc = self.slider_drawer._calc_circle_center(w, h)
            self.slider_moving = True
            self.slider_offset = (x - xc, y - yc)
        elif event.LeftUp():
            if not self.slider_moving and not self.slider_drawer.IsPosInSlider(x,y, w, h):
                xc, yc = self.slider_drawer._calc_circle_center(w, h)
                self.slider_drawer.JumpToNextSnap(x - xc > 0)
                self.Refresh()
                if self.scroll_callback is not None:
                    self.scroll_callback(self.slider_drawer.GetValue())
            self.slider_moving = False
        #elif event.Leaving():
        #    self.slider_moving = False

        if self.slider_moving:
            self.slider_drawer.PositionToValue(x, w, self.slider_offset[0])
            self.Refresh()
            if self.scroll_callback is not None:
                    self.scroll_callback(self.slider_drawer.GetValue())
        event.Skip()


