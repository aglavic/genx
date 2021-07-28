import string

import wx

class SliderControl(wx.Slider):
    """ A custom slider control."""

    def __init__(self, parent, id, value=0.0, min_value=0.0, max_value=100.0, font=None, border_frame=True,
                 size=wx.DefaultSize):
        int_value=int(100*(value-min_value)/(max_value-min_value))
        wx.Slider.__init__(self, parent, id, value=int_value, minValue=0, maxValue=100, size=size)
        self.min_value=min_value
        self.max_value=max_value
        self.value=float(value)

        self.border_frame=border_frame
        self.scroll_callback=None

    def bind_handlers(self):
        self.Bind(wx.EVT_CHAR, self.OnChar)
        self.Bind(wx.EVT_SLIDER, self.OnChange)

    def unbind_handlers(self):
        self.Unbind(wx.EVT_SLIDER)

    def OnChar(self, event):
        if chr(event.GetKeyCode())==' ':
            self.GetParent().GetParent().SaveEditControlValue()
            self.GetParent().GetParent().EnableCellEditControl(False)
            return
        else:
            event.Skip()

    def OnChange(self, event):
        self.value=self.GetValue()

        if self.scroll_callback is not None:
            self.scroll_callback(self.value)

    def set_slider_position(self):
        if self.min_value==self.max_value:
            int_value=50
        else:
            int_value=int(100*(self.value-self.min_value)/(self.max_value-self.min_value))
        wx.Slider.SetValue(self, int_value)

    def SetValue(self, value):
        self.value=value
        self.set_slider_position()

    def GetValue(self):
        int_value=wx.Slider.GetValue(self)
        return int_value/100.*(self.max_value-self.min_value)+self.min_value

    def SetMaxValue(self, value):
        self.max_value=value
        self.set_slider_position()

    def SetMinValue(self, value):
        self.min_value=value
        self.set_slider_position()

    def SetScrollCallback(self, func):
        """ Set a callback that is called on a scroll event.

        func should only take one parameter which is the value, i.e. func(value).
        """
        self.scroll_callback=func

class NumberValidator(wx.Validator):
    """Validaator to handle numerical values, accepts also 'e' as exponent value as input"""

    def __init__(self):
        wx.Validator.__init__(self)
        self.Bind(wx.EVT_CHAR, self.OnChar)

    def Clone(self):
        return NumberValidator()

    def Validate(self, win):
        tc=self.GetWindow()
        val=tc.GetValue()

        try:
            val=float(val)
        except:
            return False
        else:
            return True

    def OnChar(self, event):
        if event.AltDown() or event.ControlDown():
            event.Skip()
            return
        key=event.GetKeyCode()
        tc=self.GetWindow()
        val=tc.GetValue()
        pos=tc.GetInsertionPoint()

        if event.GetUnicodeKey()!=wx.WXK_NONE:
            # an actual character key was pressed

            if chr(key) in string.digits:
                if len(val)==0:
                    event.Skip()
                    return
                if pos==0 and val[0]=='-':
                    if not wx.Validator_IsSilent():
                        wx.Bell()
                    return
                event.Skip()
                return

            if chr(key)=='.' and '.' not in val:
                if 'e' in val:
                    # Check so that the . ends up correctly relative e
                    if pos>val.index('e'):
                        if not wx.Validator_IsSilent():
                            wx.Bell()
                        return
                if len(val)==0:
                    event.Skip()
                    return
                if pos==0 and val[0]=='-':
                    if not wx.Validator_IsSilent():
                        wx.Bell()
                    return

                event.Skip()
                return

            if chr(key)=='e' and 'e' not in val:
                if '.' in val:
                    # Check so that the e ends up correctly relative .
                    if pos<=val.index('.'):
                        if not wx.Validator_IsSilent():
                            wx.Bell()
                        return
                if len(val)==0:
                    event.Skip()
                    return
                if pos==0 and val[0]=='-':
                    if not wx.Validator_IsSilent():
                        wx.Bell()
                    return
                event.Skip()
                return

            if chr(key)=='-':
                if val[0]!='-' and pos==0:
                    event.Skip()
                elif val[pos-1]=='e' and '-' not in val[pos-1:]:
                    event.Skip()
                elif val[0]=='-':
                    tc.SetValue(val[1:])
                    tc.SetInsertionPoint(max(0,pos-1))
                else:
                    tc.SetValue('-'+val)
                    tc.SetInsertionPoint(pos+1)
                return

            if chr(key)==' ' and self.Validate(None):
                tc.GetParent().GetParent().SaveEditControlValue()
                tc.GetParent().GetParent().EnableCellEditControl(False)
                return

        if key in [wx.WXK_TAB, wx.WXK_RETURN, wx.WXK_BACK, wx.WXK_ESCAPE,
                     wx.WXK_DOWN, wx.WXK_UP, wx.WXK_LEFT, wx.WXK_RIGHT, wx.WXK_NUMPAD_LEFT,
                     wx.WXK_NUMPAD_RIGHT, wx.WXK_DELETE, wx.WXK_BACK, wx.WXK_HOME, wx.WXK_END]:
            event.Skip()
            return

        if not wx.Validator.IsSilent():
            wx.Bell()

        # Does not allow the event to propagate (kills it)
        return

class SpinCtrl(wx.TextCtrl):
    def __init__(self, parent, id=wx.NewId(), pos=wx.DefaultPosition, size=wx.DefaultSize, value=0.0,
                 min_value=None, max_value=None, steps=20, digits=6, name="SpinCtrl"):
        self.value=value
        self.digits=digits
        self.min_value=min_value
        self.max_value=max_value
        self.steps=steps
        self.value_change_func=None
        wx.TextCtrl.__init__(self, parent, id=id, pos=pos, size=size, name=name,
                             style=wx.TE_RIGHT,
                             value="%.*g"%(self.digits, self.value),
                             validator=NumberValidator())

        self.Layout()
        # self._spin=wx.SpinButton(self, wx.NewId(), pos=pos, size=wx.Size(20, self.GetSize().GetHeight()-8),
        #                          style=wx.SP_ARROW_KEYS | wx.SP_VERTICAL, name=name+"_SpinButton")
        # self._spin.Bind(wx.EVT_SPIN_UP, self.eh_spin_up,
        #                 id=self._spin.GetId())
        # self._spin.Bind(wx.EVT_SPIN_DOWN, self.eh_spin_down,
        #                 id=self._spin.GetId())

        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
        self.Bind(wx.EVT_CHAR, self.OnChar)

    def step(self, dir_up=True):
        """Steps the value up/down on increment

        :param dir_up: stepping direction
        :return:
        """
        inc=(self.max_value-self.min_value)*1.0/self.steps
        val=self.value
        if dir_up:
            val+=inc
        else:
            val-=inc

        # Check so that the bounds are kept
        val=max(val, self.min_value)
        val=min(val, self.max_value)

        self.SetValue(val)

        if self.value_change_func is not None:
            self.value_change_func(val)

    def eh_spin_up(self, event):
        self.step(dir_up=True)
        if self.value_change_func is not None:
            self.value_change_func(self.GetValue())
        event.Skip()

    def eh_spin_down(self, event):
        self.step(dir_up=False)
        if self.value_change_func is not None:
            self.value_change_func(self.GetValue())
        event.Skip()

    def OnMouseWheel(self, event):
        if event.GetWheelRotation()>0 and not event.IsWheelInverted():
            self.step(dir_up=True)
        else:
            self.step(dir_up=False)

    def OnChar(self, event):
        # print event.GetKeyCode()
        if event.GetKeyCode()==wx.WXK_UP:
            self.step(dir_up=True)
        elif event.GetKeyCode()==wx.WXK_DOWN:
            self.step(dir_up=False)
        else:
            event.Skip()

    def SetValue(self, value):
        try:
            self.value=float(value)
            wx.TextCtrl.SetValue(self, "%.*g"%(self.digits, self.value))
        except ValueError:
            wx.TextCtrl.SetValue(self, value)

    def GetValue(self):
        try:
            self.value=float(wx.TextCtrl.GetValue(self))
        except ValueError:
            pass
        return wx.TextCtrl.GetValue(self)

    def SetValueChangeCallback(self, func):
        """ Sets a callback function to execute when the value has changed
        :param func:
        :return:
        """
        self.value_change_func=func

    def SetRange(self, min_value, max_value):
        """ Sets the range of the control

        :param min_value:
        :param max_value:
        :return:
        """
        if max_value<min_value:
            tmp_value=min_value
            min_value=max_value
            max_value=tmp_value
        self.min_value=float(min_value)
        self.max_value=float(max_value)
