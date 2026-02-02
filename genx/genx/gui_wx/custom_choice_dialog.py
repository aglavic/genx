"""
An replacement for wx.SingleChoiceDialog that does not have issues with oversized width
and allows additional styling of elements.
"""

import wx


class SCDialog(wx.Dialog):
    def __init__(
        self,
        parent,
        message,
        caption,
        choices,
        background_colors=None,
        style=wx.CHOICEDLG_STYLE,
        pos=wx.DefaultPosition,
    ):
        wx.Dialog.__init__(self, parent, title=caption, style=style, pos=pos)
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        vbox.Add(wx.StaticText(self, label=message), 0, wx.FIXED_MINSIZE | wx.ALL, 4)
        self.list_widget = wx.ListBox(self, style=wx.LB_SINGLE | wx.LB_HSCROLL | wx.LB_OWNERDRAW, choices=choices)
        if background_colors:
            for i, ci in enumerate(background_colors):
                self.list_widget.SetItemBackgroundColour(i, ci)

        vbox.Add(self.list_widget, 1, wx.EXPAND)

        self.list_widget.Bind(wx.EVT_LISTBOX_DCLICK, self.selected)

    def selected(self, event):
        self.EndModal(wx.ID_OK)

    def GetSelection(self):
        return self.list_widget.GetSelection()
