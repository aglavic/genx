"""
A dialog that acts as file-like object for console output.
"""

from io import StringIO

import wx


class TextOutputDialog(wx.Dialog):
    def __init__(self, parent, autoflush=True):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER, name=f"GenX action details")
        self.SetTitle(f"Text output")
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        self.msg = wx.TextCtrl(self, value="", style=wx.TE_READONLY | wx.TE_MULTILINE | wx.TE_DONTWRAP)
        font = self.msg.GetFont()
        font.SetFamily(wx.FONTFAMILY_TELETYPE)
        self.msg.SetFont(font)
        vbox.Add(self.msg, 1, wx.EXPAND)
        self.text = StringIO()
        self.autoflush = autoflush

    def write(self, txt):
        self.text.write(txt)
        if self.autoflush:
            self.flush()

    def writable(self):
        return True

    def writelines(self, lines):
        self.text.writelines(lines)
        if self.autoflush:
            self.flush()

    def flush(self):
        self.msg.SetValue(self.text.getvalue())
        self.msg.SetScrollPos(wx.VERTICAL, self.msg.GetScrollRange(wx.VERTICAL))
        self.msg.SetInsertionPoint(-1)
        wx.YieldIfNeeded()
