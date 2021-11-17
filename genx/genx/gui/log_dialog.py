"""
An interface to show output messages from python logging with convenient fielters.
"""

import wx
from traceback import format_exception
from typing import List, Tuple
from wx.lib.mixins.listctrl import ListCtrlAutoWidthMixin
import logging

from.custom_events import log_message_event, EVT_LOG_MESSAGE

class GuiHandler(logging.Handler, wx.EvtHandler):
    def __init__(self, level=logging.DEBUG):
        logging.Handler.__init__(self, level=level)
        wx.EvtHandler.__init__(self)
        formatter = logging.Formatter('[%(levelname)s] - %(asctime)s - %(filename)s:%(lineno)i:%(funcName)s %(message)s',
                                      datefmt='%H:%M:%S')
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord):
        fmt_message = self.format(record)
        evt=log_message_event(record=record, fmt_message=fmt_message)
        wx.QueueEvent(self, evt)

class RecordDisplayDialog(wx.Dialog):
    def __init__(self, parent, record: logging.LogRecord):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE|wx.STAY_ON_TOP|wx.RESIZE_BORDER,
                           name=f'GenX log message')
        self.SetTitle(f'GenX {record.levelname} message')
        vbox=wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        vbox.Add(wx.StaticText(self, label=f'Severity: {record.levelname}'))
        vbox.Add(wx.StaticText(self, label=f'Process: {record.processName}'))
        vbox.Add(wx.StaticText(self, label=f'Thread: {record.threadName}'))
        vbox.Add(wx.StaticText(self, label=f'Module: {record.module}'))
        vbox.Add(wx.StaticText(self, label=f'Line: {record.lineno}'))
        vbox.Add(wx.StaticText(self, label=f'\nMessage:'))
        msg=wx.TextCtrl(self, value=record.msg, style=wx.TE_READONLY|wx.TE_MULTILINE|wx.TE_DONTWRAP)
        font=msg.GetFont()
        font.SetFamily(wx.FONTFAMILY_TELETYPE)
        msg.SetFont(font)
        vbox.Add(msg, 0, wx.EXPAND)
        if record.exc_info:
            vbox.Add(wx.StaticText(self, label='\nError:'))
            exc_message=''.join(format_exception(*record.exc_info))
            if record.stack_info:
                exc_message+='\n'+record.stack_info
            msg=wx.TextCtrl(self, value=exc_message, style=wx.TE_READONLY|wx.TE_MULTILINE|wx.TE_DONTWRAP)
            font=msg.GetFont()
            font.SetFamily(wx.FONTFAMILY_TELETYPE)
            msg.SetFont(font)
            vbox.Add(msg, 1, wx.EXPAND)

class LoggingDialog(wx.Dialog):
    looged_events: List[Tuple[logging.LogRecord, str]]

    def __init__(self, parent=None):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE|wx.STAY_ON_TOP|wx.RESIZE_BORDER,
                           name='GenX Log')
        self.SetTitle('GenX Logging History')
        self.handler = GuiHandler()
        logging.getLogger().addHandler(self.handler)
        self.build_layout()
        self.looged_events=[]
        self.log_level=logging.INFO

        self.Bind(wx.EVT_CLOSE, self.close_window)
        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.show_event_message, self.log_list)
        self.handler.Bind(EVT_LOG_MESSAGE, self.OnMessage)

        pos=parent.GetPosition()
        size=parent.GetSize()
        self.SetSize(400, size.height)
        self.SetPosition(wx.Point(pos.x+size.width, pos.y))

    def close_window(self, event):
        # make sure the handler is removed on close of dialog
        event.Skip()
        logging.getLogger().removeHandler(self.handler)

    def update_loglevel(self, level):
        self.log_level=level
        self.log_list.DeleteAllItems()
        for record, message in self.looged_events:
            self.append_event(record)
        self.resize_columns()

    def show_event_message(self, event):
        records=[ri for ri,mi in self.looged_events if ri.levelno>=self.log_level]
        record=records[event.GetIndex()]
        RecordDisplayDialog(self, record).Show()

    def build_layout(self):
        vbox=wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        topbox=wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(topbox)
        rb=wx.RadioButton(self, -1, 'DEBUG')
        self.Bind(wx.EVT_RADIOBUTTON, lambda evt: self.update_loglevel(logging.DEBUG), rb)
        topbox.Add(rb)
        rb=wx.RadioButton(self, -1, 'INFO')
        rb.SetValue(True)
        self.Bind(wx.EVT_RADIOBUTTON, lambda evt: self.update_loglevel(logging.INFO), rb)
        topbox.Add(rb)
        rb=wx.RadioButton(self, -1, 'WARNING')
        self.Bind(wx.EVT_RADIOBUTTON, lambda evt: self.update_loglevel(logging.WARNING), rb)
        topbox.Add(rb)
        rb=wx.RadioButton(self, -1, 'ERROR')
        self.Bind(wx.EVT_RADIOBUTTON, lambda evt: self.update_loglevel(logging.ERROR), rb)
        topbox.Add(rb)

        self.log_list=wx.ListCtrl(self, style=wx.LC_REPORT|wx.LC_SINGLE_SEL)
        self.log_list.AppendColumn('Time')
        self.log_list.AppendColumn('Level')
        self.log_list.AppendColumn('Message')
        # self.log_list.setResizeColumn(3)
        vbox.Add(self.log_list, 1, wx.EXPAND)

    def append_event(self, record: logging.LogRecord):
        if record.levelno>=self.log_level:
            self.log_list.Append((record.asctime, record.levelname, record.msg.splitlines()[0]))

    def resize_columns(self):
        self.log_list.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.log_list.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        self.log_list.SetColumnWidth(2, wx.LIST_AUTOSIZE)
        self.log_list.EnsureVisible(self.log_list.GetItemCount()-1)

    def OnMessage(self, evt):
        self.looged_events.append((evt.record, evt.fmt_message))
        self.append_event(evt.record)
        self.resize_columns()