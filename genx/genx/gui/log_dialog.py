"""
An interface to show output messages from python logging with convenient fielters.
"""

import wx
from traceback import format_exception
from typing import List, Tuple
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
        msg=wx.TextCtrl(self, value=record.message, style=wx.TE_READONLY|wx.TE_MULTILINE|wx.TE_DONTWRAP)
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
    logged_events: List[logging.LogRecord]

    def __init__(self, parent=None):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE|wx.STAY_ON_TOP|wx.RESIZE_BORDER,
                           name='GenX Log')
        self.SetTitle('GenX Logging History')
        self.handler = GuiHandler()
        logging.getLogger().addHandler(self.handler)
        self.build_layout()
        self.logged_events=[]
        self.log_level=logging.INFO

        self.Bind(wx.EVT_CLOSE, self.close_window)
        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.show_event_message, self.log_list)
        self.handler.Bind(EVT_LOG_MESSAGE, self.OnMessage)

        pos=parent.GetPosition()
        size=parent.GetSize()
        self.SetSize(400, size.height)
        self.SetPosition(wx.Point(pos.x+size.width, pos.y))

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

    def close_window(self, event):
        # make sure the handler is removed on close of dialog
        event.Skip()
        logging.getLogger().removeHandler(self.handler)

    @property
    def filtered_events(self):
        return [ri for ri in self.logged_events if ri.levelno>=self.log_level]

    def update_loglevel(self, level):
        self.log_level=level
        self.log_list.DeleteAllItems()
        for i, record in enumerate(self.filtered_events):
            if (i+1)<len(self.logged_events) and \
                    self.logged_events[i+1].message==record.message and \
                    self.logged_events[i+1].lineno==record.lineno:
                continue
            self.append_event(record)
        self.resize_columns()

    def show_event_message(self, event: wx.ListEvent):
        record=self.logged_events[event.GetData()]
        RecordDisplayDialog(self, record).Show()

    def append_event(self, record: logging.LogRecord, new=False):
        if record.levelno>=self.log_level:
            if new:
                prev_event_evts=[ri for ri in self.logged_events if ri.levelno>=self.log_level]
                if len(prev_event_evts)>1 and \
                    prev_event_evts[-2].message==record.message and \
                    prev_event_evts[-2].lineno==record.lineno:
                    # don't repeat the same error to speed up diaplsy
                    return
            self.log_list.Append((record.asctime, record.levelname, record.message.splitlines()[0]))
            last = self.log_list.GetItemCount()-1
            self.log_list.SetItemData(last, self.logged_events.index(record))

    def resize_columns(self):
        self.log_list.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.log_list.SetColumnWidth(1, wx.LIST_AUTOSIZE)
        self.log_list.SetColumnWidth(2, wx.LIST_AUTOSIZE)
        self.log_list.EnsureVisible(self.log_list.GetItemCount()-1)

    def OnMessage(self, evt):
        self.logged_events.append(evt.record)
        self.append_event(evt.record, new=True)
        self.resize_columns()