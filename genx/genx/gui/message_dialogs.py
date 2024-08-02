import sys

from logging import debug, error, info, warning

import wx


def ShowQuestionDialog(frame, message, title="Question?", yes_no=False):
    exc_info = sys.exc_info()
    debug(message, exc_info=exc_info[0] and exc_info)
    if yes_no:
        flags = wx.YES | wx.NO | wx.YES_DEFAULT | wx.ICON_QUESTION
    else:
        flags = wx.OK | wx.CANCEL | wx.OK_DEFAULT | wx.ICON_QUESTION
    dlg = wx.MessageDialog(frame, message, title, flags)
    result = dlg.ShowModal() in [wx.ID_OK, wx.ID_YES]
    dlg.Destroy()
    return result


def ShowNotificationDialog(frame, message, title="Information"):
    exc_info = sys.exc_info()
    info(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_INFORMATION)
    dlg.ShowModal()
    dlg.Destroy()


def ShowWarningDialog(frame, message, title="Warning"):
    exc_info = sys.exc_info()
    warning(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()


def ShowErrorDialog(frame, message, title="ERROR"):
    exc_info = sys.exc_info()
    error(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()
