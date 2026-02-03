def ShowInfoDialog(frame, message, title="Information"):
    if wx is None:
        print(message)
        return
    else:
        exc_info = sys.exc_info()
        info(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_INFORMATION)
    dlg.ShowModal()
    dlg.Destroy()


def ShowWarningDialog(frame, message, title="Warning"):
    if wx is None:
        print(message)
        return
    else:
        exc_info = sys.exc_info()
        warning(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()


def ShowErrorDialog(frame, message, title="ERROR"):
    if wx is None:
        print(message)
        return
    else:
        exc_info = sys.exc_info()
        error(message, exc_info=exc_info[0] and exc_info)
    dlg = wx.MessageDialog(frame, message, title, wx.OK | wx.ICON_ERROR)
    dlg.ShowModal()
    dlg.Destroy()


def ShowQuestionDialog(frame, message, title="Question"):
    if wx is None:
        result = input(message + " [y/n]").strip().lower() in ["y", "yes"]
        return result
    dlg = wx.MessageDialog(frame, message, title, wx.YES_NO | wx.ICON_QUESTION)
    result = dlg.ShowModal() == wx.ID_YES
    dlg.Destroy()
    return result
