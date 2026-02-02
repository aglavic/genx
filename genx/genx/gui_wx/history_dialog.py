"""
An interface to show output messages from python logging with convenient fielters.
"""

from typing import List

import wx

from ..model_actions import ActionBlock, ActionHistory, ModelAction
from .message_dialogs import ShowErrorDialog, ShowNotificationDialog


class ActionDisplayDialog(wx.Dialog):
    def __init__(self, parent, action: ModelAction):
        wx.Dialog.__init__(
            self, parent, style=wx.DEFAULT_DIALOG_STYLE | wx.STAY_ON_TOP | wx.RESIZE_BORDER, name=f"GenX action details"
        )
        self.SetTitle(f"Action Details")
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        vbox.Add(wx.StaticText(self, label=f"Name: {action.action_name}"))
        vbox.Add(wx.StaticText(self, label=f"\nDescription:"))
        msg = wx.TextCtrl(self, value=str(action), style=wx.TE_READONLY | wx.TE_MULTILINE | wx.TE_DONTWRAP)
        font = msg.GetFont()
        font.SetFamily(wx.FONTFAMILY_TELETYPE)
        msg.SetFont(font)
        vbox.Add(msg, 1, wx.EXPAND)


class HistoryDialog(wx.Dialog):
    actions: List[ModelAction]
    current_index: int = 0
    changed_actions: ActionBlock

    def __init__(self, parent, history: ActionHistory):
        wx.Dialog.__init__(
            self, parent, style=wx.DEFAULT_DIALOG_STYLE | wx.STAY_ON_TOP | wx.RESIZE_BORDER, name="ActionHistory"
        )
        self.SetTitle("GenX Action History")
        self.changed_actions = None
        self.history = history
        self.actions = history.undo_stack + list(reversed(history.redo_stack))
        self.current_index = len(history.undo_stack)
        self.build_layout()

        self.Bind(wx.EVT_LIST_ITEM_ACTIVATED, self.show_action_details, self.action_list)

        pos = parent.GetPosition()
        size = parent.GetSize()
        self.SetSize(400, int(size.height * 0.6))
        self.SetPosition(wx.Point(pos.x + size.width // 2, pos.y + int(size.height * 0.2)))

    def build_layout(self):
        vbox = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)

        self.action_list = wx.ListCtrl(self, style=wx.LC_REPORT)
        self.action_list.AppendColumn("Action")
        self.append_actions()

        self.resize_columns()
        vbox.Add(self.action_list, 1, wx.EXPAND)

        hbox = wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(hbox)
        cbutton = wx.Button(self, -1, label="Close")
        hbox.Add(cbutton)
        self.Bind(wx.EVT_BUTTON, self.OnDestroy, cbutton)

        ubutton = wx.Button(self, -1, label="Revert selected actions (and remove from history)")
        self.Bind(wx.EVT_BUTTON, self.OnRevert, ubutton)
        hbox.Add(ubutton)

    def OnDestroy(self, evt):
        self.EndModal(wx.ID_CANCEL)

    def OnRevert(self, evt):
        start = self.action_list.GetFirstSelected()
        if 0 <= start < self.current_index:
            success = True
            items = [[start, 1]]
            nitem = self.action_list.GetNextSelected(start)
            while 0 <= nitem <= self.current_index:
                if nitem == sum(items[-1]):
                    items[-1][1] += 1
                else:
                    items.append([nitem, 1])
                nitem = self.action_list.GetNextSelected(nitem)
            changed: List[ModelAction] = []
            for start, length in items:
                try:
                    changed += self.history.remove_actions(self.current_index - start, length).actions
                except Exception as e:
                    ShowErrorDialog(
                        self,
                        f"The actions could not be re-applied:\n{e}\n\n"
                        f"The history was reset to the previous state. "
                        f"You might need to analyze your stepts in detail to revert these actions",
                    )
                    self.actions = self.history.undo_stack + list(reversed(self.history.redo_stack))
                    self.current_index = len(self.history.undo_stack)
                    changed += self.history.redo_stack[-length:]
                    self.action_list.DeleteAllItems()
                    self.append_actions()
                    success = False
                    break
            self.changed_actions = ActionBlock(self.actions[-1].model, changed)
            if success:
                self.EndModal(wx.ID_OK)
        else:
            ShowNotificationDialog(self, "You have to select at least one action to be removed from the undo stack")

    def append_actions(self):
        for i, action in enumerate(self.actions):
            self.action_list.Append((action.action_name,))
            if i >= self.current_index:
                self.action_list.SetItemTextColour(i, wx.Colour(100, 100, 100))

    def show_action_details(self, event):
        action = self.actions[event.GetIndex()]
        ActionDisplayDialog(self, action).ShowModal()

    def resize_columns(self):
        self.action_list.SetColumnWidth(0, wx.LIST_AUTOSIZE)
        self.action_list.EnsureVisible(self.action_list.GetItemCount() - 1)
