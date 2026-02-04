"""
Qt port of history_dialog.
"""

from typing import List

from PySide6 import QtCore, QtGui, QtWidgets

from ..model_actions import ActionBlock, ActionHistory, ModelAction
from .message_dialogs import ShowErrorDialog, ShowNotificationDialog


class ActionDisplayDialog(QtWidgets.QDialog):
    def __init__(self, parent, action: ModelAction):
        super().__init__(parent)
        self.setWindowTitle("Action Details")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"Name: {action.action_name}"))
        layout.addWidget(QtWidgets.QLabel("\nDescription:"))
        msg = QtWidgets.QPlainTextEdit(self)
        msg.setReadOnly(True)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        msg.setFont(font)
        msg.setPlainText(str(action))
        layout.addWidget(msg, 1)


class HistoryDialog(QtWidgets.QDialog):
    actions: List[ModelAction]
    current_index: int = 0
    changed_actions: ActionBlock

    def __init__(self, parent, history: ActionHistory):
        super().__init__(parent)
        self.setWindowTitle("GenX Action History")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)
        self.changed_actions = None
        self.history = history
        self.actions = history.undo_stack + list(reversed(history.redo_stack))
        self.current_index = len(history.undo_stack)
        self.build_layout()

        pos = parent.frameGeometry().topLeft()
        size = parent.size()
        self.resize(400, int(size.height() * 0.6))
        self.move(pos.x() + size.width() // 2, pos.y() + int(size.height() * 0.2))

    def build_layout(self):
        vbox = QtWidgets.QVBoxLayout(self)

        self.action_list = QtWidgets.QTableWidget(0, 1, self)
        self.action_list.setHorizontalHeaderLabels(["Action"])
        self.action_list.horizontalHeader().setStretchLastSection(True)
        self.action_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.action_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.action_list.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.append_actions()
        self.action_list.cellDoubleClicked.connect(self.show_action_details)
        vbox.addWidget(self.action_list, 1)

        hbox = QtWidgets.QHBoxLayout()
        vbox.addLayout(hbox)
        cbutton = QtWidgets.QPushButton("Close", self)
        cbutton.clicked.connect(self.reject)
        hbox.addWidget(cbutton)
        ubutton = QtWidgets.QPushButton("Revert selected actions (and remove from history)", self)
        ubutton.clicked.connect(self.OnRevert)
        hbox.addWidget(ubutton)

    def OnRevert(self):
        selected = [idx.row() for idx in self.action_list.selectionModel().selectedRows()]
        if not selected or min(selected) >= self.current_index:
            ShowNotificationDialog(self, "You have to select at least one action to be removed from the undo stack")
            return

        selected.sort()
        success = True
        items: List[list[int]] = [[selected[0], 1]]
        for idx in selected[1:]:
            if idx == sum(items[-1]):
                items[-1][1] += 1
            else:
                items.append([idx, 1])

        changed: List[ModelAction] = []
        for start, length in items:
            try:
                changed += self.history.remove_actions(self.current_index - start, length).actions
            except Exception as e:
                ShowErrorDialog(
                    self,
                    f"The actions could not be re-applied:\n{e}\n\n"
                    "The history was reset to the previous state. "
                    "You might need to analyze your steps in detail to revert these actions",
                )
                self.actions = self.history.undo_stack + list(reversed(self.history.redo_stack))
                self.current_index = len(self.history.undo_stack)
                changed += self.history.redo_stack[-length:]
                self.action_list.setRowCount(0)
                self.append_actions()
                success = False
                break

        self.changed_actions = ActionBlock(self.actions[-1].model, changed)
        if success:
            self.accept()

    def append_actions(self):
        for i, action in enumerate(self.actions):
            row = self.action_list.rowCount()
            self.action_list.insertRow(row)
            item = QtWidgets.QTableWidgetItem(action.action_name)
            if i >= self.current_index:
                item.setForeground(QtGui.QBrush(QtGui.QColor(100, 100, 100)))
            self.action_list.setItem(row, 0, item)

    def show_action_details(self, row: int, _column: int):
        action = self.actions[row]
        ActionDisplayDialog(self, action).exec()
