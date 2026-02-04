import sys

from logging import debug, error, info, warning

from PySide6 import QtWidgets


def ShowQuestionDialog(frame, message, title="Question?", yes_no=False):
    exc_info = sys.exc_info()
    debug(message, exc_info=exc_info[0] and exc_info)
    if yes_no:
        buttons = QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        default = QtWidgets.QMessageBox.StandardButton.Yes
    else:
        buttons = QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel
        default = QtWidgets.QMessageBox.StandardButton.Ok
    result = QtWidgets.QMessageBox.question(frame, title, message, buttons, default)
    return result in (QtWidgets.QMessageBox.StandardButton.Ok, QtWidgets.QMessageBox.StandardButton.Yes)


def ShowNotificationDialog(frame, message, title="Information"):
    exc_info = sys.exc_info()
    info(message, exc_info=exc_info[0] and exc_info)
    QtWidgets.QMessageBox.information(frame, title, message)


def ShowWarningDialog(frame, message, title="Warning"):
    exc_info = sys.exc_info()
    warning(message, exc_info=exc_info[0] and exc_info)
    QtWidgets.QMessageBox.warning(frame, title, message)


def ShowErrorDialog(frame, message, title="ERROR"):
    exc_info = sys.exc_info()
    error(message, exc_info=exc_info[0] and exc_info)
    QtWidgets.QMessageBox.critical(frame, title, message)
