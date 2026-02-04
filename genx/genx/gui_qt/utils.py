import sys

from logging import error, info, warning

from PySide6 import QtWidgets


def ShowInfoDialog(frame, message, title="Information"):
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


def ShowQuestionDialog(frame, message, title="Question"):
    result = QtWidgets.QMessageBox.question(
        frame,
        title,
        message,
        QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
    )
    return result == QtWidgets.QMessageBox.StandardButton.Yes
