"""
Qt logging dialog to display log records and inspect details.
Ported from the wx implementation.
"""

import logging
from traceback import format_exception
from typing import List, Optional

from PySide6 import QtCore, QtGui, QtWidgets


class _LogSignalEmitter(QtCore.QObject):
    message = QtCore.Signal(object, str)  # record, formatted message


class GuiHandler(logging.Handler):
    def __init__(self, level: int = logging.DEBUG) -> None:
        super().__init__(level=level)
        self.emitter = _LogSignalEmitter()
        formatter = logging.Formatter(
            "[%(levelname)s] - %(asctime)s - %(filename)s:%(lineno)i:%(funcName)s %(message)s",
            datefmt="%H:%M:%S",
        )
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            fmt_message = self.format(record)
        except Exception:
            return
        # Ensure GUI updates happen on the Qt main thread.
        QtCore.QTimer.singleShot(0, lambda r=record, m=fmt_message: self.emitter.message.emit(r, m))


class RecordDisplayDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget], record: logging.LogRecord) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"GenX {record.levelname} message")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel(f"Severity: {record.levelname}"))
        layout.addWidget(QtWidgets.QLabel(f"Process: {record.processName}"))
        layout.addWidget(QtWidgets.QLabel(f"Thread: {record.threadName}"))
        layout.addWidget(QtWidgets.QLabel(f"Module: {record.module}"))
        layout.addWidget(QtWidgets.QLabel(f"Line: {record.lineno}"))
        layout.addWidget(QtWidgets.QLabel("\nMessage:"))

        msg = QtWidgets.QTextEdit(self)
        msg.setReadOnly(True)
        msg.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
        msg.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
        msg.setPlainText(record.getMessage())
        layout.addWidget(msg, 1)

        if record.exc_info:
            layout.addWidget(QtWidgets.QLabel("\nError:"))
            exc_message = "".join(format_exception(*record.exc_info))
            if record.stack_info:
                exc_message += "\n" + record.stack_info
            exc = QtWidgets.QTextEdit(self)
            exc.setReadOnly(True)
            exc.setLineWrapMode(QtWidgets.QTextEdit.LineWrapMode.NoWrap)
            exc.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
            exc.setPlainText(exc_message)
            layout.addWidget(exc, 1)


class LoggingDialog(QtWidgets.QDialog):
    logged_events: List[logging.LogRecord]

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("GenX Logging History")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowStaysOnTopHint)

        self.handler = GuiHandler()
        logging.getLogger().addHandler(self.handler)

        self.logged_events = []
        self.log_level = logging.INFO
        self._build_layout()

        self.handler.emitter.message.connect(self._on_message)

        if parent is not None:
            parent_geom = parent.frameGeometry()
            self.resize(400, parent_geom.height())
            self.move(parent_geom.topRight())

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        topbox = QtWidgets.QHBoxLayout()
        layout.addLayout(topbox)

        self._level_group = QtWidgets.QButtonGroup(self)
        for level, checked in [
            (logging.DEBUG, False),
            (logging.INFO, True),
            (logging.WARNING, False),
            (logging.ERROR, False),
        ]:
            rb = QtWidgets.QRadioButton(logging.getLevelName(level))
            rb.setChecked(checked)
            self._level_group.addButton(rb, level)
            rb.toggled.connect(self._on_level_toggled)
            topbox.addWidget(rb)

        self.log_list = QtWidgets.QTableWidget(0, 3, self)
        self.log_list.setHorizontalHeaderLabels(["Time", "Level", "Message"])
        self.log_list.horizontalHeader().setStretchLastSection(True)
        self.log_list.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.log_list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.log_list.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.log_list.cellDoubleClicked.connect(self._show_event_message)
        layout.addWidget(self.log_list, 1)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        logging.getLogger().removeHandler(self.handler)
        super().closeEvent(event)

    @property
    def filtered_events(self) -> List[logging.LogRecord]:
        return [ri for ri in self.logged_events if ri.levelno >= self.log_level]

    def _on_level_toggled(self, checked: bool) -> None:
        if not checked:
            return
        level = self._level_group.checkedId()
        self.update_loglevel(level)

    def update_loglevel(self, level: int) -> None:
        self.log_level = level
        self.log_list.setRowCount(0)
        for record in self.filtered_events:
            self._append_event(record)
        self._resize_columns()

    def _show_event_message(self, row: int, _column: int) -> None:
        item = self.log_list.item(row, 0)
        if item is None:
            return
        record_index = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if record_index is None:
            return
        record = self.logged_events[int(record_index)]
        RecordDisplayDialog(self, record).show()

    def _format_time(self, record: logging.LogRecord) -> str:
        formatter = self.handler.formatter
        if isinstance(formatter, logging.Formatter):
            return formatter.formatTime(record, formatter.datefmt)
        return ""

    def _append_event(self, record: logging.LogRecord, new: bool = False) -> None:
        if record.levelno < self.log_level:
            return
        if new:
            prev_event_evts = [ri for ri in self.logged_events if ri.levelno >= self.log_level]
            if (
                len(prev_event_evts) > 1
                and prev_event_evts[-2].getMessage() == record.getMessage()
                and prev_event_evts[-2].lineno == record.lineno
            ):
                # Don't repeat the same error to speed up display.
                return

        time_text = getattr(record, "asctime", None) or self._format_time(record)
        message = record.getMessage().splitlines()[0] if record.getMessage() else ""
        row = self.log_list.rowCount()
        self.log_list.insertRow(row)
        item_time = QtWidgets.QTableWidgetItem(time_text)
        item_time.setData(QtCore.Qt.ItemDataRole.UserRole, self.logged_events.index(record))
        self.log_list.setItem(row, 0, item_time)
        self.log_list.setItem(row, 1, QtWidgets.QTableWidgetItem(record.levelname))
        self.log_list.setItem(row, 2, QtWidgets.QTableWidgetItem(message))

    def _resize_columns(self) -> None:
        self.log_list.resizeColumnsToContents()
        items = self.log_list.rowCount()
        if items > 0:
            self.log_list.scrollToItem(self.log_list.item(items - 1, 0))

    def _on_message(self, record: logging.LogRecord, _fmt_message: str) -> None:
        self.logged_events.append(record)
        self._append_event(record, new=True)
        self._resize_columns()
