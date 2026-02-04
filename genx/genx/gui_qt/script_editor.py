"""
Qt-only script editor widget for GenX.
Designed to be replaceable with a QScintilla implementation later.
"""

from __future__ import annotations

import re
from typing import Optional

from PySide6 import QtCore, QtGui, QtWidgets


class PythonHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, document: QtGui.QTextDocument) -> None:
        super().__init__(document)
        self._rules: list[tuple[QtCore.QRegularExpression, QtGui.QTextCharFormat]] = []

        def _fmt(color: str, bold: bool = False, italic: bool = False) -> QtGui.QTextCharFormat:
            fmt = QtGui.QTextCharFormat()
            fmt.setForeground(QtGui.QColor(color))
            if bold:
                fmt.setFontWeight(QtGui.QFont.Weight.Bold)
            if italic:
                fmt.setFontItalic(True)
            return fmt

        keyword_fmt = _fmt("#2e6da4", bold=True)
        builtin_fmt = _fmt("#6f42c1")
        comment_fmt = _fmt("#6a737d", italic=True)
        string_fmt = _fmt("#22863a")
        number_fmt = _fmt("#b31d28")

        keywords = [
            "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
            "continue", "def", "del", "elif", "else", "except", "finally", "for", "from", "global",
            "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
            "return", "try", "while", "with", "yield",
        ]
        builtins = [
            "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "len", "list",
            "max", "min", "print", "range", "reversed", "set", "str", "sum", "tuple", "zip",
        ]

        for kw in keywords:
            self._rules.append((QtCore.QRegularExpression(rf"\b{kw}\b"), keyword_fmt))
        for bi in builtins:
            self._rules.append((QtCore.QRegularExpression(rf"\b{bi}\b"), builtin_fmt))

        self._rules.append((QtCore.QRegularExpression(r"#.*$"), comment_fmt))
        self._rules.append((QtCore.QRegularExpression(r"\b\d+(\.\d+)?\b"), number_fmt))
        self._rules.append((QtCore.QRegularExpression(r"\"([^\"\\]|\\.)*\""), string_fmt))
        self._rules.append((QtCore.QRegularExpression(r"'([^'\\]|\\.)*'"), string_fmt))

    def highlightBlock(self, text: str) -> None:
        for regex, fmt in self._rules:
            it = regex.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


class GenxScriptEditor(QtWidgets.QPlainTextEdit):
    """
    Qt-only script editor with basic Python syntax highlighting.
    Keep this API stable to allow a later swap to QScintilla.
    """

    modified = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._auto_comp_choose_single = True
        self._auto_comp_ignore_case = False
        self._backspace_unindents = True
        self._show_line_numbers = True
        self._show_indent_guides = True

        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont)
        self.setFont(font)
        self._indent_width = 4 * QtGui.QFontMetrics(font).horizontalAdvance(" ")
        self.setTabStopDistance(self._indent_width)

        self._line_number_area = _LineNumberArea(self)
        self.blockCountChanged.connect(self._update_line_number_area_width)
        self.updateRequest.connect(self._update_line_number_area)
        self.cursorPositionChanged.connect(self._highlight_current_line)
        self._update_line_number_area_width(0)
        self._highlight_current_line()

        self._highlighter = PythonHighlighter(self.document())
        self.textChanged.connect(self._on_text_changed)

        self.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)

    # Compatibility hooks (wx.py.editwindow.EditWindow API)
    def SetBackSpaceUnIndents(self, value: bool) -> None:
        self._backspace_unindents = bool(value)

    def AutoCompSetChooseSingle(self, value: bool) -> None:
        self._auto_comp_choose_single = bool(value)

    def AutoCompSetIgnoreCase(self, value: bool) -> None:
        self._auto_comp_ignore_case = bool(value)

    def EmptyUndoBuffer(self) -> None:
        if hasattr(self.document(), "clearUndoRedoStacks"):
            self.document().clearUndoRedoStacks()
        else:
            self.setUndoRedoEnabled(False)
            self.setUndoRedoEnabled(True)

    def setDisplayLineNumbers(self, _value: bool) -> None:
        self._show_line_numbers = bool(_value)
        self._update_line_number_area_width(0)
        self._line_number_area.setVisible(self._show_line_numbers)

    def setIndentGuides(self, value: bool) -> None:
        self._show_indent_guides = bool(value)
        self.viewport().update()

    # Slots mirroring wx event names (for easy porting)
    @QtCore.Slot(object)
    def on_key_down(self, event) -> None:
        # wx: EVT_KEY_DOWN -> ScriptEditorKeyEvent
        super().keyPressEvent(event)

    @QtCore.Slot()
    def on_modified(self) -> None:
        # wx: wx.stc.EVT_STC_MODIFIED
        self.modified.emit()

    def _on_text_changed(self) -> None:
        self.on_modified()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        key = event.key()
        if key == QtCore.Qt.Key.Key_Tab and not event.modifiers():
            cursor = self.textCursor()
            if cursor.hasSelection():
                self._indent_selection()
                return
            cursor.insertText("    ")
            return
        if key == QtCore.Qt.Key.Key_Backtab:
            cursor = self.textCursor()
            if cursor.hasSelection():
                self._deindent_selection()
                return
            block = cursor.block()
            text = block.text()
            if text.startswith("    "):
                cursor.beginEditBlock()
                cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)
                for _ in range(4):
                    cursor.deleteChar()
                cursor.endEditBlock()
            elif text.startswith("\t"):
                cursor.beginEditBlock()
                cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)
                cursor.deleteChar()
                cursor.endEditBlock()
            return
        if key in (QtCore.Qt.Key.Key_Return, QtCore.Qt.Key.Key_Enter):
            cursor = self.textCursor()
            cursor.beginEditBlock()
            super().keyPressEvent(event)
            block_text = cursor.block().previous().text()
            indent = re.match(r"[ \t]*", block_text).group(0)
            if block_text.rstrip().endswith(":"):
                indent += "    "
            cursor.insertText(indent)
            cursor.endEditBlock()
            self.setTextCursor(cursor)
            return
        if key == QtCore.Qt.Key.Key_Backspace and self._backspace_unindents:
            cursor = self.textCursor()
            if cursor.hasSelection():
                super().keyPressEvent(event)
                return
            block = cursor.block()
            pos_in_block = cursor.position() - block.position()
            if pos_in_block > 0:
                text = block.text()[:pos_in_block]
                if text.isspace():
                    remove = min(4, len(text))
                    for _ in range(remove):
                        cursor.deletePreviousChar()
                    return
        super().keyPressEvent(event)

    def _indent_selection(self) -> None:
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.beginEditBlock()
        cursor.setPosition(start)
        start_block = cursor.blockNumber()
        cursor.setPosition(end)
        end_block = cursor.blockNumber()
        cursor.setPosition(start)
        for _ in range(start_block, end_block + 1):
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)
            cursor.insertText("    ")
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextBlock)
        cursor.endEditBlock()

    def _deindent_selection(self) -> None:
        cursor = self.textCursor()
        start = cursor.selectionStart()
        end = cursor.selectionEnd()
        cursor.beginEditBlock()
        cursor.setPosition(start)
        start_block = cursor.blockNumber()
        cursor.setPosition(end)
        end_block = cursor.blockNumber()
        cursor.setPosition(start)
        for _ in range(start_block, end_block + 1):
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.StartOfBlock)
            block_text = cursor.block().text()
            if block_text.startswith("    "):
                for _ in range(4):
                    cursor.deleteChar()
            elif block_text.startswith("\t"):
                cursor.deleteChar()
            cursor.movePosition(QtGui.QTextCursor.MoveOperation.NextBlock)
        cursor.endEditBlock()

    def line_number_area_width(self) -> int:
        if not self._show_line_numbers:
            return 0
        digits = len(str(max(1, self.blockCount())))
        space = 3 + self.fontMetrics().horizontalAdvance("9") * digits
        return space

    def _update_line_number_area_width(self, _block_count: int) -> None:
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def _update_line_number_area(self, rect: QtCore.QRect, dy: int) -> None:
        if not self._show_line_numbers:
            return
        if dy:
            self._line_number_area.scroll(0, dy)
        else:
            self._line_number_area.update(0, rect.y(), self._line_number_area.width(), rect.height())
        if rect.contains(self.viewport().rect()):
            self._update_line_number_area_width(0)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        cr = self.contentsRect()
        self._line_number_area.setGeometry(QtCore.QRect(cr.left(), cr.top(), self.line_number_area_width(), cr.height()))

    def _highlight_current_line(self) -> None:
        if self.isReadOnly():
            return
        extra_selections = []
        selection = QtWidgets.QTextEdit.ExtraSelection()
        line_color = QtGui.QColor(240, 240, 255)
        selection.format.setBackground(line_color)
        selection.format.setProperty(QtGui.QTextFormat.Property.FullWidthSelection, True)
        selection.cursor = self.textCursor()
        selection.cursor.clearSelection()
        extra_selections.append(selection)
        self.setExtraSelections(extra_selections)

    def _line_number_area_paint_event(self, event: QtGui.QPaintEvent) -> None:
        if not self._show_line_numbers:
            return
        painter = QtGui.QPainter(self._line_number_area)
        painter.fillRect(event.rect(), QtGui.QColor(245, 245, 245))

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        top = int(self.blockBoundingGeometry(block).translated(self.contentOffset()).top())
        bottom = top + int(self.blockBoundingRect(block).height())

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                number = str(block_number + 1)
                painter.setPen(QtGui.QColor(120, 120, 120))
                painter.drawText(
                    0,
                    top,
                    self._line_number_area.width() - 4,
                    self.fontMetrics().height(),
                    QtCore.Qt.AlignmentFlag.AlignRight,
                    number,
                )
            block = block.next()
            if not block.isValid():
                break
            top = bottom
            bottom = top + int(self.blockBoundingRect(block).height())
            block_number += 1

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        if not self._show_indent_guides:
            return
        painter = QtGui.QPainter(self.viewport())
        painter.setPen(QtGui.QPen(QtGui.QColor(230, 230, 230)))
        first_block = self.firstVisibleBlock()
        block = first_block
        offset = self.contentOffset()
        viewport_rect = self.viewport().rect()
        while block.isValid():
            block_rect = self.blockBoundingGeometry(block).translated(offset)
            if block_rect.top() > viewport_rect.bottom():
                break
            text = block.text()
            if text:
                # count leading spaces as indent levels (4 spaces per level)
                leading = len(text) - len(text.lstrip(" "))
                guide_levels = leading // 4
                for level in range(1, guide_levels + 1):
                    x = int(level * self._indent_width + self.contentOffset().x())
                    painter.drawLine(x, int(block_rect.top()), x, int(block_rect.bottom()))
            block = block.next()


class _LineNumberArea(QtWidgets.QWidget):
    def __init__(self, editor: GenxScriptEditor) -> None:
        super().__init__(editor)
        self._editor = editor

    def sizeHint(self) -> QtCore.QSize:
        return QtCore.QSize(self._editor.line_number_area_width(), 0)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        self._editor._line_number_area_paint_event(event)
