# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'parametergrid.ui'
##
## Created by: Qt User Interface Compiler version 6.9.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QAction, QBrush, QColor, QConicalGradient,
    QCursor, QFont, QFontDatabase, QGradient,
    QIcon, QImage, QKeySequence, QLinearGradient,
    QPainter, QPalette, QPixmap, QRadialGradient,
    QTransform)
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QHeaderView, QSizePolicy,
    QTableWidget, QTableWidgetItem, QToolBar, QWidget)

class Ui_ParameterGrid(object):
    def setupUi(self, ParameterGrid):
        if not ParameterGrid.objectName():
            ParameterGrid.setObjectName(u"ParameterGrid")
        self.actionAddRow = QAction(ParameterGrid)
        self.actionAddRow.setObjectName(u"actionAddRow")
        icon = QIcon()
        icon.addFile(u":/main_gui/add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon.addFile(u":/main_gui/add.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionAddRow.setIcon(icon)
        self.actionDeleteRow = QAction(ParameterGrid)
        self.actionDeleteRow.setObjectName(u"actionDeleteRow")
        icon1 = QIcon()
        icon1.addFile(u":/main_gui/delete.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon1.addFile(u":/main_gui/delete.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionDeleteRow.setIcon(icon1)
        self.actionMoveUp = QAction(ParameterGrid)
        self.actionMoveUp.setObjectName(u"actionMoveUp")
        icon2 = QIcon()
        icon2.addFile(u":/main_gui/move_up.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon2.addFile(u":/main_gui/move_up.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionMoveUp.setIcon(icon2)
        self.actionMoveDown = QAction(ParameterGrid)
        self.actionMoveDown.setObjectName(u"actionMoveDown")
        icon3 = QIcon()
        icon3.addFile(u":/main_gui/move_down.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon3.addFile(u":/main_gui/move_down.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionMoveDown.setIcon(icon3)
        self.separator1 = QAction(ParameterGrid)
        self.separator1.setObjectName(u"separator1")
        self.separator1.setSeparator(True)
        self.actionSort = QAction(ParameterGrid)
        self.actionSort.setObjectName(u"actionSort")
        icon4 = QIcon()
        icon4.addFile(u":/main_gui/sort.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon4.addFile(u":/main_gui/sort.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionSort.setIcon(icon4)
        self.actionSortName = QAction(ParameterGrid)
        self.actionSortName.setObjectName(u"actionSortName")
        icon5 = QIcon()
        icon5.addFile(u":/main_gui/sort2.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon5.addFile(u":/main_gui/sort2.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionSortName.setIcon(icon5)
        self.separator2 = QAction(ParameterGrid)
        self.separator2.setObjectName(u"separator2")
        self.separator2.setSeparator(True)
        self.actionProjectFom = QAction(ParameterGrid)
        self.actionProjectFom.setObjectName(u"actionProjectFom")
        icon6 = QIcon()
        icon6.addFile(u":/main_gui/par_proj.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon6.addFile(u":/main_gui/par_proj.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionProjectFom.setIcon(icon6)
        self.actionScanFom = QAction(ParameterGrid)
        self.actionScanFom.setObjectName(u"actionScanFom")
        icon7 = QIcon()
        icon7.addFile(u":/main_gui/par_scan.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon7.addFile(u":/main_gui/par_scan.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionScanFom.setIcon(icon7)
        self.horizontalLayout = QHBoxLayout(ParameterGrid)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.toolbar = QToolBar(ParameterGrid)
        self.toolbar.setObjectName(u"toolbar")
        self.toolbar.setOrientation(Qt.Vertical)

        self.horizontalLayout.addWidget(self.toolbar)

        self.parameterTable = QTableWidget(ParameterGrid)
        self.parameterTable.setObjectName(u"parameterTable")

        self.horizontalLayout.addWidget(self.parameterTable)


        self.toolbar.addAction(self.actionAddRow)
        self.toolbar.addAction(self.actionDeleteRow)
        self.toolbar.addAction(self.actionMoveUp)
        self.toolbar.addAction(self.actionMoveDown)
        self.toolbar.addAction(self.separator1)
        self.toolbar.addAction(self.actionSort)
        self.toolbar.addAction(self.actionSortName)
        self.toolbar.addAction(self.separator2)
        self.toolbar.addAction(self.actionProjectFom)
        self.toolbar.addAction(self.actionScanFom)

        self.retranslateUi(ParameterGrid)

        QMetaObject.connectSlotsByName(ParameterGrid)
    # setupUi

    def retranslateUi(self, ParameterGrid):
        self.actionAddRow.setText(QCoreApplication.translate("ParameterGrid", u"Add row", None))
#if QT_CONFIG(tooltip)
        self.actionAddRow.setToolTip(QCoreApplication.translate("ParameterGrid", u"Insert a new parameter row", None))
#endif // QT_CONFIG(tooltip)
        self.actionDeleteRow.setText(QCoreApplication.translate("ParameterGrid", u"Delete row", None))
#if QT_CONFIG(tooltip)
        self.actionDeleteRow.setToolTip(QCoreApplication.translate("ParameterGrid", u"Delete selected parameter row(s)", None))
#endif // QT_CONFIG(tooltip)
        self.actionMoveUp.setText(QCoreApplication.translate("ParameterGrid", u"Move up", None))
#if QT_CONFIG(tooltip)
        self.actionMoveUp.setToolTip(QCoreApplication.translate("ParameterGrid", u"Move selected row up", None))
#endif // QT_CONFIG(tooltip)
        self.actionMoveDown.setText(QCoreApplication.translate("ParameterGrid", u"Move down", None))
#if QT_CONFIG(tooltip)
        self.actionMoveDown.setToolTip(QCoreApplication.translate("ParameterGrid", u"Move selected row down", None))
#endif // QT_CONFIG(tooltip)
        self.actionSort.setText(QCoreApplication.translate("ParameterGrid", u"Sort", None))
#if QT_CONFIG(tooltip)
        self.actionSort.setToolTip(QCoreApplication.translate("ParameterGrid", u"Sort parameters by attribute", None))
#endif // QT_CONFIG(tooltip)
        self.actionSortName.setText(QCoreApplication.translate("ParameterGrid", u"Sort name", None))
#if QT_CONFIG(tooltip)
        self.actionSortName.setToolTip(QCoreApplication.translate("ParameterGrid", u"Sort parameters by object name", None))
#endif // QT_CONFIG(tooltip)
        self.actionProjectFom.setText(QCoreApplication.translate("ParameterGrid", u"Project FOM", None))
#if QT_CONFIG(tooltip)
        self.actionProjectFom.setToolTip(QCoreApplication.translate("ParameterGrid", u"Project FOM for selected parameter", None))
#endif // QT_CONFIG(tooltip)
        self.actionScanFom.setText(QCoreApplication.translate("ParameterGrid", u"Scan FOM", None))
#if QT_CONFIG(tooltip)
        self.actionScanFom.setToolTip(QCoreApplication.translate("ParameterGrid", u"Scan FOM for selected parameter", None))
#endif // QT_CONFIG(tooltip)
        pass
    # retranslateUi

