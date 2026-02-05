# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'datalist.ui'
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
from PySide6.QtWidgets import (QApplication, QSizePolicy, QToolBar, QVBoxLayout,
    QWidget)

class Ui_DataListControl(object):
    def setupUi(self, DataListControl):
        if not DataListControl.objectName():
            DataListControl.setObjectName(u"DataListControl")
        self.actionAddData = QAction(DataListControl)
        self.actionAddData.setObjectName(u"actionAddData")
        icon = QIcon()
        icon.addFile(u":/main_gui/add.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon.addFile(u":/main_gui/add.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionAddData.setIcon(icon)
        self.actionImportData = QAction(DataListControl)
        self.actionImportData.setObjectName(u"actionImportData")
        icon1 = QIcon()
        icon1.addFile(u":/main_gui/open.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon1.addFile(u":/main_gui/open.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionImportData.setIcon(icon1)
        self.actionAddSimulation = QAction(DataListControl)
        self.actionAddSimulation.setObjectName(u"actionAddSimulation")
        icon2 = QIcon()
        icon2.addFile(u":/main_gui/add_simulation.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon2.addFile(u":/main_gui/add_simulation.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionAddSimulation.setIcon(icon2)
        self.actionDataInfo = QAction(DataListControl)
        self.actionDataInfo.setObjectName(u"actionDataInfo")
        icon3 = QIcon()
        icon3.addFile(u":/main_gui/info.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon3.addFile(u":/main_gui/info.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionDataInfo.setIcon(icon3)
        self.separator1 = QAction(DataListControl)
        self.separator1.setObjectName(u"separator1")
        self.separator1.setSeparator(True)
        self.actionMoveUp = QAction(DataListControl)
        self.actionMoveUp.setObjectName(u"actionMoveUp")
        icon4 = QIcon()
        icon4.addFile(u":/main_gui/move_up.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon4.addFile(u":/main_gui/move_up.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionMoveUp.setIcon(icon4)
        self.actionMoveDown = QAction(DataListControl)
        self.actionMoveDown.setObjectName(u"actionMoveDown")
        icon5 = QIcon()
        icon5.addFile(u":/main_gui/move_down.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon5.addFile(u":/main_gui/move_down.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionMoveDown.setIcon(icon5)
        self.actionDelete = QAction(DataListControl)
        self.actionDelete.setObjectName(u"actionDelete")
        icon6 = QIcon()
        icon6.addFile(u":/main_gui/delete.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon6.addFile(u":/main_gui/delete.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionDelete.setIcon(icon6)
        self.separator2 = QAction(DataListControl)
        self.separator2.setObjectName(u"separator2")
        self.separator2.setSeparator(True)
        self.actionPlotting = QAction(DataListControl)
        self.actionPlotting.setObjectName(u"actionPlotting")
        icon7 = QIcon()
        icon7.addFile(u":/main_gui/plotting.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon7.addFile(u":/main_gui/plotting.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionPlotting.setIcon(icon7)
        self.actionCalc = QAction(DataListControl)
        self.actionCalc.setObjectName(u"actionCalc")
        icon8 = QIcon()
        icon8.addFile(u":/main_gui/calc.png", QSize(), QIcon.Mode.Normal, QIcon.State.Off)
        icon8.addFile(u":/main_gui/calc.png", QSize(), QIcon.Mode.Normal, QIcon.State.On)
        self.actionCalc.setIcon(icon8)
        self.verticalLayout = QVBoxLayout(DataListControl)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.toolbar = QToolBar(DataListControl)
        self.toolbar.setObjectName(u"toolbar")

        self.verticalLayout.addWidget(self.toolbar)

        self.listContainer = QWidget(DataListControl)
        self.listContainer.setObjectName(u"listContainer")
        self.listLayout = QVBoxLayout(self.listContainer)
        self.listLayout.setObjectName(u"listLayout")
        self.listLayout.setContentsMargins(0, 0, 0, 0)

        self.verticalLayout.addWidget(self.listContainer)


        self.toolbar.addAction(self.actionAddData)
        self.toolbar.addAction(self.actionImportData)
        self.toolbar.addAction(self.actionAddSimulation)
        self.toolbar.addAction(self.actionDataInfo)
        self.toolbar.addAction(self.separator1)
        self.toolbar.addAction(self.actionMoveUp)
        self.toolbar.addAction(self.actionMoveDown)
        self.toolbar.addAction(self.actionDelete)
        self.toolbar.addAction(self.separator2)
        self.toolbar.addAction(self.actionPlotting)
        self.toolbar.addAction(self.actionCalc)

        self.retranslateUi(DataListControl)

        QMetaObject.connectSlotsByName(DataListControl)
    # setupUi

    def retranslateUi(self, DataListControl):
        self.actionAddData.setText(QCoreApplication.translate("DataListControl", u"Add data set", None))
#if QT_CONFIG(tooltip)
        self.actionAddData.setToolTip(QCoreApplication.translate("DataListControl", u"Insert empty data set", None))
#endif // QT_CONFIG(tooltip)
        self.actionImportData.setText(QCoreApplication.translate("DataListControl", u"Import data set", None))
#if QT_CONFIG(tooltip)
        self.actionImportData.setToolTip(QCoreApplication.translate("DataListControl", u"Import data into selected data set", None))
#endif // QT_CONFIG(tooltip)
        self.actionAddSimulation.setText(QCoreApplication.translate("DataListControl", u"Add simulation set", None))
#if QT_CONFIG(tooltip)
        self.actionAddSimulation.setToolTip(QCoreApplication.translate("DataListControl", u"Insert a data set for simulation", None))
#endif // QT_CONFIG(tooltip)
        self.actionDataInfo.setText(QCoreApplication.translate("DataListControl", u"Data info", None))
#if QT_CONFIG(tooltip)
        self.actionDataInfo.setToolTip(QCoreApplication.translate("DataListControl", u"Show the meta data information for the selected dataset", None))
#endif // QT_CONFIG(tooltip)
        self.actionMoveUp.setText(QCoreApplication.translate("DataListControl", u"Move up", None))
#if QT_CONFIG(tooltip)
        self.actionMoveUp.setToolTip(QCoreApplication.translate("DataListControl", u"Move selected data set(s) up", None))
#endif // QT_CONFIG(tooltip)
        self.actionMoveDown.setText(QCoreApplication.translate("DataListControl", u"Move down", None))
#if QT_CONFIG(tooltip)
        self.actionMoveDown.setToolTip(QCoreApplication.translate("DataListControl", u"Move selected data set(s) down", None))
#endif // QT_CONFIG(tooltip)
        self.actionDelete.setText(QCoreApplication.translate("DataListControl", u"Delete data set", None))
#if QT_CONFIG(tooltip)
        self.actionDelete.setToolTip(QCoreApplication.translate("DataListControl", u"Delete selected data set", None))
#endif // QT_CONFIG(tooltip)
        self.actionPlotting.setText(QCoreApplication.translate("DataListControl", u"Plot settings", None))
#if QT_CONFIG(tooltip)
        self.actionPlotting.setToolTip(QCoreApplication.translate("DataListControl", u"Plot settings", None))
#endif // QT_CONFIG(tooltip)
        self.actionCalc.setText(QCoreApplication.translate("DataListControl", u"Calculations", None))
#if QT_CONFIG(tooltip)
        self.actionCalc.setToolTip(QCoreApplication.translate("DataListControl", u"Calculation on selected data set(s)", None))
#endif // QT_CONFIG(tooltip)
        pass
    # retranslateUi

