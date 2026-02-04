# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'data_grid_panel.ui'
##
## Created by: Qt User Interface Compiler version 6.9.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QSizePolicy,
    QSpacerItem, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget)

class Ui_DataGridPanel(object):
    def setupUi(self, DataGridPanel):
        if not DataGridPanel.objectName():
            DataGridPanel.setObjectName(u"DataGridPanel")
        self.verticalLayout = QVBoxLayout(DataGridPanel)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.headerLayout = QHBoxLayout()
        self.headerLayout.setObjectName(u"headerLayout")
        self.labelDataSet = QLabel(DataGridPanel)
        self.labelDataSet.setObjectName(u"labelDataSet")

        self.headerLayout.addWidget(self.labelDataSet)

        self.dataGridChoice = QComboBox(DataGridPanel)
        self.dataGridChoice.setObjectName(u"dataGridChoice")

        self.headerLayout.addWidget(self.dataGridChoice)

        self.headerSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.headerLayout.addItem(self.headerSpacer)


        self.verticalLayout.addLayout(self.headerLayout)

        self.separatorLine = QFrame(DataGridPanel)
        self.separatorLine.setObjectName(u"separatorLine")
        self.separatorLine.setFrameShape(QFrame.HLine)
        self.separatorLine.setFrameShadow(QFrame.Sunken)

        self.verticalLayout.addWidget(self.separatorLine)

        self.dataGrid = QTableWidget(DataGridPanel)
        self.dataGrid.setObjectName(u"dataGrid")
        self.dataGrid.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.dataGrid.setSelectionMode(QAbstractItemView.ContiguousSelection)
        self.dataGrid.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.dataGrid.setColumnCount(6)
        self.dataGrid.setRowCount(0)

        self.verticalLayout.addWidget(self.dataGrid)


        self.retranslateUi(DataGridPanel)
        self.dataGridChoice.activated.connect(DataGridPanel.on_data_grid_choice_changed)
        self.dataGridChoice.currentIndexChanged.connect(DataGridPanel.on_data_grid_choice_changed)

        QMetaObject.connectSlotsByName(DataGridPanel)
    # setupUi

    def retranslateUi(self, DataGridPanel):
        self.labelDataSet.setText(QCoreApplication.translate("DataGridPanel", u"  Data set: ", None))
        self.dataGrid.setHorizontalHeaderLabels([
            QCoreApplication.translate("DataGridPanel", u"x_raw", None),
            QCoreApplication.translate("DataGridPanel", u"y_raw", None),
            QCoreApplication.translate("DataGridPanel", u"Error_raw", None),
            QCoreApplication.translate("DataGridPanel", u"x", None),
            QCoreApplication.translate("DataGridPanel", u"y", None),
            QCoreApplication.translate("DataGridPanel", u"Error", None)])
        pass
    # retranslateUi

