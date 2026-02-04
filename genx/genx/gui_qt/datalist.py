"""
Qt port of the wx-based datalist module.
Implements list view + dialogs for data handling.
"""

from dataclasses import dataclass

from PySide6 import QtCore, QtGui, QtWidgets

from .. import data
from ..core.config import BaseConfig, Configurable
from .data_loader import PluginController as DataLoaderController
from .datalist_ui import Ui_DataListControl
from .message_dialogs import ShowErrorDialog, ShowNotificationDialog, ShowQuestionDialog, ShowWarningDialog
from .metadata_dialog import MetaDataDialog


class DataController:
    """
    Interface layer class between the ListModel and the DataList class.
    """

    data: data.DataList

    def __init__(self, data_list):
        self.data = data_list

    def get_data(self):
        return self.data

    def get_column_headers(self):
        return ["Name", "Show", "Use", "Errors"]

    def get_count(self):
        return self.data.get_len()

    def has_data(self, index):
        if index < 0 or index >= self.get_count():
            return False
        return self.data[index].has_data()

    def get_item_text(self, item, col):
        bool_output = {True: "Yes", False: "No"}
        if col == 0:
            return self.data.get_name(item)
        if col == 1:
            return bool_output[self.data[item].show]
        if col == 2:
            return bool_output[self.data.get_use(item)]
        if col == 3:
            return bool_output[self.data.get_use_error(item)]
        return ""

    def set_data(self, data_list):
        self.data = data_list

    def set_name(self, pos, name):
        self.data.set_name(pos, name)

    def move_up(self, pos):
        self.data.move_up(pos)

    def move_down(self, pos):
        self.data.move_down(pos)

    def add_item(self):
        self.data.add_new()

    def delete_item(self, pos):
        self.data.delete_item(pos)

    def get_colors(self):
        colors = []
        for data_set in self.data:
            dc = data_set.data_color
            sc = data_set.sim_color
            colors.append(
                (
                    (int(dc[0] * 255), int(dc[1] * 255), int(dc[2] * 255)),
                    (int(sc[0] * 255), int(sc[1] * 255), int(sc[2] * 255)),
                )
            )
        return colors

    def get_items_plotsettings(self, pos):
        sim_list = [self.data[i].get_sim_plot_items() for i in pos]
        data_list = [self.data[i].get_data_plot_items() for i in pos]
        return sim_list, data_list

    def test_commands(self, command, pos):
        result = ""
        for i in pos:
            result = self.data[i].try_commands(command)
            if result != "":
                break
        return result

    def run_commands(self, command, pos):
        result = ""
        for i in pos:
            try:
                self.data[i].set_commands(command)
                self.data[i].run_command()
            except Exception as exc:
                result += f"Error occured for data set {i}: {exc}"
                break
        return result

    def compare_sim_y_length(self, pos):
        result = True
        for index in pos:
            if self.data[index].y.shape != self.data[index].y_sim.shape:
                result = False
                break
        return result

    def get_items_commands(self, pos):
        return [self.data[i].get_commands() for i in pos]

    def get_items_names(self):
        return [self.data.get_name(i) for i in range(self.get_count())]

    def set_items_plotsettings(self, pos, sim_list, data_list):
        lpos = list(range(len(sim_list)))
        [self.data[i].set_sim_plot_items(sim_list[j]) for i, j in zip(pos, lpos)]
        [self.data[i].set_data_plot_items(data_list[j]) for i, j in zip(pos, lpos)]

    def show_data(self, positions):
        self.data.show_items(positions)

    def toggle_show_data(self, positions):
        [self.data.toggle_show(pos) for pos in positions]

    def toggle_use_data(self, positions):
        [self.data.toggle_use(pos) for pos in positions]

    def toggle_use_error(self, positions):
        [self.data.toggle_use_error(pos) for pos in positions]


@dataclass
class DataListEvent:
    data: data.DataList
    data_changed: bool = True
    new_data: bool = False
    new_model: bool = False
    description: str = ""
    data_moved: bool = False
    position: int | list[int] | None = None
    up: bool = False
    deleted: bool = False
    name_change: bool = False


@dataclass
class VDataListConfig(BaseConfig):
    section = "data handling"
    toggle_show: bool = True


@dataclass
class DataCommandConfig(BaseConfig):
    section = "data commands"
    names: str = "A Example;Default;Simulation;Sustematic Errors"
    x_commands: str = "x+33;x;arange(0.01, 6, 0.01);x"
    y_commands: str = "y/1e5;y;arange(0.01, 6, 0.01)*0;y"
    e_commands: str = "e/2.;e;arange(0.01, 6, 0.01)*0;rms(e, fpe(1.0, 0.02), 0.01*dydx())"

class DataListModel(QtCore.QAbstractTableModel):
    def __init__(self, data_controller: DataController, parent=None):
        super().__init__(parent)
        self.data_cont = data_controller
        self._headers = self.data_cont.get_column_headers()

    def rowCount(self, parent=QtCore.QModelIndex()):
        return self.data_cont.get_count()

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self._headers)

    def data(self, index, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row, col = index.row(), index.column()
        if role == QtCore.Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return self.data_cont.get_item_text(row, col)
            return ""
        if role == QtCore.Qt.ItemDataRole.CheckStateRole and col in (1, 2, 3):
            if col == 1:
                return QtCore.Qt.CheckState.Checked if self.data_cont.data[row].show else QtCore.Qt.CheckState.Unchecked
            if col == 2:
                return QtCore.Qt.CheckState.Checked if self.data_cont.data.get_use(row) else QtCore.Qt.CheckState.Unchecked
            if col == 3:
                return (
                    QtCore.Qt.CheckState.Checked
                    if self.data_cont.data.get_use_error(row)
                    else QtCore.Qt.CheckState.Unchecked
                )
        if role == QtCore.Qt.ItemDataRole.DecorationRole and col == 0:
            colors = self.data_cont.get_colors()
            if 0 <= row < len(colors):
                data_color, _sim_color = colors[row]
                pix = QtGui.QPixmap(16, 16)
                pix.fill(QtCore.Qt.GlobalColor.transparent)
                painter = QtGui.QPainter(pix)
                try:
                    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
                    painter.setBrush(QtGui.QBrush(QtGui.QColor(*data_color)))
                    painter.setPen(QtGui.QPen(QtGui.QColor(*data_color)))
                    painter.drawEllipse(1, 1, 14, 14)
                finally:
                    painter.end()
                return QtGui.QIcon(pix)
        return None

    def headerData(self, section, orientation, role=QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            if 0 <= section < len(self._headers):
                return self._headers[section]
        return None

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemFlag.ItemIsEnabled
        flags = QtCore.Qt.ItemFlag.ItemIsEnabled | QtCore.Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        if index.column() in (1, 2, 3):
            flags |= QtCore.Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role=QtCore.Qt.ItemDataRole.EditRole):
        if not index.isValid():
            return False
        row, col = index.row(), index.column()
        if col == 0 and role == QtCore.Qt.ItemDataRole.EditRole:
            self.data_cont.set_name(row, value)
            self.dataChanged.emit(index, index, [role])
            return True
        if col in (1, 2, 3) and role == QtCore.Qt.ItemDataRole.CheckStateRole:
            if col == 1:
                self.data_cont.data.toggle_show(row)
            elif col == 2:
                self.data_cont.data.toggle_use(row)
            else:
                self.data_cont.data.toggle_use_error(row)
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def refresh(self):
        self.beginResetModel()
        self.endResetModel()

class VirtualDataList(QtWidgets.QTableView, Configurable):
    """
    The list view for the data.

    Original wx event names:
    - EVT_DATA_LIST
    - EVT_UPDATE_PLOTSETTINGS
    """

    data_list_event = QtCore.Signal(object)
    update_plotsettings = QtCore.Signal(object)

    opt: VDataListConfig

    def __init__(self, parent, data_controller: DataController, status_text=None):
        QtWidgets.QTableView.__init__(self, parent)
        Configurable.__init__(self)

        self.data_cont = data_controller
        self.parent = parent
        self.status_text = status_text
        self.data_loader = None
        self.data_loader_cont = DataLoaderController(self)
        self.show_indices = []

        self.model = DataListModel(self.data_cont, self)
        self.setModel(self.model)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
            | QtWidgets.QAbstractItemView.EditTrigger.SelectedClicked
        )
        self.horizontalHeader().setStretchLastSection(False)
        self.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.Stretch)
        for col in range(1, self.model.columnCount()):
            self.horizontalHeader().setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.setAlternatingRowColors(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QtWidgets.QAbstractItemView.DragDropMode.DropOnly)

        self.selectionModel().selectionChanged.connect(self.OnSelectionChanged)
        self.ReadConfig()

    def SetShowToggle(self, toggle):
        self.opt.toggle_show = bool(toggle)
        self.WriteConfig()

    def OnSelectionChanged(self, _selected=None, _deselected=None):
        if not self.opt.toggle_show:
            indices = self._GetSelectedItems()
            indices.sort()
            if indices != self.show_indices:
                self.data_cont.show_data(indices)
                self._UpdateData("Show data set flag toggled", data_changed=True)
        self.show_indices = self._GetSelectedItems()

    def _GetSelectedItems(self):
        return [idx.row() for idx in self.selectionModel().selectedRows()]

    def _UpdateData(
        self,
        desc,
        data_changed=True,
        new_data=False,
        position=None,
        moved=False,
        direction_up=True,
        deleted=False,
        name_change=False,
        new_model=False,
    ):
        evt = DataListEvent(
            data=self.data_cont.get_data(),
            data_changed=data_changed,
            new_data=new_data,
            new_model=new_model,
            description=desc,
            data_moved=moved,
            position=position,
            up=direction_up,
            deleted=deleted,
            name_change=name_change,
        )
        self.data_list_event.emit(evt)

    def _UpdateImageList(self):
        """
        Qt equivalent of wx image list refresh: trigger icon repaint.
        """
        self.model.refresh()

    def SetItemCount(self, _count):
        """
        Qt equivalent of wx.ListCtrl.SetItemCount: refresh the model after the
        underlying data list size has changed.
        """
        self.model.refresh()

    def _CheckSelected(self, indices):
        if len(indices) == 0:
            ShowNotificationDialog(self, "At least one data set has to be selected")
            return False
        return True

    def DeleteItem(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        result = ShowQuestionDialog(self, f"Remove {len(indices)} dataset(s) ?", title="Remove?")
        if result:
            indices.sort(reverse=True)
            [self.data_cont.delete_item(index) for index in indices]
            self.model.refresh()
            self._UpdateData("Data Deleted", deleted=True, position=indices)

    def AddItem(self):
        self.data_cont.add_item()
        self.model.refresh()
        self._UpdateData("Item added", data_changed=True, new_data=True)

    def MoveItemUp(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        indices.sort()
        if indices[0] != 0:
            [self.data_cont.move_up(index) for index in indices]
            self.model.refresh()
            self._UpdateData("Item moved", data_changed=False, moved=True, direction_up=True, position=indices)
        else:
            ShowNotificationDialog(self, "The first dataset can not be moved up")

    def MoveItemDown(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        indices.sort(reverse=True)
        if indices[0] != self.data_cont.get_count() - 1:
            [self.data_cont.move_down(index) for index in indices]
            self.model.refresh()
            self._UpdateData("Item moved", data_changed=False, moved=True, direction_up=False, position=indices)
        else:
            ShowNotificationDialog(self, "The last dataset can not be moved down")

    def LoadData(self):
        self.data_loader.SetData(self.data_cont.get_data())
        if self.data_loader.LoadDataFile(self._GetSelectedItems()):
            self._UpdateData("New data added", new_data=True)

    def load_from_files(self, files, do_update=True):
        offset = self.data_cont.get_count()
        while offset > 0 and not self.data_cont.has_data(offset - 1):
            offset -= 1
        i = 0
        for fi in files:
            for di in range(min(self.data_loader.CountDatasets(fi), 25)):
                if self.data_cont.get_count() < (i + offset + 1):
                    self.data_cont.add_item()
                    self.model.refresh()
                self.data_loader.LoadDataset(self.data_cont.get_data()[i + offset], fi, data_id=di)
                i += 1

        if do_update:
            self._UpdateData("New data added", new_data=True)
            self.model.refresh()
        return True

    def ChangeDataLoader(self):
        self.data_loader_cont.ShowDialog()

    def OnNewModel(self, model):
        self.data_cont.set_data(model.get_data())
        self.model.refresh()
        self._UpdateData("Data from model loaded", data_changed=True, new_data=True, new_model=True)
        self.ReadConfig()
        self.data_loader_cont.load_default()

    def OnPlotSettings(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        sim_list, data_list = self.data_cont.get_items_plotsettings(indices)
        dia = PlotSettingsDialog(self, sim_list[0], data_list[0])

        def on_apply(sim_par, data_par):
            sim_list_new = [sim_par for _ in indices]
            data_list_new = [data_par for _ in indices]
            self.data_cont.set_items_plotsettings(indices, sim_list_new, data_list_new)
            self.update_plotsettings.emit(DataListEvent(self.data_cont.get_data(), data_changed=True))

        dia.SetApplyFunc(on_apply)
        dia.exec()

    def OnShowData(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        self.data_cont.toggle_show_data(indices)
        self.model.refresh()
        self._UpdateData("Show data set flag toggled", data_changed=True)

    def OnUseData(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        self.data_cont.toggle_use_data(indices)
        self.model.refresh()
        self._UpdateData("Use data set flag toggled", data_changed=True)

    def OnUseError(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        self.data_cont.toggle_use_error(indices)
        self.model.refresh()
        self._UpdateData("Use error flag toggled", data_changed=True)

    def OnCalcEdit(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        command_list = self.data_cont.get_items_commands(indices)
        data_names = self.data_cont.get_items_names()
        data_commands = [self.data_cont.get_items_commands([i])[0] for i in range(self.data_cont.get_count())]
        cfg = DataCommandConfig()
        predef_names = cfg.names.split(";")
        predef_commands = [
            {"x": x, "y": y, "e": e}
            for x, y, e in zip(cfg.x_commands.split(";"), cfg.y_commands.split(";"), cfg.e_commands.split(";"))
        ]
        dia = CalcDialog(self, command_list[0], data_names, data_commands, predef_names, predef_commands)
        dia.SetCommandRunner(lambda command: self.data_cont.run_commands(command, indices))
        dia.SetCommandTester(lambda command: self.data_cont.test_commands(command, indices))
        dia.exec()

    def OnImportSettings(self):
        if self.data_loader is None:
            return
        if hasattr(self.data_loader, "SettingsDialog"):
            self.data_loader.SettingsDialog()

    def CreateSimData(self):
        wiz = CreateSimDataWizard(self)
        if wiz.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            xstr, ystr, names = wiz.GetValues()
            for name in names:
                self.data_cont.add_item()
                self.data_cont.set_name(-1, name)
                self.data_cont.run_commands({"x": xstr, "y": ystr, "e": ystr}, [-1])
                self.model.refresh()
                self._UpdateData("Item added", data_changed=True, new_data=True)

    def ShowInfo(self):
        indices = self._GetSelectedItems()
        if not self._CheckSelected(indices):
            return
        dia = MetaDataDialog(self, self.data_cont.get_data(), selected=indices[0])
        dia.exec()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return
        files = [u.toLocalFile() for u in urls if u.isLocalFile()]
        if files:
            self.load_from_files(files, do_update=True)

    @QtCore.Slot(object)
    def OnDataListEvent(self, event):
        pass

    @QtCore.Slot(object)
    def OnUpdatePlotSettings(self, event):
        pass


class DataListControl(QtWidgets.QWidget):
    """
    The Control window for the whole Data list including a small toolbar.
    """

    def __init__(self, parent, status_text=None):
        super().__init__(parent)
        self.ui = Ui_DataListControl()
        self.ui.setupUi(self)
        mydata = data.DataList()
        self.data_cont = DataController(mydata)
        self.list_ctrl = VirtualDataList(self, self.data_cont, status_text=status_text)

        self.ui.listLayout.addWidget(self.list_ctrl, 1)

    @QtCore.Slot()
    def on_actionImportData_triggered(self):
        self.list_ctrl.LoadData()

    @QtCore.Slot()
    def on_actionAddData_triggered(self):
        self.list_ctrl.AddItem()

    @QtCore.Slot()
    def on_actionAddSimulation_triggered(self):
        self.list_ctrl.CreateSimData()

    @QtCore.Slot()
    def on_actionDataInfo_triggered(self):
        self.list_ctrl.ShowInfo()

    @QtCore.Slot()
    def on_actionDelete_triggered(self):
        self.list_ctrl.DeleteItem()

    @QtCore.Slot()
    def on_actionMoveUp_triggered(self):
        self.list_ctrl.MoveItemUp()

    @QtCore.Slot()
    def on_actionMoveDown_triggered(self):
        self.list_ctrl.MoveItemDown()

    @QtCore.Slot()
    def on_actionPlotting_triggered(self):
        self.list_ctrl.OnPlotSettings()

    @QtCore.Slot()
    def on_actionCalc_triggered(self):
        self.list_ctrl.OnCalcEdit()

    def eh_external_new_model(self, model):
        self.list_ctrl.OnNewModel(model)

    def DataLoaderSettingsDialog(self):
        self.list_ctrl.ChangeDataLoader()

class PlotSettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent, sim_pars, data_pars):
        super().__init__(parent)
        self.setWindowTitle("Plot Settings")

        def func(sim_par, data_par):
            pass

        self.apply_func = func

        grid = QtWidgets.QGridLayout()
        col_labels = ["Color", "Line type", "Thickness", "Symbol", "Size"]
        row_labels = ["Simulation:", "Data:"]

        for idx, item in enumerate(col_labels):
            grid.addWidget(QtWidgets.QLabel(item), 0, idx + 1)
        for idx, item in enumerate(row_labels):
            grid.addWidget(QtWidgets.QLabel(item), idx + 1, 0)

        self.sim_colorbutton = QtWidgets.QPushButton()
        self.data_colorbutton = QtWidgets.QPushButton()
        self._set_button_color(self.sim_colorbutton, sim_pars.get("color"))
        self._set_button_color(self.data_colorbutton, data_pars.get("color"))
        self.sim_colorbutton.clicked.connect(lambda: self._pick_color(self.sim_colorbutton))
        self.data_colorbutton.clicked.connect(lambda: self._pick_color(self.data_colorbutton))
        grid.addWidget(self.sim_colorbutton, 1, 1)
        grid.addWidget(self.data_colorbutton, 2, 1)

        self.line_type = ["", "-", ":", "--", ".-", None]
        line_type = ["No line", "full", "dotted", "dashed", "dash dotted", " "]
        self.sim_linetype_choice = QtWidgets.QComboBox()
        self.sim_linetype_choice.addItems(line_type)
        self.sim_linetype_choice.setCurrentIndex(self._get_first_match(sim_pars["linetype"], self.line_type))
        self.data_linetype_choice = QtWidgets.QComboBox()
        self.data_linetype_choice.addItems(line_type)
        self.data_linetype_choice.setCurrentIndex(self._get_first_match(data_pars["linetype"], self.line_type))
        grid.addWidget(self.sim_linetype_choice, 1, 2)
        grid.addWidget(self.data_linetype_choice, 2, 2)

        self.sim_linethick_ctrl = QtWidgets.QSpinBox()
        self.data_linethick_ctrl = QtWidgets.QSpinBox()
        if sim_pars["linethickness"] is not None:
            self.sim_linethick_ctrl.setRange(1, 20)
            self.sim_linethick_ctrl.setValue(sim_pars["linethickness"])
        else:
            self.sim_linethick_ctrl.setRange(-1, 20)
            self.sim_linethick_ctrl.setValue(-1)
        if data_pars["linethickness"] is not None:
            self.data_linethick_ctrl.setRange(1, 20)
            self.data_linethick_ctrl.setValue(data_pars["linethickness"])
        else:
            self.data_linethick_ctrl.setRange(-1, 20)
            self.data_linethick_ctrl.setValue(-1)
        grid.addWidget(self.sim_linethick_ctrl, 1, 3)
        grid.addWidget(self.data_linethick_ctrl, 2, 3)

        self.symbol_type = ["", "s", "o", ".", "d", "<", None]
        symbol_type = ["No symbol", "squares", "circles", "dots", "diamonds", "triangle", " "]
        self.sim_symboltype_choice = QtWidgets.QComboBox()
        self.sim_symboltype_choice.addItems(symbol_type)
        self.sim_symboltype_choice.setCurrentIndex(self._get_first_match(sim_pars["symbol"], self.symbol_type))
        self.data_symboltype_choice = QtWidgets.QComboBox()
        self.data_symboltype_choice.addItems(symbol_type)
        self.data_symboltype_choice.setCurrentIndex(self._get_first_match(data_pars["symbol"], self.symbol_type))
        grid.addWidget(self.sim_symboltype_choice, 1, 4)
        grid.addWidget(self.data_symboltype_choice, 2, 4)

        self.sim_symbolsize_ctrl = QtWidgets.QSpinBox()
        self.data_symbolsize_ctrl = QtWidgets.QSpinBox()
        if sim_pars["symbolsize"] is not None:
            self.sim_symbolsize_ctrl.setRange(1, 20)
            self.sim_symbolsize_ctrl.setValue(sim_pars["symbolsize"])
        else:
            self.sim_symbolsize_ctrl.setRange(1, 20)
            self.sim_symbolsize_ctrl.setValue(-1)
        if data_pars["symbolsize"] is not None:
            self.data_symbolsize_ctrl.setRange(1, 20)
            self.data_symbolsize_ctrl.setValue(data_pars["symbolsize"])
        else:
            self.data_symbolsize_ctrl.setRange(0, 20)
            self.data_symbolsize_ctrl.setValue(-1)
        grid.addWidget(self.sim_symbolsize_ctrl, 1, 5)
        grid.addWidget(self.data_symbolsize_ctrl, 2, 5)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.on_ok)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self.OnApply)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(grid)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line)
        layout.addWidget(buttons)

    def _set_button_color(self, button, color):
        if color is None:
            color = (255, 255, 255, 255)
        qcolor = QtGui.QColor(*color)
        button.setStyleSheet(f"background-color: {qcolor.name()};")
        button._current_color = qcolor

    def _pick_color(self, button):
        color = QtWidgets.QColorDialog.getColor(button._current_color, self)
        if color.isValid():
            button._current_color = color
            button.setStyleSheet(f"background-color: {color.name()};")

    def _get_first_match(self, item, list1):
        position = 0
        for i in range(len(list1)):
            if list1[i] == item:
                position = i
                break
        return position

    def SetApplyFunc(self, func):
        self.apply_func = func

    def on_ok(self):
        self.OnApply()
        self.accept()

    def OnApply(self):
        sim_par = self.GetSimPar()
        data_par = self.GetDataPar()
        self.apply_func(sim_par, data_par)

    def GetSimPar(self):
        color = self.sim_colorbutton._current_color
        if color == QtGui.QColor(255, 255, 255):
            color = None
        symbolsize = self.sim_symbolsize_ctrl.value()
        if symbolsize < 0:
            symbolsize = None
        linethickness = self.sim_linethick_ctrl.value()
        if linethickness < 0:
            linethickness = None
        return {
            "color": None if color is None else color.getRgb(),
            "symbol": self.symbol_type[self.sim_symboltype_choice.currentIndex()],
            "symbolsize": symbolsize,
            "linetype": self.line_type[self.sim_linetype_choice.currentIndex()],
            "linethickness": linethickness,
        }

    def GetDataPar(self):
        color = self.data_colorbutton._current_color
        if color == QtGui.QColor(255, 255, 255):
            color = None
        symbolsize = self.sim_symbolsize_ctrl.value()
        if symbolsize < 0:
            symbolsize = None
        linethickness = self.sim_linethick_ctrl.value()
        if linethickness < 0:
            linethickness = None
        return {
            "color": None if color is None else color.getRgb(),
            "symbol": self.symbol_type[self.data_symboltype_choice.currentIndex()],
            "symbolsize": self.data_symbolsize_ctrl.value(),
            "linetype": self.line_type[self.data_linetype_choice.currentIndex()],
            "linethickness": self.data_linethick_ctrl.value(),
        }


class CalcDialog(QtWidgets.QDialog):
    def __init__(self, parent, commands, data_names, data_commands, predef_commands_names=None, predef_commands=None):
        super().__init__(parent)
        self.setWindowTitle("Data Calculations")

        self.command_runner = None
        self.command_tester = None
        self.data_list = data_names
        self.data_commands = data_commands

        if predef_commands and predef_commands_names:
            self.predef_list = predef_commands_names
            self.predef_commands = predef_commands
        else:
            self.predef_list = ["Example", "Default"]
            self.predef_commands = [{"x": "x*2", "y": "y/1000.0", "e": "e/1000"}, {"x": "x", "y": "y", "e": "e"}]

        self.predef_choice = QtWidgets.QComboBox()
        self.predef_choice.addItems(self.predef_list)
        self.data_choice = QtWidgets.QComboBox()
        self.data_choice.addItems(self.data_list)
        self.predef_choice.currentIndexChanged.connect(self.OnPredefChoice)
        self.data_choice.currentIndexChanged.connect(self.OnDataChoice)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Predefined:"))
        top.addWidget(self.predef_choice)
        top.addSpacing(20)
        top.addWidget(QtWidgets.QLabel("Data set:"))
        top.addWidget(self.data_choice)

        grid = QtWidgets.QGridLayout()
        command_names_standard = ["x", "y", "e"]
        self.command_ctrl = {}
        row = 0
        for name in command_names_standard:
            if name in commands:
                grid.addWidget(QtWidgets.QLabel(f"{name} = "), row, 0)
                ctrl = QtWidgets.QLineEdit(commands[name])
                self.command_ctrl[name] = ctrl
                grid.addWidget(ctrl, row, 1)
                row += 1
        command_names = sorted(commands.keys())
        for name in command_names:
            if name in command_names_standard:
                continue
            grid.addWidget(QtWidgets.QLabel(f"{name} = "), row, 0)
            ctrl = QtWidgets.QLineEdit(commands[name])
            self.command_ctrl[name] = ctrl
            grid.addWidget(ctrl, row, 1)
            row += 1

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.OnClickExecute)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Apply).clicked.connect(self.OnClickExecute)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(top)
        layout.addLayout(grid)
        line = QtWidgets.QFrame()
        line.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        line.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(line)
        layout.addWidget(buttons)

    def SetCommandRunner(self, function):
        self.command_runner = function

    def SetCommandTester(self, function):
        self.command_tester = function

    def OnPredefChoice(self, _event=None):
        item = self.predef_choice.currentIndex()
        self.command_ctrl["x"].setText(self.predef_commands[item]["x"])
        self.command_ctrl["y"].setText(self.predef_commands[item]["y"])
        self.command_ctrl["e"].setText(self.predef_commands[item]["e"])

    def OnDataChoice(self, _event=None):
        item = self.data_choice.currentIndex()
        failed = []
        for name in self.command_ctrl:
            val = self.command_ctrl[name].text()
            try:
                val = self.data_commands[item][name]
            except KeyError:
                failed.append(name)
            self.command_ctrl[name].setText(val)
        if len(failed) > 0:
            ShowWarningDialog(
                self,
                "The data operations for the following members of the data set could not be copied: "
                + " ,".join(failed),
                "Copy failed",
            )

    def OnClickExecute(self, _event=None):
        current_command = {name: ctrl.text() for name, ctrl in self.command_ctrl.items()}
        if self.command_tester and self.command_runner:
            result = self.command_tester(current_command)
            if result == "":
                result = self.command_runner(current_command)
                if result != "":
                    result = (
                        "There is an error that the command tester did not catch please give the following "
                        "information to the developer:\n\n"
                        + result
                    )
                    ShowErrorDialog(self, result, "Error in GenX")
            else:
                result = "There is an error in the typed expression.\n" + result
                ShowWarningDialog(self, result, "Expression not correct")
        if self.sender() and isinstance(self.sender(), QtWidgets.QDialogButtonBox):
            self.accept()


class TitledPage(QtWidgets.QWizardPage):
    def __init__(self, title):
        super().__init__()
        self.setTitle(title)


class CreateSimDataWizard(QtWidgets.QWizard):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Create Simulation Data Sets")
        self.pages = []
        self.min_val = None
        self.max_val = None

        page1 = TitledPage("X-values")
        page2 = TitledPage("Number of data sets")
        page3 = TitledPage("Data set names")

        step_types = ["const", "log"]
        form1 = QtWidgets.QFormLayout(page1)
        self.minCtrl = QtWidgets.QLineEdit("0.0")
        self.maxCtrl = QtWidgets.QLineEdit("1.0")
        self.stepChoice = QtWidgets.QComboBox()
        self.stepChoice.addItems(step_types)
        self.stepCtrl = QtWidgets.QSpinBox()
        self.stepCtrl.setRange(1, 100000)
        self.stepCtrl.setValue(100)
        form1.addRow("Start", self.minCtrl)
        form1.addRow("Stop", self.maxCtrl)
        form1.addRow("Step type", self.stepChoice)
        form1.addRow("Num steps", self.stepCtrl)

        form2 = QtWidgets.QFormLayout(page2)
        self.setsCtrl = QtWidgets.QSpinBox()
        self.setsCtrl.setRange(1, 1000)
        self.setsCtrl.setValue(1)
        form2.addRow("Data sets", self.setsCtrl)

        vbox3 = QtWidgets.QVBoxLayout(page3)
        vbox3.addWidget(QtWidgets.QLabel("Change the name of the data sets"))
        self.scrollArea = QtWidgets.QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.nameContainer = QtWidgets.QWidget()
        self.nameSizer = QtWidgets.QVBoxLayout(self.nameContainer)
        self.scrollArea.setWidget(self.nameContainer)
        vbox3.addWidget(self.scrollArea)
        self.nameCtrls = []

        self.add_page(page1)
        self.add_page(page2)
        self.add_page(page3)
        self.currentIdChanged.connect(self.on_page_changed)

    def add_page(self, page):
        self.addPage(page)
        self.pages.append(page)

    def on_page_changed(self, page_id):
        if page_id == 1:
            if not self.min_max_values_valid():
                self.back()
        if page_id == 2:
            for item in self.nameCtrls:
                item.deleteLater()
            self.nameCtrls = []
            for i in range(self.setsCtrl.value()):
                ctrl = QtWidgets.QLineEdit(f"Sim{i}")
                self.nameSizer.addWidget(ctrl)
                self.nameCtrls.append(ctrl)

    def min_max_values_valid(self):
        try:
            self.min_val = float(eval(self.minCtrl.text()))
        except Exception:
            self.min_val = None
            ShowWarningDialog(self, "The minimum value can not be evaluated to a numerical value")
            return False
        try:
            self.max_val = float(eval(self.maxCtrl.text()))
        except Exception:
            self.max_val = None
            ShowWarningDialog(self, "The maximum value can not be evaluated to a numerical value")
            return False
        if self.min_val < 1e-20 and self.stepChoice.currentText() == "log":
            ShowWarningDialog(self, "The minimum value have to be larger than 1e-20 when using log step size")
            return False
        return True

    def GetValues(self):
        if self.stepChoice.currentText() == "log":
            xstr = "logspace(log10(%s), log10(%s), %d)" % (
                self.minCtrl.text(),
                self.maxCtrl.text(),
                self.stepCtrl.value(),
            )
        else:
            xstr = "linspace(%s, %s, %d)" % (self.minCtrl.text(), self.maxCtrl.text(), self.stepCtrl.value())
        ystr = "zeros(%d)*nan" % self.stepCtrl.value()
        namestrs = [ctrl.text() for ctrl in self.nameCtrls]
        return xstr, ystr, namestrs
