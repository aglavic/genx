"""
A simple dialog window to display meta data read from files to the user.
Qt port of the wx implementation.
"""

from copy import deepcopy
from datetime import datetime
from logging import debug

import yaml
from PySide6 import QtCore, QtGui, QtWidgets

from genx.data import DataList
from genx.model import Model


class MetaDataDialog(QtWidgets.QDialog):
    datasets: DataList

    def __init__(self, parent, datasets: DataList, selected=0, filter_leaf_types=None, close_on_activate=False):
        super().__init__(parent)
        self.setWindowTitle("Dataset information")
        self.setWindowFlags(self.windowFlags() | QtCore.Qt.WindowType.WindowMaximizeButtonHint)

        self.datasets = datasets
        self.filter_leaf_types = filter_leaf_types
        self.close_on_activate = close_on_activate

        try:
            from orsopy import fileio

            self.orso_repr = [fileio.Orso(**deepcopy(di.meta)) for di in self.datasets]
        except Exception:
            self.orso_repr = [None for di in self.datasets]

        self.tree = QtWidgets.QTreeWidget(self)
        self.tree.setHeaderHidden(True)
        self.tree.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.OnRightClick)

        self.leaf_ids = set()

        btn = QtWidgets.QPushButton("Make ORSO Conform", self)
        btn.clicked.connect(self.make_orso_conform)

        self.text = QtWidgets.QPlainTextEdit(self)
        self.text.setReadOnly(True)

        left = QtWidgets.QVBoxLayout()
        left.addWidget(self.tree, 1)
        left.addWidget(btn, 0)

        main = QtWidgets.QHBoxLayout(self)
        main.addLayout(left, 1)
        main.addWidget(self.text, 2)

        self.tree.currentItemChanged.connect(self.show_item)
        self.tree.itemActivated.connect(self.item_activated)

        self.setMinimumSize(800, 800)
        self.activated_leaf = None

        self.build_tree(selected)

    def build_tree(self, selected):
        self.tree.clear()
        root = QtWidgets.QTreeWidgetItem(["datasets"])
        root.setData(0, QtCore.Qt.ItemDataRole.UserRole, ("", "Select key to show information"))
        self.tree.addTopLevelItem(root)
        for i, di in enumerate(self.datasets):
            branch = QtWidgets.QTreeWidgetItem([di.name])
            branch.setData(
                0,
                QtCore.Qt.ItemDataRole.UserRole,
                (
                    di.name,
                    yaml.dump(di.meta, indent=4, sort_keys=False).replace("    ", "\t").replace("\n", "\n\t"),
                    [i],
                    self.orso_repr[i],
                ),
            )
            root.addChild(branch)
            self.add_children(branch, di.meta, [i], self.orso_repr[i])
            if i == selected:
                branch.setExpanded(True)
        root.setExpanded(True)
        self.add_coloring(root)

    def add_children(self, node, source, path, orso_repr):
        for key, value in source.items():
            if isinstance(value, dict):
                obj = getattr(orso_repr, key, None)
                itm = QtWidgets.QTreeWidgetItem([key])
                itm.setData(
                    0,
                    QtCore.Qt.ItemDataRole.UserRole,
                    (
                        key,
                        yaml.dump(value, indent=4, sort_keys=False).replace("    ", "\t").replace("\n", "\n\t"),
                        path + [key],
                        obj,
                    ),
                )
                node.addChild(itm)
                self.add_children(itm, value, path + [key], getattr(orso_repr, key, None))
            else:
                itm = QtWidgets.QTreeWidgetItem([key])
                vtype = type(value)
                itm.setData(0, QtCore.Qt.ItemDataRole.UserRole, (key, f"{value} ({vtype.__name__})", path + [key], vtype))
                if self.filter_leaf_types is None or type(value) in self.filter_leaf_types:
                    self.leaf_ids.add(itm)
                    itm.setBackground(0, QtGui.QBrush(QtGui.QColor("aaaaff")))
                else:
                    itm.setForeground(0, QtGui.QBrush(QtGui.QColor("aaaaaa")))
                node.addChild(itm)
            if orso_repr:
                if key in getattr(orso_repr, "_orso_optionals", {}):
                    itm.setBackground(0, QtGui.QBrush(QtGui.QColor(255, 255, 150)))
                elif key in getattr(orso_repr, "user_data", {}):
                    itm.setBackground(0, QtGui.QBrush(QtGui.QColor(255, 150, 255)))
                elif key in getattr(orso_repr, "__annotations__", {}):
                    itm.setBackground(0, QtGui.QBrush(QtGui.QColor(150, 255, 150)))

    def add_coloring(self, root):
        node = QtWidgets.QTreeWidgetItem(["coloring"])
        root.addChild(node)
        itm = QtWidgets.QTreeWidgetItem(["orso required"])
        itm.setBackground(0, QtGui.QBrush(QtGui.QColor(150, 255, 150)))
        node.addChild(itm)
        itm = QtWidgets.QTreeWidgetItem(["orso optional"])
        itm.setBackground(0, QtGui.QBrush(QtGui.QColor(255, 255, 150)))
        node.addChild(itm)
        itm = QtWidgets.QTreeWidgetItem(["user defined"])
        itm.setBackground(0, QtGui.QBrush(QtGui.QColor(255, 150, 255)))
        node.addChild(itm)
        node.setExpanded(True)

    def show_item(self, item, _prev=None):
        if item is None:
            return
        data = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if not data:
            return
        name, text = data[0], data[1]
        self.text.setPlainText(f"{name}:\n\n\t{text}")

    def item_activated(self, item, _column=0):
        if item not in self.leaf_ids:
            return
        name, data, self.activated_leaf, vtype = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
        if self.close_on_activate:
            self.accept()
            return
        prev_dict = self.datasets[self.activated_leaf[0]].meta
        if vtype is type(None):
            from typing import Dict, List, Literal, Tuple, Union

            from orsopy import fileio

            item_obj = fileio.Orso(**prev_dict)
            for key in self.activated_leaf[1:-1]:
                item_obj = getattr(item_obj, key, None)
                if item_obj is None:
                    break
            if item_obj is None:
                vtype = None
            else:
                vtype = item_obj.__annotations__.get(self.activated_leaf[-1], str)
            from orsopy.fileio.base import get_args, get_origin

            if get_origin(vtype) == Union:
                vtype = get_args(vtype)[0]
            if get_origin(vtype) == Literal or vtype == fileio.Polarization:
                if vtype == fileio.Polarization:
                    options = [v.value for v in fileio.Polarization]
                else:
                    options = list(get_args(vtype))
                value, ok = QtWidgets.QInputDialog.getItem(
                    self, "Select Value", f"Select new value for {name}", options, 0, False
                )
                if ok:
                    for key in self.activated_leaf[1:-1]:
                        prev_dict = prev_dict[key]
                    prev_dict[self.activated_leaf[-1]] = value
                    item.setData(
                        0, QtCore.Qt.ItemDataRole.UserRole, (name, f"{value} ({type(None).__name__})", self.activated_leaf, type(None))
                    )
                    self.show_item(item)
                    debug(f"updated {name} to {prev_dict[self.activated_leaf[-1]]}")
                    self.update_parents(item, self.activated_leaf)
                return
        if vtype is list:
            for key in self.activated_leaf[1:-1]:
                prev_dict = prev_dict[key]
            prev_value = prev_dict[self.activated_leaf[-1]]
            options = [str(di) for di in prev_value]
            value, ok = QtWidgets.QInputDialog.getItem(
                self, "Select Value", f"Select new value to edit in {name}", options, 0, False
            )
            if ok:
                idx = options.index(value)
                new_value, ok = QtWidgets.QInputDialog.getText(
                    self, "Enter Value", "Enter new value", text=str(prev_value[idx])
                )
                if ok:
                    if type(prev_value[idx]) is dict:
                        prev_value[idx] = eval(new_value)
                    else:
                        prev_value[idx] = type(prev_value[idx])(new_value)
                item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (name, f"{prev_value} (list)", self.activated_leaf, list))
                self.update_parents(item, self.activated_leaf)
                self.text.setPlainText(f"{name}:\n\n\t{prev_value} (list)")
            return
        elif vtype is datetime:
            return
        elif not vtype in [str, int, float, None]:
            return
        for key in self.activated_leaf[1:-1]:
            prev_dict = prev_dict[key]
        prev_value = prev_dict[self.activated_leaf[-1]]
        value, ok = QtWidgets.QInputDialog.getText(
            self, "Enter Value", f"Enter new value for {name} with type {vtype}", text=str(prev_value)
        )
        if ok:
            if vtype is None:
                try:
                    value_eval = eval(value)
                except Exception:
                    value_eval = value
                vtype = type(value_eval)
                value = value_eval
            else:
                value = vtype(value)
            prev_dict[self.activated_leaf[-1]] = value
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, (name, f"{value} ({vtype.__name__})", self.activated_leaf, vtype))
            self.show_item(item)
            debug(f"updated {name} to {prev_dict[self.activated_leaf[-1]]}")
            self.update_parents(item, self.activated_leaf)

    def update_parents(self, item, path, initial=False):
        if len(path) == 1:
            return
        parent = item if initial else item.parent()
        mpath = list(path[:-1])
        obj = self.orso_repr[mpath[0]]
        mdict = self.datasets[mpath.pop(0)].meta
        while len(mpath) > 0:
            obj = getattr(obj, mpath[0], None)
            mdict = mdict[mpath.pop(0)]
        value = mdict
        parent.setData(
            0,
            QtCore.Qt.ItemDataRole.UserRole,
            (
                path[-2],
                yaml.dump(value, indent=4, sort_keys=False).replace("    ", "\t").replace("\n", "\n\t"),
                path[:-1],
                obj,
            ),
        )
        self.update_parents(parent, path[:-1])

    def make_orso_conform(self):
        Model.update_orso_meta(self.datasets)
        try:
            from orsopy import fileio

            self.orso_repr = [fileio.Orso(**di.meta) for di in self.datasets]
        except Exception:
            self.orso_repr = [None for di in self.datasets]
        self.leaf_ids = set()
        self.build_tree(selected=0)

    def OnRightClick(self, point):
        item = self.tree.itemAt(point)
        if item is None:
            return
        if item in self.leaf_ids:
            name, data, leaf, vtype = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            if vtype is list:
                entries = ["append item", "move item"]
                callback = self.list_popup_action
            else:
                entries = ["no action implemented"]
                callback = lambda _action: None
            menu = QtWidgets.QMenu(self)
            self._popup_tree_item = (item, leaf)
            for entry in entries:
                action = menu.addAction(entry)
                action.setData(entry)
            action = menu.exec(self.tree.viewport().mapToGlobal(point))
            if action is not None:
                callback(action)
        else:
            try:
                key, text, path, obj = item.data(0, QtCore.Qt.ItemDataRole.UserRole)
            except (TypeError, ValueError):
                return
            menu = QtWidgets.QMenu(self)
            if type(obj) is dict:
                entries = ["new key"] + list(obj.keys())
            else:
                entries = list(obj.__annotations__.keys())
            self._popup_tree_item = (item, path)
            for entry in entries:
                if type(obj) is dict:
                    text = entry if entry == "new key" else f"delete {entry}"
                elif entry in obj._orso_optionals:
                    text = f"new {entry}*"
                else:
                    text = f"new {entry}"
                action = menu.addAction(text)
                action.setData(entry)
            action = menu.exec(self.tree.viewport().mapToGlobal(point))
            if action is not None:
                self.make_new_key(action)

    def resolve_type(self, ktype, node, key):
        from typing import Union

        from orsopy.fileio.base import get_args, get_origin

        if get_origin(ktype) is dict:
            return dict
        if get_origin(ktype) is list:
            return list
        if get_origin(ktype) == Union:
            args = list(get_args(ktype))
            if args[-1] is type(None):
                args.pop(-1)
            if len(args) == 1:
                return self.resolve_type(args[0], node, key)
            options = [ci.__name__ for ci in args]
            value, ok = QtWidgets.QInputDialog.getItem(
                self, "Select Item", f"Select object type for {node}.{key}", options, 0, False
            )
            if ok:
                ktype = args[options.index(value)]
                return self.resolve_type(ktype, node, key)
            return None
        return ktype

    def make_new_key(self, action):
        key, text, path, obj = self._popup_tree_item[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        spath = list(path)
        mdict = self.datasets[spath.pop(0)].meta
        while len(spath) > 0:
            mdict = mdict[spath.pop(0)]
        if type(obj) is dict:
            dkey = action.data()
            if dkey == "new key":
                value, ok = QtWidgets.QInputDialog.getText(self, "Enter Key Name", f"Enter new key name for {key}")
                if ok:
                    parent = self._popup_tree_item[0].parent()
                    pkey, ptext, ppath, pobj = parent.data(0, QtCore.Qt.ItemDataRole.UserRole)
                    from typing import Union

                    from orsopy.fileio.base import get_args, get_origin

                    try:
                        ktype = pobj.__annotations__[key]
                        if get_origin(ktype) == Union:
                            ktype = [arg for arg in get_args(ktype) if get_origin(arg) == dict][0]
                        ktype = get_args(ktype)[1]
                        ktype = self.resolve_type(ktype, key, "item")
                    except (AttributeError, KeyError, IndexError):
                        ktype = None
                    if hasattr(ktype, "_orso_optionals"):
                        new = ktype.empty()
                        for attr, atype in ktype.__annotations__.items():
                            if get_origin(atype) == Union:
                                atype = get_args(atype)[0]
                            if get_origin(atype) in [list, dict]:
                                continue
                            aval, ok = QtWidgets.QInputDialog.getText(
                                self,
                                "Enter Attribute Value",
                                f"Enter value for attribute {attr} ({atype.__name__}).\nCancel for \"None\".",
                            )
                            if ok:
                                setattr(new, attr, atype(aval))
                        obj[value] = new.to_dict()
                    else:
                        obj[value] = None
            else:
                del obj[dkey]
            self._popup_tree_item[0].takeChildren()
            self.add_children(self._popup_tree_item[0], mdict, path, obj)
            self.update_parents(self._popup_tree_item[0], path + [key], initial=True)
            return
        ktype = obj.__annotations__[action.data()]
        ktype = self.resolve_type(ktype, key, action.data())
        if hasattr(ktype, "_orso_optionals"):
            new = ktype.empty()
            setattr(obj, action.data(), new)
            mdict[action.data()] = new.to_dict()
        elif ktype is dict:
            new = {}
            setattr(obj, action.data(), new)
            mdict[action.data()] = new
        elif ktype is list:
            setattr(obj, action.data(), [])
            mdict[action.data()] = []
        else:
            setattr(obj, action.data(), None)
            mdict[action.data()] = None
        self._popup_tree_item[0].takeChildren()
        self.add_children(self._popup_tree_item[0], mdict, path, obj)
        self.update_parents(self._popup_tree_item[0], path + [key], initial=True)

    def list_popup_action(self, action):
        name, data, leaf, vtype = self._popup_tree_item[0].data(0, QtCore.Qt.ItemDataRole.UserRole)
        entry = action.data()
        if entry == "move item":
            return
        if entry == "append item":
            parent = self._popup_tree_item[0].parent()
            pkey, ptext, ppath, pobj = parent.data(0, QtCore.Qt.ItemDataRole.UserRole)
            prev_dict = self.datasets[leaf[0]].meta
            for key in leaf[1:-1]:
                prev_dict = prev_dict[key]
            lobj = prev_dict[leaf[-1]]
            from typing import Union

            from orsopy.fileio.base import get_args, get_origin

            try:
                ktype = pobj.__annotations__[name]
                if get_origin(ktype) == Union:
                    ktype = [arg for arg in get_args(ktype) if get_origin(arg) == list][0]
                ktype = get_args(ktype)[0]
            except (AttributeError, KeyError, IndexError):
                ktype = None
            ktype = self.resolve_type(ktype, name, "item")
            if hasattr(ktype, "_orso_optionals"):
                new = ktype.empty()
                lobj.append(new.to_dict())
            else:
                value, ok = QtWidgets.QInputDialog.getText(self, "Enter Value", f"Enter value for new {name} item")
                if ok:
                    lobj.append(value)
                else:
                    return
            self._popup_tree_item[0].setData(0, QtCore.Qt.ItemDataRole.UserRole, (name, f"{lobj} (list)", leaf, list))
            self.update_parents(self._popup_tree_item[0], leaf)
            self.text.setPlainText(f"{name}:\n\n\t{lobj} (list)")
