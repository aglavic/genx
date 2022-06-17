"""
A simple dialog window to display meta data read from files to the user.
"""

import yaml
import wx
from logging import debug

from genx.data import DataList
from genx.model import Model


class MetaDataDialog(wx.Dialog):
    datasets: DataList

    def __init__(self, parent, datasets: DataList, selected=0, filter_leaf_types=None, close_on_activate=False):
        wx.Dialog.__init__(self, parent, title="Dataset information",
                           style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX | wx.TR_HIDE_ROOT)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(sizer)

        self.tree = wx.TreeCtrl(self)
        self.leaf_ids = []

        vsizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(vsizer, proportion=1, flag=wx.EXPAND)
        vsizer.Add(self.tree, proportion=1, flag=wx.EXPAND)

        btn = wx.Button(self, label='Make ORSO Conform')
        vsizer.Add(btn, flag=wx.FIXED_MINSIZE)
        btn.Bind(wx.EVT_BUTTON, self.make_orso_conform)

        self.text = wx.TextCtrl(self, style=wx.TE_READONLY | wx.TE_MULTILINE | wx.TE_DONTWRAP)
        sizer.Add(self.text, proportion=2, flag=wx.EXPAND)

        self.datasets = datasets
        self.filter_leaf_types = filter_leaf_types
        try:
            from orsopy import fileio
        except ImportError:
            self.orso_repr = [None for di in self.datasets]
        else:
            self.orso_repr = [fileio.Orso(**di.meta) for di in self.datasets]
        self.build_tree(selected)

        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.show_item)
        self.tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.item_activated)

        self.tree.Bind(wx.EVT_TREE_ITEM_RIGHT_CLICK, self.OnRightClick)

        self.SetSize((800, 800))
        self.activated_leaf = None
        self.close_on_activate = close_on_activate

    def build_tree(self, selected):
        root = self.tree.AddRoot('datasets')
        self.tree.SetItemData(root, ('', 'Select key to show information'))
        for i, di in enumerate(self.datasets):
            branch = self.tree.AppendItem(root, di.name)
            self.tree.SetItemData(branch, (di.name,
                                           yaml.dump(di.meta, indent=4).replace('    ', '\t').replace('\n', '\n\t')))

            self.add_children(branch, di.meta, [i], self.orso_repr[i])
            if i==selected:
                self.tree.Expand(branch)
        self.tree.Expand(root)
        self.add_coloring()

    def add_children(self, node, source, path, orso_repr):
        for key, value in source.items():
            if isinstance(value, dict):
                obj = getattr(orso_repr, key, None)
                itm = self.tree.AppendItem(node, key)
                self.tree.SetItemData(itm,
                                      (key, yaml.dump(value, indent=4).replace('    ', '\t').replace('\n', '\n\t'),
                                       path+[key], obj))
                self.add_children(itm, value, path+[key], getattr(orso_repr, key, None))
            else:
                itm = self.tree.AppendItem(node, key)
                vtype=type(value)
                self.tree.SetItemData(itm, (key, f'{value} ({vtype.__name__})', path+[key], vtype))
                if self.filter_leaf_types is None or type(value) in self.filter_leaf_types:
                    self.leaf_ids.append(itm)
                    self.tree.SetItemBackgroundColour(itm, wx.Colour('aaaaff'))
                else:
                    self.tree.SetItemTextColour(itm, wx.Colour('aaaaaa'))
            if orso_repr:
                if key in getattr(orso_repr, '_orso_optionals', {}):
                    self.tree.SetItemBackgroundColour(itm, wx.Colour(255, 255, 150))
                elif key in getattr(orso_repr, 'user_data', {}):
                    self.tree.SetItemBackgroundColour(itm, wx.Colour(255, 150, 255))
                else:
                    self.tree.SetItemBackgroundColour(itm, wx.Colour(150, 255, 150))

    def add_coloring(self):
        root = self.tree.GetRootItem()
        node = self.tree.AppendItem(root, 'coloring')
        itm = self.tree.AppendItem(node, 'orso required')
        self.tree.SetItemBackgroundColour(itm, wx.Colour(150, 255, 150))
        itm = self.tree.AppendItem(node, 'orso optional')
        self.tree.SetItemBackgroundColour(itm, wx.Colour(255, 255, 150))
        itm = self.tree.AppendItem(node, 'user defined')
        self.tree.SetItemBackgroundColour(itm, wx.Colour(255, 150, 255))
        self.tree.Expand(node)

    def show_item(self, event: wx.TreeEvent):
        item = event.GetItem()
        name, data, *_ = self.tree.GetItemData(item)
        self.text.Clear()
        self.text.AppendText('%s:\n\n\t%s'%(name, data))

    def item_activated(self, event: wx.TreeEvent):
        if event.GetItem() not in self.leaf_ids:
            event.Skip()
            return
        # a leaf item was activated
        name, data, self.activated_leaf, vtype=self.tree.GetItemData(event.GetItem())
        if self.close_on_activate:
            self.EndModal(wx.ID_OK)
        else:
            prev_dict = self.datasets[self.activated_leaf[0]].meta
            if vtype is type(None):
                from orsopy import fileio
                from typing import Dict, List, Tuple, Union, Literal
                item = fileio.Orso(**prev_dict)
                for key in self.activated_leaf[1:-1]:
                    item = getattr(item, key, None)
                    if item is None:
                        break
                if item is None:
                    vtype = str
                else:
                    vtype = item.__annotations__.get(self.activated_leaf[-1], str)
                from orsopy.fileio.base import get_args, get_origin
                if get_origin(vtype) == Union:
                    vtype = get_args(vtype)[0]
                if get_origin(vtype) == Literal or vtype==fileio.Polarization:
                    if vtype==fileio.Polarization:
                        options = [v.value for v in  fileio.Polarization]
                    else:
                        options = get_args(vtype)
                    dia = wx.SingleChoiceDialog(self, message=f'Select new value for {name}',
                                               caption='Select Value', choices=options)
                    if dia.ShowModal()==wx.ID_OK and dia.GetSelection()>=0:
                        value = options[dia.GetSelection()]
                        for key in self.activated_leaf[1:-1]:
                            prev_dict = prev_dict[key]
                        prev_dict[self.activated_leaf[-1]] = value
                        self.tree.SetItemData(event.GetItem(), (name, f'{value} ({type(None).__name__})',
                                                                self.activated_leaf, type(None)))
                        self.show_item(event)
                        debug(f'updated {name} to {prev_dict[self.activated_leaf[-1]]}')
                        self.update_parents(event.GetItem(), self.activated_leaf)
                    return
            if not vtype in [str, int, float]:
                # can only edit simple leaf items that can be converted from a str up to now
                print(vtype)
                return
            for key in self.activated_leaf[1:-1]:
                prev_dict = prev_dict[key]
            prev_value = prev_dict[self.activated_leaf[-1]]
            dia = wx.TextEntryDialog(self, message=f'Enter new value for {name} with type {vtype}', caption='Enter Value',
                                     value=str(prev_value))
            if dia.ShowModal()==wx.ID_OK:
                value = vtype(dia.GetValue())
                prev_dict[self.activated_leaf[-1]] = value
                self.tree.SetItemData(event.GetItem(), (name, f'{value} ({vtype.__name__})',
                                                        self.activated_leaf, vtype))
                self.show_item(event)
                debug(f'updated {name} to {prev_dict[self.activated_leaf[-1]]}')
                self.update_parents(event.GetItem(), self.activated_leaf)

    def update_parents(self, item, path, initial=False):
        if len(path)==1:
            return
        if initial:
            parent = item
        else:
            parent = self.tree.GetItemParent(item)
        mpath = list(path[:-1])
        obj = self.orso_repr[mpath[0]]
        mdict = self.datasets[mpath.pop(0)].meta
        while len(mpath)>0:
            obj = getattr(obj, mpath[0], None)
            mdict = mdict[mpath.pop(0)]
        value = mdict
        self.tree.SetItemData(parent,
                              (path[-2], yaml.dump(value, indent=4).replace('    ', '\t').replace('\n', '\n\t'),
                               path[:-1], obj))
        self.update_parents(parent, path[:-1])

    def make_orso_conform(self, evt):
        Model.update_orso_meta(self.datasets)
        self.tree.DeleteAllItems()
        self.leaf_ids = []
        self.build_tree(selected=0)

    def OnRightClick(self, event: wx.TreeEvent):
        item = event.GetItem()
        if item in self.leaf_ids:
            name, data, leaf, vtype = self.tree.GetItemData(event.GetItem())
        else:
            try:
                key, text, path, obj = self.tree.GetItemData(item)
            except TypeError:
                return
            popupmenu = wx.Menu()
            entries = list(obj.__annotations__.keys())
            self._popup_tree_item = (item, path)
            self._popup_menu_ids = {}
            for entry in entries:
                if entry in obj._orso_optionals:
                    text = f"new {entry}*"
                else:
                    text = f"new {entry}"
                menuItem = popupmenu.Append(-1, text)
                self._popup_menu_ids[menuItem.GetId()] = entry
            popupmenu.Bind(wx.EVT_MENU, self.make_new_key)
            self.tree.PopupMenu(popupmenu, event.GetPoint())

    def make_new_key(self, event: wx.CommandEvent):
        from typing import Union
        key, text, path, obj = self.tree.GetItemData(self._popup_tree_item[0])
        spath = list(path)
        ktype=obj.__annotations__[self._popup_menu_ids[event.GetId()]]
        mdict = self.datasets[spath.pop(0)].meta
        while len(spath)>0:
            mdict = mdict[spath.pop(0)]
        from orsopy.fileio.base import get_args, get_origin
        if get_origin(ktype)==Union:
            args =  list(get_args(ktype))
            if args[-1] is type(None):
                args.pop(-1)
            if len(args)==1:
                ktype = args[0]
            else:
                options = [ci.__name__ for ci in args]
                dia = wx.SingleChoiceDialog(self,
                                    message=f'Select object type for {key}.{self._popup_menu_ids[event.GetId()]}',
                                    caption='Select Item', choices=options)
                if dia.ShowModal()==wx.ID_OK and dia.GetSelection()>=0:
                    ktype = args[dia.GetSelection()]
        if hasattr(ktype, '_orso_optionals'):
            new = ktype.empty()
            setattr(obj, self._popup_menu_ids[event.GetId()], new)
            mdict[self._popup_menu_ids[event.GetId()]] = new.to_dict()
        else:
            setattr(obj, self._popup_menu_ids[event.GetId()], None)
            mdict[self._popup_menu_ids[event.GetId()]] = None
        self.tree.DeleteChildren(self._popup_tree_item[0])
        self.add_children(self._popup_tree_item[0], mdict, path, obj)
        self.update_parents(self._popup_tree_item[0], path+[key], initial=True)
