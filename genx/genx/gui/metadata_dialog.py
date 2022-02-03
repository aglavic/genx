"""
A simple dialog window to display meta data read from files to the user.
"""

import yaml
import wx

from genx.data import DataList


class MetaDataDialog(wx.Dialog):
    datasets: DataList

    def __init__(self, parent, datasets: DataList, selected=0, filter_leaf_types=None, close_on_activate=False):
        wx.Dialog.__init__(self, parent, title="Dataset information",
                           style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX | wx.TR_HIDE_ROOT)
        sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.SetSizer(sizer)

        self.tree = wx.TreeCtrl(self)
        self.leaf_ids = []
        sizer.Add(self.tree, proportion=1, flag=wx.EXPAND)
        self.text = wx.TextCtrl(self, style=wx.TE_READONLY | wx.TE_MULTILINE | wx.TE_DONTWRAP)
        sizer.Add(self.text, proportion=2, flag=wx.EXPAND)

        self.datasets = datasets
        self.filter_leaf_types = filter_leaf_types
        self.build_tree(selected)

        self.Bind(wx.EVT_TREE_SEL_CHANGED, self.show_item)
        self.tree.Bind(wx.EVT_TREE_ITEM_ACTIVATED, self.item_activated)

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

            self.add_children(branch, di.meta, [i])
            if i==selected:
                self.tree.Expand(branch)
        self.tree.Expand(root)

    def add_children(self, node, source, path):
        for key, value in source.items():
            if isinstance(value, dict):
                itm = self.tree.AppendItem(node, key)
                self.tree.SetItemData(itm,
                                      (key, yaml.dump(value, indent=4).replace('    ', '\t').replace('\n', '\n\t')))
                self.add_children(itm, value, path+[key])
            else:
                itm = self.tree.AppendItem(node, key)
                self.tree.SetItemData(itm, (key, f'{value} ({type(value).__name__})', path+[key]))
                if self.filter_leaf_types is None or type(value) in self.filter_leaf_types:
                    self.leaf_ids.append(itm)
                    self.tree.SetItemBackgroundColour(itm, wx.Colour('aaaaff'))
                else:
                    self.tree.SetItemTextColour(itm, wx.Colour('aaaaaa'))

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
        self.activated_leaf =self.tree.GetItemData(event.GetItem())[2]
        if self.close_on_activate:
            self.EndModal(wx.ID_OK)
