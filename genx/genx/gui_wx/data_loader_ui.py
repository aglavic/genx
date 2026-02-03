from .utils import ShowInfoDialog
from genx.data import DataSet

class UITemplate:
    def LoadDataFile(self, selected_items):
        """
        Selected items is the selcted items in the items in the current DataList
        into which data from file(s) should be loaded. Note that the default
        implementation only allows the loading of a single file!
        Overriding this function in subclasses can of course change this
        behaviour. This function calls the LoadData function which implements
        the io function by it self. The LoadData has to be overloaded in
        order to have a working plugin.
        """
        import wx

        n_selected = len(selected_items)
        if n_selected > 0:
            dlg = wx.FileDialog(
                self.parent,
                message="Choose your Datafile",
                defaultFile="",
                wildcard=self.GetWildcardString(),
                style=wx.FD_OPEN | wx.FD_CHANGE_DIR,
            )

            if dlg.ShowModal() == wx.ID_OK:
                file_path = dlg.GetPath()
                dlg.Destroy()
                dcount = self.CountDatasets(file_path)
                if n_selected > dcount:
                    ShowInfoDialog(
                        self.parent, f"You can only load {dcount} dataset(s) from this file", "Too many selections"
                    )
                    dlg.ShowModal()
                    dlg.Destroy()
                else:
                    self.SetData(self.parent.data_cont.get_data())
                    for di in range(n_selected):
                        dataset = DataSet(copy_from=self.data[selected_items[di]])
                        self.LoadDataset(dataset, file_path, data_id=di)

                        if len(dataset.x) == 0:
                            continue

                        self.data[selected_items[di]] = dataset
                    # In case the dataset name has changed
                    self.UpdateDataList()

                    # Send an update that new data has been loaded
                    self.SendUpdateDataEvent()

                    return True
            else:
                dlg.Destroy()
        else:
            ShowInfoDialog(self.parent, "Please select a dataset", "No active dataset")
        return False

