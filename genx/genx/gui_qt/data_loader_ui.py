from .message_dialogs import ShowNotificationDialog
from genx.data import DataSet


class UITemplate:
    def LoadDataFile(self, selected_items):
        """
        Selected items is the selected items in the current DataList
        into which data from file(s) should be loaded. Note that the default
        implementation only allows the loading of a single file.
        """
        from PySide6 import QtWidgets

        n_selected = len(selected_items)
        if n_selected > 0:
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self.parent,
                "Choose your Datafile",
                "",
                self.GetWildcardString(),
            )

            if file_path:
                dcount = self.CountDatasets(file_path)
                if n_selected > dcount:
                    ShowNotificationDialog(
                        self.parent, f"You can only load {dcount} dataset(s) from this file", "Too many selections"
                    )
                else:
                    self.SetData(self.parent.data_cont.get_data())
                    for di in range(n_selected):
                        dataset = DataSet(copy_from=self.data[selected_items[di]])
                        self.LoadDataset(dataset, file_path, data_id=di)

                        if len(dataset.x) == 0:
                            continue

                        self.data[selected_items[di]] = dataset
                    self.UpdateDataList()
                    self.SendUpdateDataEvent()

                    return True
        else:
            ShowNotificationDialog(self.parent, "Please select a dataset", "No active dataset")
        return False
