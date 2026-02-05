from genx.gui_qt.utils import ShowWarningDialog


def DataLoadWizard(plugin, selected_items):
    """Qt placeholder for the spec loader wizard."""
    if len(selected_items) == 0:
        ShowWarningDialog(plugin.parent, "Please select a data set before trying to load a spec file.")
        return False
    ShowWarningDialog(plugin.parent, "Spec loader wizard is not yet available in the Qt UI.")
    return False
