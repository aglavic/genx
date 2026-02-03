import wx
import wx.adv
import wx.lib.filebrowsebutton as filebrowse


def DataLoadWizard(plugin, selected_items):
    """Called when GenX wants to load data"""

    if len(selected_items)==0:
        ShowWarningDialog(plugin.parent, "Please select a data set before trying to load a spec file.")
        return False
    old_data = plugin.datalist[selected_items[0]].copy()
    plugin.dataset = plugin.datalist[selected_items[0]]
    wizard = wx.adv.Wizard(plugin.parent, -1, "Load Spec File", style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
    # page1 = TitledPage(wizard, "Page1")
    page1 = LoadSpecScanPage(wizard, plugin, default_filename=plugin.specfile_name)
    page2 = SelectCountersPage(wizard, plugin)
    page3 = CustomManipulationPage(wizard, plugin)
    page4 = SetNamePage(wizard, plugin)

    wx.adv.WizardPageSimple_Chain(page1, page2)
    wx.adv.WizardPageSimple_Chain(page2, page3)
    wx.adv.WizardPageSimple_Chain(page3, page4)

    wizard.FitToPage(page1)
    # print dir(wizard)
    # print 'data: ', self.datalist[0].x
    if wizard.RunWizard(page1):
        return True
    else:
        plugin.datalist[selected_items[0]].safe_copy(old_data)
        return False

class TitledPage(wx.adv.WizardPageSimple):
    def __init__(self, parent, title, plugin):
        wx.adv.WizardPageSimple.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        titleText = wx.StaticText(self, -1, title)
        titleText.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(titleText, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND | wx.ALL, 5)


class LoadSpecScanPage(wx.adv.WizardPageSimple):

    def __init__(self, parent, plugin, default_filename=None):
        self.plugin = plugin

        title = "Load scan"
        wx.adv.WizardPageSimple.__init__(self, parent)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        titleText = wx.StaticText(self, -1, title)
        titleText.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(titleText, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND | wx.ALL, 5)

        self.filebrowser = filebrowse.FileBrowseButton(
            self, -1, startDirectory=".", size=(450, -1), changeCallback=self.filebrowser_callback, fileMode=wx.FD_OPEN
        )
        self.filebrowser.textControl.Disable()
        self.sizer.Add(self.filebrowser, 0, wx.EXPAND | wx.ALL, 10)
        chooserText = wx.StaticText(self, -1, "Select the scans to load")
        self.sizer.Add(chooserText, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.scanchooser = wx.CheckListBox(self, -1)
        self.sizer.Add(self.scanchooser, 1, wx.EXPAND | wx.ALL, 2)

        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGING, self.OnWizPageChanging)
        self.scanchooser.Bind(wx.EVT_CHECKLISTBOX, self.OnScanSelected)

        self.SetControlEnable(wx.ID_FORWARD, False)

        # Load the defualt spec file -  if exists
        if default_filename:
            self.filebrowser.SetValue(default_filename)
            self.load_file(default_filename)

    def SetControlEnable(self, id, state):
        """Sets the control enable of window for parent to state. Use wx.ID_FORWARD or
        wx.ID_BACKWARD for the beck forward buttons.
        """
        win = self.Parent.FindWindowById(id)
        if win:
            win.Enable(state)
        else:
            raise Exception("Can not find window %s" % (id,))

    def OnScanSelected(self, evt):
        """A scan has been seelcted"""
        # check so at least one scan is selected
        if len(self.scanchooser.GetChecked()) > 0:
            self.SetControlEnable(wx.ID_FORWARD, True)
        else:
            self.SetControlEnable(wx.ID_FORWARD, False)

    def OnWizPageChanging(self, evt):
        # Check if we are moving forward
        if evt.GetDirection():
            scan_numbers = self.extract_scan_numbers()
            if self.plugin.specfile:
                try:
                    self.plugin.load_scans(scan_numbers)
                except Exception as e:
                    ShowErrorDialog(
                        self.Parent,
                        "Could not load the selected scan"
                        " - The scan might be corrupt.\n"
                        "Please choose another.\n"
                        "Python error: " + e.__str__(),
                    )
                    # print e.__str__()
                    evt.Veto()

    def extract_scan_numbers(self):
        """Extract the scan numbers from the Checklistbox"""
        scan_numbers = self.scanchooser.CheckedStrings
        # Extract the numbers from the strings
        scan_numbers = [int(ss.split(" ")[0]) for ss in scan_numbers]
        return scan_numbers

    def load_file(self, filename):
        self.scanchooser.Set(self.plugin.load_specfile(filename))
        # self.SetControlEnable(wx.ID_FORWARD, True)
        # print filename

    def filebrowser_callback(self, evt):
        filename = evt.GetString()
        self.load_file(filename)


class SelectCountersPage(wx.adv.WizardPageSimple):
    def __init__(self, parent, plugin):
        wx.adv.WizardPageSimple.__init__(self, parent)
        self.plugin = plugin
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        title = "Select Counters/Motors"
        titleText = wx.StaticText(self, -1, title)
        titleText.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(titleText, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND | wx.ALL, 5)

        self.choiceSizer = wx.FlexGridSizer(rows=4, cols=2, vgap=10, hgap=10)
        self.choiceSizer.Add(wx.StaticText(self, -1, "X value"), 0, wx.EXPAND | wx.ALL, 5)
        self.xChoice = wx.Choice(self, -1, (-1, -1), choices=[])
        self.choiceSizer.Add(self.xChoice, 0, wx.EXPAND | wx.ALL, 5)
        self.choiceSizer.Add(wx.StaticText(self, -1, "Det. value"), 0, wx.EXPAND | wx.ALL, 5)
        self.detChoice = wx.Choice(self, -1, (-1, -1), choices=[])
        self.choiceSizer.Add(self.detChoice, 0, wx.EXPAND | wx.ALL, 5)
        self.choiceSizer.Add(wx.StaticText(self, -1, "Mon. value"), 0, wx.EXPAND | wx.ALL, 5)
        self.monChoice = wx.Choice(self, -1, (-1, -1), choices=[])
        self.choiceSizer.Add(self.monChoice, 0, wx.EXPAND | wx.ALL, 5)
        self.choiceSizer.Add(wx.StaticText(self, -1, "Error value"), 0, wx.EXPAND | wx.ALL, 5)
        self.errorChoice = wx.Choice(self, -1, (-1, -1), choices=[])
        self.choiceSizer.Add(self.errorChoice, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.choiceSizer, 0, wx.ALIGN_CENTRE | wx.ALL)
        self.automergeCheckBox = wx.CheckBox(self, -1, "Auto merge data")
        self.automergeSizer = wx.FlexGridSizer(rows=1, cols=1, vgap=10, hgap=10)
        self.automergeSizer.Add(self.automergeCheckBox, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.automergeSizer, 0, wx.ALIGN_CENTRE | wx.ALL)

        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGED, self.OnWizPageChanged)
        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGING, self.OnWizPageChanging)

    def OnWizPageChanged(self, evt):
        # Check if we are moving forward
        # print 'In wiz page changing ', evt.GetDirection()
        if evt.GetDirection():
            choices = self.plugin.get_data_choices()
            self.update_counters(choices)

    def OnWizPageChanging(self, evt):
        if evt.GetDirection():
            self.plugin.update_data_cols(self.get_selected(), self.automergeCheckBox.IsChecked())

    def update_counters(self, mot_count):
        self.xChoice.SetItems(mot_count)
        if len(mot_count) > 0:
            self.xChoice.SetSelection(0)
        self.detChoice.SetItems(mot_count)
        try:
            self.detChoice.SetSelection(mot_count.index("Detector"))
        except:
            self.detChoice.SetSelection(0)
        self.monChoice.SetItems(["None"] + mot_count)
        try:
            self.monChoice.SetSelection(mot_count.index("Monitor") + 1)
        except:
            self.monChoice.SetSelection(0)
        self.errorChoice.SetItems(["None"] + mot_count)
        self.errorChoice.SetSelection(0)

    def get_selected(self):
        data = (
            self.xChoice.GetStringSelection(),
            self.detChoice.GetStringSelection(),
            self.monChoice.GetStringSelection(),
            self.errorChoice.GetStringSelection(),
        )
        return data


class CustomManipulationPage(wx.adv.WizardPageSimple):
    def __init__(self, parent, plugin, data_sets=None):
        wx.adv.WizardPageSimple.__init__(self, parent)
        if data_sets is None:
            data_sets = []
        self.plugin = plugin
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        title = "Custom Manipulation"
        titleText = wx.StaticText(self, -1, title)
        titleText.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(titleText, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND | wx.ALL, 5)
        help_text = (
            "By changing the expressions in the boxes below you can "
            + "manipulate your data with expressions written in Python.\n"
            + "These transforms can be changed later from the data calculations"
            + "dialog."
        )
        helpText = wx.StaticText(self, -1, help_text)
        helpText.Wrap(_maxWidth)
        self.sizer.Add(helpText, 0, wx.ALIGN_CENTRE | wx.ALL, 10)
        self.manipSizer = wx.FlexGridSizer(rows=4, cols=2, vgap=10, hgap=10)
        self.manipSizer.Add(wx.StaticText(self, -1, "x = "), 0, wx.EXPAND | wx.ALL, 5)
        self.xCtrl = wx.TextCtrl(self, -1, value="x")
        self.manipSizer.Add(self.xCtrl, 0, wx.EXPAND | wx.ALL, 5)

        self.manipSizer.Add(wx.StaticText(self, -1, "y = "), 0, wx.EXPAND | wx.ALL, 5)
        self.yCtrl = wx.TextCtrl(self, -1, value="det")
        self.manipSizer.Add(self.yCtrl, 0, wx.EXPAND | wx.ALL, 5)

        self.manipSizer.Add(wx.StaticText(self, -1, "Error = "), 0, wx.EXPAND | wx.ALL, 5)
        self.errorCtrl = wx.TextCtrl(self, -1, value="0*x")
        self.manipSizer.Add(self.errorCtrl, 0, wx.EXPAND | wx.ALL, 5)
        self.sizer.Add(self.manipSizer, 0, wx.ALIGN_CENTRE | wx.ALL)

        help_text = "Import the settings from data set: "
        helpText = wx.StaticText(self, -1, help_text)
        helpText.Wrap(_maxWidth)
        self.sizer.Add(helpText, 0, wx.ALIGN_CENTRE | wx.ALL, 10)
        self.importChoice = wx.Choice(self, -1, (-1, 50), choices=data_sets)
        self.sizer.Add(self.importChoice, 0, wx.ALIGN_CENTRE | wx.ALL, 10)
        self.Bind(wx.EVT_CHOICE, self.EvtImport, self.importChoice)

        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGED, self.OnWizPageChanged)
        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGING, self.OnWizPageChanging)

    def OnWizPageChanged(self, evt):
        """Event handler for entering this page"""
        manip_string = ["x", "det"]
        if self.plugin.mon_val != "None":
            manip_string.append("mon")
        else:
            manip_string.append("None")
        if self.plugin.error_val != "None":
            manip_string.append("e")
        else:
            manip_string.append("None")

        self.set_manip(manip_string)
        self.importChoice.SetItems(self.plugin.get_dataset_names())

    def OnWizPageChanging(self, evt):
        """Event Handler for leaving this page"""
        commands = {
            "x": self.xCtrl.GetValue(),
            "y": self.yCtrl.GetValue(),
            "e": self.errorCtrl.GetValue(),
        }
        if not self.plugin.set_commands(commands):
            evt.Veto()

    def EvtImport(self, evt):
        """On the event to import manipulations from another data set"""
        item = self.importChoice.GetSelection()
        com = self.plugin.datalist[item].get_commands()
        # self.set_manip((com['x'], com['y'], com['e']))
        self.xCtrl.ChangeValue(com["x"])
        self.yCtrl.ChangeValue(com["y"])
        self.errorCtrl.ChangeValue(com["e"])

    def set_manip(self, choices):
        """Sets the manipulations to defualt values as given by
        the choices in the form (x, det, mon, error)
        """
        iprint(choices)
        self.xCtrl.ChangeValue(choices[0])
        if choices[2] == "None":
            self.yCtrl.ChangeValue(choices[1])
            if choices[3] == "None":
                self.errorCtrl.ChangeValue("sqrt(%s)" % (choices[1],))
            else:
                self.errorCtrl.ChangeValue(choices[3])
        else:
            self.yCtrl.ChangeValue("%s/%s" % (choices[1], choices[2]))
            if choices[3] == "None":
                self.errorCtrl.ChangeValue(
                    "sqrt(1.0/%s + 1.0/%s)*%s/%s" % (choices[1], choices[2], choices[1], choices[2])
                )
            else:
                self.errorCtrl.ChangeValue(choices[3])


class SetNamePage(wx.adv.WizardPageSimple):

    def __init__(self, parent, plugin, data_sets=None):
        wx.adv.WizardPageSimple.__init__(self, parent)
        if data_sets is None:
            data_sets = []
        self.plugin = plugin
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)
        title = "Name"
        titleText = wx.StaticText(self, -1, title)
        titleText.SetFont(wx.Font(18, wx.SWISS, wx.NORMAL, wx.BOLD))
        self.sizer.Add(titleText, 0, wx.ALIGN_CENTRE | wx.ALL, 5)
        self.sizer.Add(wx.StaticLine(self, -1), 0, wx.EXPAND | wx.ALL, 5)
        help_text = (
            "Set the name of the data set as it will appear in GenX."
            "The name can be changed later by a slow double click on the name "
            "in the data panel."
        )
        helpText = wx.StaticText(self, -1, help_text)
        helpText.Wrap(_maxWidth)
        self.sizer.Add(helpText, 0, wx.ALIGN_CENTRE | wx.ALL, 10)

        self.manipSizer = wx.FlexGridSizer(rows=1, cols=2, vgap=10, hgap=10)
        self.manipSizer.Add(wx.StaticText(self, -1, "Name: "), 0, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 10)
        self.xCtrl = wx.TextCtrl(self, -1, value="Name")
        self.manipSizer.Add(self.xCtrl, 0, wx.EXPAND | wx.ALL, 10)
        self.sizer.Add(self.manipSizer, 0, wx.EXPAND | wx.ALIGN_CENTRE | wx.ALL)

        self.Bind(wx.adv.EVT_WIZARD_PAGE_CHANGING, self.OnWizPageChanging)

    def OnWizPageChanging(self, evt):
        """Event Handler for leaving this page"""

        self.plugin.dataset.name = self.xCtrl.GetValue()


def automerge(xval, yvals, rel_cond=100.0):
    """Merges the values according to rel_cond where the
    the merging criterion is the distance between the two first datapoints divided by
    rel_cond.

    xval - an arraray of x values
    y-vals a tuple of yvals
    rel_cond as above
    """
    min_size = np.abs(xval[1] - xval[0]) / rel_cond
    xarg = xval.argsort()
    xnew = np.array([xval[xarg[0]]]) * 1.0
    ynew = []
    for yval in yvals:
        ynew.append(np.array([yval[xarg[0]]]) * 1.0)
    inew = 0
    for i in range(len(xarg))[1:]:
        if np.abs(xnew[-1] - xval[xarg[i]]) <= min_size:
            for j in range(len(ynew)):
                ynew[j][-1] += yvals[j][xarg[i]]
        else:
            xnew = np.append(xnew, xval[xarg[i]])
            for j in range(len(ynew)):
                ynew[j] = np.append(ynew[j], yvals[j][xarg[i]])
            inew += 1

    return xnew, ynew
