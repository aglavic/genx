# -*- coding: utf8 -*-
"""
======
MagSLD
======

A plugin to show a second y-axis in the SLD plot that gives the magnetization corresponging
to the magnetic SLD and to calculate the integrated magnetic moment surface density. 

Written by Artur Glavic
Last Changes 04/28/15
"""

import wx

from numpy import trapz

from genx.models.lib.physical_constants import AAm2_to_emucc

from .. import add_on_framework as framework


class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent

        self._init_refplugin()

        menu = self.NewMenu("Mag. SLD")
        self.mb_second_axis = wx.MenuItem(
            menu, wx.NewId(), "Show second SLD axis", "Show second SLD axis for magnetization", wx.ITEM_CHECK
        )
        menu.Append(self.mb_second_axis)
        self.mb_second_axis.Check(True)
        self.mb_use_SI = wx.MenuItem(menu, wx.NewId(), "Use SI units", "Use SI units of magnetization", wx.ITEM_CHECK)
        menu.Append(self.mb_use_SI)
        self.mb_use_SI.Check(False)

        self.parent.Bind(wx.EVT_MENU, self._OnSimulate, self.mb_second_axis)
        self.parent.Bind(wx.EVT_MENU, self._OnSimulate, self.mb_use_SI)

    def _init_refplugin(self):
        ph = self.parent.plugin_control.plugin_handler
        if "Reflectivity" in ph.loaded_plugins:
            # connect to the reflectivity plugin for layer creation
            self.refplugin = ph.loaded_plugins["Reflectivity"]
        elif "SimpleReflectivity" in ph.loaded_plugins:
            # connect to the reflectivity plugin for layer creation
            self.refplugin = ph.loaded_plugins["SimpleReflectivity"]
        else:
            raise RuntimeError("Can't load plugin without Reflectivity plugin present.")

        self._orgi_call = self.refplugin.OnSimulate
        self.refplugin.OnSimulate = self._OnSimulate

        sld_plot = self.refplugin.sld_plot
        ax = sld_plot.ax
        self._orig_position = ax.get_position()
        ax2 = ax.twinx()
        sld_plot.ax2 = ax2
        box = self._orig_position
        ax2.set_position([box.x0, box.y0, box.width, box.height])
        ax2.set_visible(False)
        self._annotations = []

    def Remove(self):
        self.refplugin.OnSimulate = self._orgi_call
        sld_plot = self.refplugin.sld_plot
        sld_plot.ax2.set_visible(False)
        del sld_plot.ax2
        del self.refplugin
        framework.Template.Remove(self)

    def _OnSimulate(self, event):
        """
        Call original reflectivity plugin OnSimulate function to make sure the plot is
        already created when adding the second axis.
        """
        for ann in self._annotations:
            try:
                ann.remove()
            except:
                pass
        self._annotations = []
        self._orgi_call(event)
        sld_plot = self.refplugin.sld_plot
        model = self.GetModel()
        sld_items = len(sld_plot.plot_dicts)
        msld = None
        z = None
        unit = ""
        ax = sld_plot.ax
        ax2 = sld_plot.ax2
        for i, di in enumerate(model.data):
            if di.show and sld_items > i:
                slds = sld_plot.plot_dicts[i]
                if not "z" in slds or not "SLD unit" in slds:
                    continue
                z = slds["z"]
                unit = slds["SLD unit"]
                if unit in ["fm/\AA^{3}", "\\AA^{-2}", "", "10^{-6}\\AA^{-2}"] and self.mb_second_axis.IsChecked():
                    for key, value in list(slds.items()):
                        if key == "mag" and value.sum() > 0.0:
                            msld = value
                            mag = msld * AAm2_to_emucc
                            com = (msld * z).sum() / msld.sum()
                            if self.mb_use_SI.IsChecked():
                                self._annotations.append(
                                    ax.annotate(
                                        "Integrated M:\n%.4g $A$" % trapz(mag * 1e3, z * 1e-10),
                                        (com, 0.02),
                                        ha="center",
                                    )
                                )
                            else:
                                self._annotations.append(
                                    ax.annotate(
                                        "Integrated M:\n%.4g $emu/cm^2$" % trapz(mag, z * 1e-8),
                                        (com, 0.02),
                                        ha="center",
                                    )
                                )
        if (
            msld is not None
            and unit in ["fm/\AA^{3}", "\\AA^{-2}", "", "10^{-6}\\AA^{-2}"]
            and self.mb_second_axis.IsChecked()
        ):
            ax2.set_visible(True)
            ymin, ymax = ax.get_ylim()
            if self.mb_use_SI.IsChecked():
                ax2.set_ylim((ymin * AAm2_to_emucc * 1e3, ymax * AAm2_to_emucc * 1e3))
                ax2.yaxis.label.set_text("M [$A/m$]")
            else:
                ax2.set_ylim((ymin * AAm2_to_emucc, ymax * AAm2_to_emucc))
                ax2.yaxis.label.set_text("M [$emu/cm^3$]")
            sld_plot.ax.legend(loc="upper right", framealpha=0.5, fontsize="small", ncol=1)  # bbox_to_anchor=(1, 0.5),
        else:
            ax2.set_visible(False)
