"""
Plotting widget for sample SLDs.
"""
import colorsys
from dataclasses import dataclass

import wx
import numpy as np
import time
import os

from scipy.interpolate import interp1d

from .reflectivity_misc import ReflectivityModule
from genx.core.custom_logging import iprint
from genx.data import DataList
from genx.gui.plotpanel import BasePlotConfig, PlotPanel
from genx.model import Model


@dataclass
class SamplePlotConfig(BasePlotConfig):
    section = 'sample plot'
    data_derived_color: bool = False
    show_single_model: bool = False
    legend_outside: bool = False

    show_imag: bool = False


class SamplePlotPanel(PlotPanel):
    '''
    Widget for plotting the scattering length density of a sample.
    '''
    opt: SamplePlotConfig

    def __init__(self, parent, plugin, id=-1, color=None, dpi=None
                 , style=wx.NO_FULL_REPAINT_ON_RESIZE, **kwargs):
        PlotPanel.__init__(self, parent, id, color, dpi, SamplePlotConfig, style, **kwargs)
        self.plugin = plugin

        self.update = self.Plot
        self.create_axes()

    def create_axes(self):
        self.ax = self.figure.add_subplot(111)
        box = self.ax.get_position()
        if self.opt.legend_outside:
            self.ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        else:
            self.ax.set_position([box.x0, box.y0, box.width*0.95, box.height])

    def Plot(self):
        '''
        Plotting the sample.
        '''
        if not self.ax:
            # self.ax = self.figure.add_subplot(111)
            self.create_axes()

        data: DataList = self.plugin.GetModel().get_data()
        if self.opt.data_derived_color:
            colors = []
            for di in data:
                h, s, v = colorsys.rgb_to_hsv(*di.sim_color)
                ci = [
                    colorsys.hsv_to_rgb(h, 1.0, 1.0),
                    colorsys.hsv_to_rgb(h, 1.0, 0.5),
                    colorsys.hsv_to_rgb(h, 0.4, 1.0),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 1.0, 1.0),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 1.0, 0.5),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 0.4, 1.0),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 1.0, 1.0),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 1.0, 0.5),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 0.4, 1.0),
                    ]
                colors.append(ci)
        else:
            colors = [['b', 'r', 'g', 'c', 'm', 'y', 'k']]*len(data)

        model: ReflectivityModule = self.plugin.GetModel().script_module
        self.plot_dicts = []
        while len(self.ax.lines)>0:
            self.ax.lines[0].remove()
        while len(self.ax.collections)>0:
            self.ax.collections[0].remove()
        i = 0
        sld_units = []

        if self.plugin.sim_returns_sld and model._sim:
            # New style sim function with one sld for each simulation
            self.plot_dicts = model.SLD
            for sim in range(len(self.plot_dicts)):
                if data[sim].show:
                    for key in self.plot_dicts[sim]:
                        try:
                            if key in ['z', 'SLD unit'] or (self.plot_dicts[0][key]==0).all():
                                # skip lines that are all zero to keep legend cleaner
                                continue
                        except KeyError:
                            pass
                        is_imag = key[:2]=='Im' or key[:4]=='imag'
                        if (is_imag and self.opt.show_imag) or not is_imag:
                            if self.opt.show_single_model:
                                label = key
                            else:
                                label = data[sim].name+'\n'+key
                            self.ax.plot(self.plot_dicts[sim]['z'], self.plot_dicts[sim][key],
                                         color=colors[sim][i%len(colors[sim])], label=label)

                            if 'SLD unit' in self.plot_dicts[sim]:
                                if not self.plot_dicts[sim]['SLD unit'] in sld_units:
                                    sld_units.append(self.plot_dicts[sim]['SLD unit'])
                            i += 1
                if self.opt.show_single_model:
                    break
        else:
            # Old style plotting just one sld
            if self.plugin.GetModel().compiled:
                try:
                    sample = model.sample
                except AttributeError:
                    iprint("Warning: Could not locate the sample in the model")
                    return
                plot_dict = sample.SimSLD(None, None, model.inst)
                self.plot_dicts = [plot_dict]
                for key in self.plot_dicts[0]:
                    if key in ['z', 'SLD unit'] or (self.plot_dicts[0][key]==0).all():
                        # skip lines that are all zero to keep legend cleaner
                        continue
                    is_imag = key[:2]=='Im' or key[:4]=='imag'
                    if (is_imag and self.opt.show_imag) or not is_imag:
                        label = key
                        self.ax.plot(self.plot_dicts[0]['z'], self.plot_dicts[0][key],
                                     colors[-1][i%len(colors[-1])], label=label)

                        if 'SLD unit' in self.plot_dicts[0]:
                            if not self.plot_dicts[0]['SLD unit'] in sld_units:
                                sld_units.append(self.plot_dicts[0]['SLD unit'])
                        i += 1

        if i>0:
            if self.opt.legend_outside:
                self.ax.legend(loc='center left', framealpha=0.5, bbox_to_anchor=(1, 0.5), fontsize="small", ncol=1)
            else:
                self.ax.legend(loc='upper right', framealpha=0.5, fontsize="small", ncol=1)
            sld_unit = ', '.join(sld_units)
            self.ax.yaxis.label.set_text('$\mathrm{\mathsf{SLD\,[%s]}}$'%sld_unit)
            self.ax.xaxis.label.set_text('$\mathrm{\mathsf{ z\,[\AA]}}$')
            wx.CallAfter(self.flush_plot)
            self.AutoScale()

    def SavePlotData(self, filename):
        ''' Save all the SLD profiles to file with filename.'''
        # Check so that there are a simulation to save
        try:
            self.plot_dicts
        except:
            self.plugin.ShowWarningDialog('No SLD data to save.'
                                          ' Simulate the model first and then save.')
            return
        base, ext = os.path.splitext(filename)
        if ext=='':
            ext = '.dat'
        data = self.plugin.GetModel().get_data()
        for sim in range(len(self.plot_dicts)):
            new_filename = (base+'%03d'%sim+ext)
            save_array = np.array([self.plot_dicts[sim]['z']])
            header = ' z [Å]'+' '*12
            for key in self.plot_dicts[sim]:
                if key!='z' and key!='SLD unit':
                    save_array = np.r_[save_array, [self.plot_dicts[sim][key]]]
                    header += f' {key:19s}'
            with open(new_filename, 'w', encoding='utf-8') as f:
                f.write("# File exported from GenX's Reflectivity plugin\n")
                f.write("# File created: %s\n"%time.ctime())
                f.write("# Simulated SLD for data set: %s\n"%data[sim].name)
                if 'SLD unit' in self.plot_dicts[sim]:
                    sld_unit = self.plot_dicts[sim]['SLD unit']
                    sld_unit = sld_unit.replace('10^{-6}\\AA^{-2}', '10⁻⁶ Å⁻²')
                    sld_unit = sld_unit.replace('r_{e}/\\AA^{3}', 'rₑ/Å⁻³')
                    f.write('# SLD unit: %s\n'%sld_unit)
                f.write("# Coumns: \n")
                f.write('#'+header+'\n')
                np.savetxt(f, save_array.T, fmt='%-19.12e')

    def generate_sld_distribution(self, reference_interface=0, number_sample=1000):
        """
        Create SLD confidence interval from a sample of random parameters chosen from
        the parameter errors. A reference surface defines the zero point for the comparison.
        """
        model_obj: Model = self.plugin.GetModel()
        parameters = model_obj.parameters
        model_module: ReflectivityModule = self.plugin.GetModel().script_module

        param_funcs, best_params, par_min, par_max = model_obj.get_fit_pars(use_bounds=False)
        param_edown, param_eup = np.array(parameters.get_error_pars()).T
        NP = len(param_funcs)

        initial_SLDs = model_module.SLD
        # setup empty lists for each element of the model SLD
        data = []
        zmin = 1e6
        zmax = -1e6
        for SLDi in initial_SLDs:
            data.append(dict([(key, []) for key in SLDi if key not in ['z', 'SLD unit']]))
            zmin = min(SLDi['z'].min(), zmin)
            zmax = max(SLDi['z'].max(), zmax)
        # create a general z-range for all simulations
        zrng = zmax-zmin
        z = np.arange(round(zmin-0.1*zrng), round(zmax+0.1*zrng), 1.0)

        if reference_interface!=0:
            d = model_module.sample.resolveLayerParameters()['d']
            h = np.cumsum(d)
            z -= h[reference_interface]
            for SLDi in initial_SLDs:
                SLDi['z'] -= h[reference_interface]

        for i in range(number_sample):
            srnd = np.random.randn(NP)
            dp = np.where(srnd>0., param_eup, param_edown)*abs(srnd)
            pi = best_params+dp
            for j, fj in enumerate(param_funcs):
                fj(pi[j])
            model_module.SLD = []
            model_module.Sim(model_obj.data)

            for i, SLDi in enumerate(model_module.SLD):
                if reference_interface!=0:
                    d = model_module.sample.resolveLayerParameters()['d']
                    h = np.cumsum(d)
                    SLDi['z'] -= h[reference_interface]

                for key, value in SLDi.items():
                    if key in ['z', 'SLD unit']:
                        continue
                    zfun = interp1d(SLDi['z'], value,
                                    fill_value=(value[0], value[-1]), bounds_error=False, kind='linear')
                    data[i][key].append(zfun(z))

        # calculate the SLD curve range that falls within the 1-sigma interval of 68.2% probability
        output = []
        for i, SLDi in enumerate(data):
            output.append({'z': z, 'SLD unit': initial_SLDs[i]['SLD unit']})
            for key, value in SLDi.items():
                zfun = interp1d(initial_SLDs[i]['z'], initial_SLDs[i][key],
                                fill_value=(initial_SLDs[i][key][0], initial_SLDs[i][key][-1]),
                                bounds_error=False, kind='linear')

                output[-1][key] = (np.percentile(np.array(value), 2.5, axis=0),  # lower bound 2-sigma
                                   np.percentile(np.array(value), 15.9, axis=0),  # lower bound 1-sigma
                                   zfun(z),  # best parameter line
                                   np.percentile(np.array(value), 84.1, axis=0),  # upper bound 1-sigma
                                   np.percentile(np.array(value), 97.5, axis=0))  # upper bound 2-sigma
        return output

    def PlotConfidence(self, reference_interface=0, number_sample=1000, plot_data=None):
        '''
        Plotting the sample.
        '''
        if not self.ax:
            # self.ax = self.figure.add_subplot(111)
            self.create_axes()

        data: DataList = self.plugin.GetModel().get_data()
        if self.opt.data_derived_color:
            colors = []
            for di in data:
                h, s, v = colorsys.rgb_to_hsv(*di.sim_color)
                ci = [
                    colorsys.hsv_to_rgb(h, 1.0, 1.0),
                    colorsys.hsv_to_rgb(h, 1.0, 0.5),
                    colorsys.hsv_to_rgb(h, 0.4, 1.0),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 1.0, 1.0),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 1.0, 0.5),
                    colorsys.hsv_to_rgb((h+0.1)%1.0, 0.4, 1.0),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 1.0, 1.0),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 1.0, 0.5),
                    colorsys.hsv_to_rgb((h-0.1)%1.0, 0.4, 1.0),
                    ]
                colors.append(ci)
        else:
            colors = [['b', 'r', 'g', 'c', 'm', 'y', 'k']]*len(data)

        while len(self.ax.lines)>0:
            self.ax.lines[0].remove()
        while len(self.ax.collections)>0:
            self.ax.collections[0].remove()

        if plot_data is None:
            plot_data = self.generate_sld_distribution(reference_interface, number_sample)
        i = 0
        sld_units = []

        # New style sim function with one sld for each simulation
        for sim in range(len(plot_data)):
            if data[sim].show:
                for key in plot_data[sim]:
                    try:
                        if key in ['z', 'SLD unit'] or (plot_data[0][key][2]==0).all():
                            # skip lines that are all zero to keep legend cleaner
                            continue
                    except KeyError:
                        pass
                    is_imag = key[:2]=='Im' or key[:4]=='imag'
                    if (is_imag and self.opt.show_imag) or not is_imag:
                        if self.opt.show_single_model:
                            label = key
                        else:
                            label = data[sim].name+'\n'+key
                        # 2-sigma range
                        self.ax.fill_between(plot_data[sim]['z'],
                                             plot_data[sim][key][0],
                                             plot_data[sim][key][4],
                                             color=colors[sim][i%len(colors[sim])], alpha=0.25)
                        # 1-sigma range
                        self.ax.fill_between(plot_data[sim]['z'],
                                             plot_data[sim][key][1],
                                             plot_data[sim][key][3],
                                             color=colors[sim][i%len(colors[sim])], alpha=0.5)
                        # fit result line
                        self.ax.plot(plot_data[sim]['z'], plot_data[sim][key][2],
                                     color=colors[sim][i%len(colors[sim])], label=label)

                        if 'SLD unit' in plot_data[sim]:
                            if not plot_data[sim]['SLD unit'] in sld_units:
                                sld_units.append(plot_data[sim]['SLD unit'])
                        i += 1
            if self.opt.show_single_model:
                break

        if i>0:
            if self.opt.legend_outside:
                self.ax.legend(loc='center left', framealpha=0.5, bbox_to_anchor=(1, 0.5), fontsize="small", ncol=1)
            else:
                self.ax.legend(loc='upper right', framealpha=0.5, fontsize="small", ncol=1)
            sld_unit = ', '.join(sld_units)
            self.ax.yaxis.label.set_text('$\mathrm{\mathsf{SLD\,[%s]}}$'%sld_unit)
            self.ax.xaxis.label.set_text('$\mathrm{\mathsf{ z\,[\AA]}}$')
            wx.CallAfter(self.flush_plot)
            self.AutoScale()

    def SaveConfidenceData(self, filename, reference_interface=0, number_sample=1000, do_plot=False):
        ''' Save all the SLD confidence intervals to file with filename.'''
        plot_data = self.generate_sld_distribution(reference_interface, number_sample)
        base, ext = os.path.splitext(filename)
        if ext=='':
            ext = '.dat'
        data = self.plugin.GetModel().get_data()
        sub_names = ['-l95', '-l63', '-bst', '-h63', '-h95']
        for sim in range(len(plot_data)):
            new_filename = (base+'%03d'%sim+ext)
            save_array = np.array([plot_data[sim]['z']])
            header = ' z [Å]'+' '*12
            for key in plot_data[sim]:
                if key!='z' and key!='SLD unit':
                    for i, sn in enumerate(sub_names):
                        save_array = np.r_[save_array, [plot_data[sim][key][i]]]
                        header += f' {key+sn:19s}'
            with open(new_filename, 'w', encoding='utf-8') as f:
                f.write("# File exported from GenX's Reflectivity plugin\n")
                f.write("# File created: %s\n"%time.ctime())
                f.write("# Simulated SLD with confidence intervals for data set: %s\n"%data[sim].name)
                f.write("# Reference interface/Number of Samples: %i/%i\n"%(reference_interface, number_sample))
                f.write("# Data describes confidence intervals given as 2-sigma (95%%), "
                        "1-sigma (68%%) and best estimate from fit.")
                if 'SLD unit' in plot_data[sim]:
                    sld_unit = plot_data[sim]['SLD unit']
                    sld_unit = sld_unit.replace('10^{-6}\\AA^{-2}', '10⁻⁶ Å⁻²')
                    sld_unit = sld_unit.replace('r_{e}/\\AA^{3}', 'rₑ/Å⁻³')
                    f.write('# SLD unit: %s\n'%sld_unit)
                f.write("# Coumns: \n")
                f.write('#'+header+'\n')
                np.savetxt(f, save_array.T, fmt='%-19.12e')
        if do_plot:
            self.PlotConfidence(plot_data=plot_data)

    def OnContextMenu(self, event):
        '''
        Callback to show the popmenu for the plot which allows various
        settings to be made.
        '''
        menu = self.generate_context_menu()
        menu.AppendSeparator()

        legendID = wx.NewId()
        menu.AppendCheckItem(legendID, "Legend Outside")
        menu.Check(legendID, self.opt.legend_outside)

        def OnLegend(event):
            self.opt.legend_outside = not self.opt.legend_outside
            self.figure.clear()
            self.create_axes()
            self.Plot()

        self.Bind(wx.EVT_MENU, OnLegend, id=legendID)

        coloringID = wx.NewId()
        menu.AppendCheckItem(coloringID, "Color from Data")
        menu.Check(coloringID, self.opt.data_derived_color)

        def OnColoring(event):
            self.opt.data_derived_color = not self.opt.data_derived_color
            self.Plot()

        self.Bind(wx.EVT_MENU, OnColoring, id=coloringID)

        singlemodelID = wx.NewId()
        menu.AppendCheckItem(singlemodelID, "Single Sample Model")
        menu.Check(singlemodelID, self.opt.show_single_model)

        def OnSingleModel(event):
            self.opt.show_single_model = not self.opt.show_single_model
            self.Plot()

        self.Bind(wx.EVT_MENU, OnSingleModel, id=singlemodelID)

        # Time to show the menu
        self.PopupMenu(menu)
        self.WriteConfig()

        self.Unbind(wx.EVT_MENU)
        menu.Destroy()
