"""
Classes to interface with the bumps module for fitting and statistical analysis.
"""

import threading
from numpy import newaxis, maximum
from matplotlib.colors import LogNorm

import wx
from wx.grid import Grid, EVT_GRID_CELL_LEFT_DCLICK

from bumps import __version__ as bumps_version
from bumps.monitor import TimedUpdate
from bumps.fitproblem import nllf_scale
from bumps.formatnum import format_uncertainty
from bumps.dream.corrplot import _hists

from .plotpanel import PlotPanel, BasePlotConfig
from .exception_handling import CatchModelError
from ..bumps_optimizer import BumpsResult

class ProgressMonitor(TimedUpdate):
    """
    Display fit progress on the console
    """

    def __init__(self, problem, pbar, ptxt, progress=0.25, improvement=5.0):
        TimedUpdate.__init__(self, progress=progress, improvement=improvement)
        self.problem=problem
        self.pbar=pbar  # wx.Gauge
        self.ptxt=ptxt  # wx.StaticText
        self.chis=[]
        self.steps=[]

    def show_progress(self, history):
        scale, err=nllf_scale(self.problem)
        chisq=format_uncertainty(scale*history.value[0], err)
        self.ptxt.SetLabel('step: %s/%s  cost: %s'%(history.step[0], self.pbar.GetRange(), chisq))
        self.pbar.SetValue(history.step[0])
        self.steps.append(history.step[0])
        self.chis.append(scale*history.value[0])

    def show_improvement(self, history):
        return
        p=self.problem.getp()
        try:
            self.problem.setp(history.point[0])
            out='<table width="50%"><tr>'
            out+="</tr>\n<tr>".join(
                ["<td>%s</td><td>%s</td><td>%s</td>"%(pi.name, pi.value, pi.bounds) for pi in self.problem._parameters])
            self.result_text.value=out+"</td></tr></table>"
        finally:
            self.problem.setp(p)
        with self.plot_out:
            fig=figure()
            plot(self.steps, self.chis)
            clear_output()
            display(fig)
            close()

class StatisticsPanelConfig(BasePlotConfig):
    section='statistics plot'

class StatisticalAnalysisDialog(wx.Dialog):
    rel_cov=None
    thread: threading.Thread
    _res: BumpsResult

    def __init__(self, parent, model, prev_result: BumpsResult=None):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        vbox=wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)
        self.SetTitle('Statistical Analysis of Parameters')

        dpi_scale_factor=wx.GetApp().dpi_scale_factor

        self.ptxt=wx.StaticText(self, label="...")
        self.pbar=wx.Gauge(self, range=1000)

        hbox=wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(hbox, proportion=1, flag=wx.EXPAND)

        lpanel=wx.Panel(self)
        gpanel=wx.Panel(self)
        gbox=wx.BoxSizer(wx.VERTICAL)
        gpanel.SetSizer(gbox)
        self.grid=Grid(gpanel)
        gbox.Add(wx.StaticText(gpanel, label='Estimated covariance matrix:'),
                 proportion=0, flag=wx.FIXED_MINSIZE)
        gbox.Add(self.grid, proportion=1, flag=wx.EXPAND)
        self.normalize_checkbox=wx.CheckBox(gpanel, label='Normalize value (σ_ij/σ_i/σ_j)')
        self.normalize_checkbox.SetValue(True)
        self.normalize_checkbox.Bind(wx.EVT_CHECKBOX, self.OnToggleNormalize)
        gbox.Add(self.normalize_checkbox, proportion=0, flag=wx.FIXED_MINSIZE)

        nfparams=len([pi for pi in model.parameters if pi.fit])
        self.grid.CreateGrid(nfparams+1, nfparams)
        self.grid.SetColLabelTextOrientation(wx.VERTICAL)
        self.grid.SetColLabelSize(int(dpi_scale_factor*80))
        for i in range(nfparams):
            self.grid.SetRowSize(i, int(dpi_scale_factor*80))
            self.grid.SetColSize(i, int(dpi_scale_factor*80))
            self.grid.SetColLabelValue(i, '%i'%i)
            self.grid.SetRowLabelValue(i+1, '%i'%i)
        self.grid.SetRowSize(nfparams, int(dpi_scale_factor*80))
        self.grid.DisableCellEditControl()
        self.grid.SetMinSize((int(dpi_scale_factor*200),int(dpi_scale_factor*200)))
        self.grid.Bind(EVT_GRID_CELL_LEFT_DCLICK, self.OnSelectCell)
        rpanel=PlotPanel(self, config_class=StatisticsPanelConfig)
        rpanel.SetMinSize((int(dpi_scale_factor*200),int(dpi_scale_factor*200)))
        self.plot_panel=rpanel
        self.ax=rpanel.figure.add_subplot(111)

        hbox.Add(lpanel, proportion=0, flag=wx.EXPAND|wx.ALIGN_TOP)
        hbox.Add(gpanel, proportion=1, flag=wx.EXPAND)
        hbox.Add(rpanel, proportion=1, flag=wx.EXPAND)


        lsizer=wx.GridSizer(vgap=1, hgap=2, cols=2)
        lpanel.SetSizer(lsizer)
        self.entries={}
        for key, emin, emax, val in [('pop', 1, 20*nfparams, 2*nfparams),
                                     ('samples', 1000, 10000000, 10000),
                                     ('burn', 0, 10000, 200)]:
            lsizer.Add(wx.StaticText(lpanel, label='%s:'%key), flag=wx.FIXED_MINSIZE)
            self.entries[key]=wx.SpinCtrl(lpanel, wx.ID_ANY, min=emin, max=emax, value=str(val))
            lsizer.Add(self.entries[key], flag=wx.FIXED_MINSIZE)

        lsizer.AddStretchSpacer(10)
        self.run_button=wx.Button(self, label='Run Analysis...')
        vbox.Add(self.run_button)
        vbox.Add(self.ptxt, proportion=0, flag=wx.EXPAND)
        vbox.Add(self.pbar, proportion=0, flag=wx.EXPAND)

        self.Bind(wx.EVT_BUTTON, self.OnRunAnalysis, self.run_button)
        self.Bind(wx.EVT_CLOSE, self.OnClose)

        self.model=model
        self.thread=None

        psize=parent.GetSize()
        self.SetSize(int(psize.GetWidth()*0.75), int(psize.GetHeight()*0.75))

        if prev_result is not None:
            self._res=prev_result
            self.bproblem=prev_result.bproblem
            self.display_bumps()

    def OnRunAnalysis(self, event):
        if self.thread is not None:
            self.run_button.SetLabel('Run Analysis...')
            self.bproblem.fitness.stop_fit = True
            return
        self.thread=threading.Thread(target=self.run_bumps)
        self.thread.start()
        self.run_button.SetLabel('Stop Run')

    def run_bumps(self):
        self.bproblem=self.model.bumps_problem()
        mon=ProgressMonitor(self.bproblem, self.pbar, self.ptxt)
        pop=self.entries['pop'].GetValue()
        burn=self.entries['burn'].GetValue()
        samples=self.entries['samples'].GetValue()
        self.pbar.SetRange(int(samples/(len(self.bproblem.model_parameters())*pop))+burn)

        with CatchModelError(self, 'bumps_modeling') as mgr:
            res=self.model.bumps_fit(method='dream',
                                     pop=pop, samples=samples, burn=burn,
                                     thin=1, alpha=0, outliers='none', trim=False,
                                     monitors=[mon], problem=self.bproblem)

        self.run_button.SetLabel('Run Analysis...')
        if mgr.successful:
            self._res=res
            wx.CallAfter(self.display_bumps)

    def display_bumps(self):
        if self.thread is not None:
            self.thread.join(timeout=5.0)
            self.thread=None
        self.pbar.SetValue(0)

        res=self._res
        self.draw=res.state.draw()
        pnames=list(self.bproblem.model_parameters().keys())
        sort_indices=[pnames.index(ni) for ni in self.draw.labels]

        self.abs_cov=res.cov
        self.rel_cov=res.cov/res.dx[:,newaxis]/res.dx[newaxis,:]
        rel_max=[0, 1, abs(self.rel_cov[0,1])]
        if self.normalize_checkbox.IsChecked():
            display_cov=self.rel_cov
            fmt="%.6f"
        else:
            display_cov=self.abs_cov
            fmt="%.4g"
        self.grid.SetRowLabelValue(0, 'Value/Error:')
        for i, ci in enumerate(self.rel_cov):
            plabel='\n'.join(self.draw.labels[i].rsplit('_', 1))
            self.grid.SetColLabelValue(sort_indices[i], plabel)
            self.grid.SetRowLabelValue(sort_indices[i]+1, plabel)
            self.grid.SetCellValue(0, sort_indices[i], "%.8g\n%.4g"%(res.x[i], res.dx[i]))
            self.grid.SetCellAlignment(0, sort_indices[i], wx.ALIGN_CENTRE, wx.ALIGN_CENTRE)
            self.grid.SetReadOnly(0, sort_indices[i])
            self.grid.SetCellBackgroundColour(0, sort_indices[i], "#cccccc")
            for j, cj in enumerate(ci):
                self.grid.SetCellValue(sort_indices[i]+1, sort_indices[j], fmt%display_cov[i,j])
                self.grid.SetReadOnly(sort_indices[i]+1, sort_indices[j])
                self.grid.SetCellAlignment(sort_indices[i]+1, sort_indices[j], wx.ALIGN_CENTRE, wx.ALIGN_CENTRE)
                if i==j:
                    self.grid.SetCellBackgroundColour(sort_indices[i]+1, sort_indices[j], "#888888")
                elif abs(cj)>0.4:
                    self.grid.SetCellBackgroundColour(sort_indices[i]+1, sort_indices[j], "#ffcccc")
                elif abs(cj)>0.3:
                    self.grid.SetCellBackgroundColour(sort_indices[i]+1, sort_indices[j], "#ffdddd")
                elif abs(cj)>0.2:
                    self.grid.SetCellBackgroundColour(sort_indices[i]+1, sort_indices[j], "#ffeeee")
                if i!=j and abs(cj)>rel_max[2]:
                    rel_max=[min(i, j), max(i, j), abs(cj)]
        self.hists=_hists(self.draw.points.T, bins=50)

        fig=self.plot_panel.figure
        fig.clear()
        ax=fig.add_subplot(111)
        data, x, y=self.hists[(rel_max[0], rel_max[1])]
        vmin, vmax=data[data>0].min(), data.max()
        ax.pcolorfast(y, x, maximum(vmin, data), norm=LogNorm(vmin, vmax), cmap='inferno')
        ax.set_xlabel(self.draw.labels[rel_max[1]])
        ax.set_ylabel(self.draw.labels[rel_max[0]])
        self.plot_panel.flush_plot()

        # add analysis data do model for later storange in export header
        exdict={}
        exdict['library']='bumps'
        exdict['version']=bumps_version
        exdict['settings']=dict(pop=self.entries['pop'].GetValue(),
                                burn=self.entries['burn'].GetValue(),
                                samples=self.entries['samples'].GetValue())
        exdict['parameters']=[
            dict(name=li, value=float(xi), error=float(dxi),
                 cross_correlations=dict((self.draw.labels[j], float(res.cov[i,j]))
                                         for j in range(len(res.x))))
            for i, (li, xi, dxi) in enumerate(zip(self.draw.labels, res.x, res.dx))
            ]
        self.model.extra_analysis['statistics_mcmc']=exdict

    def OnToggleNormalize(self, evt):
        if self.rel_cov is None:
            evt.Skip()
            return
        if self.normalize_checkbox.IsChecked():
            display_cov=self.rel_cov
            fmt="%.6f"
        else:
            display_cov=self.abs_cov
            fmt="%.4g"
        pnames=list(self.bproblem.model_parameters().keys())
        sort_indices=[pnames.index(ni) for ni in self.draw.labels]
        for i, ci in enumerate(self.rel_cov):
            for j, cj in enumerate(ci):
                self.grid.SetCellValue(sort_indices[i]+1, sort_indices[j], fmt%display_cov[i,j])

    def OnSelectCell(self, evt):
        ri, rj=evt.GetCol(), evt.GetRow()-1
        pnames=list(self.bproblem.model_parameters().keys())
        reverse_indices=[self.draw.labels.index(ni) for ni in pnames]
        i=reverse_indices[ri]
        j=reverse_indices[rj]
        if i==j or j==-1:
            return
        elif i>j:
            itmp=i
            i=j
            j=itmp

        fig=self.plot_panel.figure
        fig.clear()
        ax=fig.add_subplot(111)
        data, x, y=self.hists[(i,j)]
        vmin, vmax=data[data>0].min(), data.max()
        ax.pcolorfast(y, x, maximum(vmin, data), norm=LogNorm(vmin, vmax), cmap='inferno')
        ax.set_xlabel(self.draw.labels[j])
        ax.set_ylabel(self.draw.labels[i])
        self.plot_panel.flush_plot()

    def OnClose(self, event):
        if self.thread is not None and self.thread.is_alive():
            # a running bumps simulation can't be interrupted from other thread, stop window from closing
            self.ptxt.SetLabel("Simulation running")
            self.bproblem.fitness.stop_fit=True
            self.thread.join(timeout=5.0)
        event.Skip()
