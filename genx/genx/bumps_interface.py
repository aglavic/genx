"""
Classes to interface with the bumps module for fitting and statistical analysis.
"""

import wx
from wx.grid import Grid, EVT_GRID_CELL_LEFT_DCLICK
import threading
from numpy import newaxis, maximum
from matplotlib.colors import LogNorm
from bumps.monitor import TimedUpdate
from bumps.fitproblem import nllf_scale
from bumps.formatnum import format_uncertainty
from bumps.dream.corrplot import _hists, _plot
from .plotpanel import PlotPanel

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
        self.ptxt.SetLabel('step: %s/%s\tcost: %s'%(history.step[0], self.pbar.GetRange(), chisq))
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

class StatisticalAnalysisDialog(wx.Dialog):
    def __init__(self, parent, model):
        wx.Dialog.__init__(self, parent, style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
        vbox=wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(vbox)

        self.ptxt=wx.StaticText(self, label="...")
        self.pbar=wx.Gauge(self, range=1000)

        hbox=wx.BoxSizer(wx.HORIZONTAL)
        vbox.Add(hbox, proportion=1, flag=wx.EXPAND)

        lpanel=wx.Panel(self)
        self.grid=Grid(self)
        self.grid.CreateGrid(len(model.parameters), len(model.parameters))
        for i in range(len(model.parameters)):
            self.grid.SetRowSize(i, 80)
            self.grid.SetColSize(i, 80)
            self.grid.SetColLabelValue(i, '%i'%i)
            self.grid.SetRowLabelValue(i, '%i'%i)
        self.grid.DisableCellEditControl()
        self.grid.Bind(EVT_GRID_CELL_LEFT_DCLICK, self.OnSelectCell)
        rpanel=PlotPanel(self) #wx.Panel(self)
        rpanel.SetMinSize((200,200))
        self.plot_panel=rpanel
        self.ax=rpanel.figure.add_subplot(111)

        hbox.Add(lpanel, proportion=0, flag=wx.EXPAND|wx.ALIGN_TOP)
        hbox.Add(self.grid, proportion=0, flag=wx.EXPAND)
        hbox.Add(rpanel, proportion=3, flag=wx.EXPAND)


        lsizer=wx.GridSizer(vgap=1, hgap=2, cols=2)
        lpanel.SetSizer(lsizer)
        self.entries={}
        for key, emin, emax, val in [('pop', 1, 20*len(model.parameters), 2*len(model.parameters)),
                                     ('samples', 1000, 10000000, 10000),
                                     ('burn', 0, 10000, len(model.parameters)*2)]:
            lsizer.Add(wx.StaticText(lpanel, label='%s:'%key), flag=wx.FIXED_MINSIZE)
            self.entries[key]=wx.SpinCtrl(lpanel, wx.ID_ANY, min=emin, max=emax, value=str(val))
            lsizer.Add(self.entries[key], flag=wx.FIXED_MINSIZE)

        lsizer.AddStretchSpacer(10)
        but=wx.Button(self, label='Run Analysis...')
        vbox.Add(but)
        vbox.Add(self.ptxt, proportion=0, flag=wx.EXPAND)
        vbox.Add(self.pbar, proportion=0, flag=wx.EXPAND)

        self.Bind(wx.EVT_BUTTON, self.OnRunAnalysis, but)

        self.model=model
        self.thread=None

        psize=parent.GetSize()
        self.SetSize(int(psize.GetWidth()*0.75), int(psize.GetHeight()*0.75))

    def OnRunAnalysis(self, event):
        self.thread=threading.Thread(target=self.run_bumps)
        self.thread.start()

    def run_bumps(self):
        self.bproblem=self.model.bumps_problem()
        mon=ProgressMonitor(self.bproblem, self.pbar, self.ptxt)
        pop=self.entries['pop'].GetValue()
        burn=self.entries['burn'].GetValue()
        samples=self.entries['samples'].GetValue()
        self.pbar.SetRange(int(samples/(len(self.bproblem.model_parameters())*pop))+burn)
        res=self.model.bumps_fit(method='dream',
                                 pop=pop, samples=samples, burn=burn,
                                 thin=1, alpha=0, outliers='none', trim=False,
                                 monitors=[mon], problem=self.bproblem)
        self._res=res
        wx.CallAfter(self.display_bumps)

    def display_bumps(self):
        res=self._res
        self.draw=res.state.draw()
        self.rel_cov=res.cov/res.dx[:,newaxis]/res.dx[newaxis,:]
        rel_max=[0, 1, abs(self.rel_cov[0,1])]
        for i, ci in enumerate(self.rel_cov):
            self.grid.SetColLabelValue(i, self.draw.labels[i])
            self.grid.SetRowLabelValue(i, self.draw.labels[i])
            for j, cj in enumerate(ci):
                self.grid.SetCellValue(i, j, "%.6f"%cj)
                self.grid.SetReadOnly(i, j)
                if i==j:
                    self.grid.SetCellBackgroundColour(i, j, "#888888")
                elif abs(cj)>0.4:
                    self.grid.SetCellBackgroundColour(i, j, "#ffcccc")
                elif abs(cj)>0.3:
                    self.grid.SetCellBackgroundColour(i, j, "#ffdddd")
                elif abs(cj)>0.2:
                    self.grid.SetCellBackgroundColour(i, j, "#ffeeee")
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

    def OnSelectCell(self, evt):
        i, j=evt.GetCol(), evt.GetRow()
        if i==j:
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
