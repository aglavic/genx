"""
Classes to interface with the bumps module for fitting and statistical analysis.
"""

import wx
import threading
from bumps.monitor import TimedUpdate
from bumps.fitproblem import nllf_scale
from bumps.formatnum import format_uncertainty

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
        vbox.Add(hbox, proportion=0, flag=wx.EXPAND)

        lpanel=wx.Panel(self)
        rpanel=wx.Panel(self)
        hbox.Add(lpanel, proportion=1, flag=wx.EXPAND)
        hbox.Add(rpanel, proportion=1, flag=wx.EXPAND)

        lsizer=wx.GridSizer(vgap=1, hgap=2, cols=2)
        lpanel.SetSizer(lsizer)
        self.entries={}
        for key, emin, emax, val in [('pop', 1, 150, 15),
                                     ('samples', 1000, 10000000, 100000),
                                     ('burn', 0, 10000, 100)]:
            lsizer.Add(wx.StaticText(lpanel, label='%s:'%key))
            self.entries[key]=wx.SpinCtrl(lpanel, wx.ID_ANY, min=emin, max=emax, value=str(val))
            lsizer.Add(self.entries[key], flag=wx.EXPAND)

        vbox.AddStretchSpacer(1)
        but=wx.Button(self, label='Run Analysis...')
        vbox.Add(but)
        vbox.Add(self.ptxt, proportion=0, flag=wx.EXPAND)
        vbox.Add(self.pbar, proportion=0, flag=wx.EXPAND)

        self.Bind(wx.EVT_BUTTON, self.OnRunAnalysis, but)

        self.model=model
        self.thread=None

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
