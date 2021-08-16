
from bumps.monitor import TimedUpdate
from bumps.fitproblem import nllf_scale
from bumps.formatnum import format_uncertainty

class NBMonitor(TimedUpdate):
    """
    Display fit progress on the console
    """

    def __init__(self, problem, progress=0.25, improvement=5.0):
        TimedUpdate.__init__(self, progress=progress, improvement=improvement)
        from ipywidgets import Output, HTML, HBox, VBox
        from IPython.display import display
        self.problem=problem
        self.status_text=HTML()
        self.result_text=HTML()
        self.plot_out=Output()

        vbox=VBox([self.status_text, self.result_text])
        hbox=HBox([vbox, self.plot_out])
        display(hbox)
        self.steps=[]
        self.chis=[]

    def show_progress(self, history):
        scale, err=nllf_scale(self.problem)
        chisq=format_uncertainty(scale*history.value[0], err)
        self.status_text.value='<table width="50%%"><tr><td>step</td><td>%s</td><td>cost</td><td>%s</td></tr></table>'%(
        history.step[0], chisq)
        self.steps.append(history.step[0])
        self.chis.append(scale*history.value[0])

    def show_improvement(self, history):
        from IPython.display import display, clear_output
        from matplotlib.pyplot import figure, plot, close
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

