import warnings
from dataclasses import dataclass
from logging import debug

import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from numpy import (
    arange,
    arccos,
    arcsin,
    arctan,
    array,
    cos,
    exp,
    hstack,
    linspace,
    log,
    log10,
    pi,
    sin,
    sqrt,
    tan,
    vstack,
)
from PySide6 import QtCore, QtGui, QtWidgets

from ..core.config import BaseConfig, Configurable
from ..data import DataList
from .script_editor import GenxScriptEditor


class SimplePlotPanel(QtWidgets.QWidget):
    ax: matplotlib.axes.Axes

    def __init__(self, parent, dpi=None, **kwargs):
        super().__init__(parent, **kwargs)
        if dpi is None:
            dpi = self.logicalDpiX()
        self.parent = parent
        debug("init SimplePlotPanel - setup figure")
        self.figure = Figure(figsize=(1.0, 1.0), dpi=dpi)
        debug("init SimplePlotPanel - setup canvas")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.ax = self.figure.add_subplot(111)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas, 1)

    def clear(self):
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)

    def flush_plot(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                self.figure.tight_layout(h_pad=0)
            except ValueError:
                pass
        self.canvas.draw()


@dataclass
class PublicationConfig(BaseConfig):
    section = "publication graph"

    width: float = 3.39
    heigth: float = 2.36
    font_size: int = int(matplotlib.rcParams["font.size"])
    font_familty: int = 0
    font_face: str = matplotlib.rcParams["font.sans-serif"][0]

    start_text: str = repr(
        """## Create plots using matplotlib related to your model.
## Uncomment relevant lines to save/show different graphs.
## For other matplotlib rc style options see 
## https://matplotlib.org/stable/tutorials/introductory/customizing.html
fig.set_facecolor('white')
rcParams['font.family'] = 'sans-serif'
rcParams['font.size'] = font_size # dialog entry
rcParams['font.sans-serif'] = font_face # dialog entry
rcParams['font.weight'] = font_weight # dialog entry
rcParams['mathtext.fontset'] = 'dejavusans' # 'dejavuserif'|'cm'|'stix'|'stixsans'

#outimg = save_file_dialog('*.png') # select filename for export
############ Plot of Reflectivity curves ###############
clear()
for di in data:
    if not di.show:
        continue
    errorbar(di.x, di.y, yerr=di.error, label="data-"+di.name, **di.data_kwds, zorder=2)
    semilogy(di.x, di.y_sim, label="sim-"+di.name, **di.sim_kwds, zorder=5)
xlabel(model.__xlabel__)
ylabel(model.__ylabel__)
legend()
tight_layout(pad=0.5)
#savefig(outimg, dpi=300) # save to selected image file
show() # show in GUI graph

########### Plot of SLD profiles ##############
clear()
sld = SLD[0] # assuming all dataset represent the same model
for key, value in sld.items():
    if key in ['z', 'SLD unit', 'Mass Density']:
        continue
    plot(sld['z'], value, label=key)
xlabel("z [Å]")
ylabel(f"SLD [${sld['SLD unit']}$]")
legend()
tight_layout(pad=0.5)
#savefig(outimg[:-4]+'_sld.png', dpi=300) # save to selected image file with changed suffix
#show() # show in GUI graph

########### Plot of SLD profiles ##############
clear()
md = SLD[0]['Mass Density'] # assuming all dataset represent the same model
## replace standard line colors with colormap
## See https://matplotlib.org/stable/users/explain/colors/colormaps.html
axes.set_prop_cycle(color=cm.gist_rainbow(np.linspace(0,1,len(md.keys())-1)))
for key, value in md.items():
    if key in ['z', 'SLD unit']:
        continue
    plot(md['z'], value, label=key)
xlabel("z [Å]")
ylabel(f"density [${md['SLD unit']}$]")
legend()
tight_layout(pad=0.5)
#savefig(outimg[:-4]+'_dens.png', dpi=300) # save to image file with changed suffix
#show() # show in GUI graph

""".splitlines()
    )


class PublicationDialog(QtWidgets.QDialog, Configurable):
    opt: PublicationConfig

    def __init__(self, parent, data: DataList = None, module=None, SLD: dict = None):
        QtWidgets.QDialog.__init__(self, parent)
        Configurable.__init__(self)
        self.ReadConfig()
        self.setWindowTitle("Custom plotting for Publication")

        self.data = data
        self.module = module
        self.SLD = SLD

        main_box = QtWidgets.QHBoxLayout(self)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal, self)
        main_box.addWidget(splitter, 1)

        left_panel = QtWidgets.QWidget(splitter)
        right_panel = QtWidgets.QWidget(splitter)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 1)

        left_layout = QtWidgets.QVBoxLayout(left_panel)
        self.tinput = GenxScriptEditor(left_panel)
        self.tinput.setPlainText("\n".join(eval(self.opt.start_text)))
        left_layout.addWidget(self.tinput, 1)
        button = QtWidgets.QPushButton("plot", left_panel)
        button.clicked.connect(self.OnPlot)
        left_layout.addWidget(button, 0)

        right_layout = QtWidgets.QVBoxLayout(right_panel)
        self.plot_area = QtWidgets.QScrollArea(right_panel)
        self.plot_area.setWidgetResizable(True)
        plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_container)
        self.plot = SimplePlotPanel(plot_container)
        self.plot.setMinimumSize(300, 300)
        plot_layout.addWidget(self.plot, 1)
        self.plot_area.setWidget(plot_container)
        right_layout.addWidget(self.plot_area, 1)

        self.error_text = QtWidgets.QLabel(right_panel)
        self.error_text.setStyleSheet("background-color: #ff9696;")
        self.error_text.hide()
        right_layout.addWidget(self.error_text, 0)

        size_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(size_row)
        self.width = QtWidgets.QDoubleSpinBox(right_panel)
        self.width.setRange(0.1, 30.0)
        self.width.setSingleStep(0.01)
        self.width.setValue(self.opt.width)
        self.height = QtWidgets.QDoubleSpinBox(right_panel)
        self.height.setRange(0.1, 30.0)
        self.height.setSingleStep(0.01)
        self.height.setValue(self.opt.heigth)
        size_row.addWidget(QtWidgets.QLabel("Width (inches)"))
        size_row.addWidget(self.width)
        size_row.addWidget(QtWidgets.QLabel("Height (inches)"))
        size_row.addWidget(self.height)

        font_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(font_row)
        self.font_combo = QtWidgets.QFontComboBox(right_panel)
        self.font_combo.setCurrentFont(QtGui.QFont(self.opt.font_face))
        self.font_size = QtWidgets.QSpinBox(right_panel)
        self.font_size.setRange(4, 96)
        self.font_size.setValue(self.opt.font_size)
        font_row.addWidget(QtWidgets.QLabel("Font"))
        font_row.addWidget(self.font_combo)
        font_row.addWidget(QtWidgets.QLabel("Size"))
        font_row.addWidget(self.font_size)

        save_row = QtWidgets.QHBoxLayout()
        right_layout.addLayout(save_row)
        save_button = QtWidgets.QPushButton("store current script and settings as default", right_panel)
        save_button.clicked.connect(self.OnStoreDefaults)
        save_row.addWidget(save_button)

        self.width.valueChanged.connect(self.OnPlot)
        self.height.valueChanged.connect(self.OnPlot)
        self.font_combo.currentFontChanged.connect(self.OnPlot)
        self.font_size.valueChanged.connect(self.OnPlot)

        self.resize(800, 800)

    def SaveFromScript(self, wildcard="*.png"):
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Plot",
            "",
            f"Image files ({wildcard});;All files (*.*)",
        )
        if not filename:
            return None
        return filename

    def OnPlot(self, _event=None):
        txt = self.tinput.toPlainText()
        font = self.font_combo.currentFont()
        font.setPointSize(self.font_size.value())
        weight = "bold" if font.weight() >= QtGui.QFont.Weight.Bold else "normal"
        env = dict(
            rcParams=matplotlib.rcParams,
            data=self.data,
            model=self.module,
            SLD=self.SLD,
            show=self.plot.flush_plot,
            fig=self.plot.figure,
            canvas=self.plot.canvas,
            savefig=self.plot.figure.savefig,
            tight_layout=self.plot.figure.tight_layout,
            font_size=font.pointSize(),
            font_face=font.family(),
            font_weight=weight,
            save_file_dialog=self.SaveFromScript,
            np=np,
            cm=plt.cm,
        )

        def clear():
            self.plot.clear()
            ax = self.plot.ax
            env["plot"] = ax.plot
            env["semilogy"] = ax.semilogy
            env["errorbar"] = ax.errorbar
            env["title"] = ax.set_title
            env["xlabel"] = ax.set_xlabel
            env["ylabel"] = ax.set_ylabel
            env["legend"] = ax.legend
            env["axes"] = ax

        env["clear"] = clear

        dpi = self.logicalDpiX()
        self.plot.figure.set_dpi(dpi)
        self.plot.figure.set_size_inches(self.width.value(), self.height.value())
        px_w = int(self.width.value() * dpi)
        px_h = int(self.height.value() * dpi)
        self.plot.setMinimumSize(px_w, px_h)
        self.plot.resize(px_w, px_h)
        self.plot_area.widget().adjustSize()

        clear()
        try:
            exec(txt, env)
        except Exception as e:
            self.error_text.setText(e.__class__.__name__ + ": " + str(e))
            self.error_text.show()
        else:
            self.error_text.hide()

        matplotlib.rcdefaults()
        self.plot.flush_plot()

    def OnStoreDefaults(self, _event=None):
        self.opt.start_text = repr(self.tinput.toPlainText().splitlines())
        self.opt.width = self.width.value()
        self.opt.heigth = self.height.value()
        self.opt.font_size = self.font_size.value()
        self.opt.font_face = self.font_combo.currentFont().family()
        self.opt.font_familty = 0
        self.WriteConfig(default=True)
