"""<h1> SXRD plugin </h1>
Qt port placeholder with minimal script synchronization.
"""

from PySide6 import QtWidgets

from genx.core.custom_logging import iprint
from genx.plugins import add_on_framework as framework

from .help_modules import model_interactors as mi


class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.parent = parent
        self.setup_script_interactor()

        panel = self.NewInputFolder("SXRD")
        layout = QtWidgets.QVBoxLayout(panel)
        layout.addWidget(QtWidgets.QLabel("SXRD plugin UI is not yet ported to Qt.", panel))

        try:
            if self.GetModelScript():
                self.script_interactor.parse_code(self.GetModelScript())
        except Exception:
            iprint("SXRD plugin model could not be read.")

    def setup_script_interactor(self, model_name="sxrd2"):
        model = __import__("models.%s" % model_name, globals(), locals(), [model_name])
        preamble = (
            "import models.%s as model\nfrom models.utils import UserVars\nfrom models.symmetries import *\n"
            % model_name
        )
        script_interactor = mi.ModelScriptInteractor(preamble=preamble)
        script_interactor.add_section(
            "Instruments", mi.ObjectScriptInteractor, class_name="model.Instrument", class_impl=model.Instrument
        )
        script_interactor.add_section(
            "UnitCells", mi.ObjectScriptInteractor, class_name="model.UnitCell", class_impl=model.UnitCell
        )
        script_interactor.add_section("Slabs", mi.ObjectScriptInteractor, class_name="model.Slab", class_impl=model.Slab)
        script_interactor.add_section(
            "Domains", mi.ObjectScriptInteractor, class_name="model.Domain", class_impl=model.Domain
        )
        script_interactor.add_section(
            "Samples", mi.ObjectScriptInteractor, class_name="model.Sample", class_impl=model.Sample
        )
        self.script_interactor = script_interactor
