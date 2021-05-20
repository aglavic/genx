'''
========================================================
:mod:`auto` Data loader that selects plugin by extension
========================================================

Loads files using an other data loader that fits the file extension extracted from the file.
If none if found it falls back to the resolution loader if there res column is configured,
otherwise the default loader is used.

See the other data loaders for detailed explanation.
'''

from .resolution import Plugin as ResolutionPlugin
from .default import Plugin as DefaultPlugin
from .amor import Plugin as AmorPlugin
from .sns_mr import Plugin as SNSPlugin

class Plugin(ResolutionPlugin, DefaultPlugin):
    """
    Automatic selected data loader.

    Plugins having fixed columns have to be put into the loaders list,
    if flexible columns are used they are part of the inheritance and can only use the
    columns defined in the ResolutionPlugin class.
    """

    def __init__(self, parent):
        ResolutionPlugin.__init__(self, parent)
        self.res_col=-1
        self.loaders=[AmorPlugin(None), SNSPlugin(None)]
        self.wildcard=";".join([li.wildcard for li in self.loaders])

    def LoadData(self, dataset, filename):
        for li in self.loaders:
            if filename.endswith(li.wildcard[1:]):
                return li.LoadData(dataset, filename)
        if self.res_col<0:
            self.x_col=self.q_col
            self.y_col=self.I_col
            self.e_col=self.eI_col
            return DefaultPlugin.LoadData(self, dataset, filename)
        else:
            return ResolutionPlugin.LoadData(self, dataset, filename)
