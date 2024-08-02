"""<h1>Exporter</h1>
A plugin that allows to export models for the use in other programs.
Currently, only BornAgain is supported to easy transfer from spcular
to off-specular or GISAS simulations.

"""

import wx

from .. import add_on_framework as framework


class Plugin(framework.Template):
    def __init__(self, parent):
        framework.Template.__init__(self, parent)
        self.menu = self.NewMenu("Exporter")
        self.parent = parent

        self.mb_export_ba = wx.MenuItem(
            self.menu,
            wx.NewId(),
            "BornAgain script...",
            "Export reflectometry layers to BornAgain python script.",
            wx.ITEM_NORMAL,
        )
        self.menu.Append(self.mb_export_ba)
        self.parent.Bind(wx.EVT_MENU, self.OnExportBA, self.mb_export_ba)

        self.StatusMessage("Sucessfully loaded Exporter...")

    def OnExportBA(self, event):
        """Export layer model to BornAgain script."""
        dlg = wx.FileDialog(
            self.parent,
            message="Export As",
            defaultFile="genx_model.py",
            wildcard="Python (*.py)|*.py",
            style=wx.FD_SAVE,  # | wx.FD_CHANGE_DIR
        )
        if dlg.ShowModal() != wx.ID_OK:
            return
        fname = dlg.GetPath()

        model = self.GetModel()
        model.simulate()
        m = model.script_module
        names = list(m.__dict__.keys())
        objects = list(m.__dict__.values())

        s = m.sample

        output = "# layer definitions: thickness, density, f-xray, b-neutron, roughness, muB/FU\n"
        output += "Ambient=" + self._expand_layer(s.Ambient) + "\n"
        output += "Stacks=[\n"
        for stack in s.Stacks:
            if stack in objects:
                name = "- Stack: " + names[objects.index(stack)]
            else:
                name = ""
            output += f"         [{stack.Repetitions:d}, #repetitions {name}\n"
            for l in stack.Layers:
                if l in objects:
                    name = "# " + names[objects.index(l)]
                else:
                    name = ""
                output += "           " + self._expand_layer(l) + f",{name}\n"
            output += "         ],\n"
        output += "       ]\n"
        output += "Substrate=" + self._expand_layer(s.Substrate) + "\n\n"
        output += f'probe="{m.inst.probe}"'

        open(fname, "w", encoding="utf-8").write(TEMPLATE % output)

    def _expand_layer(self, layer):
        return f"[{layer.d}, {layer.dens}, {layer.f}, {layer.b}, {layer.sigma}, {layer.magn}]"

    def clear_menu(self):
        """clear_menu(self) --> None

        Clears the menu from all items present in it
        """
        [self.menu.RemoveItem(item) for item in self.menu.GetMenuItems()]


TEMPLATE = '''import bornagain as ba
from bornagain import angstrom, deg, nm

%s

muB=9274.0101e8 #BohrMagneton/1e-5nm³ to A∕m

def get_sample():
    """
    Builds the sample from the defined GenX model.
    """
    hurst=0.4
    corr=100.*nm
    magnetizationVector    = ba.kvector_t(0, muB,0)
    multi_layer = ba.MultiLayer()
    
    if probe=="x-ray":
        midx=2
    else:
        midx=3

    m_ambient = ba.MaterialBySLD("Ambient", 1e-5*Ambient[1]*Ambient[midx].real, -1e-5*Ambient[1]*Ambient[midx].imag)
    multi_layer.addLayer(ba.Layer(m_ambient))
    
    for i, stack in enumerate(Stacks):
        j=0
        for rep in range(stack[0]):
            for li in stack[1:]:
                if 'pol' in probe and li[5]>0:
                    material=ba.MaterialBySLD("S%%iL%%i"%%(i,j), 1e-5*li[1]*li[midx].real, -1e-5*li[1]*li[midx].imag, 1e-5*magnetizationVector*li[1]*li[5])
                else:
                    material=ba.MaterialBySLD("S%%iL%%i"%%(i,j), 1e-5*li[1]*li[midx].real, -1e-5*li[1]*li[midx].imag)
                if li[4]>0:
                    roughness=ba.LayerRoughness(li[4]*angstrom, hurst, corr)
                    multi_layer.addLayerWithTopRoughness(ba.Layer(material, li[0]*angstrom), roughness)
                else:
                    multi_layer.addLayer(ba.Layer(material, li[0]*angstrom))
                j+=1
    
    
    m_substrate = ba.MaterialBySLD("Substrate", 1e-5*Substrate[1]*Substrate[midx].real, -1e-5*Substrate[1]*Substrate[midx].imag)
    if Substrate[4]>0:
        roughness=ba.LayerRoughness(Substrate[4]*angstrom, hurst, corr)
        multi_layer.addLayerWithTopRoughness(ba.Layer(m_substrate), roughness)
    else:
        multi_layer.addLayer(ba.Layer(m_substrate))
    
    return multi_layer

def get_simulation(scan_size=500):
    """
    Defines and returns specular simulation
    with a qz-defined beam
    """
    qzs = np.linspace(0.01, 3.0, scan_size)  # qz-values
    dq = 0.03 * qzs
    n_sig = 2.0
    n_samples = 25

    distr = ba.RangedDistributionGaussian(n_samples, n_sig)

    scan = ba.QSpecScan(qzs)
    scan.setAbsoluteQResolution(distr, dq)

    simulation = ba.SpecularSimulation()
    simulation.setScan(scan)

    return simulation

def run_simulation(polarization=ba.kvector_t(0.0, 1.0, 0.0),
                   analyzer=None):
    """
    Runs simulation and returns its result.
    """
    sample = get_sample()
    simulation = get_simulation()
    
    # adding polarization and analyzer operator
    if polarization:
        simulation.setBeamPolarization(polarization)
    if analyzer:
      simulation.setAnalyzerProperties(analyzer, 1.0, 0.5)
    
    simulation.setSample(sample)
    simulation.runSimulation()
    return simulation.result()

if __name__=='__main__':
    from numpy import array
    from matplotlib.pyplot import *
    smpl=get_sample()
    figure(figsize=(16,8))
    subplot(121)
    title('Reflectivity %%s'%%probe)
    if 'pol' in probe:
        ba.plot_simulation_result(run_simulation(polarization=ba.kvector_t(0.0, 1.0, 0.0)), 
            label='up', postpone_show=True)
        ba.plot_simulation_result(run_simulation(polarization=ba.kvector_t(0.0, -1.0, 0.0)), 
            label='down', postpone_show=True)
        legend()
    else:
        ba.plot_simulation_result(run_simulation(polarization=None), postpone_show=True)
    subplot(122)
    title('Scattering Length Density Profile')
    z, sld=ba.MaterialProfile(smpl)
    plot(z,array(sld).real)
    xlabel("z [nm]")
    ylabel("SLD [$\\AA^{-2}$]")
    tight_layout()
    show()
'''
