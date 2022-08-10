"""
Helper module to support ORSO file header information to build reflectivity models.
"""

from dataclasses import dataclass
from copy import deepcopy
from typing import List, TYPE_CHECKING

from orsopy import fileio

from ...add_ons.help_modules.materials_db import Formula, MASS_DENSITY_CONVERSION

if TYPE_CHECKING:
    # make sure the wx based plugins don't need to be imported at runtime
    from ...add_ons.SimpleReflectivity import Plugin as SRPlugin
    from ...add_ons.Reflectivity import Plugin as RPlugin

@dataclass
class InstrumentInformation:
    probe: str
    coords: str
    wavelength: float

@dataclass
class LayerData:
    dens: float
    formula: Formula
    f: complex
    b: complex
    d: float
    sigma: float

@dataclass
class LayerModel:
    substrate: LayerData
    ambient: LayerData
    stacks: List[List[LayerData]]
    repetitions: List[int]

class OrsoHeaderAnalyzer:
    header: fileio.Orso
    instrument: InstrumentInformation
    layer_model: LayerModel

    def __init__(self, meta):
        self.instrument = None
        self.model = None
        meta = deepcopy(meta)
        self.header = fileio.Orso(**meta)
        self.analyze_instrument_info()
        if hasattr(fileio, 'model_language') and self.header.data_source.sample.model:
            self.analyze_model()

    @property
    def instrument_settings(self) -> fileio.InstrumentSettings:
        return self.header.data_source.measurement.instrument_settings

    @property
    def experiment(self) -> fileio.Experiment:
        return self.header.data_source.experiment

    def get_probe(self):
        if self.experiment.probe=='neutron':
            pol = self.instrument_settings.polarization or 'unpolarized'
            if pol=='unpolarized':
                probe = 'neutron'
            elif pol in ['pp', 'mm', 'pm', 'mp']:
                probe = 'neutron pol spin flip'
            else:
                probe = 'neutron pol'
        else:
            probe = 'x-ray'
        return probe

    def analyze_instrument_info(self):
        probe = self.get_probe()

        # detect x-axis unit
        if self.header.columns[0].unit in [None, '1/angstrom', '1/nm']:
            coords = 'q'
        else:
            # not officially ORSO confirm
            coords = '2θ'
        wavelength = float(self.instrument_settings.wavelength.as_unit('angstrom') or 1.54)
        if not isinstance(wavelength, float):
            # in case wavelength is a ValueRange (ToF)
            wavelength = 1.0
        self.instrument = InstrumentInformation(probe=probe, coords=coords, wavelength=wavelength)

    @staticmethod
    def get_layer_data(layer) -> LayerData:
        layer.material.generate_density()
        if layer.material.formula:
            formula = Formula.from_str(layer.material.formula)
            if layer.material.number_density:
                dens = layer.material.number_density.as_unit('1/angstrom**3')
            elif layer.material.mass_density:
                dens = layer.material.mass_density.as_unit('g/cm**3')* \
                                 MASS_DENSITY_CONVERSION/formula.mFU()
            else:
                from genx.models.utils import bc
                try:
                    dens = layer.material.get_sld().real/eval(formula.b()).real
                except Exception:
                    dens = 0.1
            b = formula.b()
            f = formula.f()
        else:
            formula = None
            b = layer.material.get_sld()
            f = layer.material.get_sld()
            dens = 0.1
        d = layer.thickness.as_unit('angstrom')
        sigma = layer.roughness.as_unit('angstrom')
        return LayerData(dens=dens, formula=formula, b=b, f=f, d=d, sigma=sigma)


    def analyze_model(self):
        from orsopy.fileio import model_language
        header_model = self.header.data_source.sample.model

        res_stack = header_model.resolve_stack()
        stacks = []
        repetitions = []
        last_stack = True
        for si in res_stack:
            if hasattr(si, 'repetitions'):
                repetitions.append(si.repetitions)
                stack_layers = []
                for stack_item in si.sequence:
                    if isinstance(stack_item, model_language.Layer):
                        stack_layers.append(self.get_layer_data(stack_item))
                    else:
                        stack_layers += list(map(self.get_layer_data, stack_item.resolve_to_layers()))
                stacks.append(stack_layers)
                last_stack = True
            elif last_stack:
                last_stack = False
                repetitions.append(1)
                stacks.append([self.get_layer_data(si)])
            else:
                stacks[-1].append(self.get_layer_data(si))

        # make sure we remove ambient and substrate from a single repetition stack
        if repetitions[0]!=1:
            repetitions.insert(0, 1)
            repetitions[1] -= 1
            stacks.insert(0, stacks[0])
        if repetitions[-1]!=1:
            repetitions[-1] -= 1
            repetitions.append(1)
            stacks.append(stacks[-1])
        ambient = stacks[0].pop(0)
        substrate = stacks[-1].pop(-1)
        for i, si in reversed(list(enumerate(stacks))):
            if len(si)==0:
                stacks.pop(i)
                repetitions.pop(i)
        self.layer_model = LayerModel(ambient=ambient, substrate=substrate, stacks=stacks, repetitions=repetitions)

    def simple_refl_layer(self, layer: LayerData, pos=1):
        if layer.formula:
            dens = layer.dens*layer.formula.mFU()/MASS_DENSITY_CONVERSION
            return [None, "Formula", layer.formula,
                    False, str(dens), False, '0.0',
                    False, str(layer.d), False, str(layer.sigma),
                    pos]
        else:
            return [None, "Formula", "SLD",
                    False, str(layer.b.real if self.instrument.probe!='x-ray' else layer.f.real), False, '0.0',
                    False, str(layer.d), False, str(layer.sigma),
                    pos]

    def build_simple_model(self, refl: "SRPlugin"):
        from ...add_ons.SimpleReflectivity import TOP_LAYER, ML_LAYER, BOT_LAYER
        refl.sample_widget.sample_table.ResetModel()
        refl.sample_widget.inst_params['probe'] = self.instrument.probe
        refl.sample_widget.inst_params['wavelength'] = self.instrument.wavelength
        refl.sample_widget.inst_params['coords'] = self.instrument.coords
        refl.sample_widget.inst_params['res'] = 0.01
        if self.layer_model:
            # if any stack is repeated, use that as central one
            if any([ri>1 for ri in self.layer_model.repetitions]):
                pos = TOP_LAYER
                is_ML = True
            else:
                pos = ML_LAYER
                is_ML = False

            ID = 0
            layers = []
            repetitions = 1
            for i, (ri, si) in enumerate(zip(self.layer_model.repetitions, self.layer_model.stacks)):
                if is_ML and pos==ML_LAYER:
                    pos = BOT_LAYER
                if ri>1 and pos==TOP_LAYER:
                    pos = ML_LAYER
                    repetitions = ri
                    for li in si:
                        nl = self.simple_refl_layer(li, pos=pos)
                        nl[0] = f'Layer_{ID:02}'
                        layers.append(nl)
                        ID+=1
                else:
                    for li in ri*si:
                        nl = self.simple_refl_layer(li, pos=pos)
                        nl[0] = f'Layer_{ID:02}'
                        layers.append(nl)
                        ID+=1
            refl.sample_widget.sample_table.ambient = self.simple_refl_layer(self.layer_model.ambient)
            refl.sample_widget.sample_table.substrate = self.simple_refl_layer(self.layer_model.substrate)
            refl.sample_widget.sample_table.RebuildTable(layers)
            refl.sample_widget.sample_table.repetitions = repetitions
        refl.sample_widget.last_sample_script = refl.sample_widget.sample_table.getModelCode()
        refl.sample_widget.UpdateModel(re_color=True)

    def build_reflectivity(self, refl: "RPlugin"):
        refl.CreateNewModel('models.spec_nx')
        # detect source radiation
        inst = refl.sample_widget.instruments['inst']
        inst.probe = self.instrument.probe
        inst.coords = self.instrument.coords
        inst.wavelength = self.instrument.wavelength
        pol_names = {
            'po': 'uu', 'mo': 'dd', 'op': 'uu', 'om': 'dd',
            'pp': 'uu', 'mm': 'dd', 'pm': 'ud', 'mp': 'du'
            }
        # set resolution column
        if len(self.header.columns)>3 and self.header.columns[3].error_of==self.header.columns[0].name:
            inst.restype = 'full conv and varying res.'
            inst.respoints = 7
            inst.resintrange = 2.5
            el = refl.simulation_widget.GetExpressionList()
            for i, data_item in enumerate(refl.GetModel().data):
                if len(data_item.meta['columns'])>3 and (
                        data_item.meta['columns'][3].get('error_of', None)==
                        data_item.meta['columns'][0]['name']):
                    el[i].append(f'inst.setRes(data[{i}].res)')
                else:
                    el[i].append(f'inst.setRes(0.001)')
                # set polarization channel
                pol = data_item.meta['data_source']['measurement']['instrument_settings'].get('polarization',
                                                                                              'unpolarized')
                if pol in pol_names:
                    el[i].append(f'inst.setPol("{pol_names[pol]}")')

        if self.layer_model:
            lm = self.layer_model
            tmp = refl.sampleh.sample.Ambient
            tmp.b, tmp.f, tmp.dens = lm.ambient.b, lm.ambient.f, lm.ambient.dens
            tmp = refl.sampleh.sample.Substrate
            tmp.b, tmp.f, tmp.dens, tmp.sigma = lm.substrate.b, lm.substrate.f, lm.substrate.dens, lm.substrate.sigma

            pos = 1
            for si, (rep, stack) in enumerate(zip(lm.repetitions, lm.stacks)):
                refl.sampleh.insertItem(pos, 'Stack', f'ST_{si}')
                stack_obj = refl.sampleh.sample.Stacks[0]
                stack_obj.Repetitions = rep
                pos += 1
                for li, layer in enumerate(stack):
                    refl.sampleh.insertItem(pos, 'Layer', f'L_{si}_{li:02}')
                    pos += 1

                    layer_obj = stack_obj.Layers[0]

                    layer_obj.b, layer_obj.f, layer_obj.dens = layer.b, layer.f, layer.dens
                    layer_obj.d , layer_obj.sigma = layer.d, layer.sigma

            refl.sample_widget.Update()

        refl.WriteModel()

# include GenX materials in density resolution
try:
    from orsopy.fileio import model_language
except ImportError:
    pass
else:
    from orsopy.utils.density_resolver import DensityResolver
    from orsopy.utils.resolver_slddb import ResolverSLDDB
    from orsopy.utils.chemical_formula import Formula as OrsoFormula
    from ...add_ons.help_modules.materials_db import mdb

    class ResolverGenX(DensityResolver):
        comment = "from GenX "

        def resolve_formula(self, formula: OrsoFormula) -> float:
            frm = Formula.from_str(str(formula))
            if frm not in mdb:
                raise ValueError(f"Could not find material {formula}")
            else:
                res_formula, density = mdb[frm]
                self.comment = f"density from GenX db {res_formula}"
                return 1e3*eval(density)

        def resolve_elemental(self, formula: OrsoFormula) -> float:
            raise ValueError("GenX resolve only used for known materials.")

    model_language.DENSITY_RESOLVERS = [ResolverGenX(), ResolverSLDDB()]
