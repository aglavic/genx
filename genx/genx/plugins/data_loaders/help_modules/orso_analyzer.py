"""
Helper module to support ORSO file header information to build reflectivity models.
"""
import logging

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

from orsopy import fileio

from ...add_ons.help_modules.materials_db import MASS_DENSITY_CONVERSION, Formula

if TYPE_CHECKING:
    # make sure the wx based plugins don't need to be imported at runtime
    from ...add_ons.Reflectivity import Plugin as RPlugin
    from ...add_ons.SimpleReflectivity import Plugin as SRPlugin


def sanetize_name(name):
    if name:
        name = name.replace('\t', '_').replace(' ', '_').replace('.', 'p').replace('-', 'm')
        if not name.isidentifier():
            name = None
    return name


@dataclass
class InstrumentInformation:
    probe: str
    coords: str
    wavelength: float


@dataclass
class LayerData:
    dens: float
    formula: Optional[Formula]
    f: complex
    b: complex
    d: float
    sigma: float
    name: Optional[str]

    def __post_init__(self):
        self.name = sanetize_name(self.name)

@dataclass
class StackData:
    layers: List[LayerData]
    name: Optional[str]

    def __post_init__(self):
        self.name = sanetize_name(self.name)

@dataclass
class LayerModel:
    substrate: LayerData
    ambient: LayerData
    stacks: List[StackData]
    repetitions: List[int]

class OrsoHeaderAnalyzer:
    header: fileio.Orso
    instrument: InstrumentInformation
    layer_model: Optional[LayerModel] = None

    def __init__(self, meta):
        self.model = None
        meta = deepcopy(meta)
        self.header = fileio.Orso.from_dict(meta)
        self.analyze_instrument_info()
        if hasattr(fileio, "model_language") and self.header.data_source.sample.model:
            try:
                self.analyze_model()
            except Exception:
                logging.error('Header defined a sample model but parsing the model failed', exc_info=True)

    @classmethod
    def from_orso(cls, orso_header):
        self = object.__new__(cls)
        self.model = None
        self.header = orso_header
        self.analyze_instrument_info()
        try:
            self.analyze_model()
        except Exception:
            logging.critical('Header defined a sample model but parsing the model failed', exc_info=True)
        return self

    @classmethod
    def from_yaml(cls, yaml_text):
        import yaml
        from orsopy.fileio.model_language import SampleModel

        meta = yaml.safe_load(yaml_text)

        for parent in ['data_source', 'sample', 'model']:
            if parent in meta:
                meta = meta[parent]
        if not 'stack' in meta:
            raise ValueError('A sample model must contain the "stack" attribute')

        self = object.__new__(cls)
        self.model = None
        self.header = fileio.Orso.empty()
        self.header.data_source.sample.model = SampleModel.from_dict(meta)

        self.instrument = InstrumentInformation(probe='x-ray', coords='q', wavelength=1.54)

        try:
            self.analyze_model()
        except Exception:
            logging.critical('Header defined a sample model but parsing the model failed', exc_info=True)

        return self

    @property
    def instrument_settings(self) -> fileio.InstrumentSettings:
        return self.header.data_source.measurement.instrument_settings

    @property
    def experiment(self) -> fileio.Experiment:
        return self.header.data_source.experiment

    def get_probe(self):
        if self.experiment.probe == "neutron":
            pol = self.instrument_settings.polarization or "unpolarized"
            if pol == "unpolarized":
                probe = "neutron"
            elif pol in ["pp", "mm", "pm", "mp"]:
                probe = "neutron pol spin flip"
            else:
                probe = "neutron pol"
        else:
            probe = "x-ray"
        return probe

    def analyze_instrument_info(self):
        probe = self.get_probe()

        # detect x-axis unit
        if self.header.columns[0].unit in [None, "1/angstrom", "1/nm"]:
            coords = "q"
        else:
            # not officially ORSO confirm
            coords = "2θ"
        try:
            wavelength = float(self.instrument_settings.wavelength.as_unit("angstrom"))
        except Exception:
            wavelength = 1.54
        if not isinstance(wavelength, float):
            # in case wavelength is a ValueRange (ToF)
            wavelength = 1.0
        self.instrument = InstrumentInformation(probe=probe, coords=coords, wavelength=wavelength)

    @staticmethod
    def get_layer_data(layer) -> LayerData:
        layer.material.generate_density()
        if getattr(layer.material, 'formula', None):
            formula = Formula.from_str(layer.material.formula)
            if layer.material.number_density:
                dens = layer.material.number_density.as_unit("1/angstrom**3")
            elif layer.material.mass_density:
                dens = layer.material.mass_density.as_unit("g/cm**3") * MASS_DENSITY_CONVERSION / formula.mFU()
            else:
                from genx.models.utils import bc

                try:
                    dens = layer.material.get_sld().real*1e5 / eval(formula.b(), globals={'bc': bc}, locals={}).real
                except Exception:
                    dens = 0.1
            b = formula.b()
            f = formula.f()
        else:
            formula = None
            b = layer.material.get_sld()*1e6
            f = layer.material.get_sld()*1e6
            dens = 0.1
        name = getattr(layer, 'original_name', None) or getattr(layer.material, 'original_name', None)
        d = layer.thickness.as_unit("angstrom")
        sigma = layer.roughness.as_unit("angstrom")
        return LayerData(dens=dens, formula=formula, b=b, f=f, d=d, sigma=sigma, name=name)

    def analyze_model(self):
        from orsopy.fileio import model_language

        header_model = self.header.data_source.sample.model

        res_stack = header_model.resolve_stack()
        stacks = []
        repetitions = []
        last_stack = True
        for si in res_stack:
            if hasattr(si, "repetitions"):
                repetitions.append(si.repetitions)
                stack = StackData(layers=[], name=getattr(si, 'original_name', None))
                for stack_item in si.sequence:
                    if isinstance(stack_item, model_language.Layer):
                        stack.layers.append(self.get_layer_data(stack_item))
                    else:
                        stack.layers += list(map(self.get_layer_data, stack_item.resolve_to_layers()))
                stacks.append(stack)
                last_stack = True
            elif last_stack:
                last_stack = False
                repetitions.append(1)
                stacks.append(StackData(layers=[self.get_layer_data(si)], name=None))
            else:
                stacks[-1].layers.append(self.get_layer_data(si))

        # make sure we remove ambient and substrate from a single repetition stack
        if repetitions[0] != 1:
            repetitions.insert(0, 1)
            repetitions[1] -= 1
            stacks.insert(0, stacks[0])
        if repetitions[-1] != 1:
            repetitions[-1] -= 1
            repetitions.append(1)
            stacks.append(stacks[-1])
        ambient = stacks[0].layers.pop(0)
        substrate = stacks[-1].layers.pop(-1)
        for i, si in reversed(list(enumerate(stacks))):
            if len(si.layers) == 0:
                stacks.pop(i)
                repetitions.pop(i)
        self.layer_model = LayerModel(ambient=ambient, substrate=substrate, stacks=stacks, repetitions=repetitions)

    def simple_refl_layer(self, layer: LayerData, pos=1):
        if layer.formula:
            dens = layer.dens * layer.formula.mFU() / MASS_DENSITY_CONVERSION
            return [
                layer.name,
                "Formula",
                layer.formula,
                False,
                str(dens),
                False,
                "0.0",
                False,
                str(layer.d),
                False,
                str(layer.sigma),
                pos,
            ]
        else:
            return [
                layer.name,
                "Formula",
                "SLD",
                False,
                str(layer.b.real if self.instrument.probe != "x-ray" else layer.f.real),
                False,
                "0.0",
                False,
                str(layer.d),
                False,
                str(layer.sigma),
                pos,
            ]

    def build_simple_model(self, refl: "SRPlugin"):
        refl.sample_widget.sample_table.ResetModel()
        refl.sample_widget.inst_params["probe"] = self.instrument.probe
        refl.sample_widget.inst_params["wavelength"] = self.instrument.wavelength
        refl.sample_widget.inst_params["coords"] = self.instrument.coords
        refl.sample_widget.inst_params["res"] = 0.01
        self.build_simple_sample(refl)

    def build_simple_sample(self, refl: "SRPlugin"):
        from ...add_ons.SimpleReflectivity import BOT_LAYER, ML_LAYER, TOP_LAYER
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
                    for li in si.layers:
                        nl = self.simple_refl_layer(li, pos=pos)
                        if nl[0] is None:
                            nl[0] = f"Layer_{ID:02}"
                        elif nl[0] in [ni[0] for ni in layers]:
                            nl[0] = f"{nl[0]}_{ID:02}"
                        layers.append(nl)
                        ID += 1
                else:
                    for li in ri*si.layers:
                        nl = self.simple_refl_layer(li, pos=pos)
                        if nl[0] is None:
                            nl[0] = f"Layer_{ID:02}"
                        elif nl[0] in [ni[0] for ni in layers]:
                            nl[0] = f"{nl[0]}_{ID:02}"
                        layers.append(nl)
                        ID += 1
            refl.sample_widget.sample_table.ambient = self.simple_refl_layer(self.layer_model.ambient)
            refl.sample_widget.sample_table.substrate = self.simple_refl_layer(self.layer_model.substrate)
            refl.sample_widget.sample_table.RebuildTable(layers)
            refl.sample_widget.sample_table.repetitions = repetitions
        refl.sample_widget.last_sample_script = refl.sample_widget.sample_table.getModelCode()
        refl.sample_widget.UpdateModel(re_color=True)

    def build_reflectivity(self, refl: "RPlugin"):
        refl.CreateNewModel("models.spec_nx")
        # detect source radiation
        inst = refl.sample_widget.instruments["inst"]
        inst.probe = self.instrument.probe
        inst.coords = self.instrument.coords
        inst.wavelength = self.instrument.wavelength
        pol_names = {"po": "uu", "mo": "dd", "op": "uu", "om": "dd", "pp": "uu", "mm": "dd", "pm": "ud", "mp": "du"}
        # set resolution column
        if len(self.header.columns) > 3 and self.header.columns[3].error_of == self.header.columns[0].name:
            inst.restype = "full conv and varying res."
            inst.respoints = 7
            inst.resintrange = 2.5
            el = refl.simulation_widget.GetExpressionList()
            for i, data_item in enumerate(refl.GetModel().data):
                if len(data_item.meta["columns"]) > 3 and (
                    data_item.meta["columns"][3].get("error_of", None) == data_item.meta["columns"][0]["name"]
                ):
                    el[i].append(f"inst.setRes(data[{i}].res)")
                else:
                    el[i].append(f"inst.setRes(0.001)")
                # set polarization channel
                pol = data_item.meta["data_source"]["measurement"]["instrument_settings"].get(
                    "polarization", "unpolarized"
                )
                if pol in pol_names:
                    el[i].append(f'inst.setPol("{pol_names[pol]}")')

        self.build_sample(refl)

    def build_sample(self, refl: "RPlugin"):
        if self.layer_model:
            lm = self.layer_model
            tmp = refl.sampleh.sample.Ambient
            tmp.b, tmp.f, tmp.dens = lm.ambient.b, lm.ambient.f, lm.ambient.dens
            tmp._ca["f"] = lm.ambient.f
            tmp._ca["b"] = lm.ambient.b
            tmp = refl.sampleh.sample.Substrate
            tmp.b, tmp.f, tmp.dens, tmp.sigma = lm.substrate.b, lm.substrate.f, lm.substrate.dens, lm.substrate.sigma
            tmp._ca["f"] = lm.substrate.f
            tmp._ca["b"] = lm.substrate.b

            while len(refl.sampleh)>2:
                # clear the stacks in the sample handler
                refl.sampleh.deleteItem(1)

            pos = 1
            for si, (rep, stack) in enumerate(zip(lm.repetitions, lm.stacks)):
                name = stack.name
                if name is None:
                    name = f"Stack_{si:02}"
                elif name in refl.sampleh.names:
                    name = f"{name}_{si:02}"
                refl.sampleh.insertItem(pos, "Stack", name)
                stack_obj = refl.sampleh.sample.Stacks[0]
                stack_obj.Repetitions = rep
                pos += 1
                for li, layer in enumerate(stack.layers):
                    name = layer.name
                    if name is None:
                        name = f"Layer_{pos:02}"
                    elif name in refl.sampleh.names:
                        name = f"{name}_{pos:02}"

                    refl.sampleh.insertItem(pos, "Layer", name)
                    pos += 1

                    layer_obj = stack_obj.Layers[0]

                    layer_obj.b, layer_obj.f, layer_obj.dens = layer.b, layer.f, layer.dens
                    layer_obj.d, layer_obj.sigma = layer.d, layer.sigma
                    layer_obj._ca["b"] = layer.b
                    layer_obj._ca["f"] = layer.f

            refl.sample_widget.Update()

        refl.WriteModel()


# include GenX materials in density resolution
try:
    from orsopy.fileio import model_language
except ImportError:
    pass
else:
    from orsopy.utils.chemical_formula import Formula as OrsoFormula
    try:
        from orsopy.utils.density_resolver import DensityResolver
    except ImportError:
        from orsopy.utils.density_resolver import MaterialResolver as DensityResolver
    from orsopy.utils.resolver_slddb import ResolverSLDDB

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
                return 1e3 * eval(density)

        def resolve_elemental(self, formula: OrsoFormula) -> float:
            raise ValueError("GenX resolve only used for known materials.")

    model_language.DENSITY_RESOLVERS = [ResolverGenX(), ResolverSLDDB()]
