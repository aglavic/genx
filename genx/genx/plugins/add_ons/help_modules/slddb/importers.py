"""
Functions to create database compatible entries from other file formats.
"""
import pathlib
import os
from .material import Formula, PolymerSequence
from .dbconfig import db_lookup


class Importer(dict):
    """
    Base class for importing database entries. Includes checks for correctness used by all importers.
    """
    formula = None

    def __init__(self, filename, validate=True):
        self.filename = filename
        self.name = os.path.basename(filename).rsplit('.', 1)[0]
        data = self.build_data()
        if validate:
            self.validate(name=self.name, formula=self.formula, **data)
        dict.__init__(self, data)

    @staticmethod
    def validate(**full_entry):
        # check for all values of the data dictionary if the format is valid
        for key, value in full_entry.items():
            if not db_lookup[key][1].validate(value):
                raise ValueError(f"Can not import dataset, failed to validate value '{value}' for key '{key}'")

    def build_data(self):
        raise NotImplementedError("Importer has to be subclassed with _build_data implemented.")

    def __repr__(self):
        return f'MaterialData(name="{self.name}", formula={repr(self.formula)} , data={dict.__repr__(self)})'


class CifImporter(Importer):
    suffix = 'cif'

    def __init__(self, filename, validate=True, sequence=1):
        self.sequence = sequence
        super().__init__(filename, validate=validate)

    @staticmethod
    def float_werr(value):
        # Convert CIF entry that might have an uncertainty to float
        return float(value.split('(')[0])

    def build_data(self):
        try:
            import CifFile
        except ImportError:
            raise RuntimeError("You have to install PyCifRW python package to be able to import cif files")

        output = {
            'data_origin':    'diffraction', 'comments': 'imported from CIF file',
            'physical_state': 'solid'
            }

        cf = CifFile.ReadCif(pathlib.Path(self.filename).as_uri())
        block = cf.first_block()

        if '_chemical_formula_sum' in block:
            formula = Formula(block['_chemical_formula_sum'])
        elif '_entity_poly.pdbx_seq_one_letter_code' in block:
            txt = block['_entity_poly.pdbx_seq_one_letter_code']
            if type(txt) is list:
                formula = PolymerSequence(txt[self.sequence-1])
            else:
                formula = PolymerSequence(txt)
            output['tags'] = ['biology', 'polymer']
            output['reference'] = 'Protein Data Bank (PDB)'
            output['ref_website'] = 'https://www.rcsb.org/'
            if '_citation.pdbx_database_id_DOI' in block:
                output['doi'] = block['_citation.pdbx_database_id_DOI'][0]
        else:
            raise ValueError("Could not locate chemical formula or one letter PDB sequence")

        if '_exptl_crystal_density_diffrn' in block:
            output['density'] = self.float_werr(block['_exptl_crystal_density_diffrn'])
        elif '_cell_volume' in block and '_cell_formula_units_Z' in block:
            output['FU_volume'] = self.float_werr(block['_cell_volume'])/ \
                                  self.float_werr(block['_cell_formula_units_Z'])  # Å³
        elif '_entity_poly.pdbx_seq_one_letter_code' in block:
            # will use database FU_volume to deduce polymer density
            pass
        else:
            raise ValueError("No data to deduce material density")

        if '_chemical_name_mineral' in block:
            self.name = block['_chemical_name_mineral']

        if all([ii in block for ii in ['_journal_name_full', '_journal_volume', '_journal_year',
                                       '_publ_author_name', ]]):
            authors = ', '.join(block['_publ_author_name'])
            journal = block["_journal_name_full"]
            volume = block["_journal_volume"]
            year = block["_journal_year"]
            if '_journal_page_first' in block:
                page = block['_journal_page_first']
            else:
                page = '-'
            output['reference'] = f'{authors}; {journal}, {volume}, p. {page} ({year})'.replace('\n', ' ')

        if '_journal_paper_doi' in block:
            output['doi'] = 'https://doi.org/'+block['_journal_paper_doi']

        self.formula = formula
        return output


importers = [CifImporter]
