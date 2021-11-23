"""
Configure the database file as well as parameters for DB tables used.
"""

import os
from .converters import CType, CLimited, CArray, CDate, CComplex, CFormula, CSelect, CMultiSelect, \
    CUrl, CMail, Cdoi, Ccas

if 'APPDATA' in os.environ:
    confighome = os.environ['APPDATA']
elif 'XDG_CONFIG_HOME' in os.environ:
    confighome = os.environ['XDG_CONFIG_HOME']
else:
    confighome = os.path.join(os.environ['HOME'], '.config')
configpath = os.path.join(confighome, 'slddb')
if not os.path.exists(configpath):
    try:
        os.makedirs(configpath)
    except OSError:
        print("Could not create config path, local database storage not possible.")

cstr = CType(str, str)
cint = CType(int, int)
pint = CType(int, int, 'integer primary key')
cfloat = CType(float, float)
cposfloat = CLimited(float, float, 1e-30)
cdate = CDate()
cformula = CFormula()
ccomplex = CComplex()
carray = CArray()
curl = CUrl()
cmail = CMail()

WEBAPI_URL = 'https://slddb.esss.dk/slddb/'
DB_FILE = os.path.join(configpath, 'local_database.db')

DB_MATERIALS_NAME = 'materials'
DB_MATERIALS_COLUMNS = [
    # (Name,           converter, default, unit)
    ('ID', pint, None, None),
    ('created', cdate, 'CURRENT_TIMESTAMP', None),
    ('created_by', cmail, None, None),
    ('updated', cdate, None, None),
    ('validated', cdate, None, None),
    ('validated_by', cstr, None, None),
    ('accessed', cint, 0, None),
    ('selected', cint, 0, None),
    ('name', cstr, None, '*'),
    ('description', cstr, None, None),
    ('formula', cformula, None, '*'),
    ('HR_fomula', cstr, None, None),
    ('density', cposfloat, None, 'g/cm³ **'),
    ('FU_volume', cposfloat, None, 'Å³ **'),
    ('SLD_n', ccomplex, None, 'Å⁻² **'),
    ('SLD_x', ccomplex, None, 'r_e/Å⁻³ **'),
    ('E_x', cfloat, None, 'keV'),
    ('mu', cfloat, 0.0, 'µB/FU'),
    ('physical_state', CSelect(['solid', 'liquid', 'gas', 'solution',
                                'micellar aggregate', 'assembled monolayer/bilayer',
                                'nanoparticle']), 'solid', None),
    ('tags', CMultiSelect(['magnetic', 'polymer', 'biology', 'membrane', 'lipid',
                           'metal', 'metal alloy', 'inorganic', 'small organic',
                           'surfactant', 'protein']), None, None),
    ('ref_website', curl, None, None),
    ('reference', cstr, None, None),
    ('doi', Cdoi(), None, None),
    ('purity', cstr, None, None),
    ('CAS_No', Ccas(), None, None),
    ('crystal_data', cstr, None, None),
    ('temperature', cposfloat, None, 'K'),
    ('magnetic_field', cfloat, None, 'T'),
    ('data_origin', CSelect(['unspecified', 'textbook',
                             'x-ray reflectivity', 'neutron reflectivity',
                             'mass density', 'diffraction', 'interferometry',
                             'SANS', 'SAXS', 'molecular dynamics']), 'unspecified', None),
    ('comments', cstr, None, None),
    ('invalid', cdate, None, None),
    ('invalid_by', cstr, None, None),
    ]
DB_MATERIALS_FIELDS = [fi[0] for fi in DB_MATERIALS_COLUMNS]
DB_MATERIALS_CONVERTERS = [fi[1] for fi in DB_MATERIALS_COLUMNS]
DB_MATERIALS_FIELD_DEFAULTS = [fi[2] for fi in DB_MATERIALS_COLUMNS]
DB_MATERIALS_FIELD_UNITS = [fi[3] for fi in DB_MATERIALS_COLUMNS]
DB_MATERIALS_HIDDEN_DATA = ['created', 'created_by', 'updated',
                            'validated', 'validated_by', 'accessed', 'selected',
                            'invalid', 'invalid_by']
db_lookup = dict([(field, (i, converter, default, unit))
                  for i, (field, converter, default, unit) in
                  enumerate(zip(DB_MATERIALS_FIELDS,
                                DB_MATERIALS_CONVERTERS,
                                DB_MATERIALS_FIELD_DEFAULTS,
                                DB_MATERIALS_FIELD_UNITS))])
