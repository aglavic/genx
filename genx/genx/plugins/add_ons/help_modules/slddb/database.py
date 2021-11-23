"""
Manage database creation, insertion and access.
"""

import sqlite3
from .dbconfig import DB_MATERIALS_CONVERTERS, DB_MATERIALS_NAME, \
    DB_MATERIALS_FIELDS, DB_MATERIALS_FIELD_DEFAULTS, db_lookup
from .material import Material, Formula
from .importers import importers
from .comparators import Comparator


class SLDDB:
    """
    Database to store material parameters to calculate
    scattering length densities (SLDs) for neutron
    and x-ray scattering.
    """

    def __init__(self, dbfile):
        self.db = sqlite3.connect(dbfile)

    def import_material(self, filename, name=None, commit=True):
        suffix = filename.rsplit('.', 1)[1]
        res = None
        for importer in importers:
            if importer.suffix==suffix:
                res = importer(filename)
                break
        if res is None:
            raise IOError("File import failed for %s, no suitable importer found"%filename)
        if name is None:
            name = res.name
        return self.add_material(name, res.formula, commit=commit, **importer(filename))

    def add_material(self, name, formula, commit=True, **data):
        din = {}
        for key, value in data.items():
            if key not in DB_MATERIALS_FIELDS:
                raise KeyError('%s is not a valid data field'%key)
            din[key] = db_lookup[key][1].convert(value)

        if not ('density' in din or 'FU_volume' in din
                or 'SLD_n' in din or ('SLD_x' in din and 'E_x' in din)):
            raise ValueError("Not enough information to determine density")

        din['name'] = db_lookup['name'][1].convert(name)
        din['formula'] = db_lookup['formula'][1].convert(formula)

        c = self.db.cursor()
        # check if entry already exists
        qstr = "SELECT * FROM %s WHERE %s"%(
            DB_MATERIALS_NAME, ' AND '.join(["%s=?"%key for key in din.keys()]))
        c.execute(qstr, tuple(din.values()))
        if len(c.fetchall())!=0:
            raise ValueError("Entry with this data already exists")

        qstr = "INSERT INTO %s (%s) VALUES (%s)"%(
            DB_MATERIALS_NAME, ", ".join(din.keys()),
            ', '.join(["?" for _ in din.keys()]))
        c.execute(qstr, tuple(din.values()))
        c.close()
        if commit:
            self.db.commit()

    def update_material(self, ID, commit=True, **data):
        din = self.search_material(ID=ID, filter_invalid=False)[0]
        din.update(data)
        del (din['ID'])
        del (din['updated'])
        del (din['validated'])
        del (din['validated_by'])

        for key, value in din.items():
            if key not in DB_MATERIALS_FIELDS:
                raise KeyError('%s is not a valid data field'%key)
            if value is None:
                continue
            din[key] = db_lookup[key][1].convert(value)

        if not any([din.get(name, None) is not None for name in ['density', 'FU_volume', 'SLD_n', 'SLD_x']]):
            raise ValueError("Not enough information to determine density")

        c = self.db.cursor()

        qstr = "UPDATE %s SET %s,updated = CURRENT_TIMESTAMP,validated = NULL, validated_by = NULL WHERE ID==?"%(
            DB_MATERIALS_NAME, ", ".join(["%s = ?"%key for key in din.keys()]))
        c.execute(qstr, tuple(din.values())+(ID,))
        c.close()
        if commit:
            self.db.commit()

    def search_material(self, join_and=True, serializable=False, filter_invalid=True, limit=100, offset=0, **data):
        for key, value in data.items():
            if key not in DB_MATERIALS_FIELDS:
                raise KeyError('%s is not a valid data field'%key)

        if len(data)==0:
            sstr = 'SELECT * FROM %s'%DB_MATERIALS_NAME
            if filter_invalid:
                sstr += ' WHERE invalid IS NULL'
            qstr = ''
            qlst = []
            ustr = ''
        else:
            sstr = 'SELECT * FROM %s WHERE '%DB_MATERIALS_NAME
            if filter_invalid:
                sstr += 'invalid IS NULL AND '
            ustr = 'UPDATE %s SET accessed = accessed + 1 WHERE '%DB_MATERIALS_NAME
            qstr = ''
            qlst = []
            for key, value in data.items():
                if isinstance(value, Comparator):
                    # user has supplied a comparator instead of a value
                    cmp: Comparator = value
                    cmp.key = key
                else:
                    # use comparator for specific validator
                    cmp: Comparator = db_lookup[key][1].comparator(value, key)
                qstr += cmp.query_string()
                qlst_add = cmp.query_args()
                qlst += qlst_add

                if len(qlst_add)>0:
                    if join_and:
                        qstr += ' AND '
                    else:
                        qstr += '  OR '
            qstr = qstr[:-5]
        c = self.db.cursor()
        c.execute(sstr+qstr+' ORDER BY validated DESC, selected DESC, accessed DESC LIMIT %i,%i'%(offset, limit), qlst)
        results = c.fetchall()
        keys = [key for key, *ignore in c.description]
        # update access counter
        c.execute(ustr+qstr, qlst)
        c.close()
        self.db.commit()

        # convert values
        output = []
        if serializable:
            for row in results:
                rowdict = {key: db_lookup[key][1].revert_serializable(value) for key, value in zip(keys, row)}
                output.append(rowdict)
        else:
            for row in results:
                rowdict = {key: db_lookup[key][1].revert(value) for key, value in zip(keys, row)}
                output.append(rowdict)
        return output

    def count_material(self, join_and=True, filter_invalid=True, **data):
        for key, value in data.items():
            if key not in DB_MATERIALS_FIELDS:
                raise KeyError('%s is not a valid data field'%key)

        if len(data)==0:
            sstr = 'SELECT COUNT(*) FROM %s'%DB_MATERIALS_NAME
            if filter_invalid:
                sstr += ' WHERE invalid IS NULL'
            qstr = ''
            qlst = []
        else:
            sstr = 'SELECT COUNT(*) FROM %s WHERE '%DB_MATERIALS_NAME
            if filter_invalid:
                sstr += 'invalid IS NULL AND '
            qstr = ''
            qlst = []
            for key, value in data.items():
                if isinstance(value, Comparator):
                    # user has supplied a comparator instead of a value
                    cmp: Comparator = value
                    cmp.key = key
                else:
                    # use comparator for specific validator
                    cmp: Comparator = db_lookup[key][1].comparator(value, key)
                qstr += cmp.query_string()
                qlst_add = cmp.query_args()
                qlst += qlst_add

                if len(qlst_add)>0:
                    if join_and:
                        qstr += ' AND '
                    else:
                        qstr += '  OR '
            qstr = qstr[:-5]
        c = self.db.cursor()
        c.execute(sstr+qstr, qlst)
        result = c.fetchone()
        # update access counter
        c.close()
        self.db.commit()
        return result[0]

    def select_material(self, result) -> Material:
        # generate Material object from database entry and increment selection counter
        formula = Formula(result['formula'])
        if result['density']:
            fu_volume = None
        else:
            fu_volume = result['FU_volume']
        extra_data = {}
        if result['invalid'] is not None:
            extra_data[
                'WARNING'] = 'This entry has been invalidated by ORSO on %s, ' \
                             'please contact %s for more information.'%(
                                 result['invalid'], result['invalid_by'])
        extra_data['ID'] = int(result['ID'])
        extra_data['ORSO_validated'] = result['validated'] is not None
        extra_data['reference'] = result.get('reference', '')
        extra_data['doi'] = result.get('doi', '')
        extra_data['description'] = result.get('description', '')

        m = Material(formula,
                     dens=result['density'],
                     fu_volume=fu_volume,
                     rho_n=result['SLD_n'],
                     xsld=result['SLD_x'], xE=result['E_x'],
                     mu=result['mu'],
                     ID=result['ID'],
                     name=result['name'],
                     extra_data=extra_data)

        ustr = 'UPDATE %s SET selected = selected + 1 WHERE ID == ?'%DB_MATERIALS_NAME
        c = self.db.cursor()
        c.execute(ustr, (result['ID'],))
        c.close()
        self.db.commit()
        return m

    def validate_material(self, ID, user):
        ustr = 'UPDATE %s SET validated = CURRENT_TIMESTAMP, validated_by = ?,' \
               ' invalid = NULL, invalid_by = NULL WHERE ID == ?'%DB_MATERIALS_NAME
        c = self.db.cursor()
        c.execute(ustr, (user, ID,))
        c.close()
        self.db.commit()

    def invalidate_material(self, ID, user):
        ustr = 'UPDATE %s SET invalid = CURRENT_TIMESTAMP, invalid_by = ?, ' \
               ' validated = NULL, validated_by = NULL WHERE ID == ?'%DB_MATERIALS_NAME
        c = self.db.cursor()
        c.execute(ustr, (user, ID,))
        c.close()
        self.db.commit()

    def create_table(self):
        c = self.db.cursor()
        name_type = ['%s %s %s'%(fi, ci.sql_type, (di is not None) and "DEFAULT %s"%di or "")
                     for fi, ci, di in zip(DB_MATERIALS_FIELDS, DB_MATERIALS_CONVERTERS,
                                           DB_MATERIALS_FIELD_DEFAULTS)]
        qstr = 'CREATE TABLE %s (%s)'%(DB_MATERIALS_NAME, ", ".join(name_type))
        c.execute(qstr)
        c.close()
        self.db.commit()

    def create_database(self):
        self.create_table()
        self.db.commit()

    def add_elements(self):
        import periodictable

        for element in periodictable.elements:
            # noinspection PyUnresolvedReferences
            if element is periodictable.n or element.density is None:
                continue
            state = 'solid'
            if 'T=-' in element.density_caveat:
                state = 'liquid'
            self.add_material(element.name.capitalize(),
                              element.symbol,
                              commit=False,
                              description=element.density_caveat,
                              density=element.density,
                              physical_state=state,
                              data_origin='textbook',
                              ref_website='https://github.com/pkienzle/periodictable',
                              reference='Python module periodictable, \ndata source: ILL Neutron Data Booklet')
        self.db.commit()

    def update_fields(self):
        # add columns not currently available
        c = self.db.cursor()
        c.execute('SELECT * FROM %s LIMIT 1'%DB_MATERIALS_NAME)
        _ = c.fetchall()
        fields = [col[0] for col in c.description]
        if len(fields)==len(DB_MATERIALS_FIELDS) and DB_MATERIALS_FIELDS==fields[:len(DB_MATERIALS_FIELDS)]:
            return
        if DB_MATERIALS_FIELDS[:len(fields)]!=fields:
            # need to reorder and/or add/remove colums of the databse, requires copy of table
            name_type = ['%s %s %s'%(fi, ci.sql_type, (di is not None) and "DEFAULT %s"%di or "")
                         for fi, ci, di in zip(DB_MATERIALS_FIELDS, DB_MATERIALS_CONVERTERS,
                                               DB_MATERIALS_FIELD_DEFAULTS)]
            qstr = 'CREATE TABLE tmp_table (%s)'%(", ".join(name_type))
            c.execute(qstr)
            jf = [field for field in fields if field in DB_MATERIALS_FIELDS]
            qstr = 'INSERT INTO tmp_table (%s) SELECT %s FROM %s'%(','.join(jf), ','.join(jf), DB_MATERIALS_NAME)
            c.execute(qstr)
            c.execute('DROP TABLE %s'%DB_MATERIALS_NAME)
            c.execute('ALTER TABLE tmp_table RENAME TO %s'%DB_MATERIALS_NAME)
            c.close()
            self.db.commit()
            return
        # append new columns
        start = len(fields)
        name_type = ['%s %s %s'%(fi, ci.sql_type, (di is not None) and "DEFAULT %s"%di or "")
                     for fi, ci, di in zip(DB_MATERIALS_FIELDS[start:], DB_MATERIALS_CONVERTERS[start:],
                                           DB_MATERIALS_FIELD_DEFAULTS[start:])]
        c.execute('ALTER TABLE %s ADD %s'%(DB_MATERIALS_NAME,
                                           ", ".join(name_type)))
        c.close()
        self.db.commit()

    def backup(self, filename):
        # make a copy of the open database
        out = sqlite3.connect(filename)
        with out:
            self.db.backup(out)
        out.close()

    def __del__(self):
        self.db.close()
