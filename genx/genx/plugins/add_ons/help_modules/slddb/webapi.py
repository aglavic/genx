import json
import os
import pathlib
import datetime
import warnings
import ssl

from urllib import request, parse
from urllib.error import URLError
from . import SLDDB, DB_FILE
from .dbconfig import WEBAPI_URL
from .material import Material, Formula
from .element_table import get_element


class SLD_API:
    """
      Python API for users of the SLDDB data.

      Allows to query the online database for materials, calculate SLDs and add new materials.
      If connection to the server fails, a local copy of the database is used, instead.

      Usage:
        from slddb import api
        res=api.search(fomula="Fe2O3")
        res[0]['density'] => ....

        m=api.material(res[0]['ID']) # retreive all data for the given material, see Material class.
        sldn=m.rho_n # get nuclear neutron SLD (complex number)
        sldm=m.rho_m # get magnetic neutron SLD (real number)
        sldx=m.f_of_E(E=8.047823) # get x-ray SLD (complex number) for given energy, default is Cu-Kalpha

        # custom material just for SLD calculation, requires either dens, fu_volume, rho_n or xsld+xE
        m=api.custom(formula='Au', dens=19.3)

      Units of results/queries:
        density: g/cm³
        roh_n: Å^{-2}
        roh_m: Å^{-2}
        sldx: Å^{-2}
        fu_volume: Å³
    """
    db_suburl = 'download_db'
    max_age = 1
    db: SLDDB = None

    def __init__(self):
        self.first_access = True
        self.use_webquery = True  # only try webquery once, if error occures switch to local database

    def check(self):
        # make sure the local database file is up to date, if not try to download newest version
        if self.first_access:
            now = datetime.datetime.now()
            try:
                stat = pathlib.Path(DB_FILE).stat()
            except FileNotFoundError:
                self.download_db()
            else:
                mtime = datetime.datetime.fromtimestamp(stat.st_ctime)
                try:
                    mtime = max(mtime, datetime.datetime.fromtimestamp(stat.st_mtime))
                except AttributeError:
                    pass
                if (now-mtime).days>self.max_age:
                    try:
                        self.download_db()
                    except URLError as err:
                        warnings.warn("Can't download new version of databse; "+str(err))
                        return
            self.db = SLDDB(DB_FILE)  # after potential update, make connection with local database
            self.first_access = False
        else:
            return

    def download_db(self):
        # noinspection PyUnresolvedReferences
        context = ssl._create_unverified_context()
        res = request.urlopen(WEBAPI_URL+self.db_suburl, context=context)
        data = res.read()
        if not data.startswith(b'SQLite format 3'):
            raise ValueError('Error when downloading new database')
        if os.path.isfile(DB_FILE):
            os.remove(DB_FILE)
        with open(DB_FILE, 'wb') as fh:
            fh.write(data)

    @staticmethod
    def webquery(qdict):
        data = parse.urlencode(qdict)
        # noinspection PyUnresolvedReferences
        context = ssl._create_unverified_context()
        webdata = request.urlopen(WEBAPI_URL+'api?'+data, context=context)
        return json.loads(webdata.read())  # return decoded data

    def localquery(self, qdict):
        return self.db.search_material(**qdict)

    def localmaterial(self, ID):
        res = self.db.search_material(ID=ID)
        return self.db.select_material(res[0])

    def search(self, **opts):
        '''
        Search for a particular material using a combination of provided search keys.

        Examples:
             api.search(formula="Fe2O3")
             api.search(density=5.242)
             api.search(name='iron')
        '''
        if not self.use_webquery:
            return self.localquery(opts)

        self.check()
        try:
            res = self.webquery(opts)
        except URLError:
            self.use_webquery = False
            res = self.localquery(opts)
        return res

    def material(self, ID):
        """
        Returns the material object for a certain databse entry specified by its unique ID.

        Example:
            res=api.search(formula='Fe')
            material=api.material(res[0]['ID'])
            print(material.dens, material.rho_n, material.f_of_E(8.0))
        """
        if not self.use_webquery:
            return self.localmaterial(ID)

        self.check()
        try:
            res = self.webquery({'ID': int(ID)})
        except URLError:
            self.use_webquery = False
            return self.localmaterial(ID)
        else:
            f = Formula(res['formula'], sort=False)
            out = Material([(get_element(element), amount) for element, amount in f],
                           dens=float(res['density']))
            return out

    @staticmethod
    def custom(formula, dens=None, fu_volume=None, rho_n=None, mu=0., xsld=None, xE=None):
        """
        Returns the material object for a certain material as specified by caller.

        Example:
            res=api.custom('Fe', dens=7.8)
            print(material.dens, material.rho_n, material.f_of_E(8.0))
        """
        f = Formula(formula, sort=False)
        out = Material([(get_element(element), amount) for element, amount in f],
                       dens=dens, fu_volume=fu_volume, rho_n=rho_n, mu=mu, xsld=xsld, xE=xE)
        return out

    def bio_blender(self, sequence, molecule='protein'):
        """
        Get material for protein, DNA or RNA. Provide a letter sequence and molecule type ('protein', 'dna', 'rna').
        """
        opts = {molecule: sequence, 'sldcalc': 'true'}
        res = self.webquery(opts)
        out = Material(Formula(res['formula']), fu_volume=res['fu_volume'])
        return out
