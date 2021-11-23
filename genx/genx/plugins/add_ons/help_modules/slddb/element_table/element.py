"""
Defining the Element class that is used to hold all needed data for one element/isotope.
"""
import os
from numpy import array, load

from .masses import ATOMIC_WEIGHTS, ELEMENT_CHARGES, ELEMENT_FULLNAMES
from .nlengths_pt import NEUTRON_SCATTERING_LENGTHS
from .nabs_geant4 import NEUTRON_ABSORPTIONS, DATA_DIR as NABS_DATA_DIR
from .xray_nist import XRAY_SCATTERING_FACTORS


BASE_PATH = os.path.abspath(os.path.dirname(__file__))

ELEMENT_NAMES = dict([(value, key) for key, value in ELEMENT_CHARGES.items()])

ELEMENT_FULLNAMES['D'] = 'deuterium'
ELEMENT_FULLNAMES['Hx'] = 'exchangeable hydrogen'

for data in [ATOMIC_WEIGHTS, NEUTRON_SCATTERING_LENGTHS]:
    data['D'] = data[(1, 2)]
    data['Hx'] = data[(1, 1)]
for data in [ELEMENT_CHARGES, XRAY_SCATTERING_FACTORS]:
    data['Hx'] = data['H']


class Element():
    N = None

    def __init__(self, symbol=None, Z=None):
        # get element from database
        N = None
        self.symbol = symbol
        if Z is None and symbol is None:
            raise ValueError("Provide either Symbol or Z")
        elif Z is None:
            if '[' in symbol:
                self.symbol, N = symbol.rstrip(']').split('[', 1)
                N = int(N)
                key = (ELEMENT_CHARGES[self.symbol], N)
            else:
                key = symbol
            self.Z = ELEMENT_CHARGES[self.symbol]
        else:
            self.Z = Z
            self.symbol = ELEMENT_NAMES[Z]
            key = self.symbol

        self.N = N
        self.name = ELEMENT_FULLNAMES[self.symbol]
        self.mass = ATOMIC_WEIGHTS[key]
        try:
            self.b = NEUTRON_SCATTERING_LENGTHS[key]
        except KeyError:
            raise ValueError(f'No neutorn scattering data for {key}')
        if key in NEUTRON_ABSORPTIONS:
            self._ndata = load(os.path.join(BASE_PATH, NABS_DATA_DIR, NEUTRON_ABSORPTIONS[key]))['arr_0']
        else:
            self._ndata = None

        try:
            self._xdata = array(XRAY_SCATTERING_FACTORS[self.symbol])
        except KeyError:
            self._xdata = None

    def f_of_E(self, Ei):
        if self._xdata is None:
            return float('nan')
        E, fp, fpp = self._xdata
        fltr = (E>=Ei)
        if not fltr.any():
            return 0.-0j
        else:
            # linear interpolation between two nearest points
            E1 = E[fltr][0]
            try:
                E2 = E[fltr][1]
            except IndexError:
                return fp[fltr][0]-1j*fpp[fltr][0]
            else:
                f1 = fp[fltr][0]-1j*fpp[fltr][0]
                f2 = fp[fltr][1]-1j*fpp[fltr][1]
                return ((E2-Ei)*f1+(Ei-E1)*f2)/(E2-E1)

    def b_of_L(self, Li):
        if self._ndata is None:
            return self.b
        L, b_abs = self._ndata
        if Li>L[-1]:
            return self.b.real-1j*b_abs[-1]
        if Li<L[0]:
            return self.b.real-1j*b_abs[0]
        fltr = (L>=Li)
        # linear interpolation between two nearest points
        L1 = L[fltr][0]
        try:
            L2 = L[fltr][1]
        except IndexError:
            return self.b.real-1j*b_abs[fltr][0]
        else:
            b_abs1 = b_abs[fltr][0]
            b_abs2 = b_abs[fltr][1]
            return self.b.real-1j*((L2-Li)*b_abs1+(Li-L1)*b_abs2)/(L2-L1)

    @property
    def E(self):
        return self._xdata[0]

    @property
    def f(self):
        return self._xdata[1]-1j*self._xdata[2]

    @property
    def fp(self):
        return self._xdata[1]

    @property
    def fpp(self):
        return self._xdata[2]

    @property
    def has_ndata(self):
        return self._ndata is not None

    @property
    def Lamda(self):
        # return neutron wavelength values for energy dependant absorption
        if self.has_ndata:
            return self._ndata[0]
        else:
            return array([0.05, 50.0])

    @property
    def b_abs(self):
        return self._ndata[1]

    @property
    def b_lambda(self):
        if self.has_ndata:
            return self.b.real-1j*self._ndata[1]
        else:
            return array([self.b, self.b])

    def __str__(self):
        if self.N is None:
            return self.symbol
        else:
            return "%s[%s]"%(self.symbol, self.N)

    def __repr__(self):
        if self.N is None:
            symb = self.symbol
        else:
            symb = "%s[%s]"%(self.symbol, self.N)
        return 'Element(symbol="%s")'%symb

    def __eq__(self, other):
        if type(self)==type(other):
            return self.N==other.N and self.Z==other.Z \
                   and self.symbol==other.symbol
        else:
            return object.__eq__(self, other)

    def __hash__(self):
        return hash((self.N, self.Z, self.symbol))
