"""
Class to hold information for one material and allow calculation
of x-ray and neutron SLDs for different applications.
"""

import re
from numpy import array, pi
from collections import OrderedDict
from .constants import u2g, r_e, r_e_angstrom, muB, rho_of_M, Cu_kalpha, Mo_kalpha, \
    E_to_lambda, fm2angstrom, dens_H2O, dens_D2O
from .element_table import get_element


SUBSCRIPT_DIGITS = "₀₁₂₃₄₅₆₇₈₉"


class Formula(list):
    """
    Evaluate strings for element chemical fomula.
    """
    elements = (r"A[cglmrstu]|B[aehikr]?|C[adeflmorsu]?|D[bsy]{0,1}|E[rsu]|F[emr]?|"
                r"G[ade]|H[efgosx]?|I[nr]?|Kr?|L[airu]|M[dgnot]|N[abdeiop]?|"
                r"Os?|P[abdmortu]?|R[abefghnu]|S[bcegimnr]?|T[abcehilm]|"
                r"Uu[bhopqst]|U|V|W|Xe|Yb?|Z[nr]")
    isotopes = (r"(A[cglmrstu]|B[aehikr]?|C[adeflmorsu]?|D[bsy]{0,1}|E[rsu]|F[emr]?|"
                r"G[ade]|H[efgos]?|I[nr]?|Kr?|L[airu]|M[dgnot]|N[abdeiop]?|"
                r"Os?|P[abdmortu]?|R[abefghnu]|S[bcegimnr]?|T[abcehilm]|"
                r"Uu[bhopqst]|U|V|W|Xe|Yb?|Z[nr])"
                r"\[[1-9][0-9]{0,2}\]")

    def __init__(self, string, sort=True):
        if isinstance(string, list):
            list.__init__(self, string)
            if isinstance(string, Formula):
                self._do_sort = string._do_sort
                self.HR_formula = string.HR_formula
            else:
                self._do_sort = sort
                self.HR_formula = str(self)
        else:
            self._do_sort = sort
            self.HR_formula = string
            list.__init__(self, [])
            self.parse_string(string)
            self.merge_same()

    def parse_string(self, string):
        # remove gaps and ignored characters
        string = string.replace(' ', '').replace('\t', '').replace('\n', '')
        string = string.replace('{', '').replace('}', '').replace('_', '').replace('$', '')

        groups = self.split_groups(string)
        for group, factor in groups:
            try:
                items = self.parse_group(group, case_sensitive=True)
            except ValueError:
                items = self.parse_group(group, case_sensitive=False)
            items = [(i[0], i[1]*factor) for i in items]
            # noinspection PyMethodFirstArgAssignment
            self += items

    @staticmethod
    def split_groups(string):
        if '(' not in string:
            return [(string, 1.0)]
        out = []
        start = string.index('(')
        end = start
        if start>0:
            out.append((string[:start], 1.0))
        while end<len(string):
            end = start+string[start:].find(')')
            _next = end+1
            if end<start:
                raise ValueError('Brackets need to be closed')
            while not (_next==len(string) or string[_next].isalpha() or string[_next]=='('):
                _next += 1
            block = string[start+1:end]
            if '(' in block:
                raise ValueError("Only one level of brackets is allowed")
            number = string[end+1:_next]
            if number=='':
                out.append((block, 1.0))
            else:
                out.append((block, float(number)))
            if _next==len(string):
                break
            if '(' not in string[_next:]:
                out.append((string[_next:], 1.0))
                break
            else:
                start = _next+string[_next:].index('(')
                end = start
                if start>_next:
                    out.append((string[_next:start], 1.0))
        return out

    def parse_group(self, group, case_sensitive=True):
        if case_sensitive:
            flags = 0
        else:
            flags = re.IGNORECASE
        out = []
        mele = re.search(self.elements, group, flags=flags)
        miso = re.search(self.isotopes, group, flags=flags)
        if miso is not None and miso.start()==mele.start():
            prev = miso
        else:
            prev = mele
        if prev is None or prev.start()!=0:
            raise ValueError('Did not find any valid element in string')
        pos = prev.end()
        while pos<len(group):
            mele = re.search(self.elements, group[pos:], flags=flags)
            miso = re.search(self.isotopes, group[pos:], flags=flags)
            if miso is not None and miso.start()==mele.start():
                _next = miso
            else:
                _next = mele
            if _next is None:
                break
            if _next.start()==0:
                out.append((prev.string[prev.start():prev.end()].capitalize(), 1.0))
            else:
                out.append((prev.string[prev.start():prev.end()].capitalize(),
                            float(group[pos:pos+_next.start()])))
            prev = _next
            pos += _next.end()
        if pos==len(group):
            out.append((prev.string[prev.start():].capitalize(), 1.0))
        else:
            out.append((prev.string[prev.start():prev.end()].capitalize(), float(group[pos:])))
        return out

    def merge_same(self):
        elements = OrderedDict({})
        for ele, amount in self:
            if ele in elements:
                elements[ele] += amount
            else:
                elements[ele] = amount
        self[:] = [items for items in elements.items() if items[1]!=0]
        if self._do_sort:
            self.sort()

    def __str__(self):
        output = ''
        for element, number in self:
            if number==1.0:
                output += element
            elif number.is_integer():
                output += element+str(int(number))
            else:
                output += element+str(number)
        return output

    def __contains__(self, item):
        # check if an element is in the formula
        return item in [el[0] for el in self]

    def index(self, item, **kwargs):
        return [el[0] for el in self].index(item, **kwargs)

    def __add__(self, other):
        out = Formula(self[:]+other[:], sort=self._do_sort)
        out.merge_same()
        return out

    def __sub__(self, other):
        sother = -1*other
        out = Formula(self[:]+sother[:], sort=self._do_sort)
        out.merge_same()
        return out

    def __mul__(self, other):
        return Formula([(el[0], other*el[1]) for el in self], sort=self._do_sort)

    def __rmul__(self, other):
        return self*other


class PolymerSequence(str):
    """
    Used to represent a chain of amino acids. Currently, no checking
    is done and this is just a string containing the appropriate letters.
    """


class Material:
    """
    Units used:
    b: fm
    fu_volume: Å³
    fu_dens: 1/Å³
    dens: g/cm³
    roh_n: Å^{-2}
    roh_m: Å^{-2}
    mu: muB/FU
    M: kA/m = emu/cm³
    """

    def __init__(self, elements, dens=None, fu_volume=None, rho_n=None, mu=0., xsld=None, xE=None,
                 fu_dens=None, M=None,
                 ID=None, name=None, extra_data=None):
        if type(elements) is str:
            elements = Formula(elements)
        if type(elements) is Formula:
            elements = [(get_element(element), amount) for element, amount in elements]
        self.elements = elements
        # generate formula unit density using different priority of possible inputs
        if fu_volume is not None:
            if dens is not None or fu_dens is not None:
                raise ValueError("fu_volume can't be supplied together with a density value")
            self.fu_dens = 1./fu_volume
        elif dens is not None:
            if fu_dens is not None:
                raise ValueError("dens and fu_dens can't be provided at the same time")
            self.fu_dens = dens/self.fu_mass/u2g*1e-24
        elif fu_dens is not None:
            self.fu_dens = fu_dens
        elif rho_n is not None:
            self.fu_dens = abs(rho_n/self.fu_b)/fm2angstrom
        elif xsld is not None and xE is not None:
            self.fu_dens = abs(xsld/self.f_of_E(xE))*(1./r_e_angstrom)
        else:
            raise ValueError(
                    "Need to provide means to calculate density, {dens, fu_volume, rho_n, xsld+xE}")
        if M is not None:
            if mu!=0.:
                raise ValueError("M and mu can't be provided at the same time")
            self.M = M
        else:
            self.mu = mu
        self.ID = ID
        self.name = name
        self.extra_data = extra_data or {}

    @property
    def fu_volume(self):
        return 1./self.fu_dens

    @property
    def rho_n(self):
        return self.fu_b*self.fu_dens*fm2angstrom  # Å^-1

    @property
    def rho_m(self):
        return self.M*rho_of_M

    @property
    def M(self):
        return self.mu*muB*self.fu_dens

    @M.setter
    def M(self, value):
        self.mu = value/self.fu_dens/muB

    def f_of_E(self, E=Cu_kalpha):
        f = 0.
        for element, number in self.elements:
            f += number*element.f_of_E(E)
        return f

    def rho_of_E(self, E):
        f = self.f_of_E(E)
        return f*r_e*self.fu_dens*fm2angstrom  # Å^-2

    def delta_of_E(self, E):
        rho = self.rho_of_E(E)
        lamda = E_to_lambda/E
        return lamda**2/2./pi*rho.real

    def beta_of_E(self, E):
        rho = self.rho_of_E(E)
        lamda = E_to_lambda/E
        return -lamda**2/2./pi*rho.imag

    def mu_of_E(self, E):
        rho = self.rho_of_E(E)
        lamda = E_to_lambda/E
        return -lamda*2.*rho.imag

    def rho_vs_E(self):
        # generate full energy range data for E,SLD
        E = self.elements[0][0].E
        for element, number in self.elements:
            E = E[(E>=element.E.min()) & (E<=element.E.max())]
        rho = array([self.rho_of_E(Ei) for Ei in E])
        return E, rho

    def delta_vs_E(self):
        E, rho = self.rho_vs_E()
        lamda = E_to_lambda/E
        return E, lamda**2/2./pi*rho.real

    def beta_vs_E(self):
        E, rho = self.rho_vs_E()
        lamda = E_to_lambda/E
        return E, -lamda**2/2./pi*rho.imag

    def mu_vs_E(self):
        E, rho = self.rho_vs_E()
        lamda = E_to_lambda/E
        return E, -lamda*2.*rho.imag

    @property
    def dens(self):
        return self.fu_mass*u2g*self.fu_dens*1e24  # g/cm³

    @property
    def fu_mass(self):
        m = 0.
        for element, number in self.elements:
            if element.mass is None:
                raise ValueError(f'No mass known for element {element}')
            m += number*element.mass
        return m

    @property
    def fu_b(self):
        b = 0.
        for element, number in self.elements:
            b += number*element.b
        return b

    @property
    def has_ndata(self):
        return any([element.has_ndata for element, number in self.elements])

    def b_of_L(self, Li):
        b = 0.
        for element, number in self.elements:
            b += number*element.b_of_L(Li)
        return b

    def rho_n_of_L(self, Li):
        return self.b_of_L(Li)*self.fu_dens*fm2angstrom

    def b_vs_L(self):
        # generate full energy range data for Lambda,SLD
        if not self.has_ndata:
            # no energy dependant cross-sections
            b0 = self.fu_b
            return array([0.05, 50.0]), array([b0, b0])

        L = [el for el, n in self.elements if el.has_ndata][0].Lamda
        b = array([self.b_of_L(Li) for Li in L])
        return L, b

    def rho_n_vs_L(self):
        L, b = self.b_vs_L()
        return L, b*self.fu_dens*fm2angstrom

    @property
    def formula(self):
        output = ''
        for element, number in self.elements:
            output += element.symbol+str(number)
        return Formula(output)

    @staticmethod
    def convert_subscript(number):
        if number==1.0:
            return ''
        nstr = str(number)
        out = ''
        for digit in nstr:
            if digit=='.':
                if number.is_integer():
                    break
                out += '.'
            else:
                out += SUBSCRIPT_DIGITS[int(digit)]
        return out

    @property
    def deuterated(self):
        # returns a copy of this material with all hydrogens replaced by deuterium but same fu_volume
        dformula = Formula(self.formula)
        if 'H' in dformula:
            Hidx = dformula.index('H')
            dformula[Hidx] = ('D', dformula[Hidx][1])
        if 'Hx' in dformula:
            Hidx = dformula.index('Hx')
            dformula[Hidx] = ('D', dformula[Hidx][1])
        dformula.merge_same()
        if self.name is None:
            dname = None
        else:
            dname = 'd'+self.name
        return Material(dformula, fu_dens=self.fu_dens, name=dname, extra_data=self.extra_data)

    def deuterate(self, fraction):
        """
        Return a partially deuterated molecule with fraction of D instead of H.
        """
        fh = self.formula
        fd = self.deuterated.formula
        return Material((1.0-fraction)*fh+fraction*fd, fu_dens=self.fu_dens, extra_data=self.extra_data)

    @property
    def edeuterated(self):
        # returns a copy of this material with all non-exchangable hydrogens replaced by deuterium but same fu_volume
        dformula = Formula(self.formula)
        if 'H' in dformula:
            Hidx = dformula.index('H')
            dformula[Hidx] = ('D', dformula[Hidx][1])
        dformula.merge_same()
        if self.name is None:
            dname = None
        else:
            dname = 'd'+self.name
        return Material(dformula, fu_dens=self.fu_dens, name=dname, extra_data=self.extra_data)

    @property
    def exchanged(self):
        # returns a copy of this material with all Hx replaced by deuterium but same fu_volume
        eformula = Formula(self.formula)
        if 'Hx' in eformula:
            Hidx = eformula.index('Hx')
            eformula[Hidx] = ('D', eformula[Hidx][1])
        eformula.merge_same()
        if self.name is None:
            dname = None
        else:
            dname = 'e'+self.name
        return Material(eformula, fu_dens=self.fu_dens, name=dname, extra_data=self.extra_data)

    @property
    def not_exchanged(self):
        # returns a copy of this material with all Hx replaced by normal hydrogen but same fu_volume
        eformula = Formula(self.formula)
        if 'Hx' in eformula:
            Hidx = eformula.index('Hx')
            eformula[Hidx] = ('H', eformula[Hidx][1])
        eformula.merge_same()
        if self.name is None:
            dname = None
        else:
            dname = 'e'+self.name
        return Material(eformula, fu_dens=self.fu_dens, name=dname, extra_data=self.extra_data)

    def exchange(self, D_fraction, D2O_fraction, exchange=0.9):
        """
        Return a partially deuterated modlecule within H2O/D2O solution given amount of exchange.
        """
        # fractionally deuterated molecule without Hx
        hd = (1.0-D_fraction)*self.not_exchanged.formula+D_fraction*self.deuterated.formula
        # fully exchanged molecule
        exchange_diff = self.exchanged.formula-self.not_exchanged.formula
        exh2o = hd-D_fraction*exchange_diff
        exd2o = hd-(D_fraction-1.0)*exchange_diff
        # partial exchanged molecule
        in_h2o = (1.0-exchange)*hd+exchange*exh2o
        in_d2o = (1.0-exchange)*hd+exchange*exd2o
        res_formula = (1.0-D2O_fraction)*in_h2o+D2O_fraction*in_d2o
        res_formula.merge_same()
        return Material(res_formula, fu_volume=self.fu_volume)

    @property
    def match_point(self):
        # return the fraction of H2O required to match the material contrast
        rh2o = H2O.rho_n.real
        rd2o = D2O.rho_n.real
        rh = self.exchange(0., 0.).rho_n.real
        rd = self.exchange(0., 1.).rho_n.real
        mp = (rh2o-rh)/(rd+rh2o-rh-rd2o)
        return mp

    def match_exchange(self, D_fraction=0.0, exchange=0.9):
        # return the D2O match point for a given deuteration and exchange fraction
        rh2o = H2O.rho_n.real
        rd2o = D2O.rho_n.real
        rh = self.exchange(D_fraction, 0., exchange=exchange).rho_n.real
        rd = self.exchange(D_fraction, 1., exchange=exchange).rho_n.real
        mp = (rh2o-rh)/(rd+rh2o-rh-rd2o)
        return mp

    def export(self, xray_units='sld'):
        """
        Export material data to dictionary.

        xray_units: one of "edens", "n_db" or "sld"
        """
        if xray_units not in ['sld', 'edens', 'n_db']:
            raise ValueError(f'Not a valid xray unit {xray_units}, use "edens", "n_db" or "sld"')
        out = {'ID': self.ID}
        out.update(self.extra_data)
        out['name'] = self.name
        out['formula'] = str(self.formula)
        out['density'] = self.dens
        out['fu_dens'] = self.fu_dens
        out['fu_volume'] = self.fu_volume
        out['fu_mass'] = self.fu_mass
        out['fu_b'] = repr(self.fu_b)
        out['rho_n'] = repr(self.rho_n)
        out['rho_n_mag'] = self.rho_m
        if self.has_ndata:
            L, rho_n = self.rho_n_vs_L()
            out['neutron_lambda'] = L.tolist()
            out['neutron_rho_real'] = rho_n.real.tolist()
            out['neutron_rho_imag'] = rho_n.imag.tolist()
        out['d2o_match_point'] = 100.*self.match_point
        out['mu'] = self.mu
        out['M'] = self.M
        if xray_units=='n_db':
            out['delta_Cu_kalpha'] = self.delta_of_E(Cu_kalpha)
            out['beta_Cu_kalpha'] = self.beta_of_E(Cu_kalpha)
            out['delta_Mo_kalpha'] = self.delta_of_E(Mo_kalpha)
            out['beta_Mo_kalpha'] = self.beta_of_E(Mo_kalpha)
            E, delta = self.delta_vs_E()
            E, beta = self.beta_vs_E()
            out['xray_E'] = E.tolist()
            out['xray_delta'] = delta.tolist()
            out['xray_beta'] = beta.tolist()
        else:
            if xray_units=='edens':
                factor = 1.0e5/r_e
            else:
                factor = 1.0
            out['rho_Cu_kalpha'] = repr(factor*self.rho_of_E(Cu_kalpha))
            out['rho_Mo_kalpha'] = repr(factor*self.rho_of_E(Mo_kalpha))
            E, rho = self.rho_vs_E()
            out['xray_E'] = E.tolist()
            out['xray_real'] = (factor*rho).real.tolist()
            out['xray_imag'] = (factor*rho).imag.tolist()
        out['units'] = {
            'density': 'g/cm**3', 'fu_dens': '1/angstrom**3', 'fu_volume': 'angstrom**3', 'fu_mass': 'u',
            'fu_b':    'fm', 'rho_n': '1/angstrom**2', 'rho_n_mag': '1/angstrom**2', 'mu': 'muB', 'M': 'emu/cm**3',
            'xray_E':  'keV'
            }
        if xray_units=='n_db':
            out['units']['xray_values'] = '1'
        elif xray_units=='edens':
            out['units']['xray_values'] = 'r_e/angstrom**3'
        else:
            out['units']['xray_values'] = '1/angstrom**2'
        if self.has_ndata:
            out['units']['neutron_lambda'] = 'angstrom'
            out['units']['neutron_values'] = '1/angstrom**2'
        return out

    def __add__(self, other):
        # add two materials by adding the chemical formula and FU_volume of each element.
        if type(other)!=type(self):
            raise ValueError('Can only combine two Material instances.')
        fout = dict(self.elements)
        for ele, number in other.elements:
            if ele in fout:
                fout[ele] += number
            else:
                fout[ele] = number
        fout = list(fout.items())
        Vout = self.fu_volume+other.fu_volume
        mout = self.mu+other.mu
        return Material(fout, fu_volume=Vout, mu=mout)

    def __mul__(self, other):
        """
        Calculate a multiple of this material, mostly useful when combining
        different components. The SLD should stay the same as the formula
        as well as the FU_volume are multiplied with the same amount.
        """
        if type(other) in [int, float]:
            if other<=0:
                raise ValueError("Can only mulitply material with positive number")
            fout = [(ele, number*other) for ele, number in self.elements]
            return Material(fout, fu_volume=self.fu_volume*other,
                            mu=self.mu*other)
        else:
            raise ValueError("Can only multiply material by scalar")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __str__(self):
        output = ''
        for element, number in self.elements:
            nstr = self.convert_subscript(number)
            output += element.symbol+nstr
        return output

    def __repr__(self):
        output = ''
        if self.name:
            output += f'{self.name}='
        output += 'Material('
        output += str([(ei.symbol, num) for ei, num in self.elements])
        output += f', fu_volume={self.fu_volume}'
        if self.ID:
            output += f', ID={self.ID}'
        output += ')'
        return output


H2O = Material(Formula('H2O'), dens=dens_H2O)
D2O = Material(Formula('D2O'), dens=dens_D2O)
