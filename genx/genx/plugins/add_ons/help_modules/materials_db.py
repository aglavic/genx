'''
Handling of a table of materials with associated density parameters.
Includes parsing of chemical formula string.
'''

import os
import re
import json

from genx.models.utils import bc, fp, fw, __bc_dict__  # @UnusedImport
from genx.models.lib.physical_constants import MASS_DENSITY_CONVERSION
from genx.core.custom_logging import iprint

# configuration file to store the known materials
try:
    import appdirs
except ImportError:
    config_path=os.path.expanduser(os.path.join('~', '.genx'))
else:
    config_path=appdirs.user_data_dir('GenX3', 'ArturGlavic')
if not os.path.exists(config_path):
    os.makedirs(config_path)
config_file=os.path.join(config_path, 'materials.cfg')

default_materials=[
    [[["Cr", 1.0]], "7.19*0.602214/51.9961"],
    [[["D", 2.0], ["O", 1.0]], "1.107*0.602214/20.0276"],
    [[["Fe", 1.0]], "7.874*0.602214/55.845"],
    [[["Fe", 2.0], ["O", 3.0]], "2.0/100.713"],
    [[["H", 2.0], ["O", 1.0]], "1*0.602214/18.0152"],
    [[["La", 0.7], ["Sr", 0.3], ["Mn", 1.0], ["O", 3.0]], "6.0/349.916"],
    [[["Ni", 1.0]], "8.908*0.602214/58.6934"],
    [[["Si", 1.0]], "2.329*0.602214/28.0855"],
    [[["Si", 1.0], ["O", 2.0]], "3.0/113.005"],
    [[["Sr", 1.0], ["Ti", 1.0], ["O", 3.0]], "1.0/(3.905**3)"],
    [[["Ti", 1.0]], "4.506*0.602214/47.867"],
    [[["Ti", 1.0], ["O", 2.0]], "4.0/(4.5318*5.5019*4.9063)"],
    ]


class Formula(list):
    '''
    Holds the elements and fraction for a chemical formula.
    Includes the possibility to compare materials that
    have different element orders or base in their fraction.
    '''

    def __init__(self, data):
        # check that the data is correct form, list of [Element, fraction] items.
        for di in data:
            if len(di)!=2:
                raise ValueError('Formula has to consist of [Element, fraction] entries')
            if not self.check_atom(di[0]):
                raise ValueError('Element/Isotope %s not in database'%di[0])
            di[1]=float(di[1])
        list.__init__(self, data)

    @classmethod
    def from_str(cls, estr):
        '''
        Try to parse a formula string,
        has to be a simple formula with no brackets.
        
        Most read errors are ignored, so the result might not be
        the intended formula if the entry format is wrong.
        Format should be:
        {Element 1}{frac 1}{Element 2}{frac 2} like Fe2O3, SiO2 or H2O
        
        Fractions of 1 may be ommited and non-natural isotopes are
        entered with ^{N} before an element like ^{56}Fe2O3.
        '''
        if '(' in estr or ')' in estr:
            raise ValueError("Can't parse formula with brackets.")
        for ign_char in [" ", "\t", "_", "-"]:
            estr=estr.replace(ign_char, "")
        for i in range(10):
            estr=estr.replace(cls._get_subscript(i), '%i'%i)  # convert to normal str w/o subsccript
        if estr=="":
            return Formula([])
        extracted_elements=[]
        i=0
        mlen=len(estr)
        while i<mlen:
            # find next element
            iso_match=re.search('\^\{[0-9]{1,3}\}[A-Z][a-zA-Z]{0,1}', estr[i:])
            normal_match=re.search('[A-Z][a-zA-Z]{0,1}', estr[i:])
            if iso_match is None and normal_match is None:
                break
            elif iso_match is None or iso_match.start()>normal_match.start():
                element=estr[i+normal_match.start():i+normal_match.end()].capitalize()
            else:
                element=estr[i+iso_match.start():i+normal_match.start()]
                element+=estr[i+normal_match.start():i+normal_match.end()].capitalize()
            i+=normal_match.end()

            j=0
            while i+j<mlen and (estr[i+j]=='.' or estr[i+j].isdigit()):
                j+=1
            count_txt=estr[i:i+j]
            i+=j
            if count_txt=='':
                count=1.
            else:
                try:
                    count=float(count_txt)
                except ValueError:
                    continue
            extracted_elements.append([element, count])
        return Formula(extracted_elements)

    @classmethod
    def from_bstr(cls, bstr):
        '''
        Try to parse a string of scattering length elements,
        the same type that is returned by self.b().
        '''
        items=map(str.strip, bstr.split('+'))
        extracted_elements=[]
        for item in items:
            if not item[:3] in ['bc.', 'bw.']:
                continue
            else:
                element, count=item[3:].split('*', 1)
                count=float(count)
                extracted_elements.append([element, count])
        return Formula(extracted_elements)

    def __str__(self):
        '''Generate a string with sub- and superscript numbers for material.'''
        output=''
        for element, count in self:
            if element.startswith('^'):
                try:
                    isotope, element=element[2:].split('}')
                    output+=self._get_superscript(int(isotope))
                except (IndexError, ValueError):
                    pass
            if count==1:
                output+=element
            else:
                output+=element+self._get_subscript(count)
        return output

    def estr(self):
        '''Generates an editable (ascii) string that can be parsed back as Formul'''
        output=''
        for element, count in self:
            if count==1:
                output+=element
            else:
                output+=element+"%g"%count
        return output

    @classmethod
    def _get_superscript(cls, count):
        '''
          Return a subscript unicode string that equals the given number.
        '''
        scount='%g'%count
        result=''
        for char in scount:
            if char=='.':
                result+='﹒'
            else:
                # a superscript digit in unicode
                result+=(b'\\u207'+char.encode('utf-8')).decode('unicode-escape')
        return result

    @classmethod
    def _get_subscript(cls, count):
        '''
          Return a subscript unicode string that equals the given number.
        '''
        scount='%g'%count
        result=''
        for char in scount:
            if char=='.':
                result+='﹒'
            else:
                # a subscript digit in unicode
                result+=(b'\\u208'+char.encode('utf-8')).decode('unicode-escape')
        return result

    def elements(self):
        '''Returns alphabetically sorted list of elements in formula.'''
        return list(sorted([ei[0].split('}')[-1] for ei in self]))

    def isotopes(self):
        '''Returns same as elements but with isotopes w/ number at end of list.'''
        return list(sorted([ei[0] for ei in self]))

    def amounts(self):
        '''
        Amounts as stored by elements alphabetically.
        Can be used to compare two formulas for equality.
        '''
        items=[(ei[0].split('}')[-1], ei[1]) for ei in self]
        items.sort()
        return [i[1] for i in items]

    def fractions(self):
        '''
        Fractional numbers sorted by elements alphabetically.
        Can be used to compare two formulas for equivalency.
        '''
        fractions=self.amounts()
        total=sum(fractions)
        return [f/total for f in fractions]

    def __eq__(self, other):
        if type(other)!=Formula:
            try:
                cmpo=Formula(other)
            except:
                return False
        return self.elements()==other.elements() and self.amounts()==other.amounts()

    def equivalent(self, other):
        '''
        Returns if formula contains equivalent elemental composition,
        ignores the size of the fomula unit. For equality use ==.
        '''
        cmpo=Formula(other)
        return self.elements()==other.elements() and self.fractions()==other.fractions()

    def mFU(self):
        '''Calculate mass in u for formula unit (FU).'''
        mass=0.
        for ei, fi in self:
            try:
                mass+=fi*atomic_data[ei][2]
            except KeyError:
                if ei.startswith('^{'):
                    raise KeyError('Unknonw isotope: "%s"'%ei)
                else:
                    raise KeyError('Element %s does not exist'%ei)
        # return 1. if formulat is empty to avoid division by zero in density
        return mass or 1.0

    @staticmethod
    def check_atom(atom_string):
        return atom_string in atomic_data

    def describe(self):
        '''Return a multile string with written element content.'''
        output=''
        for ei, (element, number) in enumerate(self):
            if element.startswith('^'):
                iso='-%s'%(element[2:].split('}')[0])
                if not element in atomic_data:
                    iso='-unknown isotope'
                    element=element.split('}')[-1]
            else:
                iso=''
            output+="%g x %s\n"%(number, atomic_data[element][0]+iso)
        return output[:-1]

    def f(self):
        '''Return string to be used in models to calculate scattering power f'''
        if len(self)==0:
            return '0.j'
        fw.set_wavelength(1.54)
        elements=''
        for element, count in self:
            if element in isotopes:
                element=isotopes[element][1]
            elements+='+fp.%s*%g'%(element, count)
        return elements[1:]

    def fw(self):
        '''Return string to be used in models to calculate scattering power f'''
        if len(self)==0:
            return '0.j'
        fw.set_wavelength(1.54)
        elements=''
        total=sum(self.amounts())
        for element, count in self:
            if element in isotopes:
                element=isotopes[element][1]
            elements+='+fw.%s*%g'%(element, count/total)
        return elements[1:]

    def b(self):
        '''Return string to be used in models to calculate scattering length b'''
        if len(self)==0:
            return '0.j'
        elements=''
        for element, count in self:
            if element in isotopes:
                element=isotopes[element][0]
            elements+='+bc.%s*%g'%(element, count)
        return elements[1:]

    def bw(self):
        '''Return string to be used in models to calculate scattering length b'''
        if len(self)==0:
            return '0.j'
        elements=''
        total=sum(self.amounts())
        for element, count in self:
            if element in isotopes:
                element=isotopes[element][0]
            elements+='+bw.%s*%g'%(element, count/total)
        return elements[1:]

class MaterialsDatabase(list):
    '''
    Holds a list of materials and associated methods.
    '''

    def __init__(self):
        if os.path.exists(config_file):
            try:
                known_materials=json.loads(open(config_file, 'r').read())
            except json.JSONDecodeError:
                iprint("Can't reload material list, file corrupted.")
                known_materials=default_materials
        else:
            known_materials=default_materials
        data=[self.prepare(mi) for mi in known_materials]
        list.__init__(self, data)

    def save_data(self):
        open(config_file, 'w').write(json.dumps(self))

    def prepare(self, item):
        if len(item)!=2:
            raise ValueError("Requires [Formula, density_str] entry")
        return [Formula(item[0]), item[1]]

    def append(self, item):
        list.append(self, self.prepare(item))
        self.save_data()

    def __getitem__(self, item):
        if type(item) in [str, Formula]:
            # Try to match formula with database
            if type(item) is str:
                item=Formula.from_str(item)
            for material in self:
                if material[0]==item:
                    return material
                if material[0].equivalent(item):
                    material=[item, material[1]+'*%g'%(material[0].mFU()/item.mFU())]
                    return material
            raise IndexError("Material %s no in database"%item.estr())
        else:
            return list.__getitem__(self, item)

    def __contains__(self, item):
        if type(item) in [str, Formula]:
            # Try to match formula with database
            if type(item) is str:
                item=Formula.from_str(item)
            for material in self:
                if material[0]==item:
                    return True
                if material[0].equivalent(item):
                    return True
            return False
        else:
            return list.__contains__(self, item)

    def SLDx(self, item):
        density=self.dens_FU(item)
        fw.set_wavelength(1.54)
        return density*eval(self[item][0].f())  # *2.82

    def SLDn(self, item):
        density=self.dens_FU(item)
        return density*eval(self[item][0].b())*10.

    def dens_FU(self, item):
        '''Returns the formula unit (FU) density of the compound "item" in 1/Å³'''
        return eval(self[item][1])

    def dens_mass(self, item):
        '''Returns the mass density of the compound "item" in g/cm³'''
        return eval(self[item][1])*self[item][0].mFU()/MASS_DENSITY_CONVERSION

    def __delitem__(self, index):
        list.__delitem__(self, index)
        self.save_data()

    def __setitem__(self, index, item):
        list.__setitem__(index, self.prepare(item))
        self.save_data()

    def insert(self, index, item):
        list.insert(self, index, self.prepare(item))
        self.save_data()

    def pop(self, index):
        list.pop(self, index)
        self.save_data()

# list of elements with their name, atomic number and atomic mass values (+GenX name)
# mostly to calculate atomic density from mass density
atomic_data={
    "D": ("Deuterium", 1, 2.01410178),
    "H": ("Hydrogen", 1, 1.0079),
    "He": ("Helium", 2, 4.0026),
    "Li": ("Lithium", 3, 6.941),
    "Be": ("Beryllium", 4, 9.0122),
    "B": ("Boron", 5, 10.811),
    "C": ("Carbon", 6, 12.0107),
    "N": ("Nitrogen", 7, 14.0067),
    "O": ("Oxygen", 8, 15.9994),
    "F": ("Fluorine", 9, 18.9984),
    "Ne": ("Neon", 10, 20.1797),
    "Na": ("Sodium", 11, 22.9897),
    "Mg": ("Magnesium", 12, 24.305),
    "Al": ("Aluminum", 13, 26.9815),
    "Si": ("Silicon", 14, 28.0855),
    "P": ("Phosphorus", 15, 30.9738),
    "S": ("Sulfur", 16, 32.065),
    "Cl": ("Chlorine", 17, 35.453),
    "Ar": ("Argon", 18, 39.948),
    "K": ("Potassium", 19, 39.0983),
    "Ca": ("Calcium", 20, 40.078),
    "Sc": ("Scandium", 21, 44.9559),
    "Ti": ("Titanium", 22, 47.867),
    "V": ("Vanadium", 23, 50.9415),
    "Cr": ("Chromium", 24, 51.9961),
    "Mn": ("Manganese", 25, 54.938),
    "Fe": ("Iron", 26, 55.845),
    "Co": ("Cobalt", 27, 58.9332),
    "Ni": ("Nickel", 28, 58.6934),
    "Cu": ("Copper", 29, 63.546),
    "Zn": ("Zinc", 30, 65.39),
    "Ga": ("Gallium", 31, 69.723),
    "Ge": ("Germanium", 32, 72.64),
    "As": ("Arsenic", 33, 74.9216),
    "Se": ("Selenium", 34, 78.96),
    "Br": ("Bromine", 35, 79.904),
    "Kr": ("Krypton", 36, 83.8),
    "Rb": ("Rubidium", 37, 85.4678),
    "Sr": ("Strontium", 38, 87.62),
    "Y": ("Yttrium", 39, 88.9059),
    "Zr": ("Zirconium", 40, 91.224),
    "Nb": ("Niobium", 41, 92.9064),
    "Mo": ("Molybdenum", 42, 95.94),
    "Tc": ("Technetium", 43, 98),
    "Ru": ("Ruthenium", 44, 101.07),
    "Rh": ("Rhodium", 45, 102.906),
    "Pd": ("Palladium", 46, 106.42),
    "Ag": ("Silver", 47, 107.868),
    "Cd": ("Cadmium", 48, 112.411),
    "In": ("Indium", 49, 114.818),
    "Sn": ("Tin", 50, 118.71),
    "Sb": ("Antimony", 51, 121.76),
    "Te": ("Tellurium", 52, 127.6),
    "I": ("Iodine", 53, 126.904),
    "Xe": ("Xenon", 54, 131.293),
    "Cs": ("Cesium", 55, 132.905),
    "Ba": ("Barium", 56, 137.327),
    "La": ("Lanthanum", 57, 138.905),
    "Ce": ("Cerium", 58, 140.116),
    "Pr": ("Praseodymium", 59, 140.908),
    "Nd": ("Neodymium", 60, 144.24),
    "Pm": ("Promethium", 61, 145),
    "Sm": ("Samarium", 62, 150.36),
    "Eu": ("Europium", 63, 151.964),
    "Gd": ("Gadolinium", 64, 157.25),
    "Tb": ("Terbium", 65, 158.925),
    "Dy": ("Dysprosium", 66, 162.5),
    "Ho": ("Holmium", 67, 164.93),
    "Er": ("Erbium", 68, 167.259),
    "Tm": ("Thulium", 69, 168.934),
    "Yb": ("Ytterbium", 70, 173.04),
    "Lu": ("Lutetium", 71, 174.967),
    "Hf": ("Hafnium", 72, 178.49),
    "Ta": ("Tantalum", 73, 180.948),
    "W": ("Tungsten", 74, 183.84),
    "Re": ("Rhenium", 75, 186.207),
    "Os": ("Osmium", 76, 190.23),
    "Ir": ("Iridium", 77, 192.217),
    "Pt": ("Platinum", 78, 195.078),
    "Au": ("Gold", 79, 196.966),
    "Hg": ("Mercury", 80, 200.59),
    "Tl": ("Thallium", 81, 204.383),
    "Pb": ("Lead", 82, 207.2),
    "Bi": ("Bismuth", 83, 208.98),
    "Po": ("Polonium", 84, 209),
    "At": ("Astatine", 85, 210),
    "Rn": ("Radon", 86, 222),
    "Fr": ("Francium", 87, 223),
    "Ra": ("Radium", 88, 226),
    "Ac": ("Actinium", 89, 227),
    "Th": ("Thorium", 90, 232.038),
    "Pa": ("Protactinium", 91, 231.036),
    "U": ("Uranium", 92, 238.029),
    "Np": ("Neptunium", 93, 237),
    "Pu": ("Plutonium", 94, 244),
    "Am": ("Americium", 95, 243),
    "Cm": ("Curium", 96, 247),
    "Bk": ("Berkelium", 97, 247),
    "Cf": ("Californium", 98, 251),
    "Es": ("Einsteinium", 99, 252),
    "Fm": ("Fermium", 100, 257),
    "Md": ("Mendelevium", 101, 258),
    "No": ("Nobelium", 102, 259),
    "Lr": ("Lawrencium", 103, 262),
    "Rf": ("Rutherfordium", 104, 261),
    "Db": ("Dubnium", 105, 262),
    "Sg": ("Seaborgium", 106, 266),
    "Bh": ("Bohrium", 107, 264),
    "Hs": ("Hassium", 108, 277),
    "Mt": ("Meitnerium", 109, 268),
    }

isotopes={
    'D': ('i2H', 'H'),
    }

'''
    Go through the database of neutron scattering length and generate
    isotope list accordingly.
'''
for key in __bc_dict__.keys():
    if not key.startswith('i'):
        continue
    try:
        N=int(re.findall(r'\d+', key)[0])
    except IndexError:
        continue
    element=key[len('i%i'%N):].capitalize()
    isokey='^{%i}%s'%(N, element)
    n='i%i%s'%(N, element)
    isotopes[isokey]=(n, element)
    atomic_data[isokey]=(atomic_data[element][0],
                         atomic_data[element][1],
                         float(N))

mdb=MaterialsDatabase()
