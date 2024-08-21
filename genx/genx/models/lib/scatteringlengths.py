""" A library for handling optical constants and scattering lengths
in an efficent way. Based on a base class database which is subclassed
for each case.
Programmer: Matts Bjorck
Last changed: 2009-03-19
"""

import string

import numpy as np

# Imports needed for the dispersive table implementation
from scipy import interpolate

from genx.core.custom_logging import iprint

from . import refl_base as refl


# ==============================================================================
class Func(object):
    """A function object which stores the real function so it can be
    dynamically replaced from its parents.
    """

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        return self.func(*args)

    def replace_func(self, func):
        self.func = func


class Proxy(object):
    """Proxy class borrowed from ASPN Recipe 496741 at adress
    http://code.activestate.com/recipes/496741/
    """

    __slots__ = ["_obj", "__weakref__"]

    def __init__(self, obj):
        object.__setattr__(self, "_obj", obj)

    #
    # proxying (special cases)
    #
    def __getattribute__(self, name):
        return getattr(object.__getattribute__(self, "_obj"), name)

    def __delattr__(self, name):
        delattr(object.__getattribute__(self, "_obj"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_obj"), name, value)

    def __bool__(self):
        return bool(object.__getattribute__(self, "_obj"))

    def __str__(self):
        return str(object.__getattribute__(self, "_obj"))

    def __repr__(self):
        return repr(object.__getattribute__(self, "_obj"))

    def __call__(self, *args, **kwargs):
        return object.__getattribute__(self, "_obj").__call__(*args, **kwargs)

    def _change_object(self, obj):
        object.__setattr__(self, "_obj", obj)

    #
    # factories
    #
    _special_names = [
        "__abs__",
        "__add__",
        "__and__",
        "__cmp__",
        "__coerce__",
        "__contains__",
        "__delitem__",
        "__delslice__",
        "__div__",
        "__divmod__",
        "__eq__",
        "__float__",
        "__floordiv__",
        "__ge__",
        "__getitem__",
        "__getslice__",
        "__gt__",
        "__hash__",
        "__hex__",
        "__iadd__",
        "__iand__",
        "__idiv__",
        "__idivmod__",
        "__ifloordiv__",
        "__ilshift__",
        "__imod__",
        "__imul__",
        "__int__",
        "__invert__",
        "__ior__",
        "__ipow__",
        "__irshift__",
        "__isub__",
        "__iter__",
        "__itruediv__",
        "__ixor__",
        "__le__",
        "__len__",
        "__long__",
        "__lshift__",
        "__lt__",
        "__mod__",
        "__mul__",
        "__ne__",
        "__neg__",
        "__oct__",
        "__or__",
        "__pos__",
        "__pow__",
        "__radd__",
        "__rand__",
        "__rdiv__",
        "__rdivmod__",
        "__reduce__",
        "__reduce_ex__",
        "__repr__",
        "__reversed__",
        "__rfloorfiv__",
        "__rlshift__",
        "__rmod__",
        "__rmul__",
        "__ror__",
        "__rpow__",
        "__rrshift__",
        "__rshift__",
        "__rsub__",
        "__rtruediv__",
        "__rxor__",
        "__setitem__",
        "__setslice__",
        "__sub__",
        "__truediv__",
        "__xor__",
        "next",
    ]

    # @classmethod #This is the new way of doing a classmethod, starting from python 2.4
    def _create_class_proxy(cls, theclass):
        """creates a proxy for the given class"""

        def make_method(name):
            def method(self, *args, **kw):
                return getattr(object.__getattribute__(self, "_obj"), name)(*args, **kw)

            return method

        namespace = {}
        for name in cls._special_names:
            if hasattr(theclass, name) and not hasattr(cls, name):
                namespace[name] = make_method(name)
        return type("%s(%s)" % (cls.__name__, theclass.__name__), (cls,), namespace)

    # And, this is the old way. Which I stck with for compatiblity reasons
    _create_class_proxy = classmethod(_create_class_proxy)

    def __new__(cls, obj, *args, **kwargs):
        """
        creates an proxy instance referencing `obj`. (obj, *args, **kwargs) are
        passed to this class' __init__, so deriving classes can define an
        __init__ method of their own.
        note: _class_proxy_cache is unique per deriving class (each deriving
        class must hold its own cache)
        """
        try:
            cache = cls.__dict__["_class_proxy_cache"]
        except KeyError:
            cls._class_proxy_cache = cache = {}
        try:
            theclass = cache[obj.__class__]
        except KeyError:
            cache[obj.__class__] = theclass = cls._create_class_proxy(obj.__class__)
        ins = object.__new__(theclass)
        # theclass.__init__(ins, obj, *args, **kwargs)
        return ins


def change_proxyobject(proxy, obj):
    """
    changes the current object of proxy to the obj.
    Note that evil things can happen if not care is taken so the objects are
    of the same class
    """
    object.__setattr__(proxy, "_obj", obj)


# ==============================================================================
class Database(object):
    """A database class where object memebers are dynamically acessed
    and stored. I.e. the constants are only looked up when needed.
    """

    def __init__(self):
        """__init__(self) --> None"""
        object.__setattr__(self, "stored_values", {})

    def __getattribute__(self, name):
        """__getattr__(self, name) --> object

        Looks up and returns the attribure name
        """
        if name.startswith("__"):
            return object.__getattribute__(self, name)
        name = name.lower()
        stored_values = object.__getattribute__(self, "stored_values")
        if name in stored_values:
            return stored_values[name]
        else:
            try:
                stored_values[name] = object.__getattribute__(self, "lookup_value")(name)
            except (LookupError, IOError) as e:
                raise LookupError("The name %s does not exist in the" "database" % name)
            return stored_values[name]

    def __setattr__(self, name, value):
        """__setattr__(self, name) --> None

        Just overlaoding the setattribute so a object cant be set.
        """
        raise TypeError("This object does not support assignements")

    def lookup_value(self, name):
        """lookup_value(self, name) --> object

        Used to (externally) lookup a value in a database to be inserted in
        local one for this object.
        """
        iprint("Looking up value")
        return 1

    def reset_database(self):
        """reset_database(self) --> None

        Resets the internal database
        """
        stored_values = object.__getattribute__(self, "stored_values")
        stored_values = {}


# ==============================================================================
class FormFactor(Database):
    """A database for the x-ray formfactor which includes the
    anomulous part as well as the angle dependent part. The object will
    return a function of sin(theta)/lambda
    """

    def __init__(self, wavelength, f_calc):
        """__init__(self, wavelength, f_calc) --> None

        wavelength [float] the wavelength of the radiation in AA, f_calc a
        function which calculates (lookup the scattering factor)
        """
        Database.__init__(self)
        object.__setattr__(self, "wavelength", wavelength)
        object.__setattr__(self, "f_calc", f_calc)

    def set_wavelength(self, wavelength):
        """set_wavelength(self, wavelength) --> None

        sets the wavelength of the database. This will reset the
        all the values. I.e when called each value will be reloaded
        """
        # First check so we actually change the wavelength ...
        if abs(wavelength - object.__getattribute__(self, "wavelength")) > 1e-10:
            object.__setattr__(self, "wavelength", wavelength)
            # object.__getattribute__(self, 'reset_database')()
            stored_vals = object.__getattribute__(self, "stored_values")
            # for key in stored_vals:
            #    f = object.__getattribute__(self, 'lookup_value')(key)
            #    change_proxyobject(stored_vals[key], f)
            # Removing a bug with proxy adding does not work properly
            for key in stored_vals:
                f = object.__getattribute__(self, "lookup_value")(key)
                stored_vals[key] = f

    def __getattribute__(self, name):
        """__getattribute__(self, name) --> value

        Finds and returns a attribute
        """
        if name == "set_wavelength" or name.startswith("__"):
            return object.__getattribute__(self, "set_wavelength")
        else:
            # print 'getting value'
            return Database.__getattribute__(self, name)

    def lookup_value(self, name):
        """lookup_value(self, name) --> f [function]

        looks up a value in the external database
        """
        wl = object.__getattribute__(self, "wavelength")
        # f = Proxy((object.__getattribute__(self, 'f_calc')(name, wl)))
        # Removing a bug with proxy adding does not work properly
        f = object.__getattribute__(self, "f_calc")(name, wl)
        return f


# ==============================================================================
class ScatteringLength(Database):
    """A simpler version of the database where everything is loaded at once
    good for simple and small databases. I.e the scatteringlength for
    a neutron which is largely wavelength independent.
    """

    def __init__(self, values):
        """__init__(self, values) --> None

        values is a dictonary of key value pairs that are used in the database.
        """
        Database.__init__(self)
        object.__setattr__(self, "stored_values", values)

    def lookup_value(self, name):
        raise LookupError("The element %s does not exist in the database" % name)


# ==============================================================================
# Function to load databases and or values


def load_f0dabax(filename, create_rho=False):
    """load_dabax(filename) --> f0 (dictonary)

    loads a dabax file with f0 values and return a dictonary with
    f0(sin(theta)/lambda) for the element given by the key name.
    """

    def create_f(a, b, c):
        def f(sin_over_lambda):
            """f(sin_over_lambda) --> float

            The atomic form factor for x-rays. The non-dispertive part.
            """
            return c + sum(a[:, np.newaxis] * np.exp(-b[:, np.newaxis] * sin_over_lambda**2), 0)

        return f

    def create_rho(a, b, c):
        def rho(r, fp, wb, occ):
            return (
                sum(
                    a[:, np.newaxis]
                    / (8.0 * (np.pi * (b[:, np.newaxis] / 16 / np.pi**2 + wb)) ** 1.5)
                    * np.exp(-(r**2.0) / 4.0 / (b[:, np.newaxis] / 16 / np.pi**2 + wb)),
                    0,
                )
                + (c + fp) / (8.0 * (np.pi * wb) ** 1.5) * np.exp(-(r**2.0) / 4.0 / wb)
            ) * occ

        return rho

    with open(filename, "r") as f:
        real_label = ""
        temp_dict = {}
        for line in f.readlines():
            # Get the label for each line
            if line[0] == "#":
                label = line[1]
                ret = line[1:-1]
            else:
                label = "D"
                ret = line[:-1]
            # Gets the real label, atom name
            if label == "S":
                real_label = ret.split()[-1]
            # The row contains data
            if label == "D":
                temp_dict[real_label.lower()] = [float(x) for x in ret.split()]

    f0 = {}
    rho0 = {}
    for key in list(temp_dict.keys()):
        temp = temp_dict[key]
        # change the name to not have operators...
        if key[-1] == "-":
            key = key[:-1] + "m"
        if key[-1] == "+":
            key = key[:-1] + "p"
        f0[key] = create_f(np.array(temp[:4]), np.array(temp[5:]), temp[4])
        rho0[key] = create_rho(np.array(temp[:4]), np.array(temp[5:]), temp[4])
    if create_rho:
        return f0, rho0
    else:
        return f0


def create_fp_lookup(path):
    """create_f_lookup(filename) --> lookup_func(name, wavelength)

    Creates a lookup function to lookup element names and returns a function
    that yields dispersive scattering factors at Q = 0. NOTE wavelengths in AA
    """

    def lookup_func(name, wavelength):
        """lookup_func(name, wavelength) --> fp = f1 - 1.0J*f2

        looksup the scattering factors for a given wavelength given in AA for
        a element (note that the databases does not support ions).
        The data given is the dispersive part + f0 (non dispersive) at Q = 0.
        """
        e, f1, f2 = np.loadtxt(path + "%s.nff" % name.lower(), skiprows=1, unpack=True)
        energy = 1239.842 / wavelength * 10
        if energy >= e[-2] or energy <= e[1]:
            raise ValueError(
                "The energy/wavelength is outside the databse"
                + "range, the energy should be inside [%f,%f] " % (e[1], e[-2])
            )
        pos1 = np.argmin(abs(e - energy))
        # Is the energy point to the right or left of the current point
        # If it is ontop it doesn't matter since the interpolation will be exact at
        # the endpoints
        if (e[pos1] - energy) > 0:
            pos2 = pos1 - 1
        else:
            pos2 = pos1 + 1
        # A quick linear interpolation:
        f1_e = (energy - e[pos1]) * (f1[pos2] - f1[pos1]) / (e[pos2] - e[pos1]) + f1[pos1]
        f2_e = (energy - e[pos1]) * (f2[pos2] - f2[pos1]) / (e[pos2] - e[pos1]) + f2[pos1]

        return f1_e - 1.0j * f2_e

    return lookup_func


def create_bl_lookup(path, b_dict):
    """ """

    def lookup_func(name, wavelength):
        """lookup_func(name, wavelength) --> b = b_s - 1.0J*b_a

        looksup the energy dependent absorption cross sections
        and calculated the complex scattering length from it.
        """
        energy = (0.28601435 / wavelength) ** 2  # convert wavelength in Å to energy in eV

        if len(name) > 1 and name[0].lower() == "i" and name[1].isdigit():
            if name[-2].isdigit():
                ele = name[-1] + name[1:-1]
            else:
                ele = name[-2:] + name[1:-2]
        else:
            ele = name.capitalize()

        e, xs_s, xs_a = np.loadtxt(path + "%s.txt" % ele, unpack=True)

        if energy >= e[-2] or energy <= e[1]:
            raise ValueError(
                "The energy/wavelength is outside the databse"
                + "range, the energy should be inside [%f,%f] " % (e[1], e[-2])
            )
        pos1 = np.argmin(abs(e - energy))
        # Is the energy point to the right or left of the current point
        # If it is ontop it doesn't matter since the interpolation will be exact at
        # the endpoints
        if (e[pos1] - energy) > 0:
            pos2 = pos1 - 1
        else:
            pos2 = pos1 + 1
        # A quick linear interpolation:
        xs_si = (energy - e[pos1]) * (xs_s[pos2] - xs_s[pos1]) / (e[pos2] - e[pos1]) + xs_s[pos1]
        xs_ai = (energy - e[pos1]) * (xs_a[pos2] - xs_a[pos1]) / (e[pos2] - e[pos1]) + xs_a[pos1]

        breal = b_dict[name.lower()].real
        bimag = xs_ai / wavelength / 2.0 * 1e-3
        return breal - 1j * bimag

    return lookup_func


def create_fpdisp_lookup(path):
    """create_f_lookup(filename) --> lookup_func(name)

    Creates a lookup function to lookup element names and returns a function
    that yields a function of dispersive scattering factors f(E) at Q = 0. NOTE energy is in eV
    """

    def create_dispersion_func(name):
        e, f1, f2 = np.loadtxt(path + "%s.nff" % name.lower(), skiprows=1, unpack=True)
        f1interp = interpolate.interp1d(e, f1, kind="linear")
        f2interp = interpolate.interp1d(e, f2, kind="linear")

        def f(energy):
            return f1interp(energy) - 1.0j * f2interp(energy)

        return refl.ReflFunction(f, (np.mean(e),), {}, id="f(E)")

    return create_dispersion_func


def create_f_lookup(lookup_fp, f0):
    """create_f_lookup(lookup_fp, f0) --> lookup_func(name, wavelength)

    combines a f0 dictonary [dict] witht the lookup function for the
     anoumoulous scattering factors. Note that it is assumed that lookup_fp
    is a function on the form lookup_fp(name, wavelength) --> f1 - 1.0Jf2
    where name is the name of the element!
    """

    def lookup_func(name, wavelength):
        """Looks up the total angle dependent form factor
        f = f0 + f1 + 1.0J*f2 of element name. This dispersive part
        is independent on the ionicity but f0 not.
        """
        # Check if name corrspond to an ion
        element = None
        if len(name) > 1:
            if name[-2].isdigit():
                element = name[:-2]
            else:
                element = name
        else:
            element = name
        # Remove the non dispersive part for the given element
        fp = lookup_fp(element, wavelength) - f0[element](0)
        f0_name = f0[name]

        def f(sin_over_lambda):
            return f0_name(sin_over_lambda) + fp

        return f

    return lookup_func


def create_rho_lookup(lookup_fp, rho0, f0):
    """create_rho_lookup(lookup_fp, rho0, f0) --> lookup_func(name, wavelength)

    combines a rho0 dictonary [dict] with the lookup function for the
     anoumoulous scattering factors. Note that it is assumed that lookup_fp
    is a function on the form lookup_fp(name, wavelength) --> f1 - 1.0Jf2
    where name is the name of the element!
    """

    def lookup_func(name, wavelength):
        """Looks up the "electron density" of an atom as described by
        its formfactor, f = f0 + f1 + 1.0J*f2, of element name.
        The dispersive part is independent on the ionicity but f0 not.
        """
        # Check if name corrspond to an ion
        element = None
        if len(name) > 1:
            if name[-2].isdigit():
                element = name[:-2]
            else:
                element = name
        else:
            element = name
        # Remove the non dispersive part for the given element
        fp = lookup_fp(element, wavelength) - f0[element](0)
        rho0_name = rho0[name]

        def rho(r, wb, occ):
            return rho0_name(r, fp, wb, occ)

        return rho

    return lookup_func


def create_fw_lookup(lookup_fp, weight):
    """create_fw_lookup(lookup_fp, weight) --> lookup_func(name, wavelength)

    combines a f0 dictonary [dict] witht the lookup function for the
     anoumoulous scattering factors. Note that it is assumed that lookup_fp
    is a function on the form lookup_fp(name, wavelength) --> f1 - 1.0Jf2
    where name is the name of the element!
    """

    def lookup_func(name, wavelength):
        """Looks up the total form factor at Q = 0
        f = f1 + 1.0J*f2 of element name.
        """
        fp = lookup_fp(name, wavelength)
        w = weight[name]

        return fp / (w / 0.6022141)

    return lookup_func


def load_bdabax(filename):
    """load_bdabax(filename) --> b (dictonary)

    loads a dabax file with b (sld for neutrons) values
    and return a dictonary with b for the elements and isotopes
    given by the key name.
    """
    temp_dict = read_dabax(filename)
    # We are intrested in the scatttering lengths
    b_c = {}
    for key in temp_dict:
        val = temp_dict[key][1] + temp_dict[key][2] * 1.0j
        if key[0].isdigit():
            b_c["i" + key] = val
        else:
            b_c[key] = val

    return b_c


def load_fdabax(filename):
    """loads a dabax file with the scattering length tables returns a lookup
    function so that the wavelength can be changed."""
    table = read_dabax(filename)

    def lookup_func(name, wavelength):
        """Looks up the total form factor at Q = 0
        f = f1 + 1.0J*f2 of element name.
        """
        el = np.array(table[name])
        el = el.reshape(len(el) / 7, 7)
        e = el[:, 0]
        f1 = el[:, 1]
        f2 = el[:, 2]
        energy = 1239.842 / wavelength * 10 / 1e3  # keV
        if energy >= e[-2] or energy <= e[1]:
            raise ValueError(
                "The energy/wavelength is outside the databse"
                + "range, the energy should be inside [%f,%f] " % (e[1], e[-2])
            )
        pos1 = np.argmin(abs(e - energy))
        # Is the energy point to the right or left of the current point
        # If it is ontop it doesn't matter since the interpolation will be exact at
        # the endpoints
        if (e[pos1] - energy) > 0:
            pos2 = pos1 - 1
        else:
            pos2 = pos1 + 1
        # A quick linear interpolation:
        f1_e = (energy - e[pos1]) * (f1[pos2] - f1[pos1]) / (e[pos2] - e[pos1]) + f1[pos1]
        f2_e = (energy - e[pos1]) * (f2[pos2] - f2[pos1]) / (e[pos2] - e[pos1]) + f2[pos1]

        return f1_e - 1.0j * f2_e

    return lookup_func


def read_dabax(filename):
    """read_dabax_dict(filename) --> temp_dict [dict]

    Loads an entire dabax file to a dictonary. Scan names are the
    keys.
    """

    def tofloat(x):
        try:
            f = float(x)
        except:
            f = np.nan
        return f

    real_label = ""
    temp_dict = {}
    with open(filename) as f:
        for line in f.readlines():
            # Get the label for each line
            if line[0] == "#":
                label = line[1]
                ret = line[1:-1]
            else:
                label = "Data"
                ret = line[:-1]
            # Gets the real label, atom name
            if label == "S":
                real_label = ret.split()[-1]
            # The row contains data
            if label == "Data":
                # To get all values in the table
                if real_label.lower() not in temp_dict:
                    temp_dict[real_label.lower()] = [tofloat(x.split("(")[0]) for x in ret.split()]
                else:
                    temp_dict[real_label.lower()] += [tofloat(x.split("(")[0]) for x in ret.split()]

    return temp_dict


def load_atomic_weights_dabax(filename):
    """load__atomic_weights_dabax(filename) --> w (dictonary)

    loads a dabax file with w (sld for neutrons) values
    and return one dictonary with the atomic weights for the elements
    by the key name. Note that all isotopes are includes with the an
    i in front of them.
    """

    temp_dict = read_dabax(filename)
    # We are intrested in the scatttering lengths
    w_mean = {}
    for key in temp_dict:
        if key[0].isdigit():
            # This is an isotope
            w_mean["i" + key] = temp_dict[key][0]
            if len(temp_dict[key]) == 3:
                # The mean atomic mass for the element
                w_mean[key.lstrip(string.digits)] = temp_dict[key][2]

    return w_mean


def create_scatt_weight(scatt_dict, w_dict):
    """create_bw(scatt_dict, w_dict) --> sw_dict

    Makes a scattering length database for using with densities (g/cm3)
    """
    sw_dict = {}
    for key in scatt_dict:
        # print key, ' ',scatt_dict[key], ' ',w_dict[key]
        if key in w_dict:
            sw_dict[key] = scatt_dict[key] / complex(w_dict[key] / 0.6022141)
    return sw_dict
