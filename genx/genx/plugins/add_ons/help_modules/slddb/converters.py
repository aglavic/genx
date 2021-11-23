"""
Classes for conversion of specific data into sqlite types and back.
"""
import re
from abc import ABC, abstractmethod
from numpy import array, frombuffer
from datetime import datetime
from .material import Formula
from .comparators import GenericComparator, FormulaComparator, FuzzyFloat


SQLITE_TYPES = [int, float, str, bytes]
SQLITE_STR = {int: 'INT', float: 'REAL', str: 'TEXT', bytes: 'BLOB'}


class Converter(ABC):
    """
    Base class for all other converters, can't be used stand alone.
    """
    sql_type = "TEXT"  # if subclass does not define the SQLite type assume TEXT for compatibility
    html_list = False  # only used for types that require to query a list in html requests
    comparator = GenericComparator
    html_title = None

    @abstractmethod
    def __init__(self):
        "Converters is an abstract class"

    def validate(self, data):
        # Default behavior is to just try to convert.
        try:
            self.convert(data)
        except Exception:
            return False
        else:
            return True

    @abstractmethod
    def convert(self, data):
        """Converts data to SQLite format"""

    def revert(self, db_data):
        # Default behavior is to return the database data directly
        return db_data

    def revert_serializable(self, db_data):
        # generage json serialisable value, default is the normal value
        return self.revert(db_data)

    def html_input(self, field, value):
        # return a string containing the input key for html entry template
        if self.html_title:
            return f'<input type="text" name="{field}" id="compound {field}" value="{value}" ' \
                   f'title = "{self.html_title}">'
        else:
            return f'<input type="text" name="{field}" id="compound {field}" value="{value}">'


class CType(Converter):

    # converts between a python type and SQLite type

    def __init__(self, fromtype, dbtype, db_repstr=None):
        self._fromtype = fromtype
        if dbtype not in SQLITE_TYPES:
            raise TypeError("Type %s is not a valid SQLite type"%
                            dbtype.__name__)
        if fromtype is float:
            self.comparator = FuzzyFloat
            self.html_title = 'when searching: value|value+/-10%|range: 13.4|~13.4|13-14'
        self._dbtype = dbtype
        if db_repstr is None:
            self.sql_type = SQLITE_STR[dbtype]
        else:
            self.sql_type = db_repstr

    def convert(self, data):
        value = self._fromtype(data)
        return self._dbtype(value)

    def revert(self, db_data):
        if db_data is None:
            return None
        elif type(db_data) is not self._dbtype:
            raise ValueError(
                    "Wrong type of database data %s, expecte %s"%(
                        type(db_data).__name__,
                        self._dbtype.__name__)
                    )
        else:
            return self._fromtype(db_data)


class CDate(Converter):
    sql_type = 'TEXT'

    def __init__(self):
        pass

    def convert(self, data):
        if type(data) is datetime:
            return data.strftime('%Y-%m-%d %H:%M:%S')
        else:
            return data

    def revert(self, db_data):
        if db_data is not None:
            return datetime.strptime(db_data, '%Y-%m-%d %H:%M:%S')
        else:
            return None

    def revert_serializable(self, db_data):
        return db_data

    def html_input(self, field, value):
        return f'<input type="text" name="{field}" id="compound {field}" value="{value}"' \
               ' placeholder="date: {year}-{month}-{day} {hours}:{minutes}:{seconds}" title="2021-01-10 00:00:00" />'


class CFormula(Converter):
    comparator = FormulaComparator

    def __init__(self):
        pass

    def convert(self, data):
        res = Formula(data)
        return str(res)

    def revert(self, db_data):
        return db_data

    def html_input(self, field, value):
        return f'<input type="text" name="{field}" id="compound {field}" value="{value}"' \
               ' placeholder="Fe2O3 / H[2]2O / H2(C2H4)4" title="Chemical Formula: ' \
               'Fe2O3 / H[2]2O / H2(C2H4)4 / ~F" />' \
               '<div class="tooltip">🛈<div class="tooltiptext">' \
               'Chemical formula of a compound can be provided in various ways. ' \
               'It will typically be a sequence of ' \
               'chemical elements followed each by a number.<br/>Isotopes are written ' \
               'with the mass number N in square brackets. ' \
               '<i>D</i> is also accepted for H[2].' \
               '<br/>Repreating groups can be provided by a sub-formula in round brackets ' \
               'followed by a number of repetitions. ' \
               '<br/>For organic molecules that exchange hydrogen atoms with their environment, ' \
               'such exchangable hydrogens are written separately as <i>Hx</i> element.' \
               '<br/>In searches the match is exect (besides re-ordering the elements), ' \
               'a rough search can be indicated by ' \
               'adding a tilde (~) before the formula.<br/>Examples:<br/>' \
               'Fe2O3 / H[2]2O / H2(C2H4)4 / ~F' \
               '</div></div>'


class ValidatedString(CType):
    regex = None
    placeholder = ''

    def __init__(self):
        CType.__init__(self, str, str)

    def convert(self, data):
        if re.match(self.regex, data) is not None:
            return CType.convert(self, data)
        else:
            raise ValueError("Not a valid %s: %s"%(self.__class__.__name__[1:], data))

    def html_input(self, field, value):
        return f'<input type="text" name="{field}" id="compound {field}" value="{value}"' \
               ' placeholder="'+self.placeholder+'" title="'+self.placeholder+'" />'


class CUrl(ValidatedString):
    regex = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    placeholder = 'http://www.your_website.net'


# The user email gets encrypted, which does not allow verification on database level
# class CMail(ValidatedString):
#     regex = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', re.IGNORECASE)
#     placeholder = 'your.name@domain.net'
#
#     def html_input(self, field, value):
#         return f'<input type="email" name="{field}" id="compound {field}" value="{value}"' \
#                ' placeholder="'+self.placeholder+'" title="'+self.placeholder+'" />'

class CMail(CType):
    placeholder = 'your.name@domain.net'

    def __init__(self):
        CType.__init__(self, str, str)

    def html_input(self, field, value):
        return f'<input type="email" name="{field}" id="compound {field}" value="{value}"' \
               ' placeholder="'+self.placeholder+'" title="'+self.placeholder+'" />'

class Cdoi(ValidatedString):
    regex = re.compile(
            r'^https://doi.org/'  # https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    placeholder = 'https://doi.org/your/ref'

    def convert(self, data):
        # if entry is just the doi value it is replaced by the url
        if data.startswith('http'):
            return ValidatedString.convert(self, data)
        else:
            return ValidatedString.convert(self, 'https://doi.org/'+data)


class Ccas(ValidatedString):
    regex = re.compile(
            r'\b[1-9][0-9]{1,5}-\d{2}-\d\b', re.IGNORECASE)
    placeholder = 'xxxxxxx-yy-z'


class CArray(Converter):
    # convert numpy array to bytest representation and back
    sql_type = 'BLOB'

    def __init__(self, shape=None, ndim=None):
        self._shape = shape
        self._ndim = ndim

    def convert(self, data):
        adata = array(data)
        if self._shape is not None and adata.shape!=self._shape:
            raise ValueError("Array shape should be %s"%str(self._shape))
        if self._ndim is not None and data.ndim!=self._ndim:
            raise ValueError("Array dimension should be %s"%self._ndim)
        type_char = data.dtype.char.encode('ascii')
        dim = str(data.ndim).encode('ascii')
        pre_chars = type_char+dim
        if data.ndim!=1:
            # for arrays >1d store the array shape before the data
            # first store the length of the shape string and then
            # the string itself
            shapestr = str(data.shape).encode('ascii')
            shapelen = ("%06i"%len(shapestr)).encode('ascii')
            pre_chars += shapelen+shapestr
        str_data = adata.tobytes()
        return pre_chars+str_data

    def revert(self, db_data):
        if db_data is None:
            return None
        if type(db_data) is not bytes:
            raise TypeError("Array type needs binary string to revert")

        dtype = db_data[0:1]
        ndim = int(db_data[1:2].decode('ascii'))
        if ndim!=1:
            shapelen = int(db_data[2:8].decode('ascii'))
            shape = eval(db_data[8:8+shapelen].decode('ascii'))
            dstart = 8+shapelen
        else:
            dstart = 2
            shape = None
        return frombuffer(db_data[dstart:], dtype=dtype).reshape(shape)

    def revert_serializable(self, db_data):
        if db_data is None:
            return None
        else:
            data = self.revert(db_data)
            if data.dtype==complex:
                return str(data)
            else:
                return data.tolist()


class CLimited(CType):

    # a converter for numbers that checks the range
    def __init__(self, fromtype, db_type,
                 low_lim=None, up_lim=None, db_repstr=None):
        CType.__init__(self, fromtype, db_type, db_repstr=db_repstr)
        self._low_lim = low_lim
        self._up_lim = up_lim

    def validate(self, data):
        if CType.validate(self, data):
            data = self.convert(data)
            return (self._low_lim is None or self._low_lim<=data) and (
                    self._up_lim is None or self._up_lim>=data)
        else:
            return False

    def convert(self, data):
        value = CType.convert(self, data)
        if (self._low_lim is None or self._low_lim<=value) and (
                self._up_lim is None or self._up_lim>=value):
            return value
        else:
            raise ValueError("Value out of range, has to be %s<value<%s"%(
                self._low_lim, self._up_lim))

    def html_input(self, field, value):
        return f'<input type="text" name="{field}" id="compound {field}" value="{value}"' \
               f' placeholder="{self._low_lim}<value<{self._up_lim}"' \
               f' title="when searching: value|value+/-10%|range: 13.4|~13.4|13-14" />'


class CComplex(CArray):

    def __init__(self):
        CArray.__init__(self, shape=(1,), ndim=1)

    def convert(self, data):
        if type(data) is str:
            data = complex(data)
        if type(data) not in [float, complex]:
            raise TypeError("Needs to be complex number")
        return CArray.convert(self, array([data]))

    def revert(self, db_data):
        if db_data is None:
            return None
        else:
            return CArray.revert(self, db_data)[0]

    def revert_serializable(self, db_data):
        if db_data is None:
            return None
        else:
            return str(self.revert(db_data))

    def html_input(self, field, value):
        return f'<input type="text" name="{field}" id="compound {field}" value="{value}"' \
               ' placeholder="complex: (real)+(imag)j"/>'


class CSelect(CType):

    def __init__(self, options):
        self.options = options
        CType.__init__(self, str, str)

    def convert(self, data):
        value = CType.convert(self, data)
        if value not in self.options:
            raise ValueError("Value has to be in selection %s"%repr(self.options))
        return value

    def html_input(self, field, value):
        output = f'<select name="{field}" id="compound {field}">'
        output += '<option value=""></option>'
        for selection in self.options:
            if value==selection:
                output += f'<option value="{selection}" selected>{selection}</option>'
            else:
                output += f'<option value="{selection}">{selection}</option>'
        output += '</select>'
        return output


class CMultiSelect(CType):
    html_list = True

    def __init__(self, options):
        self.options = options
        CType.__init__(self, list, str)

    def convert(self, data):
        data = list(data)
        for value in data:
            if value not in self.options:
                raise ValueError("Item have to be in selection %s"%repr(self.options))
        return repr(data)

    def revert(self, db_data):
        if db_data is None:
            return []
        return eval(db_data)

    def html_input(self, field, value):
        output = f'<select name="{field}" id="compound {field}" multiple>'
        for selection in self.options:
            if selection in value:
                output += f'<option value="{selection}" selected>{selection}</option>'
            else:
                output += f'<option value="{selection}">{selection}</option>'
        output += '</select><br />'
        output += f'<div class="hide_mobile"><input type="button" id="btnReset {field}" ' \
                  f'value="clear" class="clear_field" data-field="{field}" />'
        output += ' use ctrl+click</div>'
        return output
