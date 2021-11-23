"""
Classes to represent comparisons in SQLite queries.
"""

from .material import Formula
from abc import ABC, abstractmethod


class Comparator(ABC):

    def __init__(self, value, key=None):
        self.key = key
        self.value = value

    @abstractmethod
    def query_string(self):
        """
        Return the SQL query string used when searching.
        Should contain one question mark for each argument that is used.
        """

    @abstractmethod
    def query_args(self):
        """
        Return the arguments used for the query. The number of arguments should
        be equal to the number of question marks returned by query_string.
        """


class GenericComparator(Comparator):
    """
    Comparator used for the generic case. Makes a simple comparison
    based on the type of the argument.
    """

    def query_string(self):
        if type(self.value) in (list, tuple):
            if len(self.value)==0:
                return ''
            qstr = '('
            qstr += ' AND '.join([f'{self.key} LIKE ?' for _ in self.value])
            qstr += ')'
            return qstr
        elif type(self.value) is str:
            return f'{self.key} LIKE ?'
        else:
            return f'{self.key} == ?'

    def query_args(self):
        if type(self.value) in (list, tuple):
            return [f"%%'{vi}'%%" for vi in self.value]
        elif type(self.value) is str:
            return [f'%%{self.value}%%']
        else:
            return [self.value]


class ExactString(Comparator):
    """
    Perform an exact string comparison.
    """

    def query_string(self):
        return f'{self.key} == ?'

    def query_args(self):
        return [self.value]


class FormulaComparator(Comparator):
    """
    Compare to formula string.
    """

    def query_string(self):
        if type(self.value) is str and self.value.startswith('~'):
            return f'{self.key} LIKE ?'
        else:
            return f'{self.key} == ?'

    def query_args(self):
        if type(self.value) is str and self.value.startswith('~'):
            formula = Formula(self.value[1:])
            return [f'%%{formula}%%']
        else:
            formula = Formula(self.value)
            return [str(formula)]


class FuzzyFloat(Comparator):
    """
    Compare a float value agains a range or +/-10% of given value.
    """

    def query_string(self):
        try:
            float(self.value)
        except ValueError:
            return f'({self.key} >= ? and {self.key} <= ?)'
        else:
            return f'{self.key} == ?'

    def query_args(self):
        svalue = str(self.value)
        if svalue.startswith('~'):
            value = float(svalue[1:])
            return [value*0.9, value*1.1]
        elif len(svalue.split('-'))==2:
            return list(map(float, svalue.split('-')))
        else:
            return [float(self.value)]
