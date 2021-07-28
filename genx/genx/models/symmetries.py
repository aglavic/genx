"""
Package to handle symmetries in surface structures.
"""

import numpy as np

class SymTrans:
    def __init__(self, P=None, t=None):
        # TODO: Check size of arrays!
        if P is None:
            P=[[1, 0], [0, 1]]
        if t is None:
            t=[0, 0]
        self.P=np.array(P, dtype=np.float64)
        self.t=np.array(t, dtype=np.float64)

    def trans_x(self, x, y):
        '''transformed x coord'''
        # print self.P[0][0]*x + self.P[0][1]*y + self.t[0]
        return self.P[0][0]*x+self.P[0][1]*y+self.t[0]

    def trans_y(self, x, y):
        '''transformed x coord'''
        # print self.P[1][0]*x + self.P[1][1]*y + self.t[1]
        return self.P[1][0]*x+self.P[1][1]*y+self.t[1]

    def apply_symmetry(self, x, y):
        return np.dot(self.P, np.c_[x, y])+self.t

class Sym(list):
    """Class to hold a symmetry i.e. a list of SymTrans."""

    def __init__(self, *args):
        list.__init__(self, args)

# Symmetries
p1=Sym(SymTrans([[1, 0], [0, 1]]))
p2=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]))
pm=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, 1]]))
pg=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, 1]], [0, 1./2]))
cm=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, 1]]),
       SymTrans([[1, 0], [0, 1]], [1./2, 1./2]), SymTrans([[-1, 0], [0, 1]], [1./2, 1./2]))
p2mm=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]), SymTrans([[-1, 0], [0, 1]]),
         SymTrans([[1, 0], [0, -1]]))
p2mg=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]), SymTrans([[-1, 0], [0, 1]], [1./2, 0]),
         SymTrans([[1, 0], [0, -1]], [1./2, 0]))
p2gg=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]), SymTrans([[-1, 0], [0, 1]], [1./2, 1./2]),
         SymTrans([[1, 0], [0, -1]], [1./2, 1./2]))
c2mm=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]), SymTrans([[-1, 0], [0, 1]]),
         SymTrans([[1, 0], [0, -1]]),
         SymTrans([[1, 0], [0, 1]], [1./2, 1./2]), SymTrans([[-1, 0], [0, -1]], [1./2, 1./2]),
         SymTrans([[-1, 0], [0, 1]], [1./2, 1./2]), SymTrans([[1, 0], [0, -1]], [1./2, 1./2]))
p4=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]), SymTrans([[0, -1], [1, 0]]),
       SymTrans([[0, 1], [-1, 0]]))
p4mm=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]), SymTrans([[0, -1], [1, 0]]),
         SymTrans([[0, 1], [-1, 0]]), SymTrans([[-1, 0], [0, 1]]), SymTrans([[1, 0], [0, -1]]),
         SymTrans([[0, 1], [1, 0]]), SymTrans([[0, -1], [-1, 0]])
         )
p4gm=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[-1, 0], [0, -1]]), SymTrans([[0, -1], [1, 0]]),
         SymTrans([[0, 1], [-1, 0]]),
         SymTrans([[-1, 0], [0, 1]], [1./2, 1./2]), SymTrans([[1, 0], [0, -1]], [1./2, 1./2]),
         SymTrans([[0, 1], [1, 0]], [1./2, 1./2]), SymTrans([[0, -1], [-1, 0]], [1./2, 1./2])
         )
p3=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[0, -1], [1, -1]]), SymTrans([[-1, 1], [-1, 0]]))
p3m1=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[0, -1], [1, -1]]), SymTrans([[-1, 1], [-1, 0]]),
         SymTrans([[0, -1], [-1, 0]]), SymTrans([[-1, 1], [0, 1]]), SymTrans([[1, 0], [1, -1]])
         )
p31m=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[0, -1], [1, -1]]), SymTrans([[-1, 1], [-1, 0]]),
         SymTrans([[0, 1], [1, 0]]), SymTrans([[1, -1], [0, -1]]), SymTrans([[-1, 0], [-1, 1]])
         )
p6=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[0, -1], [1, -1]]), SymTrans([[-1, 1], [-1, 0]]),
       SymTrans([[-1, 0], [0, -1]]), SymTrans([[0, 1], [-1, 1]]), SymTrans([[1, -1], [1, 0]])
       )
p6mm=Sym(SymTrans([[1, 0], [0, 1]]), SymTrans([[0, -1], [1, -1]]), SymTrans([[-1, 1], [-1, 0]]),
         SymTrans([[-1, 0], [0, -1]]), SymTrans([[0, 1], [-1, 1]]), SymTrans([[1, -1], [1, 0]]),
         SymTrans([[0, -1], [-1, 0]]), SymTrans([[-1, 1], [0, 1]]), SymTrans([[1, 0], [1, -1]]),
         SymTrans([[0, 1], [1, 0]]), SymTrans([[1, -1], [0, -1]]), SymTrans([[-1, 0], [-1, 1]])
         )
