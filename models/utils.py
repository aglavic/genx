from numpy import *

class UserVars:
    def __init__(self):
        pass

    def newVar(self,name,value):
        #name=name.lower()
        setattr(self,name,value)
        setattr(self,'set'+name[0].upper()+name[1:],lambda value:setattr(self,name,value))
        setattr(self,'get'+name[0].upper()+name[1:],lambda :getattr(self,name,value))


if __name__=='__main__':
    MyVars=UserVars()
    MyVars.newVar('a',3)
