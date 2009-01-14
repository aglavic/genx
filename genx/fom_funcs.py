'''fom_funcs.py
figure of merit function definitions for fitting
Programmer: Matts Bjorck
Last changed 20080819
'''

import numpy as np

#==============================================================================
# BEGIN fom function defintions
def log(simulations, data):
    ''' The absolute logartihmic difference
    '''
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return 1.0/(N-1)*np.sum([np.sum(np.abs(np.log10(dataset.y)-np.log10(sim)))\
            for (dataset, sim) in zip(data,simulations) if dataset.use])
            
def diff(simulations, data):
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return 1.0/(N-1)*np.sum([np.sum(np.abs(dataset.y - sim))\
            for (dataset, sim) in zip(data,simulations) if dataset.use])
            
def sqrt(simulations, data):
    N = np.sum([len(dataset.y)*dataset.use for dataset in data])
    return 1.0/(N-1)*np.sum([np.sum(abs(np.sqrt(dataset.y) - np.sqrt(sim)))\
            for (dataset, sim) in zip(data,simulations) if dataset.use])
            
def R1(simulations, data):
    denom = np.sum([np.sum(np.sqrt(dataset.y)) for dataset in data if dataset.use])
    return 1.0/denom*np.sum([np.sum(abs(np.sqrt(dataset.y) - np.sqrt(sim)))\
            for (dataset, sim) in zip(data,simulations) if dataset.use])

def R2(simulations, data):
    denom = np.sum([np.sum(dataset.y**2) for dataset in data if dataset.use])
    return 1.0/denom*np.sum([np.sum((dataset.y - sim)**2)\
            for (dataset, sim) in zip(data,simulations) if dataset.use])
            
# END fom function definition
#==============================================================================
# create introspection variables so that everything updates automatically
# Find all objects in this namespace
obj_list = dir()[:]
#print obj_list, dir()
# find all functions
all_func_names = [s for s in obj_list if type(eval(s)).__name__ == 'function']
func_names = [s for s in all_func_names if all_func_names[0] != '_']
