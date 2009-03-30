'''<h1> Figure of merits (fom)</h1>
Figure of merits is described here. This is the functions
that compares how well the simulation matches the measured
data. Strictly speaking for gaussian errors a Chi2 fom
is the most appropriate. However, the world is not
perfect and many times the data can be fitted easier
and more robust if another fom is chosen. The following is
a brief explanation of the foms included so far.
<h2>Fom functions</h2>
In the following the merged data set consisting of all data sets
that are marked use is written as <var>Y</var> and the equivalent
simulation is denoted as <var>S</var>. A single element of these arrays
is marked with the subscript <var>i</var>. In the same manner the
in-dependent variable (denoted as x in the data strucure) is called
<var>X</var>. The error array is denoted <var>S</var>. Finally the number
of data points is <var>N</var>

<h3>log</h3>
Absolute logarithmic (base 10) difference<br>
<HUGE>FOM<sub>log</sub> = 1/(N-1) &#8721;<sub><var>i</var></sub>
&#124;log<sub>10</sub>(<var>Y<sub>i</sub></var>) -
log<sub>10</sub>(<var>S<sub>i</sub></var>)&#124;<br></HUGE>

<h3>diff</h3>
Absolute difference<br>
<huge>FOM<sub>diff</sub> = 1/(N-1) &#8721;<sub><var>i</var></sub>
&#124;<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>&#124;<br></huge>

<h3>sqrt</h3>
Absolute squared difference<br>
<huge>FOM<sub>sqrt</sub> = 1/(N-1) &#8721;<sub><var>i</var></sub>
&#124;sqrt(<var>Y<sub>i</sub></var>) - sqrt(<var>S<sub>i</sub></var>)
&#124;<br></huge>
<h3>R1</h3>
Crystallographic R1 factor, assumes that the loaded data are intensities.
<br>
<huge>FOM<sub>R1</sub> =
&#8721;<sub><var>i</var></sub>
&#124;sqrt(<var>Y<sub>i</sub></var>) - sqrt(<var>S<sub>i</sub></var>)
&#124;/&#8721;<sub><var>i</var></sub>sqrt(<var>Y<sub>i</sub></var>)
<br></huge>
<h3>R2</h3>
Crystallographic R2 factor, assumes that the loaded data are intensities.
<br>
<huge>FOM<sub>R2</sub> =
&#8721;<sub><var>i</var></sub>
(<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>
)<sup>2</sup>/&#8721;<sub><var>i</var></sub><var>Y<sub>i</sub><sup>2</sup></var>
<br></huge>
<h2>Customisation</h2>

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
