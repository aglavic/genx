'''<h1> Figure of Merit (FOM)</h1>
The Figure of Merit (FOM) is the function that compares how well the simulation matches the measured data. Strictly speaking, for Gaussian errors, a chi squared (&chi;<sup>2</sup>) FOM is the most appropriate. However, the world is not perfect and many times the data can be fitted more easily and more robustly if another FOM is chosen. Each FOM function has its merits and drawbacks, and fitting can rely critically on choosing the right FOM function for the particular data to be analyzed. The following gives a brief summary and explanation of the FOMs included in the standard GenX distribution so far.<br>
It is also possible to create custom FOM functions to be used by GenX. For more information on this refer to the Section "Customization" below.<br>


<h2>Available FOM functions</h2>
In the following, the merged data set consisting of all data sets
that are marked for use is denoted as <var>Y</var> and the corresponding
simulation is denoted as <var>S</var>. A single element of these arrays
is indicated by a subscript <var>i</var>. In the same manner, the
independent variable (denoted as <var>x</var> in the data strucure) is called
<var>X</var>. The error array is denoted <var>E</var>. Finally the total number
of data points is given by <var>N</var> and <var>p</p> is the number of free parameters
in the fit.<br>


<h3>Unweighted FOM functions</h3>


<h4>diff</h4>
Average of the absolute difference between simulation and data.<br>
<br><huge>
    FOM<sub>diff</sub> =  1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    &#124;<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>&#124;
</huge><br>


<h4>log</h4>
Average of the absolute difference between the logarithms (base 10) of the data and the simulation.<br>
<br><huge>
    FOM<sub>log</sub> = 1/(N-p) &times;&#8721;<sub><var>i</var></sub>
    &#124;log<sub>10</sub>(<var>Y<sub>i</sub></var>) -
    log<sub>10</sub>(<var>S<sub>i</sub></var>)&#124;
</huge><br>


<h4>sqrt</h4>
Average of the absolute difference between the square roots of the data and the simulation:<br>
<br><huge>
    FOM<sub>sqrt</sub> =  1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    &#124;sqrt(<var>Y<sub>i</sub></var>) - sqrt(<var>S<sub>i</sub></var>)
    &#124;
</huge><br>


<h4>R1</h4>
Crystallographic R-factor (often denoted as R1, sometimes called residual factor or reliability factor or the R-value or R<sub>work</sub>).<br>
Gives the percentage of the summed structure factor residuals (absolute difference between data and simulation) over the entire data set with respect to the total sum of measured structure factors. For data sets spanning several orders of magnitude in intensity, R1 is dominated by the residuals at high intensities, while large residuals at low intensities have very little impact on R1.
This implementation here assumes that the loaded data are intensities (squares of the structure factors), hence the square roots of the loaded data are taken for the calculation of R1.<br>
[A.J.C. Wilson, Acta Crystallogr. A32, 994 (1976)]<br>
<br><huge>
  FOM<sub>R1</sub> =
  &#8721;<sub><var>i</var></sub> [ 
  &#124;sqrt(<var>Y<sub>i</sub></var>) - sqrt(<var>S<sub>i</sub></var>)
  &#124; ] / &#8721;<sub><var>i</var></sub> [ sqrt(<var>Y<sub>i</sub></var>) ]
</huge><br>


<h4>logR1</h4>
The logarithmic R1 factor is a modification of the crystallographic R-factor, calculated using the logarithm (base 10) of the structure factor and simulation. This scaling results in a more equal weighting of high-intensity and low-intensity data points which can be very helpful when fitting data which is spanning several orders of magnitude on the y-axis. Essentially it gives all data points equal weight when displayed in a log-plot.<br>
<br><huge>
    FOM<sub>logR1</sub> =
    &#8721;<sub><var>i</var></sub> [ &#124;
    log<sub>10</sub>(sqrt(<var>Y<sub>i</sub></var>)) -
    log<sub>10</sub>(sqrt(<var>S<sub>i</sub></var>))
    &#124; ] / 
    &#8721;<sub><var>i</var></sub> [
    log<sub>10</sub>(sqrt(<var>Y<sub>i</sub></var>) ]
</huge><br>


<h4>R2</h4>
Crystallographic R2 factor. In contrast to R1, this gives the ratio of the total sum of squared deviations to the total sum of squared structure factors. (Note that sometimes R2 is also defined as the square root of the value defined here.)
Like in the case for R1, this implementation assumes that the loaded data are intensities (squares of the structure factors).<br>
[A.J.C. Wilson, Acta Crystallogr. A32, 994 (1976)]<br>
<br><huge>
    FOM<sub>R2</sub> =
    &#8721;<sub><var>i</var></sub> [
    (<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>)<sup>2</sup> ] /
    &#8721;<sub><var>i</var></sub> [ <var>Y<sub>i</sub><sup>2</sup></var> ]
</huge><br>


<h4>logR2</h4>
The logarithmic R2 factor is a modification of the crystallographic R2 factor, calculated using the logarithm (base 10) of the structure factor and simulation. This scaling results in a more similar weighting of high-intensity and low-intensity data points which can be very helpful when fitting data which is spanning several orders of magnitude on the y-axis. Essentially it gives all data points equal weight when displayed in a log-plot.<br>
<br><huge>
    FOM<sub>logR2</sub> =
    &#8721;<sub><var>i</var></sub> [ 
    (log<sub>10</sub>(<var>Y<sub>i</sub></var>) -
    log<sub>10</sub>(<var>S<sub>i</sub></var>)
    )<sup>2</sup> ] /
    &#8721;<sub><var>i</var></sub> [ 
    log<sub>10</sub>(<var>Y<sub>i</sub>)<sup>2</sup></var> ]
</huge><br>


<h4>sintth4</h4>
Gives the average of the absolute differences scaled with a sin(2&theta;)<sup>4</sup> term (2&theta; = tth). For reflectivity data, this will divide away the Fresnel reflectivity. <br>
<br><huge>
    FOM<sub>sintth4</sub> = 1/(N-p) &times;
    &#8721;<sub><var>i</var></sub>
    &#124;<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>&#124; &times;
    sin(<var>tth</var>)<sup>4</sup>
</huge><br>

<h4>Norm</h4>
Gives the linear difference normalized by the absolute sum of data points<br>
<br><huge>
    FOM<sub>Norm</sub> = 1/(N-p) &times; &#8721;<sub><var>i</var></sub>
        &#124;<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>&#124;
        /  &#8721;<sub><var>j</var></sub> |&#124;<var>Y<sub>j</sub></var>|
</huge><br>

<h3>Weighted FOM functions</h3>

<h4>chi2bars</h4>
Chi squared (&chi;<sup>2</sup>) FOM including error bars<br>
<br><huge>
    FOM<sub>chi2bars</sub> = 1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    ((<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>) /
    <var>E<sub>i</sub></var>)<sup>2</sup>
</huge><br>


<h4>chibars</h4>
Chi squared but without the squaring! Includes error bars:<br>
<br><huge>
    FOM<sub>chibars</sub> = 1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    &#124;(<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>) /
    <var>E<sub>i</sub></var>&#124;
</huge><br>


<h4>logbars</h4>
Absolute logarithmic (base 10) difference, taking errors into account:<br>
<br><huge>
    FOM<sub>logbars</sub> = 1/(N-p) &times; &#8721;<sub><var>i</var></sub>
    &#124;log<sub>10</sub>(<var>Y<sub>i</sub></var>) -
    log<sub>10</sub>(<var>S<sub>i</sub></var>)&#124; /
    <var>E<sub>i</sub></var>*ln(10)*<var>Y<sub>i</sub></var>
</huge><br>


<h4>R1bars</h4>
Similar to the crystallographic R-factor R1, but with weighting of the data points by the experimental error values. The error values in E are assumed to be proportional to the standard deviation of the measured intensities.<br>
[A.J.C. Wilson, Acta Crystallogr. A32, 994 (1976), W.C. Hamilton, Acta Crystallogr. 18(3), 502 (1965)]<br>
<br><huge>
    FOM<sub>R1bars</sub> =
    &#8721;<sub><var>i</var></sub><var> [ sqrt(1/E<sub>i</sub></var>) &times;
    &#124;sqrt(<var>Y<sub>i</sub></var>) - sqrt(<var>S<sub>i</sub></var>)
    &#124; ] /
    &#8721;<sub><var>i</var></sub> [ sqrt(1/E<sub>i</sub></var>) &times;
    sqrt(<var>Y<sub>i</sub></var>) ]
</huge><br>


<h4>R2bars</h4>
Weighted R2 factor. The error values in E are assumed to be proportional to the standard deviation of the measured intensities.<br>
[A.J.C. Wilson, Acta Crystallogr. A32, 994 (1976), W.C. Hamilton, Acta Crystallogr. 18(3), 502 (1965)]<br>
<br><huge>
    FOM<sub>R2bars</sub> =
    &#8721;<sub><var>i</var></sub> [ (1/E<sub>i</sub></var>) &times;
    (<var>Y<sub>i</sub></var> - <var>S<sub>i</sub></var>)<sup>2</sup> ] /
    &#8721;<sub><var>i</var></sub> [ (1/E<sub>i</sub></var>) &times;
    <var>Y<sub>i</sub><sup>2</sup></var> ]
</huge><br>


<h2>Customization</h2>
Users can add their own cumstom-built FOM functions to be used in GenX. For detailed instructions on how to write the code for a custom FOM function and how to include it in the list of FOM functions available to GenX, see the manual at
<a href = "http://apps.sourceforge.net/trac/genx/wiki/DocPages/WriteFom">
http://apps.sourceforge.net/trac/genx/wiki/DocPages/WriteFom </a>
'''
# ==============================================================================

import numpy as np
from .core.custom_logging import iprint

# import also the custom FOM functions defined in fom_funcs_custom.py
# (do nothing if file does not exist)
try:
    # noinspection PyUnresolvedReferences
    from fom_funcs_custom import *
    iprint("Imported custom-defined FOM functions from fom_funcs_custom.py")
except:
    pass

# ==============================================================================
# BEGIN FOM function defintions
def _div_dof(func):
    # decorator to set function attribute
    func.__div_dof__=True
    return func

# =========================
# unweighted FOM functions
@_div_dof
def diff(simulations, data):
    ''' Average absolute difference
    '''
    N=np.sum([len(dataset.y)*dataset.use for dataset in data])
    # return 1.0/(N-1)*np.sum([np.sum(np.abs(dataset.y - sim))\
    #    for (dataset, sim) in zip(data,simulations) if dataset.use])
    return [(dataset.y-sim)
            for (dataset, sim) in zip(data, simulations)]

@_div_dof
def log(simulations, data):
    ''' Average absolute logartihmic difference
    '''
    N=np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [(np.log10(dataset.y)-np.log10(sim))
            for (dataset, sim) in zip(data, simulations)]

@_div_dof
def sqrt(simulations, data):
    ''' Average absolute difference of the square root
    '''
    N=np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [(np.sqrt(dataset.y)-np.sqrt(sim))
            for (dataset, sim) in zip(data, simulations)]

def R1(simulations, data):
    ''' Crystallographic R-factor (R1)
    '''
    denom=np.sum([np.sum(np.sqrt(np.abs(dataset.y))) for dataset in data \
                  if dataset.use])
    return [1.0/denom*(np.sqrt(np.abs(dataset.y))-np.sqrt(np.abs(sim))) \
            for (dataset, sim) in zip(data, simulations)]

def logR1(simulations, data):
    ''' logarithmic crystallographic R-factor (R1)
    '''
    denom=np.sum([np.sum(np.log10(np.sqrt(dataset.y))) for dataset in data \
                  if dataset.use])
    return [1.0/denom*(np.log10(np.sqrt(dataset.y))-
                       np.log10(np.sqrt(sim))) \
            for (dataset, sim) in zip(data, simulations)]

def R2(simulations, data):
    ''' Crystallographic R2 factor
    '''
    denom=np.sum([np.sum(dataset.y**2) for dataset in data \
                  if dataset.use])
    return [1.0/denom*np.sign(dataset.y-sim)*(dataset.y-sim)**2 \
            for (dataset, sim) in zip(data, simulations)]

def logR2(simulations, data):
    ''' logarithmic crystallographic R2 factor
    '''
    denom=np.sum([np.sum(np.log10(dataset.y)**2) for dataset in data \
                  if dataset.use])
    return [1.0/denom*np.sign(np.log10(dataset.y)-np.log10(sim))*(np.log10(dataset.y)-np.log10(sim))**2 \
            for (dataset, sim) in zip(data, simulations)]

@_div_dof
def sintth4(simulations, data):
    ''' Sin(tth)^4 scaling of the average absolute difference for reflectivity.
    '''
    N=np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [np.sin(dataset.x*np.pi/360.0)**4*
            (dataset.y-sim)
            for (dataset, sim) in zip(data, simulations)]

@_div_dof
def Norm(simulations, data):
    '''  linear difference normalized by absolute sum of values
    '''
    return [1.0/np.sum(np.abs(dataset.y))*(dataset.y-sim) \
            for (dataset, sim) in zip(data, simulations)]

# =======================
# weighted FOM functions

@_div_dof
def chi2bars(simulations, data):
    ''' Weighted chi squared
    '''
    N=np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [np.sign(dataset.y-sim)*(dataset.y-sim)**2/dataset.error**2
            for (dataset, sim) in zip(data, simulations)]

@_div_dof
def chibars(simulations, data):
    ''' Weighted chi squared but without the squaring
    '''
    N=np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [((dataset.y-sim)/dataset.error)
            for (dataset, sim) in zip(data, simulations)]

@_div_dof
def logbars(simulations, data):
    ''' Weighted average absolute difference of the logarithm of the data
    '''
    N=np.sum([len(dataset.y)*dataset.use for dataset in data])
    return [((np.log10(dataset.y)-np.log10(sim))
             /dataset.error*np.log(10)*dataset.y)
            for (dataset, sim) in zip(data, simulations)]

def R1bars(simulations, data):
    ''' Weighted crystallographic R-factor (R1)
    '''
    denom=np.sum([np.sum(np.sqrt(1/dataset.error)*np.sqrt(dataset.y))
                  for dataset in data if dataset.use])
    return [1.0/denom*np.sqrt(1/dataset.error)*
            (np.sqrt(dataset.y)-np.sqrt(sim))
            for (dataset, sim) in zip(data, simulations)]

def R2bars(simulations, data):
    ''' Weighted crystallographic R2 factor
    '''
    denom=np.sum([(1/dataset.error)*np.sum(dataset.y**2)
                  for dataset in data if dataset.use])
    return [1.0/denom*(1/dataset.error)*np.sign(dataset.y-sim)*(dataset.y-sim)**2
            for (dataset, sim) in zip(data, simulations)]

# END FOM function definition
# ==============================================================================


# create introspection variables so that everything updates automatically
# Find all objects in this namespace
# (this includes the custom-defined FOM functions from fom_funcs_custom.py)
obj_list=dir()[:]

# find all functions
all_func_names=[s for s in obj_list if type(eval(s)).__name__=='function']
func_names=[s for s in all_func_names if not s.startswith('_')]

# End of file
# ==============================================================================
