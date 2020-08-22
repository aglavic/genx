# run performance tests on python, numba and cuda versions of code

from multiprocessing import cpu_count
from numpy import *
from timeit import Timer
from genx.models import lib
lib.USE_NUMBA=False

import numba

from genx.models.lib.paratt import ReflQ as py_ReflQ
from genx.models.lib.paratt_numba import ReflQ as nb_ReflQ
from genx.models.lib.neutron_refl import Refl as py_ReflSF
from genx.models.lib.neutron_numba import Refl as nb_ReflSF
try:
    from genx.models.lib.neutron_cuda import Refl as nc_ReflSF
except:
    HAS_CUDA=False
else:
    HAS_CUDA=True

LAMBDA=4.5

sld_Si=2.1e-06
sld_Fe=8e-6
sld_Fe_p=12.9e-6
sld_Fe_m=2.9e-6
sld_Pt=6.22e-6
def pot(sld):
    return (2*pi/LAMBDA)**2*(1-(1-LAMBDA**2/2/pi*sld)**2)

###### Models for Parratt evaluation ######
SMALL_REF=[
    1000,
    linspace(0.001, 0.2, 100), # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*1+[1.], dtype=complex128), # n
    array([3.]+[100, 50]*1+[3.]), # d
    array([10., ]*4), # sigma
    ]

MED_REF=[
    1000,
    linspace(0.001, 0.2, 400), # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*10+[1.], dtype=complex128), # n
    array([3.]+[100, 50]*10+[3.]), # d
    array([10., ]*22), # sigma
    ]

LQ_REF=[
    50,
    linspace(0.001, 0.2, 4000), # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*10+[1.], dtype=complex128), # n
    array([3.]+[100, 50]*10+[3.]), # d
    array([10., ]*22), # sigma
    ]

LS_REF=[
    50,
    linspace(0.001, 0.2, 400), # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*100+[1.], dtype=complex128), # n
    array([3.]+[100, 50]*100+[3.]), # d
    array([10., ]*202), # sigma
    ]

LARGE_REF=[
    10,
    linspace(0.001, 0.2, 4000), # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*100+[1.], dtype=complex128), # n
    array([3.]+[100, 50]*100+[3.]), # d
    array([10., ]*202), # sigma
    ]

###### Models for Matrix evaluation ######
SMALL_SF=[
    500,
    linspace(0.001, 0.2, 100), # Q
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*1+[0], dtype=complex128), # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*1+[0], dtype=complex128), # Vm
    array([3.]+[100, 50]*1+[3.]), # d
    array([0.0]+[0.0, 45*pi/180]*1+[0]), # Mag_ang
    array([10., ]*4), # sigma
    ]

MED_SF=[
    50,
    linspace(0.001, 0.2, 400), # Q
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*10+[0], dtype=complex128), # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*10+[0], dtype=complex128), # Vm
    array([3.]+[100, 50]*10+[3.]), # d
    array([0.0]+[0.0, 45*pi/180]*10+[0]), # Mag_ang
    array([10., ]*22), # sigma
    ]

LQ_SF=[
    10,
    linspace(0.001, 0.2, 4000), # Q
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*10+[0], dtype=complex128), # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*10+[0], dtype=complex128), # Vm
    array([3.]+[100, 50]*10+[3.]), # d
    array([0.0]+[0.0, 45*pi/180]*10+[0]), # Mag_ang
    array([10., ]*22), # sigma
    ]

LS_SF=[
    10,
    linspace(0.001, 0.2, 400), # Q
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*100+[0], dtype=complex128), # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*100+[0], dtype=complex128), # Vm
    array([3.]+[100, 50]*100+[3.]), # d
    array([0.0]+[0.0, 45*pi/180, 0., 90*pi/180.]*50+[0]), # Mag_ang
    array([10., ]*202), # sigma
    ]

LARGE_SF=[
    5,
    hstack([linspace(0.001, 0.2, 400), ]*5), # Q - too fine steps breaks python variant
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*100+[0], dtype=complex128), # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*100+[0], dtype=complex128), # Vm
    array([3.]+[100, 50]*100+[3.]), # d
    array([0.0]+[0.0, 45*pi/180, 0., 90*pi/180.]*50+[0]), # Mag_ang
    array([10., ]*202), # sigma
    ]


REF_ITMS=['SMALL_REF', 'MED_REF', 'LQ_REF', 'LS_REF', 'LARGE_REF']
SF_ITMS=['SMALL_SF', 'MED_SF', 'LQ_SF', 'LS_SF', 'LARGE_SF']

ref_results=[]
print('%16s | %16s | %16s | %10s | %16s | %10s'%('Points/Layers', 'Numpy [ms]',
                                   'Numba (1x) [ms]', 'Speedup',
                                   'Numba (%ix) [ms]'%cpu_count(), 'Speedup'))
for name in REF_ITMS:
    mod=eval(name)
    nrep=mod[0]
    mlabel='%i/%i'%(mod[1].shape[0], mod[3].shape[0])

    T=Timer('py_ReflQ(*%s[1:])'%name, 'from __main__ import py_ReflQ, %s'%name)
    t_py=min(T.repeat(5, nrep))/nrep
    
    try:
        numba.set_num_threads(1)
    except AttributeError:
        t_nb1=nan
    else:
        T=Timer('nb_ReflQ(*%s[1:])'%name, 'from __main__ import nb_ReflQ, %s'%name)
        t_nb1=min(T.repeat(5, 5*nrep))/nrep/5.
        numba.set_num_threads(cpu_count())
    T=Timer('nb_ReflQ(*%s[1:])'%name, 'from __main__ import nb_ReflQ, %s'%name)
    t_nb=min(T.repeat(5, 5*nrep))/nrep/5.

    ref_results.append([t_py, t_nb1, t_nb])
    print('%16s | %16.3f | %16.3f | %10.1f | %16.3f | %10.1f'%(mlabel, t_py*1000,
                                             t_nb1*1000, t_py/t_nb1, t_nb*1000, t_py/t_nb))

sf_results=[]
print('%16s | %16s | %16s | %10s | %16s | %10s | %16s | %10s'%('Points/Layers', 'Numpy [ms]',
                                                'Numba (1x) [ms]', 'Speedup',
                                                'Numba (%ix) [ms]'%cpu_count(), 'Speedup',
                                                'Cuda [ms]', 'Speedup'))
for name in SF_ITMS:
    mod=eval(name)
    nrep=mod[0]
    mlabel='%i/%i'%(mod[1].shape[0], mod[3].shape[0])

    T=Timer('py_ReflSF(*%s[1:])'%name, 'from __main__ import py_ReflSF, %s'%name)
    t_py=min(T.repeat(5, nrep))/nrep

    try:
        numba.set_num_threads(1)
    except AttributeError:
        t_nb1=nan
    else:
        T=Timer('nb_ReflSF(*%s[1:])'%name, 'from __main__ import nb_ReflSF, %s'%name)
        t_nb1=min(T.repeat(5, 5*nrep))/nrep/5.
        numba.set_num_threads(cpu_count())
    T=Timer('nb_ReflSF(*%s[1:])'%name, 'from __main__ import nb_ReflSF, %s'%name)
    t_nb=min(T.repeat(5, 5*nrep))/nrep/5.

    if HAS_CUDA:
        T=Timer('nc_ReflSF(*%s[1:])'%name, 'from __main__ import nc_ReflSF, %s'%name)
        t_nc=min(T.repeat(5, 5*nrep))/nrep/5.
    else:
        t_nc=nan

    sf_results.append([t_py, t_nb1, t_nb, t_nc])
    print('%16s | %16.3f | %16.3f | %10.1f | %16.3f | %10.1f | %16.3f | %10.1f'%(
                    mlabel, t_py*1000,
                    t_nb1*1000, t_py/t_nb1, t_nb*1000, t_py/t_nb,
                    t_nc*1000, t_py/t_nc))

print("\n\nraw data:")
for i, name in enumerate(REF_ITMS):
    mod=eval(name)
    nrep=mod[0]
    mlabel='%i/%i'%(mod[1].shape[0], mod[3].shape[0])

    t_py, t_nb1, t_nb=ref_results[i]
    print('%s\t%g\t%g\t%g\t%g\t%g'%(
                    mlabel, t_py*1000,
                    t_nb1*1000, t_py/t_nb1, t_nb*1000, t_py/t_nb))
for i, name in enumerate(SF_ITMS):
    mod=eval(name)
    nrep=mod[0]
    mlabel='%i/%i'%(mod[1].shape[0], mod[3].shape[0])

    t_py, t_nb1, t_nb, t_nc=sf_results[i]
    print('%s\t%g\t%g\t%g\t%g\t%g\t%g\t%g'%(
                    mlabel, t_py*1000,
                    t_nb1*1000, t_py/t_nb1, t_nb*1000, t_py/t_nb,
                    t_nc*1000, t_py/t_nc))
