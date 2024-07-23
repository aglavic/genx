# run performance tests on python, numba and cuda versions of code

from multiprocessing import cpu_count
from numpy import pi, array, complex128, linspace, nan, hstack, log2
from timeit import Timer
from genx.models import lib


lib.USE_NUMBA = False

import platform


print(platform.processor())
import numba

from genx.models.lib.paratt import ReflQ as py_ReflQ
from genx.models.lib.paratt_numba import ReflQ as nb_ReflQ
from genx.models.lib.neutron_refl import Refl as py_ReflSF
from genx.models.lib.neutron_numba import Refl as nb_ReflSF


try:
    import numba.cuda


    numba.cuda.detect()  # display the cuda device used, in case something whent wrong and simulator is on
except:
    HAS_CUDA = False
else:
    from genx.models.lib.paratt_cuda import ReflQ as nc_ReflQ
    from genx.models.lib.neutron_cuda import Refl as nc_ReflSF
    HAS_CUDA = True

LAMBDA = 4.5

# SLD with wavlength factor included
sld_Si = 2.1e-06*LAMBDA**2/2./pi
sld_Fe = 8e-6*LAMBDA**2/2./pi
sld_Fe_p = 12.9e-6*LAMBDA**2/2./pi
sld_Fe_m = 2.9e-6*LAMBDA**2/2./pi
sld_Pt = 6.22e-6*LAMBDA**2/2./pi


def pot(sld):
    return (2*pi/LAMBDA)**2*(1.-(1.-sld)**2)


###### Models for Parratt evaluation ######
SMALL_REF = [
    linspace(0.001, 0.2, 100),  # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*1+[1.], dtype=complex128),  # n
    array([3.]+[100, 50]*1+[3.]),  # d
    array([10., ]*4),  # sigma
    ]

MED_REF = [
    linspace(0.001, 0.2, 400),  # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*10+[1.], dtype=complex128),  # n
    array([3.]+[100, 50]*10+[3.]),  # d
    array([10., ]*22),  # sigma
    ]

LQ_REF = [
    linspace(0.001, 0.2, 4000),  # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*10+[1.], dtype=complex128),  # n
    array([3.]+[100, 50]*10+[3.]),  # d
    array([10., ]*22),  # sigma
    ]

LS_REF = [
    linspace(0.001, 0.2, 400),  # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*100+[1.], dtype=complex128),  # n
    array([3.]+[100, 50]*100+[3.]),  # d
    array([10., ]*202),  # sigma
    ]

LARGE_REF = [
    linspace(0.001, 0.2, 4000),  # Q
    LAMBDA,
    array([1.-sld_Si]+[1.-sld_Pt, 1.-sld_Fe_p]*100+[1.], dtype=complex128),  # n
    array([3.]+[100, 50]*100+[3.]),  # d
    array([10., ]*202),  # sigma
    ]

###### Models for Matrix evaluation ######
SMALL_SF = [
    linspace(0.001, 0.2, 100),  # Q
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*1+[0], dtype=complex128),  # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*1+[0], dtype=complex128),  # Vm
    array([3.]+[100, 50]*1+[3.]),  # d
    array([0.0]+[0.0, 45*pi/180]*1+[0]),  # Mag_ang
    array([10., ]*4),  # sigma
    ]

MED_SF = [
    linspace(0.001, 0.2, 400),  # Q
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*10+[0], dtype=complex128),  # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*10+[0], dtype=complex128),  # Vm
    array([3.]+[100, 50]*10+[3.]),  # d
    array([0.0]+[0.0, 45*pi/180]*10+[0]),  # Mag_ang
    array([10., ]*22),  # sigma
    ]

LQ_SF = [
    linspace(0.001, 0.2, 4000),  # Q
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*10+[0], dtype=complex128),  # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*10+[0], dtype=complex128),  # Vm
    array([3.]+[100, 50]*10+[3.]),  # d
    array([0.0]+[0.0, 45*pi/180]*10+[0]),  # Mag_ang
    array([10., ]*22),  # sigma
    ]

LS_SF = [
    linspace(0.001, 0.2, 400),  # Q
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*100+[0], dtype=complex128),  # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*100+[0], dtype=complex128),  # Vm
    array([3.]+[100, 50]*100+[3.]),  # d
    array([0.0]+[0.0, 45*pi/180, 0., 90*pi/180.]*50+[0]),  # Mag_ang
    array([10., ]*202),  # sigma
    ]

LARGE_SF = [
    hstack([linspace(0.001, 0.2, 400), ]*10),  # Q - too fine steps breaks python variant
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_p)]*100+[0], dtype=complex128),  # Vp
    array([pot(sld_Si)]+[pot(sld_Pt), pot(sld_Fe_m)]*100+[0], dtype=complex128),  # Vm
    array([3.]+[100, 50]*100+[3.]),  # d
    array([0.0]+[0.0, 45*pi/180, 0., 90*pi/180.]*50+[0]),  # Mag_ang
    array([10., ]*202),  # sigma
    ]


def time_call(call, init, repeat=1):
    # time a function all repeated times, goal is 1s run time
    T = Timer(call, init)
    t1 = T.timeit(repeat)
    rep = max(1, int(1./t1/5.))
    return min(T.repeat(5, rep))/rep


REF_ITMS = ['SMALL_REF', 'MED_REF', 'LQ_REF', 'LS_REF', 'LARGE_REF']
SF_ITMS = ['SMALL_SF', 'MED_SF', 'LQ_SF', 'LS_SF', 'LARGE_SF']

if hasattr(numba, 'set_num_threads'):
    print('Multithread scaling large model:')
    mthread_LS = []
    print(f'{"Cores":8s} | {"Parratt [ms]":16s} | {"[ms*cores]":12s} | '
          f'{"Matrix [ms]":16s} | {"[ms*cores]":12s}')
    for lprocs in range(int(log2(cpu_count()))+1):
        procs = 2**lprocs
        numba.set_num_threads(procs)
        t_parratt = time_call('nb_ReflQ(*LS_REF)', 'from __main__ import nb_ReflQ, LS_REF',
                              repeat=procs)

        t_matrix = time_call('nb_ReflSF(*LS_SF)', 'from __main__ import nb_ReflSF, LS_SF',
                             repeat=procs)
        print(f'{procs:8} | {t_parratt*1000:16.3f} | {t_parratt*1000*procs:12.3f} | '
              f'{t_matrix*1000:16.3f} | {t_matrix*1000*procs:12.3f}')
        mthread_LS.append((procs, t_parratt, t_matrix))

if hasattr(numba, 'set_num_threads'):
    print('Multithread scaling large data:')
    mthread_LQ = []
    print(f'{"Cores":8s} | {"Parratt [ms]":16s} | {"[ms*cores]":12s} | '
          f'{"Matrix [ms]":16s} | {"[ms*cores]":12s}')
    for lprocs in range(int(log2(cpu_count()))+1):
        procs = 2**lprocs
        numba.set_num_threads(procs)
        t_parratt = time_call('nb_ReflQ(*LQ_REF)', 'from __main__ import nb_ReflQ, LQ_REF',
                              repeat=procs)

        t_matrix = time_call('nb_ReflSF(*LQ_SF)', 'from __main__ import nb_ReflSF, LQ_SF',
                             repeat=procs)
        print(f'{procs:8} | {t_parratt*1000:16.3f} | {t_parratt*1000*procs:12.3f} | '
              f'{t_matrix*1000:16.3f} | {t_matrix*1000*procs:12.3f}')
        mthread_LQ.append((procs, t_parratt, t_matrix))

ref_results = []
print('Parratt algorithm:')
print('%16s | %16s | %16s | %10s | %16s | %12s | %16s | %12s'%('Points/Layers', 'Numpy [ms]',
                                                               'Numba (1x) [ms]', 'Speedup',
                                                               'Numba (%ix) [ms]'%cpu_count(), 'Speedup (x1)',
                                                               'Cuda [ms]', 'Speedup (x1)'))
for name in REF_ITMS:
    mod = eval(name)
    mlabel = '%i/%i'%(mod[0].shape[0], mod[2].shape[0])

    t_py = time_call('py_ReflQ(*%s)'%name, 'from __main__ import py_ReflQ, %s'%name)

    try:
        numba.set_num_threads(1)
    except AttributeError:
        t_nb1 = nan
    else:
        t_nb1 = time_call('nb_ReflQ(*%s)'%name, 'from __main__ import nb_ReflQ, %s'%name)
        numba.set_num_threads(cpu_count())
    t_nb = time_call('nb_ReflQ(*%s)'%name, 'from __main__ import nb_ReflQ, %s'%name)

    if HAS_CUDA:
        t_nc = time_call('nc_ReflQ(*%s)'%name, 'from __main__ import nc_ReflQ, %s'%name)
    else:
        t_nc = nan

    ref_results.append([t_py, t_nb1, t_nb, t_nc])
    print(f'{mlabel:16s} | {t_py*1000:16.3f} | {t_nb1*1000:16.3f} | {t_py/t_nb1:10.1f} | '
          f'{t_nb*1000:16.3f} | {t_nb1/t_nb:12.1f} | {t_nc*1000:16.3f} | {t_nb1/t_nc:12.1f}')

sf_results = []
print('Matrix algorithm:')
print('%16s | %16s | %16s | %10s | %16s | %12s | %16s | %12s'%('Points/Layers', 'Numpy [ms]',
                                                               'Numba (1x) [ms]', 'Speedup',
                                                               'Numba (%ix) [ms]'%cpu_count(), 'Speedup (x1)',
                                                               'Cuda [ms]', 'Speedup (x1)'))
for name in SF_ITMS:
    mod = eval(name)
    mlabel = '%i/%i'%(mod[0].shape[0], mod[2].shape[0])

    t_py = time_call('py_ReflSF(*%s)'%name, 'from __main__ import py_ReflSF, %s'%name)

    try:
        numba.set_num_threads(1)
    except AttributeError:
        t_nb1 = nan
    else:
        t_nb1 = time_call('nb_ReflSF(*%s)'%name, 'from __main__ import nb_ReflSF, %s'%name)
        numba.set_num_threads(cpu_count())
    t_nb = time_call('nb_ReflSF(*%s)'%name, 'from __main__ import nb_ReflSF, %s'%name)

    if HAS_CUDA:
        t_nc = time_call('nc_ReflSF(*%s)'%name, 'from __main__ import nc_ReflSF, %s'%name)
    else:
        t_nc = nan

    sf_results.append([t_py, t_nb1, t_nb, t_nc])
    print(f'{mlabel:16s} | {t_py*1000:16.3f} | {t_nb1*1000:16.3f} | {t_py/t_nb1:10.1f} | '
          f'{t_nb*1000:16.3f} | {t_nb1/t_nb:12.1f} | {t_nc*1000:16.3f} | {t_nb1/t_nc:12.1f}')

print("\n\nraw data:")
for (procs, t_pLS, t_mLS), (_, t_pLQ, t_mLQ) in zip(mthread_LS, mthread_LQ):
    print(f'Cores {procs}\t{t_pLS*1000}\t{t_pLQ*1000}\t'
          f'{t_mLS*1000}\t{t_mLQ*1000}')
for i, name in enumerate(REF_ITMS):
    mod = eval(name)
    nrep = mod[0]
    mlabel = '%i/%i'%(mod[0].shape[0], mod[2].shape[0])

    t_py, t_nb1, t_nb, t_nc = ref_results[i]
    print('Parratt %s\t%g\t%g\t%g\t%g\t%g\t%g\t%g'%(
        mlabel, t_py*1000,
        t_nb1*1000, t_py/t_nb1, t_nb*1000, t_nb1/t_nb,
        t_nc*1000, t_nb1/t_nc))
for i, name in enumerate(SF_ITMS):
    mod = eval(name)
    nrep = mod[0]
    mlabel = '%i/%i'%(mod[0].shape[0], mod[2].shape[0])

    t_py, t_nb1, t_nb, t_nc = sf_results[i]
    print('Matrix %s\t%g\t%g\t%g\t%g\t%g\t%g\t%g'%(
        mlabel, t_py*1000,
        t_nb1*1000, t_py/t_nb1, t_nb*1000, t_nb1/t_nb,
        t_nc*1000, t_nb1/t_nc))
