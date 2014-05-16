"""
File: build_ext.py
Contains functions to build compiled extensions modules for different algorithms.
Progammed by: Matt Bjorck
First version: 2014-05-16
"""
from numpy import *
from scipy.weave import ext_tools

def paratt():
    mod = ext_tools.ext_module('paratt_ext')

    # Inline code for Refl function
    code='''
    int points=Ntheta[0];
    int interfaces=Nsigma[0]-1;
    int j=0;
    double k = 2*3.141592/lamda;
    double torad = 3.141592/180.0;
    double costheta2;
    std::complex<double> Qj, Qj1, r, rp, p, imag(0.0,1.0);

    PyObject* out_array = PyArray_SimpleNew(1, Ntheta, NPY_DOUBLE);
    double* r_array = (double*) ((PyArrayObject*) out_array)->data;


    for(int i = 0; i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        Qj = 2.0*k*sqrt(n[0]*n[0] - costheta2);
        Qj1 = 2.0*k*sqrt(n[1]*n[1] - costheta2);
        rp=(Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[0]*sigma[0]);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = 2.0*k*sqrt(n[j+1]*n[j+1] - costheta2);
            rp = (Qj-Qj1)/(Qj1+Qj)*exp(-Qj1*Qj/2.0*sigma[j]*sigma[j]);
            p = exp(imag*d[j]*Qj);
            r = (rp+r*p)/(1.0 + rp*r*p);
        }

        r_array[i]=std::real(r*conj(r));
    }

    return_val = out_array;
    Py_XDECREF(out_array);

    '''
    lamda = 1.54
    theta=array([0,1,2,3],dtype=float64)
    n = array([0,0,0]).astype(complex128)
    d = array([0,0,0]).astype(float64)
    sigma = d.astype(float64)

    refl = ext_tools.ext_function('refl',code,['theta','lamda','n','d','sigma',])
    mod.add_function(refl)

    # Generating module for reflectivity with q input.
    lamda = 1.54
    Q=array([0,1,2,3],dtype=float64)
    n = array([0,0,0]).astype(complex128)
    d = array([0,0,0]).astype(float64)
    sigma = d.astype(float64)

    code='''
    int points=NQ[0];
    int interfaces=Nsigma[0]-1;
    double Q02 = 16*3.141592*3.141592/lamda/lamda;
    std::complex<double> Q2, Qj, Qj1, r, rp, p, imag(0.0,1.0);
    std::complex<double> n2amb = n[interfaces]*n[interfaces];

    PyObject* out_array = PyArray_SimpleNew(1, NQ, NPY_DOUBLE);
    double* r_array = (double*) ((PyArrayObject*) out_array)->data;

    int j=0;
    for(int i = 0; i < points; i++){
        Q2 = Q[i]*Q[i];
        Qj = sqrt((n[0]*n[0] - n2amb)*Q02 + n2amb*Q2);
        Qj1 = sqrt((n[1]*n[1] - n2amb)*Q02 + n2amb*Q2);
        rp=(Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[0]*sigma[0]);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = sqrt((n[j+1]*n[j+1] - n2amb)*Q02 + n2amb*Q2);
            rp = (Qj-Qj1)/(Qj1+Qj)*exp(-Qj1*Qj/2.0*sigma[j]*sigma[j]);
            p = exp(imag*d[j]*Qj);
            r = (rp+r*p)/(1.0 + rp*r*p);
        }

        r_array[i] = std::real(r*conj(r));
    }

    return_val = out_array;
    Py_XDECREF(out_array);

    '''
    reflq = ext_tools.ext_function('reflq',code,['Q','lamda','n','d','sigma',])
    mod.add_function(reflq)

    # Code for generating Refl_nvary2
    theta=array([0,1,2,3],dtype=float64)
    lamda = 1.54*ones(theta.shape)
    n = array([0,0,0]).astype(complex128)[:, newaxis]*ones(theta.shape)
    d = array([0,0,0]).astype(float64)
    sigma = d.astype(float64)

    code='''
    int points=Ntheta[0];
    int layers = Nn[0];
    int interfaces=Nsigma[0]-1;
    int j, m;
    double costheta2, k;
    double pi = 3.141592;
    double torad = pi/180.0;
    std::complex<double> Qj, Qj1, r, rp, p, imag(0.0, 1.0);

    PyObject* out_array = PyArray_SimpleNew(1, Ntheta, NPY_DOUBLE);
    double* r_array = (double*) ((PyArrayObject*) out_array)->data;

    for(int i = 0;i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        k = 4*pi/lamda[i];
        Qj = k*sqrt(n[0*points + i]*n[0*points + i]  - costheta2);
        Qj1 = k*sqrt(n[1*points + i]*n[1*points + i] - costheta2);
        rp = (Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[0]*sigma[0]);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = k*sqrt(n[(j+1)*points + i]*n[(j+1)*points + i] - costheta2);
            rp = (Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[j]*sigma[j]);
            p = exp(imag*d[j]*Qj);
            r = (rp+r*p)/(1.0+rp*r*p);
        }

       r_array[i] = real(r*conj(r));
    }

    return_val = out_array;
    Py_XDECREF(out_array);

    '''
    refl_nvary2 = ext_tools.ext_function('refl_nvary2',code,['theta','lamda','n','d','sigma',])
    mod.add_function(refl_nvary2)

    #Code for generating refl_nvary2_nosigma
    code='''
    int points=Ntheta[0];
    int layers = Nn[0];
    int interfaces=Nd[0]-1;
    int j, m;
    double costheta2, k;
    double pi = 3.141592;
    double torad = pi/180.0;
    std::complex<double> Qj, Qj1, r, rp, p, imag(0.0, 1.0);

    PyObject* out_array = PyArray_SimpleNew(1, Ntheta, NPY_DOUBLE);
    double* r_array = (double*) ((PyArrayObject*) out_array)->data;

    for(int i = 0;i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        k = 4*pi/lamda[i];
        Qj = k*sqrt(n[0*points + i]*n[0*points + i]  - costheta2);
        Qj1 = k*sqrt(n[1*points + i]*n[1*points + i] - costheta2);
        rp = (Qj - Qj1)/(Qj1 + Qj);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = k*sqrt(n[(j+1)*points + i]*n[(j+1)*points + i] - costheta2);
            rp = (Qj - Qj1)/(Qj1 + Qj);
            p = exp(imag*d[j]*Qj);
            r = (rp+r*p)/(1.0+rp*r*p);
        }

       r_array[i] = real(r*conj(r));
    }

    return_val = out_array;
    Py_XDECREF(out_array);

    '''
    refl_nvary2_nosigma = ext_tools.ext_function('refl_nvary2_nosigma',code,['theta','lamda','n','d'])
    mod.add_function(refl_nvary2_nosigma)


    mod.compile()