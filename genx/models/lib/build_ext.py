"""
File: build_ext.py
Contains functions to build compiled extensions modules for different algorithms.
Progammed by: Matt Bjorck
First version: 2014-05-16
"""
from numpy import *
from scipy.weave import ext_tools

def paratt():
    """
    Function to build c extension of different reincarnations of the Paratt's
    algorithm.
    """
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
    std::complex<double> n2amb = n[interfaces]*n[interfaces];

    PyObject* out_array = PyArray_SimpleNew(1, Ntheta, NPY_DOUBLE);
    double* r_array = (double*) ((PyArrayObject*) out_array)->data;


    for(int i = 0; i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        Qj = 2.0*k*sqrt(n[0]*n[0] - n2amb*costheta2);
        Qj1 = 2.0*k*sqrt(n[1]*n[1] - n2amb*costheta2);
        rp=(Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[0]*sigma[0]);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = 2.0*k*sqrt(n[j+1]*n[j+1] - n2amb*costheta2);
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
    std::complex<double> n2amb;

    PyObject* out_array = PyArray_SimpleNew(1, Ntheta, NPY_DOUBLE);
    double* r_array = (double*) ((PyArrayObject*) out_array)->data;

    for(int i = 0;i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        n2amb = n[interfaces*points + i];
        n2amb *= n2amb;
        k = 4*pi/lamda[i];
        Qj = k*sqrt(n[0*points + i]*n[0*points + i]  - n2amb*costheta2);
        Qj1 = k*sqrt(n[1*points + i]*n[1*points + i] - n2amb*costheta2);
        rp = (Qj - Qj1)/(Qj1 + Qj)*exp(-Qj1*Qj/2.0*sigma[0]*sigma[0]);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = k*sqrt(n[(j+1)*points + i]*n[(j+1)*points + i] - n2amb*costheta2);
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
    std::complex<double> n2amb;

    PyObject* out_array = PyArray_SimpleNew(1, Ntheta, NPY_DOUBLE);
    double* r_array = (double*) ((PyArrayObject*) out_array)->data;

    for(int i = 0;i < points; i++){
        costheta2 = cos(theta[i]*torad);
        costheta2 *= costheta2;
        n2amb = n[interfaces*points + i];
        n2amb *= n2amb;
        k = 4*pi/lamda[i];
        Qj = k*sqrt(n[0*points + i]*n[0*points + i]  - n2amb*costheta2);
        Qj1 = k*sqrt(n[1*points + i]*n[1*points + i] - n2amb*costheta2);
        rp = (Qj - Qj1)/(Qj1 + Qj);
        r = rp;
        for(j = 1; j < interfaces; j++){
            Qj = Qj1;
            Qj1 = k*sqrt(n[(j+1)*points + i]*n[(j+1)*points + i] - n2amb*costheta2);
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

def offspec():
    """
    Function to build C++ extension modules used for off-specular calculations using DWBA
    """
    mod = ext_tools.ext_module('offspec_ext')

    # ext_tools need the input variables to detemine the number of dimensions? and types.
    qx =array([0,0,0], dtype=float)
    qz = qx
    sqn = array([0,0], dtype = complex)
    sigma = sqn[:].astype(dtype = complex)
    sigmaid = sqn[:].astype(dtype = complex)
    z = sqn[:].astype(dtype = float)
    qlay = sqn[:,newaxis]*qx
    G=array([qlay, qlay, qlay, qlay], dtype=complex)
    q=array([qlay, qlay, qlay, qlay], dtype=complex)
    eta = 200.0
    eta_z = 2000.0
    h = 1.0
    q_min = qx.astype(dtype = float)
    table = qx.astype(dtype = complex)
    fn = arange(1.0,3)

    code="""
    int Layers = NG[1];
    int Points = Nqx[0];
    int nind = 0;
    int npind = 0;
    double xnew = 0;
    int lower = 0;

    std::complex<double> s(0.0, 0.0), Im(0.0, 1.0), Itemp(0.0, 0.0);
    PyObject* out_array = PyArray_SimpleNew(1, Nqx, NPY_DOUBLE);
    double* I = (double*) ((PyArrayObject*) out_array)->data;

    for(int m=0;m < Points;m++){
        Itemp = 0;
        for(int i=0;i < Layers;i++){
            for(int j=0;j < Layers;j++){
                for(int k=0;k < 4; k++){
                    for(int l=0;l < 4; l++){
                        nind=(k*Layers+i)*Points+m;
                        npind=(l*Layers+j)*Points+m;
                        for(int p=0; p < Nfn[0]; p++){
                            //added abs sigan 051201
                            xnew=fabs(qx[m]*eta/pow(p+1.0,1.0/2.0/h));
                            //xnew=qx[m]*eta/pow(p+1.0,1.0/2.0/h);
                            lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]));
                            s += 2.0*pow(q[nind]*conj(q[npind])*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z),p+1.0)/fn[p]*eta/pow(p+1.0,1.0/2.0/h)*((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*(xnew-q_min[lower])+table[lower]);

                        }

                        Itemp += (sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*G[nind]*exp(-0.5*pow(sigmaid[i]*q[nind],2))*conj(G[npind]*exp(-0.5*pow(sigmaid[j]*q[npind],2)))*exp(-0.5*(pow(q[nind]*sigma[i],2)+pow(conj(q[npind])*sigma[j],2)))/q[nind]/conj(q[npind])*s;
                        //*exp(-Im*q[nind]*z[i]-conj(Im*q[npind])*z[j])
                        s = 0;

                    }
                }
            }
        }
        I[m] = std::real(Itemp);
    }
    return_val = out_array;
    Py_XDECREF(out_array);
    """
    ext = ext_tools.ext_function('dwba_interdiff_sum',code,
                                 ['qx','G','q','eta','h','sigma','sigmaid','sqn','z','table','q_min','fn','eta_z'])
    mod.add_function(ext)

    code="""
    int Layers=NG[1];
    int Points=Nqx[0];
    int nind=0;
    int npind=0;
    double xnew=0;
    int lower=0;

    std::complex<double> s(0.0, 0.0), Im(0.0, 1.0), Itemp(0.0, 0.0);
    PyObject* out_array = PyArray_SimpleNew(1, Nqx, NPY_DOUBLE);
    double* I = (double*) ((PyArrayObject*) out_array)->data;

    for(int m=0;m < Points;m++){
        Itemp = 0;
        for(int i=0;i < Layers;i++){
            for(int j=0;j < Layers;j++){
                for(int k=0;k < 4; k++){
                    for(int l=0;l < 4; l++){
                        nind=(k*Layers+i)*Points+m;
                        npind=(l*Layers+j)*Points+m;
                        for(int p=0; p < Nfn[0]; p++){
                            xnew = fabs(qx[m]*eta/pow(p+1.0,1.0/2.0/h));
                            lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]));
                            s+=2.0*pow(q[nind]*conj(q[npind])*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z),p+1.0)/fn[p]*eta/pow(p+1.0,1.0/2.0/h)*((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*(xnew-q_min[lower])+table[lower]);

                        }

                        Itemp += (sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*G[nind]*conj(G[npind])*exp(-0.5*(pow(q[nind]*sigma[i],2.0)+pow(conj(q[npind])*sigma[j],2.0)))/q[nind]/conj(q[npind])*s;
                        //*exp(-Im*q[nind]*z[i]-conj(Im*q[npind])*z[j])
                        s = 0;

                    }
                }
            }
        }
        I[m] = std::real(Itemp);
    }

    return_val = out_array;
    Py_XDECREF(out_array);

    """
    ext = ext_tools.ext_function('dwba_sum',code,
                                 ['qx','G','q','eta','h','sigma','sqn','z','table','q_min','fn','eta_z'])
    mod.add_function(ext)

    # Code for Born approximation
    code="""
    int Layers=Nsigma[0];
    int Points=Nqx[0];
    double xnew=0;
    int lower=0;

    std::complex<double> s(0.0, 0.0), Im(0.0, 1.0), Itemp(0.0, 0.0);
    PyObject* out_array = PyArray_SimpleNew(1, Nqx, NPY_DOUBLE);
    double* I = (double*) ((PyArrayObject*) out_array)->data;

    for(int m=0;m < Points;m++){
        Itemp = 0;
        for(int i=0;i < Layers;i++){
            for(int j=0;j < Layers;j++){
                for(int p=0; p < Nfn[0]; p++){
                    xnew=qx[m]*eta/pow(p+1.0,1.0/2.0/h);
                    lower=int((xnew-q_min[0])/(q_min[1]-q_min[0]));
                    s += 2.0*pow(qz[m]*qz[m]*sigma[i]*sigma[j]*exp(-abs(z[i]-z[j])/eta_z),p+1.0)/fn[p]*eta/pow(p+1.0,1.0/2.0/h)*((table[lower+1]-table[lower])/(q_min[lower+1]-q_min[lower])*(xnew-q_min[lower])+table[lower]);

                }

                Itemp+=(sqn[i]-sqn[i+1])*conj(sqn[j]-sqn[j+1])*exp(-0.5*(pow(qz[m]*sigma[i],2.0)+pow(qz[m]*sigma[j],2.0)))/qz[m]/qz[m]*s;
                s = 0;

            }
        }
        I[m] = std::real(Itemp);
    }

    return_val = out_array;
    Py_XDECREF(out_array);
    """
    ext = ext_tools.ext_function('born_sum',code,
                                 ['qx','qz','eta','h','sigma','sqn','z','table','q_min','fn','eta_z'])
    mod.add_function(ext)


    mod.compile()

def sxrd():
    """
    Function to build C++ extension modules used for sxrd calculations
    """
    mod = ext_tools.ext_module('sxrd_ext')
    # Defs to set the types of the input arrays
    h = array([0,1,2], dtype = float64)
    k, l, x, y, z, u, oc, dinv = [h]*8
    f = (x[:,newaxis]*x).astype(complex128)
    Pt = array([c_[array([[1,0],[0,1]]), array([0,0])], c_[array([[1,0],[0,1]]), array([0,0])]], dtype = float64)

    code = '''
        double pi = 3.14159265358979311599796346854418516159057617187500;
        int ij = 0;
        int offset = 0;
        std::complex<double> im(0.0, 1.0), tmp;

        PyObject* out_array = PyArray_SimpleNew(1, &Nh[0], NPY_COMPLEX128);
        std::complex<double>* fs = (std::complex<double>*) ((PyArrayObject*) out_array)->data;

        //printf("Atoms: %d, Points: %d, Symmetries: %d\\n", Noc[0], Nh[0], NPt[0]);
        // Loop over all data points
        for(int i = 0; i < Nh[0]; i++){
            fs[i] = 0;
           // Loop over all atoms
           //printf("l = %f\\n", l[i]);
           for(int j = 0; j < Noc[0]; j++){
              ij = i  + j*Nh[0];
              //printf("   x = %f, y = %f, z = %f, u = %f, oc = %f \\n", x[j], y[j], z[j], u[j], oc[j]);
              // Loop over symmetry operations
              tmp = 0;
              for(int m = 0; m < NPt[0]; m++){
                 offset = m*6;
                 tmp += exp(2.0*pi*im*(h[i]*(
                          Pt[0 + offset]*x[j] + Pt[1 + offset]*y[j] +
                              Pt[2 + offset])+
                          k[i]*(Pt[3+offset]*x[j] + Pt[4+offset]*y[j]+
                              Pt[5 + offset]) +
                          l[i]*z[j]));
                  if(i == 0 && j == 0 && false){
                     printf("P = [%f, %f] [%f, %f]",
                     Pt[0 + offset], Pt[1 + offset],
                     Pt[3 + offset], Pt[4 + offset]);
                     printf(", t = [%f, %f]\\n", Pt[2 + offset], Pt[5+offset]);

                  } // End if statement
              } // End symmetry loop index m
              fs[i] += oc[j]*f[ij]*exp(-2.0*pow(pi*dinv[i],2.0)*u[j])*tmp;
           } // End atom loop index j
        } // End data point (h,k,l) loop

        return_val = out_array;
        Py_XDECREF(out_array);
    '''
    ext = ext_tools.ext_function('surface_lattice_sum',code,
                                 ['x', 'y', 'z', 'h', 'k', 'l', 'u', 'oc', 'f', 'Pt', 'dinv'])
    mod.add_function(ext)


    mod.compile()



if __name__ == '__main__':
    print 'Building Paratt extension module'
    paratt()
    print 'Building Offspec extension module'
    offspec()
    print 'Building SXRD extension module'
    sxrd()