"""Test for the implementation of calc_fields."""

from .physical_constants import r_e
from . import grating
from . import grating_diffuse
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from genx.core.custom_logging import iprint

def test_calc_fields_single_interface():
    qz=np.linspace(0.0001, 0.5, 500)
    t, r, k_z=grating.calc_fields(qz/2., 1.54, np.array([9e-5, 9e-5, 0], dtype=np.complex128),
                                  np.array([0.0, 0.0, 0.0]))

    t_theory=2*k_z[:, -1]/(k_z[:, 0]+k_z[:, -1])
    r_theory=(k_z[:, -1]-k_z[:, 0])/(k_z[:, 0]+k_z[:, -1])

    iprint((t_theory-t[:, 0]).max())
    iprint((r_theory-r[:, 1]).max())

    if False:
        plt.subplot(211)
        plt.plot(qz, np.abs(t[:, 0])**2)
        plt.plot(qz, np.abs(t_theory)**2)
        plt.subplot(212)
        plt.semilogy(qz, np.abs(r[:, -1])**2)
        plt.semilogy(np.abs(r_theory)**2)
        plt.show()

def test_calc_fields_single_layer():
    qz=np.linspace(0.0001, 0.5, 500)
    v=np.array([9e-5-2e-6, 5e-5-8e-6, 0], dtype=np.complex128)
    d=np.array([0.0, 100.0, 0.0])
    t, r, k_z=grating.calc_fields(qz/2., 1.54, v, d)
    r1=(k_z[:, 1]-k_z[:, 0])/(k_z[:, 1]+k_z[:, 0])
    r2=(k_z[:, 2]-k_z[:, 1])/(k_z[:, 2]+k_z[:, 1])
    t1=2*k_z[:, 1]/(k_z[:, 0]+k_z[:, 1])
    t2=2*k_z[:, 2]/(k_z[:, 1]+k_z[:, 2])

    r_theory=((r1*np.exp(1.0J*d[1]*k_z[:, 1])/(t1*t2)+r2*np.exp(-1.0J*d[1]*k_z[:, 1])/(t1*t2))/
              (r1*r2*np.exp(1.0J*d[1]*k_z[:, 1])/(t1*t2)+np.exp(-1.0J*d[1]*k_z[:, 1])/(t1*t2)))

    t_lay_theory=t2*np.exp(1.0J*d[1]*k_z[:, 1])/(r1*r2*np.exp(2*1.0J*d[1]*k_z[:, 1])+1.)
    r_lay_theory=r1*t2*np.exp(1.0J*d[1]*k_z[:, 1])/(r1*r2*np.exp(2*1.0J*d[1]*k_z[:, 1])+1.)

    iprint((t_lay_theory-t[:, -2]).max())
    iprint((r_theory-r[:, -1]).max())
    iprint((r_lay_theory-r[:, -2]).max())

    if False:
        plt.subplot(211)
        plt.plot(qz, np.abs(t[:, -2]), 'b:')
        plt.subplot(212)
        plt.plot(qz, np.abs(r[:, -1]), 'g:')
        plt.plot(qz, np.abs(r[:, -2]), 'm:')

        plt.subplot(211)
        plt.plot(qz, np.abs(t_lay_theory), 'b-.')
        plt.subplot(212)
        plt.semilogy(qz, np.abs(r_theory), 'g-.')
        plt.plot(qz, np.abs(r_lay_theory), 'm-.')
        plt.show()

def test_calc_fields_slicing_single_interface():
    lamda=1.54
    v_lay=(2*np.pi/1.54)**2*(1-(1-(1.57-0.4J)*r_e*lamda**2/2/np.pi)**2)
    v_sub=(2*np.pi/1.54)**2*(1-(1-(0.71-0.09J)*r_e*lamda**2/2/np.pi)**2)
    v_layers=np.array([v_lay, v_lay, 0.0])
    sigma=np.array([0., 0., 0.])
    z_int=np.array([100., 0., -20.0])
    # NOTE: The first element has to be the substrate => largest z-value!
    z_slice=np.linspace(150, -40, 160)
    h=0.7

    plt.subplot(211)
    kz_spec=np.arange(0.001, 0.2, 0.001)
    T_1, R_1, kz_1=grating.calc_fields(kz_spec, lamda, v_layers, np.r_[0, z_int[:-1]-z_int[1:]])
    plt.plot(2*kz_spec, np.abs(T_1[:, -2])**2, 'b:')

    v_mean, int_layer_indices=grating_diffuse.calc_layer_prop(z_slice*0+1, sigma, v_layers, z_int, z_slice)
    d=np.r_[0, z_slice[:-1]-z_slice[1:]]
    T_1, R_1, kz_1=grating.calc_fields(kz_spec, lamda, v_mean, d)
    z_bottom_index=np.argmin(np.abs(z_slice-z_int[0]))
    plt.plot(2*kz_spec, np.abs(T_1[:, z_bottom_index])**2, 'g-.')

    plt.subplot(212)
    plt.plot(z_slice, v_mean.real, 'b', z_slice, v_mean.imag, 'g')
    plt.plot(z_slice, np.ones_like(z_slice)*v_lay.real, 'k', z_slice, np.ones_like(z_slice)*v_lay.imag, 'k')
    plt.show()

def test_calc_fields_slicing_single_layer():
    lamda=1.54
    v_lay=(2*np.pi/1.54)**2*(1-(1-(1.57-0.4J)*r_e*lamda**2/2/np.pi)**2)
    v_sub=(2*np.pi/1.54)**2*(1-(1-(0.71-0.09J)*r_e*lamda**2/2/np.pi)**2)
    v_layers=np.array([v_sub, v_lay, 0.0])
    sigma=np.array([0., 0., 0.])
    z_int=np.array([100., 0., -20.0])
    z_slice=np.linspace(150, -40, 190*4+1)

    kz_spec=np.arange(0.001, 0.20, 0.001)
    T_1_lay, R_1_lay, kz_1=grating.calc_fields(kz_spec, lamda, v_layers, np.r_[0, z_int[:-1]-z_int[1:]])

    v_mean, int_layer_indices=grating_diffuse.calc_layer_prop(z_slice*0+1, sigma, v_layers, z_int, z_slice)
    d=np.r_[0, z_slice[:-1]-z_slice[1:]]
    T_1_slice, R_1_slice, kz_1=grating.calc_fields(kz_spec, lamda, v_mean, d)
    z_bottom_index=np.argmin(np.abs(z_slice-z_int[0]))

    # Difficult to get the numbers to agree with high precision due to the slicing.
    iprint((((np.abs(T_1_lay[:, -2])-np.abs(T_1_slice[:, z_bottom_index]))/np.abs(T_1_lay[:, -2])).max()))
    iprint((((np.abs(R_1_lay[:, -2])-np.abs(R_1_slice[:, z_bottom_index]))/np.abs(R_1_lay[:, -2])).max()))
    iprint((((np.abs(R_1_lay[:, -1])-np.abs(R_1_slice[:, -1]))/np.abs(R_1_lay[:, -1])).max()))

    if False:
        plt.subplot(311)
        plt.plot(2*kz_spec, np.abs(T_1_lay[:, -2]), 'b:')
        plt.subplot(312)
        plt.plot(2*kz_spec, np.abs(R_1_lay[:, -1]), 'g:')
        plt.plot(2*kz_spec, np.abs(R_1_lay[:, -2]), 'm:')

        plt.subplot(311)
        plt.plot(2*kz_spec, np.abs(T_1_slice[:, z_bottom_index]), 'b-.')
        plt.subplot(312)
        plt.semilogy(2*kz_spec, np.abs(R_1_slice[:, -1]), 'g-.')
        plt.plot(2*kz_spec, np.abs(R_1_slice[:, z_bottom_index]), 'm-.')

        plt.subplot(313)
        plt.plot(z_slice, v_mean.real, 'b', z_slice, v_mean.imag, 'g')
        plt.plot(z_slice, np.ones_like(z_slice)*v_lay.real, 'k', z_slice, np.ones_like(z_slice)*v_lay.imag, 'k')
        plt.plot(z_slice, np.ones_like(z_slice)*v_sub.real, 'k', z_slice, np.ones_like(z_slice)*v_sub.imag, 'k')
        plt.show()

def test_correlation_function_h05():
    """Test so that the correlation function for h=0.5 matches the analytical solution."""
    eta_r=1050.
    q_r=np.logspace(-5., 0., 100)
    c_func=grating_diffuse.correlation_function(q_r, 0.5, eta_r, 0.0, 1e-10)
    a=1/eta_r
    analytical_func=2.*a/(a**2+q_r**2)

    iprint((np.abs((c_func-analytical_func)/c_func).max()))

    if True:
        plt.loglog(q_r, c_func)
        plt.loglog(q_r, analytical_func)
        plt.show()

def test_marginal_distributions_bvn_int():
    """Test so that the marginal distributions of polynomial_coefficentis_bvn_int is as expected"""
    delta_z_slice_int1=np.linspace(-2.9, 2.9, 200.0)
    delta_z_slice_int2=2.9*np.ones_like(delta_z_slice_int1)
    sigma1=2.0
    sigma2=1.0
    poly=grating_diffuse.polynomial_cofficents_bvn_int(delta_z_slice_int1, delta_z_slice_int2, sigma1, sigma2)
    values=np.polyval(poly, 0.5*np.ones_like(delta_z_slice_int1))

    ref_values=0.5*(1+erf(delta_z_slice_int1/np.sqrt(2)/sigma1))

    iprint((np.abs((values-ref_values)/ref_values).max()))

    if True:
        plt.plot(delta_z_slice_int1, values)
        plt.plot(delta_z_slice_int1, ref_values)
        plt.show()

# test_calc_fields()
# test_calc_fields_single_layer()
# test_calc_fields_slicing_single_interface()
# test_calc_fields_slicing_single_layer()
test_correlation_function_h05()
# test_marginal_distributions_bvn_int()
