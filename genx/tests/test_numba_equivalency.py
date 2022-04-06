"""
Test that all functions implemented in Numba yield same results as pure python functions.
"""
import unittest
import numpy as np

from genx.models import lib, sxrd


lib.USE_NUMBA = False
from genx.models.lib import instrument, instrument_numba, paratt, paratt_numba, neutron_refl, neutron_numba, \
    surface_scattering


try:
    from genx.models.lib import paratt_cuda, neutron_cuda
except:
    CUDA = False
else:
    CUDA = True


class TestInstrumentModule(unittest.TestCase):

    # Test the models.core.instruments functions implemented in models.core.instruments_numba

    def test_gauss_symmetric(self):
        x = np.linspace(0., 5., 1000)
        s12 = 10.0  # sample length/2.
        sigma = 0.5  # beam width sigma
        G1 = instrument.GaussIntensity(x, s12, s12, sigma)
        G2 = instrument_numba.GaussIntensity(x, s12, s12, sigma)
        np.testing.assert_array_almost_equal(G1, G2)

    def test_gauss_nonsymmetric(self):
        x = np.linspace(0., 5., 1000)
        s1 = 10.0  # sample length left
        s2 = 5.0  # sample length right
        sigma = 0.5  # beam width sigma
        G1 = instrument.GaussIntensity(x, s1, s2, sigma)
        G2 = instrument_numba.GaussIntensity(x, s1, s2, sigma)
        np.testing.assert_array_almost_equal(G1, G2)

    def test_square(self):
        x = np.linspace(0., 5., 1000)
        slen = 10.0  # sample length
        beamwidth = 0.5  # beam width sigma
        S1 = instrument.SquareIntensity(x, slen, beamwidth)
        S2 = instrument_numba.SquareIntensity(x, slen, beamwidth)
        np.testing.assert_array_almost_equal(S1, S2)

    def test_tth2q_scalar(self):
        x = np.linspace(0., 0.5, 1000)
        wavelength = 4.5
        Q1 = instrument.TwoThetatoQ(wavelength, x)
        Q2 = instrument_numba.TwoThetatoQ(wavelength, x)
        np.testing.assert_array_almost_equal(Q1, Q2)

    def test_tth2q_vector(self):
        x = np.linspace(0., 0.5, 1000)
        wavelength = x*0.001+4.5
        Q1 = instrument.TwoThetatoQ(wavelength, x)
        Q2 = instrument_numba.TwoThetatoQ(wavelength, x)
        np.testing.assert_array_almost_equal(Q1, Q2)

    def test_q2th_scalar(self):
        x = np.linspace(0., 0.5, 1000)
        wavelength = 4.5
        TH1 = instrument.QtoTheta(wavelength, x)
        TH2 = instrument_numba.QtoTheta(wavelength, x)
        np.testing.assert_array_almost_equal(TH1, TH2)

    def test_q2th_vector(self):
        x = np.linspace(0., 0.5, 1000)
        wavelength = x*0.001+4.5
        TH1 = instrument.QtoTheta(wavelength, x)
        TH2 = instrument_numba.QtoTheta(wavelength, x)
        np.testing.assert_array_almost_equal(TH1, TH2)

    def test_resolutionvector_scalar(self):
        x = np.linspace(0., 0.5, 1000)
        dx = 0.01
        points = 10
        range = 3.5
        Q1, weight1 = instrument.ResolutionVector(x, dx, points, range)
        Q2, weight2 = instrument_numba.ResolutionVector(x, dx, points, range)
        np.testing.assert_array_almost_equal(Q1, Q2)
        np.testing.assert_array_almost_equal(weight1, weight2)

    def test_resolutionvector_vector(self):
        x = np.linspace(0., 0.5, 1000)
        dx = np.maximum(x*0.001, 1e-8)
        points = 10
        range = 3.5
        Q1, weight1 = instrument.ResolutionVector(x, dx, points, range)
        Q2, weight2 = instrument_numba.ResolutionVector(x, dx, points, range)
        np.testing.assert_array_almost_equal(Q1, Q2)
        np.testing.assert_array_almost_equal(weight1, weight2)


class TestParattModule(unittest.TestCase):

    # Test the models.core.paratt functions implemented in models.core.paratt_numba

    def test_refl_int_no_roughness(self):
        theta = np.linspace(0., 5., 1000, dtype=np.float64)
        lamda = 4.5
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        G1 = paratt.Refl(theta, lamda, n, d, sigma, return_int=True)
        G2 = paratt_numba.Refl(theta, lamda, n, d, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.Refl(theta, lamda, n, d, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_int_roughness(self):
        theta = np.linspace(0., 5., 1000, dtype=np.float64)
        lamda = 4.5
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float64)
        G1 = paratt.Refl(theta, lamda, n, d, sigma, return_int=True)
        G2 = paratt_numba.Refl(theta, lamda, n, d, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.Refl(theta, lamda, n, d, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_amp_no_roughness(self):
        theta = np.linspace(0., 5., 1000, dtype=np.float64)
        lamda = 4.5
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        G1 = paratt.Refl(theta, lamda, n, d, sigma, return_int=False)
        G2 = paratt_numba.Refl(theta, lamda, n, d, sigma, return_int=False)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.Refl(theta, lamda, n, d, sigma, return_int=False)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_amp_roughness(self):
        theta = np.linspace(0., 5., 1000, dtype=np.float64)
        lamda = 4.5
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float64)
        G1 = paratt.Refl(theta, lamda, n, d, sigma, return_int=False)
        G2 = paratt_numba.Refl(theta, lamda, n, d, sigma, return_int=False)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.Refl(theta, lamda, n, d, sigma, return_int=False)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_reflq_int_no_roughness(self):
        Q = np.linspace(0., 0.5, 1000, dtype=np.float64)
        lamda = 4.5
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        G1 = paratt.ReflQ(Q, lamda, n, d, sigma, return_int=True)
        G2 = paratt_numba.ReflQ(Q, lamda, n, d, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.ReflQ(Q, lamda, n, d, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_reflq_int_roughness(self):
        Q = np.linspace(0., 0.5, 1000, dtype=np.float64)
        lamda = 4.5
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float64)
        G1 = paratt.ReflQ(Q, lamda, n, d, sigma, return_int=True)
        G2 = paratt_numba.ReflQ(Q, lamda, n, d, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.ReflQ(Q, lamda, n, d, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_reflq_amp_no_roughness(self):
        Q = np.linspace(0., 0.5, 1000, dtype=np.float64)
        lamda = 4.5
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)
        G1 = paratt.ReflQ(Q, lamda, n, d, sigma, return_int=False)
        G2 = paratt_numba.ReflQ(Q, lamda, n, d, sigma, return_int=False)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.ReflQ(Q, lamda, n, d, sigma, return_int=False)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_reflq_amp_roughness(self):
        Q = np.linspace(0., 0.5, 1000, dtype=np.float64)
        lamda = 4.5
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float64)
        G1 = paratt.ReflQ(Q, lamda, n, d, sigma, return_int=False)
        G2 = paratt_numba.ReflQ(Q, lamda, n, d, sigma, return_int=False)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.ReflQ(Q, lamda, n, d, sigma, return_int=False)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_nvary_int_roughness(self):
        theta = np.linspace(0., 5., 1000, dtype=np.float64)
        lamda = np.linspace(4., 5.0, 1000, dtype=np.float64)
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        n = n[:, np.newaxis]*lamda[np.newaxis, :]/4.0
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float64)
        G1 = paratt.Refl_nvary2(theta, lamda, n, d, sigma, return_int=True)
        G2 = paratt_numba.Refl_nvary2(theta, lamda, n, d, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.Refl_nvary2(theta, lamda, n, d, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_nvary_amp_roughness(self):
        theta = np.linspace(0., 5., 1000, dtype=np.float64)
        lamda = np.linspace(4., 5.0, 1000, dtype=np.float64)
        n = np.array([1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j,
                      1-7.57e-6+1.73e-7j, 1-2.24e-5+2.89e-6j, 1-7.57e-6+1.73e-7j, 1], dtype=np.complex128)
        n = n[:, np.newaxis]*lamda[np.newaxis, :]/4.0
        d = np.array([2, 80, 20, 80, 20, 80, 20, 2], dtype=np.float64)
        sigma = np.array([10, 10, 10, 10, 10, 10, 10, 10], dtype=np.float64)
        G1 = paratt.Refl_nvary2(theta, lamda, n, d, sigma, return_int=False)
        G2 = paratt_numba.Refl_nvary2(theta, lamda, n, d, sigma, return_int=False)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = paratt_cuda.Refl_nvary2(theta, lamda, n, d, sigma, return_int=False)
            np.testing.assert_array_almost_equal(G1, G2)


class TestNeutronModule(unittest.TestCase):

    # Test the models.core.neutron_refl functions implemented in models.core.neutron_numba

    def test_refl_roughness(self):
        Q = np.linspace(0.001, 0.5, 1000, dtype=np.float64)
        # sld_Fe=8e-6
        sld_Fe_p = 12.9e-6
        sld_Fe_m = 2.9e-6
        sld_Pt = 6.22e-6

        Vp = np.array([sld_Pt/np.pi, sld_Fe_p/np.pi, sld_Pt/np.pi, sld_Fe_p/np.pi, 0], dtype=np.complex128)
        Vm = np.array([sld_Pt/np.pi, sld_Fe_m/np.pi, sld_Pt/np.pi, sld_Fe_m/np.pi, 0], dtype=np.complex128)
        d = np.array([3, 100, 50, 100, 3], dtype=np.float64)
        M_ang = np.array([0.0, 45*np.pi/180, 0.0, 90*np.pi/180, 0.0, ], dtype=np.float64)
        sigma = np.array([10., 10., 10., 10., 10.0], dtype=np.float64)
        G1 = neutron_refl.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        G2 = neutron_numba.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = neutron_cuda.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_no_roughness(self):
        Q = np.linspace(0.001, 0.5, 1000, dtype=np.float64)
        # sld_Fe=8e-6
        sld_Fe_p = 12.9e-6
        sld_Fe_m = 2.9e-6
        sld_Pt = 6.22e-6

        Vp = np.array([sld_Pt/np.pi, sld_Fe_p/np.pi, sld_Pt/np.pi, sld_Fe_p/np.pi, 0], dtype=np.complex128)
        Vm = np.array([sld_Pt/np.pi, sld_Fe_m/np.pi, sld_Pt/np.pi, sld_Fe_m/np.pi, 0], dtype=np.complex128)
        d = np.array([3, 100, 50, 100, 3], dtype=np.float64)
        M_ang = np.array([0.0, 45*np.pi/180, 0.0, 90*np.pi/180, 0.0, ], dtype=np.float64)
        sigma = None
        G1 = neutron_refl.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        G2 = neutron_numba.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = neutron_cuda.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_nonmag(self):
        # test ambient layer non-air
        Q = np.linspace(0.001, 0.5, 1000, dtype=np.float64)
        SLDs = np.array([6.36e+00, (4.66e+00-1.60e-02j), 2.07e+00], dtype=np.complex128)*1e-6
        n = 1.0-SLDs
        n_prime = 1.0-(SLDs-SLDs[-1])  # use corrected SLD for matrix method

        Vp = ((2.0*np.pi/4.5)**2*(1-n_prime**2)).astype(np.complex128)
        Vm = Vp
        d = np.array([0, 100, 0], dtype=np.float64)
        M_ang = np.array([0.0, 0.0, 0.0, ], dtype=np.float64)
        sigma = np.array([10., 3., 0.], dtype=np.float64)

        G0 = paratt.ReflQ(Q, 4.5, n, d, sigma, return_int=True)
        G1 = neutron_refl.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        G2 = neutron_numba.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G0, G1[0], decimal=5)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = neutron_cuda.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_nonmag_no_roughness(self):
        # test ambient layer non-air
        Q = np.linspace(0.001, 0.5, 1000, dtype=np.float64)
        SLDs = np.array([6.36e+00, (4.66e+00-1.60e-02j), 2.07e+00], dtype=np.complex128)*1e-6
        n = 1.0-SLDs
        n_prime = 1.0-(SLDs-SLDs[-1])  # use corrected SLD for matrix method

        Vp = ((2.0*np.pi/4.5)**2*(1-n_prime**2)).astype(np.complex128)
        Vm = Vp

        d = np.array([0, 100, 0], dtype=np.float64)
        M_ang = np.array([0.0, 0.0, 0.0, ], dtype=np.float64)
        sigma = None

        G0 = paratt.ReflQ(Q, 4.5, n, d, d*0., return_int=True)
        G1 = neutron_refl.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        G2 = neutron_numba.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G0, G1[0], decimal=4)
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = neutron_cuda.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_air_nonmag(self):
        # test ambient layer non-air
        Q = np.linspace(0.001, 0.5, 1000, dtype=np.float64)
        SLDs = np.array([6.36e+00, (4.66e+00-1.60e-02j), 0e+00], dtype=np.complex128)*1e-6
        n = 1.0-SLDs

        Vp = ((2.0*np.pi/4.5)**2*(1-n**2)).astype(np.complex128)
        Vm = Vp
        d = np.array([0, 100, 0], dtype=np.float64)
        M_ang = np.array([0.0, 0.0, 0.0, ], dtype=np.float64)
        sigma = np.array([10., 3., 0.], dtype=np.float64)

        G0 = paratt.ReflQ(Q, 4.5, n, d, sigma, return_int=True)
        G1 = neutron_refl.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        G2 = neutron_numba.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G0, G1[0])
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = neutron_cuda.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)

    def test_refl_nonmag_air_no_roughness(self):
        # test ambient layer non-air
        Q = np.linspace(0.001, 0.5, 1000, dtype=np.float64)
        SLDs = np.array([6.36e+00, (4.66e+00-1.60e-02j), 0e+00], dtype=np.complex128)*1e-6
        n = 1.0-SLDs

        Vp = ((2.0*np.pi/4.5)**2*(2*SLDs-SLDs**2)).astype(
            np.complex128)  # ((2.0*np.pi/4.5)**2*(1-n**2)).astype(np.complex128)
        Vm = Vp

        d = np.array([0, 100, 0], dtype=np.float64)
        M_ang = np.array([0.0, 0.0, 0.0, ], dtype=np.float64)
        sigma = None

        G0 = paratt.ReflQ(Q, 4.5, n, d, d*0., return_int=True)
        G1 = neutron_refl.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        G2 = neutron_numba.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
        np.testing.assert_array_almost_equal(G0, G1[0])  # Matrix not so accurate without magnetization
        np.testing.assert_array_almost_equal(G1, G2)
        if CUDA:
            G2 = neutron_cuda.Refl(Q, Vp, Vm, d, M_ang, sigma, return_int=True)
            np.testing.assert_array_almost_equal(G1, G2)


class TestSXRD(unittest.TestCase):
    # test the models.lib.surface_scattering fast complex sum implementation

    def setUp(self):
        sxrd.surface_lattice_sum = surface_scattering.surface_lattice_sum

    def test_calc_f(self):
        unitcell = sxrd.UnitCell(3.9045, 3.9045, 3.9045, 90, 90, 90)

        inst = sxrd.Instrument(wavel=1.0, alpha=1.0)

        bulk = sxrd.Slab()
        bulk.add_atom('Sr', 'sr2p', 0.0, 0.0, 0.0, 0.08, 1.0)
        bulk.add_atom('Ti', 'ti4p', 0.5, 0.5, 0.5, 0.08, 1.0)
        bulk.add_atom('O1', 'o2m', 0.5, 0.5, 0.0, 0.08, 1.0)
        bulk.add_atom('O2', 'o2m', 0.5, 0.0, 0.5, 0.08, 1.0)
        bulk.add_atom('O3', 'o2m', 0.0, 0.5, 0.5, 0.08, 1.0)

        stouc = sxrd.Slab()
        stouc.add_atom('Sr', 'sr2p', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
        stouc.add_atom('Ti', 'ti4p', 0.5, 0.5, 0.5, 0.08, 1.0, 1)
        stouc.add_atom('O1', 'o2m', 0.5, 0.5, 0.0, 0.08, 1.0, 1)
        stouc.add_atom('O2', 'o2m', 0.0, 0.5, 0.5, 0.08, 1.0, 2)

        laouc = sxrd.Slab(c=1.05)
        laouc.add_atom('La', 'la3p', 0.0, 0.0, 0.0, 0.08, 1.0, 1)
        laouc.add_atom('Al', 'al3p', 0.5, 0.5, 0.5, 0.08, 1.0, 1)
        laouc.add_atom('O1', 'o2m', 0.5, 0.5, 0.0, 0.08, 1.0, 1)
        laouc.add_atom('O2', 'o2m', 0.5, 0.0, 0.5, 0.08, 1.0, 2)

        lao_interface = laouc.copy()
        lao_surface = laouc.copy()
        p4 = [sxrd.SymTrans([[1, 0], [0, 1]]), sxrd.SymTrans([[-1, 0], [0, -1]]),
              sxrd.SymTrans([[0, -1], [1, 0]]), sxrd.SymTrans([[0, 1], [-1, 0]]),
              sxrd.SymTrans([[-1, 0], [0, 1]]), sxrd.SymTrans([[1, 0], [0, -1]]),
              sxrd.SymTrans([[0, 1], [1, 0]]), sxrd.SymTrans([[0, -1], [-1, 0]])]
        sample = sxrd.Sample(inst, bulk, [stouc]+[lao_interface]+[laouc]*3+[lao_surface], unitcell)
        sample.set_surface_sym(p4)

        l = np.linspace(0., 5., 1000)
        o = l*0

        for h in range(5):
            for k in range(5):
                with self.subTest(f'h={h}, k={k}'):
                    res = sample.calc_f(o+h, o+k, l)
                    res_nb = sample.turbo_calc_f(o+h, o+k, l)
                    np.testing.assert_array_almost_equal(res, res_nb)


if __name__=='__main__':
    unittest.main()
