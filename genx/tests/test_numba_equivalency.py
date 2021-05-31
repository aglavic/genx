"""
Test that all functions implemented in Numba yield same results as pure python functions.
"""
import unittest
import numpy as np

from genx.models import lib
lib.USE_NUMBA=False
from genx.models.lib import instrument, instrument_numba

class TestInstrumentModule(unittest.TestCase):
    # Test the models.lib.instruments functions implemented in models.lib.instruments_numba

    def test_gauss_symmetric(self):
        x=np.linspace(0., 5., 1000)
        s12=10.0 # sample length/2.
        sigma=0.5 # beam width sigma
        G1=instrument.GaussIntensity(x, s12, s12, sigma)
        G2=instrument_numba.GaussIntensity(x, s12, s12, sigma)
        np.testing.assert_array_almost_equal(G1, G2)

    def test_gauss_nonsymmetric(self):
        x=np.linspace(0., 5., 1000)
        s1=10.0 #  sample length left
        s2=5.0 #  sample length right
        sigma=0.5 # beam width sigma
        G1=instrument.GaussIntensity(x, s1, s2, sigma)
        G2=instrument_numba.GaussIntensity(x, s1, s2, sigma)
        np.testing.assert_array_almost_equal(G1, G2)

    def test_square(self):
        x=np.linspace(0., 5., 1000)
        slen=10.0 # sample length
        beamwidth=0.5 # beam width sigma
        S1=instrument.SquareIntensity(x, slen, beamwidth)
        S2=instrument_numba.SquareIntensity(x, slen, beamwidth)
        np.testing.assert_array_almost_equal(S1, S2)

    def test_tth2q_scalar(self):
        x=np.linspace(0., 0.5, 1000)
        wavelength=4.5
        Q1=instrument.TwoThetatoQ(wavelength, x)
        Q2=instrument_numba.TwoThetatoQ(wavelength, x)
        np.testing.assert_array_almost_equal(Q1, Q2)

    def test_tth2q_vector(self):
        x=np.linspace(0., 0.5, 1000)
        wavelength=x*0.001+4.5
        Q1=instrument.TwoThetatoQ(wavelength, x)
        Q2=instrument_numba.TwoThetatoQ(wavelength, x)
        np.testing.assert_array_almost_equal(Q1, Q2)

    def test_q2th_scalar(self):
        x=np.linspace(0., 0.5, 1000)
        wavelength=4.5
        TH1=instrument.QtoTheta(wavelength, x)
        TH2=instrument_numba.QtoTheta(wavelength, x)
        np.testing.assert_array_almost_equal(TH1, TH2)

    def test_q2th_vector(self):
        x=np.linspace(0., 0.5, 1000)
        wavelength=x*0.001+4.5
        TH1=instrument.QtoTheta(wavelength, x)
        TH2=instrument_numba.QtoTheta(wavelength, x)
        np.testing.assert_array_almost_equal(TH1, TH2)

    def test_resolutionvector_scalar(self):
        x=np.linspace(0., 0.5, 1000)
        dx=0.01
        points=10
        range=3.5
        Q1, weight1=instrument.ResolutionVector(x, dx, points, range)
        Q2, weight2=instrument_numba.ResolutionVector(x, dx, points, range)
        np.testing.assert_array_almost_equal(Q1, Q2)
        np.testing.assert_array_almost_equal(weight1, weight2)

    def test_resolutionvector_vector(self):
        x=np.linspace(0., 0.5, 1000)
        dx=x*0.001
        points=10
        range=3.5
        Q1, weight1=instrument.ResolutionVector(x, dx, points, range)
        Q2, weight2=instrument_numba.ResolutionVector(x, dx, points, range)
        np.testing.assert_array_almost_equal(Q1, Q2)
        np.testing.assert_array_almost_equal(weight1, weight2)

if __name__=='__main__':
    unittest.main()
