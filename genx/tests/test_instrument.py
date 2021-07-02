"""
    Test of the instrument module the implements resolution convolution and footprint corrections.
"""
import unittest
import numpy as np
from importlib import reload
from unittest.mock import patch
from scipy.special import erf

from genx.models import lib
lib.USE_NUMBA=False
from genx.models.lib import instrument


class TestFootprint(unittest.TestCase):
    def test_fp_square(self):
        ai=np.linspace(0., 10., 100)
        slen=10.0
        bw=np.sin(5.0*np.pi/180.)*slen # beam with to have full illumination at 5 deg

        IFP=instrument.SquareIntensity(ai, slen, bw)
        self.assertTrue((IFP>=0.0).all())
        np.testing.assert_array_equal(IFP[ai>=5.0], 1.0)
        np.testing.assert_array_less(IFP[ai<5.0], 1.0)

    def test_fp_gauss(self):
        ai=np.linspace(0., 10., 100)
        slen=10.0
        bw=np.sin(5.0*np.pi/180.)*slen/2.0 # beam with to have +/-1 sigma illumination at 5 deg
        one_sigma=erf(1.0/np.sqrt(2.0)) # area of Gaussian between + and - sigma

        IFP=instrument.GaussIntensity(ai, slen/2., slen/2., bw)
        self.assertTrue((IFP>=0.0).all())
        self.assertTrue((IFP<1.0).all())
        np.testing.assert_array_less(one_sigma, IFP[ai>=5.0])
        np.testing.assert_array_less(IFP[ai<5.0], one_sigma)

    def test_fp_gauss(self):
        ai=np.linspace(0., 10., 100)
        slen=10.0
        bw=np.sin(5.0*np.pi/180.)*slen/2.0  # beam with to have +/-1 sigma illumination at 5 deg
        one_sigma=erf(1.0/np.sqrt(2.0))  # area of Gaussian between + and - sigma

        IFP=instrument.GaussIntensity(ai, slen/2., slen/2., bw)
        self.assertTrue((IFP>=0.0).all())
        self.assertTrue((IFP<1.0).all())
        np.testing.assert_array_less(one_sigma, IFP[ai>=5.0])
        np.testing.assert_array_less(IFP[ai<5.0], one_sigma)

    def test_fp_gauss_asym(self):
        ai=np.linspace(0., 10., 100)
        slen=10.0
        bw=np.sin(5.0*np.pi/180.)*slen/2.0  # beam with to have +/-1 sigma illumination at 5 deg
        two_sigma=erf(2.0/np.sqrt(2.0))  # area of Gaussian between + and - 2*sigma
        one_sigma=erf(1.0/np.sqrt(2.0))  # area of Gaussian between + and - sigma
        footprint=(two_sigma+one_sigma)/2.

        IFP=instrument.GaussIntensity(ai, slen, slen/2., bw)
        self.assertTrue((IFP>=0.0).all())
        self.assertTrue((IFP<1.0).all())
        np.testing.assert_array_less(footprint, IFP[ai>=5.0])
        np.testing.assert_array_less(IFP[ai<5.0], footprint)

class TestResolution(unittest.TestCase):
    def test_res_vector(self):
        q_in=np.linspace(0.01, 0.301, 300)
        n_points=5
        dq=0.001
        rng=3.0

        q_out, weights=instrument.ResolutionVector(q_in, dq, n_points, range=rng)
        self.assertEqual(q_in.shape[0]*n_points, q_out.shape[0])
        self.assertEqual((n_points, q_in.shape[0]), weights.shape)
        for i, dqi in enumerate(np.linspace(-rng*dq, rng*dq, n_points)):
            np.testing.assert_array_almost_equal(q_in, q_out.reshape(n_points, -1)[i]-dqi) # center of resolution is q-point

    def test_res_vector_vec(self):
        q_in=np.linspace(0.01, 0.301, 300)
        n_points=5
        dq=0.001
        rng=3.0

        q_out_s, weights_s=instrument.ResolutionVector(q_in, dq, n_points, range=rng)
        q_out_v, weights_v=instrument.ResolutionVector(q_in, dq*np.ones_like(q_in), n_points, range=rng)
        np.testing.assert_array_equal(q_out_s, q_out_v)
        np.testing.assert_array_equal(weights_s, weights_v)

    def test_convolution(self):
        q_in=np.linspace(0.01, 0.301, 300)
        n_points=5

        q_res=np.linspace(-0.0001, 0.0001, n_points)[:, np.newaxis]+q_in
        w_res=1./(n_points-1)*np.ones_like(q_res)
        q_res=q_res.flatten()
        I0=np.ones_like(q_res)

        # 1 intensity everywhere should convolute to 1
        Iconv=instrument.ConvoluteResolutionVector(q_res, I0, w_res)
        np.testing.assert_array_equal(Iconv, 1.0)
        # single 1 value in convolution range should give weight fraction as value
        I0=I0.reshape(n_points, -1)
        I0[np.linspace(-1,1,n_points)!=0]=0
        I0=I0.flatten()
        Iconv=instrument.ConvoluteResolutionVector(q_res, I0, w_res)
        np.testing.assert_array_almost_equal(Iconv, 1.0/(n_points-1))
        # full weight on point with 1 intensity should yield 1 again
        w_res[np.linspace(-1,1,n_points)!=0]=0
        Iconv=instrument.ConvoluteResolutionVector(q_res, I0, w_res)
        np.testing.assert_array_equal(Iconv, 1.0)

    def test_fft_convolve(self):
        q_in=np.linspace(0.01, 0.301, 300)
        I0=np.ones_like(q_in)

        Iconv=instrument.ConvoluteFast(q_in, I0, 0.001, range=3)
        np.testing.assert_array_almost_equal(Iconv, 1.0)

class TestInstrumentModuleBranching(unittest.TestCase):
    def setUp(self):
        from genx.models.lib import instrument_numba
        self.SquareIntensity=instrument_numba.SquareIntensity

    def tearDown(self):
        global instrument
        lib.USE_NUMBA=False
        reload(instrument)
        # make sure to reset the module, in case test_numba_error breaks before reset
        from genx.models.lib import instrument_numba
        instrument_numba.SquareIntensity=self.SquareIntensity

    def test_numba(self):
        global instrument
        lib.USE_NUMBA=True
        reload(instrument)
        from genx.models.lib import instrument_numba
        self.assertTrue(instrument.SquareIntensity is instrument_numba.SquareIntensity)

    def test_numba_error(self):
        global instrument
        lib.USE_NUMBA=True
        from genx.models.lib import instrument_numba
        # make sure the module misses a required function so an exception is raised in instrument when importing
        del instrument_numba.SquareIntensity
        reload(instrument)
        instrument_numba.SquareIntensity=self.SquareIntensity

if __name__=='__main__':
    unittest.main()
