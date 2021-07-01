"""
    Test of the instrument module the implements resolution convolution and footprint corrections.
"""
import unittest
import numpy as np
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

if __name__=='__main__':
    unittest.main()
