"""
Neutron cross sections extracted from Geant4 by the ESS dgcode framework (doi:10.1016/j.physb.2018.03.025).
"""

DATA_DIR = "nabs_geant4"
NEUTRON_ABSORPTIONS = {
    'Li':      'nabs_Li.npz',
    (3, 6):    'nabs_Li6.npz',
    (3, 7):    'nabs_Li7.npz',
    'B':       'nabs_B.npz',
    (5, 10):   'nabs_B10.npz',
    (5, 11):   'nabs_B11.npz',
    'Cd':      'nabs_Cd.npz',
    (48, 106): 'nabs_Cd106.npz',
    (48, 108): 'nabs_Cd108.npz',
    (48, 110): 'nabs_Cd110.npz',
    (48, 111): 'nabs_Cd111.npz',
    (48, 112): 'nabs_Cd112.npz',
    (48, 113): 'nabs_Cd113.npz',
    (48, 114): 'nabs_Cd114.npz',
    (48, 116): 'nabs_Cd116.npz',
    'Gd':      'nabs_Gd.npz',
    (64, 152): 'nabs_Gd152.npz',
    (64, 154): 'nabs_Gd154.npz',
    (64, 155): 'nabs_Gd155.npz',
    (64, 156): 'nabs_Gd156.npz',
    (64, 157): 'nabs_Gd157.npz',
    (64, 158): 'nabs_Gd158.npz',
    (64, 160): 'nabs_Gd160.npz'
    }
