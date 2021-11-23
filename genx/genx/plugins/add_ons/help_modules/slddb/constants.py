"""
Physical constants and conversion factors used in the package.

The values given here have been taken from the
CODATA Internationally recommended 2018 values of the Fundamental Physical Constants
provided at https://physics.nist.gov/cuu/Constants/index.html
"""

u2g = 1.660_539_066_60e-24  # 1/N_Avogadro (g/mol => g/atom)
h_eVs = 4.135_667_696e-15  # eV·s Planck's constant
h_Js = 6.626_070_15e-34  # J·s Planck's constant
m_n = 1.674_927_498_04e-27  # kg neutron mass
c_ms = 299_792_458.0  # m/s speed of light
# some conversion constants used
kilo = 1e3
m2angstrom = 1e10
fm2angstrom = 1e-5
eV2J = h_Js/h_eVs

r_e = 2.817_940_3262  # fm - classical electron radius
r_e_angstrom = fm2angstrom*r_e  # Å - classical electron radius
sigma_to_b_1A = 5.0e-4  # fm/barn absorption at 1.0Å (sigma=4*pi*b/k)
wavelength_for_b = h_Js/m_n/2200.0*m2angstrom  # 1.798 Å (wavelength at 2200 m/s, the default for reported b-values)
sigma_to_b = sigma_to_b_1A/wavelength_for_b  # fm/barn absorption at 1.798Å (sigma=4*pi*b/k)
E_to_lambda = h_eVs/kilo*c_ms*m2angstrom  # keV·Å conversion x-ray energy to wavelength (h*c)

muB = 9.274_010_0783e3  # kA/m Å³ - Bohr Magneton scaled to get kA/m from Ä³ FU volume

#
# Non-fundamental constant derived values with yet undocumented origin
# Transition energies from Deslattes, R.D.;
# Deslattes, R.D., et al. Rev. Mod. Phys. 75, 35-99.  (2003)
# https://doi.org/10.1103/RevModPhys.75.35
Cu_kalpha1 = 8.04782  # keV
Cu_kalpha2 = 8.02784  # keV
Cu_kalpha = (2*Cu_kalpha1+Cu_kalpha2)/3.0  # keV
Mo_kalpha1 = 17.4793  # keV
Mo_kalpha2 = 17.3743  # keV
Mo_kalpha = (2*Mo_kalpha1+Mo_kalpha2)/3.  # keV

rho_of_M = 2.853e-9  # Å^-2 from kA/m

# Standard water densities for contrast matching calculations (25 C°)
dens_H2O = 0.99707  # Density of heavy water, TSING-LIEN CHANG & LÜ-HO TUNG, Nature 163, page737 (1949)
dens_D2O = 1.10440  # https://doi.org/10.1038/163737a0
