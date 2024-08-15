"""
Constants used in various computations.
"""

r_e = 2.8179403227e-5  # classical electron radius [Å] ( 1/(4 pi epsilon0) * (e_charge**2)/(m_e*c**2) )
N_a = 6.022_140_76e23  # 1/mol

AA_to_eV = 12_398.41984 # x-ray energy conversion [eV/Å] ( h * c / 1Å)

M_to_SLD = 2.853e-6  # m/A/Å² conversion M [A/m] to SLD [Å^-2]
ApM_to_emucc = 1e-3  # conversion M [A/m] to M [emu/cm³]
Memu_to_SLD = M_to_SLD * ApM_to_emucc
muB = 9.2740101e-24  # Bohr magneton [A·m^2]

muB_to_SL = muB * M_to_SLD * 1e24  # magnetic scattering length for a moment of 1 Bohr magneton
T_to_SL = 2.31605e-6  # 1/T/Á² magnetic field in Tesla to scattering length
AAm2_to_emucc = 1e-6 / Memu_to_SLD  # conversion of SLD [Å/m²] to M [emu/cm³]

MASS_DENSITY_CONVERSION = 1e-24 * N_a  # g/cm³-> u/Å³ - used in converting mass density for SLDs
