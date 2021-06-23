"""
Constants used in various computations.
"""

r_e=2.8179403227e-5 # classical electron radius [Å] ( 1/(4 pi epsilon0) * (e_charge**2)/(m_e*c**2) )

M_to_SLD=2.853e-6 # m/A/Å² conversion M [A/m] to SLD [Å^-2]
ApM_to_emucc=1e-3 # conversion M [A/m] to M [emu/cm³]
Memu_to_SLD=M_to_SLD*ApM_to_emucc
muB=9.2740101E-24 # Bohr magneton [A·m^2]

muB_to_SL=muB*M_to_SLD*1e24 # magnetic scattering length for a moment of 1 Bohr magneton
AAm2_to_emucc=1e-5/Memu_to_SLD # conversion of SLD [Å/m²] to M [emu/cm³]

