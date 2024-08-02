"""
Implementation of the iterative algorithm to define bi-layer periods of super-mirrors
that was introduced by J.B. Hayter and H.A. Mook in J. Appl. Cryst. (1989). 22, 35-41.

Adopted from Visual Basic script of PSI optics group.
"""

from numpy import arctan, log, pi, sqrt

SCALE_CONSTANT = 3.0 / sqrt(8.0)


def delta_tau(tau, delta_sign, rho_fraction, zeta):
    # translated from VB script
    omega1 = sqrt(tau**2 - 1)
    omega2 = sqrt(tau**2 - rho_fraction)

    contrast = (omega2 - omega1) / (omega1 + omega2)
    kappa = (1 - contrast) / (1 + contrast)

    nu = log(1 - zeta) / (2 * log(kappa))

    if nu < 1:
        raise ValueError(
            "Nu should be below 1, check input parameters. "
            "This can be cause by a zeta value that is too smale, try e.g. 0.95."
        )

    rho_bar = (1 - abs(kappa) ** (1 / nu)) / (1 + abs(kappa) ** (1 / nu))
    arg = rho_bar * SCALE_CONSTANT

    d_omega1 = 2 * omega1 * arctan(arg / sqrt(1 - arg**2)) / pi
    wurzel = sqrt(1 + (omega1 + (delta_sign * d_omega1)) * (omega1 + (delta_sign * d_omega1)))

    return delta_sign * (wurzel - tau)


def sm_layers(rho1, rho2, N, zeta):
    tau = 1.1
    wert = SCALE_CONSTANT

    rho_fraction = rho2 / rho1  # bruch

    D_SL = []

    for i in range(N):
        # execute Neutrons method
        epsilon = 1.0
        while epsilon > 1e-4:
            temp1 = tau - delta_tau(tau, -1, rho_fraction, zeta) - wert
            temp2 = 1e4 * (1.0001 * tau - delta_tau((1.0001 * tau), -1, rho_fraction, zeta) - wert - temp1) / tau
            temp3 = temp1 / temp2
            tau = tau - temp3
            epsilon = abs(temp3 / tau)

        wert = tau + delta_tau(tau, 1, rho_fraction, zeta)

        omega1 = sqrt(tau**2 - 1)
        omega2 = sqrt(tau**2 - rho_fraction)

        d_1 = sqrt(pi) / (4 * omega1 * sqrt(rho1))
        d_2 = sqrt(pi) / (4 * omega2 * sqrt(rho1))

        D_SL.append([d_1, d_2])

    return D_SL
