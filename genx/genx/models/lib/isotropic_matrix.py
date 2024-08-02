import numpy as np

from genx.core.custom_logging import iprint

_ctype = np.complex128


def refl(kj, kjm1):
    return (kj + kjm1) / 2 / kjm1


def ass_X_test(k_b, k_u, k_l, dd_u, dd_l, sigma, sigma_l, sigma_u, dtype=_ctype):
    """Just a simpler version of ass_X to locate an error somewhere..."""
    k_jm1_b = k_b[..., 1:]
    k_j_b = k_b[..., :-1]
    k_jm1_l = k_l[..., 1:]
    k_j_u = k_u[..., :-1]
    sigma_jm1_l = sigma_l[..., 1:]
    sigma_j_b = sigma[..., :-1]
    sigma_j_u = sigma_u[..., :-1]
    dd_jm1_l = dd_l[..., 1:]
    dd_j_u = dd_u[..., :-1]

    iprint(dd_j_u, dd_jm1_l)

    # Reflectivites
    r_j_b = refl(k_j_b, k_j_u)
    # print 'r_j_b', r_j_b
    r_jm1_l = refl(k_jm1_l, k_jm1_b)
    # print 'r_jm1_l', r_jm1_l
    r_j_u = refl(k_j_u, k_jm1_l)
    # print 'r_j_u', r_j_u
    p_jm1_l = np.exp(-1.0j * dd_jm1_l * k_jm1_l)
    p_j_u = np.exp(-1.0j * dd_j_u * k_j_u)

    # Defining the X matrix
    X = np.empty((2, 2) + k_j_b.shape, dtype=dtype)

    X[0, 0] = p_jm1_l * r_j_u * p_j_u
    X[0, 1] = -(p_jm1_l * r_j_u - p_jm1_l) / p_j_u
    X[1, 0] = -((r_j_u - 1) * p_j_u) / p_jm1_l
    X[1, 1] = r_j_u / (p_jm1_l * p_j_u)

    return X


def ass_X_interfacelayer4(k_b, k_u, k_l, dd_u, dd_l, sigma, sigma_l, sigma_u, dtype=_ctype):
    """Function that creates a interfacelayer at the top and bottom
    of the layer. with correlated roughness for the interface layers and the
    additional roughness of sigma_l and sigma_u.
    The subscripts b, l and u reffers to bulk, lower and upper respectively.

    v 0.4
    """
    # print dd_l, dd_u
    # Variables definitions that goes into the expressions:
    k_jm1_b = k_b[..., 1:]
    k_j_b = k_b[..., :-1]
    k_jm1_l = k_l[..., 1:]
    k_j_u = k_u[..., :-1]
    sigma_jm1_l = sigma_l[..., 1:]
    sigma_j_b = sigma[..., :-1]
    sigma_j_u = sigma_u[..., :-1]
    dd_jm1_l = dd_l[..., 1:]
    dd_j_u = dd_u[..., :-1]

    # Reflectivites
    r_j_b = refl(k_j_b, k_j_u)
    r_jm1_l = refl(k_jm1_l, k_jm1_b)
    r_j = refl(k_j_u, k_jm1_l)
    p_jm1_l = np.exp(-1.0j * dd_jm1_l * k_jm1_l)
    p_j_u = np.exp(-1.0j * dd_j_u * k_j_u)
    # p_j_b = np.exp(-1.0J*dd_j_b*k_j_b)

    # Defining the X matrix
    X = np.empty((2, 2) + k_j_b.shape, dtype=dtype)

    tmp1 = k_jm1_b**2
    tmp2 = sigma_jm1_l**2
    tmp3 = -(tmp1 * tmp2) / 2
    tmp4 = -k_jm1_b * tmp2 * k_jm1_l
    tmp5 = -(tmp2 * k_jm1_l**2) / 2
    tmp6 = sigma_j_b**2
    tmp7 = -(tmp1 * tmp6) / 2
    tmp8 = k_j_b**2
    tmp9 = -(tmp6 * tmp8) / 2
    tmp10 = sigma_j_u**2
    tmp11 = -(tmp8 * tmp10) / 2
    tmp12 = -k_j_b * tmp10 * k_j_u
    tmp13 = -(tmp10 * k_j_u**2) / 2
    tmp14 = np.exp(tmp13 + tmp12 + tmp11 + tmp9 + tmp7 + tmp5 + tmp4 + tmp3)
    tmp15 = 1 / p_jm1_l
    tmp16 = 1 / p_j_u
    tmp17 = 2 * k_jm1_b * tmp2 * k_jm1_l
    tmp18 = k_jm1_b * tmp6 * k_j_b
    tmp21 = np.exp(tmp18)
    tmp19 = np.exp(tmp17) * tmp21
    tmp20 = p_jm1_l**2
    # tmp21=np.exp(tmp18)
    tmp22 = -tmp21
    tmp23 = 2 * k_j_b * tmp10 * k_j_u
    # tmp24=np.exp(tmp23+tmp18)
    # tmp24=np.exp(tmp23)*tmp21
    # tmp25=np.exp(tmp23+tmp18+tmp17)
    tmp26 = p_j_u**2
    # tmp27=np.exp(tmp13+tmp12+tmp11+tmp9-k_jm1_b*tmp6*k_j_b+tmp7+tmp5+tmp4+tmp3)
    tmp27 = tmp14 / tmp21
    tmp28 = np.exp(tmp17)
    # tmp29=np.exp(tmp23+tmp17)
    tmp30 = np.exp(tmp23)
    tmp29 = tmp28 * tmp30
    tmp24 = tmp30 * tmp21
    tmp25 = tmp30 * tmp19
    tmp31 = -tmp20
    tmp32 = tmp21 * tmp20
    tmp33 = -tmp21 * tmp20
    tmp34 = tmp24 * tmp20

    X[0, 0] = (
        tmp14
        * tmp15
        * tmp16
        * (
            (
                (((tmp25 * tmp20 + tmp24) * r_jm1_l - tmp24) * r_j - tmp24 * r_jm1_l + tmp24) * tmp26
                + ((tmp19 * tmp20 + tmp21) * r_jm1_l + tmp22) * r_j
                - tmp19 * tmp20 * r_jm1_l
            )
            * r_j_b
            + ((tmp22 - tmp19 * tmp20) * r_jm1_l + tmp21) * r_j
            + tmp19 * tmp20 * r_jm1_l
        )
    )

    X[0, 1] = (
        -tmp27
        * tmp15
        * tmp16
        * (
            (
                (((tmp28 * tmp20 + 1) * r_jm1_l - 1) * r_j - r_jm1_l + 1) * tmp26
                + ((tmp29 * tmp20 + tmp30) * r_jm1_l - tmp30) * r_j
                - tmp29 * tmp20 * r_jm1_l
            )
            * r_j_b
            + (((-tmp28 * tmp20 - 1) * r_jm1_l + 1) * r_j + r_jm1_l - 1) * tmp26
        )
    )

    X[1, 0] = (
        -tmp27
        * tmp15
        * tmp16
        * (
            (
                (((tmp30 * tmp20 + tmp29) * r_jm1_l - tmp30 * tmp20) * r_j - tmp29 * r_jm1_l) * tmp26
                + ((tmp20 + tmp28) * r_jm1_l + tmp31) * r_j
                - tmp20 * r_jm1_l
                + tmp20
            )
            * r_j_b
            + ((tmp31 - tmp28) * r_jm1_l + tmp20) * r_j
            + tmp20 * r_jm1_l
            + tmp31
        )
    )

    X[1, 1] = (
        tmp14
        * tmp15
        * tmp16
        * (
            (
                (((tmp32 + tmp19) * r_jm1_l + tmp33) * r_j - tmp19 * r_jm1_l) * tmp26
                + ((tmp34 + tmp25) * r_jm1_l - tmp24 * tmp20) * r_j
                - tmp24 * tmp20 * r_jm1_l
                + tmp34
            )
            * r_j_b
            + (((tmp33 - tmp19) * r_jm1_l + tmp32) * r_j + tmp19 * r_jm1_l) * tmp26
        )
    )

    return X


def ass_X_interfacelayer3(k_b, k_u, k_l, dd_u, dd_l, sigma, sigma_l, sigma_u, dtype=_ctype):
    """Function that creates a interfacelayer at the top and bottom
    of the layer. with correlated roughness for the interface layers and the
    additional roughness of sigma_l and sigma_u.
    The subscripts b, l and u reffers to bulk, lower and upper respectively.

    v 0.3
    """

    # Variables definitions that goes into the expressions:
    k_jm1_b = k_b[..., 1:]
    k_j_b = k_b[..., :-1]
    k_jm1_l = k_l[..., 1:]
    k_j_u = k_u[..., :-1]
    sigma_jm1_l = sigma_l[..., 1:]
    sigma_j_b = sigma[..., :-1]
    sigma_j_u = sigma_u[..., :-1]
    dd_jm1_l = dd_l[..., 1:]
    dd_j_u = dd_u[..., :-1]

    iprint(dd_j_u, dd_jm1_l)

    # Reflectivites
    r_j_b = refl(k_j_b, k_j_u)
    iprint("r_j_b", r_j_b)
    r_jm1_l = refl(k_jm1_l, k_jm1_b)
    iprint("r_jm1_l", r_jm1_l)
    r_j_u = refl(k_j_u, k_jm1_l)
    iprint("r_j_u", r_j_u)
    p_jm1_l = np.exp(-1.0j * dd_jm1_l * k_jm1_l)
    p_j_u = np.exp(-1.0j * dd_j_u * k_j_u)
    # p_j_b = np.exp(-1.0J*dd_j_b*k_j_b)

    # Defining the X matrix
    X = np.empty((2, 2) + k_j_b.shape, dtype=dtype)

    tmp1 = k_jm1_b**2
    tmp2 = sigma_jm1_l**2
    tmp3 = -(tmp1 * tmp2) / 2
    tmp4 = k_jm1_b * tmp2 * k_jm1_l
    tmp5 = -(tmp2 * k_jm1_l**2) / 2
    tmp6 = sigma_j_b**2
    tmp7 = -(tmp1 * tmp6) / 2
    tmp8 = k_jm1_b * tmp6 * k_j_b
    tmp9 = k_j_b**2
    tmp10 = -(tmp6 * tmp9) / 2
    tmp11 = sigma_j_u**2
    tmp12 = -(tmp9 * tmp11) / 2
    tmp13 = -k_j_b * tmp11 * k_j_u
    tmp14 = -(tmp11 * k_j_u**2) / 2
    tmp15 = np.exp(tmp14 + tmp13 + tmp12 + tmp10 + tmp8 + tmp7 + tmp5 + tmp4 + tmp3)
    tmp16 = 1 / p_j_u
    tmp17 = -k_jm1_b * tmp2 * k_jm1_l
    tmp18 = np.exp(tmp14 + tmp13 + tmp12 + tmp10 + tmp8 + tmp7 + tmp5 + tmp17 + tmp3)
    tmp19 = 1 / p_jm1_l
    tmp20 = k_j_b * tmp11 * k_j_u
    tmp21 = np.exp(tmp14 + tmp20 + tmp12 + tmp10 + tmp8 + tmp7 + tmp5 + tmp17 + tmp3)
    tmp22 = np.exp(tmp14 + tmp20 + tmp12 + tmp10 + tmp8 + tmp7 + tmp5 + tmp4 + tmp3)
    tmp23 = -k_jm1_b * tmp6 * k_j_b
    tmp24 = np.exp(tmp14 + tmp13 + tmp12 + tmp10 + tmp23 + tmp7 + tmp5 + tmp17 + tmp3)
    tmp25 = np.exp(tmp14 + tmp13 + tmp12 + tmp10 + tmp23 + tmp7 + tmp5 + tmp4 + tmp3)
    tmp26 = np.exp(tmp14 + tmp20 + tmp12 + tmp10 + tmp23 + tmp7 + tmp5 + tmp4 + tmp3)
    tmp27 = np.exp(tmp14 + tmp20 + tmp12 + tmp10 + tmp23 + tmp7 + tmp5 + tmp17 + tmp3)

    X[0, 0] = (
        tmp22 * p_jm1_l * r_jm1_l * r_j_b * p_j_u * r_j_u
        + tmp21 * tmp19 * r_jm1_l * r_j_b * p_j_u * r_j_u
        - tmp21 * tmp19 * r_j_b * p_j_u * r_j_u
        - tmp21 * tmp19 * r_jm1_l * p_j_u * r_j_u
        + tmp21 * tmp19 * p_j_u * r_j_u
        + tmp15 * p_jm1_l * r_jm1_l * r_j_b * tmp16 * r_j_u
        + tmp18 * tmp19 * r_jm1_l * r_j_b * tmp16 * r_j_u
        - tmp18 * tmp19 * r_j_b * tmp16 * r_j_u
        - tmp15 * p_jm1_l * r_jm1_l * tmp16 * r_j_u
        - tmp15 * p_jm1_l * r_jm1_l * r_j_b * tmp16
        - tmp18 * tmp19 * r_jm1_l * r_j_b * tmp16
        + tmp18 * tmp19 * r_j_b * tmp16
        + tmp15 * p_jm1_l * r_jm1_l * tmp16
    )

    X[0, 1] = (
        -tmp25 * p_jm1_l * r_jm1_l * r_j_b * p_j_u * r_j_u
        - tmp24 * tmp19 * r_jm1_l * r_j_b * p_j_u * r_j_u
        + tmp24 * tmp19 * r_j_b * p_j_u * r_j_u
        + tmp24 * tmp19 * r_jm1_l * p_j_u * r_j_u
        - tmp24 * tmp19 * p_j_u * r_j_u
        - tmp26 * p_jm1_l * r_jm1_l * r_j_b * tmp16 * r_j_u
        - tmp27 * tmp19 * r_jm1_l * r_j_b * tmp16 * r_j_u
        + tmp27 * tmp19 * r_j_b * tmp16 * r_j_u
        + tmp26 * p_jm1_l * r_jm1_l * tmp16 * r_j_u
        + tmp25 * p_jm1_l * r_jm1_l * r_j_b * p_j_u
        + tmp24 * tmp19 * r_jm1_l * r_j_b * p_j_u
        - tmp24 * tmp19 * r_j_b * p_j_u
        - tmp24 * tmp19 * r_jm1_l * p_j_u
        + tmp24 * tmp19 * p_j_u
    )

    X[1, 0] = (
        -tmp27 * p_jm1_l * r_jm1_l * r_j_b * p_j_u * r_j_u
        - tmp26 * tmp19 * r_jm1_l * r_j_b * p_j_u * r_j_u
        + tmp27 * p_jm1_l * r_j_b * p_j_u * r_j_u
        + tmp26 * tmp19 * r_jm1_l * p_j_u * r_j_u
        - tmp24 * p_jm1_l * r_jm1_l * r_j_b * tmp16 * r_j_u
        - tmp25 * tmp19 * r_jm1_l * r_j_b * tmp16 * r_j_u
        + tmp24 * p_jm1_l * r_j_b * tmp16 * r_j_u
        + tmp24 * p_jm1_l * r_jm1_l * tmp16 * r_j_u
        - tmp24 * p_jm1_l * tmp16 * r_j_u
        + tmp24 * p_jm1_l * r_jm1_l * r_j_b * tmp16
        + tmp25 * tmp19 * r_jm1_l * r_j_b * tmp16
        - tmp24 * p_jm1_l * r_j_b * tmp16
        - tmp24 * p_jm1_l * r_jm1_l * tmp16
        + tmp24 * p_jm1_l * tmp16
    )

    X[1, 1] = (
        tmp18 * p_jm1_l * r_jm1_l * r_j_b * p_j_u * r_j_u
        + tmp15 * tmp19 * r_jm1_l * r_j_b * p_j_u * r_j_u
        - tmp18 * p_jm1_l * r_j_b * p_j_u * r_j_u
        - tmp15 * tmp19 * r_jm1_l * p_j_u * r_j_u
        + tmp21 * p_jm1_l * r_jm1_l * r_j_b * tmp16 * r_j_u
        + tmp22 * tmp19 * r_jm1_l * r_j_b * tmp16 * r_j_u
        - tmp21 * p_jm1_l * r_j_b * tmp16 * r_j_u
        - tmp21 * p_jm1_l * r_jm1_l * tmp16 * r_j_u
        + tmp21 * p_jm1_l * tmp16 * r_j_u
        - tmp18 * p_jm1_l * r_jm1_l * r_j_b * p_j_u
        - tmp15 * tmp19 * r_jm1_l * r_j_b * p_j_u
        + tmp18 * p_jm1_l * r_j_b * p_j_u
        + tmp15 * tmp19 * r_jm1_l * p_j_u
    )

    return X


def ass_X_interfacelayer(k_b, k_u, k_l, dd_u, dd_l, sigma, sigma_l, sigma_u, dtype=_ctype):
    """Function that creates a interfacelayer at the top and bottom
    of the layer. with correlated roughness for the interface layers and the
    additional roughness of sigma_l and sigma_u.
    The subscripts b, l and u reffers to bulk, lower and upper respectively.

    v 0.2
    """
    iprint(dd_l, dd_u)
    # Variables definitions that goes into the expressions:
    k_jm1_b = k_b[..., 1:]
    k_j_b = k_b[..., :-1]
    k_jm1_l = k_l[..., 1:]
    k_j_u = k_u[..., :-1]
    sigma_jm1_l = sigma_l[..., 1:]
    sigma_j_b = sigma[..., :-1]
    sigma_j_u = sigma_u[..., :-1]
    dd_jm1_l = dd_l[..., 1:]
    dd_j_u = dd_u[..., :-1]

    # Reflectivites
    r_j_b = refl(k_j_b, k_j_u)
    r_jm1_l = refl(k_jm1_l, k_jm1_b)
    r_j_u = refl(k_j_u, k_jm1_l)
    p_jm1_l = np.exp(-1.0j * dd_jm1_l * k_jm1_b)
    p_j_u = np.exp(-1.0j * dd_j_u * k_j_u)
    # p_j_b = np.exp(-1.0J*dd_j_b*k_j_b)

    # Defining the X matrix
    X = np.empty((2, 2) + k_j_b.shape, dtype=dtype)

    tmp1 = k_jm1_b**2
    tmp2 = sigma_jm1_l**2
    tmp3 = -(tmp1 * tmp2) / 2
    tmp4 = -k_jm1_b * tmp2 * k_jm1_l
    tmp5 = -(tmp2 * k_jm1_l**2) / 2
    tmp6 = sigma_j_b**2
    tmp7 = -(tmp1 * tmp6) / 2
    tmp8 = k_j_b**2
    tmp9 = -(tmp6 * tmp8) / 2
    tmp10 = sigma_j_u**2
    tmp11 = -(tmp8 * tmp10) / 2
    tmp12 = -k_j_b * tmp10 * k_j_u
    tmp13 = -(tmp10 * k_j_u**2) / 2
    tmp14 = np.exp(tmp13 + tmp12 + tmp11 + tmp9 + tmp7 + tmp5 + tmp4 + tmp3)
    tmp15 = 1 / p_jm1_l
    tmp16 = 1 / p_j_u
    tmp17 = 2 * k_jm1_b * tmp2 * k_jm1_l
    tmp18 = k_jm1_b * tmp6 * k_j_b
    tmp21 = np.exp(tmp18)
    tmp19 = np.exp(tmp17) * tmp21
    tmp20 = p_jm1_l**2
    # tmp21=np.exp(tmp18)
    tmp22 = -tmp21
    tmp23 = 2 * k_j_b * tmp10 * k_j_u
    # tmp24=np.exp(tmp23+tmp18)
    # tmp24=np.exp(tmp23)*tmp21
    # tmp25=np.exp(tmp23+tmp18+tmp17)
    tmp26 = p_j_u**2
    # tmp27=np.exp(tmp13+tmp12+tmp11+tmp9-k_jm1_b*tmp6*k_j_b+tmp7+tmp5+tmp4+tmp3)
    tmp27 = tmp14 / tmp21
    tmp28 = np.exp(tmp17)
    # tmp29=np.exp(tmp23+tmp17)
    tmp30 = np.exp(tmp23)
    tmp29 = tmp28 * tmp30
    tmp24 = tmp30 * tmp21
    tmp25 = tmp30 * tmp19
    tmp31 = -tmp20
    tmp32 = tmp21 * tmp20
    tmp33 = -tmp21 * tmp20
    tmp34 = tmp24 * tmp20

    X[0, 0] = (
        tmp14
        * tmp15
        * tmp16
        * (
            (
                (((tmp25 * tmp20 + tmp24) * r_jm1_l - tmp24) * r_j_b - tmp24 * r_jm1_l + tmp24) * tmp26
                + ((tmp19 * tmp20 + tmp21) * r_jm1_l + tmp22) * r_j_b
                - tmp19 * tmp20 * r_jm1_l
            )
            * r_j_u
            + ((tmp22 - tmp19 * tmp20) * r_jm1_l + tmp21) * r_j_b
            + tmp19 * tmp20 * r_jm1_l
        )
    )

    X[0, 1] = (
        -tmp27
        * tmp15
        * tmp16
        * (
            (
                (((tmp28 * tmp20 + 1) * r_jm1_l - 1) * r_j_b - r_jm1_l + 1) * tmp26
                + ((tmp29 * tmp20 + tmp30) * r_jm1_l - tmp30) * r_j_b
                - tmp29 * tmp20 * r_jm1_l
            )
            * r_j_u
            + (((-tmp28 * tmp20 - 1) * r_jm1_l + 1) * r_j_b + r_jm1_l - 1) * tmp26
        )
    )

    X[1, 0] = (
        -tmp27
        * tmp15
        * tmp16
        * (
            (
                (((tmp30 * tmp20 + tmp29) * r_jm1_l - tmp30 * tmp20) * r_j_b - tmp29 * r_jm1_l) * tmp26
                + ((tmp20 + tmp28) * r_jm1_l + tmp31) * r_j_b
                - tmp20 * r_jm1_l
                + tmp20
            )
            * r_j_u
            + ((tmp31 - tmp28) * r_jm1_l + tmp20) * r_j_b
            + tmp20 * r_jm1_l
            + tmp31
        )
    )

    X[1, 1] = (
        tmp14
        * tmp15
        * tmp16
        * (
            (
                (((tmp32 + tmp19) * r_jm1_l + tmp33) * r_j_b - tmp19 * r_jm1_l) * tmp26
                + ((tmp34 + tmp25) * r_jm1_l - tmp24 * tmp20) * r_j_b
                - tmp24 * tmp20 * r_jm1_l
                + tmp34
            )
            * r_j_u
            + (((tmp33 - tmp19) * r_jm1_l + tmp32) * r_j_b + tmp19 * r_jm1_l) * tmp26
        )
    )

    return X


def ass_X_interfacelayer1(k_b, k_u, k_l, dd_u, dd_l, sigma, sigma_l, sigma_u, dtype=_ctype):
    """Function that creates a interfacelayer at the top and bottom
    of the layer. with correlated roughness for the interface layers and the
    additional roughness of sigma_l and sigma_u.
    The subscripts b, l and u reffers to bulk, lower and upper respectively.

    v 0.1
    """

    # Variables definitions that goes into the expressions:
    k_jm1_b = k_b[..., 1:]
    k_j_b = k_b[..., :-1]
    k_jm1_l = k_l[..., 1:]
    k_j_u = k_u[..., :-1]
    sigma_jm1_l = sigma_l[..., 1:]
    sigma_j_b = sigma[..., :-1]
    sigma_j_u = sigma_u[..., :-1]
    dd_jm1_l = dd_u[..., 1:]
    dd_j_u = dd_u[..., :-1]

    # Defining the X matrix
    X = np.empty((2, 2) + k_j_b.shape, dtype=dtype)

    # Temporary variables to ease the calculation
    tmp1 = k_jm1_b * k_jm1_b
    tmp2 = sigma_jm1_l * sigma_jm1_l
    tmp3 = tmp1 * tmp2
    tmp4 = (2 * k_jm1_b * tmp2 + 2 * 1.0j * dd_jm1_l) * k_jm1_l
    tmp5 = k_jm1_l * k_jm1_l
    tmp6 = tmp2 * tmp5
    tmp7 = sigma_j_b * sigma_j_b
    tmp8 = tmp1 * tmp7
    tmp9 = k_j_b**2
    tmp10 = tmp7 * tmp9
    tmp11 = sigma_j_u * sigma_j_u
    tmp12 = tmp9 * tmp11
    tmp13 = (2 * k_j_b * tmp11 + 2 * 1.0j * dd_j_u) * k_j_u
    tmp14 = k_j_u * k_j_u
    tmp15 = tmp11 * tmp14
    tmp16 = 1 / np.exp((tmp15 + tmp13 + tmp12 + tmp10 + tmp8 + tmp6 + tmp4 + tmp3) / 2)
    tmp17 = 1 / k_jm1_b
    tmp18 = 1 / k_jm1_l
    tmp19 = 1 / k_j_u
    tmp20 = np.exp(2 * 1.0j * dd_jm1_l * k_jm1_l)
    tmp21 = -tmp20
    tmp22 = np.exp(2 * k_jm1_b * tmp2 * k_jm1_l)
    tmp23 = -tmp22
    tmp24 = tmp23 + tmp21
    tmp25 = k_jm1_b * tmp7 * k_j_b
    tmp26 = 2 * 1.0j * dd_j_u * k_j_u
    tmp27 = np.exp(tmp26 + tmp25)
    tmp28 = tmp22 + tmp20
    tmp29 = 2 * k_j_b * tmp11 * k_j_u
    tmp30 = np.exp(tmp29 + tmp25)
    tmp31 = tmp28 * tmp30
    tmp32 = tmp31 + tmp24 * tmp27
    tmp33 = tmp23 + tmp20
    tmp34 = tmp22 + tmp21
    tmp35 = tmp34 * tmp30
    tmp36 = tmp35 + tmp33 * tmp27
    tmp37 = tmp31 + tmp28 * tmp27
    tmp38 = tmp35 + tmp34 * tmp27
    tmp39 = 1 / np.exp((tmp15 + tmp13 + tmp12 + tmp10 + 2 * k_jm1_b * tmp7 * k_j_b + tmp8 + tmp6 + tmp4 + tmp3) / 2)
    tmp40 = np.exp(tmp13)
    tmp41 = tmp24 * tmp40
    tmp42 = tmp33 * tmp40
    tmp43 = tmp34 * tmp40
    tmp44 = tmp28 * tmp40
    tmp45 = np.exp(tmp4)
    tmp46 = tmp45 + 1
    tmp47 = np.exp(tmp26)
    tmp48 = tmp46 * tmp47
    tmp49 = -tmp45
    tmp50 = tmp49 - 1
    tmp51 = np.exp(tmp29)
    tmp52 = tmp50 * tmp51
    tmp53 = tmp45 - 1
    tmp54 = tmp53 * tmp47
    tmp55 = tmp49 + 1
    tmp56 = tmp55 * tmp51
    tmp57 = tmp50 * tmp47
    tmp58 = tmp55 * tmp47
    tmp59 = tmp53 * tmp51
    tmp60 = tmp46 * tmp51
    tmp61 = np.exp(tmp25)
    tmp62 = np.exp(tmp13 + tmp25)
    tmp63 = tmp46 * tmp62
    tmp64 = tmp63 + tmp50 * tmp61
    tmp65 = tmp53 * tmp62
    tmp66 = tmp65 + tmp55 * tmp61
    tmp67 = tmp63 + tmp46 * tmp61
    tmp68 = tmp65 + tmp53 * tmp61

    # Assembling the interface matrices
    X[0, 0] = (
        tmp16
        * tmp17
        * tmp18
        * tmp19
        * (
            (tmp32 * k_jm1_l + tmp36 * k_jm1_b) * tmp14
            + ((tmp37 * k_jm1_l + tmp38 * k_jm1_b) * k_j_b + tmp38 * tmp5 + tmp37 * k_jm1_b * k_jm1_l) * k_j_u
            + (tmp36 * tmp5 + tmp32 * k_jm1_b * k_jm1_l) * k_j_b
        )
    ) / 8

    X[0, 1] = (
        -(
            tmp39
            * tmp17
            * tmp18
            * tmp19
            * (
                ((tmp44 + tmp23 + tmp21) * k_jm1_l + (tmp43 + tmp23 + tmp20) * k_jm1_b) * tmp14
                + (
                    ((tmp44 + tmp22 + tmp20) * k_jm1_l + (tmp43 + tmp22 + tmp21) * k_jm1_b) * k_j_b
                    + (tmp42 + tmp23 + tmp20) * tmp5
                    + (tmp41 + tmp23 + tmp21) * k_jm1_b * k_jm1_l
                )
                * k_j_u
                + ((tmp42 + tmp22 + tmp21) * tmp5 + (tmp41 + tmp22 + tmp20) * k_jm1_b * k_jm1_l) * k_j_b
            )
        )
        / 8
    )

    X[1, 0] = (
        -(
            tmp39
            * tmp17
            * tmp18
            * tmp19
            * (
                ((tmp60 + tmp57) * k_jm1_l + (tmp59 + tmp58) * k_jm1_b) * tmp14
                + (
                    ((tmp60 + tmp48) * k_jm1_l + (tmp59 + tmp54) * k_jm1_b) * k_j_b
                    + (tmp56 + tmp58) * tmp5
                    + (tmp52 + tmp57) * k_jm1_b * k_jm1_l
                )
                * k_j_u
                + ((tmp56 + tmp54) * tmp5 + (tmp52 + tmp48) * k_jm1_b * k_jm1_l) * k_j_b
            )
        )
        / 8
    )

    X[1, 1] = (
        tmp16
        * tmp17
        * tmp18
        * tmp19
        * (
            (tmp64 * k_jm1_l + tmp66 * k_jm1_b) * tmp14
            + ((tmp67 * k_jm1_l + tmp68 * k_jm1_b) * k_j_b + tmp68 * tmp5 + tmp67 * k_jm1_b * k_jm1_l) * k_j_u
            + (tmp66 * tmp5 + tmp64 * k_jm1_b * k_jm1_l) * k_j_b
        )
    ) / 8

    return X
