"""
Empirical correlations for coating loss rates, fitted by AICc-guided OLS
regression on the 19-run Bosch 2018 DoE (Run 20 excluded — anomalous SSA).

Both functions accept raw (un-centred) process variables and internally apply
the training-set means used during model fitting.

r_spraying — Step 06a
    Selected model: spray rate + coating conc + DM ratio + coating conc × DM ratio
    R2 = 0.844  Adj-R2 = 0.800  LOO-CV ratio = 1.11

r_drying — Step 06b
    Selected model: batch size + DM ratio + SSA + inlet humidity + DM ratio × inlet humidity
    R2 = 0.851  Adj-R2 = 0.794  LOO-CV ratio = 1.36
"""

# ── Training-set centering means (computed from the 19 valid DoE runs) ────────
_MEAN_SPRAY_RATE   = 119.289474   # g/min
_MEAN_COATING_CONC =   1.473684   # wt %
_MEAN_DM_RATIO     =   6.472807   # g dry coating / kg particles

_MEAN_BATCH_SIZE   =   4.578947   # kg
_MEAN_SSA          =  65.477895   # cm2/g
_MEAN_HUMIDITY     =  13.447368   # g/kg dry air


def calc_r_spraying(
    spray_rate_g_min: float,
    coating_conc_pct: float,
    dm_ratio_g_kg: float,
) -> float:
    """
    Empirical correlation for r_spraying [kg/s] — coating loss rate during spray.

    dMc/dt = -r_spraying  (order-0 attrition)

    Parameters
    ----------
    spray_rate_g_min : measured spray rate [g/min]
    coating_conc_pct : coating solution dry-matter concentration [wt %]
    dm_ratio_g_kg    : dry coating mass target / batch mass × 1000 [g/kg]
                       = Qty_solution × coating_conc_pct/100 / batch_size_kg × 1000

    Returns
    -------
    r_spraying : float [kg/s]  — clipped to >= 0
    """
    sr_c  = spray_rate_g_min - _MEAN_SPRAY_RATE
    cc_c  = coating_conc_pct - _MEAN_COATING_CONC
    dm_c  = dm_ratio_g_kg    - _MEAN_DM_RATIO

    val_1e6 = (
        15.333703
        +  0.196979 * sr_c
        +  9.204374 * cc_c
        + -2.799050 * dm_c
        + -3.697864 * cc_c * dm_c
    )
    return max(float(val_1e6) * 1e-6, 0.0)


def calc_r_drying_empirical(
    batch_size_kg: float,
    dm_ratio_g_kg: float,
    ssa_cm2g: float,
    humidity_g_kg: float,
) -> float:
    """
    Empirical correlation for r_drying [kg/s] — coating loss rate during drying.

    dMc/dt = -r_drying × (Mc / batch_size)  (order-1 attrition)

    Parameters
    ----------
    batch_size_kg : initial batch mass [kg]
    dm_ratio_g_kg : dry coating mass target / batch mass × 1000 [g/kg]
    ssa_cm2g      : specific surface area of particles [cm2/g]
    humidity_g_kg : measured absolute inlet air humidity [g/kg dry air]
                    (use the measured value from ph_humidity_g_kg, NOT the
                    coded DoE factor)

    Returns
    -------
    r_drying : float [kg/s]  — clipped to >= 0
    """
    bs_c  = batch_size_kg - _MEAN_BATCH_SIZE
    dm_c  = dm_ratio_g_kg - _MEAN_DM_RATIO
    ss_c  = ssa_cm2g      - _MEAN_SSA
    hu_c  = humidity_g_kg - _MEAN_HUMIDITY

    val_1e3 = (
        3.101623
        +  0.813251 * bs_c
        + -0.728002 * dm_c
        +  0.183477 * ss_c
        + -0.072031 * hu_c
        +  0.033859 * dm_c * hu_c
    )
    return max(float(val_1e3) * 1e-3, 0.0)
