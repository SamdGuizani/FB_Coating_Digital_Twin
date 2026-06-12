"""
Shared rig-level configuration — single source of truth for constants that were
previously duplicated across app.py, notebook 05b and the coefficient-
optimisation scripts.

Two kinds of values live here:

- Equipment / material constants of the Bosch Solidlab coater used in the
  2018 DoE campaign (air properties, bed geometry, particle material).
- Calibration constants of the first-order dissolution model (§5 of
  MODELLING_BACKGROUND_README.md), shared by the digital-twin front-ends and
  the inverse-modelling pipeline (step 02).

Per-run operating parameters (batch size, flows, temperatures, spray rate…)
do NOT belong here — they live in :class:`fluid_bed.parameters.ProcessParameters`,
one instance per run.
"""

from __future__ import annotations

from .parameters import ProcessParameters

# ── Equipment / material constants (Bosch Solidlab coater, DoE 2018) ──────────
RHO_AIR = 1.10        # kg/m³, process air density
CP_AIR = 1010.0       # J/kg/K, process air specific heat
RHO_PARTICLE = 1050.0  # kg/m³, particle density
CP_PARTICLE = 1400.0   # J/kg/K, dry-particle specific heat
D_BED = 0.60          # m, bed column diameter

# ── Dissolution model calibration (first-order permeation model) ──────────────
DISSOLUTION = dict(
    Volume_disso=1000.0,   # mL, dissolution medium volume
    rho_EC=0.4,            # g/cm³, ethylcellulose film density
    Permeability=1.5e-7,   # cm²/s, EC film permeability
    Mass_sample=1.058,     # g, dissolution sample mass
    Total_min=240,         # min, dissolution test duration
)


def default_coater_params(**overrides) -> ProcessParameters:
    """
    Base :class:`ProcessParameters` pre-filled with the rig-level constants
    above. Per-run values (diameter_eq, ssa_cm2_g, batch_size, flows,
    temperatures, spray settings…) are passed as keyword overrides.

    Example
    -------
    >>> p = default_coater_params(diameter_eq=1.0e-3, ssa_cm2_g=65.0,
    ...                           batch_size=4.6, spray_rate=0.002)
    """
    base = dict(
        diameter_eq=1.0e-3,          # m, typical DoE particle
        particle_density=RHO_PARTICLE,
        cp_particle=CP_PARTICLE,
        diameter_bed=D_BED,
        batch_size=4.6,              # kg, DoE centre point
        rho_air=RHO_AIR,
        cp_air=CP_AIR,
    )
    base.update(overrides)
    return ProcessParameters(**base)


def wg_max_noloss(qty_sol_kg: float, dmc_frac: float, batch_kg: float) -> float:
    """
    Theoretical maximum coating weight gain [%] assuming 100 % deposition
    efficiency (r_spraying = 0) and no attrition (r_drying = 0).

    WG_max = m_solution · x_DM / M_batch × 100
    """
    return qty_sol_kg * dmc_frac / batch_kg * 100.0
