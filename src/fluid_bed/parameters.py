"""Central parameter container replacing MATLAB global variables."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


@dataclass
class ProcessParameters:
    """
    All physical and operating parameters for the fluid bed coating model.

    Replaces MATLAB global variables. One instance per DoE run.

    Units: SI throughout (m, kg, s, K, Pa, J).
    Stage indices: 0 = pre-heating, 1 = spraying, 2 = drying.
    """

    # ── Particle ──────────────────────────────────────────────────────────────
    diameter_eq: float          # m, equivalent particle diameter
    particle_density: float     # kg/m³, particle bulk density
    cp_particle: float          # J/kg/K, dry-particle specific heat

    # ── Bed geometry ──────────────────────────────────────────────────────────
    diameter_bed: float         # m, bed column diameter

    # ── Batch ─────────────────────────────────────────────────────────────────
    batch_size: float           # kg, total dry batch mass

    # ── Air properties ────────────────────────────────────────────────────────
    rho_air: float              # kg/m³
    cp_air: float               # J/kg/K
    viscosity_air: float = 1.94e-5    # Pa·s  (hardcoded in MATLAB)
    conductivity_air: float = 0.0262  # W/m/K (hardcoded in MATLAB)
    mw_air: float = 0.029             # kg/mol

    # ── Solvent (acetone) ─────────────────────────────────────────────────────
    mw_acetone: float = 0.058           # kg/mol
    diffusivity_acetone: float = 1e-6   # m²/s  (TODO: validate, flagged in MATLAB)
    latent_heat_acetone: float = 518_000.0  # J/kg, at ~20 °C
    cp_acetone_liquid: float = 2160.0   # J/kg/K
    # Antoine equation: log10(P [mmHg]) = A − B / (C + T [°C])
    antoine_A: float = 7.1327
    antoine_B: float = 1219.97
    antoine_C: float = 230.653

    # ── Operating parameters per stage ────────────────────────────────────────
    # Index 0 = pre-heating, 1 = spraying, 2 = drying
    air_flow_rates: tuple[float, float, float] = (0.0, 0.0, 0.0)       # kg/s, process air
    air_temperatures: tuple[float, float, float] = (333.0, 333.0, 333.0)  # K, inlet
    air_inlet_moisture: tuple[float, float, float] = (0.0, 0.0, 0.0)   # kg/kg dry basis

    # ── Spray parameters ──────────────────────────────────────────────────────
    spray_rate: float = 0.0            # kg/s, total solution
    dry_matter_conc: float = 0.15      # fraction (0–1), EC concentration
    atomization_air_flow: float = 0.0  # kg/s, added to stage-1 air flow

    # ── System ────────────────────────────────────────────────────────────────
    pressure: float = 101_325.0    # Pa
    langmuir_alpha: float = 0.05   # Langmuir isotherm tuning constant
    gas_constant: float = 8.314    # J/mol/K
    gas_residence_time: float = 0.5  # s, mean gas residence time in bed (for Y_g dynamics)

    # ── Coating material ──────────────────────────────────────────────────────
    cp_coating: float = 1300.0     # J/kg/K, e.g. ethylcellulose

    # ── Coating deposition efficiency (spraying stage) ────────────────────────
    # dMc/dt = spray_rate·DMC  −  r_spraying
    # Constant (order-0) loss; accounts for attrition and spray losses.
    # r_spraying=0 → 100% efficiency
    r_spraying: float = 6.72e-6    # kg/s, constant coating loss rate

    # ── Coating attrition during drying ───────────────────────────────────────
    # dMc/dt = -r_drying · (Mc/batch)^n_drying
    # r_drying=0 → no attrition during drying
    r_drying: float = 3.19e-3      # kg/s, attrition rate constant
    n_drying: float = 1.0          # order (1 = loss ∝ normalised coating load)

    # ── Specific surface area override ────────────────────────────────────────
    # When > 0, total_surface uses the BET/mercury-intrusion SSA instead of the
    # geometric sphere formula.  Allows diameter (for Re) and surface (for
    # heat/mass transfer) to be set independently for non-ideal particles.
    ssa_cm2_g: float = 0.0         # cm²/g; 0 = use geometric sphere formula

    # ─────────────────────────────────────────────────────────────────────────
    # Computed properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def mass_particle(self) -> float:
        """Single-particle mass [kg]."""
        return self.particle_density * (np.pi / 6) * self.diameter_eq ** 3

    @property
    def surface_particle(self) -> float:
        """Single-particle surface area [m²]."""
        return np.pi * self.diameter_eq ** 2

    @property
    def n_particles(self) -> int:
        """Number of particles in the bed."""
        return max(1, int(self.batch_size / self.mass_particle))

    @property
    def cross_section_bed(self) -> float:
        """Bed column cross-sectional area [m²]."""
        return (np.pi / 4) * self.diameter_bed ** 2

    @property
    def total_surface(self) -> float:
        """
        Total particle surface area in bed [m²].

        Uses SSA (ssa_cm2_g) when set, falling back to the geometric sphere
        formula otherwise.  1 cm²/g = 0.1 m²/kg.
        """
        if self.ssa_cm2_g > 0:
            return self.ssa_cm2_g * 0.1 * self.batch_size
        return self.n_particles * self.surface_particle

    def stage_air_flow(self, stage: int) -> float:
        """
        Total air flow rate for a given stage [kg/s].

        Stage 1 (spraying) includes atomization air, matching MATLAB logic.
        """
        if stage == 1:
            return self.air_flow_rates[1] + self.atomization_air_flow
        return self.air_flow_rates[stage]

    def specific_gas_constant_acetone(self) -> float:
        """R / MW_acetone [J/kg/K]."""
        return self.gas_constant / self.mw_acetone


def from_doe_run(row: dict, base: ProcessParameters) -> ProcessParameters:
    """
    Build a ProcessParameters by overriding base values with a DoE run's
    operating parameters.

    `row` is a dict with keys matching column names in the DoE CSV files.
    Expected keys (adapt to actual CSV columns):
        'Inlet_T_preheating_K', 'Inlet_T_spraying_K', 'Inlet_T_drying_K',
        'Flow_preheating_kgs', 'Flow_spraying_kgs', 'Flow_drying_kgs',
        'Atomization_air_kgs', 'Spray_rate_kgs', 'EC_conc_fraction',
        'Batch_size_kg'
    """
    import dataclasses
    overrides = {}

    if "Batch_size_kg" in row:
        overrides["batch_size"] = float(row["Batch_size_kg"])
    if all(k in row for k in ("Flow_preheating_kgs", "Flow_spraying_kgs", "Flow_drying_kgs")):
        overrides["air_flow_rates"] = (
            float(row["Flow_preheating_kgs"]),
            float(row["Flow_spraying_kgs"]),
            float(row["Flow_drying_kgs"]),
        )
    if all(k in row for k in ("Inlet_T_preheating_K", "Inlet_T_spraying_K", "Inlet_T_drying_K")):
        overrides["air_temperatures"] = (
            float(row["Inlet_T_preheating_K"]),
            float(row["Inlet_T_spraying_K"]),
            float(row["Inlet_T_drying_K"]),
        )
    if "Atomization_air_kgs" in row:
        overrides["atomization_air_flow"] = float(row["Atomization_air_kgs"])
    if "Spray_rate_kgs" in row:
        overrides["spray_rate"] = float(row["Spray_rate_kgs"])
    if "EC_conc_fraction" in row:
        overrides["dry_matter_conc"] = float(row["EC_conc_fraction"])

    return dataclasses.replace(base, **overrides)
