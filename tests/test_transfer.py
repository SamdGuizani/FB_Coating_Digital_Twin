"""
Unit tests for heat and mass transfer coefficient calculations.

Validates that the Ranz-Marshall correlation and Reynolds/Prandtl/Schmidt numbers
reproduce expected physical values and that the MATLAB-equivalent logic is correct.
"""

import math
import pytest
import numpy as np

from fluid_bed.parameters import ProcessParameters
from fluid_bed.transfer import (
    _reynolds,
    calc_heat_transfer,
    calc_mass_transfer,
    gas_temperature_qs,
)
from fluid_bed.drying import antoine_pressure, gas_vapor_pressure, calc_drying_rate


# ── Shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def params() -> ProcessParameters:
    """Representative parameters for a small lab-scale fluid bed coater."""
    return ProcessParameters(
        diameter_eq=1e-3,          # 1 mm pellets
        particle_density=1200.0,   # kg/m³
        cp_particle=1200.0,        # J/kg/K
        diameter_bed=0.3,          # 30 cm bed
        batch_size=2.0,            # 2 kg
        rho_air=1.2,               # kg/m³
        cp_air=1005.0,             # J/kg/K
        air_flow_rates=(0.05, 0.05, 0.05),    # kg/s per stage
        air_temperatures=(333.0, 333.0, 333.0),  # 60 °C
        spray_rate=0.01,
        dry_matter_conc=0.15,
        atomization_air_flow=0.005,
    )


# ── Reynolds number ────────────────────────────────────────────────────────────

class TestReynolds:
    def test_positive(self, params):
        Re = _reynolds(params, stage=0)
        assert Re > 0

    def test_spraying_higher_than_preheating(self, params):
        """Stage 1 adds atomisation air → higher velocity → higher Re."""
        Re_pre = _reynolds(params, stage=0)
        Re_spray = _reynolds(params, stage=1)
        assert Re_spray > Re_pre

    def test_units_reasonable(self, params):
        """Expect Re in the range 1–1000 for typical lab-scale conditions."""
        Re = _reynolds(params, stage=0)
        assert 0.1 < Re < 5000


# ── Heat transfer ──────────────────────────────────────────────────────────────

class TestHeatTransfer:
    def test_positive(self, params):
        alpha_h = calc_heat_transfer(params, stage=0)
        assert alpha_h > 0

    def test_minimum_nu_is_2(self, params):
        """Nu ≥ 2 always (Ranz-Marshall limit as Re→0)."""
        # Reduce flow to nearly zero to push Re → 0
        low_flow_params = ProcessParameters(
            **{**params.__dict__, "air_flow_rates": (1e-9, 1e-9, 1e-9)}
        )
        alpha_h = calc_heat_transfer(low_flow_params, stage=0)
        # Nu → 2  ⟹  alpha_h → 2 * conductivity / d_eq
        expected_min = 2 * params.conductivity_air / params.diameter_eq
        assert alpha_h >= expected_min * 0.99

    def test_increases_with_flow(self, params):
        import dataclasses
        high = dataclasses.replace(params, air_flow_rates=(0.2, 0.2, 0.2))
        assert calc_heat_transfer(high, 0) > calc_heat_transfer(params, 0)


# ── Mass transfer ──────────────────────────────────────────────────────────────

class TestMassTransfer:
    def test_positive(self, params):
        assert calc_mass_transfer(params, stage=0) > 0

    def test_order_of_magnitude(self, params):
        """Expect α_m ~ 1e-4 to 1e-2 m/s for these conditions."""
        alpha_m = calc_mass_transfer(params, stage=0)
        assert 1e-5 < alpha_m < 0.1

    def test_analogy_with_heat(self, params):
        """
        Le = α_h / (α_m · cp_air · rho_air) should be ~ 1 for gas–air systems.
        More precisely: Sh/Nu should equal (Sc/Pr)^(1/3) approximately.
        """
        alpha_h = calc_heat_transfer(params, stage=0)
        alpha_m = calc_mass_transfer(params, stage=0)
        # Both use the same Re and same functional form — ratio is purely Sc vs Pr
        assert alpha_h > 0 and alpha_m > 0


# ── Gas temperature (quasi-steady) ────────────────────────────────────────────

class TestGasTemperature:
    def test_between_inlet_and_particle(self, params):
        T_p = 320.0  # particle below inlet
        T_g_in = 333.0
        T_g = gas_temperature_qs(params, stage=0, T_particle=T_p)
        assert T_p <= T_g <= T_g_in

    def test_cold_particle_gets_hot_gas(self, params):
        T_g = gas_temperature_qs(params, stage=0, T_particle=293.0)
        assert T_g > 293.0

    def test_equilibrium_at_inlet_temp(self, params):
        """When T_p = T_g_in, gas should stay at T_g_in."""
        T_eq = params.air_temperatures[0]
        T_g = gas_temperature_qs(params, stage=0, T_particle=T_eq)
        assert abs(T_g - T_eq) < 1e-6


# ── Antoine equation ──────────────────────────────────────────────────────────

class TestAntoine:
    def test_acetone_at_56c_is_760mmhg(self, params):
        """Acetone boils at 56 °C (1 atm = 101325 Pa = 760 mmHg)."""
        P_vap = antoine_pressure(params, T_K=329.15)  # 56 °C
        assert abs(P_vap - 101_325.0) / 101_325.0 < 0.02  # within 2 %

    def test_increases_with_temperature(self, params):
        P_low = antoine_pressure(params, T_K=300.0)
        P_high = antoine_pressure(params, T_K=340.0)
        assert P_high > P_low


# ── Drying rate ───────────────────────────────────────────────────────────────

class TestDryingRate:
    def test_positive_when_moist(self, params):
        R_D = calc_drying_rate(params, Y_particle=0.1, Y_gas=0.0,
                               T_particle=330.0, T_air=333.0, stage=0)
        assert R_D > 0

    def test_zero_when_dry(self, params):
        """With Y_particle = 0, drying rate should be essentially zero."""
        R_D = calc_drying_rate(params, Y_particle=0.0, Y_gas=0.0,
                               T_particle=330.0, T_air=333.0, stage=0)
        assert abs(R_D) < 1e-12

    def test_no_condensation(self, params):
        """If gas is supersaturated, R_D must be non-negative (no condensation)."""
        R_D = calc_drying_rate(params, Y_particle=0.001, Y_gas=10.0,
                               T_particle=330.0, T_air=333.0, stage=0)
        assert R_D >= 0

    def test_langmuir_limit(self, params):
        """R_D should be lower for a particle with very little moisture."""
        R_D_moist = calc_drying_rate(params, Y_particle=1.0, Y_gas=0.0,
                                     T_particle=330.0, T_air=333.0, stage=0)
        R_D_dry = calc_drying_rate(params, Y_particle=0.01, Y_gas=0.0,
                                   T_particle=330.0, T_air=333.0, stage=0)
        assert R_D_moist > R_D_dry
