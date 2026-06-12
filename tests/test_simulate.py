"""Tests for fluid_bed.config and fluid_bed.simulate (full-process orchestration)."""

import numpy as np
import pytest

from fluid_bed.config import (
    DISSOLUTION,
    RHO_AIR,
    default_coater_params,
    wg_max_noloss,
)
from fluid_bed.models.dissolution import dissolution_curve, dissolution_k
from fluid_bed.simulate import run_full_process


# Default app/notebook slider values, short durations to keep the test fast
FAST_KWARGS = dict(
    d_mm=1.00, ssa_cm2g=65.0, T0_C=20.0, batch_kg=4.6,
    humidity_g_kg=13.4, dmc_pct=1.5, coating_level=0.5,
    ph_T_C=50.0, ph_flow_m3h=250, ph_dur_min=2.0,
    sp_T_C=40.0, sp_flow_m3h=250, sp_rate_g_min=120,
    dr_T_C=40.0, dr_flow_m3h=250, dr_dur_min=2.0,
)


@pytest.fixture(scope="module")
def result():
    return run_full_process(**FAST_KWARGS)


class TestConfig:
    def test_default_coater_params_constants(self):
        p = default_coater_params()
        assert p.rho_air == RHO_AIR
        assert p.cp_air == 1010.0
        assert p.particle_density == 1050.0
        assert p.cp_particle == 1400.0
        assert p.diameter_bed == 0.60

    def test_default_coater_params_overrides(self):
        p = default_coater_params(batch_size=3.0, ssa_cm2_g=70.0)
        assert p.batch_size == 3.0
        assert p.ssa_cm2_g == 70.0
        assert p.rho_air == RHO_AIR  # constants untouched by overrides

    def test_wg_max_noloss(self):
        # 1 kg solution at 1.5 % DM on a 5 kg batch -> 0.3 % WG
        assert wg_max_noloss(1.0, 0.015, 5.0) == pytest.approx(0.3)


class TestDissolutionLink:
    def test_k_decreases_with_coating(self):
        assert dissolution_k(0.004, 65.0) > dissolution_k(0.008, 65.0)

    def test_k_inverts_exactly(self):
        # k(x_EC) * x_EC must equal the constant numerator
        x_ec = 0.003
        ssa = 65.0
        num = (DISSOLUTION["Mass_sample"] * ssa ** 2 * DISSOLUTION["Permeability"]
               * DISSOLUTION["rho_EC"] / DISSOLUTION["Volume_disso"])
        assert dissolution_k(x_ec, ssa) * x_ec == pytest.approx(num)

    def test_zero_coating_instant_release(self):
        t, F, k = dissolution_curve(0.0, 65.0)
        assert np.isinf(k)
        assert np.all(F == 100.0)

    def test_curve_shape(self):
        t, F, k = dissolution_curve(0.003, 65.0)
        assert len(t) == DISSOLUTION["Total_min"]
        assert np.all(np.diff(F) >= 0)          # monotonic release
        assert 0.0 < F[0] and F[-1] <= 100.0


class TestRunFullProcess:
    def test_stages_succeed(self, result):
        assert result.preheating.success
        assert result.spraying.success
        assert result.drying.success

    def test_state_chaining(self, result):
        # Each stage starts where the previous one ended
        assert result.spraying.T_particle[0] == pytest.approx(
            result.preheating.T_particle[-1])
        assert result.drying.T_particle[0] == pytest.approx(
            result.spraying.T_particle[-1], rel=1e-6)
        assert result.drying.M_coating[0] == pytest.approx(
            result.spraying.M_coating[-1], rel=1e-6)

    def test_concatenated_series_consistent(self, result):
        n = len(result.t_all)
        for name in ("T_product", "T_gas", "Y_particle", "Y_gas", "WG", "WG_noloss"):
            assert len(getattr(result, name)) == n, name
        assert np.all(np.diff(result.t_all) >= 0)
        assert 0 < result.ph_end < result.sp_end < result.t_end
        assert result.t_end == pytest.approx(result.t_all[-1])

    def test_wg_physical(self, result):
        assert np.all(result.WG >= 0)
        # losses: model WG never exceeds the no-loss bound (small solver tolerance)
        assert result.wg_end_spray <= result.wg_max_noloss * (1 + 1e-6)
        # order-1 attrition during drying can only reduce the coating
        assert result.wg_final <= result.wg_end_spray * (1 + 1e-9)
        assert result.wg_end_spray == pytest.approx(
            result.spraying.M_coating[-1] / FAST_KWARGS["batch_kg"] * 100)

    def test_preheating_heats_bed(self, result):
        assert result.T_product[0] == pytest.approx(FAST_KWARGS["T0_C"])
        assert result.preheating.T_particle[-1] > result.preheating.T_particle[0]

    def test_loss_rates_positive(self, result):
        assert result.r_spraying >= 0
        assert result.r_drying >= 0
