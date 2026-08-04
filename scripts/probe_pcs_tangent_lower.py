# PCS 접선하한 128면 과소계상률과 편입 전 solve_avg 동시충방전 잔차를 측정한다.
# 본코드는 수정하지 않으며 수치만 출력하고 자동 판정하지 않는다.

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import loss_coeffs
import params as PM
from lower_lp import _get_problem, _get_topology, solve_avg


_COEFF_NAMES = ("a_P", "a_Q", "b_PP", "b_QQ", "b_PQ")
_S_CASES = (0.05, 0.176, 0.5, 1.0, 2.4)
_OPERATING_CASES = ((0.176, 0.42), (0.5, 1.2), (1.0, 2.4))
_BUS = 15


def _polygon_lower_bound(p_mw, q_mvar, cos_theta, sin_theta):
    """Return max_k(P cos(theta_k) + Q sin(theta_k))."""
    projections = (
        p_mw[..., None] * cos_theta[None, None, :]
        + q_mvar[..., None] * sin_theta[None, None, :]
    )
    return np.max(projections, axis=-1)


def part1_tangent_geometry():
    theta = 2.0 * np.pi * np.arange(PM.POLY_N, dtype=float) / PM.POLY_N
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    phi = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)

    all_rel = []
    all_pcs_rel = []

    print("PART1")
    print(
        "S_MVA,grid_points,max_rel_err_pct,median_rel_err_pct,"
        "max_abs_err_MW,u_gt_true_count,max_pcs_under_pct"
    )
    for S in _S_CASES:
        # "60 divisions" includes both radial endpoints: 61 radii.
        radius = np.linspace(0.0, float(S), 61)
        p_mw = radius[:, None] * np.cos(phi)[None, :]
        q_mvar = radius[:, None] * np.sin(phi)[None, :]
        true_norm = np.hypot(p_mw, q_mvar)
        lower_norm = _polygon_lower_bound(
            p_mw, q_mvar, cos_theta, sin_theta
        )
        abs_err = true_norm - lower_norm

        nonzero = true_norm > 1e-9
        rel_err = abs_err[nonzero] / true_norm[nonzero]

        true_pcs_loss = (1.0 - PM.ETA_PCS) * (
            true_norm - np.abs(p_mw)
        )
        lower_pcs_loss = (1.0 - PM.ETA_PCS) * (
            lower_norm - np.abs(p_mw)
        )
        pcs_mask = (true_norm - np.abs(p_mw)) > 1e-9
        pcs_rel = (
            (true_pcs_loss[pcs_mask] - lower_pcs_loss[pcs_mask])
            / true_pcs_loss[pcs_mask]
        )

        # A scale-aware tolerance separates floating-point roundoff from a
        # genuine violation of the support-function lower-bound property.
        exceed_tol = 64.0 * np.finfo(float).eps * max(1.0, float(S))
        exceed_count = int(np.count_nonzero(lower_norm > true_norm + exceed_tol))

        all_rel.append(rel_err)
        all_pcs_rel.append(pcs_rel)
        print(
            f"{S:.6g},{true_norm.size:d},{100.0 * np.max(rel_err):.12g},"
            f"{100.0 * np.median(rel_err):.12g},{np.max(abs_err):.12g},"
            f"{exceed_count:d},{100.0 * np.max(pcs_rel):.12g}"
        )

    combined_rel = np.concatenate(all_rel)
    combined_pcs_rel = np.concatenate(all_pcs_rel)
    print("all_max_rel_err_pct,all_max_pcs_under_pct")
    print(
        f"{100.0 * np.max(combined_rel):.12g},"
        f"{100.0 * np.max(combined_pcs_rel):.12g}"
    )


def _assemble_coeffs_1unit(grid_result, scenario):
    """Same scenario-slice assembly as evaluate._assemble_coeffs_1unit."""
    scenarios = tuple(grid_result["scenarios"])
    try:
        scenario_index = scenarios.index(str(scenario))
    except ValueError as exc:
        raise KeyError(
            f"scenario={scenario!r} not in coefficient grid"
        ) from exc
    return {
        name: np.ascontiguousarray(
            np.asarray(
                grid_result[name][scenario_index, :], dtype=np.float64
            ).reshape(1, PM.TIME_STEPS)
        )
        for name in _COEFF_NAMES
    }


def _solved_entry(coeffs):
    """Retrieve the cache entry populated by the immediately preceding solve."""
    return _get_problem(
        "avg",
        n=1,
        force_q_zero=False,
        eta_c=PM.ETA_C,
        eta_d=PM.ETA_D,
        self_discharge=PM.SELF_DISCHARGE_HOURLY,
        soc_init=PM.SOC_INIT_FRAC,
        soc_min=PM.SOC_MIN_FRAC,
        soc_max=PM.SOC_MAX_FRAC,
        mu_volt=PM.MU_VOLT,
        coeffs=coeffs,
    )


def part2_simultaneous_charge_discharge():
    # Touching topology here verifies the requested internal inspection path
    # without changing or rebuilding the optimization problem.
    _get_topology()

    grids = {
        S: loss_coeffs.measure_coeffs_grid(
            _BUS,
            S,
            scenarios=PM.AVG_DAYS,
            times=range(PM.TIME_STEPS),
            p_center=0.0,
        )
        for S, _ in _OPERATING_CASES
    }

    global_max_simul = 0.0
    global_max_abs_gap = 0.0

    print("PART2")
    print(
        "bus,S_MVA,E_MWh,scenario,max_simul_MW,sum_simul_MW,"
        "max_abs_absPnet_minus_sum_MW,pnet_reconstruction_err_MW"
    )
    for S, E in _OPERATING_CASES:
        for scenario in PM.AVG_DAYS:
            coeffs = _assemble_coeffs_1unit(grids[S], scenario)
            p_net, _q, _soc = solve_avg(
                S,
                E,
                _BUS,
                smp=np.asarray(PM.SMP_PER_MWH[scenario], dtype=float),
                profile=np.asarray(PM.LOAD[scenario], dtype=float),
                coeffs=coeffs,
            )

            entry = _solved_entry(coeffs)
            p_ch = np.asarray(
                entry["vars"]["P_ch"].value, dtype=float
            ).reshape(1, PM.TIME_STEPS)[0]
            p_dis = np.asarray(
                entry["vars"]["P_dis"].value, dtype=float
            ).reshape(1, PM.TIME_STEPS)[0]
            p_net_returned = np.asarray(
                p_net, dtype=float
            ).reshape(1, PM.TIME_STEPS)[0]

            simul = np.minimum(p_ch, p_dis)
            max_simul = float(np.max(simul))
            sum_simul = float(np.sum(simul))
            reconstructed = p_dis - p_ch
            reconstruction_err = float(
                np.max(np.abs(reconstructed - p_net_returned))
            )
            abs_gap = float(
                np.max(np.abs((p_ch + p_dis) - np.abs(reconstructed)))
            )

            global_max_simul = max(global_max_simul, max_simul)
            global_max_abs_gap = max(global_max_abs_gap, abs_gap)
            print(
                f"{_BUS:d},{S:.6g},{E:.6g},{scenario},"
                f"{max_simul:.12e},{sum_simul:.12e},{abs_gap:.12e},"
                f"{reconstruction_err:.12e}"
            )

    print("all_max_simul_MW,all_max_abs_absPnet_minus_sum_MW")
    print(f"{global_max_simul:.12e},{global_max_abs_gap:.12e}")


def main():
    part1_tangent_geometry()
    part2_simultaneous_charge_discharge()


if __name__ == "__main__":
    main()
