"""User-run validation for loss_coeffs.py.

WARNING: stages 3 and 4 perform many pandapower runpp calls and can take a
long time.  Run this script directly from the configured ``ess`` environment.
"""

from __future__ import annotations

import inspect
import os
import sys
import time
from pathlib import Path


os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import loss_coeffs as LC
import params as PM
from build_net import build_net


BUS = 15
S_MVA = 0.2
SCENARIO = "summer_peak"
PEAK_T = int(np.argmax(np.asarray(PM.LOAD[SCENARIO], dtype=float)))


def _stage(number: int | str, title: str) -> None:
    print(f"\n[stage] {number}. {title}", flush=True)


def _model_l_cost(result: dict, p_mw: float, q_mvar: float) -> float:
    return float(
        result["a_P"] * p_mw
        + result["a_Q"] * q_mvar
        + result["b_PP"] * p_mw**2
        + result["b_QQ"] * q_mvar**2
        + result["b_PQ"] * p_mw * q_mvar
    )


def main() -> int:
    started = time.perf_counter()
    print(
        "warning=many_runpp_calls stages_3_and_4_may_take_a_long_time",
        flush=True,
    )
    print(
        f"test_point bus={BUS} S={S_MVA} scenario={SCENARIO} t={PEAK_T}",
        flush=True,
    )

    _stage(0, "configured local measurement grid")
    local_grid = LC._local_grid(S_MVA, 0.5 * S_MVA)
    expected_count = {"min7": 7, "full25": 25}[
        str(PM.LOSS_GRID_DESIGN).lower()
    ]
    print(f"loss_grid_design={PM.LOSS_GRID_DESIGN}", flush=True)
    print(f"grid_sample_count_excluding_baseline={len(local_grid)}", flush=True)
    print(f"grid_points={local_grid}", flush=True)
    assert len(local_grid) == expected_count, (
        PM.LOSS_GRID_DESIGN,
        len(local_grid),
        expected_count,
    )
    assert all(
        np.hypot(p_mw, q_mvar) <= S_MVA + LC.GRID_TOL
        for p_mw, q_mvar in local_grid
    )

    if str(PM.LOSS_GRID_DESIGN).lower() == "min7":
        _stage("0b", "min7 PCS-boundary pull regression")
        boundary_grid = LC._local_grid(S_MVA, S_MVA)
        boundary_design = np.asarray(
            [[p, q, p * p, q * q, p * q] for p, q in boundary_grid],
            dtype=float,
        )
        boundary_scale = np.max(np.abs(boundary_design), axis=0)
        boundary_scale[boundary_scale == 0.0] = 1.0
        boundary_grid_rank = int(
            np.linalg.matrix_rank(boundary_design / boundary_scale)
        )
        expected_center = PM.LOSS_PULL_TARGET * S_MVA
        print(f"boundary_grid_points={boundary_grid}", flush=True)
        print(f"boundary_grid_rank={boundary_grid_rank}", flush=True)
        print(f"boundary_pull_expected_center_mw={expected_center:.12g}", flush=True)
        assert len(boundary_grid) == 7, len(boundary_grid)
        assert boundary_grid_rank == 5, boundary_grid_rank
        assert any(abs(q_mvar) > 0.0 for _, q_mvar in boundary_grid)

        boundary_result = LC.measure_coeffs(
            bus=BUS,
            S=S_MVA,
            scenario=SCENARIO,
            t=PEAK_T,
            p_center=S_MVA,
            strict=True,
        )
        boundary_fit_rel_baseline = (
            boundary_result["fit_max_abs_error_mw"]
            / abs(boundary_result["baseline_loss_mw"])
        )
        print(f"boundary_fit_rank={boundary_result['fit_rank']}", flush=True)
        print(f"boundary_a_Q={boundary_result['a_Q']:.12g}", flush=True)
        print(
            f"boundary_h_cost_min_eig={boundary_result['h_cost_min_eig']:.12g}",
            flush=True,
        )
        print(
            f"boundary_fit_rel_baseline={boundary_fit_rel_baseline:.12g}",
            flush=True,
        )
        assert boundary_result["fit_rank"] == 5, boundary_result["fit_rank"]
        assert boundary_result["a_Q"] < 0.0, boundary_result["a_Q"]
        assert boundary_result["h_cost_psd"], boundary_result["h_cost_min_eig"]
        assert boundary_fit_rel_baseline <= 5e-4, boundary_fit_rel_baseline
        # TODO: 정확한 min7+pull 5계수 회귀값은 데스크탑 실행 결과로 확정한다.

    _stage(1, "coefficient sign, PSD, fit, and units")
    result = LC.measure_coeffs(
        bus=BUS,
        S=S_MVA,
        scenario=SCENARIO,
        t=PEAK_T,
        p_center=0.0,
        strict=True,
    )
    for name in LC.COEFF_NAMES:
        print(f"{name}={result[name]:.12g}", flush=True)
    print(f"h_cost_min_eig={result['h_cost_min_eig']:.12g}", flush=True)
    print(f"fit_residual={result['fit_residual']:.12g}", flush=True)
    print(f"fit_rmse_mw={result['fit_rmse_mw']:.12g}", flush=True)
    print(f"fit_rank={result['fit_rank']}", flush=True)
    print(
        f"runpp_attempts={result['runpp_attempts']} "
        f"retries={result['runpp_retries']} "
        f"failures={result['runpp_failures']}",
        flush=True,
    )

    direct = LC.measure_loss_reduction(
        bus=BUS,
        scenario=SCENARIO,
        t=PEAK_T,
        p_mw=0.1,
        q_mvar=0.0,
    )
    measured_l_cost = direct["L_cost_mw"]
    modeled_l_cost = _model_l_cost(result, 0.1, 0.0)
    # TODO(min7 기준 재측정): 이 출력에 수치 기대값/허용오차를 추가하려면
    # 데스크탑 AC 검증 결과로 확정한다. 여기서 새 임계를 창작하지 않는다.
    print(f"direct_L_cost_p0.1_q0_mw={measured_l_cost:.12g}", flush=True)
    print(f"model_L_cost_p0.1_q0_mw={modeled_l_cost:.12g}", flush=True)
    print(
        f"direct_minus_model_mw={measured_l_cost - modeled_l_cost:.12g}",
        flush=True,
    )

    _stage(2, "delta-loss physical direction")
    direction_net = build_net()
    direction_rows = []
    for p_mw in np.linspace(0.0, S_MVA, 5):
        point = LC.measure_loss_reduction(
            bus=BUS,
            scenario=SCENARIO,
            t=PEAK_T,
            p_mw=float(p_mw),
            q_mvar=0.0,
            net=direction_net,
        )
        delta = point["delta_loss_reduction_mw"]
        direction_rows.append(delta)
        print(
            f"P_mw={p_mw:.12g} "
            f"delta_loss_reduction_mw={delta:.12g}",
            flush=True,
        )
    print(
        "delta_loss_monotone_nondecreasing="
        f"{bool(np.all(np.diff(direction_rows) >= -1e-10))}",
        flush=True,
    )

    _stage(3, "24-hour coefficient grid and fit stability")
    grid = LC.measure_coeffs_grid(
        bus=BUS,
        S=S_MVA,
        scenarios=[SCENARIO],
        times=range(PM.TIME_STEPS),
        p_center=0.0,
        strict=True,
    )
    print(
        "a_P_24="
        + np.array2string(grid["a_P"][0], precision=9, separator=","),
        flush=True,
    )
    print(
        f"a_P_all_negative={bool(np.all(grid['a_P'] < 0.0))}",
        flush=True,
    )
    print(
        f"fit_residual_max={float(np.nanmax(grid['fit_residual'])):.12g}",
        flush=True,
    )
    print(
        f"rank_lt_5_count={int(np.count_nonzero(grid['fit_rank'] < 5))}",
        flush=True,
    )

    _stage(4, "strict=False sign and PSD survey")
    violation_locations = []
    survey_count = 0
    for bus in (6, 15, 30):
        for S in (0.1, 0.2, 0.5):
            survey = LC.measure_coeffs_grid(
                bus=bus,
                S=S,
                scenarios=PM.ALL_DAYS,
                times=range(PM.TIME_STEPS),
                p_center=0.0,
                strict=False,
            )
            for scenario_index, scenario in enumerate(PM.ALL_DAYS):
                for time_index, t in enumerate(range(PM.TIME_STEPS)):
                    survey_count += 1
                    bad_a_p = not bool(survey["a_P_negative"][scenario_index, time_index])
                    bad_a_q = not bool(survey["a_Q_negative"][scenario_index, time_index])
                    bad_psd = not bool(survey["h_cost_psd"][scenario_index, time_index])
                    bad_rank = not bool(survey["fit_rank_full"][scenario_index, time_index])
                    if bad_a_p or bad_a_q or bad_psd or bad_rank:
                        violation_locations.append(
                            {
                                "bus": bus,
                                "S": S,
                                "scenario": scenario,
                                "t": t,
                                "bad_a_P": bad_a_p,
                                "bad_a_Q": bad_a_q,
                                "bad_psd": bad_psd,
                                "bad_rank": bad_rank,
                                "a_P": survey["a_P"][scenario_index, time_index],
                                "a_Q": survey["a_Q"][scenario_index, time_index],
                                "lambda_min": survey["h_cost_min_eig"][
                                    scenario_index, time_index
                                ],
                                "rank": survey["fit_rank"][scenario_index, time_index],
                            }
                        )
    print(f"survey_case_count={survey_count}", flush=True)
    print(f"violation_count={len(violation_locations)}", flush=True)
    for location in violation_locations:
        print(f"violation={location}", flush=True)

    _stage(5, "E independence by API construction")
    signature = inspect.signature(LC.measure_coeffs)
    has_e_parameter = "E" in signature.parameters or "e" in signature.parameters
    print(f"measure_coeffs_signature={signature}", flush=True)
    print(f"has_E_parameter={has_e_parameter}", flush=True)
    print(f"E_independent_by_construction={not has_e_parameter}", flush=True)

    _stage(6, "process-local cache behavior")
    LC.clear_cache()
    before = LC.cache_info()
    first = LC.get_or_measure(
        BUS, S_MVA, SCENARIO, PEAK_T, p_center=0.0, strict=True
    )
    after_first = LC.cache_info()
    second = LC.get_or_measure(
        BUS, S_MVA, SCENARIO, PEAK_T, p_center=0.0, strict=True
    )
    after_second = LC.cache_info()
    shifted = LC.get_or_measure(
        BUS, S_MVA, SCENARIO, PEAK_T, p_center=0.01, strict=True
    )
    after_shifted = LC.cache_info()
    print(f"cache_before={before}", flush=True)
    print(f"cache_after_first={after_first}", flush=True)
    print(f"cache_after_second={after_second}", flush=True)
    print(f"cache_after_shifted_center={after_shifted}", flush=True)
    print(
        "second_call_runpp_increment="
        f"{after_second['runpp_attempts'] - after_first['runpp_attempts']}",
        flush=True,
    )
    print(
        "shifted_center_runpp_increment="
        f"{after_shifted['runpp_attempts'] - after_second['runpp_attempts']}",
        flush=True,
    )
    print(
        "cached_coefficients_equal="
        f"{all(first[name] == second[name] for name in LC.COEFF_NAMES)}",
        flush=True,
    )
    print(
        "shifted_center_key_is_distinct="
        f"{after_shifted['misses'] == after_second['misses'] + 1}",
        flush=True,
    )
    _ = shifted

    print(f"\nelapsed_s={time.perf_counter() - started:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
