"""User-run validation for evaluate.py coefficient recomputation.

This script performs many AC power-flow calls.  Run it directly in the
configured ``ess`` environment; it is intentionally not run by Codex.
"""

from __future__ import annotations

import os
import subprocess
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

import evaluate
import loss_coeffs
import lower_lp
import params as PM
from build_net import build_net


BUS = 15
S_MVA = 0.2
E_MWH = 0.42
AVG_SCENARIO = "summer"
BRANCH_SCENARIO = "summer_peak"


def _stage(number: int, title: str) -> None:
    print(f"\n[stage] {number}. {title}", flush=True)


def _avg_recompute(
    scenario: str,
    *,
    S: float = S_MVA,
    E: float = E_MWH,
    return_intermediates: bool = True,
):
    return evaluate._solve_with_recompute(
        "avg",
        BUS,
        S,
        E,
        scenario,
        smp=np.asarray(PM.SMP_PER_MWH[scenario], dtype=float),
        profile=np.asarray(PM.LOAD[scenario], dtype=float),
        return_intermediates=return_intermediates,
    )


def main() -> int:
    started = time.perf_counter()
    print(
        "warning=many_runpp_calls recompute_and_test_evaluate_may_take_a_long_time",
        flush=True,
    )

    _stage(1, "one-recompute convergence")
    avg = _avg_recompute(AVG_SCENARIO)
    X0, X1 = avg["X0"], avg["X1"]
    C2 = evaluate._measure_recomputed_coeffs_1unit(
        BUS,
        S_MVA,
        AVG_SCENARIO,
        X1["P_net"],
        net=build_net(),
    )
    P2, Q2, soc2 = lower_lp.solve_avg(
        S_MVA,
        E_MWH,
        BUS,
        np.asarray(PM.SMP_PER_MWH[AVG_SCENARIO], dtype=float),
        np.asarray(PM.LOAD[AVG_SCENARIO], dtype=float),
        C2,
        assert_physics=False,
    )
    delta_10 = float(np.max(np.abs(X1["P_net"] - X0["P_net"])))
    delta_21 = float(np.max(np.abs(P2 - X1["P_net"])))
    ratio = delta_21 / delta_10 if delta_10 > 0.0 else float("nan")
    print(f"max_abs_P_X1_minus_X0={delta_10:.12g}", flush=True)
    print(f"max_abs_P_X2_minus_X1={delta_21:.12g}", flush=True)
    print(f"convergence_ratio_X21_over_X10={ratio:.12g}", flush=True)
    print(f"expected_ratio_lt_0.05={bool(np.isfinite(ratio) and ratio < 0.05)}", flush=True)
    _ = Q2, soc2

    _stage(2, "AVG/PEAK second-coefficient branching")
    profile = np.asarray(PM.LOAD[BRANCH_SCENARIO], dtype=float)
    avg_branch = evaluate._solve_with_recompute(
        "avg",
        BUS,
        S_MVA,
        E_MWH,
        BRANCH_SCENARIO,
        smp=np.asarray(PM.SMP_PER_MWH[BRANCH_SCENARIO], dtype=float),
        profile=profile,
        return_intermediates=True,
    )
    peak_branch = evaluate._solve_with_recompute(
        "peak",
        BUS,
        S_MVA,
        E_MWH,
        BRANCH_SCENARIO,
        load_total=9.5 * profile,
        profile=profile,
        return_intermediates=True,
    )
    schedule_branch_diff = float(
        np.max(
            np.abs(
                avg_branch["X0"]["P_net"] - peak_branch["X0"]["P_net"]
            )
        )
    )
    coefficient_branch_diff = float(
        np.max(np.abs(avg_branch["C1"]["a_P"] - peak_branch["C1"]["a_P"]))
    )
    print(f"max_abs_X0_avg_minus_peak={schedule_branch_diff:.12g}", flush=True)
    print(f"max_abs_C1_a_P_avg_minus_peak={coefficient_branch_diff:.12g}", flush=True)
    print(f"schedule_branches_distinct={schedule_branch_diff > 0.0}", flush=True)
    print(f"coefficient_branches_distinct={coefficient_branch_diff > 0.0}", flush=True)

    _stage(3, "cache and runpp cost")
    loss_coeffs.clear_cache()
    before = loss_coeffs.cache_info()
    first = _avg_recompute(AVG_SCENARIO, E=E_MWH)
    after_first = loss_coeffs.cache_info()

    # Isolate the E-independent C0 cache hit: this repeated grid must add no runpp.
    loss_coeffs.measure_coeffs_grid(
        BUS,
        S_MVA,
        scenarios=[AVG_SCENARIO],
        times=range(PM.TIME_STEPS),
        p_center=0.0,
        strict=True,
    )
    after_c0_repeat = loss_coeffs.cache_info()

    second_E = _avg_recompute(AVG_SCENARIO, E=0.5)
    after_second_E = loss_coeffs.cache_info()

    loss_coeffs.measure_coeffs_grid(
        BUS,
        0.2001,
        scenarios=[AVG_SCENARIO],
        times=range(PM.TIME_STEPS),
        p_center=0.0,
        strict=True,
    )
    after_changed_S = loss_coeffs.cache_info()
    print(f"cache_before={before}", flush=True)
    print(f"cache_after_first_recompute={after_first}", flush=True)
    print(f"cache_after_repeated_C0={after_c0_repeat}", flush=True)
    print(f"cache_after_changed_E_recompute={after_second_E}", flush=True)
    print(f"cache_after_S_0.2001_C0={after_changed_S}", flush=True)
    print(
        "repeated_C0_runpp_increment="
        f"{after_c0_repeat['runpp_attempts'] - after_first['runpp_attempts']}",
        flush=True,
    )
    print(
        "changed_E_full_recompute_runpp_increment="
        f"{after_second_E['runpp_attempts'] - after_c0_repeat['runpp_attempts']}",
        flush=True,
    )
    print(
        "changed_S_C0_runpp_increment="
        f"{after_changed_S['runpp_attempts'] - after_second_E['runpp_attempts']}",
        flush=True,
    )
    _ = first, second_E

    _stage(4, "_normalize_coeffs one-call pipeline")
    T = PM.TIME_STEPS
    artificial = {
        "a_P": np.full((1, T), -0.01),
        "a_Q": np.full((1, T), -0.01),
        "b_PP": np.full((1, T), 0.02),
        "b_QQ": np.full((1, T), 0.02),
        "b_PQ": np.zeros((1, T)),
    }
    original_normalize = lower_lp._normalize_coeffs
    normalize_calls = {"count": 0}

    def counted_normalize(*args, **kwargs):
        normalize_calls["count"] += 1
        return original_normalize(*args, **kwargs)

    lower_lp._normalize_coeffs = counted_normalize
    try:
        solve_args = (
            S_MVA,
            E_MWH,
            BUS,
            np.asarray(PM.SMP_PER_MWH[AVG_SCENARIO], dtype=float),
            np.asarray(PM.LOAD[AVG_SCENARIO], dtype=float),
            artificial,
        )
        P_a, Q_a, soc_a = lower_lp.solve_avg(*solve_args)
        calls_after_first = normalize_calls["count"]
        P_b, Q_b, soc_b = lower_lp.solve_avg(*solve_args)
        calls_after_second = normalize_calls["count"]
    finally:
        lower_lp._normalize_coeffs = original_normalize
    print(f"normalize_calls_first_solve={calls_after_first}", flush=True)
    print(
        f"normalize_calls_second_solve_increment="
        f"{calls_after_second - calls_after_first}",
        flush=True,
    )
    print(f"P_bitwise_equal={np.array_equal(P_a, P_b)}", flush=True)
    print(f"Q_bitwise_equal={np.array_equal(Q_a, Q_b)}", flush=True)
    print(f"soc_bitwise_equal={np.array_equal(soc_a, soc_b)}", flush=True)
    print(f"max_abs_P_repeat_diff={np.max(np.abs(P_a - P_b)):.12g}", flush=True)
    print(f"max_abs_Q_repeat_diff={np.max(np.abs(Q_a - Q_b)):.12g}", flush=True)

    _stage(5, "final schedule AC and benefits path")
    evaluate.init_worker()
    detail = evaluate.evaluate_particle(
        np.asarray([BUS, S_MVA, E_MWH], dtype=float),
        return_detail=True,
    )
    print(f"diverged={detail.get('diverged')}", flush=True)
    print(f"fitness={detail.get('fitness')}", flush=True)
    print(f"j_net={detail.get('j_net')}", flush=True)
    print(f"j_net_finite={bool(np.isfinite(detail.get('j_net', np.nan)))}", flush=True)
    print(f"decomposition_ok={detail.get('decomposition_ok')}", flush=True)

    existing_test = ROOT / "test_evaluate.py"
    if existing_test.exists():
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    str(existing_test),
                    "-q",
                    "-x",
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=900,
                check=False,
            )
            combined = (completed.stdout or "") + (completed.stderr or "")
            print(f"test_evaluate_returncode={completed.returncode}", flush=True)
            print(
                "test_failure_n_ge_2_out_of_scope="
                f"{'n=1 only' in combined or 'currently defined for n=1 only' in combined}",
                flush=True,
            )
            print(
                "test_failure_old_coeffs_signature="
                f"{'required positional argument' in combined and 'coeffs' in combined}",
                flush=True,
            )
            print("test_evaluate_output_tail:", flush=True)
            print("\n".join(combined.splitlines()[-20:]), flush=True)
        except subprocess.TimeoutExpired as exc:
            print("test_evaluate_timeout_s=900", flush=True)
            partial = (exc.stdout or "") + (exc.stderr or "")
            print("\n".join(str(partial).splitlines()[-20:]), flush=True)
    else:
        print("test_evaluate.py=not_found", flush=True)

    print(f"\nelapsed_s={time.perf_counter() - started:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
