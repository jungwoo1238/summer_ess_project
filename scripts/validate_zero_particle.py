"""User-run validation for inactive ESS early branching.

Stages 1, 2, and 4 execute AC power flows.  Run this script directly in the
configured ``ess`` environment.
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
import params as PM


BUS = 15
ACTIVE_REFERENCE_J_NET = 2.63e6


def _stage(number: int, title: str) -> None:
    print(f"\n[stage] {number}. {title}", flush=True)


def _base_differences(detail: dict, base_flow: dict) -> tuple[float, float]:
    slack_diff = max(
        float(
            np.max(
                np.abs(
                    np.asarray(detail["p_slack_ess"][scenario])
                    - np.asarray(base_flow["p_slack"][scenario])
                )
            )
        )
        for scenario in PM.ALL_DAYS
    )
    loss_diff = max(
        float(
            np.max(
                np.abs(
                    np.asarray(detail["loss_ess"][scenario])
                    - np.asarray(base_flow["loss"][scenario])
                )
            )
        )
        for scenario in PM.ALL_DAYS
    )
    return slack_diff, loss_diff


def main() -> int:
    started = time.perf_counter()
    print("warning=stages_1_2_4_execute_many_runpp_calls", flush=True)
    print(f"S_ACTIVE_MIN={evaluate.S_ACTIVE_MIN:.12g}", flush=True)
    print(f"E_ACTIVE_MIN={evaluate.E_ACTIVE_MIN:.12g}", flush=True)

    evaluate.init_worker()
    base_flow = evaluate._get_base_flow()

    _stage(1, "S=0 and E=0 match the base feeder")
    loss_coeffs.clear_cache()
    cache_before = loss_coeffs.cache_info()
    zero = evaluate.evaluate_particle(
        np.asarray([BUS, 0.0, 0.0], dtype=float),
        return_detail=True,
    )
    cache_after = loss_coeffs.cache_info()
    slack_diff, loss_diff = _base_differences(zero, base_flow)
    max_p = max(
        float(np.max(np.abs(zero["unit_p"][scenario])))
        for scenario in PM.ALL_DAYS
    )
    max_q = max(
        float(np.max(np.abs(zero["unit_q"][scenario])))
        for scenario in PM.ALL_DAYS
    )
    print(f"diverged={zero['diverged']}", flush=True)
    print(f"max_abs_slack_minus_base_mw={slack_diff:.12g}", flush=True)
    print(f"max_abs_loss_minus_base_mw={loss_diff:.12g}", flush=True)
    print(f"max_abs_P_net_mw={max_p:.12g}", flush=True)
    print(f"max_abs_Q_mvar={max_q:.12g}", flush=True)
    print(
        "loss_coeff_runpp_increment="
        f"{cache_after['runpp_attempts'] - cache_before['runpp_attempts']}",
        flush=True,
    )
    # Repeated AC solves can differ by a few nanowatts due to the Newton solver.
    base_atol_mw = 1e-8
    print(
        f"base_match_atol_1e-8="
        f"{slack_diff <= base_atol_mw and loss_diff <= base_atol_mw}",
        flush=True,
    )

    _stage(2, "inactive threshold boundary")
    tiny = evaluate.evaluate_particle(
        np.asarray([BUS, 1e-10, 0.5], dtype=float),
        return_detail=True,
    )
    tiny_slack_diff, tiny_loss_diff = _base_differences(tiny, base_flow)
    print(f"tiny_S_diverged={tiny['diverged']}", flush=True)
    print(f"tiny_S_slack_diff_mw={tiny_slack_diff:.12g}", flush=True)
    print(f"tiny_S_loss_diff_mw={tiny_loss_diff:.12g}", flush=True)

    active_cache_before = loss_coeffs.cache_info()
    small_active = evaluate.evaluate_particle(
        np.asarray([BUS, 1e-4, 0.5], dtype=float),
        return_detail=True,
    )
    active_cache_after = loss_coeffs.cache_info()
    print(f"S_1e-4_diverged={small_active['diverged']}", flush=True)
    print(f"S_1e-4_j_net={small_active.get('j_net')}", flush=True)
    print(
        "S_1e-4_loss_coeff_runpp_increment="
        f"{active_cache_after['runpp_attempts'] - active_cache_before['runpp_attempts']}",
        flush=True,
    )
    print(
        f"S_1e-4_j_net_finite={bool(np.isfinite(small_active.get('j_net', np.nan)))}",
        flush=True,
    )

    _stage(3, "existing zero-size regression")
    target = (
        f"{ROOT / 'test_evaluate.py'}"
        "::test_zero_size_particle_matches_base_exactly"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", target, "-q"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        combined = (completed.stdout or "") + (completed.stderr or "")
        print(f"zero_size_test_returncode={completed.returncode}", flush=True)
        print("zero_size_test_output_tail:", flush=True)
        print("\n".join(combined.splitlines()[-20:]), flush=True)
    except subprocess.TimeoutExpired as exc:
        print("zero_size_test_timeout_s=300", flush=True)
        partial = (exc.stdout or "") + (exc.stderr or "")
        print("\n".join(str(partial).splitlines()[-20:]), flush=True)

    _stage(4, "active-particle regression")
    active = evaluate.evaluate_particle(
        np.asarray([BUS, 0.2, 0.42], dtype=float),
        return_detail=True,
    )
    active_j_net = float(active.get("j_net", np.nan))
    print(f"active_diverged={active['diverged']}", flush=True)
    print(f"active_j_net={active_j_net:.12g}", flush=True)
    print(f"reference_j_net={ACTIVE_REFERENCE_J_NET:.12g}", flush=True)
    print(
        f"active_j_net_minus_reference={active_j_net - ACTIVE_REFERENCE_J_NET:.12g}",
        flush=True,
    )

    _stage(5, "postprocess schedule reproduction smoke test")
    import postprocess

    schedules, base_p_sum = postprocess.compute_schedules(
        [(BUS, 0.2, 0.42)]
    )
    shapes_ok = all(
        len(schedules[scenario]) == 1
        and np.asarray(schedules[scenario][0][0]).shape == (PM.TIME_STEPS,)
        and np.asarray(schedules[scenario][0][1]).shape == (PM.TIME_STEPS + 1,)
        and np.asarray(schedules[scenario][0][2]).shape == (PM.TIME_STEPS,)
        for scenario in PM.ALL_DAYS
    )
    print("postprocess_import_ok=True", flush=True)
    print(f"postprocess_compute_schedules_ok={shapes_ok}", flush=True)
    print(f"postprocess_base_p_sum_mw={base_p_sum:.12g}", flush=True)

    print(f"\nelapsed_s={time.perf_counter() - started:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
