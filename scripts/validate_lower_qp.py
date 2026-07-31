"""lower_lp.py QP rewrite validation.

Run from the configured ``ess`` conda environment.  This script does not
modify project files; it only builds and solves small validation problems.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback
from pathlib import Path


os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _print_stage(name: str) -> None:
    print(f"\n[stage] {name}", flush=True)


def _status(entry: dict) -> tuple[str, str]:
    problem = entry["problem"]
    solver = getattr(problem.solver_stats, "solver_name", None)
    return str(problem.status), str(solver)


def main() -> int:
    started = time.perf_counter()

    _print_stage("imports")
    import cvxpy as cp
    import numpy as np

    import lower_lp as LP
    import params as PM

    print(f"python={sys.version.split()[0]}", flush=True)
    print(f"cvxpy={cp.__version__}", flush=True)

    T = PM.TIME_STEPS
    coeffs = {
        "a_P": np.full((1, T), -0.01, dtype=float),
        "a_Q": np.full((1, T), -0.01, dtype=float),
        "b_PP": np.full((1, T), 0.02, dtype=float),
        "b_QQ": np.full((1, T), 0.02, dtype=float),
        "b_PQ": np.zeros((1, T), dtype=float),
    }
    zero_coeffs = {
        name: np.zeros((1, T), dtype=float) for name in LP._COEFF_KEYS
    }
    common = (
        float(PM.ETA_C),
        float(PM.ETA_D),
        float(PM.SELF_DISCHARGE_HOURLY),
        float(PM.SOC_INIT_FRAC),
        float(PM.SOC_MIN_FRAC),
        float(PM.SOC_MAX_FRAC),
    )

    _print_stage("1. DCP/DPP gate")
    dpp_rows = []
    for kind in ("avg", "peak"):
        for force_q_zero in (False, True):
            entry = LP._get_problem(
                kind,
                1,
                force_q_zero,
                *common,
                float(PM.MU_VOLT),
                coeffs,
            )
            row = {
                "kind": kind,
                "force_q_zero": force_q_zero,
                "dcp": entry["problem"].is_dcp(),
                "dpp": entry["problem"].is_dcp(dpp=True),
                "h_full_min_eig": float(entry["h_full_min_eig"]),
            }
            dpp_rows.append(row)
            print(
                f"kind={kind} force_q_zero={force_q_zero} "
                f"dcp={row['dcp']} dpp={row['dpp']} "
                f"h_full_min_eig={row['h_full_min_eig']:.12g}",
                flush=True,
            )
    if not all(row["dcp"] and row["dpp"] for row in dpp_rows):
        raise AssertionError("DCP/DPP gate failed")

    _print_stage("2. artificial-coefficient AVG/PEAK solves")
    S = np.array([0.2], dtype=float)
    E = np.array([0.42], dtype=float)
    bus = np.array([15], dtype=int)
    profile = np.asarray(PM.LOAD["summer"], dtype=float)
    # L_cost와 P_ch/P_dis가 MW 단위이므로 목적함수의 SMP는 원/MWh를 사용한다.
    smp = np.asarray(PM.SMP_PER_MWH["summer"], dtype=float)
    print(
        f"SMP_units=won_per_MWh min={np.min(smp):.12g} max={np.max(smp):.12g}",
        flush=True,
    )
    load_total = LP.base_load_bus_arrays()[0].sum() * profile

    p_avg, q_avg, soc_avg = LP.solve_avg(
        S, E, bus, smp, profile, coeffs
    )
    avg_entry = LP._get_problem(
        "avg", 1, False, *common, float(PM.MU_VOLT), coeffs
    )
    avg_status, avg_solver = _status(avg_entry)
    print(
        f"AVG status={avg_status} solver={avg_solver} "
        f"max|P|={np.max(np.abs(p_avg)):.12g} "
        f"max|Q|={np.max(np.abs(q_avg)):.12g} "
        f"max_sqrt(P2+Q2)={np.max(np.hypot(p_avg, q_avg)):.12g}",
        flush=True,
    )

    p_peak, q_peak, soc_peak, pk = LP.solve_peak(
        S, E, bus, load_total, profile, coeffs
    )
    peak_entry = LP._get_problem(
        "peak", 1, False, *common, float(PM.MU_VOLT), coeffs
    )
    peak_status, peak_solver = _status(peak_entry)
    print(
        f"PEAK status={peak_status} solver={peak_solver} pk={pk:.12g} "
        f"max|P|={np.max(np.abs(p_peak)):.12g} "
        f"max|Q|={np.max(np.abs(q_peak)):.12g} "
        f"max_sqrt(P2+Q2)={np.max(np.hypot(p_peak, q_peak)):.12g}",
        flush=True,
    )
    valid_statuses = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
    if avg_entry["problem"].status not in valid_statuses:
        raise AssertionError(f"AVG solve status={avg_entry['problem'].status}")
    if peak_entry["problem"].status not in valid_statuses:
        raise AssertionError(f"PEAK solve status={peak_entry['problem'].status}")

    _print_stage("3. Hessian PSD")
    min_eig = min(row["h_full_min_eig"] for row in dpp_rows)
    print(f"global_h_min_eig={min_eig:.12g}", flush=True)
    if min_eig < -LP._PSD_TOL:
        raise AssertionError(f"Hessian is not PSD: lambda_min={min_eig}")

    _print_stage("4. force_q_zero regression")
    p_qzero, q_qzero, _ = LP.solve_avg(
        S, E, bus, smp, profile, coeffs, force_q_zero=True
    )
    p_zero_coeff, q_zero_coeff, _ = LP.solve_avg(
        S, E, bus, smp, profile, zero_coeffs, force_q_zero=True
    )
    max_q_qzero = float(np.max(np.abs(q_qzero)))
    p_schedule_diff = float(np.max(np.abs(p_qzero - p_zero_coeff)))
    print(f"max|Q_force_zero|={max_q_qzero:.12g}", flush=True)
    print(
        "max|P_with_loss-P_zero_coeff|="
        f"{p_schedule_diff:.12g}",
        flush=True,
    )
    if max_q_qzero > 1e-6:
        raise AssertionError(f"force_q_zero regression failed: {max_q_qzero}")

    _print_stage("5. free-Q and PCS polygon")
    max_q_free = float(np.max(np.abs(q_avg)))
    max_apparent = float(np.max(np.hypot(p_avg, q_avg)))
    polygon_tolerance = S[0] / np.cos(np.pi / PM.POLY_N) - S[0] + 1e-6
    print(f"max|Q_free|={max_q_free:.12g}", flush=True)
    print(f"max_sqrt(P2+Q2)={max_apparent:.12g}", flush=True)
    print(f"S={S[0]:.12g}", flush=True)
    print(f"polygon_radial_tolerance={polygon_tolerance:.12g}", flush=True)
    if max_q_free <= 1e-7:
        raise AssertionError("free-Q solution is numerically zero")
    if max_apparent > S[0] + polygon_tolerance:
        raise AssertionError("PCS polygon bound exceeded")

    _print_stage("6. cache key behavior")
    same_entry = LP._get_problem(
        "avg", 1, False, *common, float(PM.MU_VOLT), coeffs
    )
    changed_coeffs = {name: value.copy() for name, value in coeffs.items()}
    changed_coeffs["a_P"][0, 0] -= 1e-6
    different_entry = LP._get_problem(
        "avg",
        1,
        False,
        *common,
        float(PM.MU_VOLT),
        changed_coeffs,
    )
    print(f"same_coeff_reused={same_entry is avg_entry}", flush=True)
    print(
        f"changed_coeff_rebuilt={different_entry is not avg_entry}",
        flush=True,
    )
    print(f"cache_size={len(LP._PROBLEM_CACHE)}", flush=True)

    _print_stage("7. mu_volt=0")
    mu0_entry = LP._get_problem(
        "avg", 1, False, *common, 0.0, coeffs
    )
    p_mu0, q_mu0, soc_mu0 = LP.solve_avg(
        S, E, bus, smp, profile, coeffs, mu_volt=0.0
    )
    mu0_status, mu0_solver = _status(mu0_entry)
    print(
        f"dcp={mu0_entry['problem'].is_dcp()} "
        f"dpp={mu0_entry['problem'].is_dcp(dpp=True)} "
        f"status={mu0_status} solver={mu0_solver} "
        f"max_sqrt(P2+Q2)={np.max(np.hypot(p_mu0, q_mu0)):.12g}",
        flush=True,
    )
    if not mu0_entry["problem"].is_dcp(dpp=True):
        raise AssertionError("mu_volt=0 problem is not DPP")

    _print_stage("8. existing test_lp.py compatibility check")
    existing_test = ROOT / "test_lp.py"
    if existing_test.exists():
        completed = subprocess.run(
            [sys.executable, str(existing_test)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=300,
            check=False,
        )
        combined = (completed.stdout or "") + (completed.stderr or "")
        signature_failure = (
            completed.returncode != 0
            and "coeffs" in combined
            and (
                "required positional argument" in combined
                or "missing" in combined
            )
        )
        print(f"test_lp_returncode={completed.returncode}", flush=True)
        print(
            f"expected_required_coeffs_signature_failure={signature_failure}",
            flush=True,
        )
        if completed.returncode != 0:
            tail = "\n".join(combined.splitlines()[-12:])
            print("test_lp_output_tail:", flush=True)
            print(tail, flush=True)
    else:
        print("test_lp.py=not_found", flush=True)

    elapsed = time.perf_counter() - started
    _print_stage("complete")
    print(f"elapsed_s={elapsed:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        print("\nVALIDATION_FAILED", flush=True)
        raise SystemExit(1)
