r"""User-run validation for command G diagnostic persistence."""

from __future__ import annotations

import csv
import glob
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
import lower_lp as LP
import params as PM
import postprocess


def _stage(number: int, title: str) -> None:
    print(f"\n[stage] {number}. {title}", flush=True)


def _dummy_coeffs() -> dict:
    shape = (1, PM.TIME_STEPS)
    return {
        "a_P": np.full(shape, -0.01),
        "a_Q": np.full(shape, -0.01),
        "b_PP": np.full(shape, 0.02),
        "b_QQ": np.full(shape, 0.02),
        "b_PQ": np.zeros(shape),
    }


def _latest_runs_after(started_wall: float) -> str:
    candidates = [
        path for path in glob.glob(str(ROOT / "results" / "runs_n1_*.csv"))
        if os.path.getmtime(path) >= started_wall - 1.0
    ]
    if not candidates:
        raise RuntimeError("stage 3 runs.csv not found")
    return max(candidates, key=os.path.getmtime)


def main() -> int:
    started = time.perf_counter()
    particle = np.asarray([15.0, 0.2, 0.42], dtype=float)

    _stage(1, "lower_lp voltage return and DPP")
    coeffs = _dummy_coeffs()
    scenario = PM.AVG_DAYS[0]
    legacy = LP.solve_avg(
        0.2,
        0.42,
        15,
        PM.SMP_PER_MWH[scenario],
        PM.LOAD[scenario],
        coeffs,
        return_voltage=False,
    )
    with_voltage = LP.solve_avg(
        0.2,
        0.42,
        15,
        PM.SMP_PER_MWH[scenario],
        PM.LOAD[scenario],
        coeffs,
        return_voltage=True,
    )
    entry = LP._get_problem(
        "avg",
        1,
        False,
        PM.ETA_C,
        PM.ETA_D,
        PM.SELF_DISCHARGE_HOURLY,
        PM.SOC_INIT_FRAC,
        PM.SOC_MIN_FRAC,
        PM.SOC_MAX_FRAC,
        PM.MU_VOLT,
        LP._normalize_coeffs(coeffs, 1, PM.TIME_STEPS),
    )
    voltage_sq = np.asarray(with_voltage[-1])
    print(f"legacy_tuple_len={len(legacy)}", flush=True)
    print(f"voltage_tuple_len={len(with_voltage)}", flush=True)
    print(f"voltage_shape={voltage_sq.shape}", flush=True)
    print(f"voltage_sq_min={np.min(voltage_sq):.12g}", flush=True)
    print(f"voltage_sq_max={np.max(voltage_sq):.12g}", flush=True)
    print(f"is_dcp_dpp={entry['problem'].is_dcp(dpp=True)}", flush=True)
    assert len(legacy) == 3
    assert len(with_voltage) == 4
    assert voltage_sq.shape == (PM.N_BUS, PM.TIME_STEPS)
    assert entry["problem"].is_dcp(dpp=True)

    _stage(2, "evaluate recompute intermediates")
    t_scalar = time.perf_counter()
    scalar_fitness = evaluate.evaluate_particle(particle)
    scalar_runtime = time.perf_counter() - t_scalar
    t_detail = time.perf_counter()
    detail = evaluate.evaluate_particle(
        particle, return_detail=True, collect_diagnostics=True
    )
    detail_runtime = time.perf_counter() - t_detail
    for field in (
        "recompute_jnet_delta",
        "recompute_max_p_shift",
        "recompute_convergence_ratio",
    ):
        print(f"{field}={detail.get(field)}", flush=True)
    print(f"diagnostic_error={detail.get('diagnostic_error')!r}", flush=True)
    print(f"scalar_runtime_s={scalar_runtime:.6f}", flush=True)
    print(f"detail_runtime_s={detail_runtime:.6f}", flush=True)
    assert np.isfinite(detail["recompute_jnet_delta"])
    assert np.isfinite(detail["recompute_max_p_shift"])
    assert np.isfinite(detail["recompute_convergence_ratio"])

    _stage(3, "single-process cache statistics in runs.csv")
    wall_start = time.time()
    command = [
        sys.executable,
        str(ROOT / "main.py"),
        "--n-ess", "1",
        "--profile", "dev",
        "--n-workers", "1",
        "--n-particles", "2",
        "--n-iters", "1",
        "--n-runs", "1",
        "--no-postprocess",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    print(f"main_returncode={completed.returncode}", flush=True)
    print("\n".join((completed.stdout + completed.stderr).splitlines()[-25:]), flush=True)
    if completed.returncode != 0:
        raise RuntimeError("single-process diagnostic main run failed")
    runs_path = _latest_runs_after(wall_start)
    with open(runs_path, newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    for field in (
        "coef_cache_hit_rate",
        "coef_runpp_total",
        "coef_unique_bs_count",
        "coef_measure_wall_s",
        "recompute_jnet_delta",
        "recompute_convergence_ratio",
        "recompute_max_p_shift",
    ):
        print(f"{field}={row[field]}", flush=True)
    hit_rate = float(row["coef_cache_hit_rate"])
    assert 0.0 <= hit_rate <= 1.0

    _stage(4, "postprocess voltage and mu diagnostics")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "results" / "validate_diagnostics"
    result = postprocess.process_group(runs_path, str(out_dir), timestamp)
    if result is None or result.get("schedule9") is None:
        raise RuntimeError("postprocess diagnostic result unavailable")
    metrics = result["schedule9"]["metrics"]
    voltage_error = float(metrics["max_lindistflow_ac_voltage_err"])
    print(f"max_lindistflow_ac_voltage_err={voltage_error:.12g}", flush=True)
    print(f"voltage_gate_0p005_pass={voltage_error <= 0.005}", flush=True)
    print(f"mu_penalty_active={metrics['mu_penalty_active']}", flush=True)
    schedule_path = result["schedule9"]["path"]
    with open(schedule_path, newline="", encoding="utf-8") as stream:
        schedule_rows = list(csv.DictReader(stream))
    assert schedule_rows
    assert all(row["v_lindistflow"] != "" for row in schedule_rows)
    assert all(row["mu_penalty_active"] != "" for row in schedule_rows)

    _stage(5, "default-path regression")
    detail_light = evaluate.evaluate_particle(particle, return_detail=True)
    print(f"scalar_fitness={scalar_fitness:.12g}", flush=True)
    print(f"light_detail_fitness={detail_light['fitness']:.12g}", flush=True)
    print(
        "default_path_match="
        f"{np.isclose(scalar_fitness, detail_light['fitness'], atol=10.0, rtol=0.0)}",
        flush=True,
    )
    print(
        f"light_diagnostics_collected={detail_light.get('v_lindistflow_sq') is not None}",
        flush=True,
    )
    assert np.isclose(
        scalar_fitness, detail_light["fitness"], atol=10.0, rtol=0.0
    )
    assert detail_light.get("v_lindistflow_sq") is None

    print(f"\ntotal_runtime_s={time.perf_counter() - started:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
