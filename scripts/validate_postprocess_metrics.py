r"""User-run validation for command F postprocess metrics.

Pass an existing development runs CSV to execute stages 2-5:

    python C:\Users\PowerSL\summer_ess_project\scripts\validate_postprocess_metrics.py ^
      C:\absolute\path\to\runs_n1_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
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

import benefits
import params as PM
import postprocess


def _stage(number: int, title: str) -> None:
    print(f"\n[stage] {number}. {title}", flush=True)


def _pure_decomposition_test() -> None:
    hours = PM.TIME_STEPS
    smp = {
        scenario: np.ones(hours, dtype=float)
        for scenario in PM.ALL_DAYS
    }
    p_ch = {
        scenario: np.zeros(hours, dtype=float)
        for scenario in PM.AVG_DAYS
    }
    p_dis = {
        scenario: np.ones(hours, dtype=float)
        for scenario in PM.AVG_DAYS
    }
    loss_base = {
        scenario: np.ones(hours, dtype=float)
        for scenario in PM.AVG_DAYS
    }
    loss_ess = {
        scenario: np.full(hours, 0.9, dtype=float)
        for scenario in PM.AVG_DAYS
    }
    slack_base = {
        scenario: np.full(hours, 10.0, dtype=float)
        for scenario in PM.ALL_DAYS
    }
    slack_ess = {
        scenario: np.full(hours, 8.9, dtype=float)
        for scenario in PM.ALL_DAYS
    }

    peak_day = PM.PEAK_DAYS[0]
    adjusted_season = benefits.PEAK_TO_SEASON[peak_day]
    p_ch_peak = {peak_day: np.zeros(hours, dtype=float)}
    p_dis_peak = {peak_day: np.ones(hours, dtype=float)}
    loss_base_peak = {peak_day: np.ones(hours, dtype=float)}
    loss_ess_peak = {peak_day: np.full(hours, 0.9, dtype=float)}

    b_energy_avg = benefits.b_energy_adjusted(
        slack_base,
        slack_ess,
        smp,
        PM.N_WEEKDAYS,
        adjusted_season,
    )
    b_energy_peak = benefits.b_energy_peak_day(
        slack_base, slack_ess, smp, peak_day
    )
    b_energy_total = b_energy_avg + b_energy_peak

    b_arb_total = benefits.b_arb_total(
        p_ch,
        p_dis,
        smp,
        PM.N_WEEKDAYS,
        adjusted_season,
        peak_day,
        p_ch_peak,
        p_dis_peak,
    )
    b_loss_total = benefits.b_loss_total(
        loss_base,
        loss_ess,
        smp,
        PM.N_WEEKDAYS,
        adjusted_season,
        peak_day,
        loss_base_peak,
        loss_ess_peak,
    )
    total_ok = benefits.check_b_energy_decomposition(
        b_arb_total, b_loss_total, b_energy_total
    )
    assert total_ok

    adjusted_weights = {
        scenario: float(PM.N_WEEKDAYS[scenario])
        - (1.0 if scenario == adjusted_season else 0.0)
        for scenario in PM.AVG_DAYS
    }
    b_arb_avg = benefits.b_arb(p_ch, p_dis, smp, adjusted_weights)
    b_loss_avg = benefits.b_loss(
        loss_base, loss_ess, smp, adjusted_weights
    )
    avg_ok = benefits.check_b_energy_decomposition(
        b_arb_avg, b_loss_avg, b_energy_avg
    )
    assert avg_ok

    print(f"decomposition_ok_total={total_ok}", flush=True)
    print(f"decomposition_ok_avg={avg_ok}", flush=True)
    print(f"b_energy_total={b_energy_total:.12g}", flush=True)
    print(f"b_arb_total_plus_b_loss_total={b_arb_total + b_loss_total:.12g}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runs_csv",
        nargs="?",
        help="기존 dev runs_n1_<timestamp>.csv 절대경로",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "validate_postprocess_metrics"),
    )
    args = parser.parse_args()
    started = time.perf_counter()

    _stage(1, "F-2 pure-function decomposition")
    _pure_decomposition_test()

    if not args.runs_csv:
        absolute_script = ROOT / "scripts" / "validate_postprocess_metrics.py"
        print("\nstages_2_to_5=SKIPPED", flush=True)
        print("reason=existing dev runs CSV path required", flush=True)
        print(
            f"command=python {absolute_script} C:\\absolute\\path\\to\\runs_n1_<timestamp>.csv",
            flush=True,
        )
        return 0

    runs_path = str(Path(args.runs_csv).resolve())
    output_dir = str(Path(args.output_dir).resolve())
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result = postprocess.process_group(
        runs_path, output_dir, timestamp
    )
    if result is None or result.get("schedule9") is None:
        raise RuntimeError("postprocess schedule/metrics result unavailable")
    metrics = result["schedule9"]["metrics"]

    _stage(2, "group A schedule metrics")
    for field in (
        "q_sum_mvar",
        "q_nonzero_hours",
        "q_median_mvar",
        "q_max_mvar",
        "q_min_mvar",
        "q_to_s_ratio",
        "poly_binding_frac",
        "pcs_loss_annual_won",
        "s_util_median",
        "s_util_max",
        "s_util_high_hours",
        "peak_day_energy_benefit",
        "peak_day_defer_benefit",
    ):
        print(f"{field}={metrics.get(field)}", flush=True)
    assert 0.0 <= float(metrics["poly_binding_frac"]) <= 1.0
    assert float(metrics["s_util_max"]) <= 1.01

    _stage(3, "group B Q/P loss decomposition")
    b_loss_sum = float(metrics["b_loss_q"] + metrics["b_loss_p"])
    b_loss_reference = float(metrics["_b_loss_total_reference"])
    split_ok = np.isclose(
        b_loss_sum, b_loss_reference, atol=1e-6, rtol=0.0
    )
    print(f"b_loss_q={metrics['b_loss_q']:.12g}", flush=True)
    print(f"b_loss_p={metrics['b_loss_p']:.12g}", flush=True)
    print(f"b_loss_total_reference={b_loss_reference:.12g}", flush=True)
    print(f"b_loss_split_ok={split_ok}", flush=True)
    print(f"b_loss_q_positive={metrics['b_loss_q'] > 0.0}", flush=True)
    assert split_ok

    _stage(4, "group C missing-data handling")
    for field, reason in postprocess.GROUP_C_MISSING_REASONS.items():
        value = metrics.get(field)
        missing = isinstance(value, float) and np.isnan(value)
        print(
            f"{field}={value} "
            f"{'reason=' + reason if missing else 'source=stored_or_recomputed'}",
            flush=True,
        )

    _stage(5, "full postprocess smoke and schema")
    run_path, units_path = postprocess.write_group_outputs(
        result, output_dir, timestamp
    )
    schedule_path = result["schedule9"]["path"]
    with open(run_path, newline="", encoding="utf-8") as stream:
        run_fields = next(csv.reader(stream))
    with open(schedule_path, newline="", encoding="utf-8") as stream:
        schedule_fields = next(csv.reader(stream))

    old_run_fields = {
        "n_ess", "j_net_median", "best_run", "min_bdefer_overall"
    }
    assert old_run_fields <= set(run_fields)
    assert set(postprocess.DIAGNOSTIC_SUMMARY_FIELDS) <= set(run_fields)
    assert {"v_lindistflow", "mu_penalty_active"} <= set(schedule_fields)
    print(f"run_output={run_path}", flush=True)
    print(f"units_output={units_path}", flush=True)
    print(f"schedule_output={schedule_path}", flush=True)
    print("existing_columns_preserved=True", flush=True)
    print("new_columns_present=True", flush=True)

    print(f"\ntotal_runtime_s={time.perf_counter() - started:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
