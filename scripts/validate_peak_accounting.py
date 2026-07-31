"""User-run validation for the peak-day dual-benefit accounting change."""

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

import benefits
import evaluate
import params as PM


def _stage(number: int, title: str) -> None:
    print(f"\n[stage] {number}. {title}", flush=True)


def _one_day_energy(
    p_slack_base: dict,
    p_slack_ess: dict,
    smp_mwh: dict,
    scenario: str,
) -> float:
    delta_mw = (
        np.asarray(p_slack_base[scenario])
        - np.asarray(p_slack_ess[scenario])
    )
    return (
        float(np.sum(delta_mw * np.asarray(smp_mwh[scenario])))
        * PM.DT_HOURS
    )


def _synthetic_function_tests() -> None:
    hours = PM.TIME_STEPS
    base = {
        scenario: np.full(hours, 10.0, dtype=float)
        for scenario in PM.ALL_DAYS
    }
    ess = {
        scenario: np.full(hours, 9.0, dtype=float)
        for scenario in PM.ALL_DAYS
    }
    smp = {
        scenario: np.full(hours, 100_000.0, dtype=float)
        for scenario in PM.ALL_DAYS
    }

    # summer_peak has the larger ESS slack peak.
    peak_ess = {
        scenario: np.asarray(ess[scenario], dtype=float).copy()
        for scenario in PM.PEAK_DAYS
    }
    peak_ess["summer_peak"][0] = 9.5
    peak_ess["winter_peak"][0] = 9.4
    selected = benefits.select_peak_day(peak_ess)
    assert selected == "summer_peak", selected

    # Equal maxima retain the first PM.PEAK_DAYS item.
    tie = {
        scenario: np.zeros(hours, dtype=float)
        for scenario in PM.PEAK_DAYS
    }
    for scenario in PM.PEAK_DAYS:
        tie[scenario][0] = 7.0
    tie_selected = benefits.select_peak_day(tie)
    assert tie_selected == PM.PEAK_DAYS[0], tie_selected

    old_energy = benefits.b_energy(base, ess, smp, PM.N_WEEKDAYS)
    adjusted_energy = benefits.b_energy_adjusted(
        base, ess, smp, PM.N_WEEKDAYS, "summer"
    )
    summer_day = _one_day_energy(base, ess, smp, "summer")
    assert np.isclose(
        old_energy - adjusted_energy, summer_day, atol=1e-6, rtol=0.0
    )

    peak_day_energy = benefits.b_energy_peak_day(
        base, ess, smp, "summer_peak"
    )
    expected_peak_day = _one_day_energy(base, ess, smp, "summer_peak")
    assert np.isclose(
        peak_day_energy, expected_peak_day, atol=1e-6, rtol=0.0
    )

    # All synthetic representative days have the same daily benefit, so
    # replacing one AVG day by one PEAK day preserves the effective day count.
    new_energy = adjusted_energy + peak_day_energy
    assert np.isclose(new_energy, old_energy, atol=1e-6, rtol=0.0)

    print(f"selected_peak_day={selected}", flush=True)
    print(f"tie_selected_peak_day={tie_selected}", flush=True)
    print(f"old_minus_adjusted_won={old_energy - adjusted_energy:.12g}", flush=True)
    print(f"summer_one_day_won={summer_day:.12g}", flush=True)
    print(f"peak_day_energy_won={peak_day_energy:.12g}", flush=True)
    print(f"effective_day_count_preserved={np.isclose(new_energy, old_energy)}", flush=True)


def main() -> int:
    started = time.perf_counter()

    _stage(1, "benefits pure-function unit tests")
    _synthetic_function_tests()

    _stage(2, "old/new accounting on one active particle")
    particle = np.asarray([15.0, 0.2, 0.42], dtype=float)
    detail = evaluate.evaluate_particle(particle, return_detail=True)
    if detail.get("diverged"):
        raise RuntimeError(f"active particle diverged: {detail.get('diverge_info')}")

    base_flow = evaluate._get_base_flow()
    avg_base = {s: base_flow["p_slack"][s] for s in PM.AVG_DAYS}
    avg_ess = {s: detail["p_slack_ess"][s] for s in PM.AVG_DAYS}
    old_b_energy = benefits.b_energy(
        avg_base, avg_ess, PM.SMP_PER_MWH, PM.N_WEEKDAYS
    )
    s_total = float(np.sum(detail["S"]))
    e_total = float(np.sum(detail["E"]))
    old_j_net = benefits.j_net(
        old_b_energy, detail["b_defer"], s_total, e_total
    )
    new_minus_old = float(detail["j_net"] - old_j_net)
    removed_avg_day = float(old_b_energy - detail["b_energy_avg"])
    accounting_delta = float(
        detail["b_energy_peak_day"] - removed_avg_day
    )

    print(f"peak_day={detail['peak_day']}", flush=True)
    print(f"adjusted_season={detail['adjusted_season']}", flush=True)
    print(f"b_energy_avg_won={detail['b_energy_avg']:.12g}", flush=True)
    print(
        f"b_energy_peak_day_won={detail['b_energy_peak_day']:.12g}",
        flush=True,
    )
    print(f"b_defer_won={detail['b_defer']:.12g}", flush=True)
    print(f"old_j_net_won={old_j_net:.12g}", flush=True)
    print(f"new_j_net_won={detail['j_net']:.12g}", flush=True)
    print(f"new_minus_old_j_net_won={new_minus_old:.12g}", flush=True)
    print(f"removed_avg_day_energy_won={removed_avg_day:.12g}", flush=True)
    print(f"accounting_delta_won={accounting_delta:.12g}", flush=True)
    print(
        "accounting_delta_matches="
        f"{np.isclose(new_minus_old, accounting_delta, atol=1e-6, rtol=0.0)}",
        flush=True,
    )

    _stage(3, "peak-day energy and defer contributions")
    energy_component = float(detail["b_energy_peak_day"])
    defer_component = float(detail["b_defer"])
    component_sum = energy_component + defer_component
    print(f"peak_day_energy_benefit_won={energy_component:.12g}", flush=True)
    print(f"peak_day_defer_benefit_won={defer_component:.12g}", flush=True)
    if component_sum != 0.0:
        print(
            f"energy_share={energy_component / component_sum:.12g}",
            flush=True,
        )
        print(
            f"defer_share={defer_component / component_sum:.12g}",
            flush=True,
        )
    else:
        print("energy_share=nan", flush=True)
        print("defer_share=nan", flush=True)

    _stage(4, "AVG decomposition consistency")
    avg_decomposition_sum = float(detail["b_arb"] + detail["b_loss"])
    print(f"b_arb_plus_b_loss_won={avg_decomposition_sum:.12g}", flush=True)
    print(f"b_energy_avg_won={detail['b_energy_avg']:.12g}", flush=True)
    print(f"b_energy_total_won={detail['b_energy']:.12g}", flush=True)
    print(f"decomposition_ok_avg={detail['decomposition_ok']}", flush=True)
    print(
        "decomposition_ok_total="
        f"{benefits.check_b_energy_decomposition(detail['b_arb'], detail['b_loss'], detail['b_energy'])}",
        flush=True,
    )

    _stage(5, "existing test_evaluate.py")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "test_evaluate.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    combined = "\n".join(
        part for part in (completed.stdout, completed.stderr) if part
    )
    print(f"test_evaluate_returncode={completed.returncode}", flush=True)
    print("test_evaluate_output_tail:", flush=True)
    print("\n".join(combined.splitlines()[-30:]), flush=True)

    print(f"\ntotal_runtime_s={time.perf_counter() - started:.6f}", flush=True)
    return 0 if completed.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
