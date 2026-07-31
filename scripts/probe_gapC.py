r"""괴리 C: PEAK pk 1차 손실계수 재산출 효과를 분리 계측한다.

AVG의 P=0 중심 5계수는 전 회차 고정하고, PEAK 계수만 P=0 중심에서 실제
운영점 중심으로 두 번 갱신한다. 본코드는 수정하지 않으며 실행은 사용자가 수행한다.

    python C:\Users\PowerSL\summer_ess_project\scripts\probe_gapC.py
"""

from __future__ import annotations

import csv
import datetime as datetime_
import importlib.metadata
import os
import platform
import socket
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import benefits
import params as PM
from build_net import build_net
import probe_gapB as GAP


ROUND_LABELS = ("round0", "round1", "round2")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REFERENCE_CALLS_PER_SECOND = 200.0
COEF_NAMES = GAP.COEF_NAMES

CSV_FIELDS = [
    "point_id", "b", "S", "E", "coefficient_round", "scenario", "t",
    "p_center_mw", "P_net_mw", "Q_mvar", "p_slack_mw", "line_loss_mw",
    "lp_pk_proxy_mw", "loss_reduction_linear_mw",
    "a_P", "a_Q", "b_PP", "b_QQ", "b_PQ",
    "matrix_rank", "lambda_min_cost", "lambda_max_cost",
    "solver", "status", "objective",
]


def _paths() -> tuple[str, str]:
    stamp = datetime_.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(RESULTS_DIR, f"probe_gapC_{stamp}")
    return stem + ".csv", stem + "_report.md"


def _expected_counts() -> dict[str, float]:
    baseline = len(PM.ALL_DAYS) * PM.TIME_STEPS
    feasible_per_time = sum(
        sum(
            GAP._grid_feasible(point, p, q)
            for p, q in GAP._local_grid(float(point["S"]), 0.0)
        )
        for point in GAP.POINTS
    )
    p0_surface = (
        feasible_per_time - len(GAP.POINTS)
    ) * len(PM.ALL_DAYS) * PM.TIME_STEPS
    peak_remeasure_upper = (
        len(GAP.POINTS) * 2 * len(PM.PEAK_DAYS)
        * PM.TIME_STEPS * 25
    )
    schedule_ac = (
        len(GAP.POINTS) * len(ROUND_LABELS)
        * len(PM.ALL_DAYS) * PM.TIME_STEPS
    )
    total = baseline + p0_surface + peak_remeasure_upper + schedule_ac
    return {
        "baseline_runpp_no_retry": baseline,
        "p0_coefficient_groups": (
            len(GAP.POINTS) * len(PM.ALL_DAYS) * PM.TIME_STEPS
        ),
        "p0_feasible_grid_rows_per_time_all_points": feasible_per_time,
        "p0_surface_runpp_exact_no_retry": p0_surface,
        "peak_remeasurement_groups": (
            len(GAP.POINTS) * 2
            * len(PM.PEAK_DAYS) * PM.TIME_STEPS
        ),
        "peak_remeasurement_runpp_upper_no_retry": peak_remeasure_upper,
        "schedule_ac_runpp_no_retry": schedule_ac,
        "total_runpp_upper_no_retry": total,
        "expected_seconds_at_200_calls_per_s": (
            total / REFERENCE_CALLS_PER_SECOND
        ),
    }


def _zero_centers(scenarios: list[str]) -> dict[str, np.ndarray]:
    return {
        scenario: np.zeros(PM.TIME_STEPS, dtype=float)
        for scenario in scenarios
    }


def _merge_coefficients(
    avg_coef: dict[tuple[str, str, int], dict[str, Any]],
    peak_coef: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    merged = dict(avg_coef)
    merged.update(peak_coef)
    return merged


def _subset_coefficient_counts(
    point: dict[str, Any],
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
    scenarios: list[str],
) -> dict[str, int]:
    rows = [
        coef_map[(point["point_id"], scenario, t)]
        for scenario in scenarios for t in range(PM.TIME_STEPS)
    ]
    return {
        "rank_lt_5_count": sum(int(row["matrix_rank"]) < 5 for row in rows),
        "psd_violation_count": sum(
            float(row["lambda_min_cost"]) < 0.0 for row in rows
        ),
    }


def _evaluate_schedule(
    writer,
    csv_file,
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    sgen: int,
    counter: GAP.RunCounter,
    point: dict[str, Any],
    label: str,
    schedules: dict[str, dict[str, Any]],
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    slack: dict[str, np.ndarray] = {}
    losses: dict[str, np.ndarray] = {}
    for scenario in PM.ALL_DAYS:
        p_slack = np.full(PM.TIME_STEPS, np.nan)
        line_loss = np.full(PM.TIME_STEPS, np.nan)
        schedule = schedules[scenario]
        for t in range(PM.TIME_STEPS):
            P = float(schedule["P"][t])
            Q = float(schedule["Q"][t])
            if np.isfinite(P) and np.isfinite(Q):
                GAP._set_load(net, base_p, base_q, scenario, t)
                pcs_loss = (
                    (1.0 - PM.ETA_PCS)
                    * (np.hypot(P, Q) - abs(P))
                )
                net.sgen.at[sgen, "bus"] = int(point["b"])
                net.sgen.at[sgen, "p_mw"] = P - pcs_loss
                net.sgen.at[sgen, "q_mvar"] = Q
                ok = counter.run(
                    net,
                    f"gapC/{point['point_id']}/{label}/{scenario}/t={t}",
                )
                if ok:
                    p_slack[t] = float(net.res_ext_grid.p_mw.sum())
                    line_loss[t] = float(net.res_line.pl_mw.sum())

            if scenario in PM.PEAK_DAYS:
                coef = coef_map[(point["point_id"], scenario, t)]
                loss_reduction = (
                    -float(coef["a_Q"]) * Q
                    - float(coef["a_P"]) * P
                    if np.isfinite(P) and np.isfinite(Q)
                    else np.nan
                )
                writer.writerow({
                    "point_id": point["point_id"],
                    "b": point["b"], "S": point["S"], "E": point["E"],
                    "coefficient_round": label,
                    "scenario": scenario, "t": t,
                    "p_center_mw": coef.get("p_center_mw", np.nan),
                    "P_net_mw": P, "Q_mvar": Q,
                    "p_slack_mw": p_slack[t],
                    "line_loss_mw": line_loss[t],
                    "lp_pk_proxy_mw": schedule["pk"],
                    "loss_reduction_linear_mw": loss_reduction,
                    **{name: coef[name] for name in COEF_NAMES},
                    "matrix_rank": coef["matrix_rank"],
                    "lambda_min_cost": coef["lambda_min_cost"],
                    "lambda_max_cost": coef["lambda_max_cost"],
                    "solver": schedule["solver"],
                    "status": schedule["status"],
                    "objective": schedule["objective"],
                })
                csv_file.flush()
        slack[scenario] = p_slack
        losses[scenario] = line_loss
    return slack, losses


def _benefits(
    point: dict[str, Any],
    base_slack: dict[str, np.ndarray],
    slack: dict[str, np.ndarray],
) -> dict[str, float]:
    b_energy = benefits.b_energy(
        base_slack, slack, PM.SMP_PER_MWH, PM.N_WEEKDAYS
    )
    b_defer = benefits.b_defer(base_slack, slack)
    cost = benefits.total_cost(float(point["S"]), float(point["E"]))
    j_net = benefits.j_net(
        b_energy, b_defer, float(point["S"]), float(point["E"])
    )
    return {
        "b_energy": b_energy,
        "b_defer": b_defer,
        "cost": cost,
        "j_net": j_net,
    }


def _nanmax(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    return (
        float(np.nanmax(array))
        if np.any(np.isfinite(array))
        else np.nan
    )


def _table1_rows(
    results: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for point in GAP.POINTS:
        point_id = point["point_id"]
        r0 = results[point_id]["round0"]
        r1 = results[point_id]["round1"]
        r2 = results[point_id]["round2"]
        rows.append({
            "point_id": point_id,
            "b_defer_round0": r0["b_defer"],
            "b_defer_round1": r1["b_defer"],
            "b_defer_round2": r2["b_defer"],
            "delta_b_defer_1_minus_0": r1["b_defer"] - r0["b_defer"],
            "delta_b_defer_2_minus_1": r2["b_defer"] - r1["b_defer"],
            "j_net_round0": r0["j_net"],
            "j_net_round1": r1["j_net"],
            "j_net_round2": r2["j_net"],
            "delta_j_net_1_minus_0": r1["j_net"] - r0["j_net"],
            "delta_j_net_2_minus_1": r2["j_net"] - r1["j_net"],
        })
    return rows


def _table2_rows(
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
    slack: dict[str, dict[str, dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows = []
    for point in GAP.POINTS:
        point_id = point["point_id"]
        for scenario in PM.PEAK_DAYS:
            values: dict[str, tuple[float, float, float]] = {}
            for label in ROUND_LABELS:
                lp_pk = float(schedules[point_id][label][scenario]["pk"])
                ac_peak = _nanmax(slack[point_id][label][scenario])
                values[label] = (lp_pk, ac_peak, ac_peak - lp_pk)
            rows.append({
                "point_id": point_id,
                "scenario": scenario,
                "lp_pk_round0_mw": values["round0"][0],
                "ac_slack_peak_round0_mw": values["round0"][1],
                "gap_round0_mw": values["round0"][2],
                "lp_pk_round1_mw": values["round1"][0],
                "ac_slack_peak_round1_mw": values["round1"][1],
                "gap_round1_mw": values["round1"][2],
                "lp_pk_round2_mw": values["round2"][0],
                "ac_slack_peak_round2_mw": values["round2"][1],
                "gap_round2_mw": values["round2"][2],
                "delta_gap_1_minus_0_mw": (
                    values["round1"][2] - values["round0"][2]
                ),
                "delta_gap_2_minus_1_mw": (
                    values["round2"][2] - values["round1"][2]
                ),
            })
    return rows


def _table3_rows(
    slack: dict[str, dict[str, dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows = []
    for point in GAP.POINTS:
        point_id = point["point_id"]
        peaks = {}
        for label in ROUND_LABELS:
            peaks[label] = _nanmax(
                np.asarray([
                    _nanmax(slack[point_id][label][scenario])
                    for scenario in PM.PEAK_DAYS
                ])
            )
        rows.append({
            "point_id": point_id,
            "ac_annual_peak_round0_mw": peaks["round0"],
            "ac_annual_peak_round1_mw": peaks["round1"],
            "ac_annual_peak_round2_mw": peaks["round2"],
            "delta_peak_1_minus_0_mw": (
                peaks["round1"] - peaks["round0"]
            ),
            "delta_peak_2_minus_1_mw": (
                peaks["round2"] - peaks["round1"]
            ),
        })
    return rows


def _table4_rows(
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
    checks: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for point in GAP.POINTS:
        point_id = point["point_id"]
        for label in ROUND_LABELS:
            check = checks[point_id][label]
            for scenario in PM.ALL_DAYS:
                result = schedules[point_id][label][scenario]
                rows.append({
                    "point_id": point_id,
                    "coefficient_round": label,
                    "scenario": scenario,
                    "solver": result["solver"],
                    "status": result["status"],
                    "avg_rank_lt_5_count": check["avg_rank_lt_5_count"],
                    "peak_rank_lt_5_count": check["peak_rank_lt_5_count"],
                    "avg_psd_violation_count": check[
                        "avg_psd_violation_count"
                    ],
                    "peak_psd_violation_count": check[
                        "peak_psd_violation_count"
                    ],
                    "peak_sign_bad_count": check["peak_sign_bad"],
                    "peak_discharge_count": check["peak_discharge_count"],
                    "peak_loss_reduction_bad_count": check[
                        "peak_loss_reduction_bad_count"
                    ],
                    "peak_loss_reduction_min": check[
                        "peak_loss_reduction_min"
                    ],
                    "hessian_projection_count": 0,
                })
    return rows


def _table5_rows(
    results: dict[str, dict[str, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows = []
    for point in GAP.POINTS:
        point_id = point["point_id"]
        b0 = results[point_id]["round0"]["b_energy"]
        b1 = results[point_id]["round1"]["b_energy"]
        b2 = results[point_id]["round2"]["b_energy"]
        d10 = b1 - b0
        d21 = b2 - b1
        max_change = max(abs(d10), abs(d21))
        rows.append({
            "point_id": point_id,
            "b_energy_round0": b0,
            "b_energy_round1": b1,
            "b_energy_round2": b2,
            "delta_b_energy_1_minus_0": d10,
            "delta_b_energy_2_minus_1": d21,
            "max_abs_delta_b_energy": max_change,
            "max_abs_delta_over_abs_round0": (
                max_change / abs(b0) if b0 != 0 else np.nan
            ),
            "rounds_isclose_rtol_1e_9": bool(
                np.isclose(b0, b1, rtol=1e-9, atol=0.0)
                and np.isclose(b1, b2, rtol=1e-9, atol=0.0)
            ),
        })
    return rows


def _md_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = list(rows[0]) if rows else []
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, (float, np.floating)):
                value = f"{float(value):.10g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _write_report(
    path: str,
    table1: list[dict[str, Any]],
    table2: list[dict[str, Any]] | None = None,
    table3: list[dict[str, Any]] | None = None,
    table4: list[dict[str, Any]] | None = None,
    table5: list[dict[str, Any]] | None = None,
    environment: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# probe_gapC", "", "## 표 1", ""]
    lines.extend(_md_table(table1))
    for number, table in (
        (2, table2), (3, table3), (4, table4), (5, table5)
    ):
        if table is not None:
            lines.extend(["", f"## 표 {number}", ""])
            lines.extend(_md_table(table))
    if environment is not None:
        lines.extend(["", "## 환경", ""])
        lines.extend(_md_table(environment))
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
        file.flush()
        os.fsync(file.fileno())


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _environment(
    started: float,
    measured_elapsed: float,
    counter: GAP.RunCounter,
    expected: dict[str, float],
    runtime_parameters: dict[str, float],
) -> list[dict[str, Any]]:
    rows = [
        {
            "item": "timestamp",
            "value": datetime_.datetime.now().isoformat(timespec="seconds"),
        },
        {"item": "hostname", "value": socket.gethostname()},
        {"item": "platform", "value": platform.platform()},
        {"item": "python", "value": platform.python_version()},
        {"item": "numpy", "value": np.__version__},
        {"item": "pandapower", "value": GAP.pp.__version__},
        {
            "item": "installed_solvers",
            "value": ",".join(GAP.JBASE.cp.installed_solvers()),
        },
        {"item": "POLY_N", "value": GAP.JBASE.POLY_N},
        {"item": "AVG_coefficient_round", "value": "P=0 fixed"},
        {
            "item": "PEAK_coefficient_rounds",
            "value": "P=0,P_round0,P_round1",
        },
        {"item": "runpp_calls_total", "value": counter.runpp_calls},
        {"item": "pf_retry_events", "value": counter.retry_events},
        {"item": "pf_diverged_calls", "value": counter.diverged},
        {"item": "measured_elapsed_s", "value": measured_elapsed},
        {
            "item": "runpp_calls_per_s",
            "value": (
                counter.runpp_calls / measured_elapsed
                if measured_elapsed > 0 else np.nan
            ),
        },
        {
            "item": "total_execution_time_s",
            "value": time.perf_counter() - started,
        },
        {"item": "hessian_projection_count", "value": 0},
    ]
    rows.extend(
        {"item": key, "value": value}
        for key, value in runtime_parameters.items()
    )
    rows.extend(
        {"item": key, "value": value}
        for key, value in expected.items()
    )
    for solver, package in (
        ("CLARABEL", "clarabel"), ("SCS", "scs")
    ):
        if solver in GAP.JBASE.cp.installed_solvers():
            rows.append({
                "item": f"solver_{solver}_version",
                "value": _version(package),
            })
    return rows


def main() -> int:
    started = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path, report_path = _paths()
    GAP.JBASE._validate_benefits_signatures()

    net = build_net()
    runtime_parameters = GAP._validate_runtime_parameters(net)
    expected = _expected_counts()
    for key, value in expected.items():
        print(f"{key}={value}", flush=True)

    counter = GAP.RunCounter()
    base_p = net.load["p_mw"].to_numpy(dtype=float).copy()
    base_q = net.load["q_mvar"].to_numpy(dtype=float).copy()
    sgen = GAP._ensure_sgen(net)
    measured_started = time.perf_counter()
    base_slack, baseline_loss = GAP._measure_baseline(
        net, base_p, base_q, sgen, counter
    )

    schedules: dict[
        str, dict[str, dict[str, dict[str, Any]]]
    ] = defaultdict(dict)
    slack: dict[
        str, dict[str, dict[str, np.ndarray]]
    ] = defaultdict(dict)
    results: dict[
        str, dict[str, dict[str, float]]
    ] = defaultdict(dict)
    checks: dict[
        str, dict[str, dict[str, Any]]
    ] = defaultdict(dict)

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()

        for point in GAP.POINTS:
            point_id = point["point_id"]
            avg_coef, _avg_rows = GAP._measure_coefficient_set(
                net, base_p, base_q, sgen, counter, point,
                _zero_centers(PM.AVG_DAYS), baseline_loss,
                scenarios=PM.AVG_DAYS,
            )
            avg_counts = _subset_coefficient_counts(
                point, avg_coef, PM.AVG_DAYS
            )
            peak_centers = _zero_centers(PM.PEAK_DAYS)

            for round_index, label in enumerate(ROUND_LABELS):
                peak_coef, _peak_rows = GAP._measure_coefficient_set(
                    net, base_p, base_q, sgen, counter, point,
                    peak_centers, baseline_loss,
                    scenarios=PM.PEAK_DAYS,
                )
                merged_coef = _merge_coefficients(avg_coef, peak_coef)
                coefficient_check = GAP._validate_coefficient_set(
                    point, merged_coef
                )
                peak_counts = _subset_coefficient_counts(
                    point, peak_coef, PM.PEAK_DAYS
                )
                schedule = GAP._solve_schedule_set(point, merged_coef)
                schedules[point_id][label] = schedule
                reduction_check = GAP._validate_peak_loss_reduction(
                    point, merged_coef, schedule
                )
                checks[point_id][label] = {
                    "avg_rank_lt_5_count": avg_counts["rank_lt_5_count"],
                    "avg_psd_violation_count": avg_counts[
                        "psd_violation_count"
                    ],
                    "peak_rank_lt_5_count": peak_counts["rank_lt_5_count"],
                    "peak_psd_violation_count": peak_counts[
                        "psd_violation_count"
                    ],
                    **coefficient_check,
                    **reduction_check,
                }
                round_slack, _round_losses = _evaluate_schedule(
                    writer, csv_file, net, base_p, base_q, sgen, counter,
                    point, label, schedule, merged_coef,
                )
                slack[point_id][label] = round_slack
                results[point_id][label] = _benefits(
                    point, base_slack, round_slack
                )
                if round_index < len(ROUND_LABELS) - 1:
                    peak_centers = {
                        scenario: np.asarray(
                            schedule[scenario]["P"], dtype=float
                        ).copy()
                        for scenario in PM.PEAK_DAYS
                    }

    table1 = _table1_rows(results)
    _write_report(report_path, table1)
    print("TABLE_1", flush=True)
    for row in table1:
        print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)

    table2 = _table2_rows(schedules, slack)
    _write_report(report_path, table1, table2)
    print("TABLE_2", flush=True)
    for row in table2:
        print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)

    table3 = _table3_rows(slack)
    _write_report(report_path, table1, table2, table3)
    print("TABLE_3", flush=True)
    for row in table3:
        print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)

    table4 = _table4_rows(schedules, checks)
    _write_report(report_path, table1, table2, table3, table4)
    print("TABLE_4", flush=True)
    for row in table4:
        print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)

    table5 = _table5_rows(results)
    measured_elapsed = time.perf_counter() - measured_started
    environment = _environment(
        started, measured_elapsed, counter, expected, runtime_parameters
    )
    _write_report(
        report_path, table1, table2, table3, table4, table5, environment
    )
    print("TABLE_5", flush=True)
    for row in table5:
        print(" ".join(f"{key}={value}" for key, value in row.items()), flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"runpp_calls={counter.runpp_calls}", flush=True)
    print(f"total_execution_time_s={time.perf_counter()-started:.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
