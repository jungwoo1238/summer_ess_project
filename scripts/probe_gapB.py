r"""괴리 B: QP 손실계수의 운영점 stale 오차와 E 캐시 독립성을 계측한다.

본코드는 수정하지 않는다. 실행은 사용자가 직접 수행한다.

    python C:\Users\PowerSL\summer_ess_project\scripts\probe_gapB.py

각 통제점에서 C0(P=0 중심), C1(P0 중심), C2(P1 중심)를 새로 측정하고 완전 B
정식화로 AVG/PEAK 스케줄을 푼다. 각 회차와 C0 균일 그로스업 해를 AC/benefits로
평가한다. 계수 헤시안은 사영하지 않는다. 시작 시 확정 부하(S=10MVA, PF=0.95),
ETA_PCS=0.975, 슬랙 1.02를 검증하고 다르면 중단한다.
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
import pandapower as pp


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import benefits
import params as PM
from build_net import build_net
import probe_pq_jnet_v2 as JBASE
import probe_pq_loss_v2 as SURFACE


POINTS = [
    dict(point_id="P1_old_full_opt", b=15, S=0.176, E=0.419),
    dict(point_id="P2_dev_run1", b=17, S=0.303, E=0.404),
    dict(point_id="P3_dev_run0", b=31, S=1.045, E=0.405),
    dict(point_id="P4_normal", b=15, S=0.176, E=0.412),
]
ROUND_LABELS = ("round0", "round1", "round2")
ALL_LABELS = ROUND_LABELS + ("grossup",)
COEF_NAMES = ("a_P", "a_Q", "b_PP", "b_QQ", "b_PQ")
P_LOCAL_FACTORS = np.array([-0.25, -0.125, 0.0, 0.125, 0.25])
N_Q = 5
E_TEST_MULTIPLIERS = (0.35, 1.0, 3.0)
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REFERENCE_CALLS_PER_SECOND_LOW = 80.0
REFERENCE_CALLS_PER_SECOND = 200.0
REFERENCE_CALLS_PER_SECOND_HIGH = 230.0
PARAMETER_TOL = 1e-9
SIGN_TOL = 1e-10

CSV_FIELDS = [
    "point_id", "b", "S", "E", "solution_label", "coefficient_round",
    "scenario", "t", "p_center_mw", "P_net_mw", "Q_mvar",
    "p_slack_mw", "line_loss_mw", "a_P", "a_Q", "b_PP", "b_QQ", "b_PQ",
    "matrix_rank", "lambda_min_cost", "lambda_max_cost",
    "solver", "status", "objective", "lp_pk_proxy_mw",
]


class RunCounter:
    def __init__(self) -> None:
        self.runpp_calls = 0
        self.retry_events = 0
        self.diverged = 0

    def run(self, net, context: str) -> bool:
        self.runpp_calls += 1
        try:
            pp.runpp(net, numba=True, init="results")
            return True
        except Exception:
            self.retry_events += 1
        self.runpp_calls += 1
        try:
            pp.runpp(net, numba=True, init="flat")
            return True
        except Exception:
            self.diverged += 1
            print(f"pf_diverged={context}", flush=True)
            return False


def _paths() -> tuple[str, str]:
    stamp = datetime_.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(RESULTS_DIR, f"probe_gapB_{stamp}")
    return stem + ".csv", stem + "_report.md"


def _validate_runtime_parameters(net) -> dict[str, float]:
    p_sum = float(net.load["p_mw"].sum())
    q_sum = float(net.load["q_mvar"].sum())
    s_sum = float(np.hypot(p_sum, q_sum))
    pf = p_sum / s_sum
    values = {
        "load_p_sum_mw": p_sum,
        "load_q_sum_mvar": q_sum,
        "load_s_sum_mva": s_sum,
        "load_total_pf": pf,
        "K_P": float(PM.K_P),
        "K_Q": float(PM.K_Q),
        "ETA_PCS": float(PM.ETA_PCS),
        "SLACK_VM_PU": float(PM.SLACK_VM_PU),
    }
    for key, value in values.items():
        print(f"parameter_check {key}={value:.12g}", flush=True)
    checks = {
        "load_p_sum_mw": np.isclose(p_sum, 9.5, atol=PARAMETER_TOL, rtol=0.0),
        "load_q_sum_mvar": np.isclose(
            q_sum, 10.0 * np.sqrt(1.0 - 0.95**2),
            atol=PARAMETER_TOL, rtol=0.0,
        ),
        "load_s_sum_mva": np.isclose(
            s_sum, 10.0, atol=PARAMETER_TOL, rtol=0.0
        ),
        "load_total_pf": np.isclose(
            pf, 0.95, atol=PARAMETER_TOL, rtol=0.0
        ),
        "K_P": np.isclose(
            float(PM.K_P), 2.557201, atol=1e-6, rtol=0.0
        ),
        "K_Q": np.isclose(
            float(PM.K_Q), 1.357609, atol=1e-6, rtol=0.0
        ),
        "ETA_PCS": np.isclose(
            float(PM.ETA_PCS), 0.975, atol=PARAMETER_TOL, rtol=0.0
        ),
        "SLACK_VM_PU": np.isclose(
            float(PM.SLACK_VM_PU), 1.02, atol=PARAMETER_TOL, rtol=0.0
        ),
    }
    failed = [key for key, passed in checks.items() if not bool(passed)]
    print(f"parameter_check_failed_count={len(failed)}", flush=True)
    if failed:
        raise RuntimeError(
            "확정 물리 파라미터 불일치: " + ",".join(failed)
        )
    return values


def _e_test_values(S: float) -> tuple[float, ...]:
    return tuple(float(S * multiplier) for multiplier in E_TEST_MULTIPLIERS)


def _ensure_sgen(net) -> int:
    if len(net.sgen) == 0:
        pp.create_sgen(net, bus=1, p_mw=0.0, q_mvar=0.0, name="probe_gapB")
    return int(net.sgen.index[0])


def _set_load(net, base_p: np.ndarray, base_q: np.ndarray, scenario: str, t: int) -> None:
    scale = float(PM.LOAD[scenario][t])
    net.load["p_mw"] = base_p * scale
    net.load["q_mvar"] = base_q * scale


def _measure_baseline(
    net, base_p: np.ndarray, base_q: np.ndarray, sgen: int, counter: RunCounter,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    slack, loss = {}, {}
    for scenario in PM.ALL_DAYS:
        p_arr = np.full(PM.TIME_STEPS, np.nan)
        l_arr = np.full(PM.TIME_STEPS, np.nan)
        for t in range(PM.TIME_STEPS):
            _set_load(net, base_p, base_q, scenario, t)
            net.sgen.at[sgen, "p_mw"] = 0.0
            net.sgen.at[sgen, "q_mvar"] = 0.0
            ok = counter.run(net, f"baseline/{scenario}/t={t}")
            if ok:
                p_arr[t] = float(net.res_ext_grid.p_mw.sum())
                l_arr[t] = float(net.res_line.pl_mw.sum())
        slack[scenario], loss[scenario] = p_arr, l_arr
    return slack, loss


def _local_grid(S: float, center: float) -> list[tuple[float, float]]:
    p_values = np.unique(np.clip(center + S * P_LOCAL_FACTORS, -S, S))
    grid = []
    for p in p_values:
        q_max = float(np.sqrt(max(S * S - float(p) ** 2, 0.0)))
        for q in np.unique(np.linspace(0.0, q_max, N_Q)):
            grid.append((float(p), float(q)))
    return grid


def _grid_feasible(point: dict[str, Any], p: float, q: float) -> bool:
    return bool(SURFACE._grid_feasibility(point, p, q)[0])


def _measure_group(
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    sgen: int,
    counter: RunCounter,
    point: dict[str, Any],
    scenario: str,
    t: int,
    center: float,
    baseline_loss: float,
    *,
    measure_soc_infeasible: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for p, q in _local_grid(float(point["S"]), float(center)):
        feasible = _grid_feasible(point, p, q)
        row: dict[str, Any] = {
            "p_mw": p, "q_mvar": q, "feasible": False,
            "feasible_reason": "ok" if feasible else "soc_or_pcs",
            "L_cost_mw": "",
        }
        if not feasible and not measure_soc_infeasible:
            rows.append(row)
            continue
        if not np.isfinite(baseline_loss):
            row["feasible_reason"] = "baseline_pf_diverged"
            rows.append(row)
            continue
        if abs(p) <= 1e-15 and abs(q) <= 1e-15:
            loss = baseline_loss
            ok = True
        else:
            _set_load(net, base_p, base_q, scenario, t)
            pcs_loss = (1.0 - PM.ETA_PCS) * (np.hypot(p, q) - abs(p))
            net.sgen.at[sgen, "bus"] = int(point["b"])
            net.sgen.at[sgen, "p_mw"] = p - pcs_loss
            net.sgen.at[sgen, "q_mvar"] = q
            ok = counter.run(
                net,
                f"surface/{point['point_id']}/{scenario}/t={t}/"
                f"center={center:.9g}/p={p:.9g}/q={q:.9g}",
            )
            loss = float(net.res_line.pl_mw.sum()) if ok else np.nan
        if ok:
            row["feasible"] = feasible
            row["feasible_reason"] = "ok" if feasible else "measured_for_E_filter"
            row["L_cost_mw"] = loss - baseline_loss
        else:
            row["feasible_reason"] = "pf_diverged"
        rows.append(row)
    fit = SURFACE._fit_group(point, scenario, t, rows)
    fit["p_center_mw"] = float(center)
    return rows, fit


def _measure_coefficient_set(
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    sgen: int,
    counter: RunCounter,
    point: dict[str, Any],
    centers: dict[str, np.ndarray],
    baseline_loss: dict[str, np.ndarray],
    scenarios: tuple[str, ...] | list[str] | None = None,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    list[dict[str, Any]],
]:
    coef_map = {}
    coef_rows = []
    selected_scenarios = PM.ALL_DAYS if scenarios is None else scenarios
    for scenario in selected_scenarios:
        for t in range(PM.TIME_STEPS):
            _rows, fit = _measure_group(
                net, base_p, base_q, sgen, counter, point, scenario, t,
                float(centers[scenario][t]), float(baseline_loss[scenario][t]),
            )
            coef_map[(point["point_id"], scenario, t)] = fit
            coef_rows.append(fit)
            print(
                f"coef point={point['point_id']} scenario={scenario} t={t} "
                f"center={fit['p_center_mw']:.9g} rank={fit['matrix_rank']} "
                f"lambda_min={fit['lambda_min_cost']} calls={counter.runpp_calls}",
                flush=True,
            )
    return coef_map, coef_rows


def _validate_coefficient_set(
    point: dict[str, Any],
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, Any]:
    rows = [
        coef_map[(point["point_id"], scenario, t)]
        for scenario in PM.ALL_DAYS for t in range(PM.TIME_STEPS)
    ]
    rank_bad = sum(int(r["matrix_rank"]) < 5 for r in rows)
    psd_bad = sum(float(r["lambda_min_cost"]) < 0.0 for r in rows)
    peak_sign_bad = sum(
        float(coef_map[(point["point_id"], scenario, t)]["a_P"]) >= 0.0
        or float(coef_map[(point["point_id"], scenario, t)]["a_Q"]) >= 0.0
        for scenario in PM.PEAK_DAYS for t in range(PM.TIME_STEPS)
    )
    print(
        f"coefficient_check point={point['point_id']} rank_bad={rank_bad} "
        f"psd_bad={psd_bad} peak_sign_bad={peak_sign_bad}",
        flush=True,
    )
    assert peak_sign_bad == 0, (
        f"{point['point_id']}: 방전 loss_reduction 부호 불일치 "
        f"(a_P 또는 a_Q >= 0인 PEAK 시각 {peak_sign_bad}건)"
    )
    return {
        "rank_bad": rank_bad, "psd_bad": psd_bad,
        "peak_sign_bad": peak_sign_bad,
    }


def _solve_schedule_set(
    point: dict[str, Any],
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    output = {}
    for scenario in PM.ALL_DAYS:
        kind = "avg" if scenario in PM.AVG_DAYS else "peak"
        try:
            entry = JBASE._build_problem(
                kind, "B_peak_PQ", point, scenario, coef_map
            )
            result = JBASE._solve_fixed(entry)
        except Exception as exc:
            result = {
                "P": np.full(PM.TIME_STEPS, np.nan),
                "Q": np.full(PM.TIME_STEPS, np.nan),
                "status": f"build_exception:{type(exc).__name__}",
                "solver": "", "objective": np.nan, "pk": np.nan,
            }
            print(
                f"schedule_error point={point['point_id']} scenario={scenario} "
                f"error={type(exc).__name__}:{exc}",
                flush=True,
            )
        output[scenario] = result
        print(
            f"schedule point={point['point_id']} scenario={scenario} "
            f"status={result['status']} solver={result['solver']}",
            flush=True,
        )
    return output


def _validate_peak_loss_reduction(
    point: dict[str, Any],
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
    schedules: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    checked = 0
    bad = 0
    minimum = np.inf
    for scenario in PM.PEAK_DAYS:
        schedule = schedules[scenario]
        for t in range(PM.TIME_STEPS):
            P = float(schedule["P"][t])
            Q = float(schedule["Q"][t])
            if not (np.isfinite(P) and np.isfinite(Q)):
                continue
            if P <= SIGN_TOL:
                continue
            coef = coef_map[(point["point_id"], scenario, t)]
            reduction = (
                -float(coef["a_Q"]) * Q
                - float(coef["a_P"]) * P
            )
            checked += 1
            minimum = min(minimum, reduction)
            if not np.isfinite(reduction) or reduction <= SIGN_TOL:
                bad += 1
    print(
        f"peak_loss_reduction_check point={point['point_id']} "
        f"discharge_times={checked} bad={bad} "
        f"minimum={minimum if checked else np.nan}",
        flush=True,
    )
    assert bad == 0, (
        f"{point['point_id']}: P_net>0 PEAK 시각의 "
        f"loss_reduction<=0 또는 NaN {bad}건"
    )
    return {
        "peak_discharge_count": checked,
        "peak_loss_reduction_bad_count": bad,
        "peak_loss_reduction_min": (
            float(minimum) if checked else np.nan
        ),
    }


def _evaluate_schedule_set(
    writer,
    csv_file,
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    sgen: int,
    counter: RunCounter,
    point: dict[str, Any],
    label: str,
    coefficient_round: str,
    schedules: dict[str, dict[str, Any]],
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    slack, losses = {}, {}
    for scenario in PM.ALL_DAYS:
        p_slack = np.full(PM.TIME_STEPS, np.nan)
        line_loss = np.full(PM.TIME_STEPS, np.nan)
        schedule = schedules[scenario]
        for t in range(PM.TIME_STEPS):
            P, Q = float(schedule["P"][t]), float(schedule["Q"][t])
            if np.isfinite(P) and np.isfinite(Q):
                _set_load(net, base_p, base_q, scenario, t)
                pcs_loss = (1.0 - PM.ETA_PCS) * (np.hypot(P, Q) - abs(P))
                net.sgen.at[sgen, "bus"] = int(point["b"])
                net.sgen.at[sgen, "p_mw"] = P - pcs_loss
                net.sgen.at[sgen, "q_mvar"] = Q
                ok = counter.run(
                    net, f"schedule/{point['point_id']}/{label}/{scenario}/t={t}"
                )
                if ok:
                    p_slack[t] = float(net.res_ext_grid.p_mw.sum())
                    line_loss[t] = float(net.res_line.pl_mw.sum())
            coef = coef_map[(point["point_id"], scenario, t)]
            writer.writerow({
                "point_id": point["point_id"], "b": point["b"],
                "S": point["S"], "E": point["E"],
                "solution_label": label, "coefficient_round": coefficient_round,
                "scenario": scenario, "t": t,
                "p_center_mw": coef.get("p_center_mw", np.nan),
                "P_net_mw": P, "Q_mvar": Q,
                "p_slack_mw": p_slack[t], "line_loss_mw": line_loss[t],
                **{name: coef[name] for name in COEF_NAMES},
                "matrix_rank": coef["matrix_rank"],
                "lambda_min_cost": coef["lambda_min_cost"],
                "lambda_max_cost": coef["lambda_max_cost"],
                "solver": schedule["solver"], "status": schedule["status"],
                "objective": schedule["objective"],
                "lp_pk_proxy_mw": schedule["pk"],
            })
            csv_file.flush()
        slack[scenario], losses[scenario] = p_slack, line_loss
    return slack, losses


def _benefits_for(
    point: dict[str, Any],
    base_slack: dict[str, np.ndarray],
    slack: dict[str, np.ndarray],
) -> dict[str, float]:
    b_energy = benefits.b_energy(
        base_slack, slack, PM.SMP_PER_MWH, PM.N_WEEKDAYS
    )
    b_defer = benefits.b_defer(base_slack, slack)
    capex = benefits.capex(float(point["S"]), float(point["E"]))
    opex = benefits.opex(float(point["S"]), float(point["E"]))
    cost = benefits.total_cost(float(point["S"]), float(point["E"]))
    j_net = benefits.j_net(
        b_energy, b_defer, float(point["S"]), float(point["E"])
    )
    return {
        "b_energy": b_energy, "b_defer": b_defer, "capex": capex,
        "opex": opex, "cost": cost, "j_net": j_net,
    }


def _scale_coefficients(
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
    scale: float,
) -> dict[tuple[str, str, int], dict[str, Any]]:
    output = {}
    for key, row in coef_map.items():
        new = dict(row)
        for name in COEF_NAMES:
            new[name] = float(row[name]) * scale
        new["lambda_min_cost"] = float(row["lambda_min_cost"]) * scale
        new["lambda_max_cost"] = float(row["lambda_max_cost"]) * scale
        output[key] = new
    return output


def _model_cost(coef: dict[str, Any], P: float, Q: float) -> float:
    return (
        float(coef["a_P"]) * P + float(coef["a_Q"]) * Q
        + float(coef["b_PP"]) * P * P
        + float(coef["b_QQ"]) * Q * Q
        + float(coef["b_PQ"]) * P * Q
    )


def _grossup_alpha(
    point: dict[str, Any],
    coef0: dict[tuple[str, str, int], dict[str, Any]],
    schedules0: dict[str, dict[str, Any]],
    actual_losses0: dict[str, np.ndarray],
    baseline_loss: dict[str, np.ndarray],
) -> float:
    model, actual = [], []
    for scenario in PM.ALL_DAYS:
        for t in range(PM.TIME_STEPS):
            P = float(schedules0[scenario]["P"][t])
            Q = float(schedules0[scenario]["Q"][t])
            if not (
                np.isfinite(P) and np.isfinite(Q)
                and np.isfinite(actual_losses0[scenario][t])
                and np.isfinite(baseline_loss[scenario][t])
            ):
                continue
            model.append(_model_cost(
                coef0[(point["point_id"], scenario, t)], P, Q
            ))
            actual.append(
                float(actual_losses0[scenario][t] - baseline_loss[scenario][t])
            )
    model_arr = np.asarray(model)
    actual_arr = np.asarray(actual)
    denom = float(np.dot(model_arr, model_arr))
    return float(np.dot(actual_arr, model_arr) / denom) if denom > 0 else np.nan


def _relative_coef_rows(
    all_coef: dict[str, dict[str, dict[tuple[str, str, int], dict[str, Any]]]]
) -> list[dict[str, Any]]:
    output = []
    for point in POINTS:
        point_id = point["point_id"]
        c0, c1 = all_coef[point_id]["round0"], all_coef[point_id]["round1"]
        for name in COEF_NAMES:
            rel = []
            for scenario in PM.ALL_DAYS:
                for t in range(PM.TIME_STEPS):
                    x = float(c0[(point_id, scenario, t)][name])
                    y = float(c1[(point_id, scenario, t)][name])
                    rel.append(abs(y - x) / max(abs(x), 1e-12))
            arr = np.asarray(rel)
            output.append({
                "point_id": point_id, "coefficient": name,
                "relative_change_median": float(np.median(arr)),
                "relative_change_p95": float(np.percentile(arr, 95)),
                "relative_change_max": float(np.max(arr)),
            })
    return output


def _measure_E_independence(
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    sgen: int,
    counter: RunCounter,
    baseline_loss: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    unique = {}
    for point in POINTS:
        unique[(int(point["b"]), float(point["S"]))] = point
    group_coefficients: dict[
        tuple[int, float, str, int, float], dict[str, Any]
    ] = {}
    for (bus, S), source in unique.items():
        e_values = _e_test_values(S)
        measurement_point = dict(
            point_id=f"Etest_b{bus}_S{S}", b=bus, S=S, E=max(e_values)
        )
        for scenario in PM.ALL_DAYS:
            for t in range(PM.TIME_STEPS):
                raw_rows, _fit_unused = _measure_group(
                    net, base_p, base_q, sgen, counter, measurement_point,
                    scenario, t, 0.0, float(baseline_loss[scenario][t]),
                    measure_soc_infeasible=True,
                )
                for E in e_values:
                    point_e = dict(measurement_point, E=E)
                    filtered = []
                    for raw in raw_rows:
                        row = dict(raw)
                        row["feasible"] = (
                            raw["L_cost_mw"] not in ("", None)
                            and _grid_feasible(
                                point_e, float(raw["p_mw"]), float(raw["q_mvar"])
                            )
                        )
                        filtered.append(row)
                    fit = SURFACE._fit_group(point_e, scenario, t, filtered)
                    group_coefficients[(bus, S, scenario, t, E)] = fit
    output = []
    for bus, S in unique:
        e_values = _e_test_values(S)
        for name in COEF_NAMES:
            rel_spans = []
            rank_bad = 0
            psd_bad = 0
            for scenario in PM.ALL_DAYS:
                for t in range(PM.TIME_STEPS):
                    fits = [
                        group_coefficients[(bus, S, scenario, t, E)]
                        for E in e_values
                    ]
                    rank_bad += sum(int(f["matrix_rank"]) < 5 for f in fits)
                    psd_bad += sum(
                        float(f["lambda_min_cost"]) < 0.0 for f in fits
                    )
                    vals = np.asarray([float(f[name]) for f in fits])
                    if np.all(np.isfinite(vals)):
                        rel_spans.append(
                            float((np.max(vals) - np.min(vals))
                                  / max(np.max(np.abs(vals)), 1e-12))
                        )
            arr = np.asarray(rel_spans)
            output.append({
                "b": bus, "S": S, "E_values": str(e_values),
                "coefficient": name, "n_groups": len(arr),
                "relative_span_median": (
                    float(np.median(arr)) if len(arr) else np.nan
                ),
                "relative_span_max": (
                    float(np.max(arr)) if len(arr) else np.nan
                ),
                "rank_lt_5_count_across_E": rank_bad,
                "psd_violation_count_across_E": psd_bad,
            })
    return output


def _table1_rows(
    results: dict[str, dict[str, dict[str, float]]]
) -> list[dict[str, Any]]:
    rows = []
    for point in POINTS:
        point_id = point["point_id"]
        r0, r1, r2 = (
            results[point_id]["round0"],
            results[point_id]["round1"],
            results[point_id]["round2"],
        )
        d10, d21 = r1["j_net"] - r0["j_net"], r2["j_net"] - r1["j_net"]
        rows.append({
            "point_id": point_id,
            "j_net_round0": r0["j_net"],
            "j_net_round1": r1["j_net"],
            "j_net_round2": r2["j_net"],
            "b_energy_round0": r0["b_energy"],
            "b_energy_round1": r1["b_energy"],
            "b_energy_round2": r2["b_energy"],
            "b_defer_round0": r0["b_defer"],
            "b_defer_round1": r1["b_defer"],
            "b_defer_round2": r2["b_defer"],
            "delta_round1_minus_round0": d10,
            "delta_round1_minus_round0_pct_abs_j0": (
                100.0 * d10 / abs(r0["j_net"]) if r0["j_net"] != 0 else np.nan
            ),
            "delta_round2_minus_round1": d21,
            "delta_b_energy_1_minus_0": r1["b_energy"] - r0["b_energy"],
            "delta_b_defer_1_minus_0": r1["b_defer"] - r0["b_defer"],
            "delta_b_energy_2_minus_1": r2["b_energy"] - r1["b_energy"],
            "delta_b_defer_2_minus_1": r2["b_defer"] - r1["b_defer"],
            "delta10_sign": (
                "positive" if d10 > 0 else "negative" if d10 < 0 else "zero"
            ),
            "delta21_sign": (
                "positive" if d21 > 0 else "negative" if d21 < 0 else "zero"
            ),
        })
    return rows


def _grossup_rows(
    results: dict[str, dict[str, dict[str, float]]],
    alphas: dict[str, float],
) -> list[dict[str, Any]]:
    rows = []
    for point in POINTS:
        point_id = point["point_id"]
        j0 = results[point_id]["round0"]["j_net"]
        j1 = results[point_id]["round1"]["j_net"]
        jg = results[point_id]["grossup"]["j_net"]
        denom = abs(j1 - j0)
        rows.append({
            "point_id": point_id,
            "alpha_actual_over_model_ls": alphas[point_id],
            "coefficient_scale_1_over_alpha": (
                1.0 / alphas[point_id]
                if np.isfinite(alphas[point_id]) and alphas[point_id] != 0
                else np.nan
            ),
            "j_net_round0": j0, "j_net_round1": j1,
            "j_net_grossup": jg,
            "grossup_minus_round1": jg - j1,
            "absolute_gap_fraction_closed": (
                1.0 - abs(jg - j1) / denom if denom > 0 else np.nan
            ),
        })
    return rows


def _status_rows(
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
    checks: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for point in POINTS:
        point_id = point["point_id"]
        for label in ALL_LABELS:
            for scenario in PM.ALL_DAYS:
                result = schedules[point_id][label][scenario]
                check = checks[point_id][label]
                rows.append({
                    "point_id": point_id, "solution_label": label,
                    "scenario": scenario, "solver": result["solver"],
                    "status": result["status"],
                    "rank_lt_5_count": check["rank_bad"],
                    "psd_violation_count": check["psd_bad"],
                    "peak_sign_bad_count": check["peak_sign_bad"],
                    "peak_discharge_count": check[
                        "peak_discharge_count"
                    ],
                    "peak_loss_reduction_bad_count": check[
                        "peak_loss_reduction_bad_count"
                    ],
                    "peak_loss_reduction_min": check[
                        "peak_loss_reduction_min"
                    ],
                    "hessian_projection_count": 0,
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
    lines = ["# probe_gapB", "", "## 표 1", ""] + _md_table(table1)
    for number, table in (
        (2, table2), (3, table3), (4, table4), (5, table5)
    ):
        if table is not None:
            lines += ["", f"## 표 {number}", ""] + _md_table(table)
    if environment is not None:
        lines += ["", "## 환경", ""] + _md_table(environment)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _expected_counts() -> dict[str, float]:
    baseline = len(PM.ALL_DAYS) * PM.TIME_STEPS
    round0_feasible_rows_per_time = sum(
        sum(
            _grid_feasible(point, p, q)
            for p, q in _local_grid(float(point["S"]), 0.0)
        )
        for point in POINTS
    )
    round0_surface_exact = (
        round0_feasible_rows_per_time - len(POINTS)
    ) * len(PM.ALL_DAYS) * PM.TIME_STEPS
    # 국소 격자는 최대 25점이고 각 그룹의 (0,0)은 baseline을 재사용한다.
    core_surface_upper = (
        len(POINTS) * len(PM.ALL_DAYS) * PM.TIME_STEPS * 24
        + len(POINTS) * (len(ROUND_LABELS) - 1)
        * len(PM.ALL_DAYS) * PM.TIME_STEPS * 25
    )
    unique_bs = len({(p["b"], p["S"]) for p in POINTS})
    e_surface = unique_bs * len(PM.ALL_DAYS) * PM.TIME_STEPS * 24
    schedule_ac = len(POINTS) * len(ALL_LABELS) * len(PM.ALL_DAYS) * PM.TIME_STEPS
    total = baseline + core_surface_upper + e_surface + schedule_ac
    return {
        "coefficient_measurement_groups": (
            len(POINTS) * len(ROUND_LABELS)
            * len(PM.ALL_DAYS) * PM.TIME_STEPS
        ),
        "round0_feasible_grid_rows_per_time_all_points": (
            round0_feasible_rows_per_time
        ),
        "round0_surface_runpp_exact_no_retry": round0_surface_exact,
        "E_refit_groups": (
            unique_bs * len(PM.ALL_DAYS) * PM.TIME_STEPS
            * len(E_TEST_MULTIPLIERS)
        ),
        "baseline_runpp_no_retry": baseline,
        "core_surface_runpp_upper_no_retry": core_surface_upper,
        "E_surface_runpp_no_retry": e_surface,
        "schedule_runpp_no_retry": schedule_ac,
        "total_runpp_upper_no_retry": total,
        "expected_seconds_at_200_calls_per_s": (
            total / REFERENCE_CALLS_PER_SECOND
        ),
        "expected_seconds_at_230_calls_per_s": total / REFERENCE_CALLS_PER_SECOND_HIGH,
        "expected_seconds_at_80_calls_per_s": total / REFERENCE_CALLS_PER_SECOND_LOW,
    }


def _environment(
    started: float,
    ac_elapsed: float,
    counter: RunCounter,
    expected: dict[str, float],
    runtime_parameters: dict[str, float],
) -> list[dict[str, Any]]:
    elapsed = time.perf_counter() - started
    rows = [
        {"item": "timestamp", "value": datetime_.datetime.now().isoformat(timespec="seconds")},
        {"item": "hostname", "value": socket.gethostname()},
        {"item": "platform", "value": platform.platform()},
        {"item": "python", "value": platform.python_version()},
        {"item": "numpy", "value": np.__version__},
        {"item": "pandapower", "value": pp.__version__},
        {"item": "installed_solvers", "value": ",".join(JBASE.cp.installed_solvers())},
        {"item": "POLY_N", "value": JBASE.POLY_N},
        {"item": "E_test_multipliers", "value": str(E_TEST_MULTIPLIERS)},
        {"item": "runpp_calls_total", "value": counter.runpp_calls},
        {"item": "pf_retry_events", "value": counter.retry_events},
        {"item": "pf_diverged_calls", "value": counter.diverged},
        {"item": "ac_elapsed_s", "value": f"{ac_elapsed:.9f}"},
        {"item": "runpp_calls_per_ac_s", "value": (
            f"{counter.runpp_calls/ac_elapsed:.9f}" if ac_elapsed > 0 else "nan"
        )},
        {"item": "total_execution_time_s", "value": f"{elapsed:.9f}"},
        {"item": "alpha_definition", "value": "sum(actual_Lcost*model_Lcost)/sum(model_Lcost^2)"},
        {"item": "grossup_scale_definition", "value": "1/alpha"},
        {"item": "hessian_projection_count", "value": 0},
    ]
    rows.extend(
        {"item": key, "value": value}
        for key, value in runtime_parameters.items()
    )
    rows.extend({"item": key, "value": value} for key, value in expected.items())
    for solver, package in (
        ("CLARABEL", "clarabel"), ("SCS", "scs"),
        ("OSQP", "osqp"), ("SCIPY", "scipy"), ("HIGHS", "highspy"),
    ):
        if solver in JBASE.cp.installed_solvers():
            rows.append({"item": f"solver_{solver}_version", "value": _version(package)})
    return rows


def main() -> int:
    started = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path, report_path = _paths()
    JBASE._validate_benefits_signatures()
    net = build_net()
    runtime_parameters = _validate_runtime_parameters(net)
    expected = _expected_counts()
    for key, value in expected.items():
        print(f"{key}={value}", flush=True)

    counter = RunCounter()
    base_p = net.load["p_mw"].to_numpy().copy()
    base_q = net.load["q_mvar"].to_numpy().copy()
    sgen = _ensure_sgen(net)
    ac_started = time.perf_counter()
    base_slack, baseline_loss = _measure_baseline(
        net, base_p, base_q, sgen, counter
    )

    all_coef: dict[
        str, dict[str, dict[tuple[str, str, int], dict[str, Any]]]
    ] = defaultdict(dict)
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    ac_slack: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    ac_losses: dict[str, dict[str, dict[str, np.ndarray]]] = defaultdict(dict)
    results: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    checks: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    alphas: dict[str, float] = {}

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_file.flush()
        for point in POINTS:
            point_id = point["point_id"]
            centers = {
                scenario: np.zeros(PM.TIME_STEPS) for scenario in PM.ALL_DAYS
            }
            coef_rows_by_round = {}
            for round_index, label in enumerate(ROUND_LABELS):
                coef_map, coef_rows = _measure_coefficient_set(
                    net, base_p, base_q, sgen, counter, point,
                    centers, baseline_loss,
                )
                all_coef[point_id][label] = coef_map
                coef_rows_by_round[label] = coef_rows
                checks[point_id][label] = _validate_coefficient_set(
                    point, coef_map
                )
                schedule = _solve_schedule_set(point, coef_map)
                schedules[point_id][label] = schedule
                checks[point_id][label].update(
                    _validate_peak_loss_reduction(
                        point, coef_map, schedule
                    )
                )
                slack, losses = _evaluate_schedule_set(
                    writer, csv_file, net, base_p, base_q, sgen, counter,
                    point, label, label, schedule, coef_map,
                )
                ac_slack[point_id][label] = slack
                ac_losses[point_id][label] = losses
                results[point_id][label] = _benefits_for(
                    point, base_slack, slack
                )
                if round_index < len(ROUND_LABELS) - 1:
                    centers = {
                        scenario: np.asarray(schedule[scenario]["P"], dtype=float).copy()
                        for scenario in PM.ALL_DAYS
                    }

            alpha = _grossup_alpha(
                point, all_coef[point_id]["round0"],
                schedules[point_id]["round0"],
                ac_losses[point_id]["round0"], baseline_loss,
            )
            alphas[point_id] = alpha
            scale = 1.0 / alpha if np.isfinite(alpha) and alpha > 0 else np.nan
            gross_coef = (
                _scale_coefficients(all_coef[point_id]["round0"], scale)
                if np.isfinite(scale) else all_coef[point_id]["round0"]
            )
            checks[point_id]["grossup"] = _validate_coefficient_set(
                point, gross_coef
            )
            gross_schedule = _solve_schedule_set(point, gross_coef)
            schedules[point_id]["grossup"] = gross_schedule
            checks[point_id]["grossup"].update(
                _validate_peak_loss_reduction(
                    point, gross_coef, gross_schedule
                )
            )
            gross_slack, gross_losses = _evaluate_schedule_set(
                writer, csv_file, net, base_p, base_q, sgen, counter,
                point, "grossup", "round0", gross_schedule, gross_coef,
            )
            ac_slack[point_id]["grossup"] = gross_slack
            ac_losses[point_id]["grossup"] = gross_losses
            results[point_id]["grossup"] = _benefits_for(
                point, base_slack, gross_slack
            )

    table1 = _table1_rows(results)
    _write_report(report_path, table1)
    print("TABLE_1", flush=True)
    for row in table1:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    delta10_signs = {
        row["delta10_sign"] for row in table1
        if np.isfinite(float(row["delta_round1_minus_round0"]))
    }
    print(
        f"delta_round1_minus_round0_signs_mixed="
        f"{'positive' in delta10_signs and 'negative' in delta10_signs}",
        flush=True,
    )

    table2 = _relative_coef_rows(all_coef)
    _write_report(report_path, table1, table2)
    print("TABLE_2", flush=True)
    for row in table2:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table3 = _measure_E_independence(
        net, base_p, base_q, sgen, counter, baseline_loss
    )
    _write_report(report_path, table1, table2, table3)
    print("TABLE_3", flush=True)
    for row in table3:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table4 = _grossup_rows(results, alphas)
    _write_report(report_path, table1, table2, table3, table4)
    print("TABLE_4", flush=True)
    for row in table4:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table5 = _status_rows(schedules, checks)
    ac_elapsed = time.perf_counter() - ac_started
    environment = _environment(
        started, ac_elapsed, counter, expected, runtime_parameters
    )
    _write_report(
        report_path, table1, table2, table3, table4, table5, environment
    )
    print("TABLE_5", flush=True)
    for row in table5:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"runpp_calls={counter.runpp_calls}", flush=True)
    print(f"total_execution_time_s={time.perf_counter()-started:.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
