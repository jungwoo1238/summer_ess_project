"""(P,Q) 2D 손실 QP 프로토타입 v2: 비용 부호·적응형 격자·고정 solver 재측정.

본코드는 수정하지 않는다. 실행은 사용자가 직접 수행한다.

    python scripts/probe_pq_loss_v2.py

실행 시작 전에 적응형 격자의 전체/feasible 조합 수, retry가 없을 때의 예상 runpp 호출
수, 직전 실측 100 calls/s 기준 예상 시간을 출력한다. AC 곡면은 수 분 내외가 예상된다.
원자료·계수 CSV는 (통제점, 시나리오, 시각) 그룹마다 flush하며, 표 1과 1b를 보고서에
가장 먼저 기록한다. 헤시안 PSD 사영은 사용하지 않는다.
"""

from __future__ import annotations

import csv
import datetime as datetime_
import glob
import importlib.metadata
import os
import platform
import socket
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np
import pandapower as pp


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import lower_lp
import params as PM
from build_net import build_net
from probe_q_value import POINTS
from probe_pq_loss import RunCounter


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
P_FACTORS = np.linspace(-1.0, 1.0, 9)
N_Q_ADAPTIVE = 5
POLY_N = 128
REFERENCE_RUNPP_PER_SECOND = 100.0
BASELINE_DEFINITION = "loss_p0_q0_same_point_scenario_time"
GRID_DEFINITION = (
    "p=S*[-1,-.75,-.5,-.25,0,.25,.5,.75,1];"
    "q=unique(linspace(0,sqrt(max(S^2-p^2,0)),5))"
)
FEASIBILITY_DEFINITION = "pcs_circle_and_1h_using_(soc_max-soc_min)*E"
COEF_NAMES = ("a_P", "a_Q", "b_PP", "b_QQ", "b_PQ")

RAW_FIELDS = [
    "point_id", "b", "S", "E", "scenario", "t", "p_index", "q_index",
    "p_mw", "q_mvar", "q_max_mvar", "grid_feasible", "feasible",
    "feasible_reason", "baseline_loss_mw", "loss_ess_mw",
    "dL_reduction_mw", "L_cost_mw", "design_rank", "n_fit_samples",
    "baseline_definition", "grid_definition", "feasibility_definition", "source",
]
COEF_FIELDS = [
    "point_id", "b", "S", "E", "scenario", "t", "n_feasible", "matrix_rank",
    "n_unique_p", "n_unique_q", "a_P", "a_Q", "b_PP", "b_QQ", "b_PQ",
    "relative_max_residual", "rmse_mw", "lambda_min_cost", "lambda_max_cost",
    "lambda_min_over_lambda_max_pct", "is_psd_cost",
]


@dataclass
class AvgProblem:
    problem: cp.Problem
    params: dict[str, Any]
    vars: dict[str, Any]
    include_loss: bool
    force_q_zero: bool


def _paths() -> tuple[str, str, str]:
    stamp = datetime_.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(RESULTS_DIR, f"probe_pq_loss_v2_{stamp}")
    return stem + ".csv", stem + "_coef.csv", stem + "_report.md"


def _adaptive_grid(point: dict[str, Any]) -> list[dict[str, Any]]:
    S = float(point["S"])
    grid = []
    for pi, factor in enumerate(P_FACTORS):
        p = S * float(factor)
        q_max = float(np.sqrt(max(S * S - p * p, 0.0)))
        q_values = np.unique(np.linspace(0.0, q_max, N_Q_ADAPTIVE))
        for qi, q in enumerate(q_values):
            grid.append({
                "p_index": pi, "q_index": qi, "p_mw": float(p),
                "q_mvar": float(q), "q_max_mvar": q_max,
            })
    return grid


def _grid_feasibility(point: dict[str, Any], p: float, q: float) -> tuple[bool, str]:
    S, E = float(point["S"]), float(point["E"])
    if np.hypot(p, q) > S + 1e-12:
        return False, "pcs_circle"
    usable_mwh = (PM.SOC_MAX_FRAC - PM.SOC_MIN_FRAC) * E
    max_discharge = usable_mwh * PM.ETA_D / PM.DT_HOURS
    max_charge = usable_mwh / (PM.ETA_C * PM.DT_HOURS)
    if p > max_discharge + 1e-12:
        return False, "soc_discharge_1h"
    if -p > max_charge + 1e-12:
        return False, "soc_charge_1h"
    return True, "ok"


def _expected_counts() -> dict[str, float]:
    rows_per_day = 0
    feasible_per_day = 0
    for point in POINTS:
        grid = _adaptive_grid(point)
        rows_per_day += len(grid)
        feasible_per_day += sum(
            _grid_feasibility(point, g["p_mw"], g["q_mvar"])[0] for g in grid
        )
    multiplier = len(PM.AVG_DAYS) * PM.TIME_STEPS
    total_rows = rows_per_day * multiplier
    feasible = feasible_per_day * multiplier
    # 각 그룹의 (0,0) 표본은 별도 baseline 호출 결과를 그대로 사용하므로,
    # baseline 1회 + 나머지 feasible 표본 = feasible 표본 수와 같다.
    expected_runpp = feasible
    return {
        "grid_rows": total_rows,
        "feasible_grid_rows": feasible,
        "expected_runpp_calls_no_retry_no_reuse": expected_runpp,
        "expected_seconds_at_100_calls_per_s": expected_runpp / REFERENCE_RUNPP_PER_SECOND,
    }


def _row_key(
    point_id: str, scenario: str, t: int, p_index: int, q_index: int
) -> tuple[Any, ...]:
    return point_id, scenario, int(t), int(p_index), int(q_index)


def _load_reusable(current_path: str) -> tuple[dict[tuple[Any, ...], dict[str, str]], str]:
    candidates = sorted(
        p for p in glob.glob(os.path.join(RESULTS_DIR, "probe_pq_loss_v2_*.csv"))
        if not p.endswith("_coef.csv") and os.path.abspath(p) != os.path.abspath(current_path)
    )
    if not candidates:
        return {}, ""
    path = candidates[-1]
    points = {str(p["point_id"]): p for p in POINTS}
    reusable: dict[tuple[Any, ...], dict[str, str]] = {}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("baseline_definition") != BASELINE_DEFINITION:
                    continue
                if row.get("grid_definition") != GRID_DEFINITION:
                    continue
                if row.get("feasibility_definition") != FEASIBILITY_DEFINITION:
                    continue
                if row.get("feasible_reason") in {"pf_diverged", "baseline_pf_diverged"}:
                    continue
                try:
                    point = points[row["point_id"]]
                    if (
                        int(row["b"]) != int(point["b"])
                        or float(row["S"]) != float(point["S"])
                        or float(row["E"]) != float(point["E"])
                    ):
                        continue
                    key = _row_key(
                        row["point_id"], row["scenario"], int(row["t"]),
                        int(row["p_index"]), int(row["q_index"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                reusable[key] = row
    except (OSError, csv.Error):
        return {}, ""
    return reusable, path


def _ensure_sgen(net) -> int:
    if len(net.sgen) == 0:
        pp.create_sgen(net, bus=1, p_mw=0.0, q_mvar=0.0, name="probe_pq_loss_v2")
    return int(net.sgen.index[0])


def _set_load(net, base_p: np.ndarray, base_q: np.ndarray, scenario: str, t: int) -> None:
    scale = float(PM.LOAD[scenario][t])
    net.load["p_mw"] = base_p * scale
    net.load["q_mvar"] = base_q * scale


def _fit_group(
    point: dict[str, Any], scenario: str, t: int, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    samples = [
        r for r in rows
        if bool(r["feasible"]) and r["L_cost_mw"] not in ("", None)
    ]
    X = np.asarray([
        [
            float(r["p_mw"]), float(r["q_mvar"]),
            float(r["p_mw"]) ** 2, float(r["q_mvar"]) ** 2,
            float(r["p_mw"]) * float(r["q_mvar"]),
        ]
        for r in samples
    ], dtype=float)
    y = np.asarray([float(r["L_cost_mw"]) for r in samples], dtype=float)
    rank = int(np.linalg.matrix_rank(X)) if len(X) else 0
    unique_p = len({round(float(r["p_mw"]), 12) for r in samples})
    unique_q = len({round(float(r["q_mvar"]), 12) for r in samples})
    if len(X) >= 5 and rank == 5:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        residual = X @ beta - y
        denom = float(np.max(np.abs(y)))
        rel = float(np.max(np.abs(residual)) / denom) if denom > 0 else 0.0
        rmse = float(np.sqrt(np.mean(residual**2)))
        H = np.array([
            [beta[2], beta[4] / 2.0],
            [beta[4] / 2.0, beta[3]],
        ])
        eig = np.linalg.eigvalsh(H)
        lmin, lmax = float(eig[0]), float(eig[1])
        ratio = float(100.0 * lmin / lmax) if lmax != 0 else float("nan")
        is_psd = bool(lmin >= 0.0)
    else:
        beta = np.full(5, np.nan)
        rel = rmse = lmin = lmax = ratio = float("nan")
        is_psd = False
    return {
        "point_id": point["point_id"], "b": int(point["b"]),
        "S": float(point["S"]), "E": float(point["E"]),
        "scenario": scenario, "t": t, "n_feasible": len(samples),
        "matrix_rank": rank, "n_unique_p": unique_p, "n_unique_q": unique_q,
        **{name: float(beta[i]) for i, name in enumerate(COEF_NAMES)},
        "relative_max_residual": rel, "rmse_mw": rmse,
        "lambda_min_cost": lmin, "lambda_max_cost": lmax,
        "lambda_min_over_lambda_max_pct": ratio, "is_psd_cost": is_psd,
    }


def _measure_and_fit(
    raw_path: str,
    coef_path: str,
    reusable: dict[tuple[Any, ...], dict[str, str]],
    counter: RunCounter,
) -> tuple[
    list[dict[str, Any]],
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, int],
]:
    net = build_net()
    base_p = net.load["p_mw"].to_numpy().copy()
    base_q = net.load["q_mvar"].to_numpy().copy()
    sgen = _ensure_sgen(net)
    coef_rows: list[dict[str, Any]] = []
    coef_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    counts = {"reused": 0, "measured": 0, "infeasible": 0, "pf_diverged": 0}

    with (
        open(raw_path, "w", newline="", encoding="utf-8-sig") as raw_f,
        open(coef_path, "w", newline="", encoding="utf-8-sig") as coef_f,
    ):
        raw_writer = csv.DictWriter(raw_f, fieldnames=RAW_FIELDS)
        coef_writer = csv.DictWriter(coef_f, fieldnames=COEF_FIELDS)
        raw_writer.writeheader()
        coef_writer.writeheader()
        raw_f.flush()
        coef_f.flush()

        for point in POINTS:
            point_id = str(point["point_id"])
            bus, S, E = int(point["b"]), float(point["S"]), float(point["E"])
            grid = _adaptive_grid(point)
            for scenario in PM.AVG_DAYS:
                for t in range(PM.TIME_STEPS):
                    old_group = {
                        (g["p_index"], g["q_index"]): reusable.get(
                            _row_key(
                                point_id, scenario, t, g["p_index"], g["q_index"]
                            )
                        )
                        for g in grid
                    }
                    baseline_candidates = [
                        float(r["baseline_loss_mw"])
                        for r in old_group.values()
                        if r is not None and r.get("baseline_loss_mw") not in ("", None)
                        and np.isfinite(float(r["baseline_loss_mw"]))
                    ]
                    baseline = baseline_candidates[0] if baseline_candidates else None
                    missing_feasible = any(
                        old_group[(g["p_index"], g["q_index"])] is None
                        and _grid_feasibility(point, g["p_mw"], g["q_mvar"])[0]
                        for g in grid
                    )
                    if missing_feasible and baseline is None:
                        _set_load(net, base_p, base_q, scenario, t)
                        net.sgen.at[sgen, "bus"] = bus
                        net.sgen.at[sgen, "p_mw"] = 0.0
                        net.sgen.at[sgen, "q_mvar"] = 0.0
                        ok = counter.run(net, f"baseline/{point_id}/{scenario}/t={t}")
                        baseline = float(net.res_line.pl_mw.sum()) if ok else float("nan")
                    if baseline is None:
                        baseline = float("nan")

                    group_rows: list[dict[str, Any]] = []
                    for g in grid:
                        old = old_group[(g["p_index"], g["q_index"])]
                        if old is not None:
                            row: dict[str, Any] = {
                                name: old.get(name, "") for name in RAW_FIELDS
                            }
                            row["p_index"] = int(row["p_index"])
                            row["q_index"] = int(row["q_index"])
                            row["p_mw"] = float(row["p_mw"])
                            row["q_mvar"] = float(row["q_mvar"])
                            row["q_max_mvar"] = float(row["q_max_mvar"])
                            row["grid_feasible"] = (
                                str(row["grid_feasible"]).lower() == "true"
                            )
                            row["feasible"] = str(row["feasible"]).lower() == "true"
                            if row["loss_ess_mw"] not in ("", None):
                                row["loss_ess_mw"] = float(row["loss_ess_mw"])
                            if row["dL_reduction_mw"] not in ("", None):
                                row["dL_reduction_mw"] = float(row["dL_reduction_mw"])
                            if row["L_cost_mw"] not in ("", None):
                                row["L_cost_mw"] = float(row["L_cost_mw"])
                            row["baseline_loss_mw"] = float(row["baseline_loss_mw"])
                            row["source"] = "reused"
                            counts["reused"] += 1
                        else:
                            p, q = g["p_mw"], g["q_mvar"]
                            grid_ok, reason = _grid_feasibility(point, p, q)
                            row = {
                                "point_id": point_id, "b": bus, "S": S, "E": E,
                                "scenario": scenario, "t": t,
                                "p_index": g["p_index"], "q_index": g["q_index"],
                                "p_mw": p, "q_mvar": q, "q_max_mvar": g["q_max_mvar"],
                                "grid_feasible": grid_ok, "feasible": False,
                                "feasible_reason": reason,
                                "baseline_loss_mw": baseline, "loss_ess_mw": "",
                                "dL_reduction_mw": "", "L_cost_mw": "",
                                "design_rank": "", "n_fit_samples": "",
                                "baseline_definition": BASELINE_DEFINITION,
                                "grid_definition": GRID_DEFINITION,
                                "feasibility_definition": FEASIBILITY_DEFINITION,
                                "source": "infeasible",
                            }
                            if not grid_ok:
                                counts["infeasible"] += 1
                            elif not np.isfinite(baseline):
                                row["feasible_reason"] = "baseline_pf_diverged"
                                row["source"] = "measured"
                                counts["pf_diverged"] += 1
                            elif abs(p) <= 1e-15 and abs(q) <= 1e-15:
                                row["feasible"] = True
                                row["feasible_reason"] = "ok"
                                row["loss_ess_mw"] = baseline
                                row["dL_reduction_mw"] = 0.0
                                row["L_cost_mw"] = 0.0
                                row["source"] = "baseline_reused"
                                counts["measured"] += 1
                            else:
                                _set_load(net, base_p, base_q, scenario, t)
                                net.sgen.at[sgen, "bus"] = bus
                                net.sgen.at[sgen, "p_mw"] = p
                                net.sgen.at[sgen, "q_mvar"] = q
                                ok = counter.run(
                                    net,
                                    f"surface/{point_id}/{scenario}/t={t}/"
                                    f"pi={g['p_index']}/qi={g['q_index']}",
                                )
                                row["source"] = "measured"
                                counts["measured"] += 1
                                if ok:
                                    loss = float(net.res_line.pl_mw.sum())
                                    reduction = float(baseline - loss)
                                    row["feasible"] = True
                                    row["feasible_reason"] = "ok"
                                    row["loss_ess_mw"] = loss
                                    row["dL_reduction_mw"] = reduction
                                    row["L_cost_mw"] = -reduction
                                else:
                                    row["feasible_reason"] = "pf_diverged"
                                    counts["pf_diverged"] += 1
                        group_rows.append(row)

                    coef = _fit_group(point, scenario, t, group_rows)
                    for row in group_rows:
                        row["design_rank"] = coef["matrix_rank"]
                        row["n_fit_samples"] = coef["n_feasible"]
                        raw_writer.writerow(row)
                    raw_f.flush()
                    coef_writer.writerow(coef)
                    coef_f.flush()
                    coef_rows.append(coef)
                    coef_map[(point_id, scenario, t)] = coef
                    print(
                        f"fit point={point_id} scenario={scenario} t={t} "
                        f"rank={coef['matrix_rank']} n={coef['n_feasible']} "
                        f"lambda_min_cost={coef['lambda_min_cost']} "
                        f"runpp_calls={counter.runpp_calls}",
                        flush=True,
                    )
    return coef_rows, coef_map, counts


def _finite(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    return np.asarray([
        float(r[column]) for r in rows if np.isfinite(float(r[column]))
    ], dtype=float)


def _psd_summary(coef_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for point in POINTS:
        point_id = str(point["point_id"])
        for scenario in PM.AVG_DAYS:
            group = [
                r for r in coef_rows
                if r["point_id"] == point_id and r["scenario"] == scenario
                and np.isfinite(float(r["lambda_min_cost"]))
            ]
            bad = [r for r in group if float(r["lambda_min_cost"]) < 0.0]
            ratios = _finite(bad, "lambda_min_over_lambda_max_pct")
            output.append({
                "point_id": point_id, "scenario": scenario,
                "n_cases": len(group), "n_psd_violations": len(bad),
                "psd_violation_pct": 100.0 * len(bad) / len(group) if group else np.nan,
                "minimum_lambda_min_cost": min(
                    (float(r["lambda_min_cost"]) for r in group), default=np.nan
                ),
                "minimum_lambda_min_over_lambda_max_pct": (
                    float(np.min(ratios)) if len(ratios) else np.nan
                ),
                "psd_text": "전 케이스 PSD" if group and not bad else "",
            })
    return output


def _rank_summary(coef_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for point in POINTS:
        point_id = str(point["point_id"])
        group = [r for r in coef_rows if r["point_id"] == point_id]
        bad = [r for r in group if int(r["matrix_rank"]) < 5]
        output.append({
            "point_id": point_id, "n_cases": len(group),
            "n_rank_lt_5": len(bad),
            "rank_lt_5_pct": 100.0 * len(bad) / len(group) if group else np.nan,
            "minimum_rank": min((int(r["matrix_rank"]) for r in group), default=0),
            "minimum_unique_p": min((int(r["n_unique_p"]) for r in group), default=0),
            "minimum_unique_q": min((int(r["n_unique_q"]) for r in group), default=0),
        })
    return output


def _fit_summary(coef_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for point in POINTS:
        point_id = str(point["point_id"])
        group = [r for r in coef_rows if r["point_id"] == point_id]
        rel = _finite(group, "relative_max_residual")
        rmse = _finite(group, "rmse_mw")
        output.append({
            "point_id": point_id, "n_cases": len(rel),
            "relative_max_min": float(np.min(rel)) if len(rel) else np.nan,
            "relative_max_median": float(np.median(rel)) if len(rel) else np.nan,
            "relative_max_p95": float(np.percentile(rel, 95)) if len(rel) else np.nan,
            "relative_max_max": float(np.max(rel)) if len(rel) else np.nan,
            "rmse_mw_min": float(np.min(rmse)) if len(rmse) else np.nan,
            "rmse_mw_median": float(np.median(rmse)) if len(rmse) else np.nan,
            "rmse_mw_p95": float(np.percentile(rmse, 95)) if len(rmse) else np.nan,
            "rmse_mw_max": float(np.max(rmse)) if len(rmse) else np.nan,
        })
    return output


def _dpp_probe() -> list[dict[str, Any]]:
    # (a) PSD 속성을 가진 행렬 Parameter를 quad_form에 직접 전달한다.
    x_a = cp.Variable(2)
    H_param = cp.Parameter((2, 2), PSD=True)
    H_param.value = np.array([[1.0, 0.2], [0.2, 1.0]])
    expr_a = cp.quad_form(x_a, H_param)
    prob_a = cp.Problem(cp.Minimize(expr_a), [x_a <= 1, x_a >= -1])

    # (b) 교차항을 개별 bilinear 항으로 쓴다.
    P = cp.Variable()
    Q = cp.Variable()
    bpp = cp.Parameter(nonneg=True, value=1.0)
    bqq = cp.Parameter(nonneg=True, value=1.0)
    bpq = cp.Parameter(value=0.4)
    expr_b = bpp * cp.square(P) + bqq * cp.square(Q) + bpq * cp.multiply(P, Q)
    prob_b = cp.Problem(cp.Minimize(expr_b), [P <= 1, P >= -1, Q <= 1, Q >= -1])

    # (c) 같은 PSD 행렬을 numpy 상수로 굽는다.
    x_c = cp.Variable(2)
    H_const = np.array([[1.0, 0.2], [0.2, 1.0]])
    expr_c = cp.quad_form(x_c, H_const)
    prob_c = cp.Problem(cp.Minimize(expr_c), [x_c <= 1, x_c >= -1])

    rows = []
    for form, expr, problem in (
        ("a_quad_form_H_parameter_PSD", expr_a, prob_a),
        ("b_separate_parameter_terms", expr_b, prob_b),
        ("c_quad_form_H_constant", expr_c, prob_c),
    ):
        rows.append({
            "form": form, "expression_is_dcp": bool(expr.is_dcp()),
            "expression_is_dpp": bool(expr.is_dpp()),
            "problem_is_dcp": bool(problem.is_dcp()),
            "problem_is_dcp_dpp_true": bool(problem.is_dcp(dpp=True)),
        })
    return rows


def _build_avg_problem(include_loss: bool, force_q_zero: bool) -> AvgProblem:
    T = PM.TIME_STEPS
    topo = lower_lp._get_topology()
    D, r_pu, x_pu = topo["D"], topo["r_pu"], topo["x_pu"]
    r_mat = np.tile(r_pu[:, None], (1, T))
    x_mat = np.tile(x_pu[:, None], (1, T))

    P_ch = cp.Variable(T, nonneg=True)
    P_dis = cp.Variable(T, nonneg=True)
    Q = cp.Variable(T)
    soc = cp.Variable(T + 1)
    S = cp.Parameter(nonneg=True)
    E = cp.Parameter(nonneg=True)
    bus_onehot = cp.Parameter(PM.N_BUS)
    load_p = cp.Parameter((PM.N_BUS, T))
    load_q = cp.Parameter((PM.N_BUS, T))
    smp = cp.Parameter(T, nonneg=True)
    constraints = [P_ch <= S, P_dis <= S]
    constraints += [
        soc[0] == PM.SOC_INIT_FRAC * E, soc[T] == PM.SOC_INIT_FRAC * E,
        soc >= PM.SOC_MIN_FRAC * E, soc <= PM.SOC_MAX_FRAC * E,
    ]
    for t in range(T):
        constraints.append(
            soc[t + 1] == soc[t] * (1 - PM.SELF_DISCHARGE_HOURLY)
            + PM.ETA_C * P_ch[t] * PM.DT_HOURS
            - P_dis[t] / PM.ETA_D * PM.DT_HOURS
        )
    P_net = P_dis - P_ch
    if force_q_zero:
        constraints.append(Q == 0)
    else:
        s_cap = S * float(np.cos(np.pi / POLY_N))
        for k in range(POLY_N):
            theta = 2.0 * np.pi * k / POLY_N
            constraints.append(
                P_net * float(np.cos(theta)) + Q * float(np.sin(theta)) <= s_cap
            )

    injection_p = cp.reshape(bus_onehot, (PM.N_BUS, 1), order="C") @ cp.reshape(
        P_net, (1, T), order="C"
    )
    injection_q = cp.reshape(bus_onehot, (PM.N_BUS, 1), order="C") @ cp.reshape(
        Q, (1, T), order="C"
    )
    P_e = D @ ((load_p - injection_p) / PM.S_BASE_MVA)
    Q_e = D @ ((load_q - injection_q) / PM.S_BASE_MVA)
    v = PM.V_SLACK_SQ - 2.0 * (
        D.T @ (cp.multiply(r_mat, P_e) + cp.multiply(x_mat, Q_e))
    )
    volt_penalty = float(PM.MU_VOLT) * cp.sum(
        cp.pos(v[1:, :] - PM.V_SQ_MAX) + cp.pos(PM.V_SQ_MIN - v[1:, :])
    )
    objective = cp.sum(cp.multiply(smp, P_ch - P_dis)) * PM.DT_HOURS
    objective += 1e-6 * cp.sum(P_ch + P_dis) + volt_penalty

    params: dict[str, Any] = {
        "S": S, "E": E, "bus_onehot": bus_onehot,
        "load_p": load_p, "load_q": load_q, "smp": smp,
    }
    if include_loss:
        linear_p = cp.Parameter(T)
        linear_q = cp.Parameter(T)
        H_cost = [cp.Parameter((2, 2), PSD=True) for _ in range(T)]
        objective += cp.sum(cp.multiply(linear_p, P_net))
        objective += cp.sum(cp.multiply(linear_q, Q))
        for t in range(T):
            objective += cp.quad_form(
                cp.hstack([P_net[t], Q[t]]), H_cost[t]
            )
        params.update({
            "linear_p": linear_p, "linear_q": linear_q, "H_cost": H_cost,
        })
    problem = cp.Problem(cp.Minimize(objective), constraints)
    return AvgProblem(
        problem=problem, params=params,
        vars={"P_net": P_net, "Q": Q, "soc": soc},
        include_loss=include_loss, force_q_zero=force_q_zero,
    )


def _set_avg_values(
    entry: AvgProblem,
    point: dict[str, Any],
    scenario: str,
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    bus = int(point["b"])
    profile = np.asarray(PM.LOAD[scenario], dtype=float)
    base_p, base_q = lower_lp.base_load_bus_arrays()
    onehot = np.zeros(PM.N_BUS)
    onehot[bus] = 1.0
    p = entry.params
    p["S"].value = float(point["S"])
    p["E"].value = float(point["E"])
    p["bus_onehot"].value = onehot
    p["load_p"].value = base_p[:, None] * profile[None, :]
    p["load_q"].value = base_q[:, None] * profile[None, :]
    smp = np.asarray(PM.SMP_PER_MWH[scenario], dtype=float)
    p["smp"].value = smp
    if entry.include_loss:
        a_p = np.zeros(PM.TIME_STEPS)
        a_q = np.zeros(PM.TIME_STEPS)
        for t in range(PM.TIME_STEPS):
            coef = coef_map[(str(point["point_id"]), scenario, t)]
            a_p[t], a_q[t] = float(coef["a_P"]), float(coef["a_Q"])
            H = np.array([
                [float(coef["b_PP"]), float(coef["b_PQ"]) / 2.0],
                [float(coef["b_PQ"]) / 2.0, float(coef["b_QQ"])],
            ])
            # L_cost 자체를 목적함수에 더하므로 부호 반전이나 PSD 사영이 없다.
            p["H_cost"][t].value = smp[t] * PM.DT_HOURS * H
        p["linear_p"].value = smp * PM.DT_HOURS * a_p
        p["linear_q"].value = smp * PM.DT_HOURS * a_q


def _solve_fixed(
    entry: AvgProblem,
) -> tuple[np.ndarray, np.ndarray, str, str, float]:
    attempts = (
        ("CLARABEL", {"solver": cp.CLARABEL, "max_iter": 2000}),
        ("SCS", {"solver": cp.SCS, "max_iters": 100000, "eps": 1e-6}),
    )
    last_status = ""
    for solver_name, kwargs in attempts:
        try:
            entry.problem.solve(**kwargs)
            last_status = str(entry.problem.status)
        except Exception as exc:
            last_status = f"solver_exception:{type(exc).__name__}"
            continue
        if entry.problem.status == cp.OPTIMAL:
            p_val = np.asarray(entry.vars["P_net"].value, dtype=float).copy()
            q_val = np.asarray(entry.vars["Q"].value, dtype=float).copy()
            return p_val, q_val, last_status, solver_name, float(entry.problem.value)
    return (
        np.full(PM.TIME_STEPS, np.nan),
        np.full(PM.TIME_STEPS, np.nan),
        last_status,
        attempts[-1][0],
        float("nan"),
    )


def _ac_slack(
    net, base_p: np.ndarray, base_q: np.ndarray, sgen: int,
    point: dict[str, Any], schedules_p: dict[str, np.ndarray],
    schedules_q: dict[str, np.ndarray], counter: RunCounter, label: str,
) -> dict[str, np.ndarray]:
    output = {}
    for scenario in PM.AVG_DAYS:
        arr = np.full(PM.TIME_STEPS, np.nan)
        for t in range(PM.TIME_STEPS):
            _set_load(net, base_p, base_q, scenario, t)
            P, Q = float(schedules_p[scenario][t]), float(schedules_q[scenario][t])
            pcs_loss = (1.0 - PM.ETA_PCS) * (np.hypot(P, Q) - abs(P))
            net.sgen.at[sgen, "bus"] = int(point["b"])
            net.sgen.at[sgen, "p_mw"] = P - pcs_loss
            net.sgen.at[sgen, "q_mvar"] = Q
            ok = counter.run(
                net, f"schedule/{label}/{point['point_id']}/{scenario}/t={t}"
            )
            if ok:
                arr[t] = float(net.res_ext_grid.p_mw.sum())
        output[scenario] = arr
    return output


def _schedule_probe(
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
    counter: RunCounter,
    force_q_zero: bool,
) -> list[dict[str, Any]]:
    no_loss = _build_avg_problem(False, force_q_zero)
    with_loss = _build_avg_problem(True, force_q_zero)
    net = build_net()
    base_p = net.load["p_mw"].to_numpy().copy()
    base_q = net.load["q_mvar"].to_numpy().copy()
    sgen = _ensure_sgen(net)
    output = []
    for point in POINTS:
        point_id = str(point["point_id"])
        unavailable = []
        non_psd = []
        for scenario in PM.AVG_DAYS:
            for t in range(PM.TIME_STEPS):
                coef = coef_map[(point_id, scenario, t)]
                if not all(np.isfinite(float(coef[name])) for name in COEF_NAMES):
                    unavailable.append((scenario, t))
                elif float(coef["lambda_min_cost"]) < 0.0:
                    non_psd.append((scenario, t))
        if unavailable or non_psd:
            row = {
                "point_id": point_id, "force_q_zero": force_q_zero,
                "max_abs_delta_p_net_mw": np.nan,
                "sum_abs_delta_p_net_mwh": np.nan,
                "abs_objective_difference": np.nan,
                "objective_no_loss": np.nan, "objective_with_loss": np.nan,
                "status_no_loss": "coefficient_unavailable",
                "status_with_loss": "non_psd_or_coefficient_unavailable",
                "solver_no_loss": "", "solver_with_loss": "",
                "hessian_projected_count": 0,
                "coefficient_unavailable_count": len(unavailable),
                "non_psd_hessian_count": len(non_psd),
            }
            if not force_q_zero:
                row["delta_j_net_won_per_year"] = np.nan
            output.append(row)
            continue

        p0: dict[str, np.ndarray] = {}
        q0: dict[str, np.ndarray] = {}
        p1: dict[str, np.ndarray] = {}
        q1: dict[str, np.ndarray] = {}
        statuses0, statuses1, solvers0, solvers1 = set(), set(), set(), set()
        obj0 = obj1 = 0.0
        for scenario in PM.AVG_DAYS:
            _set_avg_values(no_loss, point, scenario, coef_map)
            pc, qc, status, solver, objective = _solve_fixed(no_loss)
            p0[scenario], q0[scenario] = pc, qc
            statuses0.add(status); solvers0.add(solver); obj0 += objective

            _set_avg_values(with_loss, point, scenario, coef_map)
            pl, ql, status, solver, objective = _solve_fixed(with_loss)
            p1[scenario], q1[scenario] = pl, ql
            statuses1.add(status); solvers1.add(solver); obj1 += objective

        delta = np.concatenate([p1[s] - p0[s] for s in PM.AVG_DAYS])
        finite = bool(np.all(np.isfinite(delta)))
        row = {
            "point_id": point_id, "force_q_zero": force_q_zero,
            "max_abs_delta_p_net_mw": (
                float(np.max(np.abs(delta))) if finite else np.nan
            ),
            "sum_abs_delta_p_net_mwh": (
                float(np.sum(np.abs(delta)) * PM.DT_HOURS) if finite else np.nan
            ),
            "abs_objective_difference": abs(obj1 - obj0),
            "objective_no_loss": obj0, "objective_with_loss": obj1,
            "status_no_loss": ",".join(sorted(statuses0)),
            "status_with_loss": ",".join(sorted(statuses1)),
            "solver_no_loss": ",".join(sorted(solvers0)),
            "solver_with_loss": ",".join(sorted(solvers1)),
            "hessian_projected_count": 0,
            "coefficient_unavailable_count": 0,
            "non_psd_hessian_count": 0,
        }
        if not force_q_zero and finite:
            slack0 = _ac_slack(
                net, base_p, base_q, sgen, point, p0, q0, counter, "no_loss"
            )
            slack1 = _ac_slack(
                net, base_p, base_q, sgen, point, p1, q1, counter, "with_loss"
            )
            delta_j = 0.0
            ac_finite = True
            for scenario in PM.AVG_DAYS:
                if not (
                    np.all(np.isfinite(slack0[scenario]))
                    and np.all(np.isfinite(slack1[scenario]))
                ):
                    ac_finite = False
                    break
                delta_j += (
                    PM.N_WEEKDAYS[scenario]
                    * float(np.sum(
                        (slack0[scenario] - slack1[scenario])
                        * PM.SMP_PER_MWH[scenario]
                    ))
                    * PM.DT_HOURS
                )
            row["delta_j_net_won_per_year"] = delta_j if ac_finite else np.nan
        elif not force_q_zero:
            row["delta_j_net_won_per_year"] = np.nan
        output.append(row)
    return output


def _md_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> list[str]:
    if columns is None:
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
    table1b: list[dict[str, Any]],
    table2: list[dict[str, Any]] | None = None,
    table3: list[dict[str, Any]] | None = None,
    table4: list[dict[str, Any]] | None = None,
    table5: list[dict[str, Any]] | None = None,
    environment: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# probe_pq_loss_v2", "", "## 표 1", ""]
    lines += _md_table(table1)
    lines += ["", "## 표 1b", ""] + _md_table(table1b)
    for number, table in ((2, table2), (3, table3), (4, table4), (5, table5)):
        if table is not None:
            lines += ["", f"## 표 {number}", ""] + _md_table(table)
    if environment is not None:
        lines += ["", "## 환경", ""] + _md_table(environment, ["item", "value"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _environment(
    started: float,
    measurement_finished: float,
    measurement_runpp_calls: int,
    counter: RunCounter,
    counts: dict[str, int],
    expected: dict[str, float],
    reusable_path: str,
) -> list[dict[str, Any]]:
    elapsed = time.perf_counter() - started
    measurement_elapsed = measurement_finished - started
    rows = [
        {"item": "timestamp", "value": datetime_.datetime.now().isoformat(timespec="seconds")},
        {"item": "hostname", "value": socket.gethostname()},
        {"item": "platform", "value": platform.platform()},
        {"item": "python", "value": platform.python_version()},
        {"item": "numpy", "value": np.__version__},
        {"item": "pandapower", "value": pp.__version__},
        {"item": "cvxpy", "value": cp.__version__},
        {"item": "installed_solvers", "value": ",".join(cp.installed_solvers())},
        {"item": "POLY_N", "value": POLY_N},
        {"item": "N_Q_ADAPTIVE", "value": N_Q_ADAPTIVE},
        {"item": "baseline_definition", "value": BASELINE_DEFINITION},
        {"item": "grid_definition", "value": GRID_DEFINITION},
        {"item": "feasibility_definition", "value": FEASIBILITY_DEFINITION},
        {"item": "reusable_source", "value": reusable_path},
        {"item": "grid_rows", "value": expected["grid_rows"]},
        {"item": "feasible_grid_rows", "value": expected["feasible_grid_rows"]},
        {"item": "rows_reused", "value": counts["reused"]},
        {"item": "rows_measured", "value": counts["measured"]},
        {"item": "rows_structural_infeasible", "value": counts["infeasible"]},
        {"item": "rows_pf_diverged", "value": counts["pf_diverged"]},
        {"item": "runpp_calls_total", "value": counter.runpp_calls},
        {"item": "pf_retry_events", "value": counter.retry_events},
        {"item": "pf_diverged_calls", "value": counter.diverged},
        {"item": "measurement_elapsed_s", "value": f"{measurement_elapsed:.9f}"},
        {"item": "measurement_runpp_calls", "value": measurement_runpp_calls},
        {"item": "runpp_calls_per_measurement_s", "value": (
            f"{measurement_runpp_calls/measurement_elapsed:.9f}"
            if measurement_elapsed > 0 else "nan"
        )},
        {"item": "total_execution_time_s", "value": f"{elapsed:.9f}"},
        {"item": "hessian_projection_count", "value": 0},
    ]
    for solver, package in (
        ("CLARABEL", "clarabel"), ("SCS", "scs"), ("OSQP", "osqp"),
        ("SCIPY", "scipy"), ("HIGHS", "highspy"),
    ):
        if solver in cp.installed_solvers():
            rows.append({"item": f"solver_{solver}_version", "value": _version(package)})
    return rows


def main() -> int:
    started = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path, coef_path, report_path = _paths()
    expected = _expected_counts()
    for key, value in expected.items():
        print(f"{key}={value}", flush=True)

    reusable, reusable_path = _load_reusable(raw_path)
    print(f"reuse_source={reusable_path}", flush=True)
    print(f"reuse_rows_available={len(reusable)}", flush=True)
    counter = RunCounter()
    coef_rows, coef_map, counts = _measure_and_fit(
        raw_path, coef_path, reusable, counter
    )
    measurement_finished = time.perf_counter()
    measurement_runpp_calls = counter.runpp_calls

    table1 = _psd_summary(coef_rows)
    table1b = _rank_summary(coef_rows)
    _write_report(report_path, table1, table1b)
    print("TABLE_1", flush=True)
    for row in table1:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print("TABLE_1B", flush=True)
    for row in table1b:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    psd_bad = [
        r for r in coef_rows
        if np.isfinite(float(r["lambda_min_cost"]))
        and float(r["lambda_min_cost"]) < 0.0
    ]
    rank_bad = [r for r in coef_rows if int(r["matrix_rank"]) < 5]
    for r in psd_bad:
        print(
            f"psd_violation point={r['point_id']} scenario={r['scenario']} t={r['t']} "
            f"lambda_min_cost={r['lambda_min_cost']} "
            f"lambda_max_cost={r['lambda_max_cost']} "
            f"ratio_pct={r['lambda_min_over_lambda_max_pct']}",
            flush=True,
        )
    for r in rank_bad:
        print(
            f"rank_lt_5 point={r['point_id']} scenario={r['scenario']} t={r['t']} "
            f"rank={r['matrix_rank']} n={r['n_feasible']} "
            f"unique_p={r['n_unique_p']} unique_q={r['n_unique_q']}",
            flush=True,
        )

    table2 = _fit_summary(coef_rows)
    _write_report(report_path, table1, table1b, table2)
    print("TABLE_2", flush=True)
    for row in table2:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table3 = _dpp_probe()
    _write_report(report_path, table1, table1b, table2, table3)
    print("TABLE_3", flush=True)
    for row in table3:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table4 = _schedule_probe(coef_map, counter, force_q_zero=False)
    _write_report(report_path, table1, table1b, table2, table3, table4)
    print("TABLE_4", flush=True)
    for row in table4:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table5 = _schedule_probe(coef_map, counter, force_q_zero=True)
    environment = _environment(
        started, measurement_finished, measurement_runpp_calls,
        counter, counts, expected, reusable_path
    )
    _write_report(
        report_path, table1, table1b, table2, table3, table4, table5, environment
    )
    print("TABLE_5", flush=True)
    for row in table5:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"raw_csv={raw_path}", flush=True)
    print(f"coef_csv={coef_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"runpp_calls={counter.runpp_calls}", flush=True)
    print(f"total_execution_time_s={time.perf_counter()-started:.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
