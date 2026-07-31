"""P 손실항의 완전 j_net 영향: 스케줄 A(Q만)와 B(P·Q 전체)를 비교한다.

본코드는 수정하지 않는다. 실행은 사용자가 직접 수행한다. 직전 v2 계수 CSV가 필수다.

    python scripts/probe_pq_jnet.py scripts/results/probe_pq_loss_v2_<timestamp>_coef.csv

P1~P3의 AVG 계수는 입력 CSV에서 재사용한다. P1~P3의 PEAK 계수와 P4의 AVG·PEAK 계수는
v2 적응형 격자로 신규 측정한다. 실행 시작 전에 신규 feasible 곡면 수와 기준/스케줄 AC
평가 수를 합산한 예상 runpp 호출 수 및 215 calls/s 기준 예상 시간을 출력한다.
보고서에는 수치와 정의만 기록한다.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_
import importlib.metadata
import inspect
import os
import platform
import socket
import sys
import time
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

import benefits
import lower_lp
import params as PM
from build_net import build_net
from probe_q_value import POINTS as ORIGINAL_POINTS
import probe_pq_loss_v2 as V2


P4 = dict(point_id="P4_normal", b=15, S=0.176, E=0.412)
POINTS = [dict(p) for p in ORIGINAL_POINTS] + [P4]
METHODS = ("A_Q_only", "B_PQ_full")
POLY_N = 128
REFERENCE_RUNPP_PER_SECOND = 215.0
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
COEF_NAMES = V2.COEF_NAMES

CSV_FIELDS = [
    "point_id", "b", "S", "E", "E_over_S", "method", "day_group",
    "scenario", "t", "P_net_mw", "Q_mvar", "p_slack_mw", "line_loss_mw",
    "lp_status", "solver_name", "lp_objective", "lp_pk_proxy_mw",
]
NEW_COEF_FIELDS = [
    "point_id", "b", "S", "E", "scenario", "t", "n_feasible", "matrix_rank",
    "n_unique_p", "n_unique_q", "A_P", "A_Q", "B_PP", "B_QQ", "B_PQ",
    "relative_max_residual", "rmse_mw", "lambda_min_cost", "lambda_max_cost",
    "lambda_min_over_lambda_max_pct", "is_psd_cost",
]


@dataclass
class ProblemEntry:
    problem: cp.Problem
    P_net: Any
    Q: Any
    pk: Any
    method: str
    kind: str


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P 손실항 완전 j_net 영향: v2 L_cost 계수 CSV를 재사용한다."
    )
    parser.add_argument(
        "coef_csv",
        help="probe_pq_loss_v2_<timestamp>_coef.csv 경로",
    )
    return parser.parse_args()


def _paths() -> tuple[str, str, str]:
    stamp = datetime_.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(RESULTS_DIR, f"probe_pq_jnet_{stamp}")
    return stem + ".csv", stem + "_coef.csv", stem + "_report.md"


def _validate_benefits_signatures() -> list[dict[str, str]]:
    expected = {
        "b_energy": ("p_slack_base", "p_slack_ess", "smp_mwh", "n_weekdays"),
        "b_defer": ("p_slack_base", "p_slack_ess"),
        "capex": ("s_mva", "e_mwh"),
        "opex": ("s_mva", "e_mwh"),
        "total_cost": ("s_mva", "e_mwh"),
        "j_net": ("b_energy_val", "b_defer_val", "s_mva", "e_mwh"),
    }
    rows = []
    for name, names in expected.items():
        func = getattr(benefits, name, None)
        if func is None:
            raise RuntimeError(f"benefits.{name} 함수가 없다")
        actual = tuple(inspect.signature(func).parameters)
        if actual != names:
            raise RuntimeError(
                f"benefits.{name} 시그니처 불일치: expected={names}, actual={actual}"
            )
        rows.append({"function": name, "signature": str(inspect.signature(func))})
    return rows


def _load_v2_coefficients(
    path: str,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    int,
    dict[str, dict[str, Any]],
]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"v2 coef CSV 없음: {path}")
    expected_ids = {str(p["point_id"]) for p in ORIGINAL_POINTS}
    expected_sizes = {
        "P1_old_full_opt": (0.176, 0.419),
        "P2_dev_run1": (0.303, 0.404),
        "P3_dev_run0": (1.045, 0.405),
    }
    coef_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    point_specs: dict[str, dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        base_required = {
            "point_id", "b", "S", "E", "scenario", "t", "matrix_rank",
            "lambda_min_cost", "lambda_max_cost",
            "relative_max_residual", "rmse_mw",
        }
        missing_columns = base_required - fieldnames
        coefficient_columns = {
            name: (
                name if name in fieldnames
                else name[0].upper() + name[1:] if name[0].islower() else name
            )
            for name in COEF_NAMES
        }
        missing_columns |= {
            column for column in coefficient_columns.values()
            if column not in fieldnames
        }
        if missing_columns:
            raise RuntimeError(f"v2 coef CSV 필수 열 누락: {sorted(missing_columns)}")
        for raw in reader:
            point_id = raw["point_id"]
            if point_id not in expected_ids:
                continue
            expected_s, expected_e = expected_sizes[point_id]
            if float(raw["S"]) != expected_s or float(raw["E"]) != expected_e:
                raise RuntimeError(
                    f"통제점 불일치: {point_id} CSV=(b={raw['b']},S={raw['S']},E={raw['E']}) "
                    f"expected=(S={expected_s},E={expected_e})"
                )
            point_specs[point_id] = {
                "point_id": point_id, "b": int(raw["b"]),
                "S": float(raw["S"]), "E": float(raw["E"]),
            }
            scenario, t = raw["scenario"], int(raw["t"])
            if scenario not in PM.AVG_DAYS or not (0 <= t < PM.TIME_STEPS):
                continue
            row: dict[str, Any] = {
                "point_id": point_id, "scenario": scenario, "t": t,
                "matrix_rank": int(raw["matrix_rank"]),
                "lambda_min_cost": float(raw["lambda_min_cost"]),
                "lambda_max_cost": float(raw["lambda_max_cost"]),
                "relative_max_residual": float(raw["relative_max_residual"]),
                "rmse_mw": float(raw["rmse_mw"]),
            }
            for name in COEF_NAMES:
                row[name] = float(raw[coefficient_columns[name]])
            coef_map[(point_id, scenario, t)] = row

    missing = [
        (point_id, scenario, t)
        for point_id in expected_ids
        for scenario in PM.AVG_DAYS
        for t in range(PM.TIME_STEPS)
        if (point_id, scenario, t) not in coef_map
    ]
    if missing:
        raise RuntimeError(f"v2 coef CSV 케이스 누락: n={len(missing)}, first={missing[:5]}")
    invalid = [
        key for key, row in coef_map.items()
        if row["matrix_rank"] < 5
        or not all(np.isfinite(float(row[name])) for name in COEF_NAMES)
        or row["lambda_min_cost"] < 0.0
    ]
    if invalid:
        raise RuntimeError(
            f"v2 coef CSV rank/PSD/유한값 불충족: n={len(invalid)}, first={invalid[:5]}"
        )
    if set(point_specs) != expected_ids:
        raise RuntimeError(
            f"v2 coef CSV 통제점 누락: {sorted(expected_ids-set(point_specs))}"
        )
    return coef_map, len(coef_map), point_specs


def _ensure_sgen(net) -> int:
    if len(net.sgen) == 0:
        pp.create_sgen(net, bus=1, p_mw=0.0, q_mvar=0.0, name="probe_pq_jnet")
    return int(net.sgen.index[0])


def _set_load(net, base_p: np.ndarray, base_q: np.ndarray, scenario: str, t: int) -> None:
    scale = float(PM.LOAD[scenario][t])
    net.load["p_mw"] = base_p * scale
    net.load["q_mvar"] = base_q * scale


def _measure_base_ac(
    net, base_p: np.ndarray, base_q: np.ndarray, sgen: int, counter: RunCounter,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    p_slack = {}
    line_loss = {}
    for scenario in PM.ALL_DAYS:
        p_arr = np.full(PM.TIME_STEPS, np.nan)
        l_arr = np.full(PM.TIME_STEPS, np.nan)
        for t in range(PM.TIME_STEPS):
            _set_load(net, base_p, base_q, scenario, t)
            net.sgen.at[sgen, "p_mw"] = 0.0
            net.sgen.at[sgen, "q_mvar"] = 0.0
            ok = counter.run(net, f"base/{scenario}/t={t}")
            if ok:
                p_arr[t] = float(net.res_ext_grid.p_mw.sum())
                l_arr[t] = float(net.res_line.pl_mw.sum())
        p_slack[scenario] = p_arr
        line_loss[scenario] = l_arr
    return p_slack, line_loss


def _measure_new_coefficients(
    coef_path: str,
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    sgen: int,
    base_loss: dict[str, np.ndarray],
    counter: RunCounter,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    list[dict[str, Any]],
    int,
]:
    coef_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    coef_rows = []
    measured_calls_before = counter.runpp_calls
    with open(coef_path, "w", newline="", encoding="utf-8-sig") as coef_f:
        writer = csv.DictWriter(coef_f, fieldnames=NEW_COEF_FIELDS)
        writer.writeheader()
        coef_f.flush()
        for point in POINTS:
            scenarios = PM.ALL_DAYS if point["point_id"] == P4["point_id"] else PM.PEAK_DAYS
            grid = V2._adaptive_grid(point)
            for scenario in scenarios:
                for t in range(PM.TIME_STEPS):
                    rows = []
                    baseline = float(base_loss[scenario][t])
                    for g in grid:
                        p, q = float(g["p_mw"]), float(g["q_mvar"])
                        grid_ok, reason = V2._grid_feasibility(point, p, q)
                        row: dict[str, Any] = {
                            "p_mw": p, "q_mvar": q, "feasible": False,
                            "feasible_reason": reason, "L_cost_mw": "",
                        }
                        if not grid_ok:
                            pass
                        elif not np.isfinite(baseline):
                            row["feasible_reason"] = "baseline_pf_diverged"
                        elif abs(p) <= 1e-15 and abs(q) <= 1e-15:
                            row["feasible"] = True
                            row["feasible_reason"] = "ok"
                            row["L_cost_mw"] = 0.0
                        else:
                            _set_load(net, base_p, base_q, scenario, t)
                            pcs_loss = (
                                (1.0 - PM.ETA_PCS)
                                * (np.hypot(p, q) - abs(p))
                            )
                            net.sgen.at[sgen, "bus"] = int(point["b"])
                            net.sgen.at[sgen, "p_mw"] = p - pcs_loss
                            net.sgen.at[sgen, "q_mvar"] = q
                            ok = counter.run(
                                net,
                                f"new_surface/{point['point_id']}/{scenario}/t={t}/"
                                f"pi={g['p_index']}/qi={g['q_index']}",
                            )
                            if ok:
                                loss = float(net.res_line.pl_mw.sum())
                                row["feasible"] = True
                                row["feasible_reason"] = "ok"
                                row["L_cost_mw"] = loss - baseline
                            else:
                                row["feasible_reason"] = "pf_diverged"
                        rows.append(row)
                    coef = V2._fit_group(point, scenario, t, rows)
                    coef_rows.append(coef)
                    coef_map[(str(point["point_id"]), scenario, t)] = coef
                    writer.writerow({
                        "point_id": coef["point_id"], "b": coef["b"],
                        "S": coef["S"], "E": coef["E"],
                        "scenario": coef["scenario"], "t": coef["t"],
                        "n_feasible": coef["n_feasible"],
                        "matrix_rank": coef["matrix_rank"],
                        "n_unique_p": coef["n_unique_p"],
                        "n_unique_q": coef["n_unique_q"],
                        "A_P": coef["a_P"], "A_Q": coef["a_Q"],
                        "B_PP": coef["b_PP"], "B_QQ": coef["b_QQ"],
                        "B_PQ": coef["b_PQ"],
                        "relative_max_residual": coef["relative_max_residual"],
                        "rmse_mw": coef["rmse_mw"],
                        "lambda_min_cost": coef["lambda_min_cost"],
                        "lambda_max_cost": coef["lambda_max_cost"],
                        "lambda_min_over_lambda_max_pct": (
                            coef["lambda_min_over_lambda_max_pct"]
                        ),
                        "is_psd_cost": coef["is_psd_cost"],
                    })
                    coef_f.flush()
                    print(
                        f"new_fit point={point['point_id']} scenario={scenario} t={t} "
                        f"rank={coef['matrix_rank']} "
                        f"lambda_min_cost={coef['lambda_min_cost']} "
                        f"runpp_calls={counter.runpp_calls}",
                        flush=True,
                    )
    invalid = [
        (r["point_id"], r["scenario"], r["t"])
        for r in coef_rows
        if int(r["matrix_rank"]) < 5
        or not all(np.isfinite(float(r[name])) for name in COEF_NAMES)
        or float(r["lambda_min_cost"]) < 0.0
    ]
    if invalid:
        print(f"new_coefficient_invalid_count={len(invalid)}", flush=True)
        for point_id, scenario, t in invalid:
            print(
                f"new_coefficient_invalid point={point_id} "
                f"scenario={scenario} t={t}",
                flush=True,
            )
    return coef_map, coef_rows, counter.runpp_calls - measured_calls_before


def _new_coefficient_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for point in POINTS:
        scenarios = PM.ALL_DAYS if point["point_id"] == P4["point_id"] else PM.PEAK_DAYS
        for scenario in scenarios:
            group = [
                r for r in rows
                if r["point_id"] == point["point_id"] and r["scenario"] == scenario
            ]
            rel = np.asarray([
                float(r["relative_max_residual"]) for r in group
                if np.isfinite(float(r["relative_max_residual"]))
            ])
            rmse = np.asarray([
                float(r["rmse_mw"]) for r in group
                if np.isfinite(float(r["rmse_mw"]))
            ])
            bad_psd = [r for r in group if float(r["lambda_min_cost"]) < 0.0]
            bad_rank = [r for r in group if int(r["matrix_rank"]) < 5]
            output.append({
                "point_id": point["point_id"], "scenario": scenario,
                "n_cases": len(group), "n_rank_lt_5": len(bad_rank),
                "n_psd_violations": len(bad_psd),
                "minimum_rank": min((int(r["matrix_rank"]) for r in group), default=0),
                "minimum_lambda_min_cost": min(
                    (float(r["lambda_min_cost"]) for r in group), default=np.nan
                ),
                "relative_max_median": float(np.median(rel)) if len(rel) else np.nan,
                "relative_max_max": float(np.max(rel)) if len(rel) else np.nan,
                "rmse_mw_median": float(np.median(rmse)) if len(rmse) else np.nan,
                "rmse_mw_max": float(np.max(rmse)) if len(rmse) else np.nan,
            })
    return output


def _topology_terms(point: dict[str, Any], scenario: str):
    T = PM.TIME_STEPS
    topo = lower_lp._get_topology()
    D, r_pu, x_pu = topo["D"], topo["r_pu"], topo["x_pu"]
    base_p, base_q = lower_lp.base_load_bus_arrays()
    profile = np.asarray(PM.LOAD[scenario], dtype=float)
    load_p = base_p[:, None] * profile[None, :]
    load_q = base_q[:, None] * profile[None, :]
    onehot = np.zeros(PM.N_BUS)
    onehot[int(point["b"])] = 1.0
    return (
        D, np.tile(r_pu[:, None], (1, T)),
        np.tile(x_pu[:, None], (1, T)),
        load_p, load_q, onehot,
    )


def _build_problem(
    kind: str,
    method: str,
    point: dict[str, Any],
    scenario: str,
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
) -> ProblemEntry:
    T = PM.TIME_STEPS
    S, E = float(point["S"]), float(point["E"])
    D, r_mat, x_mat, load_p, load_q, onehot = _topology_terms(point, scenario)
    P_ch = cp.Variable(T, nonneg=True)
    P_dis = cp.Variable(T, nonneg=True)
    Q = cp.Variable(T)
    soc = cp.Variable(T + 1)
    constraints = [P_ch <= S, P_dis <= S]
    constraints += [
        soc[0] == PM.SOC_INIT_FRAC * E,
        soc[T] == PM.SOC_INIT_FRAC * E,
        soc >= PM.SOC_MIN_FRAC * E,
        soc <= PM.SOC_MAX_FRAC * E,
    ]
    for t in range(T):
        constraints.append(
            soc[t + 1] == soc[t] * (1 - PM.SELF_DISCHARGE_HOURLY)
            + PM.ETA_C * P_ch[t] * PM.DT_HOURS
            - P_dis[t] / PM.ETA_D * PM.DT_HOURS
        )
    P_net = P_dis - P_ch
    s_cap = S * float(np.cos(np.pi / POLY_N))
    for k in range(POLY_N):
        theta = 2.0 * np.pi * k / POLY_N
        constraints.append(
            P_net * float(np.cos(theta)) + Q * float(np.sin(theta)) <= s_cap
        )

    injection_p = onehot[:, None] @ cp.reshape(P_net, (1, T), order="C")
    injection_q = onehot[:, None] @ cp.reshape(Q, (1, T), order="C")
    P_e = D @ ((load_p - injection_p) / PM.S_BASE_MVA)
    Q_e = D @ ((load_q - injection_q) / PM.S_BASE_MVA)
    v = PM.V_SLACK_SQ - 2.0 * (
        D.T @ (cp.multiply(r_mat, P_e) + cp.multiply(x_mat, Q_e))
    )
    volt_penalty = float(PM.MU_VOLT) * cp.sum(
        cp.pos(v[1:, :] - PM.V_SQ_MAX) + cp.pos(PM.V_SQ_MIN - v[1:, :])
    )
    coefficients = [
        coef_map[(str(point["point_id"]), scenario, t)] for t in range(T)
    ]
    if any(
        int(c["matrix_rank"]) < 5
        or float(c["lambda_min_cost"]) < 0.0
        or not all(np.isfinite(float(c[name])) for name in COEF_NAMES)
        for c in coefficients
    ):
        raise ValueError(
            f"invalid coefficient: point={point['point_id']} scenario={scenario}"
        )

    pk = None
    if kind == "avg":
        smp = np.asarray(PM.SMP_PER_MWH[scenario], dtype=float)
        objective: Any = cp.sum(cp.multiply(smp, P_ch - P_dis)) * PM.DT_HOURS
        objective += 1e-6 * cp.sum(P_ch + P_dis) + volt_penalty
        for t, c in enumerate(coefficients):
            if method == "A_Q_only":
                objective += (
                    smp[t] * PM.DT_HOURS
                    * (float(c["a_Q"]) * Q[t] + float(c["b_QQ"]) * cp.square(Q[t]))
                )
            elif method == "B_PQ_full":
                H = np.array([
                    [float(c["b_PP"]), float(c["b_PQ"]) / 2.0],
                    [float(c["b_PQ"]) / 2.0, float(c["b_QQ"])],
                ])
                objective += smp[t] * PM.DT_HOURS * (
                    float(c["a_P"]) * P_net[t]
                    + float(c["a_Q"]) * Q[t]
                    + cp.quad_form(cp.hstack([P_net[t], Q[t]]), H)
                )
            else:
                raise ValueError(method)
        problem = cp.Problem(cp.Minimize(objective), constraints)
    elif kind == "peak":
        pk = cp.Variable()
        load_total = float(np.sum(lower_lp.base_load_bus_arrays()[0])) * np.asarray(
            PM.LOAD[scenario], dtype=float
        )
        first_order_reduction = cp.hstack([
            (
                float(c["a_Q"]) * Q[t]
                + (float(c["a_P"]) * P_net[t] if method == "B_PQ_full" else 0.0)
            )
            for t, c in enumerate(coefficients)
        ])
        constraints.append(pk >= load_total - P_net - first_order_reduction)
        problem = cp.Problem(cp.Minimize(pk + volt_penalty), constraints)
    else:
        raise ValueError(kind)
    if not problem.is_dcp():
        raise RuntimeError(
            f"problem not DCP: kind={kind} method={method} "
            f"point={point['point_id']} scenario={scenario}"
        )
    return ProblemEntry(
        problem=problem, P_net=P_net, Q=Q, pk=pk, method=method, kind=kind
    )


def _solve_fixed(entry: ProblemEntry) -> dict[str, Any]:
    attempts = (
        ("CLARABEL", {"solver": cp.CLARABEL, "max_iter": 2000}),
        ("SCS", {"solver": cp.SCS, "max_iters": 100000, "eps": 1e-6}),
    )
    last_status = ""
    last_solver = ""
    for solver_name, kwargs in attempts:
        last_solver = solver_name
        try:
            entry.problem.solve(**kwargs)
            last_status = str(entry.problem.status)
        except Exception as exc:
            last_status = f"solver_exception:{type(exc).__name__}"
            continue
        if entry.problem.status == cp.OPTIMAL:
            return {
                "P": np.asarray(entry.P_net.value, dtype=float).copy(),
                "Q": np.asarray(entry.Q.value, dtype=float).copy(),
                "status": last_status, "solver": solver_name,
                "objective": float(entry.problem.value),
                "pk": float(entry.pk.value) if entry.pk is not None else np.nan,
            }
    return {
        "P": np.full(PM.TIME_STEPS, np.nan),
        "Q": np.full(PM.TIME_STEPS, np.nan),
        "status": last_status, "solver": last_solver,
        "objective": np.nan, "pk": np.nan,
    }


def _make_schedules(
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for point in POINTS:
        point_id = str(point["point_id"])
        schedules[point_id] = {method: {} for method in METHODS}
        for method in METHODS:
            for scenario in PM.ALL_DAYS:
                kind = "avg" if scenario in PM.AVG_DAYS else "peak"
                try:
                    entry = _build_problem(
                        kind, method, point, scenario, coef_map
                    )
                    result = _solve_fixed(entry)
                except Exception as exc:
                    result = {
                        "P": np.full(PM.TIME_STEPS, np.nan),
                        "Q": np.full(PM.TIME_STEPS, np.nan),
                        "status": f"build_exception:{type(exc).__name__}",
                        "solver": "", "objective": np.nan, "pk": np.nan,
                    }
                    print(
                        f"schedule_build_error point={point_id} method={method} "
                        f"scenario={scenario} error={type(exc).__name__}:{exc}",
                        flush=True,
                    )
                schedules[point_id][method][scenario] = result
                print(
                    f"schedule point={point_id} method={method} scenario={scenario} "
                    f"status={result['status']} solver={result['solver']}",
                    flush=True,
                )
    return schedules


def _evaluate_schedules(
    csv_path: str,
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    sgen: int,
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
    counter: RunCounter,
) -> tuple[
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, dict[str, dict[str, np.ndarray]]],
]:
    slack: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    losses: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        for point in POINTS:
            point_id = str(point["point_id"])
            slack[point_id] = {}
            losses[point_id] = {}
            for method in METHODS:
                slack[point_id][method] = {}
                losses[point_id][method] = {}
                for scenario in PM.ALL_DAYS:
                    schedule = schedules[point_id][method][scenario]
                    p_arr = np.full(PM.TIME_STEPS, np.nan)
                    l_arr = np.full(PM.TIME_STEPS, np.nan)
                    for t in range(PM.TIME_STEPS):
                        P = float(schedule["P"][t])
                        Q = float(schedule["Q"][t])
                        if np.isfinite(P) and np.isfinite(Q):
                            _set_load(net, base_p, base_q, scenario, t)
                            pcs_loss = (
                                (1.0 - PM.ETA_PCS)
                                * (np.hypot(P, Q) - abs(P))
                            )
                            net.sgen.at[sgen, "bus"] = int(point["b"])
                            net.sgen.at[sgen, "p_mw"] = P - pcs_loss
                            net.sgen.at[sgen, "q_mvar"] = Q
                            ok = counter.run(
                                net,
                                f"schedule_ac/{point_id}/{method}/{scenario}/t={t}",
                            )
                            if ok:
                                p_arr[t] = float(net.res_ext_grid.p_mw.sum())
                                l_arr[t] = float(net.res_line.pl_mw.sum())
                        writer.writerow({
                            "point_id": point_id, "b": point["b"],
                            "S": point["S"], "E": point["E"],
                            "E_over_S": float(point["E"]) / float(point["S"]),
                            "method": method,
                            "day_group": (
                                "AVG" if scenario in PM.AVG_DAYS else "PEAK"
                            ),
                            "scenario": scenario, "t": t,
                            "P_net_mw": P, "Q_mvar": Q,
                            "p_slack_mw": p_arr[t], "line_loss_mw": l_arr[t],
                            "lp_status": schedule["status"],
                            "solver_name": schedule["solver"],
                            "lp_objective": schedule["objective"],
                            "lp_pk_proxy_mw": schedule["pk"],
                        })
                        f.flush()
                    slack[point_id][method][scenario] = p_arr
                    losses[point_id][method][scenario] = l_arr
    return slack, losses


def _benefit_tables(
    base_slack: dict[str, np.ndarray],
    slack: dict[str, dict[str, dict[str, np.ndarray]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    table = []
    details: dict[str, dict[str, Any]] = {}
    for point in POINTS:
        point_id = str(point["point_id"])
        details[point_id] = {}
        for method in METHODS:
            p_slack = slack[point_id][method]
            b_energy = benefits.b_energy(
                base_slack, p_slack, PM.SMP_PER_MWH, PM.N_WEEKDAYS
            )
            b_defer = benefits.b_defer(base_slack, p_slack)
            capex = benefits.capex(float(point["S"]), float(point["E"]))
            opex = benefits.opex(float(point["S"]), float(point["E"]))
            cost = benefits.total_cost(float(point["S"]), float(point["E"]))
            j_net = benefits.j_net(
                b_energy, b_defer, float(point["S"]), float(point["E"])
            )
            details[point_id][method] = {
                "b_energy": b_energy, "b_defer": b_defer,
                "capex": capex, "opex": opex, "cost": cost, "j_net": j_net,
            }
        A, B = details[point_id]["A_Q_only"], details[point_id]["B_PQ_full"]
        delta_j = B["j_net"] - A["j_net"]
        table.append({
            "point_id": point_id,
            "j_net_A_won_per_year": A["j_net"],
            "j_net_B_won_per_year": B["j_net"],
            "delta_j_net_B_minus_A_won_per_year": delta_j,
            "delta_j_net_sign": (
                "positive" if delta_j > 0 else "negative" if delta_j < 0 else "zero"
            ),
            "delta_b_energy_B_minus_A_won_per_year": (
                B["b_energy"] - A["b_energy"]
            ),
            "delta_b_defer_B_minus_A_won_per_year": (
                B["b_defer"] - A["b_defer"]
            ),
            "b_energy_A": A["b_energy"], "b_energy_B": B["b_energy"],
            "b_defer_A": A["b_defer"], "b_defer_B": B["b_defer"],
            "capex": A["capex"], "opex": A["opex"], "cost": A["cost"],
            "delta_definition": "B-A; positive=B_benefit",
        })
    return table, details


def _schedule_diff_table(
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    output = []
    for point in POINTS:
        point_id = str(point["point_id"])
        for group, scenarios in (("AVG", PM.AVG_DAYS), ("PEAK", PM.PEAK_DAYS)):
            delta_p = np.concatenate([
                schedules[point_id]["B_PQ_full"][s]["P"]
                - schedules[point_id]["A_Q_only"][s]["P"]
                for s in scenarios
            ])
            delta_q = np.concatenate([
                schedules[point_id]["B_PQ_full"][s]["Q"]
                - schedules[point_id]["A_Q_only"][s]["Q"]
                for s in scenarios
            ])
            finite = np.all(np.isfinite(delta_p)) and np.all(np.isfinite(delta_q))
            output.append({
                "point_id": point_id, "day_group": group,
                "max_abs_delta_P_net_mw": (
                    float(np.max(np.abs(delta_p))) if finite else np.nan
                ),
                "max_abs_delta_Q_mvar": (
                    float(np.max(np.abs(delta_q))) if finite else np.nan
                ),
                "sum_abs_delta_P_net_mwh": (
                    float(np.sum(np.abs(delta_p)) * PM.DT_HOURS)
                    if finite else np.nan
                ),
            })
    return output


def _p3_split_table(table1: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_point = {r["point_id"]: r for r in table1}
    return [{
        "point_id": point["point_id"],
        "E_over_S": float(point["E"]) / float(point["S"]),
        "delta_j_net_B_minus_A_won_per_year": by_point[point["point_id"]][
            "delta_j_net_B_minus_A_won_per_year"
        ],
        "delta_b_energy_B_minus_A_won_per_year": by_point[point["point_id"]][
            "delta_b_energy_B_minus_A_won_per_year"
        ],
        "delta_b_defer_B_minus_A_won_per_year": by_point[point["point_id"]][
            "delta_b_defer_B_minus_A_won_per_year"
        ],
    } for point in POINTS]


def _solver_status_table(
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    output = []
    for point in POINTS:
        point_id = str(point["point_id"])
        for method in METHODS:
            for scenario in PM.ALL_DAYS:
                row = schedules[point_id][method][scenario]
                output.append({
                    "point_id": point_id, "method": method,
                    "scenario": scenario,
                    "day_group": "AVG" if scenario in PM.AVG_DAYS else "PEAK",
                    "solver": row["solver"], "status": row["status"],
                    "objective": row["objective"], "lp_pk_proxy_mw": row["pk"],
                    "hessian_projection_count": 0,
                })
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
    table2: list[dict[str, Any]] | None = None,
    table3: list[dict[str, Any]] | None = None,
    table4: list[dict[str, Any]] | None = None,
    table5: list[dict[str, Any]] | None = None,
    environment: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# probe_pq_jnet", "", "## 표 1", ""]
    lines += _md_table(table1)
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
    counter: RunCounter,
    input_coef_path: str,
    new_coef_path: str,
    reused_coef_count: int,
    new_surface_calls: int,
    ac_phase_elapsed_s: float,
    delta_signs_mixed: bool,
    expected: dict[str, float],
    signature_rows: list[dict[str, str]],
) -> list[dict[str, Any]]:
    elapsed = time.perf_counter() - started
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
        {"item": "input_avg_coef_csv", "value": os.path.abspath(input_coef_path)},
        {"item": "new_coef_csv", "value": os.path.abspath(new_coef_path)},
        {"item": "reused_v2_coef_cases", "value": reused_coef_count},
        {"item": "new_surface_runpp_calls", "value": new_surface_calls},
        {"item": "expected_runpp_calls_no_retry", "value": expected["total"]},
        {"item": "runpp_calls_total", "value": counter.runpp_calls},
        {"item": "pf_retry_events", "value": counter.retry_events},
        {"item": "pf_diverged_calls", "value": counter.diverged},
        {"item": "ac_phase_elapsed_s", "value": f"{ac_phase_elapsed_s:.9f}"},
        {"item": "runpp_calls_per_ac_phase_s", "value": (
            f"{counter.runpp_calls/ac_phase_elapsed_s:.9f}"
            if ac_phase_elapsed_s > 0 else "nan"
        )},
        {"item": "total_execution_time_s", "value": f"{elapsed:.9f}"},
        {"item": "delta_j_net_definition", "value": "j_net_B-j_net_A;positive=B_benefit"},
        {"item": "peak_linear_loss_definition", "value": "pk>=load_total-P-a_Q*Q-[B:a_P*P]"},
        {"item": "delta_j_net_signs_mixed", "value": delta_signs_mixed},
        {"item": "hessian_projection_count", "value": 0},
    ]
    for row in signature_rows:
        rows.append({
            "item": f"benefits_signature_{row['function']}",
            "value": row["signature"],
        })
    for solver, package in (
        ("CLARABEL", "clarabel"), ("SCS", "scs"), ("OSQP", "osqp"),
        ("SCIPY", "scipy"), ("HIGHS", "highspy"),
    ):
        if solver in cp.installed_solvers():
            rows.append({"item": f"solver_{solver}_version", "value": _version(package)})
    return rows


def _expected_counts() -> dict[str, float]:
    new_surface = 0
    new_grid_rows = 0
    new_feasible_rows = 0
    for point in POINTS:
        grid = V2._adaptive_grid(point)
        feasible = sum(
            V2._grid_feasibility(point, g["p_mw"], g["q_mvar"])[0]
            for g in grid
        )
        n_scenarios = (
            len(PM.ALL_DAYS)
            if point["point_id"] == P4["point_id"]
            else len(PM.PEAK_DAYS)
        )
        groups = n_scenarios * PM.TIME_STEPS
        new_grid_rows += len(grid) * groups
        new_feasible_rows += feasible * groups
        # 기준 AC가 (0,0)을 제공하므로 각 그룹의 (0,0)은 다시 호출하지 않는다.
        new_surface += (feasible - 1) * groups
    base_ac = len(PM.ALL_DAYS) * PM.TIME_STEPS
    schedule_ac = len(POINTS) * len(METHODS) * len(PM.ALL_DAYS) * PM.TIME_STEPS
    total = new_surface + base_ac + schedule_ac
    return {
        "new_surface_grid_rows": new_grid_rows,
        "new_surface_feasible_rows": new_feasible_rows,
        "new_surface_runpp_no_retry": new_surface,
        "base_ac_runpp_no_retry": base_ac,
        "schedule_ac_runpp_no_retry": schedule_ac,
        "total": total,
        "expected_seconds_at_215_calls_per_s": total / REFERENCE_RUNPP_PER_SECOND,
    }


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path, new_coef_path, report_path = _paths()
    signature_rows = _validate_benefits_signatures()
    coef_map, reused_coef_count, point_specs = _load_v2_coefficients(args.coef_csv)
    for point in POINTS:
        if point["point_id"] in point_specs:
            point.update(point_specs[point["point_id"]])
    expected = _expected_counts()
    for key, value in expected.items():
        print(f"{key}={value}", flush=True)

    counter = RunCounter()
    net = build_net()
    base_p = net.load["p_mw"].to_numpy().copy()
    base_q = net.load["q_mvar"].to_numpy().copy()
    sgen = _ensure_sgen(net)
    ac_phase_started = time.perf_counter()
    base_slack, base_loss = _measure_base_ac(
        net, base_p, base_q, sgen, counter
    )
    new_map, new_coef_rows, new_surface_calls = _measure_new_coefficients(
        new_coef_path, net, base_p, base_q, sgen, base_loss, counter
    )
    ac_phase_elapsed_s = time.perf_counter() - ac_phase_started
    coef_map.update(new_map)
    schedules = _make_schedules(coef_map)
    schedule_ac_started = time.perf_counter()
    slack, _losses = _evaluate_schedules(
        csv_path, net, base_p, base_q, sgen, schedules, counter
    )
    ac_phase_elapsed_s += time.perf_counter() - schedule_ac_started

    table1, _details = _benefit_tables(base_slack, slack)
    _write_report(report_path, table1)
    print("TABLE_1", flush=True)
    for row in table1:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    finite_signs = {
        row["delta_j_net_sign"] for row in table1
        if np.isfinite(float(row["delta_j_net_B_minus_A_won_per_year"]))
    }
    delta_signs_mixed = "positive" in finite_signs and "negative" in finite_signs
    print(f"delta_j_net_signs_mixed={delta_signs_mixed}", flush=True)
    if delta_signs_mixed:
        for row in table1:
            print(
                f"delta_sign point={row['point_id']} "
                f"sign={row['delta_j_net_sign']} "
                f"delta_j_net={row['delta_j_net_B_minus_A_won_per_year']}",
                flush=True,
            )

    table2 = _schedule_diff_table(schedules)
    _write_report(report_path, table1, table2)
    print("TABLE_2", flush=True)
    for row in table2:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table3 = _p3_split_table(table1)
    _write_report(report_path, table1, table2, table3)
    print("TABLE_3", flush=True)
    for row in table3:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table4 = _new_coefficient_summary(new_coef_rows)
    _write_report(report_path, table1, table2, table3, table4)
    print("TABLE_4", flush=True)
    for row in table4:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table5 = _solver_status_table(schedules)
    environment = _environment(
        started, counter, args.coef_csv, new_coef_path, reused_coef_count,
        new_surface_calls, ac_phase_elapsed_s, delta_signs_mixed,
        expected, signature_rows
    )
    _write_report(
        report_path, table1, table2, table3, table4, table5, environment
    )
    print("TABLE_5", flush=True)
    for row in table5:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"coef_csv={new_coef_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"runpp_calls={counter.runpp_calls}", flush=True)
    print(f"total_execution_time_s={time.perf_counter()-started:.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
