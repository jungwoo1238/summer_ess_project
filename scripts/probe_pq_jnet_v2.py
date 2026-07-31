"""피크일 손실항 부호 수정 후 A0/A/B 완전 j_net 재판정 프로토타입.

본코드는 수정하지 않는다. 실행은 사용자가 직접 수행한다. 계수 두 파일이 필수다.

    python scripts/probe_pq_jnet_v2.py \
      scripts/results/probe_pq_loss_v2_20260729_145353_coef.csv \
      scripts/results/probe_pq_jnet_20260729_152807_coef.csv

계수 재측정은 하지 않는다. 실행 시작 시 PEAK 전 케이스의 a_P<0, a_Q<0를 검증하고,
통과한 경우에만 스케줄을 생성한다. 예상 schedule AC runpp는 4*3*5*24=1,440회다.
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import benefits
import evaluate
import lower_lp
import params as PM
from build_net import build_net


METHODS = ("A0_no_peak_loss", "A_peak_Q", "B_peak_PQ")
POLY_N = 128
REFERENCE_RUNPP_PER_SECOND = 230.0
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COEF_NAMES = ("a_P", "a_Q", "b_PP", "b_QQ", "b_PQ")
POINT_IDS = ("P1_old_full_opt", "P2_dev_run1", "P3_dev_run0", "P4_normal")

CSV_FIELDS = [
    "point_id", "b", "S", "E", "E_over_S", "method", "day_group",
    "scenario", "t", "P_net_mw", "Q_mvar", "p_slack_mw", "line_loss_mw",
    "lp_status", "solver_name", "lp_objective", "lp_pk_proxy_mw",
]


@dataclass
class ProblemEntry:
    problem: cp.Problem
    P_net: Any
    Q: Any
    pk: Any


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
        description="피크 손실항 부호 수정: A0/A/B 완전 j_net 비교"
    )
    parser.add_argument("avg_coef_csv", help="probe_pq_loss_v2_*_coef.csv")
    parser.add_argument("peak_coef_csv", help="probe_pq_jnet_20260729_152807_coef.csv")
    return parser.parse_args()


def _paths() -> tuple[str, str]:
    stamp = datetime_.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(RESULTS_DIR, f"probe_pq_jnet_v2_{stamp}")
    return stem + ".csv", stem + "_report.md"


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


def _coefficient_columns(fieldnames: set[str]) -> dict[str, str]:
    result = {}
    for name in COEF_NAMES:
        upper = name[0].upper() + name[1:]
        if name in fieldnames:
            result[name] = name
        elif upper in fieldnames:
            result[name] = upper
        else:
            raise RuntimeError(f"계수 열 누락: {name}/{upper}")
    return result


def _read_coef_file(
    path: str,
    allowed_scenarios: set[str],
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    coef_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    point_specs: dict[str, dict[str, Any]] = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        required = {
            "point_id", "b", "S", "E", "scenario", "t", "matrix_rank",
            "lambda_min_cost", "lambda_max_cost",
        }
        missing = required - fields
        if missing:
            raise RuntimeError(f"계수 CSV 필수 열 누락: {sorted(missing)}")
        columns = _coefficient_columns(fields)
        for raw in reader:
            point_id, scenario = raw["point_id"], raw["scenario"]
            if point_id not in POINT_IDS or scenario not in allowed_scenarios:
                continue
            t = int(raw["t"])
            if not 0 <= t < PM.TIME_STEPS:
                continue
            spec = {
                "point_id": point_id, "b": int(raw["b"]),
                "S": float(raw["S"]), "E": float(raw["E"]),
            }
            previous = point_specs.get(point_id)
            if previous is not None and previous != spec:
                raise RuntimeError(
                    f"통제점 값 불일치: point={point_id}, {previous} vs {spec}"
                )
            point_specs[point_id] = spec
            row: dict[str, Any] = {
                "point_id": point_id, "scenario": scenario, "t": t,
                "matrix_rank": int(raw["matrix_rank"]),
                "lambda_min_cost": float(raw["lambda_min_cost"]),
                "lambda_max_cost": float(raw["lambda_max_cost"]),
            }
            for name, column in columns.items():
                row[name] = float(raw[column])
            coef_map[(point_id, scenario, t)] = row
    return coef_map, point_specs


def _load_coefficients(
    avg_path: str, peak_path: str,
) -> tuple[
    dict[tuple[str, str, int], dict[str, Any]],
    list[dict[str, Any]],
]:
    avg_map, avg_specs = _read_coef_file(avg_path, set(PM.AVG_DAYS))
    # 두 번째 파일은 P1~P3의 PEAK와 P4의 AVG·PEAK를 함께 담는다.
    peak_map, peak_specs = _read_coef_file(peak_path, set(PM.ALL_DAYS))
    specs = dict(peak_specs)
    for point_id, avg_spec in avg_specs.items():
        if point_id in specs and specs[point_id] != avg_spec:
            raise RuntimeError(
                f"AVG/PEAK 통제점 불일치: {point_id}, {avg_spec} vs {specs[point_id]}"
            )
        specs[point_id] = avg_spec
    if set(specs) != set(POINT_IDS):
        raise RuntimeError(f"통제점 누락: {sorted(set(POINT_IDS)-set(specs))}")

    coef_map = dict(peak_map)
    coef_map.update(avg_map)
    missing = [
        (point_id, scenario, t)
        for point_id in POINT_IDS
        for scenario in PM.ALL_DAYS
        for t in range(PM.TIME_STEPS)
        if (point_id, scenario, t) not in coef_map
    ]
    if missing:
        raise RuntimeError(f"계수 케이스 누락: n={len(missing)}, first={missing[:5]}")
    invalid = [
        key for key, row in coef_map.items()
        if row["matrix_rank"] < 5
        or row["lambda_min_cost"] < 0.0
        or not all(np.isfinite(float(row[name])) for name in COEF_NAMES)
    ]
    if invalid:
        raise RuntimeError(f"계수 rank/PSD/유한값 불충족: n={len(invalid)}, first={invalid[:5]}")
    points = [specs[point_id] for point_id in POINT_IDS]
    return coef_map, points


def _verify_peak_signs(
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
    points: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    bad = []
    for point in points:
        point_id = point["point_id"]
        for scenario in PM.PEAK_DAYS:
            group = [
                coef_map[(point_id, scenario, t)] for t in range(PM.TIME_STEPS)
            ]
            bad_ap = [r["t"] for r in group if float(r["a_P"]) >= 0.0]
            bad_aq = [r["t"] for r in group if float(r["a_Q"]) >= 0.0]
            min_reduction_coeff = min(
                min(-float(r["a_P"]), -float(r["a_Q"])) for r in group
            )
            row = {
                "point_id": point_id, "scenario": scenario,
                "n_cases": len(group),
                "n_a_P_nonnegative": len(bad_ap),
                "n_a_Q_nonnegative": len(bad_aq),
                "minimum_minus_a_P_or_minus_a_Q": min_reduction_coeff,
                "sign_check_passed": not bad_ap and not bad_aq,
            }
            rows.append(row)
            if bad_ap or bad_aq:
                bad.append((point_id, scenario, bad_ap, bad_aq))
    print("PEAK_SIGN_CHECK", flush=True)
    for row in rows:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    if bad:
        for point_id, scenario, bad_ap, bad_aq in bad:
            print(
                f"peak_sign_failure point={point_id} scenario={scenario} "
                f"a_P_nonnegative_t={bad_ap} a_Q_nonnegative_t={bad_aq}",
                flush=True,
            )
        raise AssertionError(f"PEAK 계수 부호 불일치: n_groups={len(bad)}")
    assert all(
        float(coef_map[(p["point_id"], s, t)]["a_P"]) < 0.0
        and float(coef_map[(p["point_id"], s, t)]["a_Q"]) < 0.0
        and -float(coef_map[(p["point_id"], s, t)]["a_P"]) > 0.0
        and -float(coef_map[(p["point_id"], s, t)]["a_Q"]) > 0.0
        for p in points for s in PM.PEAK_DAYS for t in range(PM.TIME_STEPS)
    )
    return rows


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
        coef_map[(point["point_id"], scenario, t)] for t in range(T)
    ]
    pk = None
    if kind == "avg":
        smp = np.asarray(PM.SMP_PER_MWH[scenario], dtype=float)
        objective: Any = cp.sum(cp.multiply(smp, P_ch - P_dis)) * PM.DT_HOURS
        objective += 1e-6 * cp.sum(P_ch + P_dis) + volt_penalty
        for t, coef in enumerate(coefficients):
            if method in ("A0_no_peak_loss", "A_peak_Q"):
                objective += smp[t] * PM.DT_HOURS * (
                    float(coef["a_Q"]) * Q[t]
                    + float(coef["b_QQ"]) * cp.square(Q[t])
                )
            elif method == "B_peak_PQ":
                H = np.array([
                    [float(coef["b_PP"]), float(coef["b_PQ"]) / 2.0],
                    [float(coef["b_PQ"]) / 2.0, float(coef["b_QQ"])],
                ])
                objective += smp[t] * PM.DT_HOURS * (
                    float(coef["a_P"]) * P_net[t]
                    + float(coef["a_Q"]) * Q[t]
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
        if method == "A0_no_peak_loss":
            loss_reduction: Any = np.zeros(T)
        elif method == "A_peak_Q":
            loss_reduction = cp.hstack([
                -float(coef["a_Q"]) * Q[t]
                for t, coef in enumerate(coefficients)
            ])
        elif method == "B_peak_PQ":
            loss_reduction = cp.hstack([
                -float(coef["a_Q"]) * Q[t]
                - float(coef["a_P"]) * P_net[t]
                for t, coef in enumerate(coefficients)
            ])
        else:
            raise ValueError(method)
        constraints.append(pk >= load_total - P_net - loss_reduction)
        problem = cp.Problem(cp.Minimize(pk + volt_penalty), constraints)
    else:
        raise ValueError(kind)
    if not problem.is_dcp():
        raise RuntimeError(
            f"problem not DCP: kind={kind} method={method} "
            f"point={point['point_id']} scenario={scenario}"
        )
    return ProblemEntry(problem=problem, P_net=P_net, Q=Q, pk=pk)


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
    points: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for point in points:
        point_id = point["point_id"]
        schedules[point_id] = {method: {} for method in METHODS}
        for scenario in PM.ALL_DAYS:
            kind = "avg" if scenario in PM.AVG_DAYS else "peak"
            # A0와 A의 AVG 정식화는 완전히 같으므로 같은 solve 결과를 복사해 공유한다.
            entry_a = _build_problem(
                kind, "A0_no_peak_loss" if kind == "avg" else "A0_no_peak_loss",
                point, scenario, coef_map,
            )
            a0 = _solve_fixed(entry_a)
            schedules[point_id]["A0_no_peak_loss"][scenario] = a0
            if kind == "avg":
                schedules[point_id]["A_peak_Q"][scenario] = {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in a0.items()
                }
            else:
                entry_a_peak = _build_problem(
                    kind, "A_peak_Q", point, scenario, coef_map
                )
                schedules[point_id]["A_peak_Q"][scenario] = _solve_fixed(entry_a_peak)
            entry_b = _build_problem(
                kind, "B_peak_PQ", point, scenario, coef_map
            )
            schedules[point_id]["B_peak_PQ"][scenario] = _solve_fixed(entry_b)
            for method in METHODS:
                result = schedules[point_id][method][scenario]
                print(
                    f"schedule point={point_id} method={method} scenario={scenario} "
                    f"status={result['status']} solver={result['solver']}",
                    flush=True,
                )
    return schedules


def _ensure_sgen(net) -> int:
    if len(net.sgen) == 0:
        pp.create_sgen(net, bus=1, p_mw=0.0, q_mvar=0.0, name="probe_pq_jnet_v2")
    return int(net.sgen.index[0])


def _set_load(net, base_p: np.ndarray, base_q: np.ndarray, scenario: str, t: int) -> None:
    scale = float(PM.LOAD[scenario][t])
    net.load["p_mw"] = base_p * scale
    net.load["q_mvar"] = base_q * scale


def _evaluate_schedules(
    csv_path: str,
    points: list[dict[str, Any]],
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
    counter: RunCounter,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    net = build_net()
    base_p = net.load["p_mw"].to_numpy().copy()
    base_q = net.load["q_mvar"].to_numpy().copy()
    sgen = _ensure_sgen(net)
    slack: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        f.flush()
        for point in points:
            point_id = point["point_id"]
            slack[point_id] = {}
            for method in METHODS:
                slack[point_id][method] = {}
                for scenario in PM.ALL_DAYS:
                    schedule = schedules[point_id][method][scenario]
                    p_slack = np.full(PM.TIME_STEPS, np.nan)
                    line_loss = np.full(PM.TIME_STEPS, np.nan)
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
                                f"{point_id}/{method}/{scenario}/t={t}",
                            )
                            if ok:
                                p_slack[t] = float(net.res_ext_grid.p_mw.sum())
                                line_loss[t] = float(net.res_line.pl_mw.sum())
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
                            "p_slack_mw": p_slack[t],
                            "line_loss_mw": line_loss[t],
                            "lp_status": schedule["status"],
                            "solver_name": schedule["solver"],
                            "lp_objective": schedule["objective"],
                            "lp_pk_proxy_mw": schedule["pk"],
                        })
                        f.flush()
                    slack[point_id][method][scenario] = p_slack
    return slack


def _base_slack() -> dict[str, np.ndarray]:
    base_flow = evaluate._get_base_flow()
    return {
        scenario: np.asarray(base_flow["p_slack"][scenario], dtype=float).copy()
        for scenario in PM.ALL_DAYS
    }


def _benefit_rows(
    points: list[dict[str, Any]],
    base_slack: dict[str, np.ndarray],
    slack: dict[str, dict[str, dict[str, np.ndarray]]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    details: dict[str, dict[str, Any]] = {}
    rows = []
    for point in points:
        point_id = point["point_id"]
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
        a0 = details[point_id]["A0_no_peak_loss"]
        a = details[point_id]["A_peak_Q"]
        b = details[point_id]["B_peak_PQ"]
        delta_a = a["j_net"] - a0["j_net"]
        delta_b = b["j_net"] - a["j_net"]
        rows.append({
            "point_id": point_id,
            "j_net_A0": a0["j_net"], "j_net_A": a["j_net"], "j_net_B": b["j_net"],
            "delta_A_minus_A0": delta_a,
            "delta_A_minus_A0_sign": (
                "positive" if delta_a > 0 else "negative" if delta_a < 0 else "zero"
            ),
            "delta_b_energy_A_minus_A0": a["b_energy"] - a0["b_energy"],
            "delta_b_defer_A_minus_A0": a["b_defer"] - a0["b_defer"],
            "delta_B_minus_A": delta_b,
            "delta_B_minus_A_sign": (
                "positive" if delta_b > 0 else "negative" if delta_b < 0 else "zero"
            ),
            "delta_b_energy_B_minus_A": b["b_energy"] - a["b_energy"],
            "delta_b_defer_B_minus_A": b["b_defer"] - a["b_defer"],
            "cost": a0["cost"], "capex": a0["capex"], "opex": a0["opex"],
            "delta_definition": "A-A0;B-A",
        })
    return rows, details


def _peak_rows(
    points: list[dict[str, Any]],
    slack: dict[str, dict[str, dict[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows = []
    for point in points:
        point_id = point["point_id"]
        values: dict[str, dict[str, float]] = {}
        for method in METHODS:
            summer = float(np.max(slack[point_id][method]["summer_peak"]))
            winter = float(np.max(slack[point_id][method]["winter_peak"]))
            values[method] = {
                "summer_peak": summer, "winter_peak": winter,
                "annual_peak": max(summer, winter),
            }
        rows.append({
            "point_id": point_id,
            "A0_summer_peak_mw": values["A0_no_peak_loss"]["summer_peak"],
            "A_summer_peak_mw": values["A_peak_Q"]["summer_peak"],
            "B_summer_peak_mw": values["B_peak_PQ"]["summer_peak"],
            "A0_winter_peak_mw": values["A0_no_peak_loss"]["winter_peak"],
            "A_winter_peak_mw": values["A_peak_Q"]["winter_peak"],
            "B_winter_peak_mw": values["B_peak_PQ"]["winter_peak"],
            "A0_annual_peak_mw": values["A0_no_peak_loss"]["annual_peak"],
            "A_annual_peak_mw": values["A_peak_Q"]["annual_peak"],
            "B_annual_peak_mw": values["B_peak_PQ"]["annual_peak"],
            "delta_A_minus_A0_peak_mw": (
                values["A_peak_Q"]["annual_peak"]
                - values["A0_no_peak_loss"]["annual_peak"]
            ),
            "delta_B_minus_A_peak_mw": (
                values["B_peak_PQ"]["annual_peak"]
                - values["A_peak_Q"]["annual_peak"]
            ),
        })
    return rows


def _schedule_diff_rows(
    points: list[dict[str, Any]],
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> list[dict[str, Any]]:
    rows = []
    for point in points:
        point_id = point["point_id"]
        for pair, left, right in (
            ("A_minus_A0", "A0_no_peak_loss", "A_peak_Q"),
            ("B_minus_A", "A_peak_Q", "B_peak_PQ"),
        ):
            for group, scenarios in (("AVG", PM.AVG_DAYS), ("PEAK", PM.PEAK_DAYS)):
                dp = np.concatenate([
                    schedules[point_id][right][s]["P"]
                    - schedules[point_id][left][s]["P"]
                    for s in scenarios
                ])
                dq = np.concatenate([
                    schedules[point_id][right][s]["Q"]
                    - schedules[point_id][left][s]["Q"]
                    for s in scenarios
                ])
                finite = np.all(np.isfinite(dp)) and np.all(np.isfinite(dq))
                rows.append({
                    "point_id": point_id, "pair": pair, "day_group": group,
                    "max_abs_delta_P_mw": (
                        float(np.max(np.abs(dp))) if finite else np.nan
                    ),
                    "max_abs_delta_Q_mvar": (
                        float(np.max(np.abs(dq))) if finite else np.nan
                    ),
                    "sum_abs_delta_P_mwh": (
                        float(np.sum(np.abs(dp)) * PM.DT_HOURS)
                        if finite else np.nan
                    ),
                })
    return rows


def _status_rows(
    points: list[dict[str, Any]],
    schedules: dict[str, dict[str, dict[str, dict[str, Any]]]],
    sign_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signs = {(r["point_id"], r["scenario"]): r for r in sign_rows}
    rows = []
    for point in points:
        point_id = point["point_id"]
        for method in METHODS:
            for scenario in PM.ALL_DAYS:
                result = schedules[point_id][method][scenario]
                sign = signs.get((point_id, scenario))
                rows.append({
                    "point_id": point_id, "method": method,
                    "scenario": scenario,
                    "solver": result["solver"], "status": result["status"],
                    "hessian_projection_count": 0,
                    "n_a_P_nonnegative": (
                        sign["n_a_P_nonnegative"] if sign is not None else ""
                    ),
                    "n_a_Q_nonnegative": (
                        sign["n_a_Q_nonnegative"] if sign is not None else ""
                    ),
                    "peak_sign_check_passed": (
                        sign["sign_check_passed"] if sign is not None else ""
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
    environment: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# probe_pq_jnet_v2", "", "## 표 1", ""] + _md_table(table1)
    for number, table in ((2, table2), (3, table3), (4, table4)):
        if table is not None:
            lines += ["", f"## 표 {number}", ""] + _md_table(table)
    if environment is not None:
        lines += ["", "## 환경", ""]
        lines += _md_table(environment)
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
    ac_elapsed: float,
    counter: RunCounter,
    args: argparse.Namespace,
    signature_rows: list[dict[str, str]],
    mixed_b_minus_a: bool,
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
        {"item": "avg_coef_csv", "value": os.path.abspath(args.avg_coef_csv)},
        {"item": "peak_coef_csv", "value": os.path.abspath(args.peak_coef_csv)},
        {"item": "expected_schedule_runpp_calls", "value": 1440},
        {"item": "schedule_runpp_calls", "value": counter.runpp_calls},
        {"item": "pf_retry_events", "value": counter.retry_events},
        {"item": "pf_diverged_calls", "value": counter.diverged},
        {"item": "ac_elapsed_s", "value": f"{ac_elapsed:.9f}"},
        {"item": "runpp_calls_per_ac_s", "value": (
            f"{counter.runpp_calls/ac_elapsed:.9f}" if ac_elapsed > 0 else "nan"
        )},
        {"item": "total_execution_time_s", "value": f"{elapsed:.9f}"},
        {"item": "peak_constraint_A0", "value": "pk>=load-P"},
        {"item": "peak_constraint_A", "value": "pk>=load-P-(-a_Q*Q)=load-P+a_Q*Q"},
        {"item": "peak_constraint_B", "value": (
            "pk>=load-P-(-a_Q*Q-a_P*P)=load-P+a_Q*Q+a_P*P"
        )},
        {"item": "delta_B_minus_A_signs_mixed", "value": mixed_b_minus_a},
        {"item": "hessian_projection_count", "value": 0},
        {"item": "baseline_source", "value": "evaluate._get_base_flow"},
    ]
    for row in signature_rows:
        rows.append({
            "item": f"benefits_signature_{row['function']}",
            "value": row["signature"],
        })
    for solver, package in (
        ("CLARABEL", "clarabel"), ("SCS", "scs"),
        ("OSQP", "osqp"), ("SCIPY", "scipy"), ("HIGHS", "highspy"),
    ):
        if solver in cp.installed_solvers():
            rows.append({"item": f"solver_{solver}_version", "value": _version(package)})
    return rows


def main() -> int:
    args = _parse_args()
    started = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path, report_path = _paths()
    signature_rows = _validate_benefits_signatures()
    coef_map, points = _load_coefficients(args.avg_coef_csv, args.peak_coef_csv)
    sign_rows = _verify_peak_signs(coef_map, points)

    expected_calls = len(points) * len(METHODS) * len(PM.ALL_DAYS) * PM.TIME_STEPS
    print(f"expected_schedule_runpp_calls={expected_calls}", flush=True)
    print(
        f"expected_seconds_at_{REFERENCE_RUNPP_PER_SECOND:g}_calls_per_s="
        f"{expected_calls/REFERENCE_RUNPP_PER_SECOND:.9f}",
        flush=True,
    )

    schedules = _make_schedules(coef_map, points)
    counter = RunCounter()
    ac_started = time.perf_counter()
    slack = _evaluate_schedules(csv_path, points, schedules, counter)
    ac_elapsed = time.perf_counter() - ac_started
    base_slack = _base_slack()

    table1, _details = _benefit_rows(points, base_slack, slack)
    _write_report(report_path, table1)
    print("TABLE_1", flush=True)
    for row in table1:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    signs = {
        r["delta_B_minus_A_sign"] for r in table1
        if np.isfinite(float(r["delta_B_minus_A"]))
    }
    mixed_b_minus_a = "positive" in signs and "negative" in signs
    print(f"delta_B_minus_A_signs_mixed={mixed_b_minus_a}", flush=True)
    if mixed_b_minus_a:
        for row in table1:
            print(
                f"delta_B_minus_A point={row['point_id']} "
                f"sign={row['delta_B_minus_A_sign']} "
                f"value={row['delta_B_minus_A']}",
                flush=True,
            )

    table2 = _peak_rows(points, slack)
    _write_report(report_path, table1, table2)
    print("TABLE_2", flush=True)
    for row in table2:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table3 = _schedule_diff_rows(points, schedules)
    _write_report(report_path, table1, table2, table3)
    print("TABLE_3", flush=True)
    for row in table3:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table4 = _status_rows(points, schedules, sign_rows)
    environment = _environment(
        started, ac_elapsed, counter, args, signature_rows, mixed_b_minus_a
    )
    _write_report(report_path, table1, table2, table3, table4, environment)
    print("TABLE_4", flush=True)
    for row in table4:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"runpp_calls={counter.runpp_calls}", flush=True)
    print(f"total_execution_time_s={time.perf_counter()-started:.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
