"""QP 손실항의 DPP 게이트와 상수 B_QQ 캐시 우회를 계측한다.

본코드는 수정하지 않는다. lower_lp에서는 토폴로지와 기저부하만 읽어 오며, 문제는 이
파일의 독립 빌더로 구성한다. 실행은 프로젝트 루트에서 다음과 같이 한다.

    python scripts/probe_dpp_gate.py

표 1에서 M2가 DPP이면 지시대로 표 1만 저장한 뒤 즉시 종료한다.
"""

from __future__ import annotations

import csv
import datetime as dt
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


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lower_lp
import params as PM


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
POLY_VALUES = (int(PM.POLY_N), 128)
TEST_BUSES = (15, 32)
N_KEY_VALUES = (1, 5, 32, 160)
REPEATS = 5
S_TEST_MVA = 1.0
E_TEST_MWH = 4.0
P_MATCH_TOL_MW = 1e-6
OBJ_MATCH_TOL_WON = 1.0

RAW_FIELDS = [
    "section",
    "n_keys",
    "key_index",
    "cache_phase",
    "cache_hit",
    "built",
    "build_time_s",
    "method",
    "poly_n",
    "bus",
    "scenario",
    "repeat",
    "solve_time_s",
    "solver_status",
    "solver_name",
    "objective_value",
    "is_dcp",
    "is_dpp",
]


@dataclass
class BuiltProblem:
    problem: cp.Problem
    params: dict[str, cp.Parameter]
    vars: dict[str, Any]
    terms: dict[str, Any]
    method: str
    expression: str
    poly_n: int
    bus_tuple: tuple[int, ...] | None
    scenario_id: str | None


def _paths() -> tuple[str, str]:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(RESULTS_DIR, f"probe_dpp_gate_{stamp}")
    return stem + ".csv", stem + "_report.md"


def _dummy_a(n: int, scenario_id: str) -> np.ndarray:
    """버스·시나리오·시각별 작은 양수 1차 계수. 정확도 검증용이 아니다."""
    s_idx = list(PM.ALL_DAYS).index(scenario_id)
    i = np.arange(n, dtype=float)[:, None]
    t = np.arange(PM.TIME_STEPS, dtype=float)[None, :]
    return 0.010 * (1.0 + 0.03 * i + 0.02 * s_idx + 0.10 * np.sin(2 * np.pi * t / 24))


def _dummy_b(bus_tuple: tuple[int, ...], scenario_id: str) -> np.ndarray:
    """버스·시나리오·시각별로 다른 작은 양수 2차 계수."""
    s_idx = list(PM.ALL_DAYS).index(scenario_id)
    buses = np.asarray(bus_tuple, dtype=float)[:, None]
    t = np.arange(PM.TIME_STEPS, dtype=float)[None, :]
    return 0.010 * (
        1.0 + 0.004 * buses + 0.025 * s_idx + 0.08 * (1.0 + np.cos(2 * np.pi * t / 24))
    )


def _build_problem(
    method: str,
    *,
    expression: str = "square",
    poly_n: int,
    force_q_zero: bool = False,
    bus_tuple: tuple[int, ...] | None = None,
    scenario_id: str | None = None,
) -> BuiltProblem:
    """lower_lp._build_problem(avg)을 복사·개조한 독립 빌더.

    method:
      M0: 손실항 없음
      M1: A_Q Parameter의 1차항
      M2: A_Q Parameter + B_QQ Parameter의 2차항
      M3: A_Q Parameter + 버스/시나리오별 상수 B_QQ의 2차항
    """
    if method not in {"M0", "M1", "M2", "M3"}:
        raise ValueError(method)
    if expression not in {"square", "power"}:
        raise ValueError(expression)
    if method == "M3" and (bus_tuple is None or scenario_id is None):
        raise ValueError("M3에는 bus_tuple과 scenario_id가 필요하다")

    n = len(bus_tuple) if bus_tuple is not None else 1
    T = PM.TIME_STEPS
    topo = lower_lp._get_topology()
    D = topo["D"]
    r_pu = topo["r_pu"]
    x_pu = topo["x_pu"]
    n_bus = topo["n_bus"]
    r_mat = np.tile(r_pu[:, None], (1, T))
    x_mat = np.tile(x_pu[:, None], (1, T))

    P_ch = cp.Variable((n, T), nonneg=True, name=f"{method}_P_ch")
    P_dis = cp.Variable((n, T), nonneg=True, name=f"{method}_P_dis")
    Q = cp.Variable((n, T), name=f"{method}_Q")
    soc = cp.Variable((n, T + 1), name=f"{method}_soc")

    S_param = cp.Parameter(n, nonneg=True, name=f"{method}_S")
    E_param = cp.Parameter(n, nonneg=True, name=f"{method}_E")
    bus_onehot = cp.Parameter((n, n_bus), name=f"{method}_bus_onehot")
    load_p_bus = cp.Parameter((n_bus, T), name=f"{method}_load_p")
    load_q_bus = cp.Parameter((n_bus, T), name=f"{method}_load_q")
    smp_param = cp.Parameter(T, name=f"{method}_smp")
    A_param = None
    B_param = None

    S_col = cp.reshape(S_param, (n, 1), order="C")
    E_col = cp.reshape(E_param, (n, 1), order="C")
    constraints = [P_ch <= S_col, P_dis <= S_col]
    constraints += [
        soc[:, 0] == PM.SOC_INIT_FRAC * E_param,
        soc[:, T] == PM.SOC_INIT_FRAC * E_param,
        soc >= PM.SOC_MIN_FRAC * E_col,
        soc <= PM.SOC_MAX_FRAC * E_col,
    ]
    for t in range(T):
        constraints.append(
            soc[:, t + 1]
            == soc[:, t] * (1 - PM.SELF_DISCHARGE_HOURLY)
            + PM.ETA_C * P_ch[:, t] * PM.DT_HOURS
            - P_dis[:, t] / PM.ETA_D * PM.DT_HOURS
        )

    P_net = P_dis - P_ch
    if force_q_zero:
        constraints.append(Q == 0)
    else:
        s_cap = S_col * float(np.cos(np.pi / poly_n))
        for k in range(poly_n):
            theta = 2.0 * np.pi * k / poly_n
            constraints.append(
                P_net * float(np.cos(theta)) + Q * float(np.sin(theta)) <= s_cap
            )

    netinj_p = (load_p_bus - bus_onehot.T @ P_net) / PM.S_BASE_MVA
    netinj_q = (load_q_bus - bus_onehot.T @ Q) / PM.S_BASE_MVA
    P_e = D @ netinj_p
    Q_e = D @ netinj_q
    v = PM.V_SLACK_SQ - 2.0 * (
        D.T @ (cp.multiply(r_mat, P_e) + cp.multiply(x_mat, Q_e))
    )
    v_nonslack = v[1:, :]
    volt_penalty = float(PM.MU_VOLT) * cp.sum(
        cp.pos(v_nonslack - PM.V_SQ_MAX)
        + cp.pos(PM.V_SQ_MIN - v_nonslack)
    )

    energy_term = (
        cp.sum(
            cp.multiply(
                cp.reshape(smp_param, (1, T), order="C"),
                P_ch - P_dis,
            )
        )
        * PM.DT_HOURS
    )
    linear_term: Any = 0.0
    square_q: Any = cp.square(Q) if expression == "square" else Q**2
    weighted_quadratic: Any = 0.0

    if method in {"M1", "M2", "M3"}:
        A_param = cp.Parameter((n, T), name=f"{method}_A_Q")
        linear_term = -cp.sum(cp.multiply(A_param, Q))
    if method == "M2":
        B_param = cp.Parameter((n, T), nonneg=True, name=f"{method}_B_QQ")
        weighted_quadratic = cp.sum(cp.multiply(B_param, square_q))
    elif method == "M3":
        B_const = _dummy_b(bus_tuple, scenario_id)
        weighted_quadratic = cp.sum(cp.multiply(B_const, square_q))

    loss_term = linear_term + weighted_quadratic
    objective_expr = (
        energy_term
        + 1e-6 * cp.sum(P_ch + P_dis)
        + volt_penalty
        + loss_term
    )
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)

    params = {
        "S": S_param,
        "E": E_param,
        "bus_onehot": bus_onehot,
        "load_p_bus": load_p_bus,
        "load_q_bus": load_q_bus,
        "smp": smp_param,
    }
    if A_param is not None:
        params["A_Q"] = A_param
    if B_param is not None:
        params["B_QQ"] = B_param

    return BuiltProblem(
        problem=problem,
        params=params,
        vars={"P_ch": P_ch, "P_dis": P_dis, "Q": Q, "soc": soc, "P_net": P_net},
        terms={
            "linear_term": linear_term,
            "square_q": square_q,
            "weighted_quadratic": weighted_quadratic,
            "loss_term": loss_term,
            "objective_expr": objective_expr,
        },
        method=method,
        expression=expression,
        poly_n=int(poly_n),
        bus_tuple=bus_tuple,
        scenario_id=scenario_id,
    )


def _term_dpp(term: Any) -> bool | None:
    return bool(term.is_dpp()) if hasattr(term, "is_dpp") else None


def _gate_rows() -> tuple[list[dict[str, Any]], bool]:
    specs = [
        ("M0", "none"),
        ("M1", "none"),
        ("M2", "square"),
        ("M2", "power"),
        ("M3", "square"),
        ("M3", "power"),
    ]
    rows = []
    m2_dpp = []
    for method, expression in specs:
        build_expression = "square" if expression == "none" else expression
        kwargs: dict[str, Any] = {}
        if method == "M3":
            kwargs = {"bus_tuple": (15,), "scenario_id": PM.AVG_DAYS[0]}
        entry = _build_problem(
            method,
            expression=build_expression,
            poly_n=int(PM.POLY_N),
            **kwargs,
        )
        row = {
            "method": method,
            "expression": expression,
            "is_dcp": bool(entry.problem.is_dcp()),
            "is_dcp_dpp_true": bool(entry.problem.is_dcp(dpp=True)),
            "linear_term_is_dpp": _term_dpp(entry.terms["linear_term"]),
            "square_q_is_dpp": _term_dpp(entry.terms["square_q"]),
            "weighted_quadratic_is_dpp": _term_dpp(entry.terms["weighted_quadratic"]),
            "loss_term_is_dpp": _term_dpp(entry.terms["loss_term"]),
            "objective_expr_is_dpp": _term_dpp(entry.terms["objective_expr"]),
        }
        rows.append(row)
        if method == "M2":
            m2_dpp.append(row["is_dcp_dpp_true"])

    print("TABLE_1", flush=True)
    for row in rows:
        print(
            " ".join(f"{key}={value}" for key, value in row.items()),
            flush=True,
        )
    return rows, any(m2_dpp)


def _set_values(entry: BuiltProblem, bus_tuple: tuple[int, ...], scenario: str) -> None:
    n = len(bus_tuple)
    profile = np.asarray(PM.LOAD[scenario], dtype=float)
    base_p, base_q = lower_lp.base_load_bus_arrays()
    onehot = np.zeros((n, PM.N_BUS))
    onehot[np.arange(n), np.asarray(bus_tuple, dtype=int)] = 1.0
    p = entry.params
    p["S"].value = np.full(n, S_TEST_MVA)
    p["E"].value = np.full(n, E_TEST_MWH)
    p["bus_onehot"].value = onehot
    p["load_p_bus"].value = base_p[:, None] * profile[None, :]
    p["load_q_bus"].value = base_q[:, None] * profile[None, :]
    p["smp"].value = np.asarray(PM.SMP[scenario], dtype=float)
    if "A_Q" in p:
        p["A_Q"].value = _dummy_a(n, scenario)
    if "B_QQ" in p:
        p["B_QQ"].value = _dummy_b(bus_tuple, scenario)


def _candidate_keys() -> list[tuple[tuple[int, ...], str]]:
    return [
        ((bus,), scenario)
        for bus in range(PM.B_BOUNDS[0], PM.B_BOUNDS[1] + 1)
        for scenario in PM.ALL_DAYS
    ]


def _cache_key(
    method: str,
    poly_n: int,
    force_q_zero: bool,
    bus_tuple: tuple[int, ...],
    scenario_id: str,
) -> tuple[Any, ...]:
    return method, int(poly_n), bool(force_q_zero), tuple(bus_tuple), str(scenario_id)


def _get_cached_m3(
    cache: dict[tuple[Any, ...], BuiltProblem],
    bus_tuple: tuple[int, ...],
    scenario_id: str,
    poly_n: int,
    *,
    force_q_zero: bool = False,
) -> tuple[BuiltProblem, bool, float]:
    key = _cache_key("M3", poly_n, force_q_zero, bus_tuple, scenario_id)
    if key in cache:
        return cache[key], True, 0.0
    t0 = time.perf_counter()
    entry = _build_problem(
        "M3",
        expression="square",
        poly_n=poly_n,
        force_q_zero=force_q_zero,
        bus_tuple=bus_tuple,
        scenario_id=scenario_id,
    )
    elapsed = time.perf_counter() - t0
    cache[key] = entry
    return entry, False, elapsed


def _cache_probe(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    all_keys = _candidate_keys()
    for n_keys in N_KEY_VALUES:
        cache: dict[tuple[Any, ...], BuiltProblem] = {}
        selected = all_keys[:n_keys]
        build_times = []
        dcp_values = []
        first_total_t0 = time.perf_counter()
        for idx, (bus_tuple, scenario) in enumerate(selected):
            entry, hit, elapsed = _get_cached_m3(
                cache, bus_tuple, scenario, int(PM.POLY_N)
            )
            dcp = bool(entry.problem.is_dcp())
            dpp = bool(entry.problem.is_dcp(dpp=True))
            build_times.append(elapsed)
            dcp_values.append(dcp)
            raw_rows.append(
                {
                    "section": "cache_build",
                    "n_keys": n_keys,
                    "key_index": idx,
                    "cache_phase": "first",
                    "cache_hit": hit,
                    "built": not hit,
                    "build_time_s": elapsed,
                    "method": "M3",
                    "poly_n": PM.POLY_N,
                    "bus": bus_tuple[0],
                    "scenario": scenario,
                    "is_dcp": dcp,
                    "is_dpp": dpp,
                }
            )
        first_total = time.perf_counter() - first_total_t0

        second_builds = 0
        hit_count = 0
        second_total_t0 = time.perf_counter()
        for idx, (bus_tuple, scenario) in enumerate(selected):
            entry, hit, elapsed = _get_cached_m3(
                cache, bus_tuple, scenario, int(PM.POLY_N)
            )
            second_builds += int(not hit)
            hit_count += int(hit)
            raw_rows.append(
                {
                    "section": "cache_build",
                    "n_keys": n_keys,
                    "key_index": idx,
                    "cache_phase": "second",
                    "cache_hit": hit,
                    "built": not hit,
                    "build_time_s": elapsed,
                    "method": "M3",
                    "poly_n": PM.POLY_N,
                    "bus": bus_tuple[0],
                    "scenario": scenario,
                    "is_dcp": bool(entry.problem.is_dcp()),
                    "is_dpp": bool(entry.problem.is_dcp(dpp=True)),
                }
            )
        second_total = time.perf_counter() - second_total_t0
        arr = np.asarray(build_times)
        summary.append(
            {
                "n_keys": n_keys,
                "is_dcp_all": all(dcp_values),
                "first_build_count": len(cache),
                "build_time_each_median_s": float(np.median(arr)),
                "build_time_each_min_s": float(np.min(arr)),
                "build_time_each_max_s": float(np.max(arr)),
                "build_time_sum_s": float(np.sum(arr)),
                "first_call_total_s": first_total,
                "second_call_build_count": second_builds,
                "second_call_cache_hits": hit_count,
                "second_call_total_s": second_total,
            }
        )
    return summary


def _solve_once(entry: BuiltProblem) -> tuple[float, str, str, float]:
    t0 = time.perf_counter()
    entry.problem.solve()
    elapsed = time.perf_counter() - t0
    stats = entry.problem.solver_stats
    solver_name = str(getattr(stats, "solver_name", "")) if stats else ""
    return elapsed, str(entry.problem.status), solver_name, float(entry.problem.value)


def _solve_probe(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for poly_n in POLY_VALUES:
        for bus in TEST_BUSES:
            for scenario in PM.AVG_DAYS:
                entries = {
                    "M0": _build_problem("M0", poly_n=poly_n),
                    "M3": _build_problem(
                        "M3",
                        poly_n=poly_n,
                        bus_tuple=(bus,),
                        scenario_id=scenario,
                    ),
                }
                for method, entry in entries.items():
                    _set_values(entry, (bus,), scenario)
                    for repeat in range(1, REPEATS + 1):
                        elapsed, status, solver_name, objective = _solve_once(entry)
                        row = {
                            "section": "solve",
                            "method": method,
                            "poly_n": poly_n,
                            "bus": bus,
                            "scenario": scenario,
                            "repeat": repeat,
                            "solve_time_s": elapsed,
                            "solver_status": status,
                            "solver_name": solver_name,
                            "objective_value": objective,
                            "is_dcp": bool(entry.problem.is_dcp()),
                            "is_dpp": bool(entry.problem.is_dcp(dpp=True)),
                        }
                        raw_rows.append(row)
                        groups[(poly_n, bus, scenario, method)].append(row)

    summary = []
    for (poly_n, bus, scenario, method), rows in groups.items():
        times = np.asarray([r["solve_time_s"] for r in rows], dtype=float)
        summary.append(
            {
                "poly_n": poly_n,
                "bus": bus,
                "scenario": scenario,
                "method": method,
                "n": len(rows),
                "solve_time_median_s": float(np.median(times)),
                "solve_time_min_s": float(np.min(times)),
                "solve_time_max_s": float(np.max(times)),
                "solver_names": ",".join(sorted({r["solver_name"] for r in rows})),
                "solver_statuses": ",".join(sorted({r["solver_status"] for r in rows})),
            }
        )
    return summary


def _q_zero_probe() -> list[dict[str, Any]]:
    rows = []
    poly_n = int(PM.POLY_N)
    for bus in TEST_BUSES:
        for scenario in PM.AVG_DAYS:
            m0 = _build_problem(
                "M0", poly_n=poly_n, force_q_zero=True
            )
            m3 = _build_problem(
                "M3",
                poly_n=poly_n,
                force_q_zero=True,
                bus_tuple=(bus,),
                scenario_id=scenario,
            )
            _set_values(m0, (bus,), scenario)
            _set_values(m3, (bus,), scenario)
            _solve_once(m0)
            _solve_once(m3)
            p0 = np.asarray(m0.vars["P_net"].value)
            p3 = np.asarray(m3.vars["P_net"].value)
            obj0 = float(m0.problem.value)
            obj3 = float(m3.problem.value)
            max_p_diff = float(np.max(np.abs(p3 - p0)))
            obj_diff = float(abs(obj3 - obj0))
            rows.append(
                {
                    "poly_n": poly_n,
                    "bus": bus,
                    "scenario": scenario,
                    "max_abs_p_net_diff_mw": max_p_diff,
                    "abs_obj_diff_won": obj_diff,
                    "p_net_match": max_p_diff <= P_MATCH_TOL_MW,
                    "objective_match_to_one_won": obj_diff <= OBJ_MATCH_TOL_WON,
                    "m0_status": str(m0.problem.status),
                    "m3_status": str(m3.problem.status),
                }
            )
    return rows


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def _environment(total_time_s: float) -> list[dict[str, Any]]:
    rows = [
        {"item": "timestamp", "value": dt.datetime.now().isoformat(timespec="seconds")},
        {"item": "hostname", "value": socket.gethostname()},
        {"item": "platform", "value": platform.platform()},
        {"item": "python", "value": platform.python_version()},
        {"item": "cvxpy", "value": cp.__version__},
        {"item": "installed_solvers", "value": ",".join(cp.installed_solvers())},
        {"item": "params_poly_n", "value": PM.POLY_N},
        {"item": "total_time_s", "value": f"{total_time_s:.9f}"},
    ]
    for solver, package in (
        ("CLARABEL", "clarabel"),
        ("OSQP", "osqp"),
        ("SCIPY", "scipy"),
        ("SCS", "scs"),
        ("ECOS", "ecos"),
        ("HIGHS", "highspy"),
        ("CVXOPT", "cvxopt"),
        ("GUROBI", "gurobipy"),
    ):
        if solver in cp.installed_solvers():
            rows.append({"item": f"solver_{solver}_version", "value": _package_version(package)})
    return rows


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> list[str]:
    if not rows:
        return ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    out = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.9g}"
            values.append(str(value))
        out.append("| " + " | ".join(values) + " |")
    return out


def _write_raw_csv(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    path: str,
    gate_rows: list[dict[str, Any]],
    cache_rows: list[dict[str, Any]],
    solve_rows: list[dict[str, Any]],
    qzero_rows: list[dict[str, Any]],
    environment: list[dict[str, Any]],
) -> None:
    lines = ["# probe_dpp_gate", "", "## 표 1", ""]
    lines += _md_table(
        gate_rows,
        [
            "method",
            "expression",
            "is_dcp",
            "is_dcp_dpp_true",
            "linear_term_is_dpp",
            "square_q_is_dpp",
            "weighted_quadratic_is_dpp",
            "loss_term_is_dpp",
            "objective_expr_is_dpp",
        ],
    )
    if cache_rows:
        lines += ["", "## 표 2", ""]
        lines += _md_table(cache_rows, list(cache_rows[0]))
    if solve_rows:
        lines += ["", "## 표 3", ""]
        lines += _md_table(solve_rows, list(solve_rows[0]))
    if qzero_rows:
        lines += ["", "## 표 4", ""]
        lines += _md_table(qzero_rows, list(qzero_rows[0]))
    lines += ["", "## 환경", ""]
    lines += _md_table(environment, ["item", "value"])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    started = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path, report_path = _paths()
    raw_rows: list[dict[str, Any]] = []

    gate_rows, m2_is_dpp = _gate_rows()
    if m2_is_dpp:
        total = time.perf_counter() - started
        _write_raw_csv(csv_path, raw_rows)
        _write_report(
            report_path,
            gate_rows,
            [],
            [],
            [],
            _environment(total),
        )
        print(f"csv={csv_path}", flush=True)
        print(f"report={report_path}", flush=True)
        print(f"total_time_s={total:.9f}", flush=True)
        return 0

    cache_rows = _cache_probe(raw_rows)
    solve_rows = _solve_probe(raw_rows)
    qzero_rows = _q_zero_probe()
    total = time.perf_counter() - started
    _write_raw_csv(csv_path, raw_rows)
    _write_report(
        report_path,
        gate_rows,
        cache_rows,
        solve_rows,
        qzero_rows,
        _environment(total),
    )
    print(f"csv={csv_path}", flush=True)
    print(f"report={report_path}", flush=True)
    print(f"total_time_s={total:.9f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
