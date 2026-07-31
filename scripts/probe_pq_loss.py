"""(P,Q) 2D 손실 QP 프로토타입: 5계수 적합, PSD/DPP, 통합 AVG 스케줄 계측.

본코드는 수정하지 않는다. 실행은 사용자가 직접 수행한다.

    python scripts/probe_pq_loss.py

예상 실행시간: 실제 AC 조류계산 약 1만 회이므로 데스크탑에서 수 분~십수 분.
원자료와 계수 CSV는 행마다 flush한다. 곡면 측정/적합 완료 직후 보고서 표 1(PSD)을 먼저
저장하며, 이후 표도 완료되는 순서대로 보고서에 추가한다.
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
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lower_lp
import params as PM
from build_net import build_net
from probe_q_value import POINTS


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
P_FACTORS = np.linspace(-1.0, 1.0, 9)
Q_GRID = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
POLY_N = 128
PSD_VIOLATION_PRINT_FRAC = 0.05
BASELINE_DEFINITION = "loss_p0_q0_same_point_scenario_time"
GRID_DEFINITION = "p=S*[-1,-.75,-.5,-.25,0,.25,.5,.75,1];q=[0,.1,.2,.3,.4]"
FEASIBILITY_DEFINITION = "pcs_circle_and_1h_using_(soc_max-soc_min)*E"
FIT_COLUMNS = ("A_P", "A_Q", "B_PP", "B_QQ", "B_PQ")

RAW_FIELDS = [
    "point_id", "b", "S", "E", "scenario", "t", "p_mw", "q_mvar",
    "grid_feasible", "feasible", "feasible_reason", "baseline_loss_mw",
    "loss_ess_mw", "delta_loss_true_mw", "baseline_definition",
    "grid_definition", "feasibility_definition", "source",
]
COEF_FIELDS = [
    "point_id", "b", "S", "E", "scenario", "t", "n_feasible", "matrix_rank",
    "A_P", "A_Q", "B_PP", "B_QQ", "B_PQ",
    "relative_max_residual", "rmse_mw", "lambda_min", "lambda_max",
    "lambda_min_over_lambda_max_pct", "is_psd",
]


@dataclass
class AvgProblem:
    problem: cp.Problem
    params: dict[str, Any]
    vars: dict[str, Any]
    include_loss: bool
    force_q_zero: bool


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


def _paths() -> tuple[str, str, str]:
    stamp = datetime_.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = os.path.join(RESULTS_DIR, f"probe_pq_loss_{stamp}")
    return stem + ".csv", stem + "_coef.csv", stem + "_report.md"


def _key(point_id: str, scenario: str, t: int, p: float, q: float) -> tuple[Any, ...]:
    return point_id, scenario, int(t), round(float(p), 12), round(float(q), 12)


def _load_reusable_rows(current_path: str) -> tuple[dict[tuple[Any, ...], dict[str, str]], str]:
    candidates = sorted(
        p for p in glob.glob(os.path.join(RESULTS_DIR, "probe_pq_loss_*.csv"))
        if not p.endswith("_coef.csv") and os.path.abspath(p) != os.path.abspath(current_path)
    )
    if not candidates:
        return {}, ""
    path = candidates[-1]
    reusable: dict[tuple[Any, ...], dict[str, str]] = {}
    points = {str(p["point_id"]): p for p in POINTS}
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("baseline_definition") != BASELINE_DEFINITION:
                    continue
                if row.get("grid_definition") != GRID_DEFINITION:
                    continue
                if row.get("feasibility_definition") != FEASIBILITY_DEFINITION:
                    continue
                if row.get("feasible_reason") == "pf_diverged":
                    continue
                try:
                    point = points[row["point_id"]]
                    if (
                        int(row["b"]) != int(point["b"])
                        or not np.isclose(float(row["S"]), float(point["S"]), atol=0, rtol=0)
                        or not np.isclose(float(row["E"]), float(point["E"]), atol=0, rtol=0)
                    ):
                        continue
                    k = _key(
                        row["point_id"], row["scenario"], int(row["t"]),
                        float(row["p_mw"]), float(row["q_mvar"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
                reusable[k] = row
    except (OSError, csv.Error):
        return {}, ""
    return reusable, path


def _grid_feasibility(point: dict[str, Any], p: float, q: float) -> tuple[bool, str]:
    S = float(point["S"])
    E = float(point["E"])
    if np.hypot(p, q) > S + 1e-12:
        return False, "pcs_circle"
    usable_mwh = (PM.SOC_MAX_FRAC - PM.SOC_MIN_FRAC) * E
    max_discharge_mw = usable_mwh * PM.ETA_D / PM.DT_HOURS
    max_charge_mw = usable_mwh / (PM.ETA_C * PM.DT_HOURS)
    if p > max_discharge_mw + 1e-12:
        return False, "soc_discharge_1h"
    if -p > max_charge_mw + 1e-12:
        return False, "soc_charge_1h"
    return True, "ok"


def _ensure_sgen(net) -> int:
    if len(net.sgen) == 0:
        pp.create_sgen(net, bus=1, p_mw=0.0, q_mvar=0.0, name="probe_pq_loss")
    return int(net.sgen.index[0])


def _set_load(net, base_p: np.ndarray, base_q: np.ndarray, scenario: str, t: int) -> None:
    scale = float(PM.LOAD[scenario][t])
    net.load["p_mw"] = base_p * scale
    net.load["q_mvar"] = base_q * scale


def _measure_surfaces(
    raw_path: str,
    reusable: dict[tuple[Any, ...], dict[str, str]],
    counter: RunCounter,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    net = build_net()
    base_p = net.load["p_mw"].to_numpy().copy()
    base_q = net.load["q_mvar"].to_numpy().copy()
    sgen = _ensure_sgen(net)
    rows: list[dict[str, Any]] = []
    source_counts = {"reused": 0, "measured": 0, "infeasible": 0, "diverged": 0}

    with open(raw_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_FIELDS)
        writer.writeheader()
        f.flush()

        for point in POINTS:
            point_id = str(point["point_id"])
            bus, S, E = int(point["b"]), float(point["S"]), float(point["E"])
            p_values = S * P_FACTORS
            for scenario in PM.AVG_DAYS:
                for t in range(PM.TIME_STEPS):
                    group_keys = [
                        _key(point_id, scenario, t, p, q)
                        for p in p_values for q in Q_GRID
                    ]
                    baseline_loss = None
                    for k in group_keys:
                        old = reusable.get(k)
                        if old and old.get("baseline_loss_mw") not in ("", None):
                            baseline_loss = float(old["baseline_loss_mw"])
                            break

                    needs_measurement = any(
                        k not in reusable
                        and _grid_feasibility(point, k[3], k[4])[0]
                        for k in group_keys
                    )
                    if needs_measurement and baseline_loss is None:
                        _set_load(net, base_p, base_q, scenario, t)
                        net.sgen.at[sgen, "bus"] = bus
                        net.sgen.at[sgen, "p_mw"] = 0.0
                        net.sgen.at[sgen, "q_mvar"] = 0.0
                        ok = counter.run(
                            net, f"baseline/{point_id}/{scenario}/t={t}"
                        )
                        baseline_loss = (
                            float(net.res_line.pl_mw.sum()) if ok else float("nan")
                        )

                    for p in p_values:
                        for q in Q_GRID:
                            k = _key(point_id, scenario, t, p, q)
                            old = reusable.get(k)
                            if old is not None:
                                row = {name: old.get(name, "") for name in RAW_FIELDS}
                                row["source"] = "reused"
                                source_counts["reused"] += 1
                            else:
                                grid_ok, reason = _grid_feasibility(point, float(p), float(q))
                                row = {
                                    "point_id": point_id, "b": bus, "S": S, "E": E,
                                    "scenario": scenario, "t": t, "p_mw": float(p),
                                    "q_mvar": float(q), "grid_feasible": grid_ok,
                                    "feasible": False, "feasible_reason": reason,
                                    "baseline_loss_mw": baseline_loss,
                                    "loss_ess_mw": "", "delta_loss_true_mw": "",
                                    "baseline_definition": BASELINE_DEFINITION,
                                    "grid_definition": GRID_DEFINITION,
                                    "feasibility_definition": FEASIBILITY_DEFINITION,
                                    "source": "infeasible",
                                }
                                if not grid_ok:
                                    source_counts["infeasible"] += 1
                                elif not np.isfinite(baseline_loss):
                                    row["feasible_reason"] = "baseline_pf_diverged"
                                    row["source"] = "measured"
                                    source_counts["diverged"] += 1
                                elif abs(float(p)) <= 1e-15 and abs(float(q)) <= 1e-15:
                                    row["feasible"] = True
                                    row["feasible_reason"] = "ok"
                                    row["loss_ess_mw"] = baseline_loss
                                    row["delta_loss_true_mw"] = 0.0
                                    row["source"] = "baseline_reused"
                                    source_counts["measured"] += 1
                                else:
                                    _set_load(net, base_p, base_q, scenario, t)
                                    net.sgen.at[sgen, "bus"] = bus
                                    net.sgen.at[sgen, "p_mw"] = float(p)
                                    net.sgen.at[sgen, "q_mvar"] = float(q)
                                    ok = counter.run(
                                        net,
                                        f"surface/{point_id}/{scenario}/t={t}/p={p}/q={q}",
                                    )
                                    row["source"] = "measured"
                                    source_counts["measured"] += 1
                                    if ok:
                                        loss = float(net.res_line.pl_mw.sum())
                                        row["feasible"] = True
                                        row["feasible_reason"] = "ok"
                                        row["loss_ess_mw"] = loss
                                        row["delta_loss_true_mw"] = baseline_loss - loss
                                    else:
                                        row["feasible_reason"] = "pf_diverged"
                                        source_counts["diverged"] += 1
                            writer.writerow(row)
                            f.flush()
                            rows.append(row)
                    print(
                        f"surface point={point_id} scenario={scenario} t={t} "
                        f"rows={len(rows)} runpp_calls={counter.runpp_calls}",
                        flush=True,
                    )
    return rows, source_counts


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _fit_surfaces(
    rows: list[dict[str, Any]], coef_path: str
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    point_by_id = {str(p["point_id"]): p for p in POINTS}
    for row in rows:
        grouped[(str(row["point_id"]), str(row["scenario"]), int(row["t"]))].append(row)

    coef_rows: list[dict[str, Any]] = []
    coef_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    with open(coef_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=COEF_FIELDS)
        writer.writeheader()
        f.flush()
        for point in POINTS:
            point_id = str(point["point_id"])
            for scenario in PM.AVG_DAYS:
                for t in range(PM.TIME_STEPS):
                    samples = [
                        r for r in grouped[(point_id, scenario, t)]
                        if _truthy(r["feasible"]) and r["delta_loss_true_mw"] not in ("", None)
                    ]
                    X = np.asarray([
                        [
                            float(r["p_mw"]), float(r["q_mvar"]),
                            float(r["p_mw"]) ** 2, float(r["q_mvar"]) ** 2,
                            float(r["p_mw"]) * float(r["q_mvar"]),
                        ]
                        for r in samples
                    ], dtype=float)
                    y = np.asarray(
                        [float(r["delta_loss_true_mw"]) for r in samples], dtype=float
                    )
                    rank = int(np.linalg.matrix_rank(X)) if len(X) else 0
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
                    out = {
                        "point_id": point_id, "b": int(point["b"]),
                        "S": float(point["S"]), "E": float(point["E"]),
                        "scenario": scenario, "t": t, "n_feasible": len(samples),
                        "matrix_rank": rank,
                        **{name: float(beta[i]) for i, name in enumerate(FIT_COLUMNS)},
                        "relative_max_residual": rel, "rmse_mw": rmse,
                        "lambda_min": lmin, "lambda_max": lmax,
                        "lambda_min_over_lambda_max_pct": ratio, "is_psd": is_psd,
                    }
                    writer.writerow(out)
                    f.flush()
                    coef_rows.append(out)
                    coef_map[(point_id, scenario, t)] = out
    return coef_rows, coef_map


def _finite(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    return np.asarray(
        [float(r[column]) for r in rows if np.isfinite(float(r[column]))], dtype=float
    )


def _psd_summary(coef_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for point in POINTS:
        point_id = str(point["point_id"])
        for scenario in PM.AVG_DAYS:
            group = [
                r for r in coef_rows
                if r["point_id"] == point_id and r["scenario"] == scenario
                and np.isfinite(float(r["lambda_min"]))
            ]
            bad = [r for r in group if float(r["lambda_min"]) < 0.0]
            ratios = _finite(bad, "lambda_min_over_lambda_max_pct")
            result.append({
                "point_id": point_id, "scenario": scenario,
                "n_cases": len(group), "n_psd_violations": len(bad),
                "psd_violation_pct": 100.0 * len(bad) / len(group) if group else np.nan,
                "minimum_lambda_min": min(
                    (float(r["lambda_min"]) for r in group), default=np.nan
                ),
                "minimum_lambda_min_over_lambda_max_pct": (
                    float(np.min(ratios)) if len(ratios) else np.nan
                ),
                "psd_text": "전 케이스 PSD" if group and not bad else "",
            })
    return result


def _fit_summary(coef_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for point in POINTS:
        point_id = str(point["point_id"])
        group = [r for r in coef_rows if r["point_id"] == point_id]
        rel = _finite(group, "relative_max_residual")
        rmse = _finite(group, "rmse_mw")
        result.append({
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
    return result


def _dpp_probe() -> list[dict[str, Any]]:
    x = cp.Variable(2)
    H = cp.Parameter((2, 2), symmetric=True)
    H.value = np.array([[1.0, 0.2], [0.2, 1.0]])
    qf = cp.quad_form(x, cp.psd_wrap(H))
    prob_a = cp.Problem(cp.Minimize(qf), [x <= 1, x >= -1])

    P = cp.Variable()
    Q = cp.Variable()
    Bpp = cp.Parameter(nonneg=True, value=1.0)
    Bqq = cp.Parameter(nonneg=True, value=1.0)
    Bpq = cp.Parameter(value=0.4)
    separate = Bpp * P**2 + Bqq * Q**2 + Bpq * cp.multiply(P, Q)
    prob_b = cp.Problem(cp.Minimize(separate), [P <= 1, P >= -1, Q <= 1, Q >= -1])
    rows = []
    for form, problem, expression in (
        ("a_quad_form_psd_wrap", prob_a, qf),
        ("b_separate_terms", prob_b, separate),
    ):
        rows.append({
            "form": form,
            "expression_is_dcp": bool(expression.is_dcp()),
            "expression_is_dpp": bool(expression.is_dpp()),
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

    injections_p = cp.reshape(bus_onehot, (PM.N_BUS, 1), order="C") @ cp.reshape(
        P_net, (1, T), order="C"
    )
    injections_q = cp.reshape(bus_onehot, (PM.N_BUS, 1), order="C") @ cp.reshape(
        Q, (1, T), order="C"
    )
    P_e = D @ ((load_p - injections_p) / PM.S_BASE_MVA)
    Q_e = D @ ((load_q - injections_q) / PM.S_BASE_MVA)
    v = PM.V_SLACK_SQ - 2.0 * (
        D.T @ (cp.multiply(r_mat, P_e) + cp.multiply(x_mat, Q_e))
    )
    volt_penalty = float(PM.MU_VOLT) * cp.sum(
        cp.pos(v[1:, :] - PM.V_SQ_MAX) + cp.pos(PM.V_SQ_MIN - v[1:, :])
    )
    base_objective = cp.sum(cp.multiply(smp, P_ch - P_dis)) * PM.DT_HOURS
    base_objective += 1e-6 * cp.sum(P_ch + P_dis) + volt_penalty

    params: dict[str, Any] = {
        "S": S, "E": E, "bus_onehot": bus_onehot,
        "load_p": load_p, "load_q": load_q, "smp": smp,
    }
    loss_expr: Any = 0.0
    if include_loss:
        linear_p = cp.Parameter(T)
        linear_q = cp.Parameter(T)
        K = [cp.Parameter((2, 2), symmetric=True) for _ in range(T)]
        loss_expr = cp.sum(cp.multiply(linear_p, P_net))
        loss_expr += cp.sum(cp.multiply(linear_q, Q))
        for t in range(T):
            loss_expr += cp.quad_form(
                cp.hstack([P_net[t], Q[t]]), cp.psd_wrap(K[t])
            )
        params.update({"linear_p": linear_p, "linear_q": linear_q, "K": K})
    problem = cp.Problem(cp.Minimize(base_objective + loss_expr), constraints)
    return AvgProblem(
        problem=problem, params=params,
        vars={"P_net": P_net, "Q": Q, "soc": soc},
        include_loss=include_loss, force_q_zero=force_q_zero,
    )


def _project_psd(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    sym = (matrix + matrix.T) / 2.0
    eig, vec = np.linalg.eigh(sym)
    projected = (vec * np.maximum(eig, 0.0)) @ vec.T
    return projected, float(np.linalg.norm(projected - sym, ord="fro"))


def _set_avg_values(
    entry: AvgProblem,
    point: dict[str, Any],
    scenario: str,
    coef_map: dict[tuple[str, str, int], dict[str, Any]],
) -> tuple[int, float]:
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
    projected_count = 0
    projection_fro_sum = 0.0
    if entry.include_loss:
        ap = np.zeros(PM.TIME_STEPS)
        aq = np.zeros(PM.TIME_STEPS)
        for t in range(PM.TIME_STEPS):
            c = coef_map[(str(point["point_id"]), scenario, t)]
            ap[t], aq[t] = float(c["A_P"]), float(c["A_Q"])
            H_delta = np.array([
                [float(c["B_PP"]), float(c["B_PQ"]) / 2.0],
                [float(c["B_PQ"]) / 2.0, float(c["B_QQ"])],
            ])
            K_obj, distance = _project_psd(-smp[t] * PM.DT_HOURS * H_delta)
            projected_count += int(distance > 1e-12)
            projection_fro_sum += distance
            p["K"][t].value = K_obj
        p["linear_p"].value = -smp * PM.DT_HOURS * ap
        p["linear_q"].value = -smp * PM.DT_HOURS * aq
    return projected_count, projection_fro_sum


def _solve_avg(entry: AvgProblem) -> tuple[np.ndarray, np.ndarray, str, str, float]:
    try:
        entry.problem.solve()
    except Exception as exc:
        return (
            np.full(PM.TIME_STEPS, np.nan),
            np.full(PM.TIME_STEPS, np.nan),
            f"solver_exception:{type(exc).__name__}",
            "",
            float("nan"),
        )
    stats = entry.problem.solver_stats
    p_value = entry.vars["P_net"].value
    q_value = entry.vars["Q"].value
    if p_value is None or q_value is None or entry.problem.value is None:
        return (
            np.full(PM.TIME_STEPS, np.nan),
            np.full(PM.TIME_STEPS, np.nan),
            str(entry.problem.status),
            str(getattr(stats, "solver_name", "")) if stats else "",
            float("nan"),
        )
    return (
        np.asarray(p_value, dtype=float).copy(),
        np.asarray(q_value, dtype=float).copy(),
        str(entry.problem.status),
        str(getattr(stats, "solver_name", "")) if stats else "",
        float(entry.problem.value),
    )


def _ac_slack(
    net, base_p: np.ndarray, base_q: np.ndarray, sgen: int,
    point: dict[str, Any], schedules_p: dict[str, np.ndarray],
    schedules_q: dict[str, np.ndarray], counter: RunCounter, label: str,
) -> dict[str, np.ndarray]:
    result = {}
    for scenario in PM.AVG_DAYS:
        arr = np.full(PM.TIME_STEPS, np.nan)
        for t in range(PM.TIME_STEPS):
            _set_load(net, base_p, base_q, scenario, t)
            P = float(schedules_p[scenario][t])
            Q = float(schedules_q[scenario][t])
            pcs_loss = (1.0 - PM.ETA_PCS) * (np.hypot(P, Q) - abs(P))
            net.sgen.at[sgen, "bus"] = int(point["b"])
            net.sgen.at[sgen, "p_mw"] = P - pcs_loss
            net.sgen.at[sgen, "q_mvar"] = Q
            ok = counter.run(
                net, f"schedule/{label}/{point['point_id']}/{scenario}/t={t}"
            )
            if ok:
                arr[t] = float(net.res_ext_grid.p_mw.sum())
        result[scenario] = arr
    return result


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
        bad_coef_count = sum(
            1
            for scenario in PM.AVG_DAYS
            for t in range(PM.TIME_STEPS)
            if not all(
                np.isfinite(float(coef_map[(point_id, scenario, t)][name]))
                for name in FIT_COLUMNS
            )
        )
        if bad_coef_count:
            row = {
                "point_id": point_id, "force_q_zero": force_q_zero,
                "max_abs_delta_p_net_mw": np.nan,
                "sum_abs_delta_p_net_mwh": np.nan,
                "abs_objective_difference": np.nan,
                "objective_no_loss": np.nan, "objective_with_loss": np.nan,
                "status_no_loss": "coefficient_unavailable",
                "status_with_loss": "coefficient_unavailable",
                "solver_no_loss": "", "solver_with_loss": "",
                "objective_hessian_projected_count": np.nan,
                "objective_hessian_projection_fro_sum": np.nan,
                "coefficient_unavailable_count": bad_coef_count,
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
        projection_count = 0
        projection_fro_sum = 0.0
        for scenario in PM.AVG_DAYS:
            _set_avg_values(no_loss, point, scenario, coef_map)
            pc, qc, status, solver, objective = _solve_avg(no_loss)
            p0[scenario], q0[scenario] = pc, qc
            statuses0.add(status); solvers0.add(solver); obj0 += objective

            nproj, dproj = _set_avg_values(with_loss, point, scenario, coef_map)
            projection_count += nproj
            projection_fro_sum += dproj
            pl, ql, status, solver, objective = _solve_avg(with_loss)
            p1[scenario], q1[scenario] = pl, ql
            statuses1.add(status); solvers1.add(solver); obj1 += objective

        delta = np.concatenate([p1[s] - p0[s] for s in PM.AVG_DAYS])
        delta_finite = bool(np.all(np.isfinite(delta)))
        row = {
            "point_id": point["point_id"], "force_q_zero": force_q_zero,
            "max_abs_delta_p_net_mw": (
                float(np.max(np.abs(delta))) if delta_finite else np.nan
            ),
            "sum_abs_delta_p_net_mwh": (
                float(np.sum(np.abs(delta)) * PM.DT_HOURS) if delta_finite else np.nan
            ),
            "abs_objective_difference": abs(obj1 - obj0),
            "objective_no_loss": obj0, "objective_with_loss": obj1,
            "status_no_loss": ",".join(sorted(statuses0)),
            "status_with_loss": ",".join(sorted(statuses1)),
            "solver_no_loss": ",".join(sorted(solvers0)),
            "solver_with_loss": ",".join(sorted(solvers1)),
            "objective_hessian_projected_count": projection_count,
            "objective_hessian_projection_fro_sum": projection_fro_sum,
        }
        if not force_q_zero and delta_finite:
            slack0 = _ac_slack(
                net, base_p, base_q, sgen, point, p0, q0, counter, "no_loss"
            )
            slack1 = _ac_slack(
                net, base_p, base_q, sgen, point, p1, q1, counter, "with_loss"
            )
            delta_j = 0.0
            finite = True
            for scenario in PM.AVG_DAYS:
                if not (np.all(np.isfinite(slack0[scenario]))
                        and np.all(np.isfinite(slack1[scenario]))):
                    finite = False
                    break
                delta_j += (
                    PM.N_WEEKDAYS[scenario]
                    * float(np.sum(
                        (slack0[scenario] - slack1[scenario])
                        * PM.SMP_PER_MWH[scenario]
                    ))
                    * PM.DT_HOURS
                )
            row["delta_j_net_won_per_year"] = delta_j if finite else np.nan
        elif not force_q_zero:
            row["delta_j_net_won_per_year"] = np.nan
        output.append(row)
    return output


def _md_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> list[str]:
    if columns is None:
        columns = list(rows[0]) if rows else []
    out = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, (float, np.floating)):
                value = f"{float(value):.10g}"
            vals.append(str(value))
        out.append("| " + " | ".join(vals) + " |")
    return out


def _write_report(
    path: str,
    psd_rows: list[dict[str, Any]],
    fit_rows: list[dict[str, Any]] | None = None,
    dpp_rows: list[dict[str, Any]] | None = None,
    schedule_rows: list[dict[str, Any]] | None = None,
    qzero_rows: list[dict[str, Any]] | None = None,
    environment: list[dict[str, Any]] | None = None,
) -> None:
    lines = ["# probe_pq_loss", "", "## 표 1", ""]
    lines += _md_table(psd_rows)
    if fit_rows is not None:
        lines += ["", "## 표 2", ""] + _md_table(fit_rows)
    if dpp_rows is not None:
        lines += ["", "## 표 3", ""] + _md_table(dpp_rows)
    if schedule_rows is not None:
        lines += ["", "## 표 4", ""] + _md_table(schedule_rows)
    if qzero_rows is not None:
        lines += ["", "## 표 5", ""] + _md_table(qzero_rows)
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
    started: float, counter: RunCounter, source_counts: dict[str, int],
    reusable_path: str,
) -> list[dict[str, Any]]:
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
        {"item": "baseline_definition", "value": BASELINE_DEFINITION},
        {"item": "grid_definition", "value": GRID_DEFINITION},
        {"item": "feasibility_definition", "value": FEASIBILITY_DEFINITION},
        {"item": "reusable_source", "value": reusable_path},
        {"item": "rows_reused", "value": source_counts["reused"]},
        {"item": "rows_measured", "value": source_counts["measured"]},
        {"item": "rows_structural_infeasible", "value": source_counts["infeasible"]},
        {"item": "rows_pf_diverged", "value": source_counts["diverged"]},
        {"item": "runpp_calls", "value": counter.runpp_calls},
        {"item": "pf_retry_events", "value": counter.retry_events},
        {"item": "pf_diverged_calls", "value": counter.diverged},
        {"item": "total_execution_time_s", "value": f"{time.perf_counter()-started:.9f}"},
    ]
    for solver, package in (
        ("CLARABEL", "clarabel"), ("OSQP", "osqp"), ("SCIPY", "scipy"),
        ("SCS", "scs"), ("HIGHS", "highspy"),
    ):
        if solver in cp.installed_solvers():
            rows.append({"item": f"solver_{solver}_version", "value": _version(package)})
    return rows


def main() -> int:
    started = time.perf_counter()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    raw_path, coef_path, report_path = _paths()
    reusable, reusable_path = _load_reusable_rows(raw_path)
    print(f"reuse_source={reusable_path}", flush=True)
    print(f"reuse_rows_available={len(reusable)}", flush=True)

    counter = RunCounter()
    raw_rows, source_counts = _measure_surfaces(raw_path, reusable, counter)
    print(
        " ".join(f"{k}={v}" for k, v in source_counts.items()),
        flush=True,
    )
    coef_rows, coef_map = _fit_surfaces(raw_rows, coef_path)

    table1 = _psd_summary(coef_rows)
    _write_report(report_path, table1)
    print("TABLE_1", flush=True)
    for row in table1:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    valid = [r for r in coef_rows if np.isfinite(float(r["lambda_min"]))]
    violations = [r for r in valid if float(r["lambda_min"]) < 0.0]
    frac = len(violations) / len(valid) if valid else float("nan")
    print(f"psd_violation_fraction={frac}", flush=True)
    if np.isfinite(frac) and frac > PSD_VIOLATION_PRINT_FRAC:
        for r in violations:
            print(
                f"psd_violation point={r['point_id']} scenario={r['scenario']} "
                f"t={r['t']} lambda_min={r['lambda_min']} "
                f"lambda_max={r['lambda_max']} "
                f"ratio_pct={r['lambda_min_over_lambda_max_pct']}",
                flush=True,
            )

    table2 = _fit_summary(coef_rows)
    _write_report(report_path, table1, table2)
    print("TABLE_2", flush=True)
    for row in table2:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table3 = _dpp_probe()
    _write_report(report_path, table1, table2, table3)
    print("TABLE_3", flush=True)
    for row in table3:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table4 = _schedule_probe(coef_map, counter, force_q_zero=False)
    _write_report(report_path, table1, table2, table3, table4)
    print("TABLE_4", flush=True)
    for row in table4:
        print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)

    table5 = _schedule_probe(coef_map, counter, force_q_zero=True)
    environment = _environment(started, counter, source_counts, reusable_path)
    _write_report(report_path, table1, table2, table3, table4, table5, environment)
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
