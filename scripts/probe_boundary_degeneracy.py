"""Compare three remedies for PCS-boundary loss-grid degeneracy.

Purpose
-------
The production ``min7`` loss-measurement grid collapses when the recomputation
centre approaches ``|p_center| = S``.  This probe compares, without changing
production modules:

1. ``full25_fallback``: use min7 normally, but choose the legacy adaptive
   full25 grid when the min7 design matrix has rank below five.
2. ``pull_*``: when ``|p_center| > SHRINK_THRESH*S``, move only the measurement
   centre to ``sign(p_center)*PULL_TARGET*S`` and retain min7.  Four
   ``(threshold, target)`` combinations are measured.
3. ``qzero_gate``: when min7 has rank below five or its Q-related design
   columns vanish, set ``a_Q=b_QQ=b_PQ=0`` and fit only ``a_P,b_PP`` from the
   legacy P-axis points at Q=0.

Ground truth is the dense nominal 81-point grid (nine local P levels by nine
adaptive Q levels) measured with AC power flow and fitted with the same
column-scaled least squares as ``loss_coeffs._fit_samples``.  For boundary
conditions ``p_center >= 0.9*S``, truth is measured at ``p_center=0.9*S`` and
therefore represents the physical surface immediately inside the rating
boundary.  PCS clipping and duplicate removal can reduce the actual number of
truth points below the nominal 81; the actual count is reported.

All prediction residuals are evaluated against the AC measurements on that
condition's truth grid.  Results are diagnostic only: this script makes no
automatic pass/fail decision.

Windows CMD::

    set MKL_THREADING_LAYER=SEQUENTIAL
    python C:\\Users\\PowerSL\\summer_ess_project\\scripts\\probe_boundary_degeneracy.py

Output::

    scripts/results/probe_boundary_degeneracy_out.csv
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import loss_coeffs as LC
from build_net import build_net


BUSES = (16, 15, 6)
S_VALUES = (0.15, 0.30, 0.60)
SCENARIOS = ("summer", "winter")
TIMES = (10, 18)
P_CENTER_FRACTIONS = (0.0, 0.5, 0.9, 0.98, 1.0)

SHRINK_THRESHOLDS = (0.9, 0.95)
PULL_TARGETS = (0.85, 0.9)
BOUNDARY_FRACTION = 0.9
TRUTH_BOUNDARY_CENTER_FRACTION = 0.9

COEFF_NAMES = tuple(LC.COEFF_NAMES)
PSD_TOL = 1e-9
POINT_TOL = 1e-12
Q_COLUMN_TOL = 1e-12
SCALE_EPS_MW = 1e-12
OUTPUT_PATH = (
    Path(__file__).resolve().parent
    / "results"
    / "probe_boundary_degeneracy_out.csv"
)


def _deduplicate_feasible(
    points: Iterable[tuple[float, float]], S: float
) -> list[tuple[float, float]]:
    """Same coordinate clipping and deduplication as the validated min7 probe."""
    unique: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for p_raw, q_raw in points:
        p = float(p_raw)
        q = float(q_raw)
        radius = float(np.hypot(p, q))
        if radius > S:
            scale = S / radius
            p *= scale
            q *= scale
        if np.hypot(p, q) > S + POINT_TOL:
            raise AssertionError(f"PCS-infeasible point: P={p}, Q={q}, S={S}")
        key = (round(p, 14), round(q, 14))
        if key not in seen:
            seen.add(key)
            unique.append((p, q))
    return unique


def _ccd_geometry(S: float, p_center: float) -> tuple[float, float]:
    """Same feasible local CCD geometry as probe_coef_grid_reduction.py."""
    q_at_center = math.sqrt(max(S * S - p_center * p_center, 0.0))
    p_axis_limit = max(S - abs(p_center), 0.0)
    corner_p_limit = math.sqrt(2.0) * max(
        math.sqrt(max(S * S - 0.5 * q_at_center * q_at_center, 0.0))
        - abs(p_center),
        0.0,
    )
    s_eff = min(p_axis_limit, corner_p_limit)
    return float(s_eff), float(q_at_center)


def _grid_min7(S: float, p_center: float) -> list[tuple[float, float]]:
    """Exact min7 coordinate construction used by the validated probe."""
    s_eff, q_center = _ccd_geometry(S, p_center)
    root2 = math.sqrt(2.0)
    points = [
        (p_center, 0.0),
        (p_center - s_eff, 0.0),
        (p_center + s_eff, 0.0),
        (p_center, -q_center),
        (p_center, q_center),
        (p_center + s_eff / root2, q_center / root2),
        (p_center - s_eff / root2, -q_center / root2),
    ]
    return _deduplicate_feasible(points, S)


def _grid_dense81(S: float, p_center: float) -> list[tuple[float, float]]:
    """Same nominal 9-by-9 dense truth grid as the earlier reduction probe."""
    p_offsets = np.linspace(float(LC.P_GRID.min()), float(LC.P_GRID.max()), 9)
    p_values = np.unique(np.clip(p_center + S * p_offsets, -S, S))
    points: list[tuple[float, float]] = []
    for p in p_values:
        q_max = math.sqrt(max(S * S - float(p) ** 2, 0.0))
        points.extend(
            (float(p), float(q)) for q in np.linspace(-q_max, q_max, 9)
        )
    return _deduplicate_feasible(points, S)


def _grid_full25(S: float, p_center: float) -> list[tuple[float, float]]:
    """Legacy production full25 grid, matching loss_coeffs._grid_full25."""
    p_values = np.unique(np.clip(p_center + S * LC.P_GRID, -S, S))
    points: list[tuple[float, float]] = []
    for p_value in p_values:
        p = float(p_value)
        q_max = float(np.sqrt(max(0.0, S * S - p * p)))
        q_values = np.unique(np.linspace(-q_max, q_max, LC.N_Q_GRID))
        for q_value in q_values:
            q = float(q_value)
            if np.hypot(p, q) <= S + LC.GRID_TOL:
                points.append((p, q))
    return points


def _grid_p_axis_1d(S: float, p_center: float) -> list[tuple[float, float]]:
    """Distinct legacy P-axis samples used by the qzero boundary remedy."""
    points = [
        (p, q)
        for p, q in _grid_full25(S, p_center)
        if abs(q) <= POINT_TOL
    ]
    return _deduplicate_feasible(points, S)


def _design_matrix(points: list[tuple[float, float]]) -> np.ndarray:
    return np.asarray(
        [[p, q, p * p, q * q, p * q] for p, q in points], dtype=float
    )


def _design_diagnostics(points: list[tuple[float, float]]) -> dict:
    design = _design_matrix(points)
    if design.size == 0:
        return {"rank": 0, "q_column_norm": 0.0, "degenerate": True}
    scale = np.max(np.abs(design), axis=0)
    scale[scale == 0.0] = 1.0
    rank = int(np.linalg.matrix_rank(design / scale))
    q_column_norm = float(np.linalg.norm(design[:, (1, 3, 4)]))
    degenerate = bool(rank < 5 or q_column_norm <= Q_COLUMN_TOL)
    return {
        "rank": rank,
        "q_column_norm": q_column_norm,
        "degenerate": degenerate,
    }


def _measure_points(
    net,
    bus: int,
    scenario: str,
    t: int,
    points: list[tuple[float, float]],
) -> dict:
    """Measure points through the same cache-bypassing API as the prior probe."""
    samples: list[dict] = []
    baselines: list[float] = []
    attempts = retries = failures = 0
    for p_mw, q_mvar in points:
        measured = LC.measure_loss_reduction(
            bus,
            scenario,
            t,
            p_mw,
            q_mvar,
            net=net,
        )
        samples.append(
            {
                "p_mw": float(p_mw),
                "q_mvar": float(q_mvar),
                "L_cost_mw": float(measured["L_cost_mw"]),
            }
        )
        baselines.append(float(measured["baseline_loss_mw"]))
        attempts += int(measured["runpp_attempts"])
        retries += int(measured["runpp_retries"])
        failures += int(measured["runpp_failures"])

    finite_baselines = np.asarray(baselines, dtype=float)
    finite_baselines = finite_baselines[np.isfinite(finite_baselines)]
    baseline_loss = (
        float(np.median(finite_baselines))
        if finite_baselines.size
        else float("nan")
    )
    return {
        "points": points,
        "samples": samples,
        "baseline_loss_mw": baseline_loss,
        "sample_count": len(points),
        "pf_attempts_actual": attempts,
        "pf_retries": retries,
        "pf_failures": failures,
    }


def _fit_full(samples: list[dict]) -> dict:
    """Use the production column-scaled five-coefficient least squares."""
    return LC._fit_samples(samples)


def _fit_p_only(samples: list[dict]) -> dict:
    """Fit a_P and b_PP only; explicitly set all Q coefficients to zero."""
    finite = [row for row in samples if np.isfinite(row["L_cost_mw"])]
    if finite:
        design = np.asarray(
            [[row["p_mw"], row["p_mw"] ** 2] for row in finite], dtype=float
        )
        y = np.asarray([row["L_cost_mw"] for row in finite], dtype=float)
        scale = np.max(np.abs(design), axis=0)
        scale[scale == 0.0] = 1.0
        beta_scaled, _, rank, _ = np.linalg.lstsq(design / scale, y, rcond=None)
        beta = beta_scaled / scale
        residual = design @ beta - y
        a_p, b_pp = (float(beta[0]), float(beta[1]))
        max_abs = float(np.max(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
    else:
        rank = 0
        a_p = b_pp = max_abs = rmse = float("nan")
    return {
        "a_P": a_p,
        "a_Q": 0.0,
        "b_PP": b_pp,
        "b_QQ": 0.0,
        "b_PQ": 0.0,
        "fit_rank": int(rank),
        "fit_rank_full": False,
        "fit_max_abs_error_mw": max_abs,
        "fit_rmse_mw": rmse,
    }


def _method_names() -> list[str]:
    names = ["full25_fallback"]
    for threshold in SHRINK_THRESHOLDS:
        for target in PULL_TARGETS:
            names.append(f"pull_t{threshold:.2f}_p{target:.2f}")
    names.append("qzero_gate")
    return names


METHODS = _method_names()


def _measure_truth(
    net,
    bus: int,
    S: float,
    scenario: str,
    t: int,
    p_fraction: float,
) -> dict:
    truth_fraction = (
        TRUTH_BOUNDARY_CENTER_FRACTION
        if p_fraction >= BOUNDARY_FRACTION
        else p_fraction
    )
    truth_center = truth_fraction * S
    measured = _measure_points(
        net, bus, scenario, t, _grid_dense81(S, truth_center)
    )
    measured["fit"] = _fit_full(measured["samples"])
    measured["measurement_center_mw"] = truth_center
    return measured


def _measure_fallback(
    net, bus: int, S: float, scenario: str, t: int, p_center: float
) -> dict:
    min7_points = _grid_min7(S, p_center)
    min7_diag = _design_diagnostics(min7_points)
    fallback_used = bool(min7_diag["rank"] < 5)
    points = _grid_full25(S, p_center) if fallback_used else min7_points
    measured = _measure_points(net, bus, scenario, t, points)
    measured.update(
        {
            "fit": _fit_full(measured["samples"]),
            "measurement_center_mw": p_center,
            "fallback_used": fallback_used,
            "source_min7_rank": min7_diag["rank"],
            "source_q_column_norm": min7_diag["q_column_norm"],
            "q_gate_exempt": False,
        }
    )
    return measured


def _measure_pull(
    net,
    bus: int,
    S: float,
    scenario: str,
    t: int,
    p_center: float,
    threshold: float,
    target: float,
) -> dict:
    if abs(p_center) > threshold * S:
        sign = -1.0 if p_center < 0.0 else 1.0
        measurement_center = sign * target * S
        pulled = True
    else:
        measurement_center = p_center
        pulled = False
    points = _grid_min7(S, measurement_center)
    diagnostics = _design_diagnostics(points)
    measured = _measure_points(net, bus, scenario, t, points)
    measured.update(
        {
            "fit": _fit_full(measured["samples"]),
            "measurement_center_mw": measurement_center,
            "fallback_used": pulled,
            "source_min7_rank": diagnostics["rank"],
            "source_q_column_norm": diagnostics["q_column_norm"],
            "q_gate_exempt": False,
        }
    )
    return measured


def _measure_qzero(
    net, bus: int, S: float, scenario: str, t: int, p_center: float
) -> dict:
    min7_points = _grid_min7(S, p_center)
    diagnostics = _design_diagnostics(min7_points)
    if diagnostics["degenerate"]:
        points = _grid_p_axis_1d(S, p_center)
        measured = _measure_points(net, bus, scenario, t, points)
        fit = _fit_p_only(measured["samples"])
        q_gate_exempt = True
    else:
        points = min7_points
        measured = _measure_points(net, bus, scenario, t, points)
        fit = _fit_full(measured["samples"])
        q_gate_exempt = False
    measured.update(
        {
            "fit": fit,
            "measurement_center_mw": p_center,
            "fallback_used": bool(diagnostics["degenerate"]),
            "source_min7_rank": diagnostics["rank"],
            "source_q_column_norm": diagnostics["q_column_norm"],
            "q_gate_exempt": q_gate_exempt,
        }
    )
    return measured


def _truth_matrix(truth: dict) -> tuple[np.ndarray, np.ndarray]:
    design = _design_matrix(
        [(row["p_mw"], row["q_mvar"]) for row in truth["samples"]]
    )
    measured = np.asarray(
        [row["L_cost_mw"] for row in truth["samples"]], dtype=float
    )
    return design, measured


def _metrics(method: dict, truth: dict) -> dict:
    coeff = np.asarray([method["fit"][name] for name in COEFF_NAMES], dtype=float)
    design, truth_measured = _truth_matrix(truth)
    residual = design @ coeff - truth_measured
    finite = np.isfinite(residual)
    if np.any(finite):
        max_abs = float(np.max(np.abs(residual[finite])))
        rmse = float(np.sqrt(np.mean(residual[finite] ** 2)))
    else:
        max_abs = rmse = float("nan")
    baseline_scale = max(abs(float(truth["baseline_loss_mw"])), SCALE_EPS_MW)
    H = np.asarray(
        [[coeff[2], coeff[4] / 2.0], [coeff[4] / 2.0, coeff[3]]],
        dtype=float,
    )
    h_min_eig = (
        float(np.linalg.eigvalsh(H)[0])
        if np.all(np.isfinite(H))
        else float("nan")
    )
    a_p_ok = bool(np.isfinite(coeff[0]) and coeff[0] < 0.0)
    a_q_identified = bool(not method["q_gate_exempt"] and coeff[1] != 0.0)
    a_q_ok = bool(method["q_gate_exempt"] or (np.isfinite(coeff[1]) and coeff[1] < 0.0))
    psd_ok = bool(np.isfinite(h_min_eig) and h_min_eig >= -PSD_TOL)
    return {
        "coeff": coeff,
        "pred_max_abs_mw": max_abs,
        "pred_rmse_mw": rmse,
        "pred_max_abs_rel_baseline": max_abs / baseline_scale,
        "pred_rmse_rel_baseline": rmse / baseline_scale,
        "h_min_eig": h_min_eig,
        "a_P_ok": a_p_ok,
        "a_Q_ok": a_q_ok,
        "a_Q_identified": a_q_identified,
        "psd_ok": psd_ok,
        "gate_all_ok": bool(a_p_ok and a_q_ok and psd_ok),
    }


def _fieldnames() -> list[str]:
    fields = [
        "bus",
        "S_mva",
        "scenario",
        "t",
        "p_center_fraction",
        "p_center_mw",
        "region",
        "truth_center_mw",
        "truth_sample_count",
        "truth_pf_attempts_actual",
        "truth_baseline_loss_mw",
    ]
    fields.extend(f"truth_{name}" for name in COEFF_NAMES)
    for method in METHODS:
        fields.extend(f"{method}_{name}" for name in COEFF_NAMES)
        fields.extend(
            [
                f"{method}_measurement_center_mw",
                f"{method}_sample_count",
                f"{method}_pf_attempts_actual",
                f"{method}_pf_retries",
                f"{method}_pf_failures",
                f"{method}_fit_rank",
                f"{method}_rank_lt5",
                f"{method}_source_min7_rank",
                f"{method}_source_q_column_norm",
                f"{method}_fallback_used",
                f"{method}_q_gate_exempt",
                f"{method}_a_Q_identified",
                f"{method}_pred_max_abs_mw",
                f"{method}_pred_rmse_mw",
                f"{method}_pred_max_abs_rel_baseline",
                f"{method}_pred_rmse_rel_baseline",
                f"{method}_h_min_eig",
                f"{method}_a_P_ok",
                f"{method}_a_Q_ok",
                f"{method}_psd_ok",
                f"{method}_gate_all_ok",
            ]
        )
    return fields


def _make_row(
    bus: int,
    S: float,
    scenario: str,
    t: int,
    p_fraction: float,
    truth: dict,
    methods: dict[str, tuple[dict, dict]],
) -> dict:
    row: dict[str, object] = {
        "bus": bus,
        "S_mva": S,
        "scenario": scenario,
        "t": t,
        "p_center_fraction": p_fraction,
        "p_center_mw": p_fraction * S,
        "region": "boundary" if p_fraction >= BOUNDARY_FRACTION else "normal",
        "truth_center_mw": truth["measurement_center_mw"],
        "truth_sample_count": truth["sample_count"],
        "truth_pf_attempts_actual": truth["pf_attempts_actual"],
        "truth_baseline_loss_mw": truth["baseline_loss_mw"],
    }
    for name in COEFF_NAMES:
        row[f"truth_{name}"] = truth["fit"][name]
    for method, (measured, metrics) in methods.items():
        for index, name in enumerate(COEFF_NAMES):
            row[f"{method}_{name}"] = metrics["coeff"][index]
        row.update(
            {
                f"{method}_measurement_center_mw": measured["measurement_center_mw"],
                f"{method}_sample_count": measured["sample_count"],
                f"{method}_pf_attempts_actual": measured["pf_attempts_actual"],
                f"{method}_pf_retries": measured["pf_retries"],
                f"{method}_pf_failures": measured["pf_failures"],
                f"{method}_fit_rank": measured["fit"]["fit_rank"],
                f"{method}_rank_lt5": measured["fit"]["fit_rank"] < 5,
                f"{method}_source_min7_rank": measured["source_min7_rank"],
                f"{method}_source_q_column_norm": measured["source_q_column_norm"],
                f"{method}_fallback_used": measured["fallback_used"],
                f"{method}_q_gate_exempt": measured["q_gate_exempt"],
            }
        )
        for key in (
            "a_Q_identified",
            "pred_max_abs_mw",
            "pred_rmse_mw",
            "pred_max_abs_rel_baseline",
            "pred_rmse_rel_baseline",
            "h_min_eig",
            "a_P_ok",
            "a_Q_ok",
            "psd_ok",
            "gate_all_ok",
        ):
            row[f"{method}_{key}"] = metrics[key]
    return row


def _finite_stat(values: list[float], percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def _print_condition(index: int, total: int, row: dict) -> None:
    print(
        f"[condition {index:03d}/{total}] bus={row['bus']} S={row['S_mva']:.3f} "
        f"scenario={row['scenario']} t={row['t']} "
        f"p_center={row['p_center_fraction']:.2f}S region={row['region']} "
        f"truth_points={row['truth_sample_count']}",
        flush=True,
    )
    for method in METHODS:
        print(
            f"  {method:22s} centre={float(row[f'{method}_measurement_center_mw']):.6f} "
            f"n={row[f'{method}_sample_count']} rank={row[f'{method}_fit_rank']} "
            f"source_rank={row[f'{method}_source_min7_rank']} "
            f"aQ_ident={row[f'{method}_a_Q_identified']} "
            f"rel(max/rmse)={float(row[f'{method}_pred_max_abs_rel_baseline']):.6e}/"
            f"{float(row[f'{method}_pred_rmse_rel_baseline']):.6e} "
            f"pf={row[f'{method}_pf_attempts_actual']} "
            f"fallback={row[f'{method}_fallback_used']} "
            f"gates(P/Q/PSD)={row[f'{method}_a_P_ok']}/"
            f"{row[f'{method}_a_Q_ok']}/{row[f'{method}_psd_ok']}",
            flush=True,
        )


def _print_aggregate(rows: list[dict]) -> None:
    print("\n[aggregate: normal/boundary separated]", flush=True)
    print(
        "region    method                  relmax_med   relmax_p95   relmax_max   "
        "rmse_p95    pf_mean fallback_rate ranklt5 aQ_unidentified gate_bad",
        flush=True,
    )
    for region in ("normal", "boundary"):
        selected = [row for row in rows if row["region"] == region]
        for method in METHODS:
            relmax = [
                float(row[f"{method}_pred_max_abs_rel_baseline"])
                for row in selected
            ]
            rmse = [
                float(row[f"{method}_pred_rmse_rel_baseline"])
                for row in selected
            ]
            pf = [float(row[f"{method}_pf_attempts_actual"]) for row in selected]
            fallback_rate = float(
                np.mean([bool(row[f"{method}_fallback_used"]) for row in selected])
            )
            rank_lt5 = sum(bool(row[f"{method}_rank_lt5"]) for row in selected)
            aq_missing = sum(
                not bool(row[f"{method}_a_Q_identified"]) for row in selected
            )
            gate_bad = sum(not bool(row[f"{method}_gate_all_ok"]) for row in selected)
            print(
                f"{region:9s} {method:22s} "
                f"{_finite_stat(relmax, 50):11.6e} "
                f"{_finite_stat(relmax, 95):12.6e} "
                f"{_finite_stat(relmax, 100):12.6e} "
                f"{_finite_stat(rmse, 95):11.6e} "
                f"{_finite_stat(pf, 50) if not pf else float(np.mean(pf)):8.2f} "
                f"{fallback_rate:13.6f} {rank_lt5:7d} "
                f"{aq_missing:15d} {gate_bad:8d}",
                flush=True,
            )

    boundary = [row for row in rows if row["region"] == "boundary"]
    fallback_pf = [
        float(row["full25_fallback_pf_attempts_actual"]) for row in boundary
    ]
    fallback_rate = float(
        np.mean([bool(row["full25_fallback_fallback_used"]) for row in boundary])
    )
    print("\n[full25 fallback cost on boundary conditions]", flush=True)
    print(f"fallback_condition_rate={fallback_rate:.12g}", flush=True)
    print(f"pf_attempts_mean={float(np.mean(fallback_pf)):.12g}", flush=True)
    print("\n[reference only; no automatic pass/fail]", flush=True)
    print("prior min7 96-condition worst relative loss-surface error = 0.054%", flush=True)
    print("CLAUDE.md observed five-coefficient fit error = 0.02% to 0.07%", flush=True)


def main() -> int:
    total_conditions = (
        len(BUSES)
        * len(S_VALUES)
        * len(SCENARIOS)
        * len(TIMES)
        * len(P_CENTER_FRACTIONS)
    )
    print("[setup] PCS-boundary degeneracy remedy probe", flush=True)
    print(f"conditions={total_conditions}", flush=True)
    print(f"methods={METHODS}", flush=True)
    print(
        f"shrink_thresholds={SHRINK_THRESHOLDS} pull_targets={PULL_TARGETS}",
        flush=True,
    )
    print(
        "truth_boundary_definition=p_center_fraction>=0.9 uses dense grid at 0.9S",
        flush=True,
    )
    print(f"output={OUTPUT_PATH}", flush=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    net = build_net()
    rows: list[dict] = []
    started = time.perf_counter()
    condition_index = 0

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=_fieldnames())
        writer.writeheader()
        for bus in BUSES:
            for S in S_VALUES:
                for scenario in SCENARIOS:
                    for t in TIMES:
                        for p_fraction in P_CENTER_FRACTIONS:
                            condition_index += 1
                            p_center = p_fraction * S
                            truth = _measure_truth(
                                net, bus, S, scenario, t, p_fraction
                            )
                            measured_methods: dict[str, tuple[dict, dict]] = {}

                            fallback = _measure_fallback(
                                net, bus, S, scenario, t, p_center
                            )
                            measured_methods["full25_fallback"] = (
                                fallback,
                                _metrics(fallback, truth),
                            )

                            for threshold in SHRINK_THRESHOLDS:
                                for target in PULL_TARGETS:
                                    name = f"pull_t{threshold:.2f}_p{target:.2f}"
                                    pulled = _measure_pull(
                                        net,
                                        bus,
                                        S,
                                        scenario,
                                        t,
                                        p_center,
                                        threshold,
                                        target,
                                    )
                                    measured_methods[name] = (
                                        pulled,
                                        _metrics(pulled, truth),
                                    )

                            qzero = _measure_qzero(
                                net, bus, S, scenario, t, p_center
                            )
                            measured_methods["qzero_gate"] = (
                                qzero,
                                _metrics(qzero, truth),
                            )

                            row = _make_row(
                                bus,
                                S,
                                scenario,
                                t,
                                p_fraction,
                                truth,
                                measured_methods,
                            )
                            writer.writerow(row)
                            handle.flush()
                            rows.append(row)
                            _print_condition(condition_index, total_conditions, row)

    _print_aggregate(rows)
    total_pf = sum(
        int(row["truth_pf_attempts_actual"])
        + sum(int(row[f"{method}_pf_attempts_actual"]) for method in METHODS)
        for row in rows
    )
    elapsed = time.perf_counter() - started
    print(f"\ntotal_pf_attempts_actual={total_pf}", flush=True)
    print(f"total_runtime_s={elapsed:.6f}", flush=True)
    print(f"csv={OUTPUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
