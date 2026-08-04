"""Measure accuracy/cost trade-offs for reduced loss-coefficient grids.

Purpose
-------
Compare three reduced sampling designs for the local five-coefficient loss
surface without changing production modules.  The reference ("truth") is a
dense 81-point design: nine local P levels and nine adaptive Q levels per P,
with every point kept inside the PCS circle.  The candidates are:

* ``ccd9``: local centre, four axial points, and four corner points (9 points),
* ``q3``: the current five P levels with Q reduced to ``-q_max, 0, +q_max``
  (up to 15 points),
* ``min7``: centre, four axial points, and two opposite diagonal points
  (7 points).

Coordinates described as a centre at (0, 0) are local coordinates around
``(p_center, 0)``; stored/fitted P values remain absolute feeder injections.
All fits use ``loss_coeffs._fit_samples``, including its column-scaled least
squares implementation.  Prediction residuals are evaluated against the AC
measurements at all 81 truth-design points.  This script prints reference
accuracy lines only; it deliberately makes no automatic pass/fail decision.

``loss_coeffs.measure_loss_reduction`` recomputes the zero-injection baseline
on every point call.  Therefore the CSV records both ``pf_design_count``
(one hypothetically shared baseline plus grid points) and the API-reported
``pf_attempts_actual``.  The latter is the actual power-flow attempt count.

Run from the project root (conceptual shell form)::

    MKL_THREADING_LAYER=SEQUENTIAL python scripts/probe_coef_grid_reduction.py

Windows CMD (absolute path)::

    set MKL_THREADING_LAYER=SEQUENTIAL
    python C:\\Users\\PowerSL\\summer_ess_project\\scripts\\probe_coef_grid_reduction.py

Output is written to ``scripts/probe_coef_grid_reduction_out.csv``.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

# Allow the documented absolute-path command to work from any current directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import loss_coeffs as LC
from build_net import build_net


BUSES = (16, 15, 30, 6)
S_VALUES = (0.15, 0.30, 0.60)
SCENARIOS = ("summer", "winter")
TIMES = (10, 18)
P_CENTER_FRACTIONS = (0.0, 0.5)

COEFF_NAMES = tuple(LC.COEFF_NAMES)
CANDIDATES = ("ccd9", "q3", "min7")
PSD_TOL = 1e-9
COEF_REL_EPS = 1e-12
SCALE_EPS_MW = 1e-12
POINT_TOL = 1e-12
OUTPUT_PATH = Path(__file__).resolve().with_name(
    "probe_coef_grid_reduction_out.csv"
)


def _deduplicate_feasible(
    points: Iterable[tuple[float, float]], S: float
) -> list[tuple[float, float]]:
    """Radially clip round-off excursions and remove duplicate PCS points."""
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
            raise AssertionError(f"PCS-infeasible grid point: P={p}, Q={q}, S={S}")
        key = (round(p, 14), round(q, 14))
        if key not in seen:
            seen.add(key)
            unique.append((p, q))
    return unique


def _ccd_geometry(S: float, p_center: float) -> tuple[float, float]:
    """Return symmetric local P radius and Q radius for a feasible CCD.

    The corner constraint is evaluated for the adverse P sign (the one moving
    away from zero), so every un-clipped point lies inside the absolute PCS
    circle even when ``p_center`` is nonzero.
    """
    q_at_center = math.sqrt(max(S * S - p_center * p_center, 0.0))
    p_axis_limit = max(S - abs(p_center), 0.0)
    corner_p_limit = math.sqrt(2.0) * max(
        math.sqrt(max(S * S - 0.5 * q_at_center * q_at_center, 0.0))
        - abs(p_center),
        0.0,
    )
    s_eff = min(p_axis_limit, corner_p_limit)
    return float(s_eff), float(q_at_center)


def _grid_ccd9(S: float, p_center: float) -> list[tuple[float, float]]:
    """Nine-point centre/axis/corner design in local (delta-P, Q) space."""
    s_eff, q_center = _ccd_geometry(S, p_center)
    root2 = math.sqrt(2.0)
    points = [(p_center, 0.0)]
    points.extend(
        [
            (p_center - s_eff, 0.0),
            (p_center + s_eff, 0.0),
            (p_center, -q_center),
            (p_center, q_center),
        ]
    )
    for p_sign in (-1.0, 1.0):
        for q_sign in (-1.0, 1.0):
            points.append(
                (
                    p_center + p_sign * s_eff / root2,
                    q_sign * q_center / root2,
                )
            )
    return _deduplicate_feasible(points, S)


def _grid_q3(S: float, p_center: float) -> list[tuple[float, float]]:
    """Current five local P levels with three adaptive Q levels per P."""
    p_values = np.unique(np.clip(p_center + S * LC.P_GRID, -S, S))
    points: list[tuple[float, float]] = []
    for p in p_values:
        q_max = math.sqrt(max(S * S - float(p) ** 2, 0.0))
        points.extend((float(p), q) for q in (-q_max, 0.0, q_max))
    return _deduplicate_feasible(points, S)


def _grid_min7(S: float, p_center: float) -> list[tuple[float, float]]:
    """Seven-point lower-bound design: centre, axes, two diagonals."""
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
    """Nine local P levels by nine adaptive Q levels, at most 81 points."""
    p_offsets = np.linspace(float(LC.P_GRID.min()), float(LC.P_GRID.max()), 9)
    p_values = np.unique(np.clip(p_center + S * p_offsets, -S, S))
    points: list[tuple[float, float]] = []
    for p in p_values:
        q_max = math.sqrt(max(S * S - float(p) ** 2, 0.0))
        points.extend(
            (float(p), float(q)) for q in np.linspace(-q_max, q_max, 9)
        )
    return _deduplicate_feasible(points, S)


GRID_BUILDERS: dict[str, Callable[[float, float], list[tuple[float, float]]]] = {
    "truth81": _grid_dense81,
    "ccd9": _grid_ccd9,
    "q3": _grid_q3,
    "min7": _grid_min7,
}


def _measure_grid(
    net,
    bus: int,
    S: float,
    scenario: str,
    t: int,
    p_center: float,
    design: str,
) -> dict:
    """Measure one design point-by-point through the production public API."""
    points = GRID_BUILDERS[design](S, p_center)
    samples: list[dict] = []
    attempts = retries = failures = 0
    baselines: list[float] = []

    for p, q in points:
        measurement = LC.measure_loss_reduction(
            bus,
            scenario,
            t,
            p,
            q,
            net=net,
        )
        attempts += int(measurement["runpp_attempts"])
        retries += int(measurement["runpp_retries"])
        failures += int(measurement["runpp_failures"])
        baselines.append(float(measurement["baseline_loss_mw"]))
        samples.append(
            {
                "p_mw": float(p),
                "q_mvar": float(q),
                "L_cost_mw": float(measurement["L_cost_mw"]),
            }
        )

    fit = LC._fit_samples(samples)
    finite_baselines = np.asarray(baselines, dtype=float)
    finite_baselines = finite_baselines[np.isfinite(finite_baselines)]
    baseline_loss = (
        float(np.median(finite_baselines))
        if finite_baselines.size
        else float("nan")
    )
    baseline_spread = (
        float(np.max(finite_baselines) - np.min(finite_baselines))
        if finite_baselines.size
        else float("nan")
    )
    return {
        "design": design,
        "points": points,
        "samples": samples,
        "fit": fit,
        "baseline_loss_mw": baseline_loss,
        "baseline_spread_mw": baseline_spread,
        "sample_count": len(points),
        "pf_design_count": len(points) + 1,
        "pf_attempts_actual": attempts,
        "pf_retries": retries,
        "pf_failures": failures,
    }


def _prediction_matrix(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    design = np.asarray(
        [
            [
                row["p_mw"],
                row["q_mvar"],
                row["p_mw"] ** 2,
                row["q_mvar"] ** 2,
                row["p_mw"] * row["q_mvar"],
            ]
            for row in samples
        ],
        dtype=float,
    )
    measured = np.asarray([row["L_cost_mw"] for row in samples], dtype=float)
    return design, measured


def _candidate_metrics(candidate: dict, truth: dict) -> dict:
    truth_coeff = np.asarray(
        [truth["fit"][name] for name in COEFF_NAMES], dtype=float
    )
    candidate_coeff = np.asarray(
        [candidate["fit"][name] for name in COEFF_NAMES], dtype=float
    )
    coef_rel = np.abs(candidate_coeff - truth_coeff) / (
        np.abs(truth_coeff) + COEF_REL_EPS
    )

    truth_design, truth_measured = _prediction_matrix(truth["samples"])
    residual = truth_design @ candidate_coeff - truth_measured
    finite = np.isfinite(residual)
    if np.any(finite):
        max_abs = float(np.max(np.abs(residual[finite])))
        rmse = float(np.sqrt(np.mean(residual[finite] ** 2)))
    else:
        max_abs = rmse = float("nan")

    baseline_scale = max(abs(float(truth["baseline_loss_mw"])), SCALE_EPS_MW)
    H = np.asarray(
        [
            [candidate_coeff[2], candidate_coeff[4] / 2.0],
            [candidate_coeff[4] / 2.0, candidate_coeff[3]],
        ],
        dtype=float,
    )
    min_eig = (
        float(np.linalg.eigvalsh(H)[0])
        if np.all(np.isfinite(H))
        else float("nan")
    )
    return {
        "coeff": candidate_coeff,
        "coef_rel": coef_rel,
        "coef_rel_max": float(np.nanmax(coef_rel)),
        "coef_rel_mean": float(np.nanmean(coef_rel)),
        "pred_max_abs_mw": max_abs,
        "pred_rmse_mw": rmse,
        "pred_max_abs_rel_baseline": max_abs / baseline_scale,
        "pred_rmse_rel_baseline": rmse / baseline_scale,
        "h_min_eig": min_eig,
        "psd_ok": bool(np.isfinite(min_eig) and min_eig >= -PSD_TOL),
        "a_P_negative": bool(np.isfinite(candidate_coeff[0]) and candidate_coeff[0] < 0.0),
        "a_Q_negative": bool(np.isfinite(candidate_coeff[1]) and candidate_coeff[1] < 0.0),
    }


def _base_fieldnames() -> list[str]:
    fields = ["bus", "S_mva", "scenario", "t", "p_center_mw"]
    fields.extend(f"truth_{name}" for name in COEFF_NAMES)
    fields.extend(
        [
            "truth_fit_max_abs_error_mw",
            "truth_fit_rmse_mw",
            "truth_baseline_loss_mw",
            "truth_baseline_spread_mw",
            "truth_sample_count",
            "truth_pf_design_count",
            "truth_pf_attempts_actual",
            "truth_pf_retries",
            "truth_pf_failures",
        ]
    )
    for candidate in CANDIDATES:
        fields.extend(f"{candidate}_{name}" for name in COEFF_NAMES)
        fields.extend(f"{candidate}_{name}_rel_error" for name in COEFF_NAMES)
        fields.extend(
            [
                f"{candidate}_coef_rel_max",
                f"{candidate}_coef_rel_mean",
                f"{candidate}_pred_max_abs_mw",
                f"{candidate}_pred_rmse_mw",
                f"{candidate}_pred_max_abs_rel_baseline",
                f"{candidate}_pred_rmse_rel_baseline",
                f"{candidate}_h_min_eig",
                f"{candidate}_psd_ok",
                f"{candidate}_a_P_negative",
                f"{candidate}_a_Q_negative",
                f"{candidate}_sample_count",
                f"{candidate}_pf_design_count",
                f"{candidate}_pf_attempts_actual",
                f"{candidate}_pf_retries",
                f"{candidate}_pf_failures",
            ]
        )
    return fields


def _make_row(
    bus: int,
    S: float,
    scenario: str,
    t: int,
    p_center: float,
    truth: dict,
    candidates: dict[str, tuple[dict, dict]],
) -> dict:
    row: dict[str, object] = {
        "bus": bus,
        "S_mva": S,
        "scenario": scenario,
        "t": t,
        "p_center_mw": p_center,
    }
    for name in COEFF_NAMES:
        row[f"truth_{name}"] = truth["fit"][name]
    row.update(
        {
            "truth_fit_max_abs_error_mw": truth["fit"]["fit_max_abs_error_mw"],
            "truth_fit_rmse_mw": truth["fit"]["fit_rmse_mw"],
            "truth_baseline_loss_mw": truth["baseline_loss_mw"],
            "truth_baseline_spread_mw": truth["baseline_spread_mw"],
            "truth_sample_count": truth["sample_count"],
            "truth_pf_design_count": truth["pf_design_count"],
            "truth_pf_attempts_actual": truth["pf_attempts_actual"],
            "truth_pf_retries": truth["pf_retries"],
            "truth_pf_failures": truth["pf_failures"],
        }
    )
    for candidate, (measurement, metrics) in candidates.items():
        for index, name in enumerate(COEFF_NAMES):
            row[f"{candidate}_{name}"] = metrics["coeff"][index]
            row[f"{candidate}_{name}_rel_error"] = metrics["coef_rel"][index]
        for name in (
            "coef_rel_max",
            "coef_rel_mean",
            "pred_max_abs_mw",
            "pred_rmse_mw",
            "pred_max_abs_rel_baseline",
            "pred_rmse_rel_baseline",
            "h_min_eig",
            "psd_ok",
            "a_P_negative",
            "a_Q_negative",
        ):
            row[f"{candidate}_{name}"] = metrics[name]
        for name in (
            "sample_count",
            "pf_design_count",
            "pf_attempts_actual",
            "pf_retries",
            "pf_failures",
        ):
            row[f"{candidate}_{name}"] = measurement[name]
    return row


def _fmt(value: float) -> str:
    return f"{value:.6e}" if np.isfinite(value) else "nan"


def _print_condition(
    condition_index: int,
    total_conditions: int,
    row: dict,
) -> None:
    print(
        f"[condition {condition_index:02d}/{total_conditions}] "
        f"bus={row['bus']} S={row['S_mva']:.3f} scenario={row['scenario']} "
        f"t={row['t']} p_center={row['p_center_mw']:.6f}"
    )
    print(
        "  truth81: "
        f"samples={row['truth_sample_count']} "
        f"pf_design={row['truth_pf_design_count']} "
        f"pf_actual={row['truth_pf_attempts_actual']} "
        f"fit_max_abs={_fmt(float(row['truth_fit_max_abs_error_mw']))} MW"
    )
    for candidate in CANDIDATES:
        coef_rel_each = ",".join(
            f"{name}={_fmt(float(row[f'{candidate}_{name}_rel_error']))}"
            for name in COEFF_NAMES
        )
        print(
            f"  {candidate:6s}: "
            f"coef_rel(max/mean)="
            f"{_fmt(float(row[f'{candidate}_coef_rel_max']))}/"
            f"{_fmt(float(row[f'{candidate}_coef_rel_mean']))}  "
            f"pred(max/rmse)="
            f"{_fmt(float(row[f'{candidate}_pred_max_abs_mw']))}/"
            f"{_fmt(float(row[f'{candidate}_pred_rmse_mw']))} MW  "
            f"rel(max/rmse)="
            f"{_fmt(float(row[f'{candidate}_pred_max_abs_rel_baseline']))}/"
            f"{_fmt(float(row[f'{candidate}_pred_rmse_rel_baseline']))}  "
            f"pf(design/actual)={row[f'{candidate}_pf_design_count']}/"
            f"{row[f'{candidate}_pf_attempts_actual']}  "
            f"eig_min={_fmt(float(row[f'{candidate}_h_min_eig']))}  "
            f"sign(P/Q)={row[f'{candidate}_a_P_negative']}/"
            f"{row[f'{candidate}_a_Q_negative']}"
        )
        print(f"           coef_rel_each: {coef_rel_each}")


def _finite_percentile(values: list[float], percentile: float) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return float(np.percentile(array, percentile)) if array.size else float("nan")


def _print_aggregate(rows: list[dict]) -> None:
    print("\n[aggregate] loss-surface prediction error over 96 conditions")
    print(
        "candidate  rel_max_median  rel_max_p95  rel_max_max  "
        "rel_rmse_p95  pf_design_med  pf_actual_med  PSD_bad  sign_bad"
    )
    for candidate in CANDIDATES:
        rel_max = [float(row[f"{candidate}_pred_max_abs_rel_baseline"]) for row in rows]
        rel_rmse = [float(row[f"{candidate}_pred_rmse_rel_baseline"]) for row in rows]
        pf_design = [float(row[f"{candidate}_pf_design_count"]) for row in rows]
        pf_actual = [float(row[f"{candidate}_pf_attempts_actual"]) for row in rows]
        psd_bad = sum(not bool(row[f"{candidate}_psd_ok"]) for row in rows)
        sign_bad = sum(
            not (
                bool(row[f"{candidate}_a_P_negative"])
                and bool(row[f"{candidate}_a_Q_negative"])
            )
            for row in rows
        )
        print(
            f"{candidate:9s} "
            f"{_finite_percentile(rel_max, 50):14.6e} "
            f"{_finite_percentile(rel_max, 95):12.6e} "
            f"{_finite_percentile(rel_max, 100):12.6e} "
            f"{_finite_percentile(rel_rmse, 95):13.6e} "
            f"{_finite_percentile(pf_design, 50):13.1f} "
            f"{_finite_percentile(pf_actual, 50):13.1f} "
            f"{psd_bad:7d} {sign_bad:9d}"
        )
    print("\n[reference only; no automatic pass/fail]")
    print("CLAUDE.md observed five-coefficient fit error: 0.02% to 0.07%")


def main() -> int:
    total_conditions = (
        len(BUSES)
        * len(S_VALUES)
        * len(SCENARIOS)
        * len(TIMES)
        * len(P_CENTER_FRACTIONS)
    )
    nominal_points_per_condition = sum(
        len(builder(0.30, 0.0)) for builder in GRID_BUILDERS.values()
    )
    print("[setup] coefficient-grid reduction accuracy probe")
    print(f"conditions={total_conditions}")
    print(
        "design_counts_at_p_center_0="
        + ", ".join(
            f"{name}:{len(builder(0.30, 0.0))}+baseline"
            for name, builder in GRID_BUILDERS.items()
        )
    )
    print(
        f"nominal_grid_points={total_conditions * nominal_points_per_condition} "
        "(actual runpp attempts are read from measure_loss_reduction)"
    )
    print(f"output={OUTPUT_PATH}")

    started = time.perf_counter()
    net = build_net()
    rows: list[dict] = []
    condition_index = 0

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=_base_fieldnames())
        writer.writeheader()
        for bus in BUSES:
            for S in S_VALUES:
                for scenario in SCENARIOS:
                    for t in TIMES:
                        for fraction in P_CENTER_FRACTIONS:
                            condition_index += 1
                            p_center = fraction * S
                            truth = _measure_grid(
                                net, bus, S, scenario, t, p_center, "truth81"
                            )
                            measured_candidates: dict[str, tuple[dict, dict]] = {}
                            for candidate in CANDIDATES:
                                measurement = _measure_grid(
                                    net,
                                    bus,
                                    S,
                                    scenario,
                                    t,
                                    p_center,
                                    candidate,
                                )
                                measured_candidates[candidate] = (
                                    measurement,
                                    _candidate_metrics(measurement, truth),
                                )
                            row = _make_row(
                                bus,
                                S,
                                scenario,
                                t,
                                p_center,
                                truth,
                                measured_candidates,
                            )
                            writer.writerow(row)
                            handle.flush()
                            rows.append(row)
                            _print_condition(condition_index, total_conditions, row)

    _print_aggregate(rows)
    total_pf_actual = sum(
        int(row["truth_pf_attempts_actual"])
        + sum(int(row[f"{name}_pf_attempts_actual"]) for name in CANDIDATES)
        for row in rows
    )
    elapsed = time.perf_counter() - started
    print(f"\ntotal_pf_attempts_actual={total_pf_actual}")
    print(f"total_runtime_s={elapsed:.3f}")
    print(f"csv={OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
