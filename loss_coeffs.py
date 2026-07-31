"""AC power-flow measurement of the five-coefficient feeder-loss surface.

This module is the measurement layer between ``build_net.py`` and the QP in
``lower_lp.py``.  It deliberately does not solve schedules.  At one operating
condition ``(bus, S, scenario, t, p_center)`` it measures

    delta_loss_reduction = Loss(0, 0) - Loss(P, Q)
    L_cost = -delta_loss_reduction = Loss(P, Q) - Loss(0, 0)

in MW and fits, without an intercept,

    L_cost = a_P P + a_Q Q + b_PP P^2 + b_QQ Q^2 + b_PQ P Q.

The module-level cache is process-local.  Under multiprocessing each worker
therefore owns an independent cache, matching the worker-local network pattern
used by ``evaluate.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import time

import numpy as np
import pandapower as pp

import params as PM
from build_net import build_net


COEFF_NAMES = ("a_P", "a_Q", "b_PP", "b_QQ", "b_PQ")
P_GRID = np.asarray(
    getattr(PM, "LOSS_COEF_P_GRID", [-0.25, -0.125, 0.0, 0.125, 0.25]),
    dtype=float,
)
N_Q_GRID = 5
PSD_TOL = 1e-9
GRID_TOL = 1e-12
PF_TOLERANCE_DEFAULT_MVA = 1e-8
PF_TOLERANCE_SMALL_S_MVA = 1e-12
PF_TOLERANCE_SMALL_S_MAX_MVA = 1e-3
_PROBE_NAME = "LOSS_COEFF_PROBE"

_measured_cache: dict[tuple[int, float, str, int, float], dict] = {}
_cache_hits = 0
_cache_misses = 0
_runpp_attempt_count = 0
_runpp_retry_count = 0
_runpp_failure_count = 0
_measure_wall_s = 0.0


@dataclass
class _RunStats:
    attempts: int = 0
    retries: int = 0
    failures: int = 0


def _cache_key_coeffs(
    bus: int, S: float, scenario: str, t: int, p_center: float
) -> tuple[int, float, str, int, float]:
    """Return the exact-float cache key; S is intentionally not rounded."""
    return int(bus), float(S), str(scenario), int(t), float(p_center)


def clear_cache() -> None:
    """Clear measured coefficients and reset cache hit/miss counters."""
    global _cache_hits, _cache_misses
    _measured_cache.clear()
    _cache_hits = 0
    _cache_misses = 0


def cache_info() -> dict:
    """Return process-local cache and power-flow counters."""
    return {
        "size": len(_measured_cache),
        "hits": int(_cache_hits),
        "misses": int(_cache_misses),
        "runpp_attempts": int(_runpp_attempt_count),
        "runpp_retries": int(_runpp_retry_count),
        "runpp_failures": int(_runpp_failure_count),
        "measure_wall_s": float(_measure_wall_s),
    }


def _copy_result(result: dict) -> dict:
    copied = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        elif isinstance(value, list):
            copied[key] = [dict(item) if isinstance(item, dict) else item for item in value]
        else:
            copied[key] = value
    return copied


def _validate_inputs(
    net, bus: int, S: float, scenario: str, t: int, p_center: float
) -> tuple[int, float, str, int, float]:
    bus = int(bus)
    S = float(S)
    scenario = str(scenario)
    t = int(t)
    p_center = float(p_center)
    if bus not in net.bus.index:
        raise ValueError(f"unknown bus={bus}")
    if not np.isfinite(S) or S <= 0.0:
        raise ValueError(f"S must be finite and positive, got {S}")
    if scenario not in PM.LOAD:
        raise ValueError(f"unknown scenario={scenario!r}")
    if not 0 <= t < PM.TIME_STEPS:
        raise ValueError(f"t={t} outside [0,{PM.TIME_STEPS})")
    if not np.isfinite(p_center):
        raise ValueError(f"p_center must be finite, got {p_center}")
    if abs(p_center) > S + GRID_TOL:
        raise ValueError(f"|p_center|={abs(p_center)} exceeds S={S}")
    return bus, S, scenario, t, p_center


def _ensure_probe_sgen(net) -> int:
    if len(net.sgen):
        names = net.sgen["name"].astype(str)
        matches = list(net.sgen.index[names == _PROBE_NAME])
        if matches:
            if len(matches) != 1:
                raise RuntimeError(f"multiple {_PROBE_NAME!r} sgens found")
            return int(matches[0])
        raise ValueError(
            "supplied net already contains non-probe sgen rows; use a dedicated "
            "measurement net to avoid contaminating feeder-loss coefficients"
        )
    return int(
        pp.create_sgen(
            net, bus=int(PM.B_BOUNDS[0]), p_mw=0.0, q_mvar=0.0, name=_PROBE_NAME
        )
    )


def _set_load(
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    scenario: str,
    t: int,
) -> None:
    scale = float(PM.LOAD[scenario][t])
    net.load["p_mw"] = base_p * scale
    net.load["q_mvar"] = base_q * scale


def _pf_tolerance_mva(S: float) -> float:
    """Use tighter power-flow convergence only for numerically tiny surfaces."""
    if S <= PF_TOLERANCE_SMALL_S_MAX_MVA:
        return PF_TOLERANCE_SMALL_S_MVA
    return PF_TOLERANCE_DEFAULT_MVA


def _run_pf_with_retry(
    net, stats: _RunStats, tolerance_mva: float
) -> bool:
    """Use the evaluate.py policy: init='results', then init='flat'."""
    global _runpp_attempt_count, _runpp_retry_count, _runpp_failure_count

    stats.attempts += 1
    _runpp_attempt_count += 1
    try:
        pp.runpp(
            net,
            numba=True,
            init="results",
            tolerance_mva=tolerance_mva,
        )
        return True
    except Exception:
        stats.retries += 1
        _runpp_retry_count += 1

    stats.attempts += 1
    _runpp_attempt_count += 1
    try:
        pp.runpp(
            net,
            numba=True,
            init="flat",
            tolerance_mva=tolerance_mva,
        )
        return True
    except Exception:
        stats.failures += 1
        _runpp_failure_count += 1
        return False


def _network_loss_mw(net) -> float:
    """Total feeder loss in MW: lines plus any in-service transformer tables."""
    total = float(net.res_line["pl_mw"].sum())
    for element, result_name in (
        ("trafo", "res_trafo"),
        ("trafo3w", "res_trafo3w"),
    ):
        table = getattr(net, element, None)
        results = getattr(net, result_name, None)
        if table is None or results is None or len(table) == 0:
            continue
        mask = (
            table["in_service"].to_numpy(dtype=bool)
            if "in_service" in table
            else np.ones(len(table), dtype=bool)
        )
        total += float(results.loc[table.index[mask], "pl_mw"].sum())
    return total


def _local_grid(S: float, p_center: float) -> list[tuple[float, float]]:
    p_values = np.unique(np.clip(p_center + S * P_GRID, -S, S))
    grid: list[tuple[float, float]] = []
    for p_value in p_values:
        p = float(p_value)
        q_max = float(np.sqrt(max(0.0, S * S - p * p)))
        q_values = np.unique(np.linspace(-q_max, q_max, N_Q_GRID))
        for q_value in q_values:
            q = float(q_value)
            if np.hypot(p, q) <= S + GRID_TOL:
                grid.append((p, q))
    return grid


def _measure_loss(
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    probe_idx: int,
    bus: int,
    scenario: str,
    t: int,
    p_mw: float,
    q_mvar: float,
    stats: _RunStats,
    tolerance_mva: float,
) -> float:
    _set_load(net, base_p, base_q, scenario, t)
    net.sgen.at[probe_idx, "bus"] = int(bus)
    net.sgen.at[probe_idx, "p_mw"] = float(p_mw)
    net.sgen.at[probe_idx, "q_mvar"] = float(q_mvar)
    if not _run_pf_with_retry(net, stats, tolerance_mva):
        return float("nan")
    return _network_loss_mw(net)


def _fit_samples(samples: list[dict]) -> dict:
    converged = [row for row in samples if np.isfinite(row["L_cost_mw"])]
    if not converged:
        beta = np.full(5, np.nan)
        rank = 0
        residual = np.empty(0, dtype=float)
        y = np.empty(0, dtype=float)
    else:
        design = np.asarray(
            [
                [
                    row["p_mw"],
                    row["q_mvar"],
                    row["p_mw"] ** 2,
                    row["q_mvar"] ** 2,
                    row["p_mw"] * row["q_mvar"],
                ]
                for row in converged
            ],
            dtype=float,
        )
        y = np.asarray([row["L_cost_mw"] for row in converged], dtype=float)
        # P/Q 열은 O(S), 2차 열은 O(S²)이므로 작은 S에서 원행렬의 조건수가
        # 급격히 커진다. 열별 크기를 1로 맞춰 적합한 뒤 물리 단위 계수로 되돌린다.
        column_scale = np.max(np.abs(design), axis=0)
        column_scale[column_scale == 0.0] = 1.0
        design_scaled = design / column_scale
        beta_scaled, _, rank_value, _ = np.linalg.lstsq(
            design_scaled, y, rcond=None
        )
        beta = beta_scaled / column_scale
        rank = int(rank_value)
        residual = design @ beta - y
        scaled_condition = float(np.linalg.cond(design_scaled))

    if len(residual):
        max_abs_error = float(np.max(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual**2)))
        scale = float(np.max(np.abs(y)))
        fit_residual = max_abs_error / scale if scale > 0.0 else 0.0
    else:
        max_abs_error = rmse = fit_residual = float("nan")
        scaled_condition = float("nan")

    H = np.asarray(
        [[beta[2], beta[4] / 2.0], [beta[4] / 2.0, beta[3]]],
        dtype=float,
    )
    h_eigenvalues = (
        np.linalg.eigvalsh(H) if np.all(np.isfinite(H)) else np.full(2, np.nan)
    )
    result = {name: float(beta[i]) for i, name in enumerate(COEFF_NAMES)}
    result.update(
        {
            "fit_residual": float(fit_residual),
            "fit_rmse_mw": float(rmse),
            "fit_max_abs_error_mw": float(max_abs_error),
            "fit_rank": int(rank),
            "fit_rank_full": bool(rank == 5),
            "fit_scaled_condition": float(scaled_condition),
            "h_cost_min_eig": float(h_eigenvalues[0]),
            "h_cost_max_eig": float(h_eigenvalues[1]),
            "a_P_negative": bool(np.isfinite(beta[0]) and beta[0] < 0.0),
            "a_Q_negative": bool(np.isfinite(beta[1]) and beta[1] < 0.0),
            "h_cost_psd": bool(
                np.isfinite(h_eigenvalues[0]) and h_eigenvalues[0] >= -PSD_TOL
            ),
            "sample_count": len(samples),
            "converged_sample_count": len(converged),
        }
    )
    return result


def _enforce_strict(result: dict) -> None:
    assert result["a_P"] < 0.0, f"a_P>=0 measured: {result['a_P']}"
    assert result["a_Q"] < 0.0, f"a_Q>=0 measured: {result['a_Q']}"
    assert result["h_cost_min_eig"] >= -PSD_TOL, (
        f"H_cost non-PSD: lambda_min={result['h_cost_min_eig']}"
    )


def _measure_coeffs_on_net(
    net,
    base_p: np.ndarray,
    base_q: np.ndarray,
    bus: int,
    S: float,
    scenario: str,
    t: int,
    p_center: float,
    *,
    strict: bool,
) -> dict:
    bus, S, scenario, t, p_center = _validate_inputs(
        net, bus, S, scenario, t, p_center
    )
    probe_idx = _ensure_probe_sgen(net)
    stats = _RunStats()
    samples: list[dict] = []
    tolerance_mva = _pf_tolerance_mva(S)

    try:
        baseline_loss = _measure_loss(
            net,
            base_p,
            base_q,
            probe_idx,
            bus,
            scenario,
            t,
            0.0,
            0.0,
            stats,
            tolerance_mva,
        )
        samples.append(
            {
                "p_mw": 0.0,
                "q_mvar": 0.0,
                "loss_mw": baseline_loss,
                "delta_loss_reduction_mw": 0.0 if np.isfinite(baseline_loss) else np.nan,
                "L_cost_mw": 0.0 if np.isfinite(baseline_loss) else np.nan,
                "is_baseline": True,
            }
        )

        for p_mw, q_mvar in _local_grid(S, p_center):
            loss_mw = _measure_loss(
                net,
                base_p,
                base_q,
                probe_idx,
                bus,
                scenario,
                t,
                p_mw,
                q_mvar,
                stats,
                tolerance_mva,
            )
            if np.isfinite(baseline_loss) and np.isfinite(loss_mw):
                delta_loss = baseline_loss - loss_mw
                l_cost = -delta_loss
            else:
                delta_loss = l_cost = float("nan")
            samples.append(
                {
                    "p_mw": p_mw,
                    "q_mvar": q_mvar,
                    "loss_mw": loss_mw,
                    "delta_loss_reduction_mw": delta_loss,
                    "L_cost_mw": l_cost,
                    "is_baseline": False,
                }
            )

        result = _fit_samples(samples)
        result.update(
            {
                "bus": bus,
                "S": S,
                "scenario": scenario,
                "t": t,
                "p_center": p_center,
                "baseline_loss_mw": float(baseline_loss),
                "pf_tolerance_mva": float(tolerance_mva),
                "runpp_attempts": stats.attempts,
                "runpp_retries": stats.retries,
                "runpp_failures": stats.failures,
            }
        )
        if strict:
            _enforce_strict(result)
        return result
    finally:
        net.load["p_mw"] = base_p
        net.load["q_mvar"] = base_q
        net.sgen.at[probe_idx, "p_mw"] = 0.0
        net.sgen.at[probe_idx, "q_mvar"] = 0.0


def measure_coeffs(
    bus,
    S,
    scenario,
    t,
    p_center=0.0,
    *,
    net=None,
    strict=True,
):
    """Measure one five-coefficient loss surface.

    Returns scalar ``a_P``, ``a_Q``, ``b_PP``, ``b_QQ``, ``b_PQ`` together
    with fit residual/rank, Hessian eigenvalues, diagnostic flags, samples,
    and runpp counters.  Coefficients produce ``L_cost`` in MW for P in MW
    and Q in Mvar.
    """
    measurement_net = build_net() if net is None else net
    base_p = measurement_net.load["p_mw"].to_numpy(dtype=float).copy()
    base_q = measurement_net.load["q_mvar"].to_numpy(dtype=float).copy()
    return _measure_coeffs_on_net(
        measurement_net,
        base_p,
        base_q,
        bus,
        S,
        scenario,
        t,
        p_center,
        strict=bool(strict),
    )


def get_or_measure(
    bus,
    S,
    scenario,
    t,
    p_center=0.0,
    *,
    net=None,
    strict=True,
):
    """Return cached coefficients or measure them on a cache miss."""
    global _cache_hits, _cache_misses, _measure_wall_s
    key = _cache_key_coeffs(bus, S, scenario, t, p_center)
    cached = _measured_cache.get(key)
    if cached is not None:
        _cache_hits += 1
        result = _copy_result(cached)
        if strict:
            _enforce_strict(result)
        return result

    _cache_misses += 1
    started = time.perf_counter()
    try:
        result = measure_coeffs(
            bus,
            S,
            scenario,
            t,
            p_center,
            net=net,
            strict=strict,
        )
    finally:
        _measure_wall_s += time.perf_counter() - started
    _measured_cache[key] = _copy_result(result)
    return _copy_result(result)


def measure_coeffs_grid(
    bus,
    S,
    scenarios: Iterable[str],
    times: Iterable[int],
    p_center=0.0,
    *,
    strict=True,
):
    """Measure a scenario/time grid while reusing one pandapower network.

    Every coefficient and scalar diagnostic is returned as an array with
    shape ``(n_scenario, n_time)``.  The five coefficient arrays can be
    sliced by scenario and passed to the assembly layer in ``evaluate.py``.
    """
    scenarios = tuple(str(scenario) for scenario in scenarios)
    times = tuple(int(t) for t in times)
    if not scenarios:
        raise ValueError("scenarios must not be empty")
    if not times:
        raise ValueError("times must not be empty")

    net = build_net()
    shape = (len(scenarios), len(times))
    array_names = COEFF_NAMES + (
        "fit_residual",
        "fit_rmse_mw",
        "fit_max_abs_error_mw",
        "fit_scaled_condition",
        "h_cost_min_eig",
        "h_cost_max_eig",
    )
    output = {name: np.full(shape, np.nan, dtype=float) for name in array_names}
    output["fit_rank"] = np.zeros(shape, dtype=int)
    for flag in (
        "fit_rank_full",
        "a_P_negative",
        "a_Q_negative",
        "h_cost_psd",
    ):
        output[flag] = np.zeros(shape, dtype=bool)

    for scenario_index, scenario in enumerate(scenarios):
        for time_index, t in enumerate(times):
            result = get_or_measure(
                bus,
                S,
                scenario,
                t,
                p_center,
                net=net,
                strict=strict,
            )
            for name in array_names:
                output[name][scenario_index, time_index] = result[name]
            output["fit_rank"][scenario_index, time_index] = result["fit_rank"]
            for flag in (
                "fit_rank_full",
                "a_P_negative",
                "a_Q_negative",
                "h_cost_psd",
            ):
                output[flag][scenario_index, time_index] = result[flag]

    output["scenarios"] = scenarios
    output["times"] = np.asarray(times, dtype=int)
    output["bus"] = int(bus)
    output["S"] = float(S)
    output["p_center"] = float(p_center)
    output["pf_tolerance_mva"] = float(_pf_tolerance_mva(float(S)))
    return output


def measure_loss_reduction(
    bus,
    scenario,
    t,
    p_mw,
    q_mvar,
    *,
    net=None,
):
    """Directly measure one point against the P=Q=0 baseline."""
    measurement_net = build_net() if net is None else net
    base_p = measurement_net.load["p_mw"].to_numpy(dtype=float).copy()
    base_q = measurement_net.load["q_mvar"].to_numpy(dtype=float).copy()
    bus = int(bus)
    scenario = str(scenario)
    t = int(t)
    p_mw = float(p_mw)
    q_mvar = float(q_mvar)
    point_scale_mva = max(float(np.hypot(p_mw, q_mvar)), GRID_TOL)
    tolerance_mva = _pf_tolerance_mva(point_scale_mva)
    _validate_inputs(
        measurement_net,
        bus,
        point_scale_mva,
        scenario,
        t,
        0.0,
    )
    probe_idx = _ensure_probe_sgen(measurement_net)
    stats = _RunStats()
    try:
        baseline_loss = _measure_loss(
            measurement_net,
            base_p,
            base_q,
            probe_idx,
            bus,
            scenario,
            t,
            0.0,
            0.0,
            stats,
            tolerance_mva,
        )
        loss_mw = _measure_loss(
            measurement_net,
            base_p,
            base_q,
            probe_idx,
            bus,
            scenario,
            t,
            p_mw,
            q_mvar,
            stats,
            tolerance_mva,
        )
        delta_loss = baseline_loss - loss_mw
        return {
            "baseline_loss_mw": float(baseline_loss),
            "loss_mw": float(loss_mw),
            "delta_loss_reduction_mw": float(delta_loss),
            "L_cost_mw": float(-delta_loss),
            "pf_tolerance_mva": float(tolerance_mva),
            "runpp_attempts": stats.attempts,
            "runpp_retries": stats.retries,
            "runpp_failures": stats.failures,
        }
    finally:
        measurement_net.load["p_mw"] = base_p
        measurement_net.load["q_mvar"] = base_q
        measurement_net.sgen.at[probe_idx, "p_mw"] = 0.0
        measurement_net.sgen.at[probe_idx, "q_mvar"] = 0.0
