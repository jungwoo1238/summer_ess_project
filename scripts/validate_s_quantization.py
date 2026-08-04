"""User-run validation for S-grid coefficient caching and bounded LRU caches.

This script performs many AC power-flow calls.  Codex only writes/compiles it;
run it directly in the configured ``ess`` environment.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

import evaluate
import loss_coeffs
import lower_lp as LP
import params as PM
from build_net import build_net


BUS = 15
SCENARIO = "summer"
COEFF_NAMES = loss_coeffs.COEFF_NAMES
GRID_CANDIDATES = (0.001, 0.0001)
S_VALUES = (0.05003, 0.17603, 0.50003, 1.00003, 2.00003)
PARTICLE = np.asarray([15.0, 0.17603, 0.42], dtype=float)


def _stage(number: int, title: str, criterion: str) -> None:
    print(f"\n[stage] {number}. {title}", flush=True)
    print(f"criterion={criterion}", flush=True)


def _relative_error(reference: float, candidate: float) -> float:
    return abs(float(candidate) - float(reference)) / max(abs(float(reference)), 1e-12)


def _max_schedule_difference(reference: dict, candidate: dict, field: str) -> float:
    return max(
        float(
            np.max(
                np.abs(
                    np.asarray(reference[field][scenario], dtype=float)
                    - np.asarray(candidate[field][scenario], dtype=float)
                )
            )
        )
        for scenario in PM.ALL_DAYS
    )


def _dummy_coeffs(index: int) -> dict:
    shape = (1, PM.TIME_STEPS)
    curvature = 0.02 + 1e-5 * int(index)
    return {
        "a_P": np.full(shape, -0.01 - 1e-6 * int(index)),
        "a_Q": np.full(shape, -0.01),
        "b_PP": np.full(shape, curvature),
        "b_QQ": np.full(shape, curvature),
        "b_PQ": np.zeros(shape),
    }


def main() -> int:
    started = time.perf_counter()
    peak_t = int(np.argmax(np.asarray(PM.LOAD[SCENARIO], dtype=float)))

    _stage(
        1,
        "coefficient approximation error",
        "report all five relative errors; no automatic grid selection",
    )
    measurement_net = build_net()
    for S in S_VALUES:
        raw = loss_coeffs.measure_coeffs(
            BUS, S, SCENARIO, peak_t, p_center=0.0,
            net=measurement_net, strict=True,
        )
        for grid in GRID_CANDIDATES:
            S_grid = loss_coeffs._quantize_s_for_cache(S, grid=grid)
            quantized = loss_coeffs.measure_coeffs(
                BUS, S_grid, SCENARIO, peak_t, p_center=0.0,
                net=measurement_net, strict=True,
            )
            errors = {
                name: _relative_error(raw[name], quantized[name])
                for name in COEFF_NAMES
            }
            print(
                f"S_requested={S:.8f} grid={grid:.4g} S_grid={S_grid:.8f} "
                + " ".join(
                    f"relerr_{name}={errors[name]:.12g}"
                    for name in COEFF_NAMES
                ),
                flush=True,
            )

    _stage(
        2,
        "j_net and schedule impact",
        "candidate signal: |delta_j_net|<100 won; adoption gate: <1000 won",
    )
    original_grid = float(PM.S_CACHE_GRID_MVA)
    variants = (("none", 0.0), ("grid_0p001", 0.001), ("grid_0p0001", 0.0001))
    details = {}
    try:
        evaluate.init_worker()
        for label, grid in variants:
            PM.S_CACHE_GRID_MVA = float(grid)
            loss_coeffs.clear_cache()
            LP._PROBLEM_CACHE.clear()
            detail = evaluate.evaluate_particle(PARTICLE, return_detail=True)
            if detail.get("diverged"):
                raise RuntimeError(f"{label} evaluation diverged: {detail}")
            details[label] = detail
            info = loss_coeffs.cache_info()
            print(
                f"variant={label} grid={grid:.4g} j_net={float(detail['j_net']):.12g} "
                f"cache_size={info['size']} hits={info['hits']} misses={info['misses']}",
                flush=True,
            )
    finally:
        PM.S_CACHE_GRID_MVA = original_grid
        loss_coeffs.clear_cache()
        LP._PROBLEM_CACHE.clear()

    reference = details["none"]
    for label in ("grid_0p001", "grid_0p0001"):
        delta_j = float(details[label]["j_net"] - reference["j_net"])
        max_p = _max_schedule_difference(reference, details[label], "unit_p")
        max_q = _max_schedule_difference(reference, details[label], "unit_q")
        print(
            f"comparison={label}_minus_none delta_j_net_won={delta_j:.12g} "
            f"max_abs_delta_P_mw={max_p:.12g} max_abs_delta_Q_mvar={max_q:.12g} "
            f"abs_delta_lt_100={abs(delta_j) < 100.0} "
            f"abs_delta_lt_1000={abs(delta_j) < 1000.0}",
            flush=True,
        )

    _stage(
        3,
        "predicted cache-key reduction",
        "smaller unique-key count means more adjacent-S coefficient reuse",
    )
    nearby_s = np.arange(0.1755, 0.17751, 0.0001)
    raw_unique = len({float(S) for S in nearby_s})
    print(f"sample_count={len(nearby_s)} unique_keys_none={raw_unique}", flush=True)
    for grid in GRID_CANDIDATES:
        keys = {
            loss_coeffs._quantize_s_for_cache(float(S), grid=grid)
            for S in nearby_s
        }
        print(
            f"grid={grid:.4g} unique_keys={len(keys)} "
            f"reduction_fraction={1.0 - len(keys) / raw_unique:.12g}",
            flush=True,
        )

    _stage(
        4,
        "LRU memory bound",
        "cache size must never exceed the configured temporary maxsize=3",
    )
    production_maxsize = int(LP._PROBLEM_CACHE_MAXSIZE)
    temporary_maxsize = 3
    LP._PROBLEM_CACHE.clear()
    LP._PROBLEM_CACHE_MAXSIZE = temporary_maxsize
    try:
        sizes = []
        for index in range(5):
            normalized = LP._normalize_coeffs(_dummy_coeffs(index), 1, PM.TIME_STEPS)
            LP._get_problem(
                "avg",
                1,
                False,
                PM.ETA_C,
                PM.ETA_D,
                PM.SELF_DISCHARGE_HOURLY,
                PM.SOC_INIT_FRAC,
                PM.SOC_MIN_FRAC,
                PM.SOC_MAX_FRAC,
                PM.MU_VOLT,
                normalized,
            )
            sizes.append(len(LP._PROBLEM_CACHE))
            assert sizes[-1] <= temporary_maxsize, sizes
        print(f"problem_cache_sizes={sizes}", flush=True)
        print(f"problem_cache_bound_ok={max(sizes) <= temporary_maxsize}", flush=True)
        print(f"production_problem_cache_maxsize={production_maxsize}", flush=True)
        print(
            f"measured_cache_maxsize={loss_coeffs._MEASURED_CACHE_MAXSIZE}",
            flush=True,
        )
    finally:
        LP._PROBLEM_CACHE.clear()
        LP._PROBLEM_CACHE_MAXSIZE = production_maxsize

    print(f"total_runtime_s={time.perf_counter() - started:.6f}", flush=True)
    print("VALIDATION_COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
