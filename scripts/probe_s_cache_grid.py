# Measure expected cache hits by S-grid/LRU size and single-optimum j_net sensitivity.
# PART1 is a no-power-flow PSO trace simulation; PART2 performs the real evaluation.

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
from collections import OrderedDict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import evaluate
import loss_coeffs
import main as MAIN
import params as PM
from pso_core import PSO


GRID_CANDIDATES = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05)
MAXSIZE_CANDIDATES = (8192, 32768, 65536)
X_OPT = np.array(
    [32.0, 0.21358313449208763, 0.4407054133130149], dtype=float
)

N_SCENARIO = 5
N_TIME = 24
N_MIN7 = 7
ENTRIES_PER_BS = N_SCENARIO * N_TIME
DEV_RUNPP_REFERENCE = 754560
DEV_HIT_RATE_REFERENCE_PCT = 5.8


def _reproduce_dev_trace():
    profile = dict(MAIN.PROFILES["dev"])
    bounds = MAIN._build_bounds(1)
    int_dims = MAIN._build_int_dims(1)
    base_seq = np.random.SeedSequence(
        entropy=MAIN.DEFAULT_BASE_SEED, spawn_key=(1,)
    )
    run_seed_seqs = base_seq.spawn(profile["n_runs"])

    trace_parts = []
    for run_seed in run_seed_seqs:
        def objective(X):
            trace_parts.append(np.asarray(X, dtype=float).copy())
            return np.zeros(X.shape[0], dtype=float)

        pso = PSO(
            objective=objective,
            bounds=bounds,
            n_particles=profile["n_particles"],
            n_iters=profile["n_iters"],
            w_max=PM.PSO_W_MAX,
            w_min=PM.PSO_W_MIN,
            c1=PM.PSO_C1,
            c2=PM.PSO_C2,
            v_clamp_k=PM.PSO_V_MAX_RATIO,
            int_dims=int_dims,
            seed=run_seed,
        )
        pso.optimize()

    trace = np.concatenate(trace_parts, axis=0)
    return trace, run_seed_seqs


def _quantized_keys(trace, grid):
    # evaluate._solve_unit_schedules returns zero schedules before coefficient
    # measurement when either rating is inactive.  Such particles generate no
    # cache lookup and must not be passed to the positive-S quantizer.
    active = (
        (trace[:, 1] >= evaluate.S_ACTIVE_MIN)
        & (trace[:, 2] >= evaluate.E_ACTIVE_MIN)
    )
    active_trace = trace[active]
    buses = np.rint(active_trace[:, 0]).astype(int)
    quantized_s = np.array([
        loss_coeffs._quantize_s_for_cache(float(S), grid=float(grid))
        for S in active_trace[:, 1]
    ])
    return list(zip(buses.tolist(), quantized_s.tolist())), int(active.sum())


def _simulate_lru(keys, capacity):
    cache = OrderedDict()
    hits = 0
    misses = 0
    for key in keys:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            misses += 1
            cache[key] = None
            if len(cache) > capacity:
                cache.popitem(last=False)
    return hits, misses


def part1_cache_simulation():
    trace, _run_seed_seqs = _reproduce_dev_trace()
    n_eval_total = int(trace.shape[0])
    unlimited = {}

    print("[PART1_UNBOUNDED]")
    print(
        "grid_MVA,n_eval_total,n_eval_cache_active,unique_bS,"
        "expected_hit_rate_pct,"
        "expected_runpp_total,reduction_vs_754560_pct"
    )
    for grid in GRID_CANDIDATES:
        keys, n_eval_active = _quantized_keys(trace, grid)
        unique_bs = len(set(keys))
        hit_rate = (
            1.0 - unique_bs / n_eval_active
            if n_eval_active > 0
            else float("nan")
        )
        expected_runpp = unique_bs * N_SCENARIO * N_TIME * N_MIN7
        reduction = 1.0 - expected_runpp / DEV_RUNPP_REFERENCE
        unlimited[grid] = (keys, hit_rate)
        print(
            f"{grid:.6g},{n_eval_total:d},{n_eval_active:d},{unique_bs:d},"
            f"{100.0 * hit_rate:.6g},"
            f"{expected_runpp:d},{100.0 * reduction:.6g}"
        )

    print("[PART1_LRU]")
    print(
        "grid_MVA,maxsize,maxsize_eff_bS,expected_hit_rate_lru_pct,"
        "hit_rate_lost_to_eviction_pct"
    )
    for grid in GRID_CANDIDATES:
        keys, unlimited_hit_rate = unlimited[grid]
        for maxsize in MAXSIZE_CANDIDATES:
            capacity = int(maxsize) // ENTRIES_PER_BS
            hits, misses = _simulate_lru(keys, capacity)
            lru_hit_rate = hits / (hits + misses)
            eviction_loss = unlimited_hit_rate - lru_hit_rate
            print(
                f"{grid:.6g},{maxsize:d},{capacity:d},"
                f"{100.0 * lru_hit_rate:.6g},"
                f"{100.0 * eviction_loss:.6g}"
            )

    print("[PART1_REFERENCE]")
    print("grid_MVA,maxsize,n_workers,measured_dev_hit_rate_pct")
    print(f"0.001,8192,8,{DEV_HIT_RATE_REFERENCE_PCT:.6g}")


def part2_accuracy_sensitivity():
    evaluate.init_worker()
    original_grid = float(PM.S_CACHE_GRID_MVA)
    rows = []
    try:
        for grid in GRID_CANDIDATES:
            PM.S_CACHE_GRID_MVA = float(grid)
            loss_coeffs.clear_cache()
            detail = evaluate.evaluate_particle(X_OPT, return_detail=True)
            quant_s = loss_coeffs._quantize_s_for_cache(
                float(X_OPT[1]), grid=float(grid)
            )
            rows.append((grid, quant_s, detail))
    finally:
        PM.S_CACHE_GRID_MVA = original_grid
        loss_coeffs.clear_cache()

    baseline = rows[0][2]
    baseline_j_net = float(baseline["j_net"])

    print("[PART2]")
    print(
        "grid_MVA,quant_S_measured_MVA,j_net_won,b_energy_won,"
        "b_defer_won,b_arb_won,b_loss_won,dj_net_vs_baseline_won,"
        "dj_net_vs_baseline_pct"
    )
    for grid, quant_s, detail in rows:
        j_net = float(detail["j_net"])
        delta = j_net - baseline_j_net
        delta_pct = (
            100.0 * delta / baseline_j_net
            if abs(baseline_j_net) > 0.0
            else float("nan")
        )
        print(
            f"{grid:.6g},{quant_s:.6g},{j_net:.6g},"
            f"{float(detail['b_energy']):.6g},"
            f"{float(detail['b_defer']):.6g},"
            f"{float(detail['b_arb']):.6g},"
            f"{float(detail['b_loss']):.6g},{delta:.0f},{delta_pct:.6g}"
        )


def main():
    part1_cache_simulation()
    part2_accuracy_sensitivity()


if __name__ == "__main__":
    main()
