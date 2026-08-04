# Benchmark PSO/ABC/L-SHADE on the real ESS objective with equal evaluation budgets.
# Eight runs per algorithm are expected to take about eight hours at ~1200 s/run.

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
import time
import multiprocessing as mp
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np

import evaluate
import main as MAIN
from probe_optimizer_bench import ABCBench, LSHADEBench, PSOBench


POP = 32
EVAL_BUDGET = POP * 31
N_RUNS = 8
N_WORKERS = MAIN.DEFAULT_N_WORKERS
SEEDS = tuple(range(N_RUNS))

ALGORITHMS = (
    ("PSO", PSOBench, 30),
    ("ABC", ABCBench, 30),
    ("L-SHADE", LSHADEBench, EVAL_BUDGET),
)


def make_objective(pool):
    def objective(X):
        particles = [np.asarray(row, dtype=float) for row in X]
        results = pool.map(MAIN._eval_for_pso, particles, chunksize=1)
        return np.asarray([result[0] for result in results], dtype=float)

    return objective


def _evaluate_final_detail(x):
    detail = evaluate.evaluate_particle(
        np.asarray(x, dtype=float), return_detail=True
    )
    return dict(detail)


def _run_algorithm(algorithm, optimizer_class, n_iters, bounds, int_dims):
    rows = []
    pool = mp.Pool(N_WORKERS, initializer=evaluate.init_worker)
    try:
        objective = make_objective(pool)
        for run, seed in enumerate(SEEDS):
            optimizer = optimizer_class(
                objective=objective,
                bounds=bounds,
                n_particles=POP,
                n_iters=n_iters,
                int_dims=int_dims,
                seed=seed,
                eval_budget=EVAL_BUDGET,
            )
            started = time.perf_counter()
            result = optimizer.optimize()
            wall_s = time.perf_counter() - started
            x = np.asarray(result["x"], dtype=float)
            detail = _evaluate_final_detail(x)
            bus = int(round(float(x[0])))
            diverged = bool(detail.get("diverged", False))
            rows.append({
                "algo": algorithm,
                "run": int(run),
                "seed": int(seed),
                "bus": bus,
                "S": float(x[1]),
                "E": float(x[2]),
                "j_net": (
                    float("nan") if diverged else float(detail["j_net"])
                ),
                "b_defer": (
                    float("nan") if diverged else float(detail["b_defer"])
                ),
                "b_arb": (
                    float("nan") if diverged else float(detail["b_arb"])
                ),
                "b_loss": (
                    float("nan") if diverged else float(detail["b_loss"])
                ),
                "eval_count": int(result["eval_count"]),
                "wall_s": float(wall_s),
            })
    finally:
        pool.close()
        pool.join()
    return rows


def _is_global_basin(row):
    return (
        row["bus"] in {31, 32}
        or (
            np.isfinite(row["j_net"])
            and row["j_net"] > 3.9e6
        )
    )


def _format_bus_distribution(rows):
    counts = Counter(int(row["bus"]) for row in rows)
    return ",".join(
        f"{bus}:{counts[bus]}" for bus in sorted(counts)
    )


def _print_summary(all_rows):
    print("[REAL_BENCH_SUMMARY]")
    print(
        "algo,n_runs,global_basin_rate_pct,j_net_median,j_net_best,"
        "j_net_worst,eval_count_median,wall_time_median_s"
    )
    for algorithm, _optimizer_class, _n_iters in ALGORITHMS:
        rows = [row for row in all_rows if row["algo"] == algorithm]
        j_net = np.asarray([row["j_net"] for row in rows], dtype=float)
        eval_count = np.asarray([
            row["eval_count"] for row in rows
        ], dtype=float)
        wall_s = np.asarray([row["wall_s"] for row in rows], dtype=float)
        global_rate = 100.0 * np.mean([
            _is_global_basin(row) for row in rows
        ])
        finite_j = j_net[np.isfinite(j_net)]
        if finite_j.size:
            median_j = float(np.median(finite_j))
            best_j = float(np.max(finite_j))
            worst_j = float(np.min(finite_j))
        else:
            median_j = best_j = worst_j = float("nan")
        print(
            f"{algorithm},{len(rows):d},{global_rate:.6g},"
            f"{median_j:.6g},{best_j:.6g},{worst_j:.6g},"
            f"{np.median(eval_count):.6g},{np.median(wall_s):.6g}"
        )


def _print_bus_distribution(all_rows):
    print("[REAL_BENCH_BUS_DIST]")
    print("algo,bus_counts")
    for algorithm, _optimizer_class, _n_iters in ALGORITHMS:
        rows = [row for row in all_rows if row["algo"] == algorithm]
        print(f"{algorithm},{_format_bus_distribution(rows)}")


def _print_per_run(all_rows):
    print("[REAL_BENCH_PER_RUN]")
    print(
        "algo,run,seed,bus,S,E,j_net,b_defer,b_arb,b_loss,"
        "eval_count,wall_s"
    )
    for row in all_rows:
        print(
            f"{row['algo']},{row['run']:d},{row['seed']:d},"
            f"{row['bus']:d},{row['S']:.6g},{row['E']:.6g},"
            f"{row['j_net']:.6g},{row['b_defer']:.6g},"
            f"{row['b_arb']:.6g},{row['b_loss']:.6g},"
            f"{row['eval_count']:d},{row['wall_s']:.6g}"
        )


def main():
    bounds = MAIN._build_bounds(1)
    int_dims = MAIN._build_int_dims(1)
    all_rows = []
    for algorithm, optimizer_class, n_iters in ALGORITHMS:
        all_rows.extend(_run_algorithm(
            algorithm, optimizer_class, n_iters, bounds, int_dims
        ))
    _print_summary(all_rows)
    _print_bus_distribution(all_rows)
    _print_per_run(all_rows)


if __name__ == "__main__":
    mp.freeze_support()
    main()
