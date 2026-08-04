# Compare PSO, GWO, ABC, and L-SHADE under one candidate-evaluation budget.
# Uses only synthetic twin-valley, Rastrigin, and Rosenbrock objectives.

import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from pso_core import PSO as CorePSO


POP = 32
EVAL_BUDGET = 6000
N_RUNS = 30
SEEDS = tuple(range(N_RUNS))


def twin_valley(X):
    X = np.asarray(X, dtype=float)
    c_inf = np.array([-2.0, 0.0])
    c_glob = np.array([2.0, 0.0])
    r2_inf = np.sum((X - c_inf) ** 2, axis=1)
    r2_glob = np.sum((X - c_glob) ** 2, axis=1)
    well_inf = -0.96 * np.exp(-r2_inf / (2.0 * 1.2 ** 2))
    well_glob = -1.00 * np.exp(-r2_glob / (2.0 * 0.5 ** 2))
    return np.minimum(well_inf, well_glob) + 0.01 * np.sum(X ** 2, axis=1)


def rastrigin(X):
    X = np.asarray(X, dtype=float)
    return 20.0 + np.sum(X ** 2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)


def rosenbrock(X):
    X = np.asarray(X, dtype=float)
    return np.sum(
        100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2
        + (1.0 - X[:, :-1]) ** 2,
        axis=1,
    )


class _OptimizerBase:
    def __init__(
        self,
        objective,
        bounds,
        n_particles=POP,
        n_iters=100,
        int_dims=None,
        seed=None,
        eval_budget=EVAL_BUDGET,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_dims = self.bounds.shape[0]
        self.n_particles = int(n_particles)
        self.n_iters = int(n_iters)
        self.int_dims = list(int_dims) if int_dims else []
        self.rng = np.random.default_rng(seed)
        self.eval_budget = int(eval_budget)
        self.eval_count = 0
        self.lo = self.bounds[:, 0]
        self.hi = self.bounds[:, 1]

    def _decode(self, X):
        if not self.int_dims:
            return X
        decoded = np.asarray(X, dtype=float).copy()
        decoded[:, self.int_dims] = np.round(decoded[:, self.int_dims])
        return decoded

    def _can_evaluate(self, n_candidates):
        return self.eval_count + int(n_candidates) <= self.eval_budget

    def _evaluate(self, X):
        X = np.asarray(X, dtype=float)
        if not self._can_evaluate(X.shape[0]):
            raise RuntimeError("candidate evaluation budget exceeded")
        values = np.asarray(self.objective(self._decode(X)), dtype=float)
        if values.shape != (X.shape[0],):
            raise ValueError(
                f"objective shape {values.shape} != ({X.shape[0]},)"
            )
        self.eval_count += X.shape[0]
        return values

    def _initial_population(self):
        return self.rng.uniform(
            self.lo, self.hi, size=(self.n_particles, self.n_dims)
        )

    def _result(self, x, f, history):
        return {
            "x": self._decode(np.asarray(x)[None, :])[0],
            "f": float(f),
            "history": np.asarray(history, dtype=float),
            "eval_count": int(self.eval_count),
        }


class PSOBench:
    """Budget-limited adapter around the project's unchanged pso_core.PSO."""

    def __init__(
        self,
        objective,
        bounds,
        n_particles=POP,
        n_iters=100,
        int_dims=None,
        seed=None,
        eval_budget=EVAL_BUDGET,
    ):
        self.objective = objective
        self.bounds = np.asarray(bounds, dtype=float)
        self.n_particles = int(n_particles)
        self.n_iters = int(n_iters)
        self.int_dims = list(int_dims) if int_dims else []
        self.seed = seed
        self.eval_budget = int(eval_budget)
        self.eval_count = 0

    def optimize(self):
        max_iters_by_budget = (
            self.eval_budget - self.n_particles
        ) // self.n_particles
        n_iters = min(self.n_iters, max_iters_by_budget)

        def counted_objective(X):
            X = np.asarray(X, dtype=float)
            if self.eval_count + X.shape[0] > self.eval_budget:
                raise RuntimeError("candidate evaluation budget exceeded")
            values = np.asarray(self.objective(X), dtype=float)
            self.eval_count += X.shape[0]
            return values

        optimizer = CorePSO(
            objective=counted_objective,
            bounds=self.bounds,
            n_particles=self.n_particles,
            n_iters=n_iters,
            w_max=0.9,
            w_min=0.4,
            c1=2.0,
            c2=2.0,
            v_clamp_k=0.2,
            int_dims=self.int_dims,
            seed=self.seed,
        )
        result = optimizer.optimize()
        result["eval_count"] = int(self.eval_count)
        return result


class GWOBench(_OptimizerBase):
    def optimize(self):
        X = self._initial_population()
        fitness = self._evaluate(X)
        best_idx = int(np.argmin(fitness))
        best_x = X[best_idx].copy()
        best_f = float(fitness[best_idx])
        history = [best_f]
        max_iters_by_budget = (
            self.eval_budget - self.n_particles
        ) // self.n_particles
        max_iter = min(self.n_iters, max_iters_by_budget)

        for t in range(max_iter):
            if not self._can_evaluate(self.n_particles):
                break
            leaders = X[np.argsort(fitness)[:3]]
            a = 2.0 - 2.0 * (t / max(max_iter, 1))
            proposals = []
            for leader in leaders:
                r1 = self.rng.random(X.shape)
                r2 = self.rng.random(X.shape)
                A = 2.0 * a * r1 - a
                C = 2.0 * r2
                distance = np.abs(C * leader[None, :] - X)
                proposals.append(leader[None, :] - A * distance)
            X = np.clip(np.mean(proposals, axis=0), self.lo, self.hi)
            fitness = self._evaluate(X)
            idx = int(np.argmin(fitness))
            if fitness[idx] < best_f:
                best_f = float(fitness[idx])
                best_x = X[idx].copy()
            history.append(best_f)
        return self._result(best_x, best_f, history)


class ABCBench(_OptimizerBase):
    def _neighbor(self, X, source_index):
        n = X.shape[0]
        k = int(self.rng.integers(n - 1))
        if k >= source_index:
            k += 1
        dimension = int(self.rng.integers(self.n_dims))
        phi = self.rng.uniform(-1.0, 1.0)
        candidate = X[source_index].copy()
        candidate[dimension] += phi * (
            X[source_index, dimension] - X[k, dimension]
        )
        return np.clip(candidate, self.lo, self.hi)

    @staticmethod
    def _roulette_quality(fitness):
        return np.where(
            fitness >= 0.0,
            1.0 / (1.0 + fitness),
            1.0 + np.abs(fitness),
        )

    def optimize(self):
        X = self._initial_population()
        fitness = self._evaluate(X)
        trials = np.zeros(self.n_particles, dtype=int)
        limit = self.n_particles * self.n_dims
        idx = int(np.argmin(fitness))
        best_x = X[idx].copy()
        best_f = float(fitness[idx])
        history = [best_f]

        for _ in range(self.n_iters):
            batch_size = 2 * self.n_particles
            if not self._can_evaluate(batch_size):
                break

            scout_mask = trials >= limit
            employed = np.empty_like(X)
            for i in range(self.n_particles):
                if scout_mask[i]:
                    employed[i] = self.rng.uniform(self.lo, self.hi)
                else:
                    employed[i] = self._neighbor(X, i)

            quality = self._roulette_quality(fitness)
            probabilities = quality / np.sum(quality)
            sources = self.rng.choice(
                self.n_particles,
                size=self.n_particles,
                replace=True,
                p=probabilities,
            )
            onlookers = np.vstack([
                self._neighbor(X, int(source)) for source in sources
            ])
            candidate_f = self._evaluate(np.vstack([employed, onlookers]))
            employed_f = candidate_f[:self.n_particles]
            onlooker_f = candidate_f[self.n_particles:]

            for i in range(self.n_particles):
                if scout_mask[i]:
                    X[i] = employed[i]
                    fitness[i] = employed_f[i]
                    trials[i] = 0
                elif employed_f[i] < fitness[i]:
                    X[i] = employed[i]
                    fitness[i] = employed_f[i]
                    trials[i] = 0
                else:
                    trials[i] += 1

            for candidate, value, source in zip(
                onlookers, onlooker_f, sources
            ):
                source = int(source)
                if value < fitness[source]:
                    X[source] = candidate
                    fitness[source] = value
                    trials[source] = 0
                else:
                    trials[source] += 1

            idx = int(np.argmin(fitness))
            if fitness[idx] < best_f:
                best_f = float(fitness[idx])
                best_x = X[idx].copy()
            history.append(best_f)
        return self._result(best_x, best_f, history)


class LSHADEBench(_OptimizerBase):
    P_BEST_RATE = 0.11
    MEMORY_SIZE = 6
    MIN_POPULATION = 4

    def _sample_f(self, location):
        while True:
            value = location + 0.1 * np.tan(
                np.pi * (self.rng.random() - 0.5)
            )
            if value > 0.0:
                return min(float(value), 1.0)

    def optimize(self):
        X = self._initial_population()
        fitness = self._evaluate(X)
        initial_population = self.n_particles
        archive = []
        memory_f = np.full(self.MEMORY_SIZE, 0.5)
        memory_cr = np.full(self.MEMORY_SIZE, 0.5)
        memory_index = 0
        idx = int(np.argmin(fitness))
        best_x = X[idx].copy()
        best_f = float(fitness[idx])
        history = [best_f]

        for _ in range(self.n_iters):
            population = X.shape[0]
            if not self._can_evaluate(population):
                break
            order = np.argsort(fitness)
            pbest_count = max(
                2, int(np.ceil(self.P_BEST_RATE * population))
            )
            trials = np.empty_like(X)
            sampled_f = np.empty(population)
            sampled_cr = np.empty(population)

            for i in range(population):
                memory_slot = int(self.rng.integers(self.MEMORY_SIZE))
                F = self._sample_f(memory_f[memory_slot])
                CR = float(np.clip(
                    self.rng.normal(memory_cr[memory_slot], 0.1), 0.0, 1.0
                ))
                sampled_f[i] = F
                sampled_cr[i] = CR

                pbest_idx = int(self.rng.choice(order[:pbest_count]))
                r1_options = [j for j in range(population) if j != i]
                r1 = int(self.rng.choice(r1_options))
                pool = [
                    X[j]
                    for j in range(population)
                    if j != i and j != r1
                ] + archive
                r2_vector = np.asarray(pool[int(self.rng.integers(len(pool)))])
                mutant = (
                    X[i]
                    + F * (X[pbest_idx] - X[i])
                    + F * (X[r1] - r2_vector)
                )
                mutant = np.clip(mutant, self.lo, self.hi)
                crossover = self.rng.random(self.n_dims) <= CR
                crossover[int(self.rng.integers(self.n_dims))] = True
                trials[i] = np.where(crossover, mutant, X[i])

            trial_fitness = self._evaluate(trials)
            success = trial_fitness < fitness
            success_indices = np.flatnonzero(success)
            improvements = fitness[success] - trial_fitness[success]
            old_parents = X[success].copy()
            for parent in old_parents:
                archive.append(parent)
            X[success] = trials[success]
            fitness[success] = trial_fitness[success]

            if success_indices.size and float(np.sum(improvements)) > 0.0:
                weights = improvements / np.sum(improvements)
                f_success = sampled_f[success]
                cr_success = sampled_cr[success]
                memory_f[memory_index] = np.sum(
                    weights * f_success ** 2
                ) / np.sum(weights * f_success)
                memory_cr[memory_index] = np.sum(weights * cr_success)
                memory_index = (memory_index + 1) % self.MEMORY_SIZE

            target_population = int(np.round(
                initial_population
                - (initial_population - self.MIN_POPULATION)
                * self.eval_count / self.eval_budget
            ))
            target_population = max(
                self.MIN_POPULATION, min(population, target_population)
            )
            if target_population < population:
                keep = np.argsort(fitness)[:target_population]
                X = X[keep]
                fitness = fitness[keep]
            if len(archive) > X.shape[0]:
                keep_archive = self.rng.choice(
                    len(archive), size=X.shape[0], replace=False
                )
                archive = [archive[int(j)] for j in keep_archive]

            idx = int(np.argmin(fitness))
            if fitness[idx] < best_f:
                best_f = float(fitness[idx])
                best_x = X[idx].copy()
            history.append(best_f)
        return self._result(best_x, best_f, history)


ALGORITHMS = {
    "PSO": (PSOBench, (EVAL_BUDGET - POP) // POP),
    "GWO": (GWOBench, (EVAL_BUDGET - POP) // POP),
    "ABC": (ABCBench, (EVAL_BUDGET - POP) // (2 * POP)),
    "L-SHADE": (LSHADEBench, EVAL_BUDGET),
}


BENCHMARKS = (
    (
        "BENCH_TWIN_VALLEY",
        twin_valley,
        np.array([[-5.0, 5.0], [-5.0, 5.0]]),
        lambda result: np.linalg.norm(
            np.asarray(result["x"]) - np.array([2.0, 0.0])
        ) < 0.5,
    ),
    (
        "BENCH_RASTRIGIN",
        rastrigin,
        np.array([[-5.12, 5.12], [-5.12, 5.12]]),
        lambda result: float(result["f"]) < 1.0,
    ),
    (
        "BENCH_ROSENBROCK",
        rosenbrock,
        np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        lambda result: float(result["f"]) < 1e-2,
    ),
)


def _run_one(optimizer_class, n_iters, objective, bounds, seed):
    optimizer = optimizer_class(
        objective=objective,
        bounds=bounds,
        n_particles=POP,
        n_iters=n_iters,
        int_dims=[],
        seed=seed,
        eval_budget=EVAL_BUDGET,
    )
    started = time.perf_counter()
    result = optimizer.optimize()
    result["wall_time_s"] = time.perf_counter() - started
    return result


def main():
    for section, objective, bounds, success_test in BENCHMARKS:
        print(f"[{section}]")
        print(
            "algo,n_runs,global_hit_rate_pct,f_median,f_best,f_worst,"
            "eval_count_median,wall_time_median_s"
        )
        for algorithm, (optimizer_class, n_iters) in ALGORITHMS.items():
            results = [
                _run_one(
                    optimizer_class, n_iters, objective, bounds, seed
                )
                for seed in SEEDS
            ]
            final_f = np.array([result["f"] for result in results])
            eval_counts = np.array([
                result["eval_count"] for result in results
            ])
            wall_times = np.array([
                result["wall_time_s"] for result in results
            ])
            hit_rate = 100.0 * np.mean([
                bool(success_test(result)) for result in results
            ])
            print(
                f"{algorithm},{N_RUNS:d},{hit_rate:.6g},"
                f"{np.median(final_f):.6g},{np.min(final_f):.6g},"
                f"{np.max(final_f):.6g},{np.median(eval_counts):.6g},"
                f"{np.median(wall_times):.6g}"
            )


if __name__ == "__main__":
    main()
