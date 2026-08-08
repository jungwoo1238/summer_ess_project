"""Budget-limited L-SHADE optimizer extracted from the validated benchmark."""

import numpy as np


POP = 32
EVAL_BUDGET = 6000


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


class LSHADEBench(_OptimizerBase):
    P_BEST_RATE = 0.11
    MEMORY_SIZE = 6
    MIN_POPULATION = 12

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
