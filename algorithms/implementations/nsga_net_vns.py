from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from algorithms.multiobjective import (
    crowding_distance,
    environmental_selection,
    extract_nondominated,
    fast_non_dominated_sort,
)
from core.fitness import HardwareAwareFitness
from core.types import RunResult


class NSGANetVNS(MetaheuristicOptimizer):
    """
    NSGA-NET + VNS hybrid for discrete NAS search.

    The algorithm optimizes three minimization objectives derived from
    HardwareAwareFitness.compute_multi():
    - normalized latency
    - normalized energy
    - normalized error (1 - normalized accuracy)

    A VNS phase is applied to a subset of elite individuals each iteration
    to improve local exploitation while NSGA-II keeps global diversity.
    """

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 32,
        max_iterations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.12,
        mutation_rate_dim: float = 0.25,
        tournament_size: int = 2,
        vns_elite_fraction: float = 0.25,
        vns_k_max: int = 3,
        vns_trials_per_k: int = 2,
        archive_max_size: int = 256,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            fitness_function,
            population_size=population_size,
            max_iterations=max_iterations,
            seed=seed,
            **kwargs,
        )
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.mutation_rate_dim = mutation_rate_dim
        self.tournament_size = max(2, tournament_size)
        self.vns_elite_fraction = float(np.clip(vns_elite_fraction, 0.0, 1.0))
        self.vns_k_max = max(1, vns_k_max)
        self.vns_trials_per_k = max(1, vns_trials_per_k)
        self.archive_max_size = max(population_size, archive_max_size)

        self.objectives = np.empty((0, 3), dtype=float)
        self.front_ranks = np.empty(0, dtype=int)
        self.crowding = np.empty(0, dtype=float)
        self.archive_population = np.empty((0, self.dim), dtype=int)
        self.archive_objectives = np.empty((0, 3), dtype=float)
        self.pareto_history: list[int] = []
        self._last_front_size = 0

    def _evaluate_objectives(self, population: np.ndarray) -> np.ndarray:
        objective_matrix = np.empty((len(population), 3), dtype=float)
        for i, row in enumerate(population):
            arch = self._row_to_architecture(row)
            objective_matrix[i] = self.fitness_function.compute_multi(arch)
        return objective_matrix

    def _assign_rank_and_crowding(self, objectives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ranks = np.full(len(objectives), fill_value=np.iinfo(np.int32).max, dtype=int)
        crowd = np.zeros(len(objectives), dtype=float)
        fronts = fast_non_dominated_sort(objectives)

        for rank_idx, front in enumerate(fronts):
            ranks[front] = rank_idx
            front_crowding = crowding_distance(objectives[front])
            crowd[front] = front_crowding
        return ranks, crowd

    def _binary_tournament(self) -> int:
        contenders = self._rng.integers(0, self.population_size, size=self.tournament_size)
        best = int(contenders[0])
        for idx in contenders[1:]:
            idx = int(idx)
            if self.front_ranks[idx] < self.front_ranks[best]:
                best = idx
            elif self.front_ranks[idx] == self.front_ranks[best] and self.crowding[idx] > self.crowding[best]:
                best = idx
        return best

    def _crossover(self, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        if self._rng.random() >= self.crossover_prob:
            return p1.copy()

        if self.dim == 1:
            return np.array([int(self._rng.choice([p1[0], p2[0]]))], dtype=int)

        mask = self._rng.random(self.dim) < 0.5
        child = np.where(mask, p1, p2)
        return child.astype(int)

    def _mutate(self, child: np.ndarray, strength: int = 1) -> np.ndarray:
        if self._rng.random() >= self.mutation_prob:
            return child

        if self.dim == 1:
            step = int(self._rng.integers(-strength, strength + 1))
            if step == 0:
                step = 1
            child[0] = self._clip(child[0] + step)
            return child

        p = min(1.0, self.mutation_rate_dim * max(1, strength))
        mask = self._rng.random(self.dim) < p
        if not np.any(mask):
            mask[self._rng.integers(0, self.dim)] = True
        child[mask] = self._rng.integers(0, self.search_space_size, size=int(mask.sum()))
        return self._clip_row(child)

    def _dominates(self, lhs: np.ndarray, rhs: np.ndarray) -> bool:
        return bool(np.all(lhs <= rhs) and np.any(lhs < rhs))

    def _vns_local_search(self, start: np.ndarray) -> np.ndarray:
        best = start.copy()
        best_obj = self.fitness_function.compute_multi(self._row_to_architecture(best))

        k = 1
        while k <= self.vns_k_max:
            improved = False
            for _ in range(self.vns_trials_per_k):
                candidate = best.copy()
                candidate = self._mutate(candidate, strength=k)
                cand_obj = self.fitness_function.compute_multi(self._row_to_architecture(candidate))

                if self._dominates(cand_obj, best_obj):
                    best = candidate
                    best_obj = cand_obj
                    improved = True
                    k = 1
                    break

            if not improved:
                k += 1

        return best

    def _update_archive(self, candidates_pop: np.ndarray, candidates_obj: np.ndarray) -> None:
        if len(self.archive_population) == 0:
            merged_pop = candidates_pop
            merged_obj = candidates_obj
        else:
            merged_pop = np.vstack([self.archive_population, candidates_pop])
            merged_obj = np.vstack([self.archive_objectives, candidates_obj])

        nd_pop, nd_obj = extract_nondominated(merged_pop, merged_obj)
        if len(nd_pop) > self.archive_max_size:
            nd_pop, nd_obj = environmental_selection(nd_pop, nd_obj, self.archive_max_size)

        self.archive_population = nd_pop
        self.archive_objectives = nd_obj

    def _on_init(self) -> None:
        self.objectives = self._evaluate_objectives(self.population)
        self.front_ranks, self.crowding = self._assign_rank_and_crowding(self.objectives)
        self._update_archive(self.population, self.objectives)
        self._last_front_size = int(len(fast_non_dominated_sort(self.objectives)[0]))

    def _on_iter_end(self, iteration: int) -> None:
        self.objectives = self._evaluate_objectives(self.population)
        self.front_ranks, self.crowding = self._assign_rank_and_crowding(self.objectives)
        self._update_archive(self.population, self.objectives)

        fronts = fast_non_dominated_sort(self.objectives)
        self._last_front_size = int(len(fronts[0])) if fronts else 0
        self.pareto_history.append(self._last_front_size)

    def _select_scalar_best(self) -> tuple[np.ndarray, float]:
        scores = np.empty(len(self.population), dtype=float)
        for i, row in enumerate(self.population):
            scores[i] = self.fitness_function.compute(self._row_to_architecture(row))
        idx = int(np.argmax(scores))
        return self.population[idx].copy(), float(scores[idx])

    def _update_population(self, iteration: int) -> None:
        offspring = np.empty_like(self.population)
        for i in range(self.population_size):
            p1 = self.population[self._binary_tournament()]
            p2 = self.population[self._binary_tournament()]
            child = self._crossover(p1, p2)
            child = self._mutate(child)
            offspring[i] = self._clip_row(child)

        elite_count = max(1, int(self.population_size * self.vns_elite_fraction))
        fronts = fast_non_dominated_sort(self.objectives)
        elite_front = fronts[0] if fronts else np.array([], dtype=int)
        if len(elite_front) > 0:
            if len(elite_front) > elite_count:
                elite_front = self._rng.choice(elite_front, size=elite_count, replace=False)
            for idx in elite_front:
                offspring[int(idx)] = self._vns_local_search(offspring[int(idx)])

        off_obj = self._evaluate_objectives(offspring)
        combined_pop = np.vstack([self.population, offspring])
        combined_obj = np.vstack([self.objectives, off_obj])

        self.population, self.objectives = environmental_selection(
            combined_pop,
            combined_obj,
            self.population_size,
        )
        self.front_ranks, self.crowding = self._assign_rank_and_crowding(self.objectives)

        scalar_scores = np.empty(self.population_size, dtype=float)
        for i, row in enumerate(self.population):
            scalar_scores[i] = self.fitness_function.compute(self._row_to_architecture(row))
        self.fitness_values = scalar_scores

    def optimize(self) -> tuple[np.ndarray, float, list[float]]:
        self.population = self._initialize_population()
        self.fitness_values = self._evaluate_population(self.population)
        self.fitness_history = []

        self._on_init()

        best_sol, best_fit = self._select_scalar_best()
        self.best_solution = best_sol
        self.best_fitness = best_fit
        self.fitness_history.append(best_fit)
        self._convergence_iter = 0

        for iteration in range(self.max_iterations):
            self._update_population(iteration)

            current_sol, current_fit = self._select_scalar_best()
            if current_fit > self.best_fitness:
                self.best_solution = current_sol
                self.best_fitness = current_fit
                self._convergence_iter = iteration

            self.fitness_history.append(self.best_fitness)
            self._on_iter_end(iteration)

        return self.best_solution.copy(), float(self.best_fitness), self.fitness_history

    def run(self, run_id: int = 0) -> RunResult:
        best_sol, best_fit, history = self.optimize()

        arch = self._row_to_architecture(best_sol)
        metrics = self.fitness_function.get_hardware_metrics(arch)
        stats = self.fitness_function.get_statistics()

        best_arch_idx = arch.arch_idx if arch.arch_idx is not None else -1
        best_arch_vector = (
            arch.to_vector().astype(int).tolist() if arch.encoding is not None else None
        )

        result = RunResult(
            run_id=run_id,
            algorithm_name=self.__class__.__name__,
            best_arch_idx=best_arch_idx,
            best_arch_vector=best_arch_vector,
            best_fitness=best_fit,
            fitness_history=history,
            hardware_metrics=metrics,
            query_count=stats["api_queries"],
            convergence_iter=self._convergence_iter,
        )
        return result
