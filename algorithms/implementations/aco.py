from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class AntColonyOptimization(MetaheuristicOptimizer):
    """ACO with a pheromone trail over sampled architectures."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
        n_candidates: int = 200,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        Q: float = 1.0,
        refresh_interval: int = 25,
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
        self.n_candidates = min(n_candidates, max(self.population_size, 1) * 10)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.refresh_interval = refresh_interval

    def _on_init(self) -> None:
        self._refresh_candidates()

    def _refresh_candidates(self) -> None:
        self.candidates = self._rng.integers(
            0, self.search_space_size, size=(self.n_candidates, self.dim)
        )
        self.pheromones = np.ones(self.n_candidates, dtype=float)

    def _update_population(self, iteration: int) -> None:
        if iteration % self.refresh_interval == 0 and iteration > 0:
            top_k = min(self.population_size, self.n_candidates)
            top_indices = np.argsort(self.fitness_values)[::-1][:top_k]
            top_archs = self.population[top_indices].copy()
            self._refresh_candidates()
            self.candidates[:top_k] = top_archs

        heuristic = 1.0 / (np.abs(np.arange(self.n_candidates) - self.n_candidates // 2) + 1)
        prob = (self.pheromones**self.alpha) * (heuristic**self.beta)
        prob /= prob.sum()

        chosen = self._rng.choice(self.n_candidates, size=self.population_size, p=prob)
        self.population = self.candidates[chosen].copy()

        self.pheromones *= 1 - self.rho

        new_fit = self._evaluate_population(self.population)
        baseline = float(min(self.fitness_values.min(), new_fit.min()))
        for c, f in zip(chosen, new_fit):
            if f > -np.inf:
                self.pheromones[c] += self.Q * (f - baseline + 1e-9)

        self.fitness_values = new_fit
        self.pheromones = np.clip(self.pheromones, 1e-6, None)
