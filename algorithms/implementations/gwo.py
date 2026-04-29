from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class GrayWolfOptimizer(MetaheuristicOptimizer):
    """Grey Wolf Optimizer adapted for discrete NAS."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
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

    def _update_population(self, iteration: int) -> None:
        order = np.argsort(self.fitness_values)[::-1]
        alpha_pos = self.population[order[0]].astype(float)
        beta_pos = self.population[order[min(1, len(order) - 1)]].astype(float)
        delta_pos = self.population[order[min(2, len(order) - 1)]].astype(float)

        a = 2.0 * (1.0 - iteration / self.max_iterations)

        for i in range(self.population_size):
            x = self.population[i].astype(float)
            r1 = self._rng.random(self.dim)
            r2 = self._rng.random(self.dim)
            a1, c1 = 2 * a * r1 - a, 2 * r2
            d_alpha = np.abs(c1 * alpha_pos - x)
            x1 = alpha_pos - a1 * d_alpha

            r1 = self._rng.random(self.dim)
            r2 = self._rng.random(self.dim)
            a2, c2 = 2 * a * r1 - a, 2 * r2
            d_beta = np.abs(c2 * beta_pos - x)
            x2 = beta_pos - a2 * d_beta

            r1 = self._rng.random(self.dim)
            r2 = self._rng.random(self.dim)
            a3, c3 = 2 * a * r1 - a, 2 * r2
            d_delta = np.abs(c3 * delta_pos - x)
            x3 = delta_pos - a3 * d_delta

            self.population[i] = self._clip_row((x1 + x2 + x3) / 3.0)
