from __future__ import annotations

import math

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class WhaleOptimizationAlgorithm(MetaheuristicOptimizer):
    """Whale Optimization Algorithm adapted for discrete NAS."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
        b: float = 1.0,
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
        self.b = b

    def _update_population(self, iteration: int) -> None:
        order = np.argsort(self.fitness_values)[::-1]
        best_pos = self.population[order[0]].astype(float)
        a = 2.0 * (1.0 - iteration / self.max_iterations)

        for i in range(self.population_size):
            A = 2 * a * self._rng.random(self.dim) - a
            C = 2 * self._rng.random(self.dim)
            l = self._rng.uniform(-1, 1)
            p = self._rng.random()
            x = self.population[i].astype(float)

            if p < 0.5:
                if np.all(np.abs(A) < 1):
                    d = np.abs(C * best_pos - x)
                    new_row = best_pos - A * d
                else:
                    rand_pos = self._rng.integers(0, self.search_space_size, size=self.dim).astype(float)
                    d = np.abs(C * rand_pos - x)
                    new_row = rand_pos - A * d
            else:
                d = np.abs(best_pos - x)
                new_row = d * math.exp(self.b * l) * np.cos(2 * math.pi * l) + best_pos

            self.population[i] = self._clip_row(new_row)
