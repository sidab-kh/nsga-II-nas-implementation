from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class ParticleSwarmOptimization(MetaheuristicOptimizer):
    """Standard PSO adapted for discrete NAS search."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
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
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _on_init(self) -> None:
        n = self.population_size
        self.velocities = self._rng.uniform(-100, 100, size=(n, self.dim))
        self.personal_best = self.population.copy().astype(float)
        self.personal_best_fitness = self.fitness_values.copy()

        gb_idx = int(np.argmax(self.fitness_values))
        self.global_best = self.population[gb_idx].copy().astype(float)
        self.global_best_fitness = float(self.fitness_values[gb_idx])

    def _update_population(self, iteration: int) -> None:
        r1 = self._rng.random((self.population_size, self.dim))
        r2 = self._rng.random((self.population_size, self.dim))

        self.velocities = (
            self.w * self.velocities
            + self.c1 * r1 * (self.personal_best - self.population)
            + self.c2 * r2 * (self.global_best - self.population)
        )

        self.population = self._clip_row(self.population + self.velocities)

        new_fit = self._evaluate_population(self.population)
        improved = new_fit > self.personal_best_fitness
        self.personal_best[improved] = self.population[improved].astype(float)
        self.personal_best_fitness[improved] = new_fit[improved]

        gb_idx = int(np.argmax(self.personal_best_fitness))
        if self.personal_best_fitness[gb_idx] > self.global_best_fitness:
            self.global_best = self.personal_best[gb_idx].copy()
            self.global_best_fitness = float(self.personal_best_fitness[gb_idx])

        self.fitness_values = new_fit
