from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class HarmonySearch(MetaheuristicOptimizer):
    """Harmony Search adapted for discrete NAS."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
        HMCR: float = 0.9,
        PAR: float = 0.3,
        bw: float = 100.0,
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
        self.HMCR = HMCR
        self.PAR = PAR
        self.bw = bw

    def _update_population(self, iteration: int) -> None:
        new_row = np.empty(self.dim, dtype=int)
        for d in range(self.dim):
            if self._rng.random() < self.HMCR:
                value = int(self._rng.choice(self.population[:, d]))
                if self._rng.random() < self.PAR:
                    value = self._clip(value + int(self._rng.uniform(-self.bw, self.bw)))
            else:
                value = int(self._rng.integers(0, self.search_space_size))
            new_row[d] = value

        new_fitness = self.fitness_function.compute(self._row_to_architecture(new_row))

        worst_pos = int(np.argmin(self.fitness_values))
        if new_fitness > self.fitness_values[worst_pos]:
            self.population[worst_pos] = new_row
            self.fitness_values[worst_pos] = new_fitness
