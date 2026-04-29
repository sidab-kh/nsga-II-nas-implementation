from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class DifferentialEvolution(MetaheuristicOptimizer):
    """DE/rand/1/bin adapted for discrete NAS search."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
        F: float = 0.8,
        CR: float = 0.9,
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
        self.F = F
        self.CR = CR

    def _update_population(self, iteration: int) -> None:
        pop_f = self.population.astype(float)

        for i in range(self.population_size):
            candidates = [j for j in range(self.population_size) if j != i]
            a, b, c = self._rng.choice(candidates, size=3, replace=False)

            mutant = pop_f[a] + self.F * (pop_f[b] - pop_f[c])

            mask = self._rng.random(self.dim) < self.CR
            if not mask.any():
                mask[self._rng.integers(self.dim)] = True
            trial = np.where(mask, mutant, pop_f[i])

            trial_int = self._clip_row(trial)
            trial_fit = self.fitness_function.compute(self._row_to_architecture(trial_int))

            if trial_fit > self.fitness_values[i]:
                self.population[i] = trial_int
                self.fitness_values[i] = trial_fit
                pop_f[i] = trial_int.astype(float)
