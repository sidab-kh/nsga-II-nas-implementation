from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class EnhancedFireflyAlgorithm(MetaheuristicOptimizer):
    """Exploration and exploitation enhanced firefly algorithm."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
        alpha: float = 0.5,
        gamma: float = 1.0,
        beta_0: float = 0.2,
        sigma_0: float = 100.0,
        random_replace_frac: float = 0.2,
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
        self.alpha = alpha
        self.gamma = gamma
        self.beta_0 = beta_0
        self.sigma_0 = sigma_0
        self.random_replace_frac = random_replace_frac

    def _attractiveness(self, distance: float) -> float:
        return self.beta_0 / (1.0 + self.gamma * distance**2)

    def _update_population(self, iteration: int) -> None:
        order = np.argsort(self.fitness_values)[::-1]
        self.population = self.population[order]
        self.fitness_values = self.fitness_values[order]

        best_pos = self.population[0].astype(float)
        sigma_t = self.sigma_0 * (1.0 - iteration / self.max_iterations)
        new_pop = self.population.copy()

        for i in range(self.population_size):
            for j in range(i):
                if self.fitness_values[j] > self.fitness_values[i]:
                    pos_i = self.population[i].astype(float)
                    pos_j = self.population[j].astype(float)
                    dist = float(np.linalg.norm(pos_i - pos_j, ord=2))
                    attract = self._attractiveness(dist)

                    if self._rng.random() > 0.5:
                        new_row = best_pos + self._rng.standard_normal(self.dim) * sigma_t
                    else:
                        delta = attract * (pos_j - pos_i)
                        noise = self.alpha * (self._rng.random(self.dim) - 0.5) * max(1.0, self.search_space_size / 2)
                        new_row = pos_i + delta + noise

                    new_pop[i] = self._clip_row(new_row)

        self.population = new_pop

        num_replace = max(1, int(self.population_size * self.random_replace_frac))
        for i in range(self.population_size - num_replace, self.population_size):
            self.population[i] = self._rng.integers(0, self.search_space_size, size=self.dim)
