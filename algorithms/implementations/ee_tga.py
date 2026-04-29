from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class EnhancedTreeGrowthAlgorithm(MetaheuristicOptimizer):
    """Exploitation-Enhanced Tree Growth Algorithm (EE-TGA)."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
        theta: float = 0.2,
        lambda_param: float = 0.5,
        sigma_0: float = 100.0,
        exploit_fraction: float = 0.4,
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
        self.theta = theta
        self.lambda_param = lambda_param
        self.sigma_0 = sigma_0
        self.exploit_fraction = exploit_fraction

        n = population_size
        self.n1 = max(1, n // 3)
        self.n2 = max(1, n // 3)
        self.n3 = n - self.n1 - self.n2

    def _update_population(self, iteration: int) -> None:
        order = np.argsort(self.fitness_values)[::-1]
        self.population = self.population[order]
        self.fitness_values = self.fitness_values[order]

        best = self.population[0].astype(float)
        second = self.population[min(1, len(self.population) - 1)].astype(float)
        sigma_t = self.sigma_0 * (1.0 - iteration / self.max_iterations)

        n1_exploit = max(1, int(self.n1 * self.exploit_fraction))
        for i in range(n1_exploit):
            new_row = best + self._rng.standard_normal(self.dim) * sigma_t
            self.population[i] = self._clip_row(new_row)

        for i in range(n1_exploit, self.n1):
            r = self._rng.random(self.dim)
            current = self.population[i].astype(float)
            new_row = current / self.theta + r * current
            self.population[i] = self._clip_row(new_row)

        for i in range(self.n1, self.n1 + self.n2):
            y = self.lambda_param * best + (1 - self.lambda_param) * second
            alpha = self._rng.random(self.dim)
            current = self.population[i].astype(float)
            new_row = current + alpha * y
            self.population[i] = self._clip_row(new_row)

        for i in range(self.n1 + self.n2, self.population_size):
            self.population[i] = self._rng.integers(0, self.search_space_size, size=self.dim)
