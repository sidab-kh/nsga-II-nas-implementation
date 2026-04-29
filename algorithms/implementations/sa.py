from __future__ import annotations

import math

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class SimulatedAnnealing(MetaheuristicOptimizer):
    """Simulated Annealing with Gaussian neighborhood moves."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 1,
        max_iterations: int = 500,
        T_init: float = 1.0,
        T_min: float = 1e-4,
        cooling: float = 0.995,
        neighbour_sigma: float = 200.0,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(
            fitness_function,
            population_size=1,
            max_iterations=max_iterations,
            seed=seed,
            **kwargs,
        )
        self.T = T_init
        self.T_min = T_min
        self.cooling = cooling
        self.neighbour_sigma = neighbour_sigma

    def _on_init(self) -> None:
        self.current_solution = self.population[0].copy()
        self.current_fitness = float(self.fitness_values[0])

    def _update_population(self, iteration: int) -> None:
        candidate = self.current_solution.astype(float) + self._rng.standard_normal(self.dim) * self.neighbour_sigma
        candidate = self._clip_row(candidate)
        candidate_fitness = self.fitness_function.compute(self._row_to_architecture(candidate))

        delta_f = candidate_fitness - self.current_fitness
        if delta_f > 0 or self._rng.random() < math.exp(delta_f / max(self.T, 1e-10)):
            self.current_solution = candidate
            self.current_fitness = candidate_fitness

        self.T = max(self.T * self.cooling, self.T_min)
        self.population[0] = self.current_solution
        self.fitness_values[0] = self.current_fitness
