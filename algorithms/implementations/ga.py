from __future__ import annotations

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from core.fitness import HardwareAwareFitness


class GeneticAlgorithm(MetaheuristicOptimizer):
    """Discrete GA for either index- or vector-based NAS search spaces."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_size: int = 20,
        max_iterations: int = 100,
        crossover_prob: float = 0.8,
        mutation_prob: float = 0.1,
        mutation_sigma: float = 200.0,
        tournament_size: int = 3,
        elitism: int = 2,
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
        self.mutation_sigma = mutation_sigma
        self.tournament_size = tournament_size
        self.elitism = elitism

    def _tournament_select(self) -> int:
        contestants = self._rng.integers(0, self.population_size, size=self.tournament_size)
        return int(contestants[np.argmax(self.fitness_values[contestants])])

    def _update_population(self, iteration: int) -> None:
        new_pop = np.empty_like(self.population)

        elite_order = np.argsort(self.fitness_values)[::-1]
        for e in range(self.elitism):
            new_pop[e] = self.population[elite_order[e]]

        for i in range(self.elitism, self.population_size):
            p1 = self._tournament_select()
            p2 = self._tournament_select()

            if self.dim == 1:
                if self._rng.random() < self.crossover_prob:
                    alpha = self._rng.random()
                    child_idx = int(
                        alpha * self.population[p1, 0]
                        + (1 - alpha) * self.population[p2, 0]
                    )
                else:
                    child_idx = int(self.population[p1, 0])

                if self._rng.random() < self.mutation_prob:
                    child_idx += int(self._rng.standard_normal() * self.mutation_sigma)

                new_pop[i, 0] = self._clip(child_idx)
            else:
                parent1 = self.population[p1]
                parent2 = self.population[p2]

                if self._rng.random() < self.crossover_prob:
                    mask = self._rng.random(self.dim) < 0.5
                    child = np.where(mask, parent1, parent2)
                else:
                    child = parent1.copy()

                mut_mask = self._rng.random(self.dim) < self.mutation_prob
                if np.any(mut_mask):
                    child[mut_mask] = self._rng.integers(
                        0, self.search_space_size, size=int(mut_mask.sum())
                    )

                new_pop[i] = self._clip_row(child)

        self.population = new_pop
