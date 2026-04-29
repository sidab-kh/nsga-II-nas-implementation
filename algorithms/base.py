"""
algorithms/base.py
==================
Abstract base class for all metaheuristic optimizers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from core.fitness import HardwareAwareFitness
from core.types import NASBench201Architecture, RunResult


class MetaheuristicOptimizer(ABC):
    """Abstract base class for metaheuristic NAS optimizers."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        search_space_size: int = 15625,
        dim: int = 1,
        population_size: int = 20,
        max_iterations: int = 100,
        seed: int = 42,
    ) -> None:
        self.fitness_function = fitness_function
        self.search_space_size = search_space_size
        self.dim = dim
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        np.random.seed(seed)

        self.population: np.ndarray = np.empty(0)
        self.fitness_values: np.ndarray = np.empty(0)
        self.best_solution: Optional[np.ndarray] = None
        self.best_fitness = -np.inf
        self.fitness_history: List[float] = []
        self._convergence_iter: Optional[int] = None

    @abstractmethod
    def _update_population(self, iteration: int) -> None:
        """Update ``self.population`` based on algorithm-specific logic."""

    def _on_init(self) -> None:
        """Called once after the initial population is created and evaluated."""

    def _on_iter_end(self, iteration: int) -> None:
        """Called at the end of each iteration after evaluation."""

    def _initialize_population(self) -> np.ndarray:
        return self._rng.integers(
            0, self.search_space_size, size=(self.population_size, self.dim)
        )

    def _row_to_architecture(self, row: np.ndarray) -> NASBench201Architecture:
        return NASBench201Architecture.from_vector(
            row,
            search_space=self.fitness_function.search_space,
            search_space_size=self.search_space_size,
        )

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        fitness = np.empty(len(population))
        for i, row in enumerate(population):
            arch = self._row_to_architecture(row)
            fitness[i] = self.fitness_function.compute(arch)
        return fitness

    def _clip(self, value: int | float) -> int:
        return int(np.clip(value, 0, self.search_space_size - 1))

    def _clip_row(self, row: np.ndarray) -> np.ndarray:
        return np.clip(np.round(row).astype(int), 0, self.search_space_size - 1)

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        self.population = self._initialize_population()
        self.fitness_values = self._evaluate_population(self.population)
        self.fitness_history = []

        best_idx = int(np.argmax(self.fitness_values))
        self.best_solution = self.population[best_idx].copy()
        self.best_fitness = float(self.fitness_values[best_idx])
        self.fitness_history.append(self.best_fitness)
        self._convergence_iter = 0

        self._on_init()

        for iteration in range(self.max_iterations):
            self._update_population(iteration)
            self.fitness_values = self._evaluate_population(self.population)

            best_idx = int(np.argmax(self.fitness_values))
            if float(self.fitness_values[best_idx]) > self.best_fitness:
                self.best_fitness = float(self.fitness_values[best_idx])
                self.best_solution = self.population[best_idx].copy()
                self._convergence_iter = iteration

            self.fitness_history.append(self.best_fitness)
            self._on_iter_end(iteration)

        return self.best_solution, self.best_fitness, self.fitness_history

    def run(self, run_id: int = 0) -> RunResult:
        best_sol, best_fit, history = self.optimize()

        arch = self._row_to_architecture(best_sol)
        metrics = self.fitness_function.get_hardware_metrics(arch)
        stats = self.fitness_function.get_statistics()

        best_arch_idx = arch.arch_idx if arch.arch_idx is not None else -1
        best_arch_vector = (
            arch.to_vector().astype(int).tolist() if arch.encoding is not None else None
        )

        return RunResult(
            run_id=run_id,
            algorithm_name=self.__class__.__name__,
            best_arch_idx=best_arch_idx,
            best_arch_vector=best_arch_vector,
            best_fitness=best_fit,
            fitness_history=history,
            hardware_metrics=metrics,
            query_count=stats["api_queries"],
            convergence_iter=self._convergence_iter,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"pop={self.population_size}, "
            f"iters={self.max_iterations}, "
            f"seed={self.seed})"
        )
