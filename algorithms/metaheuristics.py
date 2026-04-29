"""
algorithms/metaheuristics.py
============================
Registry and public entrypoint for the available metaheuristic algorithms.

Individual implementations live in ``algorithms/implementations``.
"""

from __future__ import annotations

from typing import Dict, Type

from algorithms.base import MetaheuristicOptimizer
from algorithms.implementations import (
    AntColonyOptimization,
    DifferentialEvolution,
    EnhancedFireflyAlgorithm,
    EnhancedTreeGrowthAlgorithm,
    GeneticAlgorithm,
    GrayWolfOptimizer,
    HarmonySearch,
    NSGA2,
    NSGA2Partitioned,
    NSGANet,
    NSGANetVNS,
    ParticleSwarmOptimization,
    SimulatedAnnealing,
    WhaleOptimizationAlgorithm,
)


REGISTRY: Dict[str, Type[MetaheuristicOptimizer]] = {
    "EE-TGA": EnhancedTreeGrowthAlgorithm,
    "E3-FA": EnhancedFireflyAlgorithm,
    "PSO": ParticleSwarmOptimization,
    "DE": DifferentialEvolution,
    "GA": GeneticAlgorithm,
    "SA": SimulatedAnnealing,
    "ACO": AntColonyOptimization,
    "HS": HarmonySearch,
    "NSGA2": NSGA2,
    "NSGA2-PARTITIONED": NSGA2Partitioned,
    "NSGA-NET": NSGANet,
    "NSGA-NET-VNS": NSGANetVNS,
    "GWO": GrayWolfOptimizer,
    "WOA": WhaleOptimizationAlgorithm,
}


def get_algorithm(name: str) -> Type[MetaheuristicOptimizer]:
    """Look up an algorithm class by name from the registry."""
    if name not in REGISTRY:
        raise KeyError(
            f"Algorithm '{name}' not found. Available: {sorted(REGISTRY.keys())}"
        )
    return REGISTRY[name]


__all__ = [
    "REGISTRY",
    "get_algorithm",
    "AntColonyOptimization",
    "DifferentialEvolution",
    "EnhancedFireflyAlgorithm",
    "EnhancedTreeGrowthAlgorithm",
    "GeneticAlgorithm",
    "GrayWolfOptimizer",
    "HarmonySearch",
    "NSGA2",
    "NSGA2Partitioned",
    "NSGANet",
    "NSGANetVNS",
    "ParticleSwarmOptimization",
    "SimulatedAnnealing",
    "WhaleOptimizationAlgorithm",
]


