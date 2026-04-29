from __future__ import annotations

from core.fitness import HardwareAwareFitness

from .nsga_net_vns import NSGANetVNS


class NSGA2(NSGANetVNS):
    """Standalone NSGA-II baseline (without VNS local search)."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        **kwargs,
    ) -> None:
        kwargs.setdefault("vns_elite_fraction", 0.0)
        kwargs.setdefault("vns_k_max", 1)
        kwargs.setdefault("vns_trials_per_k", 1)
        super().__init__(fitness_function, **kwargs)
