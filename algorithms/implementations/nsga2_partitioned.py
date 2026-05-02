from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from algorithms.base import MetaheuristicOptimizer
from algorithms.multiobjective import (
    environmental_selection,
    extract_nondominated,
    fast_non_dominated_sort,
)
from core.fitness import HardwareAwareFitness
from core.types import RunResult, SearchSpace

from .nsga_net_vns import NSGANetVNS


@dataclass(frozen=True)
class NicheDefinition:
    """Complexity-driven niche definition for NAS-Bench-201 architectures."""

    niche_id: int

    def contains(self, *, conv3x3: int, conv1x1: int) -> bool:
        if self.niche_id == 0:
            return conv3x3 == 0 and conv1x1 == 0
        if self.niche_id == 1:
            return conv3x3 == 0 and conv1x1 >= 1
        if self.niche_id == 2:
            return conv3x3 == 1
        if self.niche_id == 3:
            return conv3x3 == 2
        if self.niche_id == 4:
            return conv3x3 == 3
        if self.niche_id == 5:
            return conv3x3 >= 4
        raise ValueError(f"Invalid niche_id={self.niche_id}")


def _count_nb201_convs_from_arch_str(arch_str: str) -> Tuple[int, int]:
    """Return (#nor_conv_3x3, #nor_conv_1x1) from a NAS-Bench-201 arch_str."""

    # HW-NAS-Bench stores NAS-Bench-201 configs with an `arch_str` like:
    #   |avg_pool_3x3~0|+|nor_conv_1x1~0|skip_connect~1|+|nor_conv_3x3~0|...
    # We only count the *normal convolution* ops, as used by the niche table.
    conv3 = int(arch_str.count("nor_conv_3x3"))
    conv1 = int(arch_str.count("nor_conv_1x1"))
    return conv3, conv1


class _NicheConstrainedNSGA2(NSGANetVNS):
    """NSGA-II variant constrained to sample/mutate inside one niche."""

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        niche: NicheDefinition,
        niche_pool: np.ndarray,
        max_resample_attempts: int = 5000,
        **kwargs,
    ) -> None:
        kwargs.setdefault("vns_elite_fraction", 0.0)
        kwargs.setdefault("vns_k_max", 1)
        kwargs.setdefault("vns_trials_per_k", 1)
        super().__init__(fitness_function, **kwargs)

        if self.fitness_function.search_space != SearchSpace.NASBENCH201:
            raise NotImplementedError(
                "Partitioned NSGA-II niches are only defined for nasbench201."
            )
        if self.dim != 1:
            raise NotImplementedError(
                "Partitioned NSGA-II niche constraints currently require dim=1 (NB201 index)."
            )

        self.niche = niche
        self.max_resample_attempts = max(10, int(max_resample_attempts))
        self.niche_pool = np.asarray(niche_pool, dtype=int).reshape(-1)
        if len(self.niche_pool) == 0:
            raise ValueError(
                f"Empty niche_pool for niche_id={self.niche.niche_id}. "
                "This indicates a broken niche cache or niche definition."
            )
        # Fast membership test (used only for safety checks / fallbacks).
        self._niche_pool_set = set(int(x) for x in self.niche_pool.tolist())
        # Legacy cache kept for compatibility / debugging; no longer needed in hot paths.
        self._conv_cache: Dict[int, Tuple[int, int]] = {}

    def _conv_counts_for_idx(self, arch_idx: int) -> Tuple[int, int]:
        arch_idx = int(arch_idx)
        if arch_idx in self._conv_cache:
            return self._conv_cache[arch_idx]

        cfg = self.fitness_function.api.get_net_config(arch_idx, self.fitness_function.dataset)
        arch_str = cfg.get("arch_str") if isinstance(cfg, dict) else None
        if not isinstance(arch_str, str):
            counts = (0, 0)
        else:
            counts = _count_nb201_convs_from_arch_str(arch_str)

        self._conv_cache[arch_idx] = counts
        return counts

    def _in_niche(self, row: np.ndarray) -> bool:
        # NB201 dim=1 => the architecture identity is the integer index itself.
        try:
            idx = int(row[0])
        except Exception:
            return False
        return idx in self._niche_pool_set

    def _sample_row_in_niche(self, *, avoid: Optional[set[int]] = None) -> np.ndarray:
        avoid = avoid or set()

        # If the avoid set covers the whole niche, allow repeats.
        if len(avoid) >= len(self.niche_pool):
            avoid = set()

        attempts = min(self.max_resample_attempts, 256)
        for _ in range(attempts):
            idx = int(self._rng.choice(self.niche_pool))
            if idx in avoid:
                continue
            return np.array([idx], dtype=int)

        # Fallback: return any niche member.
        return np.array([int(self._rng.choice(self.niche_pool))], dtype=int)

    def _initialize_population(self) -> np.ndarray:
        population = np.empty((self.population_size, self.dim), dtype=int)
        seen: set[int] = set()
        for i in range(self.population_size):
            row = self._sample_row_in_niche(avoid=seen)
            population[i] = row
            seen.add(int(row[0]))
        return population

    def _on_iter_end(self, iteration: int) -> None:
        # NSGANetVNS._on_iter_end recomputes objectives; for this constrained
        # optimizer we reuse the objectives computed in `_update_population()`.
        self.front_ranks, self.crowding = self._assign_rank_and_crowding(self.objectives)
        self._update_archive(self.population, self.objectives)

        fronts = fast_non_dominated_sort(self.objectives)
        self._last_front_size = int(len(fronts[0])) if fronts else 0
        self.pareto_history.append(self._last_front_size)

    def _mutate(self, child: np.ndarray, strength: int = 1) -> np.ndarray:
        if self._rng.random() >= self.mutation_prob:
            return child

        # Mutation is restricted to the niche by sampling directly from the
        # precomputed niche pool.
        if len(self.niche_pool) <= 1:
            return child

        current = int(child[0])
        for _ in range(8):
            idx = int(self._rng.choice(self.niche_pool))
            if idx != current:
                out = child.copy()
                out[0] = idx
                return out
        return child


class NSGA2Partitioned(MetaheuristicOptimizer):
    """NSGA-II with complexity-driven partitioning into 6 disjoint niches.

    This is a co-evolution scheme inspired by the provided pseudocode:
    - Maintain 6 independent NSGA-II populations, each constrained to a niche S_k
    - At each generation, evolve all niches (conceptually in parallel)
    - Aggregate the union of niche archives into a final non-dominated set

    Notes
    -----
    - Implemented for NAS-Bench-201 only (dim=1 integer index), since niches are
      defined in terms of counts of `nor_conv_3x3` and `nor_conv_1x1` ops.
    - The runner passes a single `population_size`; here we interpret it as the
      *total* budget and split it across the 6 niches.
    """

    def __init__(
        self,
        fitness_function: HardwareAwareFitness,
        *,
        population_per_niche: Optional[int] = None,
        niche_cache_path: str | Path | None = None,
        rebuild_niche_cache: bool = False,
        max_resample_attempts: int = 5000,
        # NSGA-II parameters (forwarded to each niche optimizer)
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.12,
        mutation_rate_dim: float = 0.25,
        tournament_size: int = 2,
        archive_max_size: int = 256,
        seed: int = 42,
        **kwargs,
    ) -> None:
        super().__init__(fitness_function, seed=seed, **kwargs)

        if self.fitness_function.search_space != SearchSpace.NASBENCH201:
            raise NotImplementedError(
                "NSGA2Partitioned is only supported for search_space='nasbench201'."
            )
        if self.dim != 1:
            raise NotImplementedError("NSGA2Partitioned currently requires dim=1.")

        self.population_per_niche = population_per_niche
        self.niche_cache_path = (
            Path(niche_cache_path) if niche_cache_path is not None else None
        )
        self.rebuild_niche_cache = bool(rebuild_niche_cache)
        self.max_resample_attempts = max(10, int(max_resample_attempts))
        self.crossover_prob = float(crossover_prob)
        self.mutation_prob = float(mutation_prob)
        self.mutation_rate_dim = float(mutation_rate_dim)
        self.tournament_size = int(tournament_size)
        self.archive_max_size = int(archive_max_size)

        self.niches: List[NicheDefinition] = [NicheDefinition(i) for i in range(6)]

        self.archive_population = np.empty((0, self.dim), dtype=int)
        self.archive_objectives = np.empty((0, 3), dtype=float)
        self.pareto_history: List[int] = []

        self._niche_pools: Optional[List[np.ndarray]] = None

    @staticmethod
    def _default_niche_cache_path() -> Path:
        # nsga-II-nas-implementation/algorithms/implementations/nsga2_partitioned.py
        # -> project root is parents[2]
        project_root = Path(__file__).resolve().parents[2]
        cache_dir = project_root / "data" / "niche_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Dataset-agnostic: NB201 arch_str (and thus niches) are independent of the dataset.
        return cache_dir / "nb201_niche_pools_v1.npz"

    def _load_or_build_niche_pools(self) -> List[np.ndarray]:
        if self._niche_pools is not None:
            return self._niche_pools

        dataset = str(self.fitness_function.dataset)
        cache_path = self.niche_cache_path or self._default_niche_cache_path()

        if cache_path.exists() and not self.rebuild_niche_cache:
            try:
                data = np.load(cache_path, allow_pickle=False)
                if int(data["search_space_size"].item()) != int(self.search_space_size):
                    raise ValueError("Cache search_space_size mismatch")
                pools = [
                    np.asarray(data[f"pool_{i}"], dtype=int).reshape(-1)
                    for i in range(6)
                ]
                if any(len(p) == 0 for p in pools):
                    raise ValueError("Cache contains empty niche pool")
                self._niche_pools = pools
                return pools
            except Exception:
                # If cache is corrupted or incompatible, rebuild it.
                pass

        # Build niche pools from the benchmark configs.
        pools_list: List[List[int]] = [[] for _ in range(6)]
        for arch_idx in range(int(self.search_space_size)):
            cfg = self.fitness_function.api.get_net_config(arch_idx, dataset)
            arch_str = cfg.get("arch_str") if isinstance(cfg, dict) else None
            if not isinstance(arch_str, str):
                conv3, conv1 = 0, 0
            else:
                conv3, conv1 = _count_nb201_convs_from_arch_str(arch_str)

            assigned = False
            for niche in self.niches:
                if niche.contains(conv3x3=conv3, conv1x1=conv1):
                    pools_list[niche.niche_id].append(int(arch_idx))
                    assigned = True
                    break
            if not assigned:
                pools_list[0].append(int(arch_idx))

        pools = [np.asarray(p, dtype=int) for p in pools_list]
        if any(len(p) == 0 for p in pools):
            raise RuntimeError(
                "Failed to build niche pools: at least one niche is empty. "
                "Check niche definitions and pickle config availability."
            )

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            search_space_size=np.asarray(int(self.search_space_size), dtype=int),
            pool_0=pools[0],
            pool_1=pools[1],
            pool_2=pools[2],
            pool_3=pools[3],
            pool_4=pools[4],
            pool_5=pools[5],
        )

        self._niche_pools = pools
        return pools

    def _update_population(self, iteration: int) -> None:
        # Unused: this optimizer overrides `optimize()` to coordinate 6 niches.
        raise NotImplementedError

    def _split_population_sizes(self) -> List[int]:
        if self.population_per_niche is not None:
            sizes = [int(self.population_per_niche)] * 6
            if any(s < 2 for s in sizes):
                raise ValueError("population_per_niche must be >= 2")
            return sizes

        total = int(self.population_size)
        base = total // 6
        rem = total % 6
        sizes = [base + (1 if i < rem else 0) for i in range(6)]
        if any(s < 2 for s in sizes):
            raise ValueError(
                "population_size is too small to split across 6 niches "
                "(need at least 12 to ensure 2 individuals per niche)."
            )
        return sizes

    def _merge_archives(self, niche_opts: List[_NicheConstrainedNSGA2]) -> None:
        pops = [o.archive_population for o in niche_opts if len(o.archive_population) > 0]
        objs = [o.archive_objectives for o in niche_opts if len(o.archive_objectives) > 0]
        if not pops:
            self.archive_population = np.empty((0, self.dim), dtype=int)
            self.archive_objectives = np.empty((0, 3), dtype=float)
            return

        merged_pop = np.vstack(pops)
        merged_obj = np.vstack(objs)
        nd_pop, nd_obj = extract_nondominated(merged_pop, merged_obj)
        if len(nd_pop) > self.archive_max_size:
            nd_pop, nd_obj = environmental_selection(nd_pop, nd_obj, self.archive_max_size)
        self.archive_population = nd_pop
        self.archive_objectives = nd_obj

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        sizes = self._split_population_sizes()

        niche_pools = self._load_or_build_niche_pools()

        niche_opts: List[_NicheConstrainedNSGA2] = []
        for niche_id, niche in enumerate(self.niches):
            niche_seed = int(self.seed + 1000 * niche_id)
            opt = _NicheConstrainedNSGA2(
                self.fitness_function,
                niche=niche,
                niche_pool=niche_pools[niche_id],
                max_resample_attempts=self.max_resample_attempts,
                search_space_size=self.search_space_size,
                dim=self.dim,
                population_size=sizes[niche_id],
                max_iterations=self.max_iterations,
                seed=niche_seed,
                crossover_prob=self.crossover_prob,
                mutation_prob=self.mutation_prob,
                mutation_rate_dim=self.mutation_rate_dim,
                tournament_size=self.tournament_size,
                archive_max_size=self.archive_max_size,
            )
            niche_opts.append(opt)

        # Phase 1: niche initialization
        global_best_fit = -np.inf
        global_best_sol: Optional[np.ndarray] = None
        self.fitness_history = []
        self._convergence_iter = 0

        for opt in niche_opts:
            opt.population = opt._initialize_population()
            opt.fitness_values = opt._evaluate_population(opt.population)
            opt._on_init()

            best_idx = int(np.argmax(opt.fitness_values))
            niche_best_fit = float(opt.fitness_values[best_idx])
            niche_best_sol = opt.population[best_idx].copy()
            if niche_best_fit > global_best_fit:
                global_best_fit = niche_best_fit
                global_best_sol = niche_best_sol

        if global_best_sol is None:
            global_best_sol = np.zeros((self.dim,), dtype=int)
            global_best_fit = -np.inf

        self.best_solution = global_best_sol.copy()
        self.best_fitness = float(global_best_fit)
        self.fitness_history.append(self.best_fitness)

        # Phase 2: partitioned co-evolution
        for iteration in range(self.max_iterations):
            for opt in niche_opts:
                opt._update_population(iteration)

            for opt in niche_opts:
                opt._on_iter_end(iteration)

            # Update global best (scalar fitness) across all niches
            for opt in niche_opts:
                idx = int(np.argmax(opt.fitness_values))
                fit = float(opt.fitness_values[idx])
                if fit > self.best_fitness:
                    self.best_fitness = fit
                    self.best_solution = opt.population[idx].copy()
                    self._convergence_iter = iteration

            self.fitness_history.append(float(self.best_fitness))

            # Keep a light Pareto history: total archive size after merge
            self._merge_archives(niche_opts)
            self.pareto_history.append(int(len(self.archive_population)))

        # Phase 3: final aggregation
        self._merge_archives(niche_opts)

        assert self.best_solution is not None
        return self.best_solution.copy(), float(self.best_fitness), list(self.fitness_history)

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
            best_fitness=float(best_fit),
            fitness_history=history,
            hardware_metrics=metrics,
            query_count=stats["api_queries"],
            convergence_iter=self._convergence_iter,
        )

    def __repr__(self) -> str:
        sizes = None
        try:
            sizes = self._split_population_sizes()
        except Exception:
            sizes = None
        split = f"split={sizes}" if sizes is not None else "split=?"
        return (
            f"{self.__class__.__name__}(pop={self.population_size}, {split}, "
            f"iters={self.max_iterations}, seed={self.seed})"
        )
