"""
experiments/runner.py
=====================
Experiment runner — algorithm-agnostic, reproducible, results-oriented.

Usage
-----
    from experiments.runner import ExperimentRunner
    from algorithms.metaheuristics import REGISTRY

    runner = ExperimentRunner(num_runs=10, seed_base=42)
    results = runner.run(
        algorithms=REGISTRY,          # or a subset
        api=hw_api,
        target_device="edgegpu",
        dataset="cifar10",
        population_size=20,
        max_iterations=100,
    )
    df = runner.summarize()
    runner.save("./outputs")
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from algorithms.base import MetaheuristicOptimizer
from core.api import HWNASBenchAPI
from core.fitness import HardwareAwareFitness
from core.types import RunResult, VALID_DATASETS, VALID_DEVICES


class ExperimentRunner:
    """
    Runs multiple algorithms × multiple runs on HW-NAS-Bench.

    Parameters
    ----------
    num_runs : int
        Number of independent runs per algorithm.
    seed_base : int
        Run r uses seed ``seed_base + r`` for reproducibility.
    verbose : bool
        Print progress to stdout.
    """

    def __init__(
        self,
        num_runs: int = 10,
        seed_base: int = 42,
        verbose: bool = True,
    ) -> None:
        self.num_runs = num_runs
        self.seed_base = seed_base
        self.verbose = verbose
        self._results: Dict[str, List[RunResult]] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        algorithms: Dict[str, Type[MetaheuristicOptimizer]],
        api: HWNASBenchAPI,
        *,
        target_device: str = "edgegpu",
        dataset: str = "cifar10",
        latency_weight: float = 0.5,
        energy_weight: float = 0.2,
        accuracy_weight: float = 0.3,
        nb201_path: Optional[str] = None,
        population_size: int = 20,
        max_iterations: int = 100,
        extra_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, List[RunResult]]:
        """
        Run all algorithms for ``num_runs`` independent runs each.

        Parameters
        ----------
        algorithms : dict
            Mapping of name → class (use ``REGISTRY`` or a subset).
        api : HWNASBenchAPI
        target_device : str
        dataset : str
        latency_weight, energy_weight : float
        population_size, max_iterations : int
        extra_kwargs : dict, optional
            Per-algorithm keyword overrides, e.g.
            ``{"SA": {"T_init": 2.0, "cooling": 0.99}}``.

        Returns
        -------
        dict mapping algorithm name → list of RunResult
        """
        if target_device not in VALID_DEVICES:
            raise ValueError(f"target_device must be one of {VALID_DEVICES}")
        if dataset not in VALID_DATASETS:
            raise ValueError(f"dataset must be one of {VALID_DATASETS}")

        extra_kwargs = extra_kwargs or {}
        self._results = {}

        for algo_name, AlgoClass in algorithms.items():
            self._log(f"\n{'='*70}")
            self._log(f"  Algorithm : {algo_name}")
            self._log(f"  Device    : {target_device} | Dataset : {dataset}")
            self._log(f"  Pop={population_size}, Iters={max_iterations}, Runs={self.num_runs}")
            self._log(f"{'='*70}")

            self._results[algo_name] = []
            algo_extra = extra_kwargs.get(algo_name, {})

            for run_id in range(self.num_runs):
                result = self._single_run(
                    AlgoClass=AlgoClass,
                    algo_name=algo_name,
                    run_id=run_id,
                    api=api,
                    target_device=target_device,
                    dataset=dataset,
                    latency_weight=latency_weight,
                    energy_weight=energy_weight,
                    accuracy_weight=accuracy_weight,
                    nb201_path=nb201_path,
                    population_size=population_size,
                    max_iterations=max_iterations,
                    **algo_extra,
                )
                self._results[algo_name].append(result)
                self._log(
                    f"  Run {run_id + 1:>2}/{self.num_runs}  "
                    f"best_fitness={result.best_fitness:.6f}  "
                    + (
                        f"arch_idx={result.best_arch_idx}"
                        if result.best_arch_vector is None
                        else f"arch={result.best_arch_vector}"
                    )
                )

        return self._results

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def _single_run(
        self,
        AlgoClass: Type[MetaheuristicOptimizer],
        algo_name: str,
        run_id: int,
        api: HWNASBenchAPI,
        target_device: str,
        dataset: str,
        latency_weight: float,
        energy_weight: float,
        accuracy_weight: float,
        nb201_path: Optional[str],
        population_size: int,
        max_iterations: int,
        **extra,
    ) -> RunResult:
        fitness_fn = HardwareAwareFitness(
            api,
            nb201_path=nb201_path,
            target_device=target_device,
            dataset=dataset,
            latency_weight=latency_weight,
            energy_weight=energy_weight,
            accuracy_weight=accuracy_weight,
        )

        optimizer = AlgoClass(
            fitness_function=fitness_fn,
            search_space_size=api.search_space_size(dataset),
            dim=api.search_dimension(dataset),
            population_size=population_size,
            max_iterations=max_iterations,
            seed=self.seed_base + run_id,
            **extra,
        )

        t0 = time.perf_counter()
        result = optimizer.run(run_id=run_id)
        result.algorithm_name = algo_name   # override class name with registry name
        elapsed = time.perf_counter() - t0

        if self.verbose:
            print(
                f"    elapsed={elapsed:.1f}s  queries={result.query_count}  "
                f"conv_iter={result.convergence_iter}"
            )

        return result

    # ------------------------------------------------------------------
    # Summarise
    # ------------------------------------------------------------------

    def summarize(self) -> pd.DataFrame:
        """Return a per-algorithm summary DataFrame."""
        if not self._results:
            return pd.DataFrame()

        rows = []
        for algo_name, runs in self._results.items():
            valid = [r for r in runs if r.is_valid()]
            if not valid:
                continue

            fitness_vals = [r.best_fitness for r in valid]
            query_counts = [r.query_count for r in valid]
            conv_iters = [r.convergence_iter for r in valid if r.convergence_iter is not None]

            best_run = valid[int(np.argmax(fitness_vals))]
            hw = best_run.hardware_metrics

            row: Dict[str, Any] = {
                "Algorithm":       algo_name,
                "Best_Fitness":    float(np.max(fitness_vals)),
                "Mean_Fitness":    float(np.mean(fitness_vals)),
                "Std_Fitness":     float(np.std(fitness_vals)),
                "Worst_Fitness":   float(np.min(fitness_vals)),
                "Mean_Queries":    float(np.mean(query_counts)),
                "Mean_Conv_Iter":  float(np.mean(conv_iters)) if conv_iters else None,
                "Valid_Runs":      len(valid),
                "Total_Runs":      len(runs),
            }

            if best_run.best_arch_vector is None:
                row["Best_Arch_Idx"] = best_run.best_arch_idx
            else:
                row["Best_Arch_Vector"] = str(best_run.best_arch_vector)

            if hw is not None:
                row["EdgeGPU_Latency_ms"]  = hw.edgegpu_latency
                row["RasPi4_Latency_ms"]   = hw.raspi4_latency
                row["FPGA_Latency_ms"]     = hw.fpga_latency

            rows.append(row)

        return pd.DataFrame(rows).sort_values("Best_Fitness", ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path, run_name: str | None = None) -> None:
        """Save summary CSV + detailed JSON to *output_dir*[/run_name]."""
        output_dir = Path(output_dir)
        if run_name:
            output_dir = output_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Summary CSV
        df = self.summarize()
        csv_path = output_dir / "summary.csv"
        df.to_csv(csv_path, index=False)
        self._log(f"\n✓ Summary saved  → {csv_path}")

        # Detailed JSON
        detailed: Dict = {}
        for algo_name, runs in self._results.items():
            detailed[algo_name] = [r.to_dict() for r in runs]

        json_path = output_dir / "detailed_results.json"
        with open(json_path, "w") as fh:
            json.dump(detailed, fh, indent=2, default=str)
        self._log(f"✓ Details saved  → {json_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)
