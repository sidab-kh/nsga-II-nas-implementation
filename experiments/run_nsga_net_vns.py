"""
Run NSGA-NET+VNS on NAS-Bench-201 and HW-NAS(FBNet) in one command.

Usage:
    uv run python experiments/run_nsga_net_vns.py
    uv run python experiments/run_nsga_net_vns.py --only nb201
    uv run python experiments/run_nsga_net_vns.py --only fbnet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.metaheuristics import get_algorithm
from core.api import HWNASBenchAPI
from core.config import ExperimentConfig
from experiments.runner import ExperimentRunner


def _resolve_path(path_value: str | None) -> str | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)


def _run_from_config(
    config_path: Path,
    *,
    pickle_override: str | None,
    nb201_override: str | None,
) -> Dict[str, object]:
    config = ExperimentConfig.from_yaml(config_path)

    if pickle_override is not None:
        config.paths.pickle = pickle_override
    if nb201_override is not None:
        config.paths.nb201 = nb201_override

    config.paths.pickle = _resolve_path(config.paths.pickle) or config.paths.pickle
    config.paths.nb201 = _resolve_path(config.paths.nb201)

    config.validate()

    algo_name = config.search.algorithms[0]
    algo_class = get_algorithm(algo_name)

    api = HWNASBenchAPI(
        config.paths.pickle,
        search_space=config.search.search_space,
    )

    runner = ExperimentRunner(
        num_runs=config.runner.runs,
        seed_base=config.runner.seed,
        verbose=not config.runner.quiet,
    )
    runner.run(
        algorithms={algo_name: algo_class},
        api=api,
        target_device=config.search.device,
        dataset=config.search.dataset,
        nb201_path=config.paths.nb201,
        latency_weight=config.fitness.latency_weight,
        energy_weight=config.fitness.energy_weight,
        accuracy_weight=config.fitness.accuracy_weight,
        population_size=config.runner.population_size,
        max_iterations=config.runner.max_iterations,
        extra_kwargs=config.extra_kwargs,
    )

    runner.save(config.output.directory, run_name=config.output.run_name)
    summary = runner.summarize()
    if summary.empty:
        raise RuntimeError(f"No valid results for {config_path}")

    row = summary.iloc[0].to_dict()
    return {
        "config": str(config_path),
        "algorithm": row.get("Algorithm"),
        "best_fitness": float(row.get("Best_Fitness")),
        "mean_fitness": float(row.get("Mean_Fitness")),
        "output_dir": str(Path(config.output.directory) / str(config.output.run_name)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NSGA-NET+VNS benchmark suite.")
    parser.add_argument(
        "--only",
        type=str,
        choices=["nb201", "fbnet"],
        default=None,
        help="Run only one benchmark instead of both.",
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default=None,
        help="Override HW-NAS-Bench pickle path (absolute or repo-relative).",
    )
    parser.add_argument(
        "--nb201",
        type=str,
        default=None,
        help="Override NAS-Bench-201 .pth path (absolute or repo-relative).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dir = ROOT / "experiments" / "configs"

    jobs = []
    if args.only in (None, "nb201"):
        jobs.append(cfg_dir / "nsga_net_vns_nb201.yaml")
    if args.only in (None, "fbnet"):
        jobs.append(cfg_dir / "nsga_net_vns_fbnet.yaml")

    print("\nRunning NSGA-NET+VNS suite")
    print("=" * 50)

    reports = []
    for cfg in jobs:
        print(f"\n[RUN] {cfg.name}")
        try:
            report = _run_from_config(
                cfg,
                pickle_override=args.pickle,
                nb201_override=args.nb201,
            )
        except FileNotFoundError as exc:
            print("\n[ERROR] Benchmark file not found")
            print(f"  {exc}")
            print("\nPlace the datasets in repo-local data/ or pass explicit paths:")
            print(
                "  python experiments/run_nsga_net_vns.py "
                "--pickle <path-to-HW-NAS-Bench-v1_0.pickle> "
                "--nb201 <path-to-NAS-Bench-201-v1_1-096897.pth>"
            )
            raise SystemExit(1) from exc

        reports.append(report)
        print(
            f"  best={report['best_fitness']:.6f}  "
            f"mean={report['mean_fitness']:.6f}  "
            f"out={report['output_dir']}"
        )

    print("\nSuite completed.")


if __name__ == "__main__":
    main()
