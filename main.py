"""
main.py
=======
Main entry point for HW-NAS-Bench metaheuristic experiments.

Quick start
-----------
    python main.py --config config.yaml

Run all algorithms
------------------
    python main.py --config config.yaml --algorithms all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Make sure the project root is on the Python path when running directly
sys.path.insert(0, str(Path(__file__).parent))

from algorithms.metaheuristics import REGISTRY
from core.api import HWNASBenchAPI
from core.config import ExperimentConfig
from core.types import VALID_DATASETS, VALID_DEVICES
from experiments.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HW-NAS-Bench metaheuristic NAS experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--pickle",
        type=str,
        default=None,
        help="Override paths.pickle from config.",
    )
    parser.add_argument(
        "--search-space",
        type=str,
        default=None,
        choices=["nasbench201", "fbnet"],
        help="Override search.search_space from config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=list(VALID_DEVICES),
        help="Override search.device from config.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        choices=list(VALID_DATASETS),
        help="Override search.dataset from config.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help=f"Algorithms to run (or 'all'). Available: {sorted(REGISTRY.keys())}",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override runner.runs from config.",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=None,
        help="Override runner.population_size from config.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override runner.max_iterations from config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override runner.seed from config.",
    )
    parser.add_argument(
        "--nb201",
        type=str,
        default=None,
        help="Override paths.nb201 from config.",
    )
    parser.add_argument(
        "--lat-weight",
        type=float,
        default=None,
        help="Override fitness.latency_weight from config.",
    )
    parser.add_argument(
        "--eng-weight",
        type=float,
        default=None,
        help="Override fitness.energy_weight from config.",
    )
    parser.add_argument(
        "--acc-weight",
        type=float,
        default=None,
        help="Override fitness.accuracy_weight from config.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output.directory from config.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Override output.run_name from config.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=None,
        help="Override runner.quiet to true.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(args.config)
    return config.apply_overrides(
        {
            "paths": {
                "pickle": args.pickle,
                "nb201": args.nb201,
            },
            "search": {
                "search_space": args.search_space,
                "device": args.device,
                "dataset": args.dataset,
                "algorithms": args.algorithms,
            },
            "runner": {
                "runs": args.runs,
                "population_size": args.pop,
                "max_iterations": args.iters,
                "seed": args.seed,
                "quiet": args.quiet,
            },
            "fitness": {
                "latency_weight": args.lat_weight,
                "energy_weight": args.eng_weight,
                "accuracy_weight": args.acc_weight,
            },
            "output": {
                "directory": args.output,
                "run_name": args.run_name,
            },
        }
    )


def main() -> None:
    args = parse_args()
    config = load_config(args)
    config.validate()

    if config.search.algorithms == ["all"]:
        selected = dict(REGISTRY)
    else:
        missing = [name for name in config.search.algorithms if name not in REGISTRY]
        if missing:
            print(f"[ERROR] Unknown algorithms: {missing}")
            print(f"        Available: {sorted(REGISTRY.keys())}")
            sys.exit(1)
        selected = {name: REGISTRY[name] for name in config.search.algorithms}

    print("\n" + "=" * 80)
    print("  HW-NAS-BENCH METAHEURISTIC SEARCH")
    print("=" * 80)
    print(f"  Config       : {args.config}")
    print(f"  Pickle       : {config.paths.pickle}")
    print(f"  Search space : {config.search.search_space}")
    print(
        f"  Device       : {config.search.device}"
        f"  |  Dataset : {config.search.dataset}"
    )
    print(f"  Algorithms   : {list(selected.keys())}")
    print(
        "  Runs / algo  : "
        f"{config.runner.runs}  |  Pop : {config.runner.population_size}"
        f"  |  Iters : {config.runner.max_iterations}"
    )
    print("=" * 80 + "\n")

    try:
        api = HWNASBenchAPI(
            config.paths.pickle,
            search_space=config.search.search_space,
        )
        print(f"Loaded {api}\n")
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    runner = ExperimentRunner(
        num_runs=config.runner.runs,
        seed_base=config.runner.seed,
        verbose=not config.runner.quiet,
    )
    try:
        runner.run(
            algorithms=selected,
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
    except NotImplementedError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("  RESULTS SUMMARY")
    print("=" * 80)
    df = runner.summarize()
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))

    runner.save(
        config.output.directory,
        run_name=config.output.run_name,
    )

    print("\n" + "=" * 80)
    print("  BEST ARCHITECTURE HARDWARE METRICS (per algorithm)")
    print("=" * 80)
    for algo_name, runs in runner._results.items():
        valid = [r for r in runs if r.is_valid() and r.hardware_metrics is not None]
        if not valid:
            continue

        best = max(valid, key=lambda r: r.best_fitness)
        hw = best.hardware_metrics
        arch_label = (
            f"arch_idx={best.best_arch_idx}"
            if best.best_arch_vector is None
            else f"arch={best.best_arch_vector}"
        )
        print(f"\n  {algo_name}  ->  {arch_label}")
        print(
            f"    EdgeGPU  : {hw.edgegpu_latency:.3f} ms"
            + (f"  /  {hw.edgegpu_energy:.3f} mJ" if hw.edgegpu_energy else "")
        )
        print(f"    RasPi4   : {hw.raspi4_latency:.3f} ms")
        if config.search.search_space == "fbnet":
            print("    EdgeTPU  : n/a")
        else:
            print(f"    EdgeTPU  : {hw.edgetpu_latency:.3f} ms")
        print(f"    Pixel3   : {hw.pixel3_latency:.3f} ms")
        print(
            f"    Eyeriss  : {hw.eyeriss_latency:.3f} ms"
            + (f"  /  {hw.eyeriss_energy:.3f} mJ" if hw.eyeriss_energy else "")
        )
        print(
            f"    FPGA     : {hw.fpga_latency:.3f} ms"
            + (f"  /  {hw.fpga_energy:.3f} mJ" if hw.fpga_energy else "")
        )

    print("\nDone.\n")


if __name__ == "__main__":
    main()
