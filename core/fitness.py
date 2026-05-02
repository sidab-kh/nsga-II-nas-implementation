"""
core/fitness.py
===============
Hardware-aware and accuracy-aware fitness functions for HW-NAS-Bench.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Protocol, Tuple

import numpy as np

from core.api import HWNASBenchAPI
from core.types import (
    HardwareMetrics,
    NASBench201Architecture,
    SearchSpace,
    VALID_DATASETS,
    VALID_DEVICES,
)


_NB201_DATASET_MAP: Dict[str, str] = {
    "cifar10": "cifar10-valid",
    "cifar100": "cifar100",
    "ImageNet16-120": "ImageNet16-120",
}
_NB201_METRIC_SET: Dict[str, str] = {
    "cifar10": "x-valid",
    "cifar100": "x-valid",
    "ImageNet16-120": "x-valid",
}

_ACC_BASELINES: Dict[str, float] = {
    "cifar10": 94.37,
    "cifar100": 73.49,
    "ImageNet16-120": 47.31,
}

_DEFAULT_LATENCY_BASELINES: Dict[str, float] = {
    "edgegpu": 100.0,
    "raspi4": 2000.0,
    "edgetpu": 10.0,
    "pixel3": 150.0,
    "eyeriss": 5000.0,
    "fpga": 50.0,
}
_DEFAULT_ENERGY_BASELINES: Dict[str, float] = {
    "edgegpu": 50.0,
    "eyeriss": 20000.0,
    "fpga": 30.0,
}


class NASBench201AccuracyAPI:
    """Thin wrapper around the official NAS-Bench-201 API."""

    _DOWNLOAD_MSG = (
        "NAS-Bench-201 is unavailable. Install `nas-bench-201` and provide "
        "`NAS-Bench-201-v1_1-096897.pth` to enable accuracy-aware fitness."
    )

    def __init__(self, nb201_path: str, hp: str = "200") -> None:
        self._api = None
        self._hp = hp
        self._cache: Dict[Tuple[int, str], float] = {}
        self._available = False

        try:
            from nas_201_api import NASBench201API  # type: ignore

            try:
                self._api = NASBench201API(nb201_path, verbose=False)
                self._available = True
                print(f"NAS-Bench-201 loaded (hp={hp} epochs)")
            except Exception as exc:
                # PyTorch >=2.6 defaults torch.load(weights_only=True), which can
                # break legacy NAS-Bench-201 checkpoint loading. Retry with
                # weights_only=False only for this trusted benchmark file.
                if "Weights only load failed" not in str(exc):
                    raise

                try:
                    import torch  # type: ignore
                except Exception:
                    raise exc

                original_torch_load = torch.load

                def _torch_load_compat(*args, **kwargs):
                    if "weights_only" not in kwargs:
                        kwargs["weights_only"] = False
                    return original_torch_load(*args, **kwargs)

                torch.load = _torch_load_compat
                try:
                    self._api = NASBench201API(nb201_path, verbose=False)
                    self._available = True
                    print(f"NAS-Bench-201 loaded (hp={hp} epochs, torch compat mode)")
                finally:
                    torch.load = original_torch_load
        except ImportError:
            warnings.warn(
                "nas_201_api not installed. Run: pip install nas-bench-201\n"
                + self._DOWNLOAD_MSG,
                UserWarning,
                stacklevel=3,
            )
        except FileNotFoundError:
            warnings.warn(self._DOWNLOAD_MSG, UserWarning, stacklevel=3)
        except Exception as exc:
            warnings.warn(
                f"NAS-Bench-201 failed to load ({exc}). Accuracy disabled.",
                UserWarning,
                stacklevel=3,
            )

    @property
    def available(self) -> bool:
        return self._available

    def get_accuracy(self, arch_idx: int, dataset: str = "cifar10") -> float:
        if not self._available or self._api is None:
            return 0.0

        key = (arch_idx, dataset)
        if key in self._cache:
            return self._cache[key]

        nb201_ds = _NB201_DATASET_MAP.get(dataset, dataset)
        metric_set = _NB201_METRIC_SET.get(dataset, "x-valid")

        try:
            info = self._api.query_meta_info_by_index(arch_idx, hp=self._hp)
            acc = float(info.get_metrics(nb201_ds, metric_set).get("accuracy", 0.0))
        except Exception:
            acc = 0.0

        self._cache[key] = acc
        return acc


class _AccuracyAPI(Protocol):
    @property
    def available(self) -> bool: ...

    def get_accuracy(self, arch_idx: int, dataset: str = "cifar10") -> float: ...


class NASBench201PickleAccuracyAPI:
    """Accuracy lookup from a precomputed pickle mapping.

    Expected file format matches `data/nasbench201_full_mapping.pkl` as demonstrated
    in `use_nasbench201_pickle.py`:

    - top-level: dict[int, arch_info]
    - arch_info['datasets'][dataset_key] contains:
        - 'metrics' (may include 'accuracy')
        - 'more_info' (may include 'valid-accuracy'/'test-accuracy'/'train-accuracy')
    """

    _UNAVAILABLE_MSG = (
        "NAS-Bench-201 pickle mapping is unavailable. Provide "
        "`data/nasbench201_full_mapping.pkl` to enable accuracy-aware fitness."
    )

    def __init__(self, mapping_path: str) -> None:
        self._mapping_path = mapping_path
        self._data: Optional[Dict[int, Any]] = None
        self._cache: Dict[Tuple[int, str], float] = {}
        self._available = False

        try:
            import pickle

            with open(mapping_path, "rb") as f:
                loaded = pickle.load(f)

            if not isinstance(loaded, dict):
                raise TypeError(
                    f"Expected dict in mapping pickle, got {type(loaded).__name__}"
                )

            # Normalize keys to int where possible.
            data: Dict[int, Any] = {}
            for k, v in loaded.items():
                try:
                    ik = int(k)
                except Exception:
                    continue
                data[ik] = v

            self._data = data
            self._available = len(data) > 0
            if self._available:
                print("NAS-Bench-201 pickle mapping loaded")
        except FileNotFoundError:
            warnings.warn(self._UNAVAILABLE_MSG, UserWarning, stacklevel=3)
        except Exception as exc:
            warnings.warn(
                f"NAS-Bench-201 pickle mapping failed to load ({exc}). Accuracy disabled.",
                UserWarning,
                stacklevel=3,
            )

    @staticmethod
    def _extract_accuracy(dataset_info: Dict[str, Any]) -> Optional[float]:
        metrics = dataset_info.get("metrics", {})
        if isinstance(metrics, dict) and "accuracy" in metrics:
            try:
                return float(metrics["accuracy"])
            except Exception:
                return None

        more_info = dataset_info.get("more_info", {})
        if isinstance(more_info, dict):
            for key in ("valid-accuracy", "test-accuracy", "train-accuracy"):
                if key in more_info:
                    try:
                        return float(more_info[key])
                    except Exception:
                        return None

        return None

    @property
    def available(self) -> bool:
        return self._available

    def get_accuracy(self, arch_idx: int, dataset: str = "cifar10") -> float:
        if not self._available or self._data is None:
            return 0.0

        key = (int(arch_idx), dataset)
        if key in self._cache:
            return self._cache[key]

        arch = self._data.get(int(arch_idx))
        if not isinstance(arch, dict):
            self._cache[key] = 0.0
            return 0.0

        datasets = arch.get("datasets", {})
        if not isinstance(datasets, dict):
            self._cache[key] = 0.0
            return 0.0

        # Prefer the same dataset mapping as the official API wrapper.
        nb201_ds = _NB201_DATASET_MAP.get(dataset, dataset)

        # Try mapped key, then original key.
        dataset_info = datasets.get(nb201_ds) or datasets.get(dataset)
        if not isinstance(dataset_info, dict):
            self._cache[key] = 0.0
            return 0.0

        acc = self._extract_accuracy(dataset_info)
        if acc is None:
            acc = 0.0

        self._cache[key] = float(acc)
        return float(acc)


class HardwareAwareFitness:
    """Joint hardware and optional accuracy fitness."""

    def __init__(
        self,
        api: HWNASBenchAPI,
        *,
        nb201_path: Optional[str] = None,
        nb201_mapping_path: Optional[str] = None,
        target_device: str = "edgegpu",
        dataset: str = "cifar10",
        latency_weight: float = 0.5,
        energy_weight: float = 0.2,
        accuracy_weight: float = 0.3,
        latency_baseline: Optional[float] = None,
        energy_baseline: Optional[float] = None,
        nb201_hp: str = "200",
    ) -> None:
        if target_device not in VALID_DEVICES:
            raise ValueError(f"target_device must be one of {VALID_DEVICES}")
        if dataset not in VALID_DATASETS:
            warnings.warn(f"dataset {dataset!r} is non-standard.", UserWarning)

        self.api = api
        self.target_device = target_device
        self.dataset = dataset
        self.search_space = api.search_space

        self._acc_api: Optional[_AccuracyAPI] = None
        self._acc_available = False

        if self.search_space == SearchSpace.FBNET and accuracy_weight > 0:
            warnings.warn(
                "FBNet in HW-NAS-Bench does not provide accuracy labels. "
                "Forcing accuracy_weight=0 and running in hardware-only mode.",
                UserWarning,
            )
            accuracy_weight = 0.0

        if self.search_space == SearchSpace.NASBENCH201 and accuracy_weight > 0:
            if nb201_mapping_path is not None:
                self._acc_api = NASBench201PickleAccuracyAPI(nb201_mapping_path)
                self._acc_available = self._acc_api.available
            elif nb201_path is not None:
                self._acc_api = NASBench201AccuracyAPI(nb201_path, hp=nb201_hp)
                self._acc_available = self._acc_api.available

        if not self._acc_available and accuracy_weight > 0:
            warnings.warn(
                "accuracy_weight > 0 but NAS-Bench-201 is unavailable. "
                "Forcing accuracy_weight=0 and running in hardware-only mode.",
                UserWarning,
            )
            accuracy_weight = 0.0

        total = latency_weight + energy_weight + accuracy_weight
        if total <= 0:
            raise ValueError("At least one weight must be positive.")
        self.w_lat = latency_weight / total
        self.w_eng = energy_weight / total
        self.w_acc = accuracy_weight / total

        self.lat_baseline = latency_baseline or _DEFAULT_LATENCY_BASELINES.get(
            target_device, 100.0
        )
        self.eng_baseline = energy_baseline or _DEFAULT_ENERGY_BASELINES.get(
            target_device, 50.0
        )
        self.acc_baseline = _ACC_BASELINES.get(dataset, 100.0)

        self._hw_cache: Dict[Tuple[str, int | Tuple[int, ...]], HardwareMetrics] = {}
        self._query_count = 0
        self._cache_hits = 0

    def get_hardware_metrics(
        self, architecture: NASBench201Architecture
    ) -> Optional[HardwareMetrics]:
        cache_key = architecture.cache_key()
        if cache_key in self._hw_cache:
            self._cache_hits += 1
            return self._hw_cache[cache_key]

        raw = self.api.query_by_architecture(architecture, self.dataset)
        if raw is None:
            return None

        metrics = HardwareMetrics.from_dict(raw)
        self._hw_cache[cache_key] = metrics
        self._query_count += 1
        return metrics

    def get_accuracy(self, architecture: NASBench201Architecture) -> float:
        if self._acc_api is None or not self._acc_available or architecture.arch_idx is None:
            return 0.0
        return self._acc_api.get_accuracy(architecture.arch_idx, self.dataset)

    def compute(self, architecture: NASBench201Architecture) -> float:
        metrics = self.get_hardware_metrics(architecture)
        if metrics is None:
            return -np.inf

        lat = metrics.get_latency(self.target_device)
        eng = metrics.get_energy(self.target_device)

        lat_norm = lat / self.lat_baseline if self.lat_baseline > 0 else lat
        eng_norm = (
            eng / self.eng_baseline
            if eng is not None and self.eng_baseline > 0
            else 0.0
        )

        hw_penalty = self.w_lat * lat_norm + self.w_eng * eng_norm
        if self.w_acc > 0 and self._acc_available:
            acc_norm = self.get_accuracy(architecture) / self.acc_baseline
            return self.w_acc * acc_norm - hw_penalty
        return -hw_penalty

    def compute_objectives(
        self, architecture: NASBench201Architecture
    ) -> Dict[str, float]:
        metrics = self.get_hardware_metrics(architecture)
        if metrics is None:
            return {
                "accuracy": 0.0,
                "latency": np.inf,
                "energy": np.inf,
                "fitness": -np.inf,
            }

        lat = metrics.get_latency(self.target_device)
        eng = metrics.get_energy(self.target_device) or 0.0
        acc = self.get_accuracy(architecture)

        return {
            "accuracy": acc,
            "latency": lat,
            "energy": eng,
            "fitness": self.compute(architecture),
        }

    def compute_multi(
        self, architecture: NASBench201Architecture
    ) -> Tuple[float, float, float]:
        metrics = self.get_hardware_metrics(architecture)
        if metrics is None:
            return (np.inf, np.inf, np.inf)

        lat = metrics.get_latency(self.target_device)
        eng = metrics.get_energy(self.target_device) or 0.0
        acc = self.get_accuracy(architecture)

        lat_norm = lat / self.lat_baseline if self.lat_baseline > 0 else lat
        eng_norm = eng / self.eng_baseline if self.eng_baseline > 0 else 0.0
        err_norm = 1.0 - acc / self.acc_baseline if self.acc_baseline > 0 else 1.0
        return lat_norm, eng_norm, err_norm

    def get_statistics(self) -> Dict:
        return {
            "api_queries": self._query_count,
            "cache_hits": self._cache_hits,
            "cached_archs": len(self._hw_cache),
            "target_device": self.target_device,
            "dataset": self.dataset,
            "accuracy_enabled": self._acc_available,
            "weights": {
                "latency": round(self.w_lat, 4),
                "energy": round(self.w_eng, 4),
                "accuracy": round(self.w_acc, 4),
            },
        }

    def __repr__(self) -> str:
        acc_status = "ON" if self._acc_available else "OFF"
        return (
            "HardwareAwareFitness("
            f"device={self.target_device!r}, dataset={self.dataset!r}, "
            f"w_lat={self.w_lat:.2f}, w_eng={self.w_eng:.2f}, "
            f"w_acc={self.w_acc:.2f} [{acc_status}])"
        )
