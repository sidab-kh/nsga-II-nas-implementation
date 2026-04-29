"""
core/api.py
===========
Thin, robust wrapper around the HW-NAS-Bench pickle database.
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.fbnet import (
    FBNET_CHOICE_COUNT,
    FBNET_SEARCH_DIM,
    FBNET_SUPPORTED_DATASETS,
    fbnet_architecture_to_keys,
)
from core.types import NASBench201Architecture, SearchSpace, VALID_DATASETS


_NB201_METRICS: List[str] = [
    "edgegpu_latency",
    "edgegpu_energy",
    "raspi4_latency",
    "edgetpu_latency",
    "pixel3_latency",
    "eyeriss_latency",
    "eyeriss_energy",
    "eyeriss_arithmetic_intensity",
    "fpga_latency",
    "fpga_energy",
]

_FBNET_METRICS: List[str] = [
    "edgegpu_latency",
    "edgegpu_energy",
    "raspi4_latency",
    "pixel3_latency",
    "eyeriss_latency",
    "eyeriss_energy",
    "fpga_latency",
    "fpga_energy",
]


class HWNASBenchAPI:
    """Hardware-aware NAS Benchmark API."""

    def __init__(
        self,
        file_path: str | Path,
        search_space: str = SearchSpace.NASBENCH201,
    ) -> None:
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"HW-NAS-Bench pickle not found: {file_path}")

        with open(file_path, "rb") as fh:
            # NumPy 2.4 emits a deprecation warning while unpickling this
            # legacy benchmark file; the data still loads correctly.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
                    category=Warning,
                )
                self._data: Dict[str, Any] = pickle.load(fh)

        self.search_space = SearchSpace(search_space)
        self._validate_search_space()

    def _validate_search_space(self) -> None:
        if self.search_space not in self._data:
            available = list(self._data.keys())
            raise KeyError(
                f"Search space '{self.search_space}' not found in pickle. "
                f"Available: {available}"
            )

    def _validate_dataset(self, dataname: str) -> None:
        if self.search_space == SearchSpace.FBNET and dataname not in FBNET_SUPPORTED_DATASETS:
            raise ValueError(
                f"FBNet is only supported for {FBNET_SUPPORTED_DATASETS}, got {dataname!r}."
            )
        if dataname not in VALID_DATASETS:
            warnings.warn(
                f"Dataset '{dataname}' is not in the standard set {VALID_DATASETS}. "
                "Proceeding anyway; your pickle may support additional datasets.",
                UserWarning,
                stacklevel=3,
            )

    def query_by_index(
        self,
        arch_index: int,
        dataname: str = "cifar10",
    ) -> Optional[Dict[str, float]]:
        self._validate_dataset(dataname)
        try:
            if self.search_space == SearchSpace.NASBENCH201:
                return self._query_nb201(arch_index, dataname)
            raise NotImplementedError(
                "FBNet uses vector encodings in this codebase. Use query_by_architecture()."
            )
        except (KeyError, IndexError, TypeError) as exc:
            warnings.warn(
                f"query_by_index({arch_index}, {dataname!r}) failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def query_by_architecture(
        self,
        architecture: NASBench201Architecture,
        dataname: str = "cifar10",
    ) -> Optional[Dict[str, float]]:
        self._validate_dataset(dataname)
        try:
            if self.search_space == SearchSpace.NASBENCH201:
                if architecture.arch_idx is None:
                    raise ValueError("NAS-Bench-201 queries require arch_idx.")
                return self._query_nb201(architecture.arch_idx, dataname)
            if architecture.encoding is None:
                raise ValueError("FBNet queries require a vector encoding.")
            return self._query_fbnet(architecture.encoding, dataname)
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            warnings.warn(
                f"query_by_architecture({architecture!r}, {dataname!r}) failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return None

    def get_net_config(
        self,
        arch_index: int,
        dataname: str = "cifar10",
    ) -> Optional[Dict]:
        try:
            if self.search_space == SearchSpace.NASBENCH201:
                return self._data[self.search_space][dataname]["config"][arch_index]
            return None
        except (KeyError, IndexError):
            return None

    def search_space_size(self, dataname: str = "cifar10") -> int:
        if self.search_space == SearchSpace.NASBENCH201:
            return 15625
        self._validate_dataset(dataname)
        return FBNET_CHOICE_COUNT

    def search_dimension(self, dataname: str = "cifar10") -> int:
        if self.search_space == SearchSpace.NASBENCH201:
            return 1
        self._validate_dataset(dataname)
        return FBNET_SEARCH_DIM

    def available_datasets(self) -> List[str]:
        if self.search_space == SearchSpace.NASBENCH201:
            try:
                return [
                    k
                    for k in self._data[self.search_space]
                    if isinstance(self._data[self.search_space][k], dict)
                    and "edgegpu_latency" in self._data[self.search_space][k]
                ]
            except KeyError:
                return []
        return list(FBNET_SUPPORTED_DATASETS)

    def _query_nb201(self, arch_index: int, dataname: str) -> Dict[str, float]:
        space_data = self._data[self.search_space][dataname]
        results: Dict[str, float] = {}

        for metric in _NB201_METRICS:
            if metric in space_data:
                results[metric] = float(space_data[metric][arch_index])

        results["average_hw_metric"] = self._composite_metric(results)
        return results

    def _query_fbnet(
        self,
        encoding: tuple[int, ...],
        dataname: str,
    ) -> Dict[str, float]:
        space_data = self._data[self.search_space]
        op_keys = fbnet_architecture_to_keys(encoding, dataname)
        results: Dict[str, float] = {}

        for metric in _FBNET_METRICS:
            table = space_data.get(metric)
            if table is None:
                continue
            total = 0.0
            for op_key in op_keys:
                total += float(table[op_key])
            results[metric] = total

        results["average_hw_metric"] = self._composite_metric(results)
        return results

    @staticmethod
    def _composite_metric(results: Dict[str, float]) -> float:
        product = 1.0
        for key, val in results.items():
            if val and ("latency" in key or "energy" in key):
                product *= val
        return product

    def __repr__(self) -> str:
        if self.search_space == SearchSpace.FBNET:
            return (
                "HWNASBenchAPI("
                f"search_space={self.search_space!r}, "
                f"dim={FBNET_SEARCH_DIM}, "
                f"choices={FBNET_CHOICE_COUNT})"
            )
        return (
            "HWNASBenchAPI("
            f"search_space={self.search_space!r}, "
            f"size={self.search_space_size()})"
        )
