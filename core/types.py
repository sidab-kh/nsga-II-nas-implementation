"""
core/types.py
=============
Shared data types, enums, and constants for HW-NAS-Bench experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NASBENCH201_SIZE = 15625
FBNET_SEARCH_SPACE = "fbnet"
NB201_SEARCH_SPACE = "nasbench201"

VALID_DATASETS = ("cifar10", "cifar100", "ImageNet16-120")
VALID_DEVICES = ("edgegpu", "raspi4", "edgetpu", "pixel3", "eyeriss", "fpga")


# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

class SearchSpace(str, Enum):
    NASBENCH201 = "nasbench201"
    FBNET = "fbnet"


# ---------------------------------------------------------------------------
# Architecture representation
# ---------------------------------------------------------------------------

@dataclass
class NASBench201Architecture:
    """
    Represents one architecture in either supported search space.

    NAS-Bench-201 uses a single integer index. FBNet uses a length-22 vector
    of categorical block choices.
    """

    arch_idx: Optional[int] = None
    encoding: Optional[Tuple[int, ...]] = None
    search_space: SearchSpace = SearchSpace.NASBENCH201

    @classmethod
    def from_index(cls, idx: int) -> "NASBench201Architecture":
        idx = int(np.clip(idx, 0, NASBENCH201_SIZE - 1))
        return cls(
            arch_idx=idx,
            encoding=None,
            search_space=SearchSpace.NASBENCH201,
        )

    @classmethod
    def from_vector(
        cls,
        vector: np.ndarray,
        *,
        search_space: SearchSpace = SearchSpace.NASBENCH201,
        search_space_size: int = NASBENCH201_SIZE,
    ) -> "NASBench201Architecture":
        arr = np.asarray(vector, dtype=np.int64).reshape(-1)
        if search_space == SearchSpace.FBNET:
            clipped = np.clip(arr, 0, search_space_size - 1).astype(np.int64)
            return cls(
                arch_idx=None,
                encoding=tuple(int(v) for v in clipped.tolist()),
                search_space=SearchSpace.FBNET,
            )
        raw = float(arr[0]) if len(arr) >= 1 else 0.0
        return cls.from_index(int(round(raw)))

    @classmethod
    def random(
        cls,
        rng: Optional[np.random.Generator] = None,
        *,
        search_space: SearchSpace = SearchSpace.NASBENCH201,
        dim: int = 1,
        search_space_size: int = NASBENCH201_SIZE,
    ) -> "NASBench201Architecture":
        if rng is None:
            rng = np.random.default_rng()
        if search_space == SearchSpace.FBNET:
            return cls.from_vector(
                rng.integers(0, search_space_size, size=dim),
                search_space=search_space,
                search_space_size=search_space_size,
            )
        return cls(arch_idx=int(rng.integers(0, NASBENCH201_SIZE)))

    def to_index(self) -> int:
        if self.arch_idx is None:
            raise ValueError("This architecture is not represented by a single index.")
        return self.arch_idx

    def to_vector(self) -> np.ndarray:
        if self.encoding is not None:
            return np.array(self.encoding, dtype=np.int64)
        if self.arch_idx is None:
            return np.empty(0, dtype=np.int64)
        return np.array([self.arch_idx], dtype=np.int64)

    def cache_key(self) -> Tuple[str, int | Tuple[int, ...]]:
        if self.search_space == SearchSpace.FBNET:
            if self.encoding is None:
                raise ValueError("FBNet architecture is missing its encoding.")
            return (self.search_space.value, self.encoding)
        if self.arch_idx is None:
            raise ValueError("NAS-Bench-201 architecture is missing its index.")
        return (self.search_space.value, self.arch_idx)

    def label(self) -> str:
        if self.search_space == SearchSpace.FBNET:
            return f"arch={list(self.encoding or [])}"
        return f"arch_idx={self.arch_idx}"

    def __repr__(self) -> str:
        if self.search_space == SearchSpace.FBNET:
            return (
                "NASBench201Architecture("
                f"search_space={self.search_space.value!r}, "
                f"encoding={list(self.encoding or [])})"
            )
        return (
            "NASBench201Architecture("
            f"search_space={self.search_space.value!r}, "
            f"arch_idx={self.arch_idx})"
        )


# ---------------------------------------------------------------------------
# Hardware metrics container
# ---------------------------------------------------------------------------

@dataclass
class HardwareMetrics:
    """
    Hardware performance metrics retrieved from HW-NAS-Bench.

    Units
    -----
    latency : milliseconds (ms)
    energy  : millijoules (mJ)
    arithmetic_intensity : ops / byte  (Eyeriss only)
    """

    edgegpu_latency: float
    raspi4_latency: float
    edgetpu_latency: float
    pixel3_latency: float
    eyeriss_latency: float
    fpga_latency: float

    edgegpu_energy: Optional[float] = None
    eyeriss_energy: Optional[float] = None
    fpga_energy: Optional[float] = None
    eyeriss_arithmetic_intensity: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "HardwareMetrics":
        return cls(
            edgegpu_latency=float(d.get("edgegpu_latency", 0.0)),
            raspi4_latency=float(d.get("raspi4_latency", 0.0)),
            edgetpu_latency=float(d.get("edgetpu_latency", 0.0)),
            pixel3_latency=float(d.get("pixel3_latency", 0.0)),
            eyeriss_latency=float(d.get("eyeriss_latency", 0.0)),
            fpga_latency=float(d.get("fpga_latency", 0.0)),
            edgegpu_energy=d.get("edgegpu_energy"),
            eyeriss_energy=d.get("eyeriss_energy"),
            fpga_energy=d.get("fpga_energy"),
            eyeriss_arithmetic_intensity=d.get("eyeriss_arithmetic_intensity"),
        )

    _LATENCY_MAP = {
        "edgegpu": "edgegpu_latency",
        "raspi4": "raspi4_latency",
        "edgetpu": "edgetpu_latency",
        "pixel3": "pixel3_latency",
        "eyeriss": "eyeriss_latency",
        "fpga": "fpga_latency",
    }
    _ENERGY_MAP = {
        "edgegpu": "edgegpu_energy",
        "eyeriss": "eyeriss_energy",
        "fpga": "fpga_energy",
    }

    def get_latency(self, device: str) -> float:
        attr = self._LATENCY_MAP.get(device)
        if attr is None:
            raise ValueError(f"Unknown device '{device}'. Valid: {list(self._LATENCY_MAP)}")
        return getattr(self, attr)

    def get_energy(self, device: str) -> Optional[float]:
        attr = self._ENERGY_MAP.get(device)
        return getattr(self, attr, None) if attr else None

    def to_dict(self) -> Dict:
        return asdict(self)

    def __repr__(self) -> str:
        parts = [
            f"edgegpu={self.edgegpu_latency:.2f}ms",
            f"raspi4={self.raspi4_latency:.2f}ms",
            f"fpga={self.fpga_latency:.2f}ms",
        ]
        return f"HardwareMetrics({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Run result
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Stores the outcome of a single optimizer run."""

    run_id: int
    algorithm_name: str
    best_arch_idx: int
    best_arch_vector: Optional[List[int]]
    best_fitness: float
    fitness_history: List[float]
    hardware_metrics: Optional[HardwareMetrics]
    query_count: int
    convergence_iter: Optional[int] = None

    def is_valid(self) -> bool:
        return self.best_fitness > -np.inf

    def to_dict(self) -> Dict:
        return {
            "run_id": self.run_id,
            "algorithm_name": self.algorithm_name,
            "best_arch_idx": self.best_arch_idx,
            "best_arch_vector": self.best_arch_vector,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
            "query_count": self.query_count,
            "convergence_iter": self.convergence_iter,
            "hardware_metrics": self.hardware_metrics.to_dict()
            if self.hardware_metrics
            else None,
        }
