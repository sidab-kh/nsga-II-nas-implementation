from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.fbnet import FBNET_SUPPORTED_DATASETS


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]

    if value == "{}":
        return {}

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    return value


def _load_simple_yaml(path: Path) -> Dict[str, Any]:
    lines: List[tuple[int, str]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        cleaned = _strip_comment(raw_line).rstrip()
        if not cleaned.strip():
            continue
        indent = len(cleaned) - len(cleaned.lstrip(" "))
        lines.append((indent, cleaned.lstrip(" ")))

    def parse_block(start: int, indent: int) -> tuple[Any, int]:
        if start >= len(lines):
            return {}, start

        if lines[start][0] != indent:
            raise ValueError(f"Invalid indentation in {path}")

        if lines[start][1].startswith("- "):
            items: List[Any] = []
            idx = start
            while idx < len(lines):
                current_indent, content = lines[idx]
                if current_indent < indent or not content.startswith("- "):
                    break
                if current_indent != indent:
                    raise ValueError(f"Invalid list indentation in {path}")

                remainder = content[2:].strip()
                idx += 1
                if remainder:
                    items.append(_parse_scalar(remainder))
                    continue

                if idx >= len(lines) or lines[idx][0] <= indent:
                    items.append(None)
                    continue

                item, idx = parse_block(idx, lines[idx][0])
                items.append(item)

            return items, idx

        mapping: Dict[str, Any] = {}
        idx = start
        while idx < len(lines):
            current_indent, content = lines[idx]
            if current_indent < indent or content.startswith("- "):
                break
            if current_indent != indent:
                raise ValueError(f"Invalid mapping indentation in {path}")
            if ":" not in content:
                raise ValueError(f"Invalid YAML line in {path}: {content}")

            key, remainder = content.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()
            idx += 1

            if remainder:
                mapping[key] = _parse_scalar(remainder)
                continue

            if idx >= len(lines) or lines[idx][0] <= indent:
                mapping[key] = {}
                continue

            value, idx = parse_block(idx, lines[idx][0])
            mapping[key] = value

        return mapping, idx

    if not lines:
        return {}

    parsed, next_index = parse_block(0, lines[0][0])
    if next_index != len(lines):
        raise ValueError(f"Could not fully parse YAML config {path}")
    if not isinstance(parsed, dict):
        raise ValueError(f"Top-level YAML object must be a mapping in {path}")
    return parsed


def load_yaml_file(path: str | Path) -> Dict[str, Any]:
    config_path = Path(path)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _load_simple_yaml(config_path)

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML object must be a mapping in {config_path}")
    return data


@dataclass
class PathsConfig:
    pickle: str = "HW-NAS-Bench-v1_0.pickle"
    nb201: Optional[str] = None


@dataclass
class SearchConfig:
    search_space: str = "nasbench201"
    device: str = "edgegpu"
    dataset: str = "cifar10"
    algorithms: List[str] = field(default_factory=lambda: ["EE-TGA", "E3-FA"])


@dataclass
class RunnerConfig:
    runs: int = 10
    population_size: int = 20
    max_iterations: int = 100
    seed: int = 42
    quiet: bool = False


@dataclass
class FitnessConfig:
    latency_weight: float = 0.5
    energy_weight: float = 0.2
    accuracy_weight: float = 0.3


@dataclass
class OutputConfig:
    directory: str = "./outputs"
    run_name: Optional[str] = None


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    extra_kwargs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(load_yaml_file(path))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        return cls(
            paths=PathsConfig(**data.get("paths", {})),
            search=SearchConfig(**data.get("search", {})),
            runner=RunnerConfig(**data.get("runner", {})),
            fitness=FitnessConfig(**data.get("fitness", {})),
            output=OutputConfig(**data.get("output", {})),
            extra_kwargs=data.get("extra_kwargs", {}),
        )

    def apply_overrides(self, overrides: Dict[str, Any]) -> "ExperimentConfig":
        for section_name in ("paths", "search", "runner", "fitness", "output"):
            section_overrides = overrides.get(section_name, {})
            section = getattr(self, section_name)
            for key, value in section_overrides.items():
                if value is not None:
                    setattr(section, key, value)

        extra_kwargs = overrides.get("extra_kwargs")
        if extra_kwargs is not None:
            self.extra_kwargs = extra_kwargs

        return self

    def normalize(self) -> "ExperimentConfig":
        if self.search.search_space == "fbnet" and self.fitness.accuracy_weight > 0:
            hw_total = self.fitness.latency_weight + self.fitness.energy_weight
            if hw_total <= 0:
                raise ValueError(
                    "fbnet runs require at least one positive hardware weight."
                )
            self.fitness.latency_weight /= hw_total
            self.fitness.energy_weight /= hw_total
            self.fitness.accuracy_weight = 0.0
        return self

    def validate(self) -> None:
        self.normalize()
        total = (
            self.fitness.latency_weight
            + self.fitness.energy_weight
            + self.fitness.accuracy_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"Fitness weights must sum to 1.0, got {total:.6f}."
            )
        if (
            self.search.search_space == "fbnet"
            and self.search.dataset not in FBNET_SUPPORTED_DATASETS
        ):
            raise ValueError(
                f"fbnet is only supported for datasets {FBNET_SUPPORTED_DATASETS}, "
                f"got {self.search.dataset!r}."
            )
        if self.search.search_space == "fbnet" and self.search.device == "edgetpu":
            raise ValueError(
                "fbnet does not include EdgeTPU measurements in HW-NAS-Bench."
            )
        if (
            self.search.search_space == "nasbench201"
            and self.fitness.accuracy_weight > 0
            and not self.paths.nb201
        ):
            raise ValueError(
                "accuracy_weight > 0 requires paths.nb201 or --nb201."
            )
        if not self.search.algorithms:
            raise ValueError("At least one algorithm must be configured.")
