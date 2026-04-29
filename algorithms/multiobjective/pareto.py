from __future__ import annotations

from typing import List

import numpy as np


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if objective vector a Pareto-dominates b (minimization)."""
    return bool(np.all(a <= b) and np.any(a < b))


def fast_non_dominated_sort(objectives: np.ndarray) -> List[np.ndarray]:
    """Compute Pareto fronts using NSGA-II fast non-dominated sorting."""
    n = objectives.shape[0]
    dominated_sets: List[List[int]] = [[] for _ in range(n)]
    domination_count = np.zeros(n, dtype=int)
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objectives[p], objectives[q]):
                dominated_sets[p].append(q)
            elif dominates(objectives[q], objectives[p]):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        next_front: List[int] = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    return [np.array(front, dtype=int) for front in fronts if front]


def crowding_distance(front_objectives: np.ndarray) -> np.ndarray:
    """Compute crowding distance for one front (higher is better)."""
    m = front_objectives.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    if m <= 2:
        return np.full(m, np.inf, dtype=float)

    distances = np.zeros(m, dtype=float)
    n_obj = front_objectives.shape[1]

    for obj_idx in range(n_obj):
        order = np.argsort(front_objectives[:, obj_idx])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf

        min_val = front_objectives[order[0], obj_idx]
        max_val = front_objectives[order[-1], obj_idx]
        denom = max_val - min_val
        if denom <= 1e-12:
            continue

        for i in range(1, m - 1):
            left_val = front_objectives[order[i - 1], obj_idx]
            right_val = front_objectives[order[i + 1], obj_idx]
            distances[order[i]] += (right_val - left_val) / denom

    return distances


def environmental_selection(
    population: np.ndarray,
    objectives: np.ndarray,
    target_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select next generation population using NSGA-II environmental selection."""
    fronts = fast_non_dominated_sort(objectives)

    selected_indices: List[int] = []
    for front in fronts:
        remaining = target_size - len(selected_indices)
        if remaining <= 0:
            break

        if len(front) <= remaining:
            selected_indices.extend(front.tolist())
            continue

        front_obj = objectives[front]
        distances = crowding_distance(front_obj)
        order = np.argsort(distances)[::-1]
        selected_indices.extend(front[order[:remaining]].tolist())
        break

    sel = np.array(selected_indices, dtype=int)
    return population[sel].copy(), objectives[sel].copy()


def extract_nondominated(
    population: np.ndarray,
    objectives: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first Pareto front of a population/objective matrix."""
    fronts = fast_non_dominated_sort(objectives)
    if not fronts:
        return np.empty((0, population.shape[1]), dtype=int), np.empty((0, objectives.shape[1]))
    first = fronts[0]
    return population[first].copy(), objectives[first].copy()
