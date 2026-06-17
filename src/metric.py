"""Reward_{0.85} metric for the LLM-routing task.

Official definition (Kaggle):

    Reward_{0.85} = 0.85 * P_bar  -  0.15 * (C_bar / C_max)

where, over the set of routed queries:
    P_bar = mean performance of the chosen model per query
    C_bar = mean cost        of the chosen model per query
    C_max = a fixed cost normalizer (the one value not given in the spec).

The metric is linear and separable per query, so the per-query optimal choice is
the model maximizing  0.85*perf_m - (0.15/C_max)*cost_m, which empirically equals
"cheapest model among the top performers" (>=99.96% of train rows) and is
invariant to the exact value of C_max. We expose C_max as a parameter and default
it to the global max cost cell in the training matrix.
"""
from __future__ import annotations
import numpy as np

PERF_WEIGHT = 0.85
COST_WEIGHT = 0.15


def default_cmax(cost_matrix: np.ndarray) -> float:
    """Convention used for internal validation: global max cost cell."""
    return float(np.asarray(cost_matrix).max())


def reward(perf_chosen: np.ndarray, cost_chosen: np.ndarray, cmax: float) -> float:
    """Reward_{0.85} for a routing decision.

    perf_chosen / cost_chosen: 1-D arrays of the chosen model's perf/cost per query.
    """
    perf_chosen = np.asarray(perf_chosen, dtype=float)
    cost_chosen = np.asarray(cost_chosen, dtype=float)
    p_bar = perf_chosen.mean()
    c_bar = cost_chosen.mean()
    return PERF_WEIGHT * p_bar - COST_WEIGHT * (c_bar / cmax)


def reward_from_choice(perf_matrix: np.ndarray, cost_matrix: np.ndarray,
                       choice_idx: np.ndarray, cmax: float) -> float:
    """Reward given a per-query model index choice (0..K-1)."""
    perf_matrix = np.asarray(perf_matrix, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    rows = np.arange(len(choice_idx))
    return reward(perf_matrix[rows, choice_idx], cost_matrix[rows, choice_idx], cmax)


def oracle_choice(perf_matrix: np.ndarray, cost_matrix: np.ndarray, cmax: float) -> np.ndarray:
    """Per-query reward-maximizing model index (the achievable ceiling)."""
    perf_matrix = np.asarray(perf_matrix, dtype=float)
    cost_matrix = np.asarray(cost_matrix, dtype=float)
    score = PERF_WEIGHT * perf_matrix - (COST_WEIGHT / cmax) * cost_matrix
    return score.argmax(axis=1)
