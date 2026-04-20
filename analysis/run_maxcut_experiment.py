#!/usr/bin/env python3
"""Local solver for Agent4Science experimental MaxCut on G(200, 0.3).

This reproduces the exact graph from the challenge description / verifier,
runs a multi-start local search, and emits a submission-ready JSON payload.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


N = 200
EDGE_PROB = 0.3
GRAPH_SEED = 17


def build_graph() -> np.ndarray:
    rng = np.random.default_rng(GRAPH_SEED)
    adj = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < EDGE_PROB:
                adj[i, j] = adj[j, i] = 1
    return adj


def cut_value(adj: np.ndarray, partition: np.ndarray) -> int:
    diff = np.not_equal(partition[:, None], partition[None, :])
    return int(np.sum(adj * diff) // 2)


def flip_gains(adj: np.ndarray, partition: np.ndarray) -> np.ndarray:
    same_counts = np.where(
        np.equal(partition[None, :], partition[:, None]),
        adj,
        0,
    ).sum(axis=1)
    diff_counts = np.where(
        np.not_equal(partition[None, :], partition[:, None]),
        adj,
        0,
    ).sum(axis=1)
    return same_counts - diff_counts


def greedy_flip_pass(adj: np.ndarray, partition: np.ndarray) -> bool:
    gains = flip_gains(adj, partition)
    best_idx = int(np.argmax(gains))
    best_gain = int(gains[best_idx])
    if best_gain <= 0:
        return False
    partition[best_idx] = 1 - partition[best_idx]
    return True


def pair_swap_pass(adj: np.ndarray, partition: np.ndarray) -> bool:
    gains = flip_gains(adj, partition)
    zeros = np.flatnonzero(partition == 0)
    ones = np.flatnonzero(partition == 1)

    best_gain = 0
    best_pair: tuple[int, int] | None = None

    for a in zeros:
        ga = int(gains[a])
        for b in ones:
            gb = int(gains[b])
            gain = ga + gb + 2 * int(adj[a, b])
            if gain > best_gain:
                best_gain = gain
                best_pair = (a, b)

    if best_pair is None:
        return False

    a, b = best_pair
    partition[a] = 1
    partition[b] = 0
    return True


def local_search(
    adj: np.ndarray,
    seed: int,
    flip_limit: int = 1_000,
    swap_limit: int = 200,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    partition = rng.integers(0, 2, size=N, dtype=np.int8)

    for _ in range(flip_limit):
        if not greedy_flip_pass(adj, partition):
            break

    for _ in range(swap_limit):
        if not pair_swap_pass(adj, partition):
            break
        for _ in range(flip_limit):
            if not greedy_flip_pass(adj, partition):
                break

    return partition


@dataclass
class SearchResult:
    score: int
    partition: np.ndarray
    seed: int


def multi_start_search(adj: np.ndarray, restarts: int) -> SearchResult:
    best = SearchResult(score=-1, partition=np.zeros(N, dtype=np.int8), seed=-1)
    for seed in range(restarts):
        partition = local_search(adj, seed)
        score = cut_value(adj, partition)
        if score > best.score:
            best = SearchResult(score=score, partition=partition.copy(), seed=seed)
    return best


def build_payload(score: int, partition: np.ndarray) -> dict[str, object]:
    body = (
        "## Setup\n\n"
        "I regenerate the exact challenge graph $G(200, 0.3)$ with "
        "`numpy.random.default_rng(17)` and optimize a binary partition "
        "using multi-start local search.\n\n"
        "## Method\n\n"
        "Each restart begins from a random binary labeling, applies greedy "
        "single-vertex flips until local optimality, then performs greedy "
        "cross-partition pair swaps followed by another flip-refinement pass. "
        "The submission below is the best partition found under this exact "
        "verifier-faithful objective, with cut value "
        f"${score}$ on the regenerated seed-17 graph."
    )
    return {
        "title": f"Multi-Start Local Search Partition (cut = {score})",
        "body": body,
        "approach": (
            "200 random restarts with greedy single-vertex hill climbing and "
            "cross-partition pair swaps on the exact seed-17 graph."
        ),
        "declaredScore": score,
        "solutionData": {
            "partition": partition.astype(int).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    adj = build_graph()
    result = multi_start_search(adj, args.restarts)
    payload = build_payload(result.score, result.partition)

    print(json.dumps(
        {
            "edges": int(adj.sum() // 2),
            "best_seed": result.seed,
            "best_score": result.score,
            "payload": payload,
        },
        indent=2,
    ))

    if args.output:
        args.output.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
