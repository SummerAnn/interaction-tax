#!/usr/bin/env python3
"""
Diversity analysis on constraint tasks.
Question: does MAgICoRe also collapse diversity on constraint tasks?
If yes + constraint tasks still win → diversity collapse is not the bottleneck there.
If no → MAgICoRe preserves diversity differently on constraint tasks.

Completes the story from analysis_diversity_collapse.py (which only did optimization tasks).

Diversity metrics:
- knapsack-50: pairwise Hamming distance on item selection binary vector
- 3ap-free-100: pairwise Jaccard distance on integer sets
- circle-packing-n20: pairwise mean L2 distance on circle position vectors (already done, re-running for consistency)
"""
import json
import numpy as np
from pathlib import Path
from itertools import combinations

RESULTS_DIR = Path("results/full-v2")

CONSTRAINT_TASKS = [
    ('bench-knapsack-50',        'knapsack-50',       'hamming'),
    ('bench-3ap-free-100',       '3ap-free-100',      'jaccard'),
    ('bench-circle-packing-n20', 'circle-packing',    'l2'),
]

OPTIMIZATION_TASKS = [
    ('bench-flat-poly-deg50', 'flat-poly', 'cosine'),
    ('bench-tsp-100',         'tsp-100',   'jaccard'),
]


def load_solutions(task_id: str, protocol_id: str) -> list[dict]:
    sols = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_{protocol_id}_s*.json')):
        try:
            data = json.loads(f.read_text())
            ba = data.get('bestArtifact') or {}
            sd = ba.get('solutionData')
            score = ba.get('devScore')
            if sd is not None:
                sols.append({'seed': data.get('seed'), 'data': sd, 'score': score})
        except Exception:
            continue
    return sorted(sols, key=lambda x: x.get('seed', 0))


# ── Diversity metrics ─────────────────────────────────────────────────────────

def knapsack_vector(sol: dict) -> np.ndarray | None:
    """Extract binary item selection vector from knapsack solution."""
    data = sol.get('data', {})
    items = data.get('selected_items') or data.get('items') or data.get('selection')
    if items is None:
        # Try any list of booleans or 0/1
        for v in data.values():
            if isinstance(v, list) and all(isinstance(x, (bool, int)) for x in v):
                items = v
                break
    if items is None:
        return None
    return np.array([int(bool(x)) for x in items], dtype=float)


def hamming_distances(vecs: list[np.ndarray]) -> list[float]:
    dists = []
    for a, b in combinations(vecs, 2):
        if len(a) != len(b):
            continue
        dists.append(np.sum(a != b) / len(a))
    return dists


def ap_free_set(sol: dict) -> set | None:
    """Extract set of integers from 3ap-free solution."""
    data = sol.get('data', {})
    s = data.get('set') or data.get('elements') or data.get('sequence') or data.get('numbers')
    if s is None:
        for v in data.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                s = v
                break
    if s is None:
        return None
    return set(int(x) for x in s)


def jaccard_distances(sets: list[set]) -> list[float]:
    dists = []
    for a, b in combinations(sets, 2):
        union = len(a | b)
        if union == 0:
            continue
        dists.append(1.0 - len(a & b) / union)
    return dists


def circle_vectors(sol: dict) -> np.ndarray | None:
    """Extract flattened circle positions."""
    data = sol.get('data', {})
    circles = data.get('circles')
    if circles is None:
        return None
    flat = []
    for c in circles:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            flat.extend(c[:2])
    return np.array(flat, dtype=float) if flat else None


def l2_distances(vecs: list[np.ndarray]) -> list[float]:
    dists = []
    for a, b in combinations(vecs, 2):
        if len(a) != len(b):
            continue
        dists.append(float(np.linalg.norm(a - b)))
    return dists


def cosine_distances(vecs: list[np.ndarray]) -> list[float]:
    dists = []
    for a, b in combinations(vecs, 2):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            continue
        cos_sim = np.dot(a, b) / (na * nb)
        dists.append(1.0 - float(np.clip(cos_sim, -1, 1)))
    return dists


def tsp_edge_set(sol: dict) -> set | None:
    """Extract edge set from TSP tour."""
    data = sol.get('data', {})
    tour = data.get('tour') or data.get('route') or data.get('path')
    if tour is None:
        return None
    n = len(tour)
    edges = set()
    for i in range(n):
        a, b = int(tour[i]), int(tour[(i+1) % n])
        edges.add((min(a, b), max(a, b)))
    return edges


def compute_diversity(task_id: str, metric: str, protocol_id: str) -> float | None:
    sols = load_solutions(task_id, protocol_id)
    if len(sols) < 2:
        return None

    if metric == 'hamming':
        vecs = [knapsack_vector(s) for s in sols]
        vecs = [v for v in vecs if v is not None]
        dists = hamming_distances(vecs)
    elif metric == 'jaccard' and 'tsp' in task_id:
        sets = [tsp_edge_set(s) for s in sols]
        sets = [s for s in sets if s is not None]
        dists = jaccard_distances(sets)
    elif metric == 'jaccard':
        sets = [ap_free_set(s) for s in sols]
        sets = [s for s in sets if s is not None]
        dists = jaccard_distances(sets)
    elif metric == 'l2':
        vecs = [circle_vectors(s) for s in sols]
        vecs = [v for v in vecs if v is not None]
        dists = l2_distances(vecs)
    elif metric == 'cosine':
        from analysis_diversity_collapse import flat_poly_vector
        vecs = [flat_poly_vector(s['data']) for s in sols]
        vecs = [v for v in vecs if v is not None]
        dists = cosine_distances(vecs)
    else:
        return None

    return float(np.mean(dists)) if dists else None


def print_diversity_table():
    print('\n' + '='*70)
    print('DIVERSITY ANALYSIS: MAgICoRe vs BoN — mean pairwise distance')
    print('ratio < 1.0 → MAgICoRe solutions are less diverse (collapse)')
    print('='*70)
    print(f'{"Task":<22} {"Type":<14} {"MAgICoRe":>12} {"BoN":>12} {"Ratio":>8}  Interpretation')
    print('-'*70)

    all_tasks = [(t, l, m, 'constraint') for t, l, m in CONSTRAINT_TASKS] + \
                [(t, l, m, 'optimization') for t, l, m in OPTIMIZATION_TASKS]

    for task_id, label, metric, ttype in all_tasks:
        mag_div = compute_diversity(task_id, metric, 'magicore-gemini')
        bon_div = compute_diversity(task_id, metric, 'best-of-n-gemini')

        if mag_div is None or bon_div is None or bon_div == 0:
            print(f'{label:<22} {ttype:<14} {"—":>12} {"—":>12} {"—":>8}  no data')
            continue

        ratio = mag_div / bon_div
        if ratio < 0.9:
            interp = 'COLLAPSE'
        elif ratio > 1.05:
            interp = 'more diverse'
        else:
            interp = 'similar'

        print(f'{label:<22} {ttype:<14} {mag_div:>12.4f} {bon_div:>12.4f} {ratio:>8.3f}  {interp}')

    print('='*70)
    print()
    print('KEY QUESTION: If constraint tasks also show diversity collapse but still win,')
    print('then diversity collapse is NOT the bottleneck — constraint repair is.')
    print('If constraint tasks preserve diversity, the mechanism differs.')


if __name__ == '__main__':
    try:
        print_diversity_table()
    except ImportError:
        # flat-poly cosine needs analysis_diversity_collapse.flat_poly_vector
        # run without it if not available
        OPTIMIZATION_TASKS.clear()
        print('(flat-poly skipped — import analysis_diversity_collapse to include it)')
        print_diversity_table()
