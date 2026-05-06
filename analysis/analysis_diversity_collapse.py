#!/usr/bin/env python3
"""
Inter-seed solution diversity analysis.
Question: does MAgICoRe collapse diversity ACROSS seeds compared to BoN?
If so, "interaction erases diversity" extends beyond within-run to protocol-level.

Metrics:
- Flat-poly (coefficients): pairwise cosine distance between final coefficient vectors
- TSP-100 (tour): pairwise Jaccard distance between final edge sets
- Circle-packing (coordinates): pairwise L2 distance matrix mean
"""
import json
import numpy as np
from pathlib import Path
from itertools import combinations

RESULTS_DIR = Path("results/full-v2")


# ── Load helpers ─────────────────────────────────────────────────────────────

def load_solution_data(task_id: str, protocol_id: str) -> list[dict]:
    """Load all final solutionData for a task×protocol."""
    solutions = []
    for f in RESULTS_DIR.glob(f'{task_id}_{protocol_id}_s*.json'):
        try:
            data = json.loads(f.read_text())
            ba = data.get('bestArtifact') or {}
            sd = ba.get('solutionData')
            score = ba.get('devScore')
            if sd is not None:
                solutions.append({'seed': data.get('seed'), 'data': sd, 'score': score})
        except Exception:
            continue
    return sorted(solutions, key=lambda x: x['seed'])


# ── Flat-poly diversity ───────────────────────────────────────────────────────

def flat_poly_vector(sol: dict) -> np.ndarray | None:
    """Extract coefficient vector from flat-poly solution."""
    coeffs = sol.get('coefficients') or sol.get('polynomial') or sol.get('coeffs')
    if coeffs is None:
        # Try to find a list-like field
        for v in sol.values():
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                coeffs = v
                break
    if coeffs is None:
        return None
    v = np.array(coeffs, dtype=float)
    return v


def cosine_distances(vectors: list[np.ndarray]) -> list[float]:
    """Pairwise cosine distances (1 - cosine_similarity) for a list of vectors."""
    dists = []
    for a, b in combinations(vectors, 2):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            dists.append(1.0)
        else:
            sim = np.dot(a, b) / (na * nb)
            dists.append(1.0 - float(np.clip(sim, -1, 1)))
    return dists


# ── TSP diversity ─────────────────────────────────────────────────────────────

def tsp_edge_set(sol: dict) -> set | None:
    """Extract edge set from TSP solution (list of city indices)."""
    tour = sol.get('tour') or sol.get('route') or sol.get('path') or sol.get('cities')
    if tour is None:
        for v in sol.values():
            if isinstance(v, list) and len(v) > 10:
                tour = v
                break
    if tour is None or len(tour) < 2:
        return None
    edges = set()
    for i in range(len(tour)):
        a, b = tour[i], tour[(i+1) % len(tour)]
        edges.add((min(a, b), max(a, b)))
    return edges


def jaccard_distances(edge_sets: list[set]) -> list[float]:
    dists = []
    for a, b in combinations(edge_sets, 2):
        if not a and not b:
            dists.append(0.0)
        else:
            dists.append(1.0 - len(a & b) / len(a | b))
    return dists


# ── Circle-packing diversity ──────────────────────────────────────────────────

def circ_pack_vector(sol: dict) -> np.ndarray | None:
    """Flatten circle coordinates to a vector: [(x1,y1,r1), ...]."""
    circles = sol.get('circles') or sol.get('packing')
    if circles is None:
        return None
    flat = []
    for c in circles:
        if isinstance(c, (list, tuple)) and len(c) >= 2:
            flat.extend(c[:3])
    return np.array(flat, dtype=float) if flat else None


def l2_distances(vectors: list[np.ndarray]) -> list[float]:
    dists = []
    for a, b in combinations(vectors, 2):
        dists.append(float(np.linalg.norm(a - b)))
    return dists


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_task(task_id: str, proto_iter: str, proto_bon: str,
                 extract_fn, distance_fn, label: str):
    iter_sols = load_solution_data(task_id, proto_iter)
    bon_sols  = load_solution_data(task_id, proto_bon)

    iter_vecs = [v for s in iter_sols if (v := extract_fn(s['data'])) is not None]
    bon_vecs  = [v for s in bon_sols  if (v := extract_fn(s['data'])) is not None]

    if len(iter_vecs) < 2 or len(bon_vecs) < 2:
        print(f'  {label}: insufficient valid solutions (iter={len(iter_vecs)}, bon={len(bon_vecs)})')
        return

    iter_dists = distance_fn(iter_vecs)
    bon_dists  = distance_fn(bon_vecs)

    print(f'  {label}')
    print(f'    {proto_iter:30s}: mean_pairwise={np.mean(iter_dists):.4f}  std={np.std(iter_dists):.4f}  n_pairs={len(iter_dists)}')
    print(f'    {proto_bon:30s}: mean_pairwise={np.mean(bon_dists):.4f}  std={np.std(bon_dists):.4f}  n_pairs={len(bon_dists)}')

    ratio = np.mean(iter_dists) / np.mean(bon_dists) if np.mean(bon_dists) > 0 else float('nan')
    direction = 'LESS diverse' if ratio < 1 else 'MORE diverse'
    print(f'    iter/bon ratio = {ratio:.3f}  → {proto_iter} is {direction} than {proto_bon}')
    print()


print('\n══════════════════════════════════════════════════════════════════════')
print(' Inter-seed solution diversity: MAgICoRe vs BoN (do seeds converge?)')
print('══════════════════════════════════════════════════════════════════════\n')

print('── Flat-poly (cosine distance on coefficient vectors) ──')
analyze_task('bench-flat-poly-deg50', 'magicore-gemini', 'best-of-n-gemini',
             flat_poly_vector, cosine_distances, 'magicore-gemini vs BoN-Gemini')
analyze_task('bench-flat-poly-deg50', 'vgs-gemini', 'best-of-n-gemini',
             flat_poly_vector, cosine_distances, 'vgs-gemini vs BoN-Gemini')

print('── TSP-100 (Jaccard distance on edge sets) ──')
analyze_task('bench-tsp-100', 'magicore-gemini', 'best-of-n-gemini',
             tsp_edge_set, jaccard_distances, 'magicore-gemini vs BoN-Gemini')
analyze_task('bench-tsp-100', 'vgs-gemini', 'best-of-n-gemini',
             tsp_edge_set, jaccard_distances, 'vgs-gemini vs BoN-Gemini')

print('── Circle-packing (L2 on flattened coordinate vectors) ──')
analyze_task('bench-circle-packing-n20', 'magicore-gemini', 'best-of-n-gemini',
             circ_pack_vector, l2_distances, 'magicore-gemini vs BoN-Gemini')
analyze_task('bench-circle-packing-n20', 'vgs-gemini', 'best-of-n-gemini',
             circ_pack_vector, l2_distances, 'vgs-gemini vs BoN-Gemini')

print('── Molecule-QED (check solution field names first) ──')
# QED: solutionData likely has SMILES string or fingerprint
mol_sols = load_solution_data('bench-molecule-qed', 'magicore-gemini')
if mol_sols:
    print(f'  molecule-qed solutionData keys: {list(mol_sols[0]["data"].keys())[:8]}')
    print(f'  sample value: {str(list(mol_sols[0]["data"].values())[0])[:100]}')
