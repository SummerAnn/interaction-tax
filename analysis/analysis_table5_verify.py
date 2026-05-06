#!/usr/bin/env python3
"""
Analysis 3: Verify Table 5 — mean pairwise solution distance.

Paper Table 5:
  MoA (no synthesis)          0.48
  Best-of-N (same model)      0.40
  Debate                      0.38
  MoA (diverse + synthesis)   0.34
  MoA (same-model)            0.34

Metric: Binary tasks use Hamming distance; continuous tasks use cosine distance.
"""
import json
import numpy as np
from pathlib import Path
from itertools import combinations

RESULTS_DIR = Path("results/full-v2")

INVALID_THRESHOLD = 1e10

# Task type classification
BINARY_TASKS = {
    'bench-erdos-overlap':    'values',       # binary indicator vector
    'bench-maxcut-500':       None,           # try auto-detect
    'bench-difference-bases': 'set',          # convert set to indicator
}
CONTINUOUS_TASKS = {
    'bench-circle-packing-n20': 'circles',    # coordinate vectors
    'bench-tsp-50':             None,         # tour permutation
    'bench-tsp-100':            None,
    'bench-flat-poly-deg50':    None,
    'bench-mol-qed':            None,
    'bench-lj-cluster-n41':     None,
}

ALL_TASKS = list(BINARY_TASKS) + list(CONTINUOUS_TASKS)


def get_solution_vec(solutionData: dict, task_id: str) -> np.ndarray | None:
    if not solutionData or not isinstance(solutionData, dict):
        return None

    if task_id == 'bench-erdos-overlap':
        v = solutionData.get('values')
        if v and len(v) == 100:
            return np.array(v, dtype=float)

    elif task_id == 'bench-maxcut-500':
        v = solutionData.get('assignment') or solutionData.get('cut') or solutionData.get('labels')
        if v and isinstance(v, list):
            return np.array(v[:500], dtype=float)

    elif task_id == 'bench-difference-bases':
        s = solutionData.get('set')
        if s and isinstance(s, list):
            n = 600  # fixed domain size
            indicator = np.zeros(n)
            for x in s:
                if isinstance(x, int) and 0 <= x < n:
                    indicator[x] = 1.0
            return indicator

    elif task_id == 'bench-circle-packing-n20':
        circles = solutionData.get('circles')
        if circles:
            # flatten [x,y,r] × n_circles
            flat = []
            for c in circles[:20]:
                if isinstance(c, list) and len(c) >= 2:
                    flat.extend(c[:2])  # just x, y
            if flat:
                return np.array(flat, dtype=float)

    elif task_id in ('bench-tsp-50', 'bench-tsp-100'):
        tour = solutionData.get('tour') or solutionData.get('path')
        if tour and isinstance(tour, list):
            return np.array(tour, dtype=float)

    elif task_id == 'bench-mol-qed':
        smiles = solutionData.get('smiles') or solutionData.get('molecule')
        if smiles and isinstance(smiles, str):
            # convert to a simple hash-based fingerprint
            chars = [ord(c) % 32 for c in smiles[:128]]
            vec = np.zeros(128)
            vec[:len(chars)] = chars
            return vec

    elif task_id == 'bench-lj-cluster-n41':
        coords = solutionData.get('positions') or solutionData.get('coords')
        if coords and isinstance(coords, list):
            flat = []
            for pt in coords[:41]:
                if isinstance(pt, list):
                    flat.extend(pt[:3])
            if flat:
                return np.array(flat, dtype=float)

    elif task_id == 'bench-flat-poly-deg50':
        coeffs = solutionData.get('coefficients') or solutionData.get('coeff')
        if coeffs and isinstance(coeffs, list):
            return np.array(coeffs[:51], dtype=float)

    return None


def pairwise_distance(vecs: list[np.ndarray], task_id: str) -> list[float]:
    """Compute pairwise distances."""
    distances = []
    is_binary = task_id in BINARY_TASKS
    for i, j in combinations(range(len(vecs)), 2):
        v1, v2 = vecs[i], vecs[j]
        if len(v1) != len(v2):
            # skip mismatched lengths
            continue
        if is_binary:
            # Hamming distance (normalized)
            d = float(np.mean(v1 != v2))
        else:
            # cosine distance
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                d = float(np.mean(v1 != v2))
            else:
                cos_sim = np.dot(v1, v2) / (n1 * n2)
                d = 1.0 - float(cos_sim)
        distances.append(d)
    return distances


def compute_protocol_distance(protocol: str, proposer_only: bool = False,
                               tasks=None) -> dict:
    """
    Compute mean pairwise solution distance for a protocol.
    proposer_only: if True, exclude last artifact (synthesizer) for MoA.
    """
    if tasks is None:
        tasks = ALL_TASKS

    all_distances = []
    task_results = {}

    for task_id in tasks:
        distances = []
        files = sorted(RESULTS_DIR.glob(f'{task_id}_{protocol}_s*.json'))
        if not files:
            continue

        for f in files:
            data = json.loads(f.read_text())
            arts = data.get('allArtifacts', [])

            if proposer_only and len(arts) > 1:
                arts = arts[:-1]

            vecs = []
            for a in arts:
                sd = a.get('solutionData')
                if sd is None:
                    continue
                v = get_solution_vec(sd, task_id)
                if v is not None:
                    vecs.append(v)

            if len(vecs) >= 2:
                dists = pairwise_distance(vecs, task_id)
                distances.extend(dists)

        if distances:
            task_results[task_id] = np.mean(distances)
            all_distances.extend(distances)

    return {
        'overall_mean': np.mean(all_distances) if all_distances else float('nan'),
        'overall_se': np.std(all_distances, ddof=1) / np.sqrt(len(all_distances)) if len(all_distances) > 1 else 0,
        'n_pairs': len(all_distances),
        'per_task': task_results,
    }


def main():
    print('=' * 72)
    print('ANALYSIS 3: Table 5 Verification — Mean Pairwise Solution Distance')
    print('=' * 72)
    print(f'\n{"Protocol":<35} {"Paper":>6} {"Computed":>9} {"SE":>6} {"N_pairs":>8}')
    print('-' * 72)

    paper_values = {
        'MoA (no synthesis)': 0.48,
        'Best-of-N (same model)': 0.40,
        'Debate': 0.38,
        'MoA (diverse + synthesis)': 0.34,
        'MoA (same-model)': 0.34,
    }

    configs = [
        ('moa-nosynth',  True,  'MoA (no synthesis)'),
        ('best-of-n',    False, 'Best-of-N (same model)'),
        ('debate',       False, 'Debate'),
        ('moa',          False, 'MoA (diverse + synthesis)'),  # all arts incl. synth
        ('moa-gemini',   True,  'MoA (same-model)'),
    ]

    for protocol, proposer_only, label in configs:
        result = compute_protocol_distance(protocol, proposer_only=proposer_only)
        m = result['overall_mean']
        se = result['overall_se']
        n = result['n_pairs']
        paper = paper_values.get(label, float('nan'))
        diff = m - paper if not np.isnan(m) else float('nan')
        diff_str = f'({diff:+.3f})' if not np.isnan(diff) else ''
        print(f'{label:<35} {paper:>6.2f} {m:>9.3f} {se:>6.3f} {n:>8d}  {diff_str}')

    # Per-task breakdown for MoA proposers
    print('\n--- MoA-nosynth per-task diversity (proposers only) ---')
    result = compute_protocol_distance('moa-nosynth', proposer_only=True)
    for task, d in sorted(result['per_task'].items()):
        print(f'  {task.replace("bench-",""):<25} {d:.3f}')

    print('\n--- MoA same-model per-task diversity ---')
    result = compute_protocol_distance('moa-gemini', proposer_only=True)
    for task, d in sorted(result['per_task'].items()):
        print(f'  {task.replace("bench-",""):<25} {d:.3f}')

    # Also try moa-nosynth with synthesis output included vs excluding synthesizer
    print('\n--- Sanity check: synthesis output vs proposer diversity ---')
    for protocol, label in [
        ('moa',         'MoA (all artifacts incl. synthesizer)'),
        ('moa-nosynth', 'MoA-nosynth (3 proposers, no synth)'),
    ]:
        r = compute_protocol_distance(protocol, proposer_only=False)
        print(f'  {label}: {r["overall_mean"]:.3f} (n={r["n_pairs"]})')


if __name__ == '__main__':
    main()
