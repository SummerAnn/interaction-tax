#!/usr/bin/env python3
"""
Analysis 2: Diversity-quality scatter by task.
For each task: initial proposer diversity (pre-interaction) vs
Q-gain = Q(diverse protocol) - Q(same-backbone protocol).

Tests whether tasks where diversity is higher in proposals also benefit more from diverse backbones.
"""
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

RESULTS_DIR = Path("results/full-v2")

# Normalization anchors (frozen from paper)
ANCHORS = {
    'bench-maxcut-500':          {'base': 3139, 'ref': 3517, 'direction': 'max'},
    'bench-circle-packing-n20':  {'base': 0.5039, 'ref': 2.0,    'direction': 'max'},
    'bench-difference-bases':    {'base': 14.0,   'ref': 1.0,    'direction': 'min'},
    'bench-flat-poly-deg50':     {'base': 2.0489, 'ref': 1.0,    'direction': 'min'},
    'bench-tsp-100':             {'base': 50835,  'ref': 7517,   'direction': 'min'},
    'bench-lj-cluster-n41':      {'base': -136.0, 'ref': -198.06,'direction': 'min'},
    'bench-erdos-overlap':       {'base': 0.2939, 'ref': 0.2321, 'direction': 'min'},
    'bench-mol-qed':             {'base': 0.60,   'ref': 1.0,    'direction': 'max'},
    'bench-tsp-50':              {'base': 25400,  'ref': 5500,   'direction': 'min'},
}

INVALID_THRESHOLD = 1e10


def q_normalize(s, task_id):
    if s is None or abs(s) > INVALID_THRESHOLD:
        return float('nan')
    a = ANCHORS.get(task_id)
    if not a:
        return float('nan')
    base, ref, direction = a['base'], a['ref'], a['direction']
    if direction == 'max':
        if abs(ref - base) < 1e-12:
            return float('nan')
        return max(0.0, min(1.0, (s - base) / (ref - base)))
    else:
        if abs(base - ref) < 1e-12:
            return float('nan')
        return max(0.0, min(1.0, (base - s) / (base - ref)))


def get_mean_q(task_id, protocol, use_hidden=True):
    """Mean Q-normalized score for a protocol on a task."""
    scores = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_{protocol}_s*.json')):
        data = json.loads(f.read_text())
        s = data.get('hiddenScore') if use_hidden else None
        if s is None:
            ba = data.get('bestArtifact', {})
            s = ba.get('devScore') if ba else None
        q = q_normalize(s, task_id)
        if not np.isnan(q):
            scores.append(q)
    return np.mean(scores) if scores else float('nan')


def tokenize(text: str) -> Counter:
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return Counter(words)


def cosine_sim(c1: Counter, c2: Counter) -> float:
    if not c1 or not c2:
        return 0.0
    intersection = set(c1) & set(c2)
    num = sum(c1[w] * c2[w] for w in intersection)
    d1 = sum(v * v for v in c1.values()) ** 0.5
    d2 = sum(v * v for v in c2.values()) ** 0.5
    if d1 == 0 or d2 == 0:
        return 0.0
    return num / (d1 * d2)


def get_initial_proposer_diversity(task_id, protocol='moa-nosynth'):
    """
    Mean pairwise cosine distance (1 - similarity) between proposers' rawOutputs,
    averaged over all seeds. Uses moa-nosynth so proposers never see each other.
    """
    distances = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_{protocol}_s*.json')):
        data = json.loads(f.read_text())
        arts = data.get('allArtifacts', [])
        texts = [a.get('rawOutput', '') for a in arts if a.get('rawOutput', '')]
        if len(texts) < 2:
            continue
        toks = [tokenize(t) for t in texts]
        for i, j in combinations(range(len(toks)), 2):
            sim = cosine_sim(toks[i], toks[j])
            distances.append(1.0 - sim)  # cosine distance
    return np.mean(distances) if distances else float('nan')


def get_solutiondata_diversity(task_id, protocol='moa-nosynth'):
    """
    Hamming distance for binary solutionData (Erdős 'values', maxcut 'assignment').
    Normalized by vector length.
    """
    distances = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_{protocol}_s*.json')):
        data = json.loads(f.read_text())
        arts = data.get('allArtifacts', [])
        vecs = []
        for a in arts:
            sd = a.get('solutionData', {})
            if not sd:
                continue
            # try different field names
            v = sd.get('values') or sd.get('assignment') or sd.get('coloring')
            if v and isinstance(v, list) and all(isinstance(x, (int, float)) for x in v[:5]):
                vecs.append(np.array(v, dtype=float))

        if len(vecs) < 2:
            continue

        # Only compare same-length vectors
        lengths = [len(v) for v in vecs]
        most_common_len = max(set(lengths), key=lengths.count)
        vecs = [v for v in vecs if len(v) == most_common_len]

        if len(vecs) < 2:
            continue

        for i, j in combinations(range(len(vecs)), 2):
            v1, v2 = vecs[i], vecs[j]
            # Normalize
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                # Hamming for binary
                d = np.mean(v1 != v2)
            else:
                cos_sim = np.dot(v1, v2) / (n1 * n2)
                d = 1.0 - cos_sim
            distances.append(float(d))

    return np.mean(distances) if distances else float('nan')


def main():
    print('=' * 72)
    print('ANALYSIS 2: Diversity-quality scatter by task')
    print('Q-gain = Q(diverse MoA) - Q(same-model MoA)')
    print('Initial diversity = cosine distance between moa-nosynth proposers')
    print('=' * 72)

    tasks = list(ANCHORS.keys())

    results = []
    for task in tasks:
        # Q-gain: diverse MoA vs same-model MoA (using canonical seeds)
        q_diverse = get_mean_q(task, 'moa')
        # Same-model controls (pick the best same-backbone)
        q_same_claude = get_mean_q(task, 'moa-claude')
        q_same_gemini = get_mean_q(task, 'moa-gemini')
        q_bon_claude  = get_mean_q(task, 'best-of-n')

        # Q-gain vs best same-backbone BoN (the control)
        q_controls = [q for q in [q_bon_claude, q_same_claude, q_same_gemini]
                      if not np.isnan(q)]
        q_best_control = max(q_controls) if q_controls else float('nan')
        q_gain = q_diverse - q_best_control if not np.isnan(q_diverse) and not np.isnan(q_best_control) else float('nan')

        # Initial diversity (moa-nosynth proposers = independent proposals)
        diversity_text = get_initial_proposer_diversity(task, 'moa-nosynth')
        diversity_solution = get_solutiondata_diversity(task, 'moa-nosynth')

        short = task.replace('bench-', '')
        results.append({
            'task': short,
            'q_diverse': q_diverse,
            'q_best_control': q_best_control,
            'q_gain': q_gain,
            'div_text': diversity_text,
            'div_solution': diversity_solution,
        })

    print(f'\n{"Task":<22} {"Q(diverse)":>10} {"Q(ctrl)":>8} {"Q-gain":>8} {"TextDiv":>9} {"SolDiv":>8}')
    print('-' * 72)
    for r in results:
        print(f'{r["task"]:<22} {r["q_diverse"]:>10.3f} {r["q_best_control"]:>8.3f} '
              f'{r["q_gain"]:>8.3f} {r["div_text"]:>9.3f} {r["div_solution"]:>8.3f}')

    # Compute correlation
    valid = [(r['div_text'], r['q_gain']) for r in results
             if not np.isnan(r['div_text']) and not np.isnan(r['q_gain'])]
    if len(valid) >= 3:
        divs, gains = zip(*valid)
        from scipy.stats import spearmanr, pearsonr
        rho, p_rho = spearmanr(divs, gains)
        r_pearson, p_pearson = pearsonr(divs, gains)
        print(f'\nCorrelation (text diversity vs Q-gain):')
        print(f'  Spearman rho={rho:.3f} (p={p_rho:.3f})')
        print(f'  Pearson r={r_pearson:.3f} (p={p_pearson:.3f})')

    valid_sol = [(r['div_solution'], r['q_gain']) for r in results
                 if not np.isnan(r['div_solution']) and not np.isnan(r['q_gain'])]
    if len(valid_sol) >= 3:
        divs, gains = zip(*valid_sol)
        from scipy.stats import spearmanr, pearsonr
        rho, p_rho = spearmanr(divs, gains)
        print(f'\nCorrelation (solution diversity vs Q-gain):')
        print(f'  Spearman rho={rho:.3f} (p={p_rho:.3f})')

    print('\n--- Tasks where diversity helps vs hurts ---')
    for r in sorted(results, key=lambda x: x['q_gain'] if not np.isnan(x['q_gain']) else -999, reverse=True):
        sign = '+' if r['q_gain'] > 0 else ''
        print(f'  {r["task"]:<22} Q-gain={sign}{r["q_gain"]:.3f}  TextDiv={r["div_text"]:.3f}')


if __name__ == '__main__':
    main()
