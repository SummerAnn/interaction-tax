#!/usr/bin/env python3
"""
Analysis 1: Round-by-round output TEXT similarity.
Computes pairwise cosine similarity of rawOutput strings across interaction steps
for debate (diverse vs same-backbone) and chain protocols.

Shows whether outputs converge (similarity increases) as agents interact.
"""
import json
import re
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

RESULTS_DIR = Path('/Users/summerann/Desktop/neurips-experiments/results/full-v2')

TASKS = [
    'bench-difference-bases',
    'bench-erdos-overlap',
    'bench-circle-packing-n20',
    'bench-tsp-50',
]

PROTOCOLS = {
    'debate':           'Debate diverse',
    'debate-gemini':    'Debate same (Gemini×3)',
    'homo-chain-gemini':'Homo-chain (Gemini×3)',
    'cross-chain':      'Cross-chain (diverse)',
    'moa':              'MoA (diverse)',
    'moa-nosynth':      'MoA-nosynth (diverse)',
}

INVALID_THRESHOLD = 1e10


def tokenize(text: str) -> Counter:
    """Simple word-level bag of words."""
    words = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return Counter(words)


def cosine_sim(c1: Counter, c2: Counter) -> float:
    """Cosine similarity between two counters."""
    if not c1 or not c2:
        return 0.0
    intersection = set(c1) & set(c2)
    num = sum(c1[w] * c2[w] for w in intersection)
    d1 = sum(v * v for v in c1.values()) ** 0.5
    d2 = sum(v * v for v in c2.values()) ** 0.5
    if d1 == 0 or d2 == 0:
        return 0.0
    return num / (d1 * d2)


def load_artifact_texts(task: str, protocol: str) -> list[list[str]]:
    """Returns list of runs, each run = list of rawOutput strings per artifact."""
    runs = []
    for f in sorted(RESULTS_DIR.glob(f'{task}_{protocol}_s*.json')):
        data = json.loads(f.read_text())
        arts = data.get('allArtifacts', [])
        texts = [a.get('rawOutput', '') for a in arts if a.get('rawOutput')]
        if len(texts) >= 2:
            runs.append(texts)
    return runs


def per_round_pairwise_similarity(runs: list[list[str]]) -> dict:
    """
    For protocols with parallel agents (like MoA): compare agent_i vs agent_j at same step.
    Returns mean pairwise cosine similarity per round.

    For sequential protocols (debate, chain): compare artifact[i] vs artifact[i+1].
    """
    if not runs:
        return {}

    # Detect if parallel or sequential
    # MoA-like: proposers run in parallel and we compare them to each other
    # Debate/chain: artifacts are sequential, compare round-to-round

    max_artifacts = max(len(r) for r in runs)

    # Per-artifact pairwise: for each run, compare all pairs at each position
    # For sequential protocols: compare artifact[0] vs artifact[1], artifact[1] vs artifact[2]
    step_sims = []

    for run in runs:
        toks = [tokenize(t) for t in run]
        run_sims = []
        for i in range(len(toks) - 1):
            sim = cosine_sim(toks[i], toks[i + 1])
            run_sims.append(sim)
        step_sims.append(run_sims)

    # Average per step transition
    max_steps = max(len(s) for s in step_sims)
    result = {}
    for step in range(max_steps):
        vals = [s[step] for s in step_sims if step < len(s)]
        if vals:
            result[f'step_{step}_to_{step+1}'] = {
                'mean': np.mean(vals),
                'se': np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0,
                'n': len(vals),
            }
    return result


def parallel_pairwise_similarity(runs: list[list[str]], proposer_only=True) -> float:
    """
    For MoA-like protocols: compare all pairs among artifacts in the same run.
    proposer_only: if True, exclude the last artifact (synthesizer).
    Returns mean pairwise cosine sim across all runs and pairs.
    """
    all_sims = []
    for run in runs:
        texts = run[:-1] if proposer_only and len(run) > 1 else run
        if len(texts) < 2:
            continue
        toks = [tokenize(t) for t in texts]
        for i, j in combinations(range(len(toks)), 2):
            sim = cosine_sim(toks[i], toks[j])
            all_sims.append(sim)
    if not all_sims:
        return float('nan')
    return float(np.mean(all_sims))


def main():
    print('=' * 72)
    print('ANALYSIS 1: Round-by-round output text similarity')
    print('(Cosine similarity of rawOutput between consecutive interaction steps)')
    print('=' * 72)

    # Sequential protocols — how similar are consecutive outputs?
    print('\n--- Sequential interaction protocols (step-to-step similarity) ---')
    print('Higher similarity = outputs converging = diversity being erased\n')

    for protocol in ['debate', 'debate-gemini', 'homo-chain-gemini', 'cross-chain']:
        label = PROTOCOLS.get(protocol, protocol)
        all_step_sims = {}
        n_runs_total = 0

        for task in TASKS:
            runs = load_artifact_texts(task, protocol)
            n_runs_total += len(runs)
            sims = per_round_pairwise_similarity(runs)
            for k, v in sims.items():
                if k not in all_step_sims:
                    all_step_sims[k] = []
                all_step_sims[k].extend([v['mean']] * v['n'])

        if not all_step_sims:
            print(f'{label}: no data')
            continue

        print(f'{label} (n={n_runs_total} runs):')
        for step, vals in sorted(all_step_sims.items()):
            m = np.mean(vals)
            print(f'  {step}: mean_cosine={m:.3f}')
        print()

    # MoA parallel proposer similarity
    print('\n--- Parallel proposer similarity (within-run) ---')
    print('MoA proposers: are diverse-backbone proposers less similar to each other?\n')

    for protocol, label in [
        ('moa', 'MoA diverse (proposers, excl. synthesizer)'),
        ('moa-nosynth', 'MoA-nosynth (proposers)'),
        ('moa-claude', 'MoA same-model (Claude×3)'),
        ('moa-gemini', 'MoA same-model (Gemini×3)'),
        ('best-of-n', 'BoN (Claude)'),
        ('best-of-n-gemini', 'BoN (Gemini)'),
    ]:
        all_sims = []
        n_runs = 0
        for task in TASKS:
            runs = load_artifact_texts(task, protocol)
            n_runs += len(runs)
            for run in runs:
                texts = run[:-1] if protocol.startswith('moa') and not protocol.startswith('moa-nosynth') and len(run) > 1 else run
                if len(texts) < 2:
                    continue
                toks = [tokenize(t) for t in texts]
                for i, j in combinations(range(len(toks)), 2):
                    sim = cosine_sim(toks[i], toks[j])
                    all_sims.append(sim)

        if all_sims:
            m = np.mean(all_sims)
            se = np.std(all_sims, ddof=1) / np.sqrt(len(all_sims)) if len(all_sims) > 1 else 0
            print(f'{label}: mean_cosine={m:.3f} ± {se:.3f} (n={n_runs} runs, {len(all_sims)} pairs)')
        else:
            print(f'{label}: no data')

    # Convergence: does debate output similarity increase across rounds?
    print('\n--- Convergence over rounds (does similarity increase?) ---')
    print('Compares artifacts across interaction positions within a run\n')

    for protocol in ['debate', 'debate-gemini']:
        label = PROTOCOLS.get(protocol, protocol)
        # Compare first vs last artifact: does the last look more like the first?
        # Or: compare agent pair cosine at round 1 vs round 2 if multi-round

        round_sims = {}
        for task in TASKS:
            runs = load_artifact_texts(task, protocol)
            for run in runs:
                if len(run) < 2:
                    continue
                toks = [tokenize(t) for t in run]
                # First vs second
                s01 = cosine_sim(toks[0], toks[1])
                round_sims.setdefault('first_to_second', []).append(s01)
                # First vs last (synthesis)
                if len(run) >= 3:
                    s0n = cosine_sim(toks[0], toks[-1])
                    round_sims.setdefault('first_to_synth', []).append(s0n)
                    s1n = cosine_sim(toks[1], toks[-1])
                    round_sims.setdefault('second_to_synth', []).append(s1n)

        print(f'{label}:')
        for k, v in round_sims.items():
            m = np.mean(v)
            print(f'  {k}: {m:.3f} (n={len(v)})')
        print()


if __name__ == '__main__':
    main()
