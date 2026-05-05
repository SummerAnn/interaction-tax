#!/usr/bin/env python3
"""
Analysis 4: Debate-nosynth ablation.
Compare:
  1. Full debate (with synthesis step)
  2. Debate-nosynth: pick best round score instead of using synthesis

Also compare same-backbone vs diverse debate, and debate-mixed-oss
(open source models in debate) to assess diversity source.
"""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path('/Users/summerann/Desktop/neurips-experiments/results/full-v2')

INVALID_THRESHOLD = 1e10

ANCHORS = {
    'bench-maxcut-500':          {'base': 3139,   'ref': 3517,   'direction': 'max'},
    'bench-circle-packing-n20':  {'base': 0.5039, 'ref': 2.0,    'direction': 'max'},
    'bench-difference-bases':    {'base': 14.0,   'ref': 1.0,    'direction': 'min'},
    'bench-flat-poly-deg50':     {'base': 2.0489, 'ref': 1.0,    'direction': 'min'},
    'bench-tsp-100':             {'base': 50835,  'ref': 7517,   'direction': 'min'},
    'bench-lj-cluster-n41':      {'base': -136.0, 'ref': -198.06,'direction': 'min'},
    'bench-erdos-overlap':       {'base': 0.2939, 'ref': 0.2321, 'direction': 'min'},
    'bench-mol-qed':             {'base': 0.60,   'ref': 1.0,    'direction': 'max'},
    'bench-tsp-50':              {'base': 25400,  'ref': 5500,   'direction': 'min'},
}


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


def best_dev_score_among_artifacts(arts, direction='max'):
    """Best devScore across allArtifacts (excluding synthesizer = last)."""
    scores = [a.get('devScore') for a in arts if a.get('devScore') is not None
              and abs(a.get('devScore', 0)) < INVALID_THRESHOLD]
    if not scores:
        return None
    return max(scores) if direction == 'max' else min(scores)


def compute_debate_nosynth_q(task_id) -> float:
    """
    Debate-nosynth: for each seed, take the best devScore among all
    intermediate rounds (all artifacts). Return mean Q.
    Direction inferred from anchors.
    """
    direction = ANCHORS.get(task_id, {}).get('direction', 'max')
    qs = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_debate_s*.json')):
        data = json.loads(f.read_text())
        arts = data.get('allArtifacts', [])
        best = best_dev_score_among_artifacts(arts, direction)
        q = q_normalize(best, task_id)
        if not np.isnan(q):
            qs.append(q)
    return (np.mean(qs), len(qs)) if qs else (float('nan'), 0)


def compute_protocol_q(task_id, protocol) -> tuple[float, int]:
    """Mean Q for a protocol on a task, using hiddenScore then devScore fallback."""
    qs = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_{protocol}_s*.json')):
        data = json.loads(f.read_text())
        s = data.get('hiddenScore')
        if s is None:
            ba = data.get('bestArtifact', {})
            s = ba.get('devScore') if ba else None
        q = q_normalize(s, task_id)
        if not np.isnan(q):
            qs.append(q)
    return (float(np.mean(qs)), len(qs)) if qs else (float('nan'), 0)


def main():
    tasks = list(ANCHORS.keys())

    print('=' * 80)
    print('ANALYSIS 4: Debate-nosynth ablation + same vs diverse backbone comparison')
    print('=' * 80)

    # ── A. Debate full vs debate-nosynth ─────────────────────────────────────
    print('\n--- A. Does synthesis help or hurt debate? ---')
    print(f'{"Task":<22} {"Debate full":>12} {"Debate-nosynth":>15} {"Delta":>8}')
    print('-' * 60)

    total_full, total_nosyn = [], []
    for task_id in tasks:
        q_full, n_full = compute_protocol_q(task_id, 'debate')
        q_nosyn, n_nosyn = compute_debate_nosynth_q(task_id)
        delta = q_nosyn - q_full if not np.isnan(q_full) and not np.isnan(q_nosyn) else float('nan')
        if not np.isnan(q_full):
            total_full.append(q_full)
        if not np.isnan(q_nosyn):
            total_nosyn.append(q_nosyn)
        short = task_id.replace('bench-', '')
        print(f'{short:<22} {q_full:>12.3f} {q_nosyn:>15.3f} {delta:>8.3f}')

    if total_full and total_nosyn:
        print(f'{"MEAN":<22} {np.mean(total_full):>12.3f} {np.mean(total_nosyn):>15.3f} '
              f'{np.mean(total_nosyn)-np.mean(total_full):>8.3f}')

    # ── B. Same vs diverse backbone: debate ──────────────────────────────────
    print('\n--- B. Diverse vs same-backbone debate ---')
    print(f'{"Task":<22} {"Debate div":>11} {"Debate-Gem":>11} {"Debate-GPT":>11} {"Delta D-S":>10}')
    print('-' * 68)

    for task_id in tasks:
        q_div,  n_div  = compute_protocol_q(task_id, 'debate')
        q_gem,  n_gem  = compute_protocol_q(task_id, 'debate-gemini')
        q_gpt,  n_gpt  = compute_protocol_q(task_id, 'debate-gpt4o')
        best_same = max(q for q in [q_gem, q_gpt] if not np.isnan(q)) if any(not np.isnan(q) for q in [q_gem, q_gpt]) else float('nan')
        delta = q_div - best_same if not np.isnan(q_div) and not np.isnan(best_same) else float('nan')
        short = task_id.replace('bench-', '')
        print(f'{short:<22} {q_div:>11.3f} {q_gem:>11.3f} {q_gpt:>11.3f} {delta:>10.3f}')

    # ── C. Open-source backbone debate (debate-mixed-oss) ────────────────────
    print('\n--- C. Commercial vs open-source backbone diversity ---')
    print('debate = commercial diverse (Claude+GPT-4o+Gemini)')
    print('debate-mixed = alternative diverse composition')
    print('debate-mixed-oss = open-source models\n')
    print(f'{"Task":<22} {"Commercial":>11} {"Alt-mix":>9} {"OSS-mix":>9}')
    print('-' * 55)

    for task_id in tasks:
        q_commercial = compute_protocol_q(task_id, 'debate')[0]
        q_mixed      = compute_protocol_q(task_id, 'debate-mixed')[0]
        q_oss        = compute_protocol_q(task_id, 'debate-mixed-oss')[0]
        short = task_id.replace('bench-', '')
        print(f'{short:<22} {q_commercial:>11.3f} {q_mixed:>9.3f} {q_oss:>9.3f}')

    # ── D. Round-by-round Q-normalized scores for debate ─────────────────────
    print('\n--- D. Round-by-round mean Q-score for debate (diverse) ---')
    print('Shows whether intermediate rounds are better than final synthesis\n')
    print(f'{"Task":<22} {"Round 1":>8} {"Round 2":>8} {"Round 3":>8} {"Final":>8} {"Best round > final?":>20}')
    print('-' * 75)

    for task_id in tasks:
        direction = ANCHORS.get(task_id, {}).get('direction', 'max')
        round_qs = {1: [], 2: [], 3: []}
        final_qs = []

        for f in sorted(RESULTS_DIR.glob(f'{task_id}_debate_s*.json')):
            data = json.loads(f.read_text())
            arts = data.get('allArtifacts', [])
            final_s = data.get('hiddenScore')
            final_q = q_normalize(final_s, task_id)
            if not np.isnan(final_q):
                final_qs.append(final_q)

            for i, a in enumerate(arts[:3], start=1):
                ds = a.get('devScore')
                q = q_normalize(ds, task_id)
                if not np.isnan(q) and abs(ds or 0) < INVALID_THRESHOLD:
                    round_qs[i].append(q)

        means = {r: np.mean(v) if v else float('nan') for r, v in round_qs.items()}
        final_mean = np.mean(final_qs) if final_qs else float('nan')
        best_round = max(m for m in means.values() if not np.isnan(m)) if any(not np.isnan(m) for m in means.values()) else float('nan')
        better = 'YES' if not np.isnan(best_round) and not np.isnan(final_mean) and best_round > final_mean + 0.01 else 'no'

        short = task_id.replace('bench-', '')
        print(f'{short:<22} {means[1]:>8.3f} {means[2]:>8.3f} {means[3]:>8.3f} {final_mean:>8.3f} {better:>20}')

    print('\nNOTE: "Best round > final?" = synthesis step hurts on this task.')


if __name__ == '__main__':
    main()
