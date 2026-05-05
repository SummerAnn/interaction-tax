#!/usr/bin/env python3
"""
Analysis: MoA-NoSynth same-model ablation.

Compares:
  moa-nosynth         : Claude+GPT-4o+Gemini, pick best (diverse, N=3)
  moa-nosynth-gemini  : Gemini×3, pick best (same-model, N=3)
  moa-nosynth-gpt4o   : GPT-4o×3, pick best (same-model, N=3)

Tests the capability confound:
  If moa-nosynth(diverse) > moa-nosynth-gpt4o on Erdős, the diversity
  claim survives against the strongest single-model baseline on that task.
  If not, the observed diversity coefficient in the 2x2 may be explained
  by GPT-4o capability alone rather than diversity.
"""
import json
import numpy as np
from pathlib import Path

RESULTS_DIR = Path('/Users/summerann/Desktop/neurips-experiments/results/full-v2')

INVALID_THRESHOLD = 1e10

ANCHORS = {
    'bench-maxcut-g200':         {'base': 3139,   'ref': 3517,   'direction': 'max'},
    'bench-circle-packing-n20':  {'base': 0.5039, 'ref': 2.0,    'direction': 'max'},
    'bench-difference-bases':    {'base': 14.0,   'ref': 1.0,    'direction': 'min'},
    'bench-flat-poly-deg50':     {'base': 2.0489, 'ref': 1.0,    'direction': 'min'},
    'bench-tsp-100':             {'base': 50835,  'ref': 7517,   'direction': 'min'},
    'bench-lj-n41':              {'base': -136.0, 'ref': -198.06,'direction': 'min'},
    'bench-erdos-overlap':       {'base': 0.2939, 'ref': 0.2321, 'direction': 'min'},
    'bench-molecule-qed':        {'base': 0.60,   'ref': 1.0,    'direction': 'max'},
    'bench-tsp-50':              {'base': 25400,  'ref': 5500,   'direction': 'min'},
}

# Original 3 tasks from the 2×2 ablation (GPT-4o-favourable subset)
TASKS_2X2 = ['bench-difference-bases', 'bench-erdos-overlap', 'bench-molecule-qed']

# Full 9-task benchmark (the capability-confound fix: no single model dominates)
TASKS_ALL9 = [
    'bench-maxcut-g200',
    'bench-circle-packing-n20',
    'bench-difference-bases',
    'bench-flat-poly-deg50',
    'bench-tsp-100',
    'bench-lj-n41',
    'bench-erdos-overlap',
    'bench-tsp-50',
    'bench-molecule-qed',
]


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


def get_scores(task_id, protocol, use_hidden=True):
    """Return list of Q-normalized scores for a protocol on a task."""
    scores = []
    files = sorted(RESULTS_DIR.glob(f'{task_id}_{protocol}_s*.json'))
    for f in files:
        data = json.loads(f.read_text())
        s = data.get('hiddenScore') if use_hidden else None
        if s is None:
            ba = data.get('bestArtifact', {})
            s = ba.get('devScore') if ba else None
        q = q_normalize(s, task_id)
        if not np.isnan(q):
            scores.append(q)
    return scores


def bootstrap_ci(scores, n_boot=2000):
    if len(scores) < 2:
        return float('nan'), float('nan')
    boots = [np.mean(np.random.choice(scores, size=len(scores), replace=True))
             for _ in range(n_boot)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def print_task_table(task_ids, protocols, label=''):
    """Print a Q-score table for a list of tasks and protocols. Returns task_results dict."""
    task_results = {p_id: {} for p_id, _ in protocols}

    print(f'\n{"Task":<22}', end='')
    for _, lbl in protocols:
        print(f'  {lbl[:18]:>18}', end='')
    print()
    print('-' * (22 + 20 * len(protocols)))

    for task_id in task_ids:
        short = task_id.replace('bench-', '')
        print(f'{short:<22}', end='')
        for p_id, _ in protocols:
            scores = get_scores(task_id, p_id, use_hidden=False)
            if not scores:
                scores = get_scores(task_id, p_id, use_hidden=True)
            q = np.mean(scores) if scores else float('nan')
            task_results[p_id][task_id] = (q, len(scores))
            s = f'{q:.3f}(n={len(scores)})' if not np.isnan(q) else 'N/A'
            print(f'  {s:>18}', end='')
        print()

    n_tasks = len(task_ids)
    print(f'\n{"MEAN ("+str(n_tasks)+" tasks)":<22}', end='')
    for p_id, _ in protocols:
        qs = [v for v, _ in task_results[p_id].values() if not np.isnan(v)]
        q_mean = np.mean(qs) if qs else float('nan')
        n_covered = len(qs)
        s = f'{q_mean:.3f}({n_covered}/{n_tasks})' if not np.isnan(q_mean) else 'N/A'
        print(f'  {s:>18}', end='')
    print()

    return task_results


def print_winner_table(task_ids, task_results):
    """Print per-task winner among the 3 same-model protocols."""
    print('\n--- Per-task winner (diverse vs Gemini×3 vs GPT-4o×3) ---')
    diverse_wins = gemini_wins = gpt_wins = 0
    for task_id in task_ids:
        short = task_id.replace('bench-', '')
        q_div = task_results['moa-nosynth'].get(task_id, (float('nan'), 0))[0]
        q_gem = task_results['moa-nosynth-gemini'].get(task_id, (float('nan'), 0))[0]
        q_gpt = task_results['moa-nosynth-gpt4o'].get(task_id, (float('nan'), 0))[0]
        candidates = {'diverse': q_div, 'gemini3': q_gem, 'gpt4o3': q_gpt}
        valid = {k: v for k, v in candidates.items() if not np.isnan(v)}
        if valid:
            winner = max(valid, key=valid.get)
            if winner == 'diverse': diverse_wins += 1
            elif winner == 'gemini3': gemini_wins += 1
            else: gpt_wins += 1
            print(f'  {short:<22}  winner={winner:<8} '
                  f'[{", ".join(f"{k}={v:.3f}" for k,v in valid.items() if not np.isnan(v))}]')
        else:
            print(f'  {short:<22}  no data yet')
    print(f'\n  Wins: diverse={diverse_wins}  gemini3={gemini_wins}  gpt4o3={gpt_wins}')


def main():
    np.random.seed(42)

    protocols_n3 = [
        ('moa-nosynth',        'MoA-NoSynth (diverse)'),
        ('moa-nosynth-gemini', 'Gemini×3'),
        ('moa-nosynth-gpt4o',  'GPT-4o×3'),
    ]

    # ── Section 1: Original 3-task 2×2 subset (for reference) ─────────────────
    print('=' * 80)
    print('SECTION 1 — Original 3-task subset (2×2 tasks, GPT-4o-favourable)')
    print('These are the tasks where GPT-4o×3 beats diverse. Showing for reference.')
    print('=' * 80)

    task_results_3 = print_task_table(TASKS_2X2, protocols_n3)

    print('\n--- Delta vs diverse ---')
    for task_id in TASKS_2X2:
        short = task_id.replace('bench-', '')
        q_div = task_results_3['moa-nosynth'].get(task_id, (float('nan'), 0))[0]
        q_gem = task_results_3['moa-nosynth-gemini'].get(task_id, (float('nan'), 0))[0]
        q_gpt = task_results_3['moa-nosynth-gpt4o'].get(task_id, (float('nan'), 0))[0]
        d_gem = q_div - q_gem if not (np.isnan(q_div) or np.isnan(q_gem)) else float('nan')
        d_gpt = q_div - q_gpt if not (np.isnan(q_div) or np.isnan(q_gpt)) else float('nan')
        sg = '+' if not np.isnan(d_gem) and d_gem > 0 else ''
        sp = '+' if not np.isnan(d_gpt) and d_gpt > 0 else ''
        print(f'  {short:<22}  div={q_div:.3f}  '
              f'gem3={q_gem:.3f}(Δ={sg}{d_gem:.3f})  '
              f'gpt3={q_gpt:.3f}(Δ={sp}{d_gpt:.3f})')

    # ── Section 2: Full 9-task benchmark (the fix) ─────────────────────────────
    print('\n' + '=' * 80)
    print('SECTION 2 — Full 9-task benchmark (capability confound fix)')
    print('No single model dominates across all 9 tasks. Coverage = the advantage.')
    print('=' * 80)

    task_results_9 = print_task_table(TASKS_ALL9, protocols_n3)
    print_winner_table(TASKS_ALL9, task_results_9)

    # ── Section 3: Bootstrap CIs on full-9 mean ────────────────────────────────
    print('\n--- Bootstrap 95% CI on mean Q (all 9 tasks) ---')
    for p_id, label in protocols_n3:
        all_q = []
        for task_id in TASKS_ALL9:
            scores = get_scores(task_id, p_id, use_hidden=False)
            if not scores:
                scores = get_scores(task_id, p_id, use_hidden=True)
            all_q.extend(scores)
        if len(all_q) >= 2:
            lo, hi = bootstrap_ci(all_q)
            print(f'  {label:<30} mean={np.mean(all_q):.3f}  '
                  f'95%CI=[{lo:.3f},{hi:.3f}]  n={len(all_q)}')
        else:
            print(f'  {label:<30} N/A (n={len(all_q)})')

    # ── Section 4: Tasks with missing same-model data ──────────────────────────
    missing = []
    for task_id in TASKS_ALL9:
        for p_id in ['moa-nosynth-gemini', 'moa-nosynth-gpt4o']:
            scores = get_scores(task_id, p_id, use_hidden=False)
            if not scores:
                scores = get_scores(task_id, p_id, use_hidden=True)
            if not scores:
                missing.append((task_id, p_id))
    if missing:
        print('\n--- Missing data (run run-nosynth-ablation.ts to fill these) ---')
        for tid, pid in missing:
            print(f'  {tid}  {pid}')
    else:
        print('\n[All 9-task cells have data]')


if __name__ == '__main__':
    main()
