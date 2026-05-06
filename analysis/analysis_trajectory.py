#!/usr/bin/env python3
"""
Per-round score trajectory for MAgICoRe runs.
Extracts allArtifacts[0,1,2].devScore from each run and plots/prints
round-by-round performance for constraint vs optimization tasks.

Constraint tasks: circle-packing-n20, knapsack-50, 3ap-free-100
Optimization tasks: flat-poly-deg50, tsp-100
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path("results/full-v2")

# ── Task definitions ──────────────────────────────────────────────────────────

TASKS = {
    # Constraint: higher = better (feasibility + quality)
    'bench-circle-packing-n20': {'label': 'circle-packing', 'type': 'constraint', 'higher_better': True},
    'bench-knapsack-50':        {'label': 'knapsack-50',    'type': 'constraint', 'higher_better': True},
    'bench-3ap-free-100':       {'label': '3ap-free-100',   'type': 'constraint', 'higher_better': True},
    # Optimization: higher = better (normalized dev scores go higher)
    # NOTE: TSP tour length — devScore is likely normalized such that higher = better
    # We check sign of BoN - MAgICoRe to determine direction
    'bench-flat-poly-deg50':    {'label': 'flat-poly',      'type': 'optimization', 'higher_better': True},
    'bench-tsp-100':            {'label': 'tsp-100',        'type': 'optimization', 'higher_better': True},
}

PROTOCOL = 'magicore-gemini'


def load_trajectories(task_id: str) -> list[list[float]]:
    """Load per-run round-by-round scores from allArtifacts."""
    trajectories = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_{PROTOCOL}_s*.json')):
        try:
            data = json.loads(f.read_text())
            arts = data.get('allArtifacts', [])
            if len(arts) >= 2:
                scores = [a.get('devScore') for a in arts]
                scores = [s for s in scores if s is not None]
                if len(scores) >= 2:
                    trajectories.append(scores)
        except Exception:
            continue
    return trajectories


def load_bon_scores(task_id: str) -> list[float]:
    """Load BoN-Gemini final scores for normalizing."""
    scores = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_best-of-n-gemini_s*.json')):
        try:
            data = json.loads(f.read_text())
            ba = data.get('bestArtifact') or {}
            s = ba.get('devScore')
            if s is not None:
                scores.append(float(s))
        except Exception:
            continue
    return scores


def print_trajectory_table():
    print('\n' + '='*72)
    print('SCORE TRAJECTORY — MAgICoRe (Gemini) round-by-round mean ± SE')
    print('='*72)
    print(f'{"Task":<22} {"Type":<14} {"Round 0":>10} {"Round 1":>10} {"Round 2":>10} {"n_runs":>7}')
    print('-'*72)

    for task_id, meta in TASKS.items():
        trajs = load_trajectories(task_id)
        if not trajs:
            print(f'{meta["label"]:<22} {"NO DATA":<14}')
            continue

        # Pad to 3 rounds if some have only 2
        max_rounds = max(len(t) for t in trajs)
        padded = []
        for t in trajs:
            if len(t) >= 3:
                padded.append(t[:3])
            elif len(t) == 2:
                padded.append(t[:2] + [None])

        rounds = [[] for _ in range(max_rounds)]
        for t in padded:
            for r, s in enumerate(t):
                if s is not None:
                    rounds[r].append(s)

        means = [np.mean(r) if r else float('nan') for r in rounds]
        ses   = [stats.sem(r) if len(r) > 1 else float('nan') for r in rounds]

        def fmt(m, se):
            if np.isnan(m):
                return '    —'
            return f'{m:10.2f}'

        row = f'{meta["label"]:<22} {meta["type"]:<14}'
        for i in range(3):
            m = means[i] if i < len(means) else float('nan')
            se = ses[i] if i < len(ses) else float('nan')
            row += fmt(m, se)
        row += f'{len(trajs):>7}'
        print(row)

        # Also print improvement direction
        if len(means) >= 2 and not np.isnan(means[0]) and not np.isnan(means[1]):
            delta_01 = means[1] - means[0]
            sign = '▲' if delta_01 > 0 else '▼'
            if len(means) >= 3 and not np.isnan(means[2]):
                delta_12 = means[2] - means[1]
                sign2 = '▲' if delta_12 > 0 else '▼'
                print(f'  {"":>20} Δ(0→1)={delta_01:+.2f} {sign}   Δ(1→2)={delta_12:+.2f} {sign2}')
            else:
                print(f'  {"":>20} Δ(0→1)={delta_01:+.2f} {sign}')

    print('='*72)


def print_regression_within_run():
    """Does round 1→2 regress on optimization but not constraint?"""
    print('\n' + '='*72)
    print('MONOTONE vs REGRESSION BREAKDOWN by task type')
    print('(A run "regresses" if any round score < previous round score)')
    print('='*72)

    for task_type in ['constraint', 'optimization']:
        task_ids = [k for k, v in TASKS.items() if v['type'] == task_type]
        all_trajs = []
        for tid in task_ids:
            all_trajs.extend(load_trajectories(tid))

        if not all_trajs:
            continue

        n_runs = len(all_trajs)
        n_reg_01 = sum(1 for t in all_trajs if len(t) >= 2 and t[1] < t[0])
        n_reg_12 = sum(1 for t in all_trajs if len(t) >= 3 and t[2] < t[1])
        n_any_reg = sum(
            1 for t in all_trajs
            if (len(t) >= 2 and t[1] < t[0]) or (len(t) >= 3 and t[2] < t[1])
        )

        print(f'\n  {task_type.upper()} tasks ({len(task_ids)} tasks, {n_runs} total runs)')
        print(f'    Regress round 0→1: {n_reg_01}/{n_runs} = {100*n_reg_01/n_runs:.0f}%')
        print(f'    Regress round 1→2: {n_reg_12}/{n_runs} = {100*n_reg_12/n_runs:.0f}%')
        print(f'    Any regression:    {n_any_reg}/{n_runs} = {100*n_any_reg/n_runs:.0f}%')

    print()


def print_normalized_trajectory():
    """Normalize each task to BoN mean, then show trajectory."""
    print('\n' + '='*72)
    print('NORMALIZED TRAJECTORY (each task normalized to BoN-Gemini mean = 1.0)')
    print('='*72)
    print(f'{"Task":<22} {"Type":<14} {"Round 0":>10} {"Round 1":>10} {"Round 2":>10}')
    print('-'*72)

    for task_id, meta in TASKS.items():
        trajs = load_trajectories(task_id)
        bon_scores = load_bon_scores(task_id)
        if not trajs or not bon_scores:
            continue

        bon_mean = np.mean(bon_scores)
        if bon_mean == 0:
            continue

        rounds = [[], [], []]
        for t in trajs:
            for r in range(min(len(t), 3)):
                if t[r] is not None:
                    rounds[r].append(t[r] / bon_mean)

        means = [np.mean(r) if r else float('nan') for r in rounds]

        def fmt(m):
            return f'{m:10.3f}' if not np.isnan(m) else '         —'

        row = f'{meta["label"]:<22} {meta["type"]:<14}'
        for m in means:
            row += fmt(m)
        print(row)

    print('='*72)


if __name__ == '__main__':
    print_trajectory_table()
    print_regression_within_run()
    print_normalized_trajectory()
