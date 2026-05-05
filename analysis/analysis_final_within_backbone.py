#!/usr/bin/env python3
"""
Final within-backbone analysis: Gemini iterative protocols vs BoN-Gemini baseline.
All comparisons at n=15 vs n=15 after Experiment D completes.
"""
import json
import os
import sys
from pathlib import Path
from scipy import stats
from scipy.stats import fisher_exact
import numpy as np

RESULTS_DIR = Path('/Users/summerann/Desktop/neurips-experiments/results/full-v2')

# Task metadata
TASK_META = {
    'bench-circle-packing-n20': {'dir': 'max', 'label': 'circle-packing'},
    'bench-flat-poly-deg50':    {'dir': 'min', 'label': 'flat-poly'},
    'bench-tsp-100':            {'dir': 'min', 'label': 'tsp-100'},
    'bench-tsp-50':             {'dir': 'min', 'label': 'tsp-50'},
    'bench-molecule-qed':       {'dir': 'max', 'label': 'molecule-qed'},
    'bench-erdos-overlap':      {'dir': 'max', 'label': 'erdos-overlap'},
    'bench-difference-bases':   {'dir': 'max', 'label': 'diff-bases'},
}

CIRCLE_FAIL = -1e28  # scores below this are invalid packings

def load_scores(task_id: str, protocol_id: str) -> list[float]:
    """Load all devScores for a given task × protocol."""
    scores = []
    for f in RESULTS_DIR.glob('*.json'):
        if f.name in ('results.json', 'salvage-index.json'):
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            continue
        if data.get('taskId') == task_id and data.get('protocolId') == protocol_id:
            ba = data.get('bestArtifact') or {}
            s = ba.get('devScore')
            if s is None:
                s = data.get('devScore')
            if s is not None:
                scores.append(float(s))
    return scores


def success_rate(scores: list[float], sentinel: float = CIRCLE_FAIL) -> tuple[int, int]:
    """Returns (successes, total) for tasks with failure sentinels."""
    successes = sum(1 for s in scores if s > sentinel)
    return successes, len(scores)


def compare(task_id: str, proto_iter: str, proto_bon: str) -> dict:
    meta = TASK_META[task_id]
    iter_scores = load_scores(task_id, proto_iter)
    bon_scores  = load_scores(task_id, proto_bon)

    result = {
        'task':       meta['label'],
        'direction':  meta['dir'],
        'protocol':   proto_iter,
        'n_iter':     len(iter_scores),
        'n_bon':      len(bon_scores),
    }

    if len(iter_scores) < 2 or len(bon_scores) < 2:
        result['note'] = 'insufficient data'
        return result

    # Circle-packing: use success rate + Fisher's exact
    if task_id == 'bench-circle-packing-n20':
        iter_ok, iter_n = success_rate(iter_scores)
        bon_ok,  bon_n  = success_rate(bon_scores)
        table = [[iter_ok, iter_n - iter_ok],
                 [bon_ok,  bon_n  - bon_ok]]
        _, p = fisher_exact(table, alternative='greater')
        result.update({
            'iter_success': f'{iter_ok}/{iter_n} ({100*iter_ok/iter_n:.0f}%)',
            'bon_success':  f'{bon_ok}/{bon_n} ({100*bon_ok/bon_n:.0f}%)',
            'test':  'Fisher (iter > bon)',
            'p':     p,
            'win':   p < 0.05,
        })
        return result

    # Everything else: two-sided t-test, then compute direction-aware d
    t, p_twosided = stats.ttest_ind(iter_scores, bon_scores)
    iter_mean = np.mean(iter_scores)
    bon_mean  = np.mean(bon_scores)
    delta = iter_mean - bon_mean  # positive = iter is higher

    # Cohen's d
    pooled_std = np.sqrt((np.std(iter_scores, ddof=1)**2 + np.std(bon_scores, ddof=1)**2) / 2)
    d = delta / pooled_std if pooled_std > 0 else 0.0

    # Win = iter is better in the task's direction
    if meta['dir'] == 'max':
        win = (delta > 0) and (p_twosided < 0.05)
        loss = (delta < 0) and (p_twosided < 0.05)
    else:
        win  = (delta < 0) and (p_twosided < 0.05)
        loss = (delta > 0) and (p_twosided < 0.05)

    result.update({
        'iter_mean': iter_mean,
        'bon_mean':  bon_mean,
        'delta':     delta,
        'd':         d,
        'p':         p_twosided,
        'test':      't-test (2-sided)',
        'win':  win,
        'loss': loss,
    })
    return result


def fmt_float(v):
    if isinstance(v, float):
        if abs(v) > 1000:
            return f'{v:,.0f}'
        return f'{v:.4f}'
    return str(v)


def print_table(rows):
    header = ['Task', 'Dir', 'Protocol', 'n_iter', 'n_bon', 'iter_mean/rate', 'bon_mean/rate', 'delta/d', 'p', 'verdict']
    col_w = [18, 4, 22, 7, 7, 18, 18, 14, 10, 10]

    def pad(s, w): return str(s)[:w].ljust(w)

    print('  '.join(pad(h, w) for h, w in zip(header, col_w)))
    print('  '.join('-'*w for w in col_w))
    for r in rows:
        if 'note' in r:
            print(f"  {r['task']} / {r['protocol']}: {r['note']}")
            continue
        if 'iter_success' in r:
            # circle-packing row
            verdict = 'WIN' if r['win'] else 'ns'
            cols = [
                r['task'], r['direction'], r['protocol'],
                str(r['n_iter']), str(r['n_bon']),
                r['iter_success'], r['bon_success'],
                '',
                f"{r['p']:.4f}",
                verdict,
            ]
        else:
            verdict = 'WIN' if r.get('win') else ('LOSS' if r.get('loss') else 'ns')
            cols = [
                r['task'], r['direction'], r['protocol'],
                str(r['n_iter']), str(r['n_bon']),
                fmt_float(r['iter_mean']), fmt_float(r['bon_mean']),
                f"d={r['d']:.2f}",
                f"{r['p']:.4f}",
                verdict,
            ]
        print('  '.join(pad(c, w) for c, w in zip(cols, col_w)))


# ─── Run comparisons ──────────────────────────────────────────────────────────

ITERATIVE_PROTOCOLS = ['magicore-gemini', 'vgs-gemini']
BASELINE = 'best-of-n-gemini'

TASKS = [
    'bench-circle-packing-n20',
    'bench-flat-poly-deg50',
    'bench-tsp-100',
    'bench-tsp-50',
    'bench-molecule-qed',
    'bench-erdos-overlap',
    'bench-difference-bases',
]

print('\n══════════════════════════════════════════════════════════════════════')
print(' Within-backbone: Gemini iterative vs BoN-Gemini (n=15 vs n=15)')
print('══════════════════════════════════════════════════════════════════════\n')

all_rows = []
for task_id in TASKS:
    for proto in ITERATIVE_PROTOCOLS:
        r = compare(task_id, proto, BASELINE)
        all_rows.append(r)

print_table(all_rows)

# Summary
wins  = [r for r in all_rows if r.get('win')]
losses = [r for r in all_rows if r.get('loss')]
print(f'\nWins (p<0.05, correct direction): {len(wins)}')
for r in wins:
    print(f'  {r["task"]} / {r["protocol"]}  p={r["p"]:.4f}')
print(f'Losses (p<0.05, wrong direction): {len(losses)}')
for r in losses:
    print(f'  {r["task"]} / {r["protocol"]}  p={r["p"]:.4f}  d={r["d"]:.2f}')

# Per-task n check
print('\n── Data availability ──')
for task_id in TASKS:
    for proto in ITERATIVE_PROTOCOLS + [BASELINE]:
        s = load_scores(task_id, proto)
        print(f'  {TASK_META[task_id]["label"]:25s}  {proto:30s}  n={len(s)}')
