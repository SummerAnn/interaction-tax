#!/usr/bin/env python3
"""
Complete self-refine ablation: MAgICoRe vs BoN vs self-refine (Gemini, n=15).
All data already exists in full-v2 for all 5 tasks.

Constraint tasks: feasibility rate (devScore > 0)
Optimization tasks: mean devScore, Mann-Whitney, Cohen's d

Critical question: On optimization tasks, does self-refine also lose vs BoN?
If yes: iteration itself (not just multi-agent) hurts on optimization.
If no: multi-agent critique specifically is the problem.
"""
import json
import numpy as np
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path('/Users/summerann/Desktop/neurips-experiments/results/full-v2')

CONSTRAINT_TASKS = [
    ('bench-circle-packing-n20', 'circle-packing'),
    ('bench-knapsack-50',        'knapsack-50'),
    ('bench-3ap-free-100',       '3ap-free-100'),
]

OPTIMIZATION_TASKS = [
    ('bench-flat-poly-deg50', 'flat-poly'),
    ('bench-tsp-100',         'tsp-100'),
]

PROTOCOLS = {
    'magicore-gemini':    'MAgICoRe',
    'best-of-n-gemini':   'BoN',
    'self-refine-gemini': 'Self-Refine',
}


def load_scores(task_id: str, protocol_id: str) -> list[float]:
    scores = []
    for f in sorted(RESULTS_DIR.glob(f'{task_id}_{protocol_id}_s*.json')):
        try:
            data = json.loads(f.read_text())
            ba = data.get('bestArtifact') or {}
            s = ba.get('devScore')
            if s is not None:
                scores.append(float(s))
        except Exception:
            continue
    return scores


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float('nan')
    pooled = np.sqrt(((n1-1)*np.var(a, ddof=1) + (n2-1)*np.var(b, ddof=1)) / (n1+n2-2))
    if pooled == 0:
        return float('nan')
    return (np.mean(a) - np.mean(b)) / pooled


def feasible(scores: list[float]) -> tuple[int, int]:
    """Count feasible (score > 0) and total."""
    return sum(1 for s in scores if s > 0), len(scores)


def fisher_pvalue(a_feas, a_tot, b_feas, b_tot):
    table = [[a_feas, a_tot - a_feas], [b_feas, b_tot - b_feas]]
    _, p = stats.fisher_exact(table, alternative='greater')
    return p


def print_constraint_table():
    print('\n' + '='*76)
    print('CONSTRAINT TASKS — feasibility rate (% runs with devScore > 0)')
    print('='*76)
    print(f'{"Task":<22} {"MAgICoRe":>12} {"BoN":>12} {"Self-Refine":>12} {"p(M>B)":>10} {"p(M>SR)":>10}')
    print('-'*76)

    for task_id, label in CONSTRAINT_TASKS:
        mag = load_scores(task_id, 'magicore-gemini')
        bon = load_scores(task_id, 'best-of-n-gemini')
        srf = load_scores(task_id, 'self-refine-gemini')

        m_f, m_t = feasible(mag)
        b_f, b_t = feasible(bon)
        s_f, s_t = feasible(srf)

        p_mb = fisher_pvalue(m_f, m_t, b_f, b_t) if b_t > 0 else float('nan')
        p_ms = fisher_pvalue(m_f, m_t, s_f, s_t) if s_t > 0 else float('nan')

        def pct(f, t):
            if t == 0: return '     —'
            return f'{f}/{t} ({100*f/t:.0f}%)'

        def pfmt(p):
            if np.isnan(p): return '         —'
            return f'{p:.4f}' if p >= 0.0001 else f'{p:.2e}'

        print(f'{label:<22} {pct(m_f,m_t):>12} {pct(b_f,b_t):>12} {pct(s_f,s_t):>12} {pfmt(p_mb):>10} {pfmt(p_ms):>10}')

        # Interpretation
        if s_t > 0:
            if s_f <= b_f:
                note = 'self-refine ≈ BoN → external critic is active ingredient'
            elif m_f > s_f:
                note = 'self-refine > BoN but MAgICoRe > self-refine → multi-agent adds value'
            else:
                note = 'self-refine ≈ MAgICoRe → iteration alone sufficient'
            print(f'  → {note}')

    print('='*76)


def print_optimization_table():
    print('\n' + '='*84)
    print('OPTIMIZATION TASKS — mean devScore (higher = better for normalized score)')
    print('Key question: does self-refine also hurt vs BoN? (shows if iteration itself is the problem)')
    print('='*84)
    print(f'{"Task":<16} {"MAgICoRe":>13} {"BoN":>13} {"Self-Refine":>13} {"p(B>M)":>9} {"d(B-M)":>8} {"p(B>SR)":>9} {"d(B-SR)":>8}')
    print('-'*84)

    for task_id, label in OPTIMIZATION_TASKS:
        mag = load_scores(task_id, 'magicore-gemini')
        bon = load_scores(task_id, 'best-of-n-gemini')
        srf = load_scores(task_id, 'self-refine-gemini')

        def fmt(sc):
            if not sc: return '            —'
            return f'{np.mean(sc):>8.1f} ±{np.std(sc):.1f}'

        # BoN > MAgICoRe (MAgICoRe loses)
        p_bm = stats.mannwhitneyu(bon, mag, alternative='greater').pvalue if mag and bon else float('nan')
        d_bm = cohens_d(bon, mag)

        # BoN > self-refine (self-refine loses)
        p_bs = stats.mannwhitneyu(bon, srf, alternative='greater').pvalue if srf and bon else float('nan')
        d_bs = cohens_d(bon, srf)

        def pfmt(p):
            if np.isnan(p): return '        —'
            return f'{p:.4f}' if p >= 0.0001 else f'{p:.2e}'
        def dfmt(d):
            if np.isnan(d): return '       —'
            return f'{d:.2f}'

        n_mag = len(mag); n_bon = len(bon); n_srf = len(srf)
        print(f'{label:<16} {fmt(mag):>13} {fmt(bon):>13} {fmt(srf) if srf else "—(no data)":>13} '
              f'{pfmt(p_bm):>9} {dfmt(d_bm):>8} {pfmt(p_bs):>9} {dfmt(d_bs):>8}')
        print(f'  n: MAgICoRe={n_mag}, BoN={n_bon}, SelfRefine={n_srf}')

        if srf and bon and mag:
            if p_bs < 0.05:
                note = 'self-refine also loses → ITERATION itself hurts on optimization'
            elif p_bm < 0.05:
                note = 'self-refine OK but MAgICoRe loses → multi-agent critique specifically hurts'
            else:
                note = 'no significant differences'
            print(f'  → {note}')

    print('='*84)


def print_summary():
    print('\n' + '='*60)
    print('MECHANISTIC SUMMARY')
    print('='*60)

    # Constraint: check if self-refine ≈ BoN across all tasks
    sref_bon_diffs = []
    for task_id, label in CONSTRAINT_TASKS:
        bon = load_scores(task_id, 'best-of-n-gemini')
        srf = load_scores(task_id, 'self-refine-gemini')
        if bon and srf:
            b_f = sum(1 for s in bon if s > 0) / len(bon)
            s_f = sum(1 for s in srf if s > 0) / len(srf)
            sref_bon_diffs.append(s_f - b_f)

    if sref_bon_diffs:
        mean_diff = np.mean(sref_bon_diffs)
        if mean_diff < 0.05:
            print('Constraint tasks: self-refine ≈ BoN (all tasks)')
            print('  → External critic agent is the active ingredient, not iteration')
        else:
            print(f'Constraint tasks: self-refine > BoN by {mean_diff:.0%} avg feasibility')
            print('  → Iteration alone partially helps; multi-agent amplifies it')

    print()


if __name__ == '__main__':
    print_constraint_table()
    print_optimization_table()
    print_summary()
