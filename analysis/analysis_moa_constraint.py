#!/usr/bin/env python3
"""
MOA-nosynth vs MOA-same-model vs MAgICoRe on constraint tasks.
All data exists in full-v2 (s1-s10 for MOA variants, s1-s15 for MAgICoRe).

MOA-nosynth: multiple models generate independently, no synthesizer (pure diversity)
MOA-same-model: same model 3x, no synthesizer (ablates model diversity)
MAgICoRe: iterative critique with external critic

Question: can MOA (which preserves diversity) match MAgICoRe on constraint tasks?
Or is iterative critique uniquely needed for constraint satisfaction?
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

PROTOCOLS = [
    ('magicore-gemini',  'MAgICoRe'),
    ('best-of-n-gemini', 'BoN-Gemini'),
    ('moa-nosynth',      'MOA-nosynth'),
    ('moa-same-model',   'MOA-same-model'),
    ('moa',              'MOA-mixed'),
]


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


def feasibility_rate(scores: list[float]) -> tuple[int, int]:
    return sum(1 for s in scores if s > 0), len(scores)


def fisher_p(a_feas, a_tot, b_feas, b_tot, alternative='greater'):
    table = [[a_feas, a_tot - a_feas], [b_feas, b_tot - b_feas]]
    _, p = stats.fisher_exact(table, alternative=alternative)
    return p


def print_constraint_table():
    print('\n' + '='*90)
    print('CONSTRAINT TASKS — feasibility rate by protocol')
    print('MOA variants: s1-s10; MAgICoRe + BoN-Gemini: s1-s15')
    print('='*90)

    header = f'{"Task":<22}'
    for _, label in PROTOCOLS:
        header += f'  {label:>14}'
    print(header)
    print('-'*90)

    for task_id, task_label in CONSTRAINT_TASKS:
        row = f'{task_label:<22}'
        magi_feas = None
        for proto_id, proto_label in PROTOCOLS:
            scores = load_scores(task_id, proto_id)
            if not scores:
                row += f'  {"—":>14}'
                continue
            f, t = feasibility_rate(scores)
            pct = f'{f}/{t} ({100*f/t:.0f}%)'
            row += f'  {pct:>14}'
            if proto_id == 'magicore-gemini':
                magi_feas = (f, t)
        print(row)

        # p-values vs MAgICoRe
        if magi_feas:
            prow = f'  {"p vs MAgICoRe":<20}'
            for proto_id, proto_label in PROTOCOLS:
                if proto_id == 'magicore-gemini':
                    prow += f'  {"(baseline)":>14}'
                    continue
                scores = load_scores(task_id, proto_id)
                if not scores:
                    prow += f'  {"—":>14}'
                    continue
                f, t = feasibility_rate(scores)
                try:
                    p = fisher_p(magi_feas[0], magi_feas[1], f, t, alternative='greater')
                    prow += f'  {p:>14.4f}'
                except Exception:
                    prow += f'  {"—":>14}'
            print(prow)
        print()

    print('='*90)


def print_optimization_table():
    print('\n' + '='*90)
    print('OPTIMIZATION TASKS — mean devScore by protocol')
    print('='*90)

    header = f'{"Task":<18}'
    for _, label in PROTOCOLS:
        header += f'  {label:>14}'
    print(header)
    print('-'*90)

    for task_id, task_label in OPTIMIZATION_TASKS:
        row = f'{task_label:<18}'
        bon_scores = None
        for proto_id, proto_label in PROTOCOLS:
            scores = load_scores(task_id, proto_id)
            if not scores:
                row += f'  {"—":>14}'
                continue
            mean = np.mean(scores)
            row += f'  {mean:>14.1f}'
            if proto_id == 'best-of-n-gemini':
                bon_scores = scores
        print(row)

        # Cohen's d vs BoN
        if bon_scores:
            drow = f'  {"d vs BoN-Gemini":<18}'
            for proto_id, proto_label in PROTOCOLS:
                if proto_id == 'best-of-n-gemini':
                    drow += f'  {"(baseline)":>14}'
                    continue
                scores = load_scores(task_id, proto_id)
                if len(scores) < 2:
                    drow += f'  {"—":>14}'
                    continue
                n1, n2 = len(bon_scores), len(scores)
                pool_std = np.sqrt(
                    ((n1-1)*np.var(bon_scores, ddof=1) + (n2-1)*np.var(scores, ddof=1)) / (n1+n2-2)
                )
                d = (np.mean(bon_scores) - np.mean(scores)) / pool_std if pool_std > 0 else float('nan')
                drow += f'  {d:>14.2f}'
            print(drow)
        print()

    print('='*90)


def print_key_finding():
    print('\n' + '='*60)
    print('KEY FINDING: Can diversity (MOA) substitute for critique (MAgICoRe)?')
    print('='*60)

    for task_id, label in CONSTRAINT_TASKS:
        mag = load_scores(task_id, 'magicore-gemini')
        moa = load_scores(task_id, 'moa-nosynth')
        bon = load_scores(task_id, 'best-of-n-gemini')

        if not mag or not bon:
            continue

        m_f, m_t = feasibility_rate(mag)
        b_f, b_t = feasibility_rate(bon)

        if moa:
            mo_f, mo_t = feasibility_rate(moa)
            moa_vs_mag = 'wins' if mo_f/mo_t >= m_f/m_t - 0.05 else 'loses'
            print(f'  {label}: MAgICoRe {m_f}/{m_t}, MOA-nosynth {mo_f}/{mo_t} → MOA {moa_vs_mag} vs MAgICoRe')
        else:
            print(f'  {label}: MAgICoRe {m_f}/{m_t}, MOA no data')

    print()


if __name__ == '__main__':
    print_constraint_table()
    print_optimization_table()
    print_key_finding()
