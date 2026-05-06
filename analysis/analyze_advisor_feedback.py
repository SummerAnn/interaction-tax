"""
Analyses requested by advisor:
1. Step-by-step diversity tracking across interaction rounds
2. Per-model error correlation (do models make different mistakes?)
3. Qualitative comparison: Erdős (diversity helps) vs MolQED (diversity doesn't)
"""

import json, os, glob
import numpy as np
from collections import defaultdict

BASE = "results/full-v2"

# ─── Distance functions per task ───

def cosine_distance(a, b):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)

def jaccard_distance(a, b):
    sa, sb = set(a), set(b)
    if len(sa | sb) == 0:
        return 0.0
    return 1.0 - len(sa & sb) / len(sa | sb)

def hamming_distance(a, b):
    a, b = np.array(a), np.array(b)
    return np.mean(a != b)

def normalized_levenshtein(a, b):
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if a[i-1]==b[j-1] else 1))
    return dp[m][n] / max(m, n)

def tour_distance(a, b):
    """Edge-set Jaccard for TSP tours."""
    def edges(tour):
        return set((min(tour[i], tour[(i+1)%len(tour)]), max(tour[i], tour[(i+1)%len(tour)])) for i in range(len(tour)))
    ea, eb = edges(a), edges(b)
    if len(ea | eb) == 0:
        return 0.0
    return 1.0 - len(ea & eb) / len(ea | eb)

TASK_DISTANCE = {
    'erdos-overlap': lambda a, b: cosine_distance(a['values'], b['values']),
    'difference-bases': lambda a, b: jaccard_distance(a['set'], b['set']),
    'molecule-qed': lambda a, b: normalized_levenshtein(a['smiles'], b['smiles']),
    'circle-packing-n20': lambda a, b: cosine_distance(
        [x for c in a['circles'] for x in c], [x for c in b['circles'] for x in c]),
    'flat-poly-deg50': lambda a, b: cosine_distance(a['coefficients'], b['coefficients']),
    'tsp-100': lambda a, b: tour_distance(a['tour'], b['tour']),
    'tsp-50': lambda a, b: tour_distance(a['tour'], b['tour']),
    'maxcut-g200': lambda a, b: hamming_distance(a['partition'], b['partition']),
    'lj-n41': lambda a, b: cosine_distance(
        [x for p in a['positions'] for x in p], [x for p in b['positions'] for x in p]),
}

def load_results(task, protocol):
    pattern = os.path.join(BASE, f"bench-{task}_{protocol}_s*.json")
    results = []
    for f in sorted(glob.glob(pattern)):
        results.append(json.load(open(f)))
    return results

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 1: Step-by-step diversity within runs
# For protocols with multiple artifacts, compute pairwise distance
# between consecutive artifacts to show convergence
# ═══════════════════════════════════════════════════════════════

def analysis_1_step_diversity():
    print("=" * 70)
    print("ANALYSIS 1: Step-by-step artifact diversity within runs")
    print("=" * 70)

    core_tasks = ['erdos-overlap', 'difference-bases', 'molecule-qed']

    # Protocols with multiple artifacts per run
    protocols = {
        'moa': 'MoA (diverse)',
        'moa-same-model': 'MoA (same-model)',
        'moa-nosynth': 'MoA (no synthesis)',
        'debate': 'Debate (same)',
        'debate-mixed': 'Debate (diverse)',
        'homo-chain': 'Homo-Chain',
        'cross-chain': 'Cross-Chain',
        'magicore': 'MAgICoRe (same)',
        'magicore-mixed': 'MAgICoRe (diverse)',
    }

    for task in core_tasks:
        dist_fn = TASK_DISTANCE.get(task)
        if not dist_fn:
            continue
        print(f"\n--- {task} ---")
        for proto_id, proto_name in protocols.items():
            runs = load_results(task, proto_id)
            if not runs:
                continue

            all_pairwise = []
            all_per_step_scores = []
            for run in runs:
                arts = run['allArtifacts']
                if len(arts) < 2:
                    continue

                # Pairwise distance between all artifacts
                dists = []
                for i in range(len(arts)):
                    for j in range(i+1, len(arts)):
                        try:
                            d = dist_fn(arts[i]['solutionData'], arts[j]['solutionData'])
                            dists.append(d)
                        except:
                            pass
                if dists:
                    all_pairwise.append(np.mean(dists))

                # Score trajectory
                scores = [a.get('devScore', None) for a in arts]
                all_per_step_scores.append(scores)

            if all_pairwise:
                print(f"  {proto_name:25s} pairwise_dist={np.mean(all_pairwise):.3f} (±{np.std(all_pairwise):.3f})  n_runs={len(all_pairwise)}")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 2: Cross-model error correlation
# Do different models make different mistakes on the same task?
# ═══════════════════════════════════════════════════════════════

def analysis_2_error_correlation():
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Per-model hidden scores (do models differ systematically?)")
    print("=" * 70)

    core_tasks = ['erdos-overlap', 'difference-bases', 'molecule-qed',
                  'circle-packing-n20', 'flat-poly-deg50', 'tsp-100', 'tsp-50']

    # Single-model baselines to compare
    model_protocols = {
        'Claude': 'best-of-n',
        'GPT-4o': 'best-of-n-gpt4o',
        'Gemini': 'best-of-n-gemini',
    }

    print(f"\n{'Task':25s} {'Claude':>10s} {'GPT-4o':>10s} {'Gemini':>10s} {'Spread':>10s}")
    print("-" * 70)

    model_scores = defaultdict(dict)

    for task in core_tasks:
        scores = {}
        for model_name, proto_id in model_protocols.items():
            runs = load_results(task, proto_id)
            if runs:
                hidden = [r['hiddenScore'] for r in runs if r.get('hiddenScore') is not None]
                if hidden:
                    scores[model_name] = np.mean(hidden)
                    model_scores[task][model_name] = hidden

        if len(scores) == 3:
            spread = max(scores.values()) - min(scores.values())
            print(f"{task:25s} {scores.get('Claude',0):10.4f} {scores.get('GPT-4o',0):10.4f} {scores.get('Gemini',0):10.4f} {spread:10.4f}")

    # Now compute per-task solution diversity between models
    print(f"\n{'Task':25s} {'C-G dist':>10s} {'C-Gem dist':>10s} {'G-Gem dist':>10s}")
    print("-" * 70)

    for task in core_tasks:
        dist_fn = TASK_DISTANCE.get(task)
        if not dist_fn:
            continue

        model_solutions = {}
        for model_name, proto_id in model_protocols.items():
            runs = load_results(task, proto_id)
            if runs:
                # Collect all best artifacts
                model_solutions[model_name] = [r['bestArtifact']['solutionData'] for r in runs]

        if len(model_solutions) >= 3:
            # Compute cross-model distances
            pairs = [('Claude', 'GPT-4o'), ('Claude', 'Gemini'), ('GPT-4o', 'Gemini')]
            dists = {}
            for m1, m2 in pairs:
                cross_dists = []
                for s1 in model_solutions.get(m1, []):
                    for s2 in model_solutions.get(m2, []):
                        try:
                            cross_dists.append(dist_fn(s1, s2))
                        except:
                            pass
                dists[f"{m1[0]}-{m2[0]}"] = np.mean(cross_dists) if cross_dists else float('nan')

            print(f"{task:25s} {dists.get('C-G', float('nan')):10.3f} {dists.get('C-G', float('nan')):10.3f} {dists.get('G-G', float('nan')):10.3f}")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 3: MoA proposer diversity — before vs after synthesis
# Compare what each proposer produced vs the final synthesis
# ═══════════════════════════════════════════════════════════════

def analysis_3_moa_proposer_diversity():
    print("\n" + "=" * 70)
    print("ANALYSIS 3: MoA proposer outputs vs synthesis output")
    print("=" * 70)

    core_tasks = ['erdos-overlap', 'difference-bases', 'molecule-qed']

    for task in core_tasks:
        dist_fn = TASK_DISTANCE.get(task)
        if not dist_fn:
            continue

        print(f"\n--- {task} ---")

        # MoA diverse: 3 artifacts = proposer1, proposer2, synthesis
        for proto_id, proto_name in [('moa', 'MoA diverse'), ('moa-same-model', 'MoA same'), ('moa-nosynth', 'MoA no-synth')]:
            runs = load_results(task, proto_id)
            if not runs:
                continue

            proposer_dists = []
            proposer_scores = []
            final_scores = []

            for run in runs:
                arts = run['allArtifacts']
                if len(arts) < 2:
                    continue

                # Proposer artifacts are all but last (if MoA with synthesis)
                # or all (if no synthesis)
                scores = [a.get('devScore', 0) for a in arts]
                proposer_scores.append(scores[:-1] if len(arts) > 2 else scores)
                final_scores.append(run.get('hiddenScore', scores[-1]))

                # Pairwise distance between proposers (exclude synthesis)
                n_proposers = len(arts) - 1 if proto_id != 'moa-nosynth' else len(arts)
                for i in range(n_proposers):
                    for j in range(i+1, n_proposers):
                        try:
                            proposer_dists.append(dist_fn(arts[i]['solutionData'], arts[j]['solutionData']))
                        except:
                            pass

            avg_dist = np.mean(proposer_dists) if proposer_dists else float('nan')
            avg_hidden = np.mean(final_scores) if final_scores else float('nan')
            print(f"  {proto_name:20s} proposer_dist={avg_dist:.3f}  hidden={avg_hidden:.4f}  n={len(runs)}")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 4: Qualitative — rawOutput comparison
# Show what models actually said on Erdős vs MolQED
# ═══════════════════════════════════════════════════════════════

def analysis_4_qualitative():
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Qualitative rawOutput comparison (Erdős vs MolQED)")
    print("=" * 70)

    for task in ['erdos-overlap', 'molecule-qed']:
        print(f"\n{'='*40} {task} {'='*40}")

        # MoA diverse run 1
        runs = load_results(task, 'moa')
        if runs:
            run = runs[0]
            arts = run['allArtifacts']
            print(f"\nMoA diverse (seed {run.get('seed','?')}):")
            for i, a in enumerate(arts):
                label = f"Proposer {i+1}" if i < len(arts)-1 else "Synthesizer"
                score = a.get('devScore', '?')
                raw = a.get('rawOutput', '')[:300]
                print(f"\n  [{label}] devScore={score}")
                print(f"  {raw}...")
            print(f"\n  Hidden score: {run.get('hiddenScore', '?')}")

        # Debate mixed run 1
        runs = load_results(task, 'debate-mixed')
        if runs:
            run = runs[0]
            arts = run['allArtifacts']
            print(f"\nDebate diverse (seed {run.get('seed','?')}):")
            for i, a in enumerate(arts):
                score = a.get('devScore', '?')
                raw = a.get('rawOutput', '')[:300]
                print(f"\n  [Round {i+1}] devScore={score}")
                print(f"  {raw}...")
            print(f"\n  Hidden score: {run.get('hiddenScore', '?')}")

# ═══════════════════════════════════════════════════════════════
# ANALYSIS 5: Error correlation matrix
# For each task, do models' per-seed scores correlate?
# ═══════════════════════════════════════════════════════════════

def analysis_5_error_correlation_matrix():
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Per-seed hidden score by model (error independence)")
    print("=" * 70)

    core_tasks = ['erdos-overlap', 'difference-bases', 'molecule-qed',
                  'circle-packing-n20', 'flat-poly-deg50', 'tsp-100', 'tsp-50']

    single_protos = {
        'Claude': 'single-shot',
        'GPT-4o': 'single-shot-gpt4o',
        'Gemini': 'single-shot-gemini',
    }

    for task in core_tasks:
        model_hidden = {}
        for model_name, proto_id in single_protos.items():
            runs = load_results(task, proto_id)
            if runs:
                model_hidden[model_name] = sorted([(r['seed'], r['hiddenScore']) for r in runs if r.get('hiddenScore') is not None])

        if len(model_hidden) >= 2:
            print(f"\n--- {task} ---")
            for model_name, scores in model_hidden.items():
                vals = [s[1] for s in scores]
                print(f"  {model_name:10s} hidden scores: {[f'{v:.4f}' for v in vals[:5]]}")


if __name__ == '__main__':
    analysis_1_step_diversity()
    analysis_2_error_correlation()
    analysis_3_moa_proposer_diversity()
    analysis_4_qualitative()
    analysis_5_error_correlation_matrix()
