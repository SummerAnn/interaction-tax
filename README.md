# The Interaction Tax: How Communication Erases Diversity in Multi-Agent Teams

Code and data artifact for the NeurIPS 2026 submission.

This is an empirical study, not a benchmark release. We use a frozen testbed of eleven scientific optimization and constraint satisfaction tasks to investigate when multi-agent interaction helps or hurts. Ten configurations (four single-model baselines and six multi-agent systems) are built from three model families (Claude Sonnet 4, GPT-4o, Gemini 2.5 Flash) and evaluated under hidden scoring where agents never see the final evaluator. All configurations share identical budget caps (200k tokens, 600s wall-clock, 25 visible-evaluator calls). Each (task, configuration) cell runs 5 seeds; the 2x2 factorial uses N=120 runs. This repository contains the testbed code, all saved results, and the analysis scripts needed to reproduce every table and figure in the paper.

## Key findings

1. **Diversity helps.** Each family excels on different tasks and fails completely on others. A team combining all three avoids every catastrophic failure. Diverse proposers more than double the average quality score, but only when proposals remain independent.
2. **The interaction tax.** In every configuration where agents read and revise one another's solutions, proposals converge within a single round. The same interaction that refines same-model outputs destroys cross-model differences. Under same backbones, Chain (+0.051), MAgICoRe (+0.044), and Debate (+0.012) all show positive interaction gain, but under diverse backbones, the same three flip negative (-0.024, -0.035, -0.078).
3. **MoA escapes because proposers stay independent.** Only Mixture-of-Agents escapes the interaction tax, because its proposers never see each other's work. It is the only configuration whose aggregate MEG confidence interval includes zero.
4. **Verifiable feedback determines when critique helps.** When violations are checkable, critique raises success rates from 0% to 47-73%. When there is no specific flaw to target, critique degrades solutions 57% of the time.
5. **Hidden evaluation matters.** Visible and hidden rankings diverge on three of nine optimization tasks (Difference Bases, MaxCut, Molecule QED).

## Contributions

1. A hidden-evaluator benchmark for scientific optimization that prevents agents from gaming the scoring function, with visible and hidden rankings diverging on three of nine tasks.
2. Empirical evidence that agent interaction erases backbone diversity on optimization tasks, with the interaction tax mechanism isolated via factorial analysis.
3. Preliminary evidence of a task-structure boundary where interaction helps on constraint satisfaction but hurts on open-ended optimization.

## What's here

```
src/                Testbed, configuration, evaluator, and offline runner code (TypeScript)
analysis/           Analysis scripts that reproduce paper tables and summary files (Python)
scripts/            Figure-generation scripts (Python)
results/full-v2/    Saved JSON results (1556 files) for all task/configuration/seed cells
results/analysis/   Derived TSV tables used by the paper and robustness appendix
tools/evaluate.ts   Re-runs dev/hidden verifiers locally on saved results
figures/            Generated paper figures
CONFIGURATIONS.md   Exact mechanics of each agent configuration
```

## Tasks

| Task | Type | Direction | Domain |
|------|------|-----------|--------|
| MaxCut G(200,0.3) | Optimization | max | Graph |
| Circle Packing n=20 | Optimization | max | Geometry |
| Difference Bases | Optimization | max | Combinatorics |
| Flat Polynomials deg. 50 | Optimization | min | Analysis |
| TSP-100 | Optimization | min | Routing |
| LJ n=41 | Optimization | min | Molecular |
| Erdos Overlap | Optimization | min | Combinatorics |
| Molecule QED | Optimization | max | Drug-likeness |
| TSP-50 | Optimization | min | Routing |
| Knapsack-50 | Constraint | max | Combinatorics |
| 3AP-Free-100 | Constraint | max | Combinatorics |

Four optimization tasks (Circle Packing, Flat Polynomials, Difference Bases, Erdos Overlap) are adapted from the AlphaEvolve problem suite with smaller parameterizations and independent scoring implementations.

Each optimization task pairs a visible evaluator (queryable during search) with a hidden evaluator that runs once on the final solution. Agents never observe their hidden-evaluator score during the run. Constraint tasks are compared by visible-evaluator feasibility under matched budgets.

## Configurations

| Configuration | Type | Description |
|---------------|------|-------------|
| Single-Shot | Single | One LLM call, no iteration |
| Best-of-N | Single | N=8 independent samples, best by dev score |
| Self-Refine | Single | Single model iterates on its own output |
| VGS | Single | Verifier-guided search with a small population |
| MoA | Multi | Diverse models propose independently, one synthesizes |
| Debate | Multi | Two agents argue, one synthesizes |
| MAgICoRe | Multi | Solver, reviewer, and refiner in a loop |
| HPE | Multi | Hierarchical planner with parallel executors |
| Homo-Chain | Multi | Sequential chain with same-model agents |
| Cross-Chain | Multi | Sequential chain with diverse-model agents |

Appendix ablations: `moa-nosynth`, `moa-same-model`, and within-backbone follow-ups (suffixed `-gpt4o`, `-gemini`) used in coverage, capability-confound, and robustness analyses.

## Key metrics

**MEG (Marginal Epistemic Gain)** measures whether a configuration beats the best single-agent baseline on hidden-evaluator scores:

```
MEG(p, t) = Q_hidden(p, t) - max_{c in C} Q_hidden(c, t)
```

where C = {Self-Refine, Best-of-N, VGS}. Positive MEG means the configuration outperforms the strongest single-agent baseline. Aggregate MEG averages across nine optimization tasks with 10,000 bootstrap CIs.

**MIG (Marginal Interaction Gain)** measures whether the interaction step adds value over running the same agents independently without communication:

```
MIG(p, t) = Q_hidden(p, t) - Q_hidden(parallel(A), t)
```

Positive MIG means interaction helps; negative means it hurts.

**Q-normalization** maps raw scores to a common [0, 1] scale: `Q(s) = (s - b) / (r - b)`, where `b` is the trivial-solution baseline and `r` is the best known published result.

## Step 1: Install dependencies

```bash
# Requires Node.js >= 20 and Python >= 3.10
npm install
pip install -r requirements.txt
```

No API keys needed for steps 2 and 3.

## Step 2: Verify the saved scores

Each task has two evaluators. The **dev evaluator** is the one agents can call during their runs to guide search. The **hidden evaluator** is a stricter variant agents never see; all paper results are based on hidden scores. Both are embedded as Python code in `src/tasks/benchmark-challenges.ts` and run locally as subprocesses.

```bash
# Verify hidden scores for one task
npx tsx tools/evaluate.ts --results results/full-v2 --task bench-maxcut-g200 --verifier hidden

# Verify dev scores for one task
npx tsx tools/evaluate.ts --results results/full-v2 --task bench-maxcut-g200 --verifier dev

# Verify both evaluators across all tasks (~20 minutes)
npx tsx tools/evaluate.ts --results results/full-v2 --verifier both
```

The output is a table of recorded vs. recomputed scores. They should match exactly. Any mismatch indicates a verifier version mismatch.

## Step 3: Reproduce the paper tables and figures

```bash
# AggMEG table and per-task MEG table (Table 2 in paper)
# -> results/analysis/meg_full.tsv
python analysis/analyze_bench.py

# MIG by configuration and backbone (Table 3)
# -> results/analysis/mig_extended.tsv
python analysis/analyze_mig.py

# 2x2 factorial: diversity vs. synthesis coefficients (Table 4)
# -> results/analysis/2x2_hierarchical.tsv
python analysis/analyze_2x2_hierarchical.py

# Diversity convergence: fraction of interaction steps that improve dev score (Table 5)
# -> results/analysis/convergence.tsv
python analysis/analyze_convergence.py

# Robustness checks: jackknife leave-one-task-out, denominator sensitivity
# -> results/analysis/jackknife_meg.tsv, meg_denom_sensitivity.tsv
python analysis/analyze_robustness.py

# MEG denominator bias correction (Appendix)
# -> results/analysis/meg_bias_correction.tsv
python analysis/analyze_meg_bias.py

# Proxy-sensitivity analysis: visible/hidden mismatch sweep (Appendix)
# -> results/analysis/mismatch_sweep.tsv
python analysis/analyze_mismatch_sweep.py

# Dev/hidden rank agreement per task (Appendix)
# -> results/analysis/rank_core.tsv, rank_all.tsv
python analysis/dev_hidden_rank_analysis.py

# Figure 1: AggMEG lollipop + MIG quadrant scatter
# -> figures/paper_fig1_combined.{pdf,png}
python scripts/make_paper_fig1_combined.py

# Figure 2: 2x2 coefficients + per-synthesizer robustness
# -> figures/paper_fig2_mechanism.{pdf,png}
python scripts/make_paper_figures.py
```

## Step 4: Re-run experiments from scratch (optional)

Re-running requires an `OPENROUTER_API_KEY`. Add it to `.env` (copy from `.env.example`).

```bash
# Full testbed: 11 tasks, 10 core configurations, 5 seeds each
npx tsx src/runner/run-offline.ts

# Single task and configuration, 3 seeds
npx tsx src/runner/run-offline.ts --task bench-difference-bases --protocol moa --seeds 1,2,3

# Validate config without spending tokens
npx tsx src/runner/run-offline.ts --dry-run
```

Both dev and hidden evaluation run as local Python subprocesses. No platform account needed. Results are written to `results/offline/`.

## Replication note

The original hidden-evaluator runs used a remote service that kept the hidden verifier server-side so agents could not inspect it during search. In this release, both dev and hidden verifiers are embedded directly in `src/tasks/benchmark-challenges.ts`, so all scores can be checked locally. Step 2 above confirms that the saved scores match those local verifiers.

Re-running experiments from scratch (Step 4) will produce numerically similar but not identical results. LLM outputs are stochastic: the same prompt with the same model may return different text across API calls, even at temperature 0, due to non-determinism in serving infrastructure. Aggregate MEG values across seeds are expected to fall within the 95% confidence intervals reported in the paper, but exact point estimates will vary. The saved results in `results/full-v2/` are the canonical numbers the paper reports.

## Result file format

Each file in `results/full-v2/` is named `bench-{task}_{configuration}_s{seed}.json`:

```json
{
  "taskId": "bench-difference-bases",
  "protocolId": "moa",
  "seed": 1,
  "bestArtifact": { "solutionData": {}, "devScore": 5.2, "rawOutput": "..." },
  "budgetTrace": { "tokenUsage": {}, "wallClockMs": 45000, "evalCalls": [] },
  "hiddenScore": 3.1
}
```

## Citation

Preprint link TBD.

## License

MIT
