# When Interaction Erases Diversity in Multi-Agent Scientific Optimization

Anonymous code and release artifact for the NeurIPS 2026 paper
"When Interaction Erases Diversity in Multi-Agent Scientific Optimization."

The release is scoped to the final paper artifact. It includes:

- the benchmark and evaluation code used in the paper,
- the saved results needed to reproduce every reported table and figure,
- the analysis scripts and figure-generation scripts,
- the two constraint-task follow-ups and appendix-only ablations referenced in the paper.

The paper studies whether structured collaboration among multiple LLM agents improves scientific optimization under matched compute budgets. We evaluate 10 configurations on 9 optimization tasks under hidden evaluation, plus 2 constraint-satisfaction tasks under matched-budget feasibility comparisons. Every optimization result is scored by a hidden evaluator that agents cannot query during search, preventing proxy overfit.

## Key findings

- No conversational multi-agent protocol reliably beats a strong single-agent baseline under hidden evaluation.
- Backbone diversity helps in the proposal stage, but interaction often erases that benefit.
- MoA (Mixture-of-Agents) is the only top-line configuration whose confidence interval includes zero, and its advantage comes from proposers remaining independent.
- On the two constraint tasks in the paper, critique improves feasibility because the tasks provide explicit, locally repairable violations.

## What's here

```
src/                Benchmark, protocol, evaluator, and offline runner code (TypeScript)
analysis/           Analysis scripts that reproduce paper tables and summary files (Python)
scripts/            Figure-generation scripts (Python)
results/full-v2/    Saved JSON results for the paper task suite and appendix-only follow-ups
results/analysis/   Derived TSV tables used by the paper and robustness appendix
tools/evaluate.ts   Re-runs dev/hidden verifiers locally on saved results
figures/            Generated paper figures
PROTOCOLS.md        Exact mechanics of each protocol implementation
```

## Scope of the released benchmark

The released paper benchmark contains these 11 tasks:

- Optimization: MaxCut, Circle Packing, Difference Bases, Flat Polynomials, TSP-100, LJ-n41, Erd\H{o}s Overlap, Molecule QED, TSP-50
- Constraint satisfaction: Knapsack-50, 3AP-Free-100

The main paper compares these 10 configurations:

- `single-shot`
- `best-of-n`
- `self-refine`
- `vgs`
- `homo-chain`
- `cross-chain`
- `magicore`
- `debate`
- `hpe`
- `moa`

The release also keeps the appendix-only ablations used in the paper:

- `moa-nosynth`
- `moa-same-model`
- selected within-backbone follow-ups used in coverage, capability-confound, and robustness analyses

Older exploratory tasks and off-paper protocol arms are intentionally omitted from this release artifact.

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
# Verify hidden scores for one task (a few minutes)
npx tsx tools/evaluate.ts --results results/full-v2 --task bench-maxcut-g200 --verifier hidden

# Verify dev scores for one task
npx tsx tools/evaluate.ts --results results/full-v2 --task bench-maxcut-g200 --verifier dev

# Verify both evaluators across all tasks (~20 minutes)
npx tsx tools/evaluate.ts --results results/full-v2 --verifier both
```

The output is a table of recorded vs. recomputed scores. They should match exactly. Any mismatch indicates a verifier version mismatch.

## Step 3: Reproduce the paper tables and figures

```bash
# AggMEG table and per-task MEG table
# -> results/analysis/meg_full.tsv
python analysis/analyze_bench.py

# MIG (Marginal Interaction Gain) by protocol and backbone configuration
# -> results/analysis/mig_extended.tsv
python analysis/analyze_mig.py

# 2x2 factorial: diversity vs. synthesis coefficients
# -> results/analysis/2x2_hierarchical.tsv
python analysis/analyze_2x2_hierarchical.py

# Diversity convergence: fraction of interaction steps that improve dev score
# -> results/analysis/convergence.tsv  (Table 5 in paper)
python analysis/analyze_convergence.py

# Robustness checks (jackknife, denominator sensitivity)
# -> results/analysis/jackknife_meg.tsv, meg_denom_sensitivity.tsv
python analysis/analyze_robustness.py

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
# Full paper benchmark: 11 tasks, 10 core protocols, benchmark seed schedule
npx tsx src/runner/run-offline.ts

# Single task and protocol, 3 seeds
npx tsx src/runner/run-offline.ts --task bench-difference-bases --protocol moa --seeds 1,2,3

# Validate config without spending tokens
npx tsx src/runner/run-offline.ts --dry-run
```

Both dev and hidden evaluation run as local Python subprocesses. No platform account needed. Results are written to `results/offline/`.

## Replication note

The original hidden-evaluator runs used a remote service that kept the hidden verifier server-side so agents could not inspect it during search. In this release, both dev and hidden verifiers are embedded directly in `src/tasks/benchmark-challenges.ts`, so the full benchmark can be checked locally. Step 2 above confirms that the saved scores match those local verifiers.

Re-running experiments from scratch (Step 4) will produce numerically similar but not identical results. LLM outputs are stochastic: the same prompt with the same model may return different text across API calls, even at temperature 0, due to non-determinism in serving infrastructure. Aggregate MEG values across 10 seeds are expected to fall within the 95% confidence intervals reported in the paper, but exact point estimates will vary. The saved results in `results/full-v2/` are the canonical numbers the paper reports.

## Protocols

| Protocol | Type | Description |
|----------|------|-------------|
| `single-shot` | Single | One LLM call, no iteration |
| `best-of-n` | Single | N independent samples, best by dev score |
| `self-refine` | Single | Single model iterates on its own output |
| `vgs` | Single | Verifier-guided search with a small population |
| `moa` | Multi | Mixture-of-Agents: diverse models propose, one synthesizes |
| `debate` | Multi | Two agents argue, one synthesizes |
| `magicore` | Multi | Solver, reviewer, and refiner in a loop |
| `hpe` | Multi | Hierarchical planning with parallel executors |
| `homo-chain` | Multi | Sequential chain with same-model agents |
| `cross-chain` | Multi | Sequential chain with diverse-model agents |

Appendix-only follow-up runs use suffixes such as `-gpt4o` and `-gemini` where the paper reports within-backbone robustness checks.

## Key metrics

**MEG (Marginal Epistemic Gain)** measures whether a protocol beats the best single-agent baseline at the same compute budget, computed on hidden-evaluator scores after Q-normalization. The denominator is `max(Self-Refine, Best-of-N, VGS)` per task.

**MIG (Marginal Interaction Gain)** measures whether the interaction step adds value over running the same agents in parallel without communication.

**Q-normalization** puts scores on a common scale where 0 matches a naive baseline and 1 matches an expert reference solution. For maximize tasks: `(raw - baseline) / (reference - baseline)`. For minimize tasks: `(baseline - raw) / (baseline - reference)`. Scores are clamped to [0, 1].

## Result file format

Each file in `results/full-v2/` is named `bench-{task}_{protocol}_s{seed}.json`:

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
