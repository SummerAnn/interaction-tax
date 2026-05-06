# Interaction Erases Diversity in Multi-Agent Scientific Optimization

Code and data for the paper "Interaction Erases Diversity in Multi-Agent Scientific Optimization."

This paper tests whether structured collaboration among multiple AI models produces better solutions to hard scientific optimization problems than the best solution a single model can find at the same compute budget. We evaluate 10 protocol configurations on 9 scientific optimization tasks. Every answer is scored by a hidden evaluator that agents cannot query during their runs, preventing any form of proxy overfit. Results for all ~2,400 runs ship with this repo so you can reproduce every table and figure without re-running experiments.

## Key findings

- No conversational multi-agent protocol reliably beats a strong single-agent baseline. Every protocol that uses inter-agent interaction ranks below Self-Refine, a single model critiquing its own output in a loop.
- Backbone diversity helps (+0.188 OLS coefficient, p < 0.01) but inter-agent interaction erases it. Diverse agents start with genuinely different proposals; once they exchange intermediate work, their outputs converge to the quality of a same-model team.
- MoA (Mixture-of-Agents) is the only protocol whose confidence interval includes zero. Its proposers never interact, so diversity survives to the synthesis step.
- The MIG flip: Chain, MAgICoRe, and Debate each show positive Marginal Interaction Gain with same-model agents, but all three flip negative when agents come from different model families. MoA stays positive in both configurations.

## What's here

```
src/                Protocol and runner code (TypeScript)
analysis/           Analysis scripts that produce the paper tables and figures (Python)
scripts/            Figure generation scripts (Python)
results/full-v2/    All experiment results (~2,400 JSON files, 10 paper tasks)
results/2x2-multisynth/   2x2 factorial ablation results (diversity x synthesis)
results/prospective-2x2/  Additional 2x2 runs on prospective tasks
tools/evaluate.ts   Verifies saved scores by re-running verifiers locally
figures/            Generated figures
PROTOCOLS.md        Exact mechanics of each protocol implementation
```

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
# Full benchmark: 9 tasks, 10 protocols, 10 seeds (~$200, ~8 hours)
npx tsx src/runner/run-offline.ts

# Single task and protocol, 3 seeds
npx tsx src/runner/run-offline.ts --task bench-difference-bases --protocol moa --seeds 1,2,3

# Validate config without spending tokens
npx tsx src/runner/run-offline.ts --dry-run
```

Both dev and hidden evaluation run as local Python subprocesses. No platform account needed. Results are written to `results/offline/`.

## Replication note

The original experiments used a server-side hidden verifier that agents could not query during runs. For replication, both the dev and hidden verifiers are embedded as Python code strings in `src/tasks/benchmark-challenges.ts` and run as local subprocesses. Step 2 above confirms that the saved scores match.

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

Backbone suffixes: `-gpt4o`, `-gemini`, `-deepseek` (single backbone), `-mixed` (Claude + GPT-4o + Gemini), `-mixed-oss` (DeepSeek + Llama + Qwen).

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

NeurIPS 2026 Evaluations and Datasets Track (under review). Citation will be added upon acceptance.

## License

MIT
