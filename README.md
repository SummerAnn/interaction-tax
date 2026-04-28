# Agent4Science-Bench: Experiment Code and Results

Experiment code and data for the paper "Agent4Science-Bench: A Hidden-Evaluation Benchmark for Multi-Agent Structured Optimization."

## Structure

```
src/                TypeScript experiment infrastructure
  config/           Model configs, normalization anchors, prompts, seeds
  protocols/        Protocol implementations (MoA, Debate, HPE, VGS, MAgICoRe, etc.)
  runner/           Experiment runners (offline, full, ablations)
  tasks/            Benchmark task definitions and challenge IDs
  lib/              LLM client, local evaluator, platform client
  budget/           Budget vector enforcement
  types.ts          Shared type definitions
analysis/           Python analysis scripts (bootstrap, 2x2 factorial, MEG, rank)
tools/              Evaluation utilities
  evaluate.ts       Offline score verification tool
scripts/            Figure generation
figures/            Generated figures (PDF + PNG)
results/
  full-v2/          Primary experiment results (~2,000 JSON files)
  kcurve-maxcut/    K-curve analysis results
  analysis/         Computed analysis outputs (TSV)
```

## Setup

```bash
# Requirements: Node.js >= 20, Python >= 3.10
npm install
pip install -r requirements.txt
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env
```

## Replication Note

The original experiments ran on [agent4science.org](https://agent4science.org), which executed the hidden verifier server-side and returned only the score — agents never received the verifier code. For replication, both the dev and hidden verifiers are embedded as Python code strings in `src/tasks/benchmark-challenges.ts` and run as local subprocesses. They are identical to what the platform executed. The `tools/evaluate.ts` script re-runs them on the saved result files and prints a comparison table so you can confirm the local verifiers reproduce the recorded scores before re-running anything.

No account or platform access is needed to reproduce any result in this paper.

## Reproducing Paper Results from Saved Data

All analysis scripts read from `results/full-v2/`, which ships with this repo.
No API keys or external services are needed to reproduce the analysis.

```bash
# 2x2 factorial analysis (Table 1 in paper)
python analysis/analyze_2x2_hierarchical.py

# Full MEG/MIG analysis (Table 2, heatmaps)
python analysis/analyze_bench.py

# Robustness checks (jackknife, denominator sensitivity)
python analysis/analyze_robustness.py

# Generate paper figures from analysis TSVs
python scripts/make_paper_figures.py
```

Analysis outputs are written to `results/analysis/*.tsv`.

## Verifying Saved Scores

The `tools/evaluate.ts` script re-runs all verifiers locally on saved result files
and prints a comparison table of recorded vs. recomputed scores. Use this to confirm
that the local verifiers are identical to what was run during the benchmark.

```bash
# Verify all results in full-v2 (dev + hidden verifiers)
npx tsx tools/evaluate.ts --results results/full-v2

# Verify a single task
npx tsx tools/evaluate.ts --results results/full-v2 --task bench-difference-bases

# Hidden verifier only
npx tsx tools/evaluate.ts --results results/full-v2 --verifier hidden
```

## Re-running Experiments (Offline)

Re-running from scratch requires only `OPENROUTER_API_KEY` — no platform access needed.
Dev and hidden evaluation both run as local Python subprocesses.

```bash
# Full benchmark — all 19 tasks, 12 protocols, 3 seeds (~$200 API cost, ~8 hours)
npx tsx src/runner/run-offline.ts

# Single task, single protocol
npx tsx src/runner/run-offline.ts --task bench-difference-bases --protocol moa --seeds 1,2,3

# Dry run to validate config without spending tokens
npx tsx src/runner/run-offline.ts --dry-run
```

Results are written to `results/offline/`. Hidden scores are written directly into
each result JSON at run time (no separate backfill step required).

Secondary ablation runners (budget scaling, composability 2x2, crossover, etc.) follow
the same pattern; see `src/runner/run-*.ts` for available entrypoints.

### Platform mode (optional)

`src/runner/run-full.ts` submits to agent4science.org for platform-hosted hidden evaluation.
This requires `ADMIN_SECRET` in `.env` and is not needed to reproduce the paper results.

```bash
# Platform run — requires ADMIN_SECRET
npx tsx src/runner/run-full.ts

# Backfill hidden scores from platform after a platform run
npx tsx src/runner/backfill-hidden-scores.ts --resultsDir results/full-v2
```

## Protocol Variants

The benchmark evaluates 41 protocol variants across a structured design:

| Suffix | Meaning |
|--------|---------|
| *(none)* | Default backbone (Claude Sonnet 4) |
| `-gpt4o` | GPT-4o backbone |
| `-gemini` | Gemini 2.5 Flash backbone |
| `-deepseek` | DeepSeek V3 backbone |
| `-mixed` | Frontier mixed brains (Claude + GPT-4o + Gemini) |
| `-mixed-oss` | Open-source mixed brains (DeepSeek + Llama + Qwen) |
| `-2x` | Doubled budget cap |

Core protocols: `single-shot`, `self-refine`, `best-of-n`, `vgs` (verifier-guided search),
`moa` (mixture-of-agents), `debate`, `hpe` (hierarchical planning), `magicore`.

Ablation protocols: `moa-nosynth` (no synthesis step), `moa-same-model` (no diversity),
`best-of-5-diverse` (5 families, zero interaction), `crossover`, `crossover-refine`.

## Key Concepts

- **MEG (Marginal Epistemic Gain)**: Does a protocol beat the best single-agent control? Computed on Q-normalized hidden scores.
- **MIG (Marginal Interaction Gain)**: Does the interaction step itself add or destroy value? Same agents, with vs. without communication.
- **Q-normalization**: Task-normalized scores using frozen anchors (see `src/config/normalization-anchors.ts`). Direction-aware: maximize tasks use `(raw - sBase)/(sRef - sBase)`, minimize tasks use `(sBase - raw)/(sBase - sRef)`. Clamped to [0, 1].
- **Hidden evaluation**: Dev evaluator gives partial feedback during search; hidden evaluator scores the final best artifact only. Agents never see the hidden verifier. Both verifiers are embedded as Python code strings in `src/tasks/benchmark-challenges.ts` and run as local subprocesses — no external service or credentials needed to reproduce scores.
- **Budget vector**: Each protocol runs under identical caps: T=200K tokens, W=600s wall clock, K=25 evaluator calls.

## Data Format

Each result file (`bench-{task}_{protocol}_s{seed}.json`) contains:

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

If you use this benchmark or code, please cite the accompanying paper (preprint link TBD).

## License

MIT
