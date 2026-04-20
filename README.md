# VEIL: A Hidden-Evaluation Benchmark for Multi-Agent Structured Optimization

Experiment code and data for the NeurIPS 2026 submission.

## Structure

```
paper/              LaTeX source for the paper
src/                TypeScript experiment infrastructure
  config/           Model configs, normalization anchors, prompts
  protocols/        Protocol implementations (MoA, debate, HPE, VGS, etc.)
  runner/           Experiment runners (full benchmark, ablations, backfill)
  tasks/            Benchmark task definitions and challenge loaders
  lib/              Platform client, LLM client, local evaluator
  types.ts          Shared type definitions
analysis/           Python analysis scripts (bootstrap, 2x2 factorial, MEG)
scripts/            Figure generation and model preflight
figures/            Generated figures (PDF + PNG)
results/
  full-v2/          Primary experiment results (~2,000 JSON files)
  kcurve-maxcut/    K-curve analysis results
  analysis/         Computed analysis outputs (TSV)
tools/internal/     Development utilities (not needed for reproduction)
```

## Setup

```bash
# Requirements: Node.js >= 20, Python >= 3.10
npm install
pip install -r requirements.txt
cp .env.example .env
# Fill in API keys in .env
```

## Reproducing Paper Results

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

## Re-running Experiments

Re-running experiments from scratch requires API keys for LLM inference
(OpenRouter) and the Agent4Science platform (for hidden evaluation).

```bash
# Full benchmark run (~$200 in API costs, ~8 hours wall clock)
npm run full

# Single protocol on one task
npm run run -- --protocol=moa --task=difference-bases --seed=1

# Backfill hidden scores from platform after runs complete
npx tsx src/runner/backfill-hidden-scores.ts --resultsDir=results/full-v2
```

## Protocol Variants

The benchmark evaluates 41 protocol variants across a structured design:

| Suffix | Meaning |
|--------|---------|
| *(none)* | Default backbone (Claude 3.5 Sonnet) |
| `-gpt4o` | GPT-4o backbone |
| `-gemini` | Gemini 2.5 Flash backbone |
| `-deepseek` | DeepSeek V3 backbone |
| `-mixed` | Frontier mixed brains (Claude + GPT-4o + Gemini) |
| `-mixed-oss` | Open-source mixed brains (DeepSeek + Llama + Qwen) |
| `-2x` | Doubled budget cap |

Core protocols: `single-shot`, `self-refine`, `best-of-n`, `vgs` (verifier-guided search), `moa` (mixture-of-agents), `debate`, `hpe` (hierarchical planning), `magicore`.

Ablation protocols: `moa-nosynth` (no synthesis step), `moa-same-model` (no diversity), `best-of-5-diverse` (5 families, zero interaction).

## Key Concepts

- **MEG (Marginal Epistemic Gain)**: Does a protocol beat the best single-agent control? Computed on Q-normalized hidden scores.
- **MIG (Marginal Interaction Gain)**: Does the interaction step itself add or destroy value? Same agents, with vs. without communication.
- **Q-normalization**: Task-normalized scores using frozen anchors (see `src/config/normalization-anchors.ts`). Direction-aware: maximize tasks use `(raw - sBase)/(sRef - sBase)`, minimize tasks use `(sBase - raw)/(sBase - sRef)`. Clamped to [0, 1].
- **Hidden evaluation**: Dev evaluator visible during search; hidden evaluator for final scoring. Agents never see the hidden verifier.
- **Budget vector**: Each protocol runs under identical caps: T=200K tokens, W=600s wall clock, K=25 evaluator calls.

## Data Format

Each result file (`bench-{task}_{protocol}_s{seed}.json`) contains:

```json
{
  "taskId": "difference-bases",
  "protocolId": "moa",
  "seed": 1,
  "bestArtifact": { "solutionData": {}, "devScore": 5.2, "rawOutput": "..." },
  "budgetTrace": { "tokenUsage": {}, "wallClockMs": 45000, "evalCalls": [] },
  "submissionId": "sub_...",
  "hiddenScore": 3.1
}
```

## License

See paper for citation details.
