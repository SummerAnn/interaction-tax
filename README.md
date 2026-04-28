# Do Multi-Agent LLMs Help? Collaboration, Diversity, and Hidden Evaluation on Scientific Optimization

Code and data for the paper "Do Multi-Agent LLMs Help? Collaboration, Diversity, and Hidden Evaluation on Scientific Optimization."

This paper asks whether structured collaboration among multiple AI models produces better solutions to hard scientific problems than the best solution a single model can find at the same compute budget. We test 10 protocols (4 single-agent baselines and 6 multi-agent) on 9 scientific optimization tasks. Each protocol is scored by a hidden evaluator that agents cannot access during their runs, which prevents them from optimizing the scoring function directly. Results for all ~2,000 runs ship with this repo so you can reproduce every table and figure without re-running experiments.

## What's here

```
src/                Protocol and runner code (TypeScript)
analysis/           Analysis scripts that produce the paper tables and figures (Python)
results/full-v2/    All experiment results (~2,000 JSON files)
results/2x2-*/      Ablation results (multi-synthesizer, composability, crossover)
tools/evaluate.ts   Verifies saved scores by re-running verifiers locally
figures/            Generated figures
```

## Step 1: Install dependencies

```bash
# Requires Node.js >= 20 and Python >= 3.10
npm install
pip install -r requirements.txt
```

No API keys needed for steps 2 and 3.

## Step 2: Verify the saved scores

Before running any analysis, confirm that the verifiers in this repo reproduce the scores in the result files. This takes a few minutes per task.

```bash
# Re-run the hidden verifier on all maxcut results and compare to saved scores
npx tsx tools/evaluate.ts --results results/full-v2 --task bench-maxcut-g200 --verifier hidden

# Run across all tasks (takes ~20 minutes)
npx tsx tools/evaluate.ts --results results/full-v2
```

The output is a table of recorded vs. recomputed scores. They should match exactly.

## Step 3: Reproduce the paper tables and figures

```bash
# MEG and MIG tables (main results)
python analysis/analyze_bench.py

# 2x2 ablation: diversity vs. synthesis (Table 1)
python analysis/analyze_2x2_hierarchical.py

# Robustness checks (jackknife, denominator sensitivity)
python analysis/analyze_robustness.py

# Figures
python scripts/make_paper_figures.py
```

Outputs are written to `results/analysis/*.tsv` and `figures/`.

## Step 4: Re-run experiments from scratch (optional)

Re-running requires an `OPENROUTER_API_KEY`. Add it to `.env` (copy from `.env.example`).

```bash
# Full benchmark: 19 tasks, 12 protocols, 5 seeds (~$200, ~8 hours)
npx tsx src/runner/run-offline.ts

# Single task and protocol, 3 seeds
npx tsx src/runner/run-offline.ts --task bench-difference-bases --protocol moa --seeds 1,2,3

# Validate config without spending tokens
npx tsx src/runner/run-offline.ts --dry-run
```

Both dev and hidden evaluation run as local Python subprocesses. No platform account needed. Results are written to `results/offline/`.

## Replication note

The original experiments ran on [agent4science.org](https://agent4science.org), which kept the hidden verifier server-side so agents could not see it during their runs. For replication, both verifiers are embedded as Python code strings in `src/tasks/benchmark-challenges.ts` and are identical to what the platform used. Step 2 above confirms this.

## Protocols

The benchmark runs 8 core protocols across multiple backbone configurations.

| Protocol | Description |
|----------|-------------|
| `single-shot` | One LLM call, no iteration |
| `best-of-n` | N independent samples, best by dev score |
| `self-refine` | Single model iterates on its own output |
| `vgs` | Verifier-guided search with a small population |
| `moa` | Mixture-of-Agents: diverse models propose, one synthesizes |
| `debate` | Two agents argue, one synthesizes |
| `magicore` | Solver, reviewer, and refiner in a loop |
| `hpe` | Hierarchical planning with parallel executors |

Backbone suffixes: `-gpt4o`, `-gemini`, `-deepseek` (single backbone), `-mixed` (Claude + GPT-4o + Gemini), `-mixed-oss` (DeepSeek + Llama + Qwen), `-2x` (doubled budget).

Ablation variants: `moa-nosynth` (no synthesis step), `moa-same-model` (single backbone, no diversity), `best-of-5-diverse`, `crossover`, `crossover-refine`.

## Key metrics

**MEG (Marginal Epistemic Gain)** measures whether a protocol beats the best single-agent baseline at the same compute budget, computed on hidden-evaluator scores after Q-normalization.

**MIG (Marginal Interaction Gain)** measures whether the interaction step in a protocol adds value over running the same agents in parallel and picking the best output.

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
