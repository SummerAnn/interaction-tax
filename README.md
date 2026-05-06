# Interaction Erases Diversity in Multi-Agent Scientific Optimization

Code and data for the paper "Interaction Erases Diversity in Multi-Agent Scientific Optimization," NeurIPS 2026 Evaluations and Datasets Track.

We evaluate ten agent protocol configurations on nine scientific optimization tasks at matched compute. Every answer is scored by a hidden evaluator that agents cannot query during search, preventing proxy overfit. All ~2,400 result files are included so every table and figure in the paper can be reproduced without re-running experiments.

## Key findings

- No protocol achieves positive aggregate MEG. Every protocol where agents exchange intermediate work ranks below Self-Refine ($-0.038$). MoA is the only protocol whose confidence interval includes zero ($-0.022$, CI $[-0.059, +0.015]$).
- Backbone diversity helps in isolation ($+0.188$ coefficient, CI entirely above zero) but interaction erases it. Once agents exchange outputs, diverse proposals converge to the level of same-model generation. The synthesis step contributes nothing ($-0.010$, CI straddles zero).
- Chain, MAgICoRe, and Debate each show positive Marginal Interaction Gain with same-model agents but flip negative under diverse backbones. MoA stays positive in both configurations because its proposers generate without seeing each other.
- Synthesis collapses mean pairwise proposal distance from $0.48$ (diverse, pre-synthesis) to $0.34$, statistically indistinguishable from same-model generation.
- On constraint satisfaction tasks the pattern reverses. External critique raises feasibility from 0–27% to 47–80% across three tasks; self-refinement with the same single agent produces no improvement.

## Tasks

Nine scientific optimization tasks, each with a visible evaluator (queryable during search) and a stricter hidden evaluator (run once on the final solution):

| Task | Domain | Dir. |
|------|--------|------|
| MaxCut $G(200,0.3)$ | Graph | max |
| Circle Packing $n=20$ | Geometry | max |
| Difference Bases | Combinatorics | min |
| Flat Polynomials deg. 50 | Analysis | min |
| TSP-100 | Routing | min |
| LJ $n=41$ | Chemistry | min |
| Erdős Overlap | Analysis | min |
| Molecule QED | Chemistry | max |
| TSP-50 | Routing | min |

## Protocols

| Protocol | Type | Description |
|----------|------|-------------|
| `single-shot` | Control | One LLM call, no iteration |
| `best-of-n` | Control | 8 independent samples; return visible-best |
| `self-refine` | Control | Single model generates, critiques, and revises for 3 rounds |
| `vgs` | Control | Population-based evolutionary search (pop. 4, gen. 5, elite 2) |
| `homo-chain` | Multi | 4 same-model agents in sequence; each attempts improvement on prior output |
| `cross-chain` | Multi | 3-agent chain rotating Claude → GPT-4o → Gemini |
| `magicore` | Multi | Solver generates; external critic critiques; refiner revises. 2 rounds |
| `debate` | Multi | 2 agents propose independently, exchange critiques across 2 rounds, synthesizer merges |
| `hpe` | Multi | Planner decomposes into 3 subproblems; executors solve independently; integrator assembles |
| `moa` | Multi | 3 proposers (Claude, GPT-4o, Gemini) generate with zero mutual knowledge; 1 synthesizer combines |
| `moa-same-model` | Ablation | Identical to MoA but all 3 proposers use Claude Sonnet 4 |
| `moa-nosynth` | Ablation | Identical to MoA (diverse) but returns the best-scoring proposal directly |

Backbones: Claude Sonnet 4 (`anthropic/claude-sonnet-4`), GPT-4o (`openai/gpt-4o`), Gemini 2.5 Flash (`google/gemini-2.5-flash`), all via OpenRouter. Budget per run: 200K tokens, 600s wall clock, 30s evaluator CPU, 25 visible-evaluator calls.

## What's here

```
src/                Protocol and runner code (TypeScript)
analysis/           Analysis scripts that reproduce every paper table (Python)
scripts/            Figure generation scripts (Python)
results/full-v2/    All experiment results (~2,400 JSON files, 9 paper tasks)
results/2x2-multisynth/   2x2 factorial results (diversity x synthesis, 3 tasks x 120 runs)
results/prospective-2x2/  Additional 2x2 runs on prospective tasks
tools/evaluate.ts   Re-runs verifiers on saved results to confirm scores
figures/            Generated figures (PDF and PNG)
PROTOCOLS.md        Full mechanics of each protocol implementation
```

## Step 1: Install dependencies

```bash
# Requires Node.js >= 20 and Python >= 3.10
npm install
pip install -r requirements.txt
```

No API keys needed for steps 2 and 3.

## Step 2: Verify the saved scores

Each result file contains both the dev score (recorded during the run) and the hidden score (recorded after). Both verifiers are embedded as Python code in `src/tasks/benchmark-challenges.ts` and run as local subprocesses.

```bash
# Verify hidden scores for one task (a few minutes)
npx tsx tools/evaluate.ts --results results/full-v2 --task bench-maxcut-g200 --verifier hidden

# Verify dev scores for one task
npx tsx tools/evaluate.ts --results results/full-v2 --task bench-maxcut-g200 --verifier dev

# Verify both evaluators across all tasks (~20 minutes)
npx tsx tools/evaluate.ts --results results/full-v2 --verifier both
```

The output is a table of recorded vs. recomputed scores. They should match exactly.

## Step 3: Reproduce the paper tables and figures

```bash
# Table: Aggregate MEG and per-task MEG (Tab. aggmeg, Tab. pertask)
python analysis/analyze_bench.py

# Table: MIG by protocol family, same-model vs diverse (Tab. mig)
python analysis/analyze_mig.py

# Table: 2x2 factorial estimates — diversity vs synthesis (Tab. 2x2)
python analysis/analyze_2x2_hierarchical.py

# Table: Per-task 2x2 cell means (Tab. 2x2cells)
python analysis/analyze_2x2_matched.py

# Table: LOO sensitivity of diversity coefficient (Tab. loo)
python analysis/analyze_robustness.py

# Table: Per-round visible scores for Debate (Tab. traj_scores)
python analysis/analysis_trajectory.py

# Table: Initial proposal quality vs final hidden score (Tab. initfinal)
python analysis/analysis_final_within_backbone.py

# Table: Synthesis decision quality relative to best proposer (Tab. synthesis_decisions)
python analysis/analysis_nosynth_ablation.py

# Table: Mean pairwise solution distance (Tab. diversity)
python analysis/analysis_output_similarity.py

# Table: Round-by-round Jaccard distance for Debate (Tab. jaccard)
python analysis/analysis_diversity_collapse.py

# Table: Feasibility success rates on constraint tasks (Tab. constraint)
python analysis/analysis_moa_constraint.py

# Step-level improvement fraction across protocols (Section 4.1 text)
python analysis/analyze_convergence.py

# Figure 1: AggMEG lollipop + MIG quadrant scatter
python scripts/make_paper_fig1_combined.py

# Figure 2: 2x2 coefficients + per-synthesizer robustness
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

Results are written to `results/offline/`.

## Replication note

The original experiments used a server-side hidden verifier that agents could not query during runs. For replication, both the dev and hidden verifiers are embedded as Python code strings in `src/tasks/benchmark-challenges.ts` and run as local subprocesses. Step 2 above confirms that the saved scores match.

Re-running from scratch (Step 4) will produce numerically similar but not identical results. LLM outputs are stochastic: the same prompt with the same model may return different text across API calls even at temperature 0, due to non-determinism in serving infrastructure. Aggregate MEG values across 10 seeds are expected to fall within the 95% confidence intervals reported in the paper, but exact point estimates will vary. The saved results in `results/full-v2/` are the canonical numbers the paper reports.

## Key metrics

**MEG (Marginal Epistemic Gain)** — did the protocol beat the best single-agent control? Computed as the protocol's hidden-evaluator Q-score minus `max(Self-Refine, Best-of-N, VGS)` per task, then averaged across tasks with bootstrap confidence intervals.

**MIG (Marginal Interaction Gain)** — did interaction help? Computed as the protocol's Q-score minus the best single-shot score from the same backbone(s) run without communication. Reported separately for same-model and diverse-backbone configurations.

**Q-normalization** maps raw scores to $[0,1]$ where 0 = trivial baseline and 1 = best known published result. For maximization: `(raw - baseline) / (reference - baseline)`; for minimization: `(baseline - raw) / (baseline - reference)`. Scores are clipped to $[0,1]$.

## Result file format

Each file in `results/full-v2/` is named `{task}_{protocol}_s{seed}.json`:

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
