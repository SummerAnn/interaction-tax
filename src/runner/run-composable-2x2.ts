#!/usr/bin/env tsx
/**
 * run-composable-2x2.ts — 2×2 diversity ablation for composable benchmark tasks
 *
 * Tests whether MoA's diversity advantage generalizes to strongly composable tasks
 * (Set Cover, Knapsack-50) beyond the three already-positive tasks in the main benchmark
 * (DiffBases +0.090, Erdős +0.021, FlatPoly +0.013).
 *
 * Design:
 *   ┌─────────────┬───────────────┬──────────────────┐
 *   │             │ Same Model    │ Diverse Models   │
 *   ├─────────────┼───────────────┼──────────────────┤
 *   │ No Synthesis│ Best-of-3     │ MoA-NoSynth      │
 *   │ Synthesis   │ MoA-Same-Model│ MoA (diverse)    │
 *   └─────────────┴───────────────┴──────────────────┘
 *
 * Cells: 2 tasks × 4 protocols × 10 seeds = 80 cells
 *
 * Prerequisite: deploy Set Cover hidden verifier first:
 *   npx tsx src/tasks/update-setcover-verifier.ts
 *
 * Usage:
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-composable-2x2.ts
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-composable-2x2.ts --dry-run
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-composable-2x2.ts --task bench-set-cover-60
 */

import 'dotenv/config';
import { runExperiment } from './experiment-runner.js';
import {
  primaryBackbone,
  moaBackbones,
  moaBackbonesSameModel,
} from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

// ── CLI ───────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const taskIdx = args.indexOf('--task');
const taskFilter = taskIdx >= 0 ? args[taskIdx + 1] : undefined;

// ── Task conversion ───────────────────────────────────────────────────────────

function challengeToTask(ch: typeof ALL_BENCHMARK_CHALLENGES[0]): BenchmarkTask {
  return {
    id: ch.id,
    title: ch.title,
    track: 'scientific',
    prompt: ch.description,
    devVerifier: ch.verifier,
    solutionSchema: { format: ch.solutionSchema },
    scoringDirection: ch.scoringDirection,
    challengeId: BENCHMARK_CHALLENGE_IDS[ch.id],
    parameters: {},
  };
}

const COMPOSABLE_TASK_IDS = ['bench-set-cover-60', 'bench-knapsack-50'];

const tasks: BenchmarkTask[] = ALL_BENCHMARK_CHALLENGES
  .filter(ch => {
    if (taskFilter) return ch.id === taskFilter;
    return COMPOSABLE_TASK_IDS.includes(ch.id);
  })
  .map(ch => {
    const task = challengeToTask(ch);
    if (!task.challengeId) {
      console.warn(`WARNING: No challenge ID for ${ch.id} — register task first.`);
    }
    return task;
  });

if (tasks.length === 0) {
  console.error(`ERROR: No tasks found. Filter: ${taskFilter ?? COMPOSABLE_TASK_IDS.join(', ')}`);
  process.exit(1);
}

// ── 2×2 Protocols ─────────────────────────────────────────────────────────────

const protocols: ProtocolConfig[] = [
  {
    id: 'moa',
    name: 'MoA Diverse (Claude+GPT-4o+Gemini)',
    backbones: moaBackbones(),
    params: {},
  },
  {
    id: 'moa-same-model',
    name: 'MoA Same-Model (Claude×3)',
    backbones: moaBackbonesSameModel(),
    params: {},
  },
  {
    id: 'moa-nosynth',
    name: 'MoA No-Synth (diverse, no synthesis)',
    backbones: moaBackbones(),
    params: {},
  },
  {
    id: 'best-of-3',
    name: 'Best-of-3 (Claude×3, matched N=3)',
    backbones: [primaryBackbone()],
    params: { n: 3 },
  },
];

// ── Budget (matches main benchmark) ───────────────────────────────────────────

const BUDGET = {
  tokenCap: 200_000,
  wallClockSeconds: 600,
  evalCpuSeconds: 30,
  evalCallCap: 25,
};

// ── Seeds ─────────────────────────────────────────────────────────────────────

const seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// ── Config ────────────────────────────────────────────────────────────────────

const config: ExperimentConfig = {
  id: 'bench-composable-2x2',
  name: 'Composable Benchmark Tasks 2×2 Diversity Ablation (Set Cover + Knapsack)',
  tasks,
  protocols,
  budget: BUDGET,
  seeds,
  outputDir: 'results/composable-2x2',
  submitForHiddenEval: true,
  apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
  apiKey: process.env.ADMIN_SECRET,
  agentId: 'agent_mnrvr457qvb4rt48',
  protocolAgentMap: {
    'moa':            'agent_mnrvr457qvb4rt48',
    'moa-same-model': 'agent_mnrvr457qvb4rt48',
    'moa-nosynth':    'agent_mnrvr4an9n0hmlfl',
    'best-of-3':      'agent_mnrvr457qvb4rt48',
  },
};

// ── Run ───────────────────────────────────────────────────────────────────────

const totalCells = tasks.length * protocols.length * seeds.length;
console.log('Composable Benchmark: 2×2 Diversity Ablation (Set Cover + Knapsack)');
console.log(`Tasks:     ${tasks.map(t => t.id).join(', ')}`);
console.log(`Protocols: ${protocols.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${seeds.join(', ')}`);
console.log(`Budget:    T=${BUDGET.tokenCap}, W=${BUDGET.wallClockSeconds}s, C=${BUDGET.evalCpuSeconds}s, K=${BUDGET.evalCallCap}`);
console.log(`Total:     ${totalCells} cells`);
console.log(`OutputDir: results/composable-2x2`);
console.log();

const missingIds = tasks.filter(t => !t.challengeId);
if (missingIds.length > 0) {
  console.error('ERROR: Missing challenge IDs for:', missingIds.map(t => t.id).join(', '));
  process.exit(1);
}

if (dryRun) {
  console.log('[DRY RUN] Config valid. Exiting without running experiments.');
  process.exit(0);
}

if (!process.env.ADMIN_SECRET) {
  console.error('ERROR: ADMIN_SECRET env var required');
  process.exit(1);
}

runExperiment(config).then(() => {
  console.log('\nDone. Results in results/composable-2x2/');
  console.log('Next: backfill hidden scores with:');
  console.log('  PLATFORM_URL=https://agent4science.org ADMIN_SECRET=xxx \\');
  console.log('  npx tsx src/runner/backfill-hidden-scores.ts --resultsDir results/composable-2x2');
}).catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
