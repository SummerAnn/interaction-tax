#!/usr/bin/env tsx
/**
 * run-prospective-2x2.ts — 2×2 diversity ablation for prospective composability tasks
 *
 * Runs the diversity × synthesis 2×2 factorial on the two new prospective tasks
 * (Max k-Coverage, Latin Square Completion) whose composability predictions were
 * pre-registered in composability_prediction_v2.md (2026-04-23).
 *
 * Design (Table 2):
 *   ┌─────────────┬───────────────┬──────────────────┐
 *   │             │ Same Model    │ Diverse Models   │
 *   ├─────────────┼───────────────┼──────────────────┤
 *   │ No Synthesis│ Best-of-3     │ MoA-NoSynth      │
 *   │ Synthesis   │ MoA-Same-Model│ MoA (diverse)    │
 *   └─────────────┴───────────────┴──────────────────┘
 *
 * Cells: 2 tasks × 4 protocols × 10 seeds = 80 cells
 *
 * Prediction (composability_prediction_v2.md):
 *   Max k-Coverage (HIGH): Δ_div > +0.10 (significant positive diversity effect)
 *   Latin Square (LOW):    Δ_div ≈ 0 or negative
 *
 * Usage:
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-prospective-2x2.ts
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-prospective-2x2.ts --dry-run
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-prospective-2x2.ts --task bench-max-k-coverage-n100
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

const PROSPECTIVE_TASK_IDS = ['bench-max-k-coverage-n100', 'bench-latin-square-9'];

const tasks: BenchmarkTask[] = ALL_BENCHMARK_CHALLENGES
  .filter(ch => {
    if (taskFilter) return ch.id === taskFilter;
    return PROSPECTIVE_TASK_IDS.includes(ch.id);
  })
  .map(ch => {
    const task = challengeToTask(ch);
    if (!task.challengeId) {
      console.warn(`WARNING: No challenge ID for ${ch.id} — register task first.`);
    }
    return task;
  });

if (tasks.length === 0) {
  console.error(`ERROR: No tasks found. Filter: ${taskFilter ?? PROSPECTIVE_TASK_IDS.join(', ')}`);
  console.error('Make sure tasks are in ALL_BENCHMARK_CHALLENGES and have challenge IDs in challenge-ids.ts.');
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

// ── Budget ────────────────────────────────────────────────────────────────────

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
  id: 'bench-prospective-2x2-composability',
  name: 'Prospective Tasks (Max k-Coverage, Latin Square) 2×2 Diversity Ablation',
  tasks,
  protocols,
  budget: BUDGET,
  seeds,
  outputDir: 'results/prospective-2x2',
  submitForHiddenEval: true,
  apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
  apiKey: process.env.ADMIN_SECRET,
  agentId: '',
  protocolAgentMap: {
    'moa':            '',
    'moa-same-model': '',
    'moa-nosynth':    '',
    'best-of-3':      '',
  },
};

// ── Run ───────────────────────────────────────────────────────────────────────

const totalCells = tasks.length * protocols.length * seeds.length;
console.log('Prospective Tasks: 2×2 Diversity Ablation (Max k-Coverage + Latin Square)');
console.log(`Tasks:     ${tasks.map(t => t.id).join(', ')}`);
console.log(`Protocols: ${protocols.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${seeds.join(', ')}`);
console.log(`Budget:    T=${BUDGET.tokenCap}, W=${BUDGET.wallClockSeconds}s, C=${BUDGET.evalCpuSeconds}s, K=${BUDGET.evalCallCap}`);
console.log(`Total:     ${totalCells} cells`);
console.log(`OutputDir: results/prospective-2x2`);
console.log();

const missingIds = tasks.filter(t => !t.challengeId);
if (missingIds.length > 0) {
  console.error('ERROR: Missing challenge IDs for:', missingIds.map(t => t.id).join(', '));
  console.error('Run register-prospective-tasks.ts first, then add IDs to challenge-ids.ts');
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
  console.log('\nDone.');
  console.log('Next: backfill hidden scores with:');
  console.log('  PLATFORM_URL=https://agent4science.org ADMIN_SECRET=xxx \\');
  console.log('  npx tsx src/runner/backfill-hidden-scores.ts --resultsDir results/prospective-2x2');
}).catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
