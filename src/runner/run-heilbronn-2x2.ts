#!/usr/bin/env tsx
/**
 * run-heilbronn-2x2.ts — 2×2 diversity ablation for Heilbronn n=12
 *
 * Heilbronn shows CV=1.369 in existing runs (similar to Erdős at 1.374),
 * making it a strong candidate for a 4th discriminative task. This runner
 * executes the standard diversity × synthesis 2×2 factorial (N=3 proposals,
 * 10 seeds) to test whether model-family diversity helps on this HIGH-composability
 * task: different agents find different local point configurations, and those
 * can sometimes be merged geometrically.
 *
 * Cells: 1 task × 4 protocols × 10 seeds = 40 cells
 * Output: results/heilbronn-2x2/
 *
 * Usage:
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-heilbronn-2x2.ts
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-heilbronn-2x2.ts --dry-run
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

// ── Task ──────────────────────────────────────────────────────────────────────

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
  };
}

const heilbronnChallenge = ALL_BENCHMARK_CHALLENGES.find(c => c.id === 'bench-heilbronn-n12');
if (!heilbronnChallenge) {
  console.error('ERROR: bench-heilbronn-n12 not found in benchmark-challenges.ts');
  process.exit(1);
}

const tasks: BenchmarkTask[] = [challengeToTask(heilbronnChallenge)];

// ── Protocols — standard 2×2 ──────────────────────────────────────────────────

const protocols: ProtocolConfig[] = [
  {
    id: 'moa',
    name: 'MoA (diverse models + synthesis)',
    backbones: moaBackbones(),
    params: {},
  },
  {
    id: 'moa-same-model',
    name: 'MoA-Same-Model (same model + synthesis)',
    backbones: moaBackbonesSameModel(),
    params: {},
  },
  {
    id: 'moa-nosynth',
    name: 'MoA-NoSynth (diverse models, selection only)',
    backbones: moaBackbones(),
    params: { noSynth: true },
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
  id: 'bench-heilbronn-2x2-diversity',
  name: 'Heilbronn n=12: 2×2 Diversity Ablation',
  tasks,
  protocols,
  budget: BUDGET,
  seeds,
  outputDir: 'results/heilbronn-2x2',
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
console.log('Heilbronn n=12: 2×2 Diversity Ablation');
console.log(`Task:      ${tasks[0].id} (challengeId: ${tasks[0].challengeId})`);
console.log(`Protocols: ${protocols.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${seeds.join(', ')}`);
console.log(`Budget:    T=${BUDGET.tokenCap}, W=${BUDGET.wallClockSeconds}s`);
console.log(`Total:     ${totalCells} cells`);
console.log(`Output:    results/heilbronn-2x2/`);
console.log();

if (!tasks[0].challengeId) {
  console.error('ERROR: Missing challenge ID for bench-heilbronn-n12');
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
  console.log('\nDone. Results in results/heilbronn-2x2/');
  process.exit(0);
}).catch(err => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
