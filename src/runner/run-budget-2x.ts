#!/usr/bin/env tsx
/**
 * run-budget-2x.ts — Budget-scaling experiment (2× budget)
 *
 * Runs key protocols at 2× the standard budget on discriminative tasks only.
 * Purpose: address M11 reviewer concern — distinguish capability limits
 * from budget limits for the negative MEG finding.
 *
 * Budget: 2× standard = (400K tokens, 1200s wall, 60s eval CPU, 50 eval calls)
 * Tasks: DiffBases, Erdős, MolQED (the 3 discriminative tasks)
 * Protocols: single-shot, self-refine, best-of-n, vgs, moa (5 key protocols)
 * Seeds: 1-5
 * Total: 3 × 5 × 5 = 75 cells
 *
 * Usage:
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-budget-2x.ts
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-budget-2x.ts --dry-run
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-budget-2x.ts --protocol moa
 */

import 'dotenv/config';
import { runExperiment } from './experiment-runner.js';
import { BENCHMARK_SEEDS } from '../config/seeds.js';
import { primaryBackbone, moaBackbones } from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

// ── CLI args ─────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const protocolIdx = args.indexOf('--protocol');
const protocolFilter = protocolIdx >= 0
  ? args[protocolIdx + 1].split(',').map(s => s.trim()).filter(Boolean)
  : undefined;
const seedsIdx = args.indexOf('--seeds');
const seedsFilter = seedsIdx >= 0
  ? args[seedsIdx + 1].split(',').map(s => Number(s.trim())).filter(n => Number.isFinite(n))
  : undefined;

// ── Discriminative tasks only ────────────────────────────────────────────────

const DISCRIMINATIVE_TASK_IDS = [
  'bench-difference-bases',
  'bench-erdos-overlap',
  'bench-molecule-qed',
];

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

const tasks: BenchmarkTask[] = ALL_BENCHMARK_CHALLENGES
  .filter(ch => DISCRIMINATIVE_TASK_IDS.includes(ch.id))
  .map(challengeToTask);

// ── Key protocols for budget-scaling ─────────────────────────────────────────

const ALL_PROTOCOLS: ProtocolConfig[] = [
  {
    id: 'single-shot-2x',
    name: 'Single Shot (2× budget)',
    backbones: [primaryBackbone()],
    params: {},
  },
  {
    id: 'self-refine-2x',
    name: 'Self-Refine (2× budget, 6 rounds)',
    backbones: [primaryBackbone()],
    params: { rounds: 6 },  // 2× rounds to use budget
  },
  {
    id: 'best-of-n-2x',
    name: 'Best-of-N (2× budget, N=16)',
    backbones: [primaryBackbone()],
    params: { n: 16 },  // 2× samples
  },
  {
    id: 'vgs-2x',
    name: 'VGS (2× budget, pop=4, gen=10)',
    backbones: [primaryBackbone()],
    params: { populationSize: 4, generations: 10, eliteCount: 2, convergenceWindow: 2 },
  },
  {
    id: 'moa-2x',
    name: 'MoA (2× budget, Claude+GPT-4o+Gemini)',
    backbones: moaBackbones(),
    params: {},
  },
];

const protocols = protocolFilter
  ? ALL_PROTOCOLS.filter(p => protocolFilter.includes(p.id))
  : ALL_PROTOCOLS;

// ── 2× Budget vector ────────────────────────────────────────────────────────

const BUDGET_2X = {
  tokenCap: 400_000,        // 2× T
  wallClockSeconds: 1200,   // 2× W
  evalCpuSeconds: 60,       // 2× C
  evalCallCap: 50,          // 2× K
};

// ── Experiment config ───────────────────────────────────────────────────────

const seeds: number[] = seedsFilter ?? [...BENCHMARK_SEEDS];

const config: ExperimentConfig = {
  id: 'agent4science-bench-budget-2x',
  name: 'Budget Scaling (2× budget on discriminative tasks)',
  tasks,
  protocols,
  budget: BUDGET_2X,
  seeds,
  outputDir: 'results/budget-2x',
  submitForHiddenEval: true,
  apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
  apiKey: process.env.ADMIN_SECRET,
  agentId: '',
  protocolAgentMap: {
    'single-shot-2x':  '',
    'self-refine-2x':  '',
    'best-of-n-2x':    '',
    'vgs-2x':          '',
    'moa-2x':          '',
  },
};

// ── Run ─────────────────────────────────────────────────────────────────────

const totalCells = tasks.length * protocols.length * seeds.length;
console.log('Budget Scaling Experiment (2× budget)');
console.log(`Tasks:     ${tasks.map(t => t.id).join(', ')}`);
console.log(`Protocols: ${protocols.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${seeds.join(', ')}`);
console.log(`Budget:    T=${BUDGET_2X.tokenCap}, W=${BUDGET_2X.wallClockSeconds}s, C=${BUDGET_2X.evalCpuSeconds}s, K=${BUDGET_2X.evalCallCap}`);
console.log(`Total:     ${totalCells} cells`);

if (dryRun) {
  console.log('\n[DRY RUN] Exiting without running experiments.');
  process.exit(0);
}

runExperiment(config).catch(err => {
  console.error('Experiment failed:', err);
  process.exit(1);
});
