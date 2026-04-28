#!/usr/bin/env tsx
/**
 * run-crossover-2x2.ts — Evolutionary crossover protocol experiment
 *
 * Tests the evolutionary crossover protocol (W7 fix) against the standard
 * 2×2 factorial protocols on three discriminative tasks (DiffBases, Erdős,
 * TSP-50) and the two new HIGH-composability tasks (Knapsack-50, Set Cover).
 *
 * The crossover protocol is structurally different from all existing protocols:
 *   - Not consensus-building (unlike debate, MAgICoRe)
 *   - Not one-shot synthesis (unlike MoA)
 *   - Explicit component-level recombination (genetic crossover analogy)
 *
 * If crossover achieves positive MIG on composable tasks but not non-composable
 * tasks, this strengthens the composability hypothesis: the key is not whether
 * interaction occurs, but whether the interaction exploits structural composability.
 *
 * Protocols tested:
 *   - crossover:        Phase 1 (independent) + Phase 2 (recombination), N=3
 *   - crossover-refine: Phase 1 + 2 + 3 (local improvement), higher budget
 *   - moa:              Standard MoA diverse (comparison baseline)
 *   - moa-nosynth:      Best-of-3 diverse (comparison baseline)
 *
 * Tasks: DiffBases, Erdős, TSP-50, Knapsack-50 (if seeded), Set Cover (if seeded)
 * Seeds: 1-10
 *
 * Usage:
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-crossover-2x2.ts
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-crossover-2x2.ts --dry-run
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-crossover-2x2.ts --task bench-difference-bases
 */

import 'dotenv/config';
import { runExperiment } from './experiment-runner.js';
import {
  primaryBackbone,
  moaBackbones,
} from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

// ── CLI ───────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const taskIdx = args.indexOf('--task');
const taskFilter = taskIdx >= 0 ? args[taskIdx + 1].split(',').map(s => s.trim()) : undefined;

// ── Tasks ─────────────────────────────────────────────────────────────────────

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

// Primary discriminative tasks + new composable tasks
const CROSSOVER_TASK_IDS = [
  'bench-difference-bases',  // HIGH composability, good diversity spread
  'bench-erdos-overlap',     // HIGH composability, strongest diversity signal
  'bench-tsp-50',            // LOW composability, null diversity (should show crossover fails)
  'bench-knapsack-50',       // HIGH composability (requires platform seeding)
  'bench-set-cover-60',      // HIGH composability (requires platform seeding)
];

const tasks: BenchmarkTask[] = ALL_BENCHMARK_CHALLENGES
  .filter(ch => {
    if (taskFilter) return taskFilter.includes(ch.id);
    return CROSSOVER_TASK_IDS.includes(ch.id);
  })
  .map(challengeToTask)
  .filter(t => {
    if (!t.challengeId) {
      console.warn(`SKIP: ${t.id} — no challenge ID (seed to platform first)`);
      return false;
    }
    return true;
  });

if (tasks.length === 0) {
  console.error('ERROR: No tasks with challenge IDs. Seed new tasks to platform first.');
  process.exit(1);
}

// ── Protocols ─────────────────────────────────────────────────────────────────

const protocols: ProtocolConfig[] = [
  // Crossover protocols — the W7 fix
  {
    id: 'crossover',
    name: 'Evolutionary Crossover (diverse, N=3 + recombination)',
    backbones: moaBackbones(),
    params: {},
  },
  {
    id: 'crossover-refine',
    name: 'Evolutionary Crossover + Refine (diverse, N=3 + recombine + local improve)',
    backbones: moaBackbones(),
    params: {},
  },
  // Standard comparisons
  {
    id: 'moa',
    name: 'MoA Diverse (comparison)',
    backbones: moaBackbones(),
    params: {},
  },
  {
    id: 'moa-nosynth',
    name: 'MoA No-Synth / Best-of-3-Diverse (comparison)',
    backbones: moaBackbones(),
    params: {},
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
  id: 'bench-crossover-2x2',
  name: 'Evolutionary Crossover Protocol Experiment',
  tasks,
  protocols,
  budget: BUDGET,
  seeds,
  outputDir: 'results/crossover',
  submitForHiddenEval: true,
  apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
  apiKey: process.env.ADMIN_SECRET,
  agentId: 'agent_mnrvr457qvb4rt48',
  protocolAgentMap: {
    'crossover':        'agent_mnrvr457qvb4rt48',
    'crossover-refine': 'agent_mnrvr457qvb4rt48',
    'moa':              'agent_mnrvr457qvb4rt48',
    'moa-nosynth':      'agent_mnrvr4an9n0hmlfl',
  },
};

// ── Run ───────────────────────────────────────────────────────────────────────

const totalCells = tasks.length * protocols.length * seeds.length;
console.log('Evolutionary Crossover Protocol Experiment');
console.log(`Tasks:     ${tasks.map(t => t.id).join(', ')}`);
console.log(`Protocols: ${protocols.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${seeds.join(', ')}`);
console.log(`Budget:    T=${BUDGET.tokenCap}, W=${BUDGET.wallClockSeconds}s`);
console.log(`Total:     ${totalCells} cells`);
console.log();
console.log('NOTE: Crossover tests W7 (adaptive composition) from the NeurIPS review.');
console.log('      Results in results/crossover/ — analyze separately from main 2×2.');
console.log();

if (dryRun) {
  console.log('[DRY RUN] Config valid. Exiting.');
  process.exit(0);
}

if (!process.env.ADMIN_SECRET) {
  console.error('ERROR: ADMIN_SECRET env var required');
  process.exit(1);
}

runExperiment(config).then(() => {
  console.log('\nDone. Analyze with: node -e "..." results/crossover/');
  process.exit(0);
}).catch(err => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
