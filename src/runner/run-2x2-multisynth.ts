#!/usr/bin/env tsx
/**
 * run-2x2-multisynth.ts — Multi-synthesizer 2×2 expansion
 *
 * Runs the synthesis cells of the 2×2 diversity ablation with GPT-4o and
 * Gemini as synthesizers (instead of the default Claude). Tests whether the
 * diversity advantage holds regardless of which model does the synthesis.
 *
 * Design:
 *   - Same 2×2 structure: diversity (same/diverse) × aggregation (synthesis/selection)
 *   - Selection cells (moa-nosynth, best-of-3) already exist and are reused
 *   - Only the synthesis cells need re-running with different synthesizers
 *   - 4 new cells × 3 tasks × 10 seeds = 120 runs
 *
 * Usage:
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-2x2-multisynth.ts
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-2x2-multisynth.ts --dry-run
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-2x2-multisynth.ts --protocol moa-synth-gpt4o
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-2x2-multisynth.ts --seeds 1,2,3
 */

import 'dotenv/config';
import { runExperiment } from './experiment-runner.js';
import { moaBackbones, moaBackbonesSameModel, gpt4oBackbone, geminiBackbone } from '../config/models.js';
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

// ── Discriminative tasks (same 3 tasks as the original 2×2) ──────────────────

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

// ── Multi-synthesizer protocols ──────────────────────────────────────────────
//
// 4 cells: {diverse, same} × {GPT-4o synth, Gemini synth}
//
// Proposer backbones stay the same as the original 2×2:
//   diverse = Claude + GPT-4o + Gemini (moaBackbones)
//   same    = Claude × 3 (moaBackbonesSameModel)
//
// Only the aggregator/synthesizer model changes.

const ALL_PROTOCOLS: ProtocolConfig[] = [
  // GPT-4o as synthesizer
  {
    id: 'moa-synth-gpt4o',
    name: 'MoA Diverse → GPT-4o Synth',
    backbones: moaBackbones(),              // Claude, GPT-4o, Gemini proposers
    params: { aggregatorBackbone: gpt4oBackbone() },
  },
  {
    id: 'moa-same-synth-gpt4o',
    name: 'MoA Same → GPT-4o Synth',
    backbones: moaBackbonesSameModel(),     // 3× Claude proposers
    params: { aggregatorBackbone: gpt4oBackbone() },
  },
  // Gemini as synthesizer
  {
    id: 'moa-synth-gemini',
    name: 'MoA Diverse → Gemini Synth',
    backbones: moaBackbones(),              // Claude, GPT-4o, Gemini proposers
    params: { aggregatorBackbone: geminiBackbone() },
  },
  {
    id: 'moa-same-synth-gemini',
    name: 'MoA Same → Gemini Synth',
    backbones: moaBackbonesSameModel(),     // 3× Claude proposers
    params: { aggregatorBackbone: geminiBackbone() },
  },
];

const protocols = protocolFilter
  ? ALL_PROTOCOLS.filter(p => protocolFilter.includes(p.id))
  : ALL_PROTOCOLS;

if (protocols.length === 0) {
  console.error(`No protocols matched filter: ${protocolFilter}`);
  process.exit(1);
}

// ── Standard 1× budget (same as original 2×2) ──────────────────────────────

const BUDGET = {
  tokenCap: 200_000,
  wallClockSeconds: 600,
  evalCpuSeconds: 30,
  evalCallCap: 25,
};

// ── Seeds: 1-10 (same as original 2×2 MoA arms) ────────────────────────────

const seeds: number[] = seedsFilter ?? [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// ── Experiment config ────────────────────────────────────────────────────────

const config: ExperimentConfig = {
  id: 'agent4science-bench-2x2-multisynth',
  name: 'Multi-Synthesizer 2×2 Expansion',
  tasks,
  protocols,
  budget: BUDGET,
  seeds,
  outputDir: 'results/2x2-multisynth',
  submitForHiddenEval: true,
  apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
  apiKey: process.env.ADMIN_SECRET,
  agentId: 'agent_mnrvr457qvb4rt48',
  protocolAgentMap: {
    'moa-synth-gpt4o':      'agent_mnrvr4an9n0hmlfl',
    'moa-same-synth-gpt4o': 'agent_mnrvr457qvb4rt48',
    'moa-synth-gemini':     'agent_mnrvr4an9n0hmlfl',
    'moa-same-synth-gemini':'agent_mnrvr457qvb4rt48',
  },
};

// ── Run ──────────────────────────────────────────────────────────────────────

const totalCells = tasks.length * protocols.length * seeds.length;
console.log('Multi-Synthesizer 2×2 Expansion');
console.log(`Tasks:     ${tasks.map(t => t.id).join(', ')}`);
console.log(`Protocols: ${protocols.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${seeds.join(', ')}`);
console.log(`Budget:    T=${BUDGET.tokenCap}, W=${BUDGET.wallClockSeconds}s, C=${BUDGET.evalCpuSeconds}s, K=${BUDGET.evalCallCap}`);
console.log(`Total:     ${totalCells} cells`);
console.log();

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
  process.exit(0);
}).catch(err => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
