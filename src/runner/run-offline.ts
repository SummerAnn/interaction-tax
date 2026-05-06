#!/usr/bin/env tsx
/**
 * run-offline.ts — Fully offline benchmark entrypoint
 *
 * Runs the complete benchmark without touching agent4science.org.
 * Dev evaluation and hidden evaluation both run as local Python subprocesses.
 * No ADMIN_SECRET required — only LLM API keys.
 *
 * Hidden scores are written directly into each result JSON (no backfill step).
 *
 * Usage:
 *   ANTHROPIC_API_KEY=xxx npx tsx src/runner/run-offline.ts
 *   ANTHROPIC_API_KEY=xxx npx tsx src/runner/run-offline.ts --task bench-maxcut-g200
 *   ANTHROPIC_API_KEY=xxx npx tsx src/runner/run-offline.ts --protocol single-shot --seeds 1,2,3
 *   ANTHROPIC_API_KEY=xxx npx tsx src/runner/run-offline.ts --dry-run
 *
 * Required env vars:
 *   OPENROUTER_API_KEY   all models route through OpenRouter (Claude, GPT-4o, Gemini, etc.)
 */

import 'dotenv/config';
import { runExperiment } from './experiment-runner.js';
import { BENCHMARK_SEEDS } from '../config/seeds.js';
import {
  primaryBackbone, crossChainBackbones, moaBackbones, moaBackbonesSameModel,
  gpt4oBackbone, geminiBackbone, deepseekBackbone, crossChainOSSBackbones,
  fiveDiverseBackbones,
} from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import { LocalEvaluator } from '../lib/local-evaluator.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

// ── CLI args ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const taskIdx = args.indexOf('--task');
const taskFilter = taskIdx >= 0
  ? args[taskIdx + 1].split(',').map(s => s.trim()).filter(Boolean)
  : undefined;
const protocolIdx = args.indexOf('--protocol');
const protocolFilter = protocolIdx >= 0
  ? args[protocolIdx + 1].split(',').map(s => s.trim()).filter(Boolean)
  : undefined;
const seedsIdx = args.indexOf('--seeds');
const seedsFilter = seedsIdx >= 0
  ? args[seedsIdx + 1].split(',').map(s => Number(s.trim())).filter(n => Number.isFinite(n))
  : undefined;

// ── Task definitions ──────────────────────────────────────────────────────────

function normalizeMaxCut(data: Record<string, unknown>): Record<string, unknown> {
  const raw = data['partition'];
  if (!Array.isArray(raw)) return data;
  let arr = (raw as unknown[]).map((v) => (Number(v) >= 1 ? 1 : 0));
  if (arr.length > 200) arr = arr.slice(0, 200);
  while (arr.length < 200) arr.push(0);
  return { ...data, partition: arr };
}

const NORMALIZERS: Partial<Record<string, (d: Record<string, unknown>) => Record<string, unknown>>> = {
  'bench-maxcut-g200': normalizeMaxCut,
};

function challengeToTask(ch: typeof ALL_BENCHMARK_CHALLENGES[0]): BenchmarkTask {
  return {
    id: ch.id,
    title: ch.title,
    track: 'scientific',
    prompt: ch.description,
    devVerifier: ch.verifier,
    hiddenVerifier: ch.hiddenVerifier,
    solutionSchema: { format: ch.solutionSchema },
    scoringDirection: ch.scoringDirection,
    challengeId: BENCHMARK_CHALLENGE_IDS[ch.id],
    parameters: {},
    normalizeOutput: NORMALIZERS[ch.id],
  };
}

const ALL_TASKS: BenchmarkTask[] = ALL_BENCHMARK_CHALLENGES.map(challengeToTask);

const tasks = taskFilter
  ? ALL_TASKS.filter(t => taskFilter.includes(t.id))
  : ALL_TASKS;

if (tasks.length === 0) {
  console.error(`No tasks matched filter: ${taskFilter}`);
  process.exit(1);
}

// ── Protocol definitions ──────────────────────────────────────────────────────

const ALL_PROTOCOLS: ProtocolConfig[] = [
  { id: 'single-shot',   name: 'Single Shot',            backbones: [primaryBackbone()],      params: {} },
  { id: 'self-refine',   name: 'Self-Refine (3 rounds)', backbones: [primaryBackbone()],      params: { rounds: 3 } },
  { id: 'best-of-n',     name: 'Best-of-8',              backbones: [primaryBackbone()],      params: { n: 8 } },
  { id: 'vgs',           name: 'VGS',                    backbones: [primaryBackbone()],      params: { populationSize: 4, generations: 5, eliteCount: 2, convergenceWindow: 2 } },
  { id: 'homo-chain',    name: 'Homo-Chain (4 agents)',  backbones: [primaryBackbone()],      params: { chainLength: 4 } },
  { id: 'cross-chain',   name: 'Cross-Chain',            backbones: crossChainBackbones(),    params: {} },
  { id: 'magicore',      name: 'MAgICoRe',               backbones: [primaryBackbone()],      params: { rounds: 2 } },
  { id: 'debate',        name: 'Debate (2 rounds)',      backbones: [primaryBackbone()],      params: { debateRounds: 2 } },
  { id: 'moa',           name: 'MoA (diverse)',          backbones: moaBackbones(),           params: {} },
  { id: 'moa-nosynth',   name: 'MoA (no synthesis)',     backbones: moaBackbones(),           params: {} },
  { id: 'moa-same-model',name: 'MoA (same model)',       backbones: moaBackbonesSameModel(),  params: {} },
  { id: 'hpe',           name: 'HPE',                    backbones: [primaryBackbone()],      params: { numExecutors: 3, rounds: 2 } },
];

const protocols = protocolFilter
  ? ALL_PROTOCOLS.filter(p => protocolFilter.includes(p.id))
  : ALL_PROTOCOLS;

if (protocols.length === 0) {
  console.error(`No protocols matched filter: ${protocolFilter}`);
  process.exit(1);
}

// ── Build local evaluator ─────────────────────────────────────────────────────
// Maps challengeId → verifier code so protocols can call budget.evalDev()
// using the same task.challengeId key as the online runner.

const devVerifiers = new Map<string, string>();
for (const ch of ALL_BENCHMARK_CHALLENGES) {
  const cid = BENCHMARK_CHALLENGE_IDS[ch.id];
  if (cid) devVerifiers.set(cid, ch.verifier);
}

const localEvaluator = new LocalEvaluator(devVerifiers);

// ── Budget and seeds ──────────────────────────────────────────────────────────

const BUDGET = {
  tokenCap:         200_000,
  wallClockSeconds: 600,
  evalCpuSeconds:   30,
  evalCallCap:      25,
};

const seeds: number[] = seedsFilter ?? [...BENCHMARK_SEEDS];

// ── Experiment config ─────────────────────────────────────────────────────────

const config: ExperimentConfig = {
  id:                 'agent4science-bench-offline',
  name:               'the benchmark (Offline)',
  tasks,
  protocols,
  budget:             BUDGET,
  seeds,
  outputDir:          'results/offline',
  submitForHiddenEval: false,  // hidden eval runs locally via task.hiddenVerifier
};

// ── Dry run ───────────────────────────────────────────────────────────────────

const totalCells = tasks.length * protocols.length * seeds.length;
console.log('the benchmark — Offline Mode');
console.log(`Tasks:     ${tasks.map(t => t.id).join(', ')}`);
console.log(`Protocols: ${protocols.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${seeds.join(', ')}`);
console.log(`Budget:    T=${BUDGET.tokenCap} W=${BUDGET.wallClockSeconds}s C=${BUDGET.evalCpuSeconds}s K=${BUDGET.evalCallCap}`);
console.log(`Total:     ${totalCells} cells`);
console.log(`Platform:  local (no ADMIN_SECRET required)`);
console.log();

if (dryRun) {
  console.log('[DRY RUN] Config valid. Exiting without running experiments.');
  process.exit(0);
}

// ── Run ───────────────────────────────────────────────────────────────────────

runExperiment(config, localEvaluator).then(() => {
  console.log('\nDone.');
  process.exit(0);
}).catch(err => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
