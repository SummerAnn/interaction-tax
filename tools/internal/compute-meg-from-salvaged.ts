#!/usr/bin/env tsx
/**
 * compute-meg-from-salvaged.ts — Compute MEG from already-salvaged Run 1 cells
 *
 * Run 2 crashed with OpenRouter 403s and overwrote Run 1's per-cell JSONs.
 * `src/tasks/salvage-run1.ts` recovered 114 cells (best-artifact + hiddenScore)
 * from run.log + Firestore into `results/full-v1-salvaged/`.
 *
 * This script loads those per-cell files and feeds them through the exported
 * `computeMEG` / `computeAggregateMEG` / `writeMEGTable` / `printMEGSummary`
 * functions from experiment-runner.ts — no LLM calls, no platform calls, no
 * budget tracking. Purely an offline statistics pass.
 *
 * Usage:
 *   npx tsx src/runner/compute-meg-from-salvaged.ts
 *
 * Output:
 *   results/full-v1-salvaged/meg_table.tsv
 *   stdout: printMEGSummary() table
 *
 * Notes:
 * - Cells without a hidden score are silently dropped by computeMEG
 *   (see `hiddenScoresFor` in experiment-runner.ts).
 * - Protocols or tasks with no baseline coverage are skipped — MEG is
 *   undefined when none of {SR, BoN, VGS} have hidden data.
 * - The ExperimentConfig we build here is a stub: computeMEG only reads
 *   config.tasks[*].id and config.protocols[*].id, so backbones/apiKey
 *   are left empty.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';

import {
  computeMEG,
  computeAggregateMEG,
  writeMEGTable,
  printMEGSummary,
} from './experiment-runner.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type {
  BenchmarkTask,
  ProtocolConfig,
  ProtocolId,
  RunResult,
  ExperimentConfig,
} from '../types.js';

const SALVAGE_DIR = 'results/full-v1-salvaged';

// ── Load runs ────────────────────────────────────────────────────────────────

function loadSalvagedRuns(dir: string): RunResult[] {
  const files = readdirSync(dir).filter(
    f => f.endsWith('.json') && f !== 'salvage-index.json'
  );
  const runs: RunResult[] = [];
  for (const f of files) {
    const raw = readFileSync(join(dir, f), 'utf-8');
    const parsed = JSON.parse(raw) as RunResult;
    // Sanity check: salvaged per-cell file has taskId + protocolId
    if (!parsed.taskId || !parsed.protocolId) {
      console.warn(`skip ${f}: missing taskId/protocolId`);
      continue;
    }
    runs.push(parsed);
  }
  return runs;
}

// ── Build stub ExperimentConfig ──────────────────────────────────────────────

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

const ALL_PROTOCOL_IDS: ProtocolId[] = [
  'single-shot',
  'self-refine',
  'best-of-n',
  'vgs',
  'homo-chain',
  'cross-chain',
  'magicore',
  'debate',
  'moa',
];

const stubProtocols: ProtocolConfig[] = ALL_PROTOCOL_IDS.map(id => ({
  id,
  name: id,
  backbones: [], // computeMEG doesn't touch backbones
  params: {},
}));

const config: ExperimentConfig = {
  id: 'full-v1-salvaged-meg',
  name: 'MEG on salvaged Run 1 cells',
  tasks: ALL_BENCHMARK_CHALLENGES.map(challengeToTask),
  protocols: stubProtocols,
  budget: {
    tokenCap: 0,
    wallClockSeconds: 0,
    evalCpuSeconds: 0,
    evalCallCap: 0,
  },
  seeds: [1, 2, 3, 4, 5],
  outputDir: SALVAGE_DIR,
};

// ── Main ─────────────────────────────────────────────────────────────────────

const runs = loadSalvagedRuns(SALVAGE_DIR);
console.log(`Loaded ${runs.length} salvaged runs from ${SALVAGE_DIR}/`);

// Coverage breakdown: how many cells per (task, protocol) have hiddenScore
const coverage = new Map<string, number>();
for (const r of runs) {
  if (typeof r.hiddenScore === 'number') {
    const key = `${r.taskId}::${r.protocolId}`;
    coverage.set(key, (coverage.get(key) ?? 0) + 1);
  }
}

console.log('\n── Coverage (cells with hiddenScore) ──');
console.log('task'.padEnd(30) + ALL_PROTOCOL_IDS.map(p => p.slice(0, 8).padStart(9)).join(''));
for (const task of config.tasks) {
  const row = ALL_PROTOCOL_IDS
    .map(pid => String(coverage.get(`${task.id}::${pid}`) ?? 0).padStart(9))
    .join('');
  console.log(task.id.padEnd(30) + row);
}

const meg = computeMEG(runs, config);
const aggregate = computeAggregateMEG(meg);

writeMEGTable(meg, aggregate, SALVAGE_DIR);
printMEGSummary(meg, aggregate);

console.log(`\nWrote ${SALVAGE_DIR}/meg_table.tsv`);
