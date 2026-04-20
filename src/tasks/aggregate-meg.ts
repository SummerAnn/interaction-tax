#!/usr/bin/env npx tsx
/**
 * Aggregate MEG from a directory of per-cell RunResult JSONs.
 *
 * Re-uses computeMEG / computeAggregateMEG / writeMEGTable from
 * experiment-runner.ts so the aggregation logic matches what a live run
 * would produce. Intended to run on the salvaged Run 1 data, but works on
 * any directory of per-cell JSONs conforming to RunResult.
 *
 * Usage:
 *   npx tsx src/tasks/aggregate-meg.ts --dir results/full-v1-salvaged
 *   npx tsx src/tasks/aggregate-meg.ts --dir results/full-v1-salvaged --out results/full-v1-salvaged
 *
 * Missing cells are silently dropped from MEG math (no back-fill).
 * Cells with hiddenScore == null are also dropped by computeMEG itself.
 */

import { readFileSync, readdirSync, existsSync } from 'fs';
import { resolve, join, dirname } from 'path';
import { fileURLToPath } from 'url';

import type { RunResult, BenchmarkTask, ProtocolConfig, ExperimentConfig, BudgetVector } from '../types.js';
import {
  computeMEG,
  computeAggregateMEG,
  writeMEGTable,
  printMEGSummary,
} from '../runner/experiment-runner.js';
import { ALL_BENCHMARK_CHALLENGES } from './benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from './challenge-ids.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ── CLI ──

const args = process.argv.slice(2);
function argVal(flag: string): string | undefined {
  const inline = args.find(a => a.startsWith(`${flag}=`));
  if (inline) return inline.slice(flag.length + 1);
  const i = args.indexOf(flag);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : undefined;
}

const dir = resolve(argVal('--dir') ?? resolve(__dirname, '../../results/full-v1-salvaged'));
const outDir = resolve(argVal('--out') ?? dir);

// ── Load cells ──

function loadRunResults(d: string): RunResult[] {
  if (!existsSync(d)) {
    console.error(`Directory not found: ${d}`);
    process.exit(1);
  }
  const files = readdirSync(d).filter(f => f.endsWith('.json') && f !== 'salvage-index.json' && f !== 'results.json');
  const runs: RunResult[] = [];
  for (const f of files) {
    try {
      const r = JSON.parse(readFileSync(join(d, f), 'utf-8')) as RunResult;
      if (r && r.taskId && r.protocolId) runs.push(r);
    } catch (err) {
      console.warn(`  skip ${f}: ${err}`);
    }
  }
  return runs;
}

// ── Synthetic config ──

/**
 * Build a synthetic ExperimentConfig that covers only the (task, protocol)
 * pairs actually present in the loaded runs. computeMEG iterates over
 * config.tasks × config.protocols, so restricting the config avoids
 * emitting MEG rows for tasks we have no data for.
 */
function buildConfig(runs: RunResult[]): ExperimentConfig {
  const taskIds = [...new Set(runs.map(r => r.taskId))];
  const protocolIds = [...new Set(runs.map(r => r.protocolId))];

  const tasks: BenchmarkTask[] = taskIds.map(id => {
    const def = ALL_BENCHMARK_CHALLENGES.find(c => c.id === id);
    if (!def) throw new Error(`Unknown task: ${id}`);
    return {
      id: def.id,
      title: def.title,
      track: 'scientific',
      prompt: def.description,
      devVerifier: def.verifier,
      solutionSchema: { raw: def.solutionSchema } as Record<string, unknown>,
      scoringDirection: def.scoringDirection,
      challengeId: BENCHMARK_CHALLENGE_IDS[id],
    };
  });

  const protocols: ProtocolConfig[] = protocolIds.map(id => ({
    id,
    name: id,
    backbones: [],
    params: {},
  }));

  const budget: BudgetVector = {
    tokenCap: 200000,
    wallClockSeconds: 600,
    evalCpuSeconds: 30,
    evalCallCap: 25,
  };

  return {
    id: 'salvage-aggregation',
    name: 'Agent4Science-Bench Salvage Aggregation',
    tasks,
    protocols,
    budget,
    seeds: [1, 2, 3, 4, 5],
    outputDir: outDir,
  };
}

// ── Main ──

function main() {
  console.log('Agent4Science-Bench — Aggregate MEG from Salvaged Cells');
  console.log(`Input dir:  ${dir}`);
  console.log(`Output dir: ${outDir}`);
  console.log('---');

  const runs = loadRunResults(dir);
  console.log(`Loaded ${runs.length} RunResult cells`);

  const withHidden = runs.filter(r => typeof r.hiddenScore === 'number');
  console.log(`Cells with hiddenScore: ${withHidden.length} / ${runs.length}`);

  if (withHidden.length === 0) {
    console.error('No cells have hiddenScore populated — cannot compute MEG.');
    process.exit(2);
  }

  const config = buildConfig(runs);
  console.log(`Synthetic config: ${config.tasks.length} tasks × ${config.protocols.length} protocols`);

  const meg = computeMEG(runs, config);
  const aggregate = computeAggregateMEG(meg);

  writeMEGTable(meg, aggregate, outDir);
  console.log(`\nWrote ${join(outDir, 'meg_table.tsv')}`);

  printMEGSummary(meg, aggregate);
}

main();
