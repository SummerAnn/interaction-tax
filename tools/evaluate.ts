#!/usr/bin/env tsx
/**
 * evaluate.ts — Standalone offline evaluation pipeline
 *
 * Reads result JSON files from a directory, runs dev and hidden verifiers
 * locally on each saved solution, and prints a score table.
 *
 * Use this to:
 *   - Reproduce all reported scores from scratch (no platform access needed)
 *   - Verify dev/hidden score parity on existing result files
 *   - Populate hiddenScore on result files produced by run-offline.ts
 *
 * Usage:
 *   npx tsx tools/evaluate.ts --results results/full-v2
 *   npx tsx tools/evaluate.ts --results results/offline --write   # write scores back to JSON
 *   npx tsx tools/evaluate.ts --results results/full-v2 --task bench-maxcut-g200
 *   npx tsx tools/evaluate.ts --results results/full-v2 --verifier hidden  # hidden only
 *   npx tsx tools/evaluate.ts --results results/full-v2 --verifier dev     # dev only
 */

import { readFileSync, writeFileSync, readdirSync } from 'fs';
import { join } from 'path';
import { runPythonVerifier } from '../src/lib/local-evaluator.js';
import { ALL_BENCHMARK_CHALLENGES } from '../src/tasks/benchmark-challenges.js';
import type { RunResult } from '../src/types.js';

// ── CLI args ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);

function getArg(flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 ? args[i + 1] : undefined;
}

const resultsDir = getArg('--results');
const taskFilter  = getArg('--task');
const verifierMode = (getArg('--verifier') ?? 'both') as 'dev' | 'hidden' | 'both';
const writeBack   = args.includes('--write');

if (!resultsDir) {
  console.error('Usage: npx tsx tools/evaluate.ts --results <dir> [--task <id>] [--verifier dev|hidden|both] [--write]');
  process.exit(1);
}

// ── Build verifier lookup by task ID ─────────────────────────────────────────

const challengeById = new Map(ALL_BENCHMARK_CHALLENGES.map(c => [c.id, c]));

// ── Load result files ─────────────────────────────────────────────────────────

const files = readdirSync(resultsDir)
  .filter(f => f.endsWith('.json') && f !== 'results.json' && f !== 'meg_table.tsv')
  .sort();

if (files.length === 0) {
  console.error(`No result JSON files found in ${resultsDir}`);
  process.exit(1);
}

// ── Evaluate ──────────────────────────────────────────────────────────────────

interface Row {
  file: string;
  taskId: string;
  protocolId: string;
  seed: number;
  devScoreRecorded: number | null;
  devScoreRerun: number | null;
  hiddenScoreRecorded: number | null;
  hiddenScoreRerun: number | null;
  error: string | null;
}

const rows: Row[] = [];
let processed = 0;

console.log(`Evaluating ${files.length} result files in ${resultsDir} ...`);
if (verifierMode !== 'both') console.log(`Verifier mode: ${verifierMode} only`);
console.log();

for (const file of files) {
  const filepath = join(resultsDir, file);
  let result: RunResult;
  try {
    result = JSON.parse(readFileSync(filepath, 'utf-8')) as RunResult;
  } catch {
    console.error(`  SKIP (parse error): ${file}`);
    continue;
  }

  if (taskFilter && result.taskId !== taskFilter) continue;

  const challenge = challengeById.get(result.taskId);
  if (!challenge) {
    console.warn(`  SKIP (unknown task ${result.taskId}): ${file}`);
    continue;
  }

  const solutionData = result.bestArtifact?.solutionData;
  if (!solutionData) {
    console.warn(`  SKIP (no bestArtifact): ${file}`);
    continue;
  }

  const row: Row = {
    file,
    taskId: result.taskId,
    protocolId: result.protocolId,
    seed: result.seed,
    devScoreRecorded: result.bestArtifact.devScore ?? null,
    devScoreRerun: null,
    hiddenScoreRecorded: result.hiddenScore ?? null,
    hiddenScoreRerun: null,
    error: null,
  };

  try {
    if (verifierMode === 'dev' || verifierMode === 'both') {
      row.devScoreRerun = await runPythonVerifier(challenge.verifier, solutionData);
    }
    if ((verifierMode === 'hidden' || verifierMode === 'both') && challenge.hiddenVerifier) {
      row.hiddenScoreRerun = await runPythonVerifier(challenge.hiddenVerifier, solutionData);
    }
  } catch (err) {
    row.error = err instanceof Error ? err.message : String(err);
  }

  rows.push(row);
  processed++;

  // Write scores back into the JSON if requested
  if (writeBack && row.hiddenScoreRerun !== null && !row.error) {
    result.hiddenScore = row.hiddenScoreRerun;
    writeFileSync(filepath, JSON.stringify(result, null, 2));
  }

  process.stdout.write(`\r  ${processed}/${files.length}`);
}

console.log('\n');

// ── Print table ───────────────────────────────────────────────────────────────

const header = [
  'task'.padEnd(30),
  'protocol'.padEnd(20),
  'seed'.padStart(4),
  'dev_rec'.padStart(10),
  'dev_rerun'.padStart(10),
  'hid_rec'.padStart(10),
  'hid_rerun'.padStart(10),
  'error',
].join('  ');

console.log(header);
console.log('-'.repeat(header.length));

for (const r of rows) {
  const fmt = (v: number | null) => v === null ? '       ---' : v.toFixed(4).padStart(10);
  console.log([
    r.taskId.padEnd(30),
    r.protocolId.padEnd(20),
    String(r.seed).padStart(4),
    fmt(r.devScoreRecorded),
    fmt(r.devScoreRerun),
    fmt(r.hiddenScoreRecorded),
    fmt(r.hiddenScoreRerun),
    r.error ?? '',
  ].join('  '));
}

// ── Summary ───────────────────────────────────────────────────────────────────

const withDev    = rows.filter(r => r.devScoreRerun !== null);
const withHidden = rows.filter(r => r.hiddenScoreRerun !== null);
const errors     = rows.filter(r => r.error !== null);

console.log(`\n${rows.length} results evaluated.`);
if (withDev.length)    console.log(`  Dev verifier rerun:    ${withDev.length} succeeded`);
if (withHidden.length) console.log(`  Hidden verifier rerun: ${withHidden.length} succeeded`);
if (errors.length)     console.log(`  Errors:                ${errors.length}`);
if (writeBack && withHidden.length > 0) {
  console.log(`  Written back to disk:  ${withHidden.filter(r => !r.error).length} files updated`);
}
