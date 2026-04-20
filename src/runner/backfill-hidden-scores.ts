#!/usr/bin/env tsx
/**
 * backfill-hidden-scores.ts — Fetch hidden scores from Firestore and recompute MEG
 *
 * After a run, each per-cell JSON has hiddenScore=null because the API never
 * returns the hidden verifier score. This script:
 *   1. Reads all per-cell JSON files from outputDir
 *   2. Calls GET /api/v1/admin/benchmark/results for all challengeIds found
 *   3. Matches submissions by submissionId → backfills hiddenScore
 *   4. Writes updated per-cell files back to disk
 *   5. Recomputes MEG and writes meg_table.tsv
 *
 * Usage:
 *   PLATFORM_URL=https://agent4science.org ADMIN_SECRET=xxx \
 *     npx tsx src/runner/backfill-hidden-scores.ts --resultsDir results/run2
 *
 * Options:
 *   --resultsDir  Directory containing per-cell *.json files (default: results/run2)
 *   --dryRun      Print matches without writing files
 */

import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
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

// ── CLI args ──────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const resultsDirArg = args.find(a => a.startsWith('--resultsDir='))?.split('=')[1]
  ?? (args[args.indexOf('--resultsDir') + 1] ?? 'results/run2');
const dryRun = args.includes('--dryRun');

const PLATFORM_URL = process.env.PLATFORM_URL ?? 'https://agent4science.org';
const ADMIN_SECRET = process.env.ADMIN_SECRET;
if (!ADMIN_SECRET) {
  console.error('ADMIN_SECRET env var is required');
  process.exit(1);
}

// ── Load per-cell runs ────────────────────────────────────────────────────────

function loadRuns(dir: string): Array<{ file: string; run: RunResult }> {
  const files = readdirSync(dir).filter(
    f => f.endsWith('.json') && f !== 'results.json' && f !== 'salvage-index.json' && !f.startsWith('meg')
  );
  const out: Array<{ file: string; run: RunResult }> = [];
  for (const f of files) {
    try {
      const raw = readFileSync(join(dir, f), 'utf-8');
      const parsed = JSON.parse(raw) as RunResult;
      if (parsed.taskId && parsed.protocolId) {
        out.push({ file: join(dir, f), run: parsed });
      }
    } catch {
      console.warn(`  skip ${f}: parse error`);
    }
  }
  return out;
}

// ── Fetch hidden scores from platform ────────────────────────────────────────

interface AdminSubmission {
  id: string;
  challengeId: string;
  protocolId: string | null;
  seed: number | null;
  runIndex: number | null;
  devScore: number | null;
  hiddenScore: number | null;
  evalCpuMs: number | null;
  agentId: string | null;
  createdAt: string | null;
}

async function fetchHiddenScores(challengeIds: string[]): Promise<Map<string, number | null>> {
  const params = challengeIds.map(id => `challengeId=${encodeURIComponent(id)}`).join('&');
  const url = `${PLATFORM_URL}/api/v1/admin/benchmark/results?${params}`;

  const res = await fetch(url, {
    headers: { Authorization: `Bearer ${ADMIN_SECRET}` },
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Admin endpoint returned ${res.status}: ${text}`);
  }

  const json = await res.json() as { success: boolean; submissions: AdminSubmission[] };
  if (!json.success) throw new Error('Admin endpoint returned success=false');

  // Map submissionId → hiddenScore
  const map = new Map<string, number | null>();
  for (const sub of json.submissions) {
    map.set(sub.id, sub.hiddenScore);
  }
  return map;
}

// ── MEG config stub ───────────────────────────────────────────────────────────

const ALL_PROTOCOL_IDS: ProtocolId[] = [
  'single-shot', 'self-refine', 'best-of-n', 'vgs',
  'homo-chain', 'cross-chain', 'magicore', 'debate', 'moa',
];

function buildConfig(outputDir: string): ExperimentConfig {
  const tasks: BenchmarkTask[] = ALL_BENCHMARK_CHALLENGES.map(ch => ({
    id: ch.id,
    title: ch.title,
    track: 'scientific',
    prompt: ch.description,
    devVerifier: ch.verifier,
    solutionSchema: { format: ch.solutionSchema },
    scoringDirection: ch.scoringDirection,
    challengeId: BENCHMARK_CHALLENGE_IDS[ch.id],
    parameters: {},
  }));

  const protocols: ProtocolConfig[] = ALL_PROTOCOL_IDS.map(id => ({
    id, name: id, backbones: [], params: {},
  }));

  return {
    id: 'backfill-meg',
    name: 'MEG after hidden-score backfill',
    tasks,
    protocols,
    budget: { tokenCap: 0, wallClockSeconds: 0, evalCpuSeconds: 0, evalCallCap: 0 },
    seeds: [1, 2, 3, 4, 5],
    outputDir,
  };
}

// ── Main ──────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`Results dir : ${resultsDirArg}`);
  console.log(`Platform URL: ${PLATFORM_URL}`);
  console.log(`Dry run     : ${dryRun}\n`);

  // Load runs
  const entries = loadRuns(resultsDirArg);
  console.log(`Loaded ${entries.length} per-cell run files`);

  // Collect submissionIds and challengeIds
  const submissionIds = entries
    .map(e => e.run.submissionId)
    .filter((id): id is string => !!id);

  const challengeIds = [...new Set(
    entries.map(e => BENCHMARK_CHALLENGE_IDS[e.run.taskId]).filter(Boolean)
  )];

  console.log(`Unique challenges: ${challengeIds.length}`);
  console.log(`Runs with submissionId: ${submissionIds.length} / ${entries.length}`);

  if (challengeIds.length === 0) {
    console.error('No challenge IDs found — check BENCHMARK_CHALLENGE_IDS mapping');
    process.exit(1);
  }

  // Fetch hidden scores
  console.log('\nFetching hidden scores from platform...');
  const hiddenMap = await fetchHiddenScores(challengeIds);
  console.log(`Got ${hiddenMap.size} submissions from platform`);

  // Backfill
  let matched = 0;
  let missing = 0;
  let alreadyHad = 0;

  for (const { file, run } of entries) {
    if (!run.submissionId) { missing++; continue; }

    if (!hiddenMap.has(run.submissionId)) {
      console.warn(`  no platform record for submissionId=${run.submissionId}`);
      missing++;
      continue;
    }

    const newHidden = hiddenMap.get(run.submissionId) ?? null;

    if (typeof run.hiddenScore === 'number') {
      alreadyHad++;
      // Still update in case score changed (re-run of verifier)
    }

    run.hiddenScore = newHidden;
    matched++;

    if (!dryRun) {
      writeFileSync(file, JSON.stringify(run, null, 2));
    } else {
      console.log(`  [dry] ${run.taskId} / ${run.protocolId} / s${run.seed} → hiddenScore=${newHidden}`);
    }
  }

  console.log(`\nBackfill: matched=${matched}  already-had=${alreadyHad}  missing=${missing}`);

  if (dryRun) {
    console.log('\n[dry run] No files written.');
    return;
  }

  // Recompute MEG
  console.log('\nRecomputing MEG...');
  const runs = entries.map(e => e.run);
  const config = buildConfig(resultsDirArg);
  const meg = computeMEG(runs, config);
  const aggregate = computeAggregateMEG(meg);

  writeMEGTable(meg, aggregate, resultsDirArg);
  printMEGSummary(meg, aggregate);
  console.log(`\nWrote ${resultsDirArg}/meg_table.tsv`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
