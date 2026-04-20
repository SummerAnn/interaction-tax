#!/usr/bin/env tsx
/**
 * run-smoke4.ts — Four-cell smoke test for Phase-3 fixes.
 *
 * Validates the three zero-LLM-credit fixes from Phase 3 before committing to a
 * bigger backfill run:
 *
 *   Cell 1: cross-chain × difference-bases × seed 1
 *     → verifies Gemini model ID fix (models.ts: google/gemini-2.5-flash)
 *
 *   Cell 2: single-shot × circle-packing-n20 × seed 1
 *     → verifies Firestore nested-array fix (solutionDataJson fallback)
 *
 *   Cell 3: single-shot × lj-n41 × seed 1
 *     → verifies OpenRouter key is healthy (cheapest early-termination task)
 *
 *   Cell 4: moa × difference-bases × seed 1
 *     → verifies multi-model MoA flow works with the new Gemini ID
 *
 * Small budget to keep cost low (~5K tokens/cell). All 4 cells should submit
 * successfully to agent4science.org and return both dev + hidden scores.
 *
 * Usage:
 *   npx tsx src/runner/run-smoke4.ts
 */

import 'dotenv/config';

import { runExperiment } from './experiment-runner.js';
import {
  primaryBackbone,
  crossChainBackbones,
  moaBackbones,
} from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

// ── Pre-flight ───────────────────────────────────────────────────────────────

const missing: string[] = [];
if (!process.env.OPENROUTER_API_KEY) missing.push('OPENROUTER_API_KEY');
if (!process.env.ADMIN_SECRET) missing.push('ADMIN_SECRET');
if (missing.length > 0) {
  console.error(`FATAL: missing env vars: ${missing.join(', ')}`);
  process.exit(1);
}

function findChallenge(id: string) {
  const ch = ALL_BENCHMARK_CHALLENGES.find(c => c.id === id);
  if (!ch) {
    console.error(`FATAL: challenge ${id} not found`);
    process.exit(1);
  }
  return ch;
}

function challengeToTask(id: string): BenchmarkTask {
  const ch = findChallenge(id);
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

// ── Four smoke cells ──────────────────────────────────────────────────────────

const diffBasesTask = challengeToTask('bench-difference-bases');
const circlePackingTask = challengeToTask('bench-circle-packing-n20');
const ljTask = challengeToTask('bench-lj-n41');

const singleShotProto: ProtocolConfig = {
  id: 'single-shot',
  name: 'Single Shot',
  backbones: [primaryBackbone()],
  params: {},
};

const crossChainProto: ProtocolConfig = {
  id: 'cross-chain',
  name: 'Cross-Chain (Claude→GPT-4o→Gemini)',
  backbones: crossChainBackbones(),
  params: {},
};

const moaProto: ProtocolConfig = {
  id: 'moa',
  name: 'Mixture-of-Agents',
  backbones: moaBackbones(),
  params: {},
};

// The experiment-runner iterates tasks × protocols × seeds, so to get exactly
// 4 specific cells we split into four configs and run them sequentially.

async function runOne(label: string, task: BenchmarkTask, protocol: ProtocolConfig) {
  const config: ExperimentConfig = {
    id: `smoke4-${label}`,
    name: `Smoke4: ${label}`,
    tasks: [task],
    protocols: [protocol],
    budget: {
      tokenCap: 12_000,       // small — one or two LLM calls, not a full search
      wallClockSeconds: 300,
      evalCpuSeconds: 30,
      evalCallCap: 5,
    },
    seeds: [1],
    outputDir: `results/smoke4/${label}`,
    submitForHiddenEval: true,
    apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
    apiKey: process.env.ADMIN_SECRET,
    agentId: 'agent_mn9unhblb3o5swf2',
  };

  console.log(`\n── Running ${label}: ${task.id} × ${protocol.id} ──`);
  const t0 = Date.now();
  try {
    const results = await runExperiment(config);
    const run = results.runs[0];
    const elapsed = Math.round((Date.now() - t0) / 1000);

    if (!run) {
      return {
        label,
        task: task.id,
        protocol: protocol.id,
        status: 'NO_RUN',
        reason: 'cell was dropped (LLM failure, parse failure, or verifier rejection)',
        elapsed,
      };
    }

    const devScore = run.bestArtifact.devScore;
    const tokens = run.budgetTrace.tokenUsage.total;
    const evalCalls = run.budgetTrace.evalCalls.length;
    const submissionId = run.submissionId;

    return {
      label,
      task: task.id,
      protocol: protocol.id,
      status: submissionId ? 'PASS' : 'NO_SUBMISSION',
      devScore,
      tokens,
      evalCalls,
      submissionId,
      elapsed,
    };
  } catch (err) {
    const elapsed = Math.round((Date.now() - t0) / 1000);
    return {
      label,
      task: task.id,
      protocol: protocol.id,
      status: 'THROW',
      reason: err instanceof Error ? err.message : String(err),
      elapsed,
    };
  }
}

async function main() {
  console.log('══════════════════════════════════════════════════════════════');
  console.log('Smoke4 — Phase-3 fix validation');
  console.log('══════════════════════════════════════════════════════════════');

  const cells: Array<[string, BenchmarkTask, ProtocolConfig]> = [
    ['1-crosschain-diffbases',  diffBasesTask,     crossChainProto],
    ['2-single-circlepacking',  circlePackingTask, singleShotProto],
    ['3-single-lj',             ljTask,            singleShotProto],
    ['4-moa-diffbases',         diffBasesTask,     moaProto],
  ];

  const results = [];
  for (const [label, task, protocol] of cells) {
    results.push(await runOne(label, task, protocol));
  }

  console.log('\n══════════════════════════════════════════════════════════════');
  console.log('Smoke4 Summary');
  console.log('══════════════════════════════════════════════════════════════');

  for (const r of results) {
    const tag = r.status === 'PASS' ? '✓ PASS' : '✗ ' + r.status;
    console.log(`\n[${tag}] ${r.label}  (${r.elapsed}s)`);
    console.log(`  task=${r.task}  protocol=${r.protocol}`);
    if (r.status === 'PASS') {
      console.log(`  devScore=${r.devScore}  tokens=${r.tokens}/${12_000}  evalCalls=${r.evalCalls}`);
      console.log(`  submissionId=${r.submissionId}`);
    } else if ('reason' in r) {
      console.log(`  reason: ${r.reason}`);
    }
  }

  const passed = results.filter(r => r.status === 'PASS').length;
  console.log(`\n── ${passed} / ${results.length} smoke cells passed ──`);
  process.exit(passed === results.length ? 0 : 2);
}

main().catch(err => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
