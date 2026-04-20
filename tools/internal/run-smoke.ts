#!/usr/bin/env tsx
/**
 * run-smoke.ts — Minimal end-to-end smoke test
 *
 * One task × one protocol × one seed, tiny budget.
 * Purpose: verify the orchestrator + platform client + benchmark/evaluate
 * endpoint all work end-to-end before committing to a full re-run.
 *
 * Checks:
 *   1. dotenv loads OPENROUTER_API_KEY + ADMIN_SECRET from .env
 *   2. primaryBackbone() returns a non-empty apiKey
 *   3. runSingleShot actually calls the LLM and parses a solution
 *   4. POST /api/v1/benchmark/evaluate (dev + final) returns a score
 *   5. experiment-runner writes a real RunResult (not an empty placeholder)
 *
 * Usage:
 *   npx tsx src/runner/run-smoke.ts
 */

import 'dotenv/config';

import { runExperiment } from './experiment-runner.js';
import { primaryBackbone } from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

// Pre-flight env check — fail loud instead of silently producing empty JSONs
const missing: string[] = [];
if (!process.env.OPENROUTER_API_KEY) missing.push('OPENROUTER_API_KEY');
if (!process.env.ADMIN_SECRET) missing.push('ADMIN_SECRET');
if (missing.length > 0) {
  console.error(`FATAL: missing env vars: ${missing.join(', ')}`);
  console.error('Populate .env in the neurips-experiments/ root.');
  process.exit(1);
}

// MaxCut is the simplest task (200-int partition array), most likely to
// round-trip cleanly through LLM → JSON parse → verifier.
const SMOKE_CHALLENGE = ALL_BENCHMARK_CHALLENGES.find(c => c.id === 'bench-maxcut-g200');
if (!SMOKE_CHALLENGE) {
  console.error('FATAL: bench-maxcut-g200 not found in ALL_BENCHMARK_CHALLENGES');
  process.exit(1);
}

// MaxCut-specific output normalizer (copied from run-full.ts, kept inline
// to avoid importing from a module with top-level side effects).
function normalizeMaxCut(data: Record<string, unknown>): Record<string, unknown> {
  const raw = data['partition'];
  if (!Array.isArray(raw)) return data;
  let arr = (raw as unknown[]).map(v => (Number(v) >= 1 ? 1 : 0));
  if (arr.length > 200) arr = arr.slice(0, 200);
  while (arr.length < 200) arr.push(0);
  return { ...data, partition: arr };
}

const smokeTask: BenchmarkTask = {
  id: SMOKE_CHALLENGE.id,
  title: SMOKE_CHALLENGE.title,
  track: 'scientific',
  prompt: SMOKE_CHALLENGE.description,
  devVerifier: SMOKE_CHALLENGE.verifier,
  solutionSchema: { format: SMOKE_CHALLENGE.solutionSchema },
  scoringDirection: SMOKE_CHALLENGE.scoringDirection,
  challengeId: BENCHMARK_CHALLENGE_IDS[SMOKE_CHALLENGE.id],
  parameters: {},
  normalizeOutput: normalizeMaxCut,
};

const smokeProtocol: ProtocolConfig = {
  id: 'single-shot',
  name: 'Single Shot (smoke)',
  backbones: [primaryBackbone()],
  params: {},
};

// Assert backbone apiKey actually resolved — catches dotenv-ordering bugs.
if (!smokeProtocol.backbones[0].apiKey) {
  console.error('FATAL: primaryBackbone() returned empty apiKey. Is OPENROUTER_API_KEY set in .env?');
  process.exit(1);
}

const smokeConfig: ExperimentConfig = {
  id: 'smoke-v1',
  name: 'Smoke Test (MaxCut × single-shot × seed 1)',
  tasks: [smokeTask],
  protocols: [smokeProtocol],
  budget: {
    tokenCap: 10_000,
    wallClockSeconds: 300,
    evalCpuSeconds: 30,
    evalCallCap: 5,
  },
  seeds: [1],
  outputDir: 'results/smoke-v1',
  submitForHiddenEval: true,
  apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
  apiKey: process.env.ADMIN_SECRET,
  agentId: 'agent_mn9unhblb3o5swf2',
};

console.log('── Smoke Test ──');
console.log(`Task:     ${smokeTask.id}`);
console.log(`Protocol: ${smokeProtocol.id} (${smokeProtocol.backbones[0].model})`);
console.log(`Seed:     ${smokeConfig.seeds[0]}`);
console.log(`Budget:   T=${smokeConfig.budget.tokenCap} W=${smokeConfig.budget.wallClockSeconds}s C=${smokeConfig.budget.evalCpuSeconds}s K=${smokeConfig.budget.evalCallCap}`);
console.log(`Output:   ${smokeConfig.outputDir}/`);
console.log(`Submit:   ${smokeConfig.submitForHiddenEval ? 'YES' : 'no'}`);
console.log('');

runExperiment(smokeConfig)
  .then(results => {
    const run = results.runs[0];
    console.log('\n── Smoke Diagnostics ──');
    if (!run) {
      console.error('FAIL: no run produced (cell was dropped).');
      console.error('Likely causes: LLM call failed, JSON parse failed, or dev verifier rejected the solution.');
      process.exit(2);
    }
    console.log(`devScore:      ${run.bestArtifact.devScore}`);
    console.log(`tokensUsed:    ${run.budgetTrace.tokenUsage.total} / ${smokeConfig.budget.tokenCap}`);
    console.log(`wallClock:     ${Math.round(run.budgetTrace.wallClockMs / 1000)}s / ${smokeConfig.budget.wallClockSeconds}s`);
    console.log(`evalCalls:     ${run.budgetTrace.evalCalls.length} / ${smokeConfig.budget.evalCallCap}`);
    console.log(`submissionId:  ${run.submissionId || '(not submitted)'}`);

    const partition = run.bestArtifact.solutionData['partition'];
    console.log(`partition len: ${Array.isArray(partition) ? partition.length : 'not an array'}`);

    if (run.bestArtifact.devScore <= 0 || run.budgetTrace.tokenUsage.total === 0) {
      console.error('\nFAIL: smoke returned a zero-score or zero-token cell. Orchestrator is broken.');
      process.exit(3);
    }

    console.log('\nPASS: orchestrator round-trip works.');
    process.exit(0);
  })
  .catch(err => {
    console.error('\nFATAL:', err);
    process.exit(1);
  });
