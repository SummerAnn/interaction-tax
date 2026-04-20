#!/usr/bin/env tsx
/**
 * run-kcurve.ts — MaxCut proxy-overfit K-curve experiment
 *
 * Runs VGS on MaxCut with varying evaluator-call caps K ∈ {1, 3, 5, 10, 15, 20, 25}.
 * If dev score rises with K but hidden score does not (or drops), that is direct
 * evidence of proxy overfit on the 80%-edge public evaluator.
 *
 * Usage:
 *   ADMIN_SECRET=xxx OPENROUTER_API_KEY=yyy npx tsx src/runner/run-kcurve.ts
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-kcurve.ts --dry-run
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-kcurve.ts --k 1,5,25   # subset of K values
 *
 * Output:  results/kcurve-maxcut/  (one JSON per cell)
 * Cost estimate: ~$2-3 (7 K values × 5 seeds × ~8K tokens/cell via OpenRouter)
 */

import 'dotenv/config';
import { runExperiment } from './experiment-runner.js';
import { primaryBackbone } from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

// ── CLI args ─────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const kIdx = args.indexOf('--k');
const kFilter = kIdx >= 0
  ? args[kIdx + 1].split(',').map(s => Number(s.trim())).filter(n => Number.isFinite(n) && n > 0)
  : undefined;

// ── K values to sweep ────────────────────────────────────────────────────────

const K_VALUES = kFilter ?? [1, 3, 5, 10, 15, 20, 25];
const SEEDS = [1, 2, 3, 4, 5];

// ── MaxCut task definition ───────────────────────────────────────────────────

const maxcutChallenge = ALL_BENCHMARK_CHALLENGES.find(ch => ch.id === 'bench-maxcut-g200');
if (!maxcutChallenge) {
  console.error('ERROR: bench-maxcut-g200 not found in benchmark challenges');
  process.exit(1);
}

function normalizeMaxCut(data: Record<string, unknown>): Record<string, unknown> {
  const raw = data['partition'];
  if (!Array.isArray(raw)) return data;
  let arr = (raw as unknown[]).map((v) => (Number(v) >= 1 ? 1 : 0));
  if (arr.length > 200) arr = arr.slice(0, 200);
  while (arr.length < 200) arr.push(0);
  return { ...data, partition: arr };
}

const maxcutTask: BenchmarkTask = {
  id: maxcutChallenge.id,
  title: maxcutChallenge.title,
  track: 'scientific',
  prompt: maxcutChallenge.description,
  devVerifier: maxcutChallenge.verifier,
  solutionSchema: { format: maxcutChallenge.solutionSchema },
  scoringDirection: maxcutChallenge.scoringDirection,
  challengeId: BENCHMARK_CHALLENGE_IDS[maxcutChallenge.id],
  parameters: {},
  normalizeOutput: normalizeMaxCut,
};

// ── Run one experiment per K value ───────────────────────────────────────────

console.log('MaxCut K-Curve Experiment');
console.log(`K values:  ${K_VALUES.join(', ')}`);
console.log(`Seeds:     ${SEEDS.join(', ')}`);
console.log(`Cells:     ${K_VALUES.length * SEEDS.length}`);
console.log(`Output:    results/kcurve-maxcut/`);
console.log(`Submit:    YES — agent4science.org (for hidden scores)`);

if (dryRun) {
  console.log('\nDry run — config valid. Exiting without running.');
  process.exit(0);
}

if (!process.env.ADMIN_SECRET) {
  console.error('\nERROR: ADMIN_SECRET env var required');
  process.exit(1);
}

async function main() {
  for (const K of K_VALUES) {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`Running K=${K} (${SEEDS.length} seeds)`);
    console.log('═'.repeat(60));

    // VGS config adapted for this K:
    // For very low K (1, 3), reduce population/generations so VGS doesn't just
    // exhaust its eval budget on initialization and never mutate.
    const pop = K <= 3 ? Math.min(K, 2) : 4;
    const gen = K <= 3 ? 1 : 5;

    const vgsProtocol: ProtocolConfig = {
      id: 'vgs',
      name: `VGS (K=${K}, pop=${pop}, gen=${gen})`,
      backbones: [primaryBackbone()],
      params: { populationSize: pop, generations: gen, eliteCount: Math.min(2, pop), convergenceWindow: 2 },
    };

    const config: ExperimentConfig = {
      id: `kcurve-maxcut-k${K}`,
      name: `K-Curve MaxCut K=${K}`,
      tasks: [maxcutTask],
      protocols: [vgsProtocol],
      budget: {
        tokenCap: 200_000,
        wallClockSeconds: 600,
        evalCpuSeconds: 30,
        evalCallCap: K,
      },
      seeds: SEEDS,
      outputDir: `results/kcurve-maxcut/k${K}`,
      submitForHiddenEval: true,
      apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
      apiKey: process.env.ADMIN_SECRET,
      agentId: 'agent_mnrvr457qvb4rt48',
    };

    await runExperiment(config);
    console.log(`K=${K} complete.`);
  }

  // ── Summary ──────────────────────────────────────────────────────────────────
  console.log('\n\nK-Curve experiment complete.');
  console.log('Results in: results/kcurve-maxcut/');
  console.log('Run analyze_kcurve.py to plot dev vs hidden score as a function of K.');
}

main().then(() => process.exit(0)).catch((err) => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
