#!/usr/bin/env tsx
/**
 * run-external-2x2.ts — External validation of the diversity–synthesis 2×2
 *
 * Runs the same 2×2 factorial (diversity × synthesis) from Finding 2 on
 * external CO-Bench-inspired tasks (TSP, Knapsack, Number Partition) to
 * test whether the coordination-diversity tradeoff replicates outside
 * the benchmark suite.
 *
 * Design (matches Table 2 in the paper):
 *   ┌─────────────┬───────────────┬──────────────────┐
 *   │             │ Same Model    │ Diverse Models   │
 *   ├─────────────┼───────────────┼──────────────────┤
 *   │ No Synthesis│ Best-of-3     │ MoA-NoSynth      │
 *   │ Synthesis   │ MoA-Same-Model│ MoA (diverse)    │
 *   └─────────────┴───────────────┴──────────────────┘
 *
 * All N=3 proposals per cell. 5 seeds per cell. Local Python verifiers.
 *
 * Usage:
 *   npx tsx src/runner/run-external-2x2.ts
 *   npx tsx src/runner/run-external-2x2.ts --task ext-tsp-25
 *   npx tsx src/runner/run-external-2x2.ts --dry-run
 */

import 'dotenv/config';
import { writeFileSync, readFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import type { BenchmarkTask, ProtocolConfig, SolutionArtifact, RunResult, BudgetVector } from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import { PROTOCOL_RUNNERS, getBestArtifact } from '../protocols/index.js';
import { primaryBackbone, moaBackbones, moaBackbonesSameModel } from '../config/models.js';
import { LocalEvaluator } from '../lib/local-evaluator.js';
import { EXTERNAL_TASKS, EXTERNAL_VERIFIERS } from '../tasks/external-tasks.js';

// ── CLI ──────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const taskIdx = args.indexOf('--task');
const taskFilter = taskIdx >= 0
  ? args[taskIdx + 1].split(',').map(s => s.trim()).filter(Boolean)
  : undefined;

// ── Config ───────────────────────────────────────────────────────────────────

const SEEDS = [42, 137, 256, 314, 500];

const BUDGET: BudgetVector = {
  tokenCap: 200_000,
  wallClockSeconds: 300,   // 5 min per cell (simpler tasks)
  evalCpuSeconds: 30,
  evalCallCap: 25,
};

const PROTOCOLS_2x2: ProtocolConfig[] = [
  {
    id: 'best-of-3',
    name: 'Best-of-3 (Claude×3, same model, no synthesis)',
    backbones: [primaryBackbone()],
    params: { n: 3 },
  },
  {
    id: 'moa-nosynth',
    name: 'MoA-NoSynth (diverse, no synthesis)',
    backbones: moaBackbones(),
    params: {},
  },
  {
    id: 'moa-same-model',
    name: 'MoA-Same-Model (Claude×3, same model, synthesis)',
    backbones: moaBackbonesSameModel(),
    params: {},
  },
  {
    id: 'moa',
    name: 'MoA (diverse, synthesis)',
    backbones: moaBackbones(),
    params: {},
  },
];

// ── Run ──────────────────────────────────────────────────────────────────────

const tasks = taskFilter
  ? EXTERNAL_TASKS.filter(t => taskFilter.includes(t.id))
  : EXTERNAL_TASKS;

if (tasks.length === 0) {
  console.error(`No tasks matched filter: ${taskFilter}`);
  process.exit(1);
}

const outputDir = 'results/external-2x2';
mkdirSync(outputDir, { recursive: true });

const totalCells = tasks.length * PROTOCOLS_2x2.length * SEEDS.length;

console.log('External Validation — Diversity × Synthesis 2×2');
console.log(`Tasks:     ${tasks.map(t => t.id).join(', ')}`);
console.log(`Protocols: ${PROTOCOLS_2x2.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${SEEDS.join(', ')}`);
console.log(`Cells:     ${totalCells}`);
console.log(`Output:    ${outputDir}/`);

if (dryRun) {
  console.log('\nDry run — config valid. Verifying local evaluators...');
  const evaluator = new LocalEvaluator(EXTERNAL_VERIFIERS);
  for (const task of tasks) {
    try {
      // Smoke-test each verifier with a dummy solution
      const dummy = makeDummy(task);
      const result = await evaluator.evalDev(task.challengeId!, dummy);
      console.log(`  ${task.id}: verifier OK (dummy score=${result.devScore.toFixed(2)})`);
    } catch (err) {
      console.error(`  ${task.id}: verifier FAILED: ${err}`);
    }
  }
  process.exit(0);
}

if (!process.env.OPENROUTER_API_KEY) {
  console.error('\nERROR: OPENROUTER_API_KEY env var required');
  process.exit(1);
}

// ── Main loop ────────────────────────────────────────────────────────────────

const evaluator = new LocalEvaluator(EXTERNAL_VERIFIERS);
const runs: RunResult[] = [];
let completed = 0;
let skipped = 0;

for (const task of tasks) {
  for (const protocol of PROTOCOLS_2x2) {
    for (const seed of SEEDS) {
      const filename = `${task.id}_${protocol.id}_s${seed}.json`;
      const filepath = join(outputDir, filename);

      // Resume support
      if (existsSync(filepath)) {
        try {
          const existing = JSON.parse(readFileSync(filepath, 'utf-8')) as RunResult;
          runs.push(existing);
          skipped++;
        } catch { /* re-run corrupt file */ }
        completed++;
        continue;
      }

      console.log(`\n  [${task.id}] × [${protocol.id}] seed=${seed} ...`);

      const budget = new BudgetEnforcer(BUDGET, evaluator);
      const runner = PROTOCOL_RUNNERS[protocol.id];

      let allArtifacts: SolutionArtifact[] = [];
      try {
        allArtifacts = await runner(task, protocol, budget);
      } catch (err) {
        console.error(`  ERROR: ${err}`);
      }

      const best = getBestArtifact(allArtifacts, task.scoringDirection);
      const trace = budget.getTrace();

      if (!best) {
        console.log(`  → SKIP (no valid artifact)`);
        completed++;
        continue;
      }

      const result: RunResult = {
        taskId: task.id,
        protocolId: protocol.id,
        runIndex: SEEDS.indexOf(seed),
        seed,
        bestArtifact: best,
        allArtifacts,
        budgetTrace: trace,
        hiddenScore: null,
        timestamp: new Date().toISOString(),
      };

      runs.push(result);
      writeFileSync(filepath, JSON.stringify(result, null, 2));

      console.log(`  → devScore=${best.devScore.toFixed(2)} tokens=${trace.tokenUsage.total} wall=${Math.round(trace.wallClockMs / 1000)}s`);

      completed++;
      console.log(`  Progress: ${completed}/${totalCells}`);
    }
  }
}

// ── Analysis ─────────────────────────────────────────────────────────────────

console.log('\n\n========================================');
console.log('  External 2×2 Results Summary');
console.log('========================================\n');

if (skipped > 0) console.log(`Resumed ${skipped} existing cells.\n`);

interface CellStats {
  mean: number;
  std: number;
  n: number;
  scores: number[];
}

const cellStats = new Map<string, CellStats>();

for (const task of tasks) {
  console.log(`── ${task.id} ──`);

  for (const protocol of PROTOCOLS_2x2) {
    const cellRuns = runs.filter(r => r.taskId === task.id && r.protocolId === protocol.id);
    const scores = cellRuns.map(r => r.bestArtifact.devScore);

    if (scores.length === 0) {
      console.log(`  ${protocol.id.padEnd(18)} — no data`);
      continue;
    }

    const m = scores.reduce((a, b) => a + b, 0) / scores.length;
    const variance = scores.reduce((a, s) => a + (s - m) ** 2, 0) / scores.length;
    const std = Math.sqrt(variance);

    const key = `${task.id}::${protocol.id}`;
    cellStats.set(key, { mean: m, std, n: scores.length, scores });

    console.log(`  ${protocol.id.padEnd(18)} mean=${m.toFixed(2).padStart(10)}  std=${std.toFixed(2).padStart(8)}  n=${scores.length}`);
  }
  console.log();
}

// ── 2×2 ANOVA-style decomposition ───────────────────────────────────────────

console.log('── Diversity × Synthesis Factorial ──\n');
console.log('                  Same-Model    Diverse     Diversity Δ');
console.log('                  ──────────    ───────     ───────────');

for (const task of tasks) {
  const bo3   = cellStats.get(`${task.id}::best-of-3`);
  const noSyn = cellStats.get(`${task.id}::moa-nosynth`);
  const same  = cellStats.get(`${task.id}::moa-same-model`);
  const moa   = cellStats.get(`${task.id}::moa`);

  if (!bo3 || !noSyn || !same || !moa) {
    console.log(`  ${task.id}: incomplete data\n`);
    continue;
  }

  const noSynthRow = `No-Synth  ${bo3.mean.toFixed(2).padStart(10)}  ${noSyn.mean.toFixed(2).padStart(10)}  ${(noSyn.mean - bo3.mean).toFixed(2).padStart(10)}`;
  const synthRow   = `Synth     ${same.mean.toFixed(2).padStart(10)}  ${moa.mean.toFixed(2).padStart(10)}  ${(moa.mean - same.mean).toFixed(2).padStart(10)}`;
  const synthDelta = `Synth Δ   ${(same.mean - bo3.mean).toFixed(2).padStart(10)}  ${(moa.mean - noSyn.mean).toFixed(2).padStart(10)}`;

  console.log(`  ${task.id}:`);
  console.log(`    ${noSynthRow}`);
  console.log(`    ${synthRow}`);
  console.log(`    ${synthDelta}`);

  // Main effects
  const diverseEffect = ((noSyn.mean - bo3.mean) + (moa.mean - same.mean)) / 2;
  const synthEffect = ((same.mean - bo3.mean) + (moa.mean - noSyn.mean)) / 2;
  console.log(`    Diversity main effect: ${diverseEffect >= 0 ? '+' : ''}${diverseEffect.toFixed(2)}`);
  console.log(`    Synthesis main effect: ${synthEffect >= 0 ? '+' : ''}${synthEffect.toFixed(2)}`);
  console.log();
}

// ── Aggregate across tasks ──────────────────────────────────────────────────

console.log('── Aggregate Main Effects (across all external tasks) ──\n');

let aggDiverseEffects: number[] = [];
let aggSynthEffects: number[] = [];

for (const task of tasks) {
  const bo3   = cellStats.get(`${task.id}::best-of-3`);
  const noSyn = cellStats.get(`${task.id}::moa-nosynth`);
  const same  = cellStats.get(`${task.id}::moa-same-model`);
  const moa   = cellStats.get(`${task.id}::moa`);

  if (!bo3 || !noSyn || !same || !moa) continue;

  aggDiverseEffects.push(((noSyn.mean - bo3.mean) + (moa.mean - same.mean)) / 2);
  aggSynthEffects.push(((same.mean - bo3.mean) + (moa.mean - noSyn.mean)) / 2);
}

if (aggDiverseEffects.length > 0) {
  const avgDiverse = aggDiverseEffects.reduce((a, b) => a + b, 0) / aggDiverseEffects.length;
  const avgSynth = aggSynthEffects.reduce((a, b) => a + b, 0) / aggSynthEffects.length;
  console.log(`  Mean diversity effect: ${avgDiverse >= 0 ? '+' : ''}${avgDiverse.toFixed(2)}`);
  console.log(`  Mean synthesis effect: ${avgSynth >= 0 ? '+' : ''}${avgSynth.toFixed(2)}`);
  console.log(`  Diversity / |Synthesis| ratio: ${Math.abs(avgDiverse / avgSynth).toFixed(1)}×`);
}

// Save summary
const summary = {
  tasks: tasks.map(t => t.id),
  protocols: PROTOCOLS_2x2.map(p => p.id),
  seeds: SEEDS,
  cellStats: Object.fromEntries(cellStats),
  timestamp: new Date().toISOString(),
};
writeFileSync(join(outputDir, 'summary.json'), JSON.stringify(summary, null, 2));
console.log(`\nResults saved to ${outputDir}/`);

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeDummy(task: BenchmarkTask): Record<string, unknown> {
  if (task.id === 'ext-tsp-25') {
    return { tour: Array.from({ length: 25 }, (_, i) => i) };
  }
  if (task.id === 'ext-maxsat-60') {
    return { assignment: Array(60).fill(0) };
  }
  if (task.id === 'ext-partition-40') {
    return { assignment: Array(40).fill(0) };
  }
  return {};
}
