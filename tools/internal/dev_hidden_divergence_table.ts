#!/usr/bin/env npx tsx
/**
 * Dev/Hidden Divergence Table — per-task dev↔hidden divergence analysis.
 *
 * The benchmark's theoretical framing relies on the dev verifier being a
 * noisy-but-aligned proxy for V_c^final (the hidden verifier). When that
 * assumption holds, protocols that optimize against the dev verifier will
 * make honest progress on the hidden one. When it doesn't — when dev and
 * hidden diverge under pressure — you get proxy overfitting: protocols game the
 * dev proxy and lose on the hidden evaluator.
 *
 * This script quantifies the dev↔hidden relationship per task using the
 * salvaged Run 1 data. For each task we report:
 *
 *   n              number of salvaged cells
 *   Pearson        linear correlation between dev and hidden
 *   Spearman       rank correlation between dev and hidden
 *   dev_best       best dev score observed (by task scoringDirection)
 *   hidden_best    best hidden score observed (by task scoringDirection)
 *   best_cell_gap  hidden score of the DEV-best cell minus the
 *                  hidden score of the HIDDEN-best cell (negative means
 *                  dev-best is also hidden-best, bigger means proxy overfit)
 *   rank_flip      fraction of ordered cell pairs (i,j) where dev and
 *                  hidden disagree on which is better. 0 = perfect
 *                  alignment, 0.5 = pure noise.
 *
 * Usage:
 *   npx tsx src/tasks/dev-hidden-divergence-table.ts --dir results/full-v1-salvaged
 */

import { readFileSync, readdirSync, writeFileSync, existsSync } from 'fs';
import { resolve, join, dirname } from 'path';
import { fileURLToPath } from 'url';

import type { RunResult } from '../types.js';
import { ALL_BENCHMARK_CHALLENGES } from './benchmark-challenges.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const args = process.argv.slice(2);
function argVal(flag: string): string | undefined {
  const inline = args.find(a => a.startsWith(`${flag}=`));
  if (inline) return inline.slice(flag.length + 1);
  const i = args.indexOf(flag);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : undefined;
}
const dir = resolve(argVal('--dir') ?? resolve(__dirname, '../../results/full-v1-salvaged'));
const outPath = resolve(argVal('--out') ?? join(dir, 'dev_hidden_divergence_table.tsv'));

// ── Load RunResult cells ──

function loadRuns(d: string): RunResult[] {
  if (!existsSync(d)) {
    console.error(`Directory not found: ${d}`);
    process.exit(1);
  }
  const files = readdirSync(d).filter(
    f => f.endsWith('.json') && f !== 'salvage-index.json' && f !== 'results.json'
  );
  const runs: RunResult[] = [];
  for (const f of files) {
    try {
      const r = JSON.parse(readFileSync(join(d, f), 'utf-8')) as RunResult;
      if (r && r.taskId && r.protocolId) runs.push(r);
    } catch {
      // skip
    }
  }
  return runs;
}

// ── Statistics ──

function mean(xs: number[]): number {
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN;
}

function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 2 || ys.length !== n) return NaN;
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? NaN : num / denom;
}

function ranks(xs: number[]): number[] {
  // Fractional ranks, handles ties by averaging.
  const sorted = xs
    .map((v, i) => ({ v, i }))
    .sort((a, b) => a.v - b.v);
  const r = new Array<number>(xs.length);
  let i = 0;
  while (i < sorted.length) {
    let j = i;
    while (j + 1 < sorted.length && sorted[j + 1].v === sorted[i].v) j++;
    const avg = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) r[sorted[k].i] = avg;
    i = j + 1;
  }
  return r;
}

function spearman(xs: number[], ys: number[]): number {
  if (xs.length !== ys.length || xs.length < 2) return NaN;
  return pearson(ranks(xs), ranks(ys));
}

/**
 * Fraction of ordered pairs (i != j) where the dev ordering contradicts the
 * hidden ordering, given the task's scoring direction. A value of 0 means
 * dev and hidden agree on every pairwise comparison; 0.5 is pure noise.
 * Ties on either axis are ignored in the denominator.
 */
function rankFlipRate(
  devs: number[],
  hids: number[],
  direction: 'maximize' | 'minimize',
): number {
  const n = devs.length;
  if (n < 2) return NaN;
  let considered = 0;
  let flipped = 0;
  const sign = direction === 'maximize' ? 1 : -1;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const d = sign * (devs[i] - devs[j]);
      const h = sign * (hids[i] - hids[j]);
      if (d === 0 || h === 0) continue;
      considered++;
      if ((d > 0) !== (h > 0)) flipped++;
    }
  }
  return considered === 0 ? NaN : flipped / considered;
}

// ── Main ──

function main() {
  console.log('Agent4Science-Bench — Dev/Hidden Divergence Table');
  console.log(`Input: ${dir}`);
  console.log(`Output: ${outPath}`);
  console.log('---');

  const runs = loadRuns(dir).filter(r => typeof r.hiddenScore === 'number');
  console.log(`Loaded ${runs.length} cells with hiddenScore`);

  const byTask = new Map<string, RunResult[]>();
  for (const r of runs) {
    if (!byTask.has(r.taskId)) byTask.set(r.taskId, []);
    byTask.get(r.taskId)!.push(r);
  }

  const rows: Array<{
    taskId: string;
    n: number;
    pearson: number;
    spearman: number;
    devBest: number;
    hidBest: number;
    bestCellGap: number;
    rankFlip: number;
    direction: 'maximize' | 'minimize';
  }> = [];

  for (const [taskId, cells] of byTask) {
    const def = ALL_BENCHMARK_CHALLENGES.find(c => c.id === taskId);
    if (!def) continue;
    const direction = def.scoringDirection;

    const devs = cells.map(c => c.bestArtifact.devScore);
    const hids = cells.map(c => c.hiddenScore as number);

    // Best by direction
    const pickBest = (xs: number[], dir: 'maximize' | 'minimize') =>
      dir === 'maximize' ? Math.max(...xs) : Math.min(...xs);
    const devBest = pickBest(devs, direction);
    const hidBest = pickBest(hids, direction);

    // Dev-best cell's hidden score vs best hidden score
    const devBestIdx = devs.findIndex(v => v === devBest);
    const devBestHidden = hids[devBestIdx];
    // For maximize: gap = hidBest - devBestHidden (>= 0, larger = worse alignment)
    // For minimize: gap = devBestHidden - hidBest (>= 0, larger = worse alignment)
    const bestCellGap =
      direction === 'maximize' ? hidBest - devBestHidden : devBestHidden - hidBest;

    rows.push({
      taskId,
      n: cells.length,
      pearson: pearson(devs, hids),
      spearman: spearman(devs, hids),
      devBest,
      hidBest,
      bestCellGap,
      rankFlip: rankFlipRate(devs, hids, direction),
      direction,
    });
  }

  rows.sort((a, b) => a.taskId.localeCompare(b.taskId));

  // ── Write TSV ──
  const lines = [
    'task\tdirection\tn\tpearson\tspearman\tdev_best\thidden_best\tbest_cell_gap\trank_flip_rate',
  ];
  for (const r of rows) {
    lines.push(
      [
        r.taskId,
        r.direction,
        r.n,
        isFinite(r.pearson) ? r.pearson.toFixed(3) : '—',
        isFinite(r.spearman) ? r.spearman.toFixed(3) : '—',
        isFinite(r.devBest) ? r.devBest.toFixed(3) : '—',
        isFinite(r.hidBest) ? r.hidBest.toFixed(3) : '—',
        isFinite(r.bestCellGap) ? r.bestCellGap.toFixed(3) : '—',
        isFinite(r.rankFlip) ? r.rankFlip.toFixed(3) : '—',
      ].join('\t')
    );
  }
  writeFileSync(outPath, lines.join('\n') + '\n');
  console.log(`\nWrote ${outPath}`);

  // ── Print ──
  console.log('\n── Dev/Hidden Divergence Table ──');
  console.log(
    'task'.padEnd(28) +
      'n'.padEnd(5) +
      'pearson'.padEnd(10) +
      'spearman'.padEnd(10) +
      'gap'.padEnd(12) +
      'rank_flip'
  );
  for (const r of rows) {
    console.log(
      r.taskId.padEnd(28) +
        String(r.n).padEnd(5) +
        (isFinite(r.pearson) ? r.pearson.toFixed(3) : '—').padEnd(10) +
        (isFinite(r.spearman) ? r.spearman.toFixed(3) : '—').padEnd(10) +
        (isFinite(r.bestCellGap) ? r.bestCellGap.toFixed(3) : '—').padEnd(12) +
        (isFinite(r.rankFlip) ? r.rankFlip.toFixed(3) : '—')
    );
  }
}

main();
