#!/usr/bin/env npx tsx
/**
 * Salvage Run 1 — reconstruct full-v1 results from the platform.
 *
 * The v1 full run had two problems:
 *   1. A second run (Run 2) crashed early with OpenRouter 403 errors and
 *      overwrote the first run's per-cell JSONs with empty placeholders.
 *   2. MEG was computed on devScore (visible to runners) instead of
 *      hiddenScore (V_c^final), which would have understated protocol
 *      differences even if the JSONs were intact.
 *
 * Recovery strategy:
 *   Run 1's run.log still contains the full list of submission IDs created
 *   on agent4science.org (157 of them). Every record in Firestore has
 *   verifierScore (devScore), hiddenVerifierScore (hiddenScore), and
 *   benchmarkMeta ({protocolId, runIndex, seed}), plus challengeId.
 *
 *   We walk the log, pull every sub_xxx we see, call
 *   GET /api/v1/benchmark/submission/[id] for each, then group by
 *   (taskId, protocolId, seed). For chain protocols where multiple
 *   submissions were made per cell, we keep the best-scoring artifact.
 *
 * Output: results/full-v1-salvaged/*.json matching the RunResult schema,
 * plus a summary printed to stdout.
 *
 * Two retrieval modes:
 *   --via=firestore (default) — read scibook's agent4science Firestore database
 *                               directly via service account. Matches what
 *                               seed-to-platform.ts does. Credentials come
 *                               from scibook/.env.local (FIREBASE_SERVICE_ACCOUNT_BASE64).
 *                               Recommended: no dependency on production
 *                               ADMIN_SECRET rotation state.
 *   --via=api                 — call GET /api/v1/benchmark/submission/[id] on
 *                               agent4science.org. Requires ADMIN_SECRET in .env
 *                               to match the production env var.
 *
 * Usage:
 *   npx tsx src/tasks/salvage-run1.ts                       # default: firestore
 *   npx tsx src/tasks/salvage-run1.ts --via=api             # use HTTP endpoint
 *   npx tsx src/tasks/salvage-run1.ts --dry-run             # parse log only
 *   npx tsx src/tasks/salvage-run1.ts --limit=20            # smoke test
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { resolve, join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { config as loadEnv } from 'dotenv';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load .env from this repo, then overlay scibook's .env.local (for Firebase).
// scibook's .env.local takes precedence so FIREBASE_SERVICE_ACCOUNT_BASE64 is
// available even if neurips-experiments/.env exists without it.
loadEnv({ path: resolve(__dirname, '../../.env') });
loadEnv({ path: resolve(__dirname, '../../../scibook/.env.local'), override: false });

import { PlatformClient, type BenchmarkSubmissionRecord } from '../lib/platform-client.js';
import { ALL_BENCHMARK_CHALLENGES } from './benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from './challenge-ids.js';
import { NORMALIZATION_ANCHORS, normalizeQ } from '../config/normalization-anchors.js';
import type { RunResult, ProtocolId, SolutionArtifact, BudgetTrace } from '../types.js';

// ── Config ──

const DEFAULT_LOG_PATH = resolve(__dirname, '../../results/full-v1/run.log');
const DEFAULT_OUT_DIR = resolve(__dirname, '../../results/full-v1-salvaged');

// Run 1 occupies lines 1..RUN1_END in run.log; everything at or after
// RUN2_START is a retry that failed with OpenRouter 403s and produced no
// submissions. We only parse Run 1.
const RUN1_END_LINE_EXCLUSIVE = 1731;

// ── CLI ──

const args = process.argv.slice(2);
function argVal(flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : undefined;
}
const logPath = argVal('--log') ?? DEFAULT_LOG_PATH;
const outDir = argVal('--out') ?? DEFAULT_OUT_DIR;
const dryRun = args.includes('--dry-run');
const limitArg = argVal('--limit') ?? (args.find(a => a.startsWith('--limit='))?.split('=')[1]);
const limit = limitArg ? parseInt(limitArg, 10) : undefined;
const viaArg = argVal('--via') ?? (args.find(a => a.startsWith('--via='))?.split('=')[1]) ?? 'firestore';
if (viaArg !== 'firestore' && viaArg !== 'api') {
  console.error(`Unknown --via value: ${viaArg}. Use 'firestore' or 'api'.`);
  process.exit(1);
}
const via = viaArg as 'firestore' | 'api';

// ── Log parser ──

/**
 * Extract unique submission IDs from Run 1 lines of run.log.
 *
 * The log has output-buffering quirks where Progress lines and submission
 * lines sometimes merge onto the same physical line, so we ignore line
 * structure entirely and just pull every sub_xxx token out of Run 1.
 */
function extractSubmissionIdsFromLog(text: string): string[] {
  const lines = text.split('\n');
  const run1 = lines.slice(0, RUN1_END_LINE_EXCLUSIVE - 1).join('\n');
  const ids = new Set<string>();
  const re = /sub_[a-z0-9]+/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(run1)) !== null) {
    ids.add(m[0]);
  }
  return [...ids];
}

// ── Challenge ↔ Task mapping ──

const CHALLENGE_TO_TASK: Record<string, { taskId: string; direction: 'maximize' | 'minimize' }> = {};
for (const [taskId, challengeId] of Object.entries(BENCHMARK_CHALLENGE_IDS)) {
  const def = ALL_BENCHMARK_CHALLENGES.find(c => c.id === taskId);
  if (!def) continue;
  CHALLENGE_TO_TASK[challengeId] = { taskId, direction: def.scoringDirection };
}

// ── Main ──

interface FetchResult {
  id: string;
  ok: boolean;
  record?: BenchmarkSubmissionRecord;
  error?: string;
}

async function fetchAllViaApi(platform: PlatformClient, ids: string[]): Promise<FetchResult[]> {
  const results: FetchResult[] = [];
  let i = 0;
  for (const id of ids) {
    i++;
    try {
      const rec = await platform.getSubmission(id);
      results.push({ id, ok: true, record: rec });
      if (i % 20 === 0) console.log(`  fetched ${i}/${ids.length}`);
    } catch (err) {
      results.push({ id, ok: false, error: String(err) });
    }
  }
  return results;
}

/**
 * Read submissions directly from the scibook Firestore admin database.
 * Mirrors seed-to-platform.ts's approach: load the service account JSON from
 * FIREBASE_SERVICE_ACCOUNT_BASE64 and use the named 'agent4science' database.
 *
 * This bypasses the platform HTTP endpoint entirely — useful when the
 * production ADMIN_SECRET has rotated or the GET endpoint isn't deployed yet.
 */
async function fetchAllViaFirestore(ids: string[]): Promise<FetchResult[]> {
  const { initializeApp, cert, getApps } = await import('firebase-admin/app');
  const { getFirestore } = await import('firebase-admin/firestore');

  const b64 = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64;
  if (!b64) {
    throw new Error(
      'FIREBASE_SERVICE_ACCOUNT_BASE64 not set. Check scibook/.env.local (salvage-run1 auto-loads it).'
    );
  }
  const sa = JSON.parse(Buffer.from(b64, 'base64').toString('utf-8')) as { projectId: string };
  if (getApps().length === 0) {
    initializeApp({ credential: cert(sa as object as Parameters<typeof cert>[0]), projectId: sa.projectId });
  }
  const db = getFirestore(getApps()[0], 'agent4science');

  const results: FetchResult[] = [];
  const BATCH = 30;
  for (let i = 0; i < ids.length; i += BATCH) {
    const batch = ids.slice(i, i + BATCH);
    const snaps = await Promise.all(
      batch.map(id => db.collection('submissions').doc(id).get())
    );
    for (let j = 0; j < batch.length; j++) {
      const id = batch[j];
      const snap = snaps[j];
      if (!snap.exists) {
        results.push({ id, ok: false, error: 'Submission not found in Firestore' });
        continue;
      }
      const d = snap.data() as Record<string, unknown>;
      const rec: BenchmarkSubmissionRecord = {
        submissionId: id,
        challengeId: (d.challengeId as string | null | undefined) ?? null,
        devScore: (d.verifierScore as number | null | undefined) ?? null,
        hiddenScore: (d.hiddenVerifierScore as number | null | undefined) ?? null,
        benchmarkMeta: (d.benchmarkMeta as BenchmarkSubmissionRecord['benchmarkMeta']) ?? null,
        improvesUpon: (d.improvesUpon as string | null | undefined) ?? null,
        createdAt: (d.createdAt as string | null | undefined) ?? null,
      };
      results.push({ id, ok: true, record: rec });
    }
    console.log(`  fetched ${Math.min(i + BATCH, ids.length)}/${ids.length}`);
  }
  return results;
}

type CellKey = string; // `${taskId}::${protocolId}::${seed}`

interface CellGroup {
  taskId: string;
  protocolId: ProtocolId;
  seed: number;
  runIndex: number;
  direction: 'maximize' | 'minimize';
  records: BenchmarkSubmissionRecord[];
}

function groupByCell(records: BenchmarkSubmissionRecord[]): Map<CellKey, CellGroup> {
  const groups = new Map<CellKey, CellGroup>();
  for (const rec of records) {
    if (!rec.challengeId) continue;
    const taskInfo = CHALLENGE_TO_TASK[rec.challengeId];
    if (!taskInfo) {
      console.warn(`  skip ${rec.submissionId}: unknown challengeId ${rec.challengeId}`);
      continue;
    }
    const meta = rec.benchmarkMeta;
    if (!meta || !meta.protocolId || meta.seed == null) {
      console.warn(`  skip ${rec.submissionId}: missing benchmarkMeta (protocolId/seed)`);
      continue;
    }

    const protocolId = meta.protocolId as ProtocolId;
    const seed = meta.seed as number;
    const runIndex = (meta.runIndex as number | null | undefined) ?? 0;
    const key: CellKey = `${taskInfo.taskId}::${protocolId}::${seed}`;

    let group = groups.get(key);
    if (!group) {
      group = {
        taskId: taskInfo.taskId,
        protocolId,
        seed,
        runIndex,
        direction: taskInfo.direction,
        records: [],
      };
      groups.set(key, group);
    }
    group.records.push(rec);
  }
  return groups;
}

function pickBestRecord(group: CellGroup): BenchmarkSubmissionRecord {
  const valid = group.records.filter(r => typeof r.devScore === 'number');
  if (valid.length === 0) return group.records[0];
  return valid.reduce((best, cur) => {
    const bd = best.devScore as number;
    const cd = cur.devScore as number;
    if (group.direction === 'maximize') return cd > bd ? cur : best;
    return cd < bd ? cur : best;
  });
}

function emptyTrace(evalCpuMs: number): BudgetTrace {
  return {
    tokenUsage: { prompt: 0, completion: 0, total: 0 },
    wallClockMs: 0,
    evalCpuMs,
    evalCalls: [],
    exhausted: { tokens: false, wallClock: false, evalCpu: false, evalCalls: false },
  };
}

function buildRunResult(group: CellGroup): RunResult {
  const best = pickBestRecord(group);
  const devScore = (best.devScore as number) ?? 0;

  const bestArtifact: SolutionArtifact = {
    // Actual solutionData is not stored in the log; leave empty. This
    // is fine for MEG / Q / correlation analysis, which only needs
    // (devScore, hiddenScore) per cell. Downstream scripts that need
    // the solution bytes should refetch directly from Firestore.
    solutionData: {},
    devScore,
    rawOutput: '',
  };

  const evalCpuMs =
    (best.benchmarkMeta && (best.benchmarkMeta.evalCpuMs as number | null | undefined)) ?? 0;

  return {
    taskId: group.taskId,
    protocolId: group.protocolId,
    runIndex: group.runIndex,
    seed: group.seed,
    bestArtifact,
    allArtifacts: [bestArtifact],
    budgetTrace: emptyTrace(evalCpuMs),
    submissionId: best.submissionId,
    hiddenScore: (best.hiddenScore as number | null | undefined) ?? null,
    timestamp: best.createdAt ?? new Date().toISOString(),
  };
}

// ── Analysis helpers ──

function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n === 0 || ys.length !== n) return NaN;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
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

function printSummary(runs: RunResult[]): void {
  console.log('\n=== Salvage Summary ===');
  console.log(`Total salvaged cells: ${runs.length} / 360 expected`);
  const withHidden = runs.filter(r => typeof r.hiddenScore === 'number');
  console.log(`Cells with hidden score: ${withHidden.length}`);

  const byTask = new Map<string, RunResult[]>();
  for (const r of runs) {
    if (!byTask.has(r.taskId)) byTask.set(r.taskId, []);
    byTask.get(r.taskId)!.push(r);
  }

  console.log('\n── Per-task coverage & dev↔hidden correlation ──');
  console.log('task'.padEnd(28) + 'n'.padEnd(6) + 'devMean'.padEnd(12) + 'hidMean'.padEnd(12) + 'corr'.padEnd(10) + 'meanQ');
  for (const taskId of Object.keys(BENCHMARK_CHALLENGE_IDS)) {
    const taskRuns = (byTask.get(taskId) ?? []).filter(r => typeof r.hiddenScore === 'number');
    const anchor = NORMALIZATION_ANCHORS[taskId];
    const devs = taskRuns.map(r => r.bestArtifact.devScore);
    const hids = taskRuns.map(r => r.hiddenScore as number);
    const corr = pearson(devs, hids);
    const devMean = devs.length ? devs.reduce((a, b) => a + b, 0) / devs.length : NaN;
    const hidMean = hids.length ? hids.reduce((a, b) => a + b, 0) / hids.length : NaN;
    const qMean = anchor && hids.length
      ? hids.map(s => normalizeQ(s, anchor)).reduce((a, b) => a + b, 0) / hids.length
      : NaN;
    console.log(
      taskId.padEnd(28) +
      String(taskRuns.length).padEnd(6) +
      (isFinite(devMean) ? devMean.toFixed(3) : '—').padEnd(12) +
      (isFinite(hidMean) ? hidMean.toFixed(3) : '—').padEnd(12) +
      (isFinite(corr) ? corr.toFixed(3) : '—').padEnd(10) +
      (isFinite(qMean) ? qMean.toFixed(3) : '—')
    );
  }

  const byProtocol = new Map<ProtocolId, RunResult[]>();
  for (const r of withHidden) {
    if (!byProtocol.has(r.protocolId)) byProtocol.set(r.protocolId, []);
    byProtocol.get(r.protocolId)!.push(r);
  }
  console.log('\n── Per-protocol coverage & mean Q (averaged across tasks) ──');
  console.log('protocol'.padEnd(16) + 'n'.padEnd(6) + 'meanQ');
  for (const [pid, prs] of byProtocol) {
    // Mean Q across cells (unweighted; for quick orientation only)
    const qs: number[] = [];
    for (const r of prs) {
      const anchor = NORMALIZATION_ANCHORS[r.taskId];
      if (!anchor || r.hiddenScore == null) continue;
      qs.push(normalizeQ(r.hiddenScore, anchor));
    }
    const qMean = qs.length ? qs.reduce((a, b) => a + b, 0) / qs.length : NaN;
    console.log(pid.padEnd(16) + String(prs.length).padEnd(6) + (isFinite(qMean) ? qMean.toFixed(3) : '—'));
  }
}

async function main() {
  console.log('Agent4Science-Bench — Run 1 Salvage');
  console.log(`Log:       ${logPath}`);
  console.log(`Output:    ${outDir}`);
  console.log(`Via:       ${via}`);
  console.log(`Mode:      ${dryRun ? 'DRY RUN' : 'LIVE'}`);
  if (limit) console.log(`Limit:     ${limit} submissions`);
  console.log('---');

  if (!existsSync(logPath)) {
    console.error(`run.log not found at ${logPath}`);
    process.exit(1);
  }

  const text = readFileSync(logPath, 'utf-8');
  const ids = extractSubmissionIdsFromLog(text);
  console.log(`Extracted ${ids.length} unique submission IDs from Run 1 (lines 1..${RUN1_END_LINE_EXCLUSIVE - 1})`);

  const idsToFetch = limit ? ids.slice(0, limit) : ids;

  if (dryRun) {
    console.log('Sample IDs:', idsToFetch.slice(0, 5));
    return;
  }

  console.log(`\nFetching ${idsToFetch.length} submissions via ${via}...`);
  let fetched: FetchResult[];
  if (via === 'firestore') {
    fetched = await fetchAllViaFirestore(idsToFetch);
  } else {
    const platform = new PlatformClient();
    fetched = await fetchAllViaApi(platform, idsToFetch);
  }
  const ok = fetched.filter(r => r.ok && r.record) as (FetchResult & { record: BenchmarkSubmissionRecord })[];
  const fail = fetched.filter(r => !r.ok);
  console.log(`\nFetch results: ${ok.length} ok, ${fail.length} failed`);
  if (fail.length > 0) {
    console.log('First 5 failures:');
    fail.slice(0, 5).forEach(f => console.log(`  ${f.id}: ${f.error}`));
  }

  const records = ok.map(r => r.record);
  const withHidden = records.filter(r => typeof r.hiddenScore === 'number').length;
  console.log(`Records with hiddenScore populated: ${withHidden}/${records.length}`);

  const groups = groupByCell(records);
  console.log(`Grouped into ${groups.size} cells`);

  mkdirSync(outDir, { recursive: true });

  const runs: RunResult[] = [];
  for (const group of groups.values()) {
    const result = buildRunResult(group);
    runs.push(result);
    const filename = `${result.taskId}_${result.protocolId}_s${result.seed}.json`;
    writeFileSync(join(outDir, filename), JSON.stringify(result, null, 2));
  }

  // Write a flat index for quick inspection
  writeFileSync(
    join(outDir, 'salvage-index.json'),
    JSON.stringify({
      salvagedAt: new Date().toISOString(),
      source: { log: logPath, run1EndLine: RUN1_END_LINE_EXCLUSIVE - 1 },
      submissions: {
        logExtracted: ids.length,
        fetched: idsToFetch.length,
        fetchOk: ok.length,
        fetchFailed: fail.length,
        withHiddenScore: withHidden,
      },
      cells: runs.length,
      failures: fail.map(f => ({ id: f.id, error: f.error })),
    }, null, 2)
  );

  printSummary(runs);

  console.log(`\nSalvaged JSONs written to: ${outDir}/`);
  console.log('Next: run MEG aggregation on the salvaged dir.');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
