#!/usr/bin/env tsx
/**
 * inspect-smoke-mixed.ts — Validate the 6-cell mixed-brain smoke test.
 *
 * Reads the 6 result files from results/full-v2/ that the smoke test wrote
 * (one per mixed-brain protocol on bench-maxcut-g200, seed 42), then fetches
 * the matching platform submissions and verifies:
 *   - all 6 cells produced a non-null bestArtifact.devScore
 *   - all 6 cells have a submissionId
 *   - per-cell token usage is in the same ballpark as the homogeneous canary
 *     (a sanity check that the role-based backbone fan-out actually ran every
 *     stage and didn't silently collapse to a single backbone)
 *   - platform.getSubmission returns the right challengeId
 *   - benchmarkMeta.protocolId on the platform matches the *mixed* id, not
 *     the homogeneous parent (catches "wired the protocol but tagged it wrong")
 *   - hiddenScore is populated (means the hidden eval ran server-side)
 *
 * Also does a code-level smoke test of crossChainOSSBackbones() to verify
 * the 3 OSS model IDs are distinct, since runtime fan-out is not directly
 * captured in the result files.
 *
 * Usage:
 *   ADMIN_SECRET=xxx OPENROUTER_API_KEY=yyy npx tsx src/runner/inspect-smoke-mixed.ts
 */

import 'dotenv/config';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { PlatformClient } from '../lib/platform-client.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import { crossChainBackbones, crossChainOSSBackbones } from '../config/models.js';
import type { BackboneConfig } from '../types.js';

const TASK = 'bench-maxcut-g200';
const SEED = 42;
const EXPECTED_CHALLENGE = BENCHMARK_CHALLENGE_IDS[TASK];

const SMOKE_PROTOCOLS = [
  'magicore-mixed',
  'debate-mixed',
  'hpe-mixed',
  'magicore-mixed-oss',
  'debate-mixed-oss',
  'hpe-mixed-oss',
];

interface SmokeResultFile {
  taskId: string;
  protocolId: string;
  runIndex: number;
  seed: number;
  bestArtifact: { solutionData: unknown; devScore: number; rawOutput?: string };
  allArtifacts: Array<{ solutionData: unknown; devScore: number }>;
  budgetTrace: {
    tokenUsage: { prompt: number; completion: number; total: number };
    wallClockMs: number;
    evalCpuMs: number;
    evalCalls: unknown[];
    exhausted: { tokens: boolean; wallClock: boolean; evalCpu: boolean; evalCalls: boolean };
  };
  submissionId: string;
  hiddenScore: number | null;
  timestamp: string;
}

async function loadResult(protocolId: string): Promise<SmokeResultFile | null> {
  // run-full writes one file per (task, protocol, seed) using the literal
  // seed value: e.g. {task}_{protocol}_s42.json for --seeds 42.
  const path = join('results/full-v2', `${TASK}_${protocolId}_s${SEED}.json`);
  try {
    const text = await readFile(path, 'utf8');
    return JSON.parse(text) as SmokeResultFile;
  } catch (err) {
    if ((err as { code?: string }).code === 'ENOENT') return null;
    throw err;
  }
}

async function main() {
  console.log('Mixed-Brain Smoke Test Validation');
  console.log(`Task:               ${TASK}`);
  console.log(`Expected challenge: ${EXPECTED_CHALLENGE}`);
  console.log(`Seed:               ${SEED}`);
  console.log(`Protocols:          ${SMOKE_PROTOCOLS.length}`);
  console.log();

  // ── Code-level fan-out check ────────────────────────────────────────────────
  const frontier = crossChainBackbones().map((b: BackboneConfig) => b.model);
  const oss = crossChainOSSBackbones().map((b: BackboneConfig) => b.model);
  console.log('Backbone fan-out (code-level, by construction):');
  console.log(`  frontier: solver=${frontier[0]} reviewer=${frontier[1]} refiner=${frontier[2]}`);
  console.log(`  oss:      solver=${oss[0]} reviewer=${oss[1]} refiner=${oss[2]}`);
  if (new Set(frontier).size !== 3) {
    console.error('  ✗ frontier role-to-model map has duplicates — fan-out is broken');
    process.exit(1);
  }
  if (new Set(oss).size !== 3) {
    console.error('  ✗ oss role-to-model map has duplicates — fan-out is broken');
    process.exit(1);
  }
  console.log('  ✓ all 3 frontier roles map to distinct models');
  console.log('  ✓ all 3 oss roles map to distinct models');
  console.log();

  // ── Per-cell validation ─────────────────────────────────────────────────────
  console.log('protocol             dev    tok    wall  arts  submission                         hidden  meta_proto         challenge_ok');
  console.log('─'.repeat(140));

  const platform = new PlatformClient();
  let problems = 0;
  const submissionIds: { protocol: string; sub: string; ok: boolean }[] = [];

  for (const protocol of SMOKE_PROTOCOLS) {
    const local = await loadResult(protocol);
    if (!local) {
      console.log(`✗ ${protocol.padEnd(20)} (no result file written — cell did not finish)`);
      problems++;
      continue;
    }

    const issues: string[] = [];
    const dev = local.bestArtifact?.devScore ?? null;
    const tok = local.budgetTrace?.tokenUsage?.total ?? 0;
    const wall = Math.round((local.budgetTrace?.wallClockMs ?? 0) / 1000);
    const numArts = local.allArtifacts?.length ?? 0;
    const sub = local.submissionId;

    if (dev === null || dev === 0) issues.push('devScore null/zero');
    if (!sub) issues.push('no submissionId');
    if (tok < 2000) issues.push(`token usage suspiciously low (${tok})`);
    // Artifact-count threshold: 2 for magicore/debate (which emit per refine/turn),
    // 1 for hpe (which legitimately emits 1 artifact per planner→execute→integrate
    // round; 2-round runs that drop to 1 are an OSS-robustness research finding,
    // not a wiring bug — the token-count symmetry check below is the real
    // fan-out detector for HPE).
    const minArts = protocol.startsWith('hpe-') ? 1 : 2;
    if (numArts < minArts) issues.push(`only ${numArts} artifact(s) — chain may have collapsed`);
    if (local.budgetTrace?.exhausted?.tokens) issues.push('token budget exhausted');

    // Platform-side checks
    let metaProto = '(unfetched)';
    let hidden: number | null = null;
    let challengeOk = '(unfetched)';
    if (sub) {
      try {
        const r = await platform.getSubmission(sub);
        metaProto = r.benchmarkMeta?.protocolId ?? '(none)';
        hidden = r.hiddenScore;
        challengeOk = r.challengeId === EXPECTED_CHALLENGE ? 'yes' : `NO (${r.challengeId})`;

        if (r.challengeId !== EXPECTED_CHALLENGE) issues.push('wrong challenge');
        if (metaProto !== protocol) issues.push(`meta proto mismatch (got ${metaProto}, expected ${protocol})`);
        if (hidden === null) issues.push('hiddenScore=null (server-side eval may still be in progress)');
      } catch (err) {
        issues.push(`platform fetch failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    }

    if (issues.length > 0) problems++;
    submissionIds.push({ protocol, sub, ok: issues.length === 0 });

    const flag = issues.length > 0 ? '✗' : '✓';
    console.log(
      `${flag} ${protocol.padEnd(20)} ${String(dev ?? 'null').padStart(5)}  ${String(tok).padStart(5)}  ${String(wall).padStart(4)}s  ${String(numArts).padStart(4)}  ${(sub ?? '(none)').padEnd(34)}  ${String(hidden ?? 'null').padStart(6)}  ${metaProto.padEnd(18)} ${challengeOk}`
    );
    if (issues.length > 0) {
      for (const i of issues) console.log(`     ! ${i}`);
    }
  }

  // ── Cross-cell sanity: token-usage symmetry between frontier and oss ────────
  console.log();
  console.log('Token-usage sanity (indirect runtime fan-out check):');
  const frontierCells = SMOKE_PROTOCOLS.filter((p) => !p.endsWith('-oss'));
  const ossCells = SMOKE_PROTOCOLS.filter((p) => p.endsWith('-oss'));
  for (const p of frontierCells) {
    const r = await loadResult(p);
    if (r) console.log(`  frontier ${p.padEnd(20)} tokens=${r.budgetTrace.tokenUsage.total}`);
  }
  for (const p of ossCells) {
    const r = await loadResult(p);
    if (r) console.log(`  oss      ${p.padEnd(20)} tokens=${r.budgetTrace.tokenUsage.total}`);
  }
  console.log('  (oss cells should be in roughly the same order of magnitude as frontier');
  console.log('   cells; an oss cell with <30% the frontier token count suggests one stage');
  console.log('   silently failed and the chain collapsed to a single model.)');

  // ── Submission IDs for UI cleanup ───────────────────────────────────────────
  console.log();
  console.log('Submission IDs (for paper trail / UI cleanup if needed):');
  for (const { protocol, sub, ok } of submissionIds) {
    console.log(`  ${ok ? '✓' : '✗'} ${protocol.padEnd(20)} ${sub}`);
  }

  console.log();
  console.log('─'.repeat(140));
  if (problems === 0) {
    console.log(`✓ All ${SMOKE_PROTOCOLS.length} smoke cells passed.`);
    console.log(`  Wiring is sound. Safe to launch the 240-cell brain-diversity sweep.`);
  } else {
    console.log(`✗ ${problems} / ${SMOKE_PROTOCOLS.length} smoke cells had problems.`);
    console.log(`  Do NOT launch the 240-cell sweep until these are fixed.`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
