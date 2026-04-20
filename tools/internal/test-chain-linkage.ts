#!/usr/bin/env tsx
/**
 * test-chain-linkage.ts — Integration test for the chain-submission path.
 *
 * Runs one cell of each chain protocol on MaxCut, captures every submissionId
 * produced during that cell, and then walks the improvesUpon chain via
 * platform.getSubmission() to verify Firestore actually stored the links.
 *
 * This is the test that bridges the gap we found: diagnose-chain.ts proved the
 * runner produces N artifacts locally, but didn't prove that submitArtifacts →
 * Firestore writes N rows with correct improvesUpon linkage.
 *
 * Chain-length expectations (at defaults):
 *   self-refine : 4 artifacts  (1 initial + 3 refine)
 *   homo-chain  : 4 artifacts  (chainLength=4)
 *   cross-chain : 3 artifacts  (3 backbones)
 *   magicore    : 3 artifacts  (1 solver + 2 refine)
 *
 * LLM/budget failures can truncate the chain short. The test accepts N >= 2
 * as passing (at least one link present), but prints the actual N so we can
 * see if anything looks short.
 *
 * Usage:
 *   npx tsx src/runner/test-chain-linkage.ts
 */

import 'dotenv/config';

import { runExperiment } from './experiment-runner.js';
import {
  primaryBackbone,
  crossChainBackbones,
} from '../config/models.js';
import { PlatformClient } from '../lib/platform-client.js';
import type { BenchmarkSubmissionRecord } from '../lib/platform-client.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type {
  ExperimentConfig,
  BenchmarkTask,
  ProtocolConfig,
  ProtocolId,
} from '../types.js';

// ── Pre-flight ───────────────────────────────────────────────────────────────

const missing: string[] = [];
if (!process.env.OPENROUTER_API_KEY) missing.push('OPENROUTER_API_KEY');
if (!process.env.ADMIN_SECRET) missing.push('ADMIN_SECRET');
if (missing.length > 0) {
  console.error(`FATAL: missing env vars: ${missing.join(', ')}`);
  process.exit(1);
}

// ── Capture-on-submit PlatformClient ────────────────────────────────────────
//
// The experiment runner doesn't expose the full list of submission IDs that
// submitArtifacts produced — it only returns the best one. To verify the
// full chain we instrument evalFinal to record every submissionId it returns,
// so the test harness can later walk the improvesUpon chain via getSubmission.

const apiUrl = process.env.A4S_API_URL || 'https://agent4science.org';
const adminSecret = process.env.ADMIN_SECRET!;

class CapturingPlatformClient extends PlatformClient {
  public capturedIds: string[] = [];

  async evalFinal(challengeId: string, solutionData: Record<string, unknown>, meta: Parameters<PlatformClient['evalFinal']>[2]) {
    const result = await super.evalFinal(challengeId, solutionData, meta);
    this.capturedIds.push(result.submissionId);
    return result;
  }

  reset() {
    this.capturedIds = [];
  }
}

// The capturing client is passed to runExperiment via its platformOverride
// parameter (added specifically to support instrumentation without duplicating
// budget/runner plumbing).

// ── Setup task (MaxCut) ──────────────────────────────────────────────────────

function findChallenge(id: string) {
  const ch = ALL_BENCHMARK_CHALLENGES.find(c => c.id === id);
  if (!ch) {
    console.error(`FATAL: challenge ${id} not found`);
    process.exit(1);
  }
  return ch;
}

const maxcutCh = findChallenge('bench-maxcut-g200');
const maxcutTask: BenchmarkTask = {
  id: maxcutCh.id,
  title: maxcutCh.title,
  track: 'scientific',
  prompt: maxcutCh.description,
  devVerifier: maxcutCh.verifier,
  solutionSchema: { format: maxcutCh.solutionSchema },
  scoringDirection: maxcutCh.scoringDirection,
  challengeId: BENCHMARK_CHALLENGE_IDS[maxcutCh.id],
  parameters: {},
};

// ── Protocols to exercise ────────────────────────────────────────────────────

interface ChainTestCase {
  protocolId: ProtocolId;
  name: string;
  protocolConfig: ProtocolConfig;
  /** Expected minimum chain length (passes if >= this). */
  expectedMin: number;
  /** Expected maximum (at defaults), for sanity printing only. */
  expectedAtDefault: number;
  /** Expected to be a true chain (improvesUpon linkage required)? */
  isChain: boolean;
}

const testCases: ChainTestCase[] = [
  {
    protocolId: 'self-refine',
    name: 'Self-Refine (3 rounds)',
    protocolConfig: {
      id: 'self-refine',
      name: 'Self-Refine (3 rounds)',
      backbones: [primaryBackbone()],
      params: { rounds: 3 },
    },
    expectedMin: 2,
    expectedAtDefault: 4,
    isChain: true,
  },
  {
    protocolId: 'homo-chain',
    name: 'Homo-Chain (length=4)',
    protocolConfig: {
      id: 'homo-chain',
      name: 'Homo-Chain (length=4)',
      backbones: [primaryBackbone()],
      params: { chainLength: 4 },
    },
    expectedMin: 2,
    expectedAtDefault: 4,
    isChain: true,
  },
  {
    protocolId: 'cross-chain',
    name: 'Cross-Chain (Claude→GPT-4o→Gemini)',
    protocolConfig: {
      id: 'cross-chain',
      name: 'Cross-Chain (Claude→GPT-4o→Gemini)',
      backbones: crossChainBackbones(),
      params: {},
    },
    expectedMin: 2,
    expectedAtDefault: 3,
    isChain: true,
  },
  {
    protocolId: 'magicore',
    name: 'MAgICoRe (2 rounds)',
    protocolConfig: {
      id: 'magicore',
      name: 'MAgICoRe (2 rounds)',
      backbones: [primaryBackbone()],
      params: { rounds: 2 },
    },
    expectedMin: 2,
    expectedAtDefault: 3,
    isChain: true,
  },
];

// ── Verify chain: walk improvesUpon backwards via getSubmission ─────────────

interface ChainReport {
  protocolId: ProtocolId;
  capturedCount: number;
  expectedAtDefault: number;
  records: BenchmarkSubmissionRecord[];
  linkageValid: boolean;
  linkageError?: string;
}

async function verifyChain(
  client: PlatformClient,
  capturedIds: string[],
): Promise<{ records: BenchmarkSubmissionRecord[]; linkageValid: boolean; linkageError?: string }> {
  const records: BenchmarkSubmissionRecord[] = [];
  for (const id of capturedIds) {
    try {
      records.push(await client.getSubmission(id));
    } catch (err) {
      return {
        records,
        linkageValid: false,
        linkageError: `getSubmission(${id}) failed: ${err instanceof Error ? err.message : String(err)}`,
      };
    }
  }

  // Linkage rules:
  //   1. First submitted row has improvesUpon=null (root)
  //   2. Every subsequent row's improvesUpon === the previous row's submissionId
  for (let i = 0; i < records.length; i++) {
    const rec = records[i];
    if (i === 0) {
      if (rec.improvesUpon !== null) {
        return {
          records,
          linkageValid: false,
          linkageError: `root row ${rec.submissionId} has improvesUpon=${rec.improvesUpon}, expected null`,
        };
      }
    } else {
      const expectedParent = records[i - 1].submissionId;
      if (rec.improvesUpon !== expectedParent) {
        return {
          records,
          linkageValid: false,
          linkageError: `row ${i} (${rec.submissionId}) improvesUpon=${rec.improvesUpon}, expected ${expectedParent}`,
        };
      }
    }
  }

  return { records, linkageValid: true };
}

// ── Per-cell runner using a CapturingPlatformClient injected in place ───────
//
// We run each protocol in its own ExperimentConfig so the budget is fresh and
// the capturing client only sees that cell's submissions. The client is
// threaded into runExperiment via its platformOverride parameter.

async function runOneCell(tc: ChainTestCase): Promise<ChainReport> {
  const client = new CapturingPlatformClient(apiUrl, adminSecret);

  const config: ExperimentConfig = {
    id: `chain-test-${tc.protocolId}`,
    name: `Chain test: ${tc.name}`,
    tasks: [maxcutTask],
    protocols: [tc.protocolConfig],
    budget: {
      tokenCap: 20_000,
      wallClockSeconds: 300,
      evalCpuSeconds: 60,
      evalCallCap: 10,
    },
    seeds: [1],
    outputDir: `results/chain-test/${tc.protocolId}`,
    submitForHiddenEval: true,
    apiUrl,
    apiKey: adminSecret,
    agentId: 'agent_mn9unhblb3o5swf2',
  };

  console.log(`\n── Running ${tc.name} ──`);
  await runExperiment(config, client);

  console.log(`  captured ${client.capturedIds.length} submission(s): ${client.capturedIds.join(', ')}`);

  if (client.capturedIds.length === 0) {
    return {
      protocolId: tc.protocolId,
      capturedCount: 0,
      expectedAtDefault: tc.expectedAtDefault,
      records: [],
      linkageValid: false,
      linkageError: 'no submissions captured (runner produced no valid artifact, or chain path not taken)',
    };
  }

  const { records, linkageValid, linkageError } = await verifyChain(client, client.capturedIds);

  return {
    protocolId: tc.protocolId,
    capturedCount: client.capturedIds.length,
    expectedAtDefault: tc.expectedAtDefault,
    records,
    linkageValid,
    linkageError,
  };
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log('══════════════════════════════════════════════════════════════');
  console.log('Chain linkage integration test');
  console.log('  task: bench-maxcut-g200');
  console.log('  protocols: self-refine, homo-chain, cross-chain, magicore');
  console.log('══════════════════════════════════════════════════════════════');

  const reports: ChainReport[] = [];
  for (const tc of testCases) {
    reports.push(await runOneCell(tc));
  }

  console.log('\n══════════════════════════════════════════════════════════════');
  console.log('Chain linkage summary');
  console.log('══════════════════════════════════════════════════════════════\n');

  let allPass = true;
  for (const r of reports) {
    const pass = r.capturedCount >= 2 && r.linkageValid;
    const tag = pass ? '✓ PASS' : '✗ FAIL';
    if (!pass) allPass = false;

    console.log(`[${tag}] ${r.protocolId}`);
    console.log(`   captured=${r.capturedCount}  expectedAtDefault=${r.expectedAtDefault}`);
    if (r.records.length > 0) {
      console.log('   chain:');
      for (let i = 0; i < r.records.length; i++) {
        const rec = r.records[i];
        const parent = rec.improvesUpon ? `↩ ${rec.improvesUpon.slice(0, 22)}` : '(root)';
        console.log(`     ${i}. ${rec.submissionId}  dev=${rec.devScore}  ${parent}`);
      }
    }
    if (r.linkageError) {
      console.log(`   ERROR: ${r.linkageError}`);
    }
    console.log('');
  }

  console.log(`── ${reports.filter(r => r.capturedCount >= 2 && r.linkageValid).length} / ${reports.length} chain tests passed ──`);
  process.exit(allPass ? 0 : 2);
}

main().catch(err => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
