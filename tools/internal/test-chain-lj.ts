#!/usr/bin/env tsx
/**
 * test-chain-lj.ts — Spot-check: chain submission path on a task with
 * NEGATIVE valid scores (LJ-n41).
 *
 * Before the submitArtifacts filter fix, every chain protocol on LJ-n41 would
 * silently collapse to a single-row submission because `devScore > 0` dropped
 * all legitimate negative cluster energies. This script verifies that after
 * the fix, a chain protocol (homo-chain) actually writes N linked rows on
 * LJ-n41.
 *
 * Usage:
 *   npx tsx src/runner/test-chain-lj.ts
 */

import 'dotenv/config';

import { runExperiment } from './experiment-runner.js';
import { primaryBackbone } from '../config/models.js';
import { PlatformClient } from '../lib/platform-client.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

if (!process.env.OPENROUTER_API_KEY || !process.env.ADMIN_SECRET) {
  console.error('FATAL: OPENROUTER_API_KEY and ADMIN_SECRET required');
  process.exit(1);
}

const apiUrl = process.env.A4S_API_URL || 'https://agent4science.org';
const adminSecret = process.env.ADMIN_SECRET!;

class CapturingPlatformClient extends PlatformClient {
  public capturedIds: string[] = [];

  async evalFinal(challengeId: string, solutionData: Record<string, unknown>, meta: Parameters<PlatformClient['evalFinal']>[2]) {
    const result = await super.evalFinal(challengeId, solutionData, meta);
    this.capturedIds.push(result.submissionId);
    return result;
  }
}

const ch = ALL_BENCHMARK_CHALLENGES.find(c => c.id === 'bench-lj-n41')!;
const ljTask: BenchmarkTask = {
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

const homoProto: ProtocolConfig = {
  id: 'homo-chain',
  name: 'Homo-Chain (length=4)',
  backbones: [primaryBackbone()],
  params: { chainLength: 4 },
};

async function main() {
  const client = new CapturingPlatformClient(apiUrl, adminSecret);
  const config: ExperimentConfig = {
    id: 'chain-lj-spot',
    name: 'Chain spot-check: homo-chain × lj-n41',
    tasks: [ljTask],
    protocols: [homoProto],
    budget: {
      tokenCap: 20_000,
      wallClockSeconds: 300,
      evalCpuSeconds: 60,
      evalCallCap: 10,
    },
    seeds: [1],
    outputDir: 'results/chain-test/lj-spot',
    submitForHiddenEval: true,
    apiUrl,
    apiKey: adminSecret,
    agentId: 'agent_mn9unhblb3o5swf2',
  };

  console.log('══════════════════════════════════════════════════════════════');
  console.log('Chain spot-check on negative-score task');
  console.log('  task: bench-lj-n41  (valid scores are negative)');
  console.log('  protocol: homo-chain (chainLength=4)');
  console.log('══════════════════════════════════════════════════════════════');

  await runExperiment(config, client);

  console.log(`\ncaptured ${client.capturedIds.length} submission(s):`);
  for (const id of client.capturedIds) {
    const rec = await client.getSubmission(id);
    const parent = rec.improvesUpon ? `↩ ${rec.improvesUpon.slice(0, 22)}` : '(root)';
    console.log(`  ${rec.submissionId}  dev=${rec.devScore}  ${parent}`);
  }

  // Verify linkage
  const records = await Promise.all(client.capturedIds.map(id => client.getSubmission(id)));
  let linkageValid = true;
  let linkageError: string | undefined;
  for (let i = 0; i < records.length; i++) {
    if (i === 0 && records[i].improvesUpon !== null) {
      linkageValid = false;
      linkageError = `root row has improvesUpon=${records[i].improvesUpon}, expected null`;
      break;
    }
    if (i > 0 && records[i].improvesUpon !== records[i - 1].submissionId) {
      linkageValid = false;
      linkageError = `row ${i} improvesUpon=${records[i].improvesUpon}, expected ${records[i - 1].submissionId}`;
      break;
    }
  }

  const pass = client.capturedIds.length >= 2 && linkageValid;
  console.log(`\n${pass ? '✓ PASS' : '✗ FAIL'}  captured=${client.capturedIds.length}  linkageValid=${linkageValid}`);
  if (linkageError) console.log(`  ERROR: ${linkageError}`);
  process.exit(pass ? 0 : 2);
}

main().catch(err => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
