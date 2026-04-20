#!/usr/bin/env tsx
/**
 * resubmit-for-hidden.ts — Re-submit saved solutions to get hidden scores
 *
 * Some per-cell JSONs have hiddenScore=null because their Firestore submissions
 * were lost. This script reads every such file, re-submits the saved
 * bestArtifact.solutionData via evalFinal, and writes back the hidden score.
 *
 * NO LLM calls. NO experiment reruns. Just platform eval + Firestore write.
 *
 * Usage:
 *   ADMIN_SECRET=xxx npx tsx src/runner/resubmit-for-hidden.ts --resultsDir results/full-v2
 *   ADMIN_SECRET=xxx npx tsx src/runner/resubmit-for-hidden.ts --resultsDir results/full-v2 --dryRun
 */

import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { RunResult } from '../types.js';

// ── CLI ──────────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const resultsDir = args.find(a => a.startsWith('--resultsDir='))?.split('=')[1]
  ?? (args.includes('--resultsDir') ? args[args.indexOf('--resultsDir') + 1] : 'results/full-v2');
const dryRun = args.includes('--dryRun');

const PLATFORM_URL = process.env.PLATFORM_URL ?? 'https://agent4science.org';
const ADMIN_SECRET = process.env.ADMIN_SECRET;
if (!ADMIN_SECRET) {
  console.error('ADMIN_SECRET env var is required');
  process.exit(1);
}

// ── Agent mapping (must match run-full.ts exactly) ───────────────────────────

const PROTOCOL_AGENT_MAP: Record<string, string> = {
  // Claude Sonnet 4 Solo
  'single-shot':    'agent_mnrvr457qvb4rt48',
  'best-of-n':      'agent_mnrvr457qvb4rt48',
  'self-refine':    'agent_mnrvr457qvb4rt48',
  'vgs':            'agent_mnrvr457qvb4rt48',
  'homo-chain':     'agent_mnrvr457qvb4rt48',
  'magicore':       'agent_mnrvr457qvb4rt48',
  'debate':         'agent_mnrvr457qvb4rt48',
  'moa':            'agent_mnrvr457qvb4rt48',
  'hpe':            'agent_mnrvr457qvb4rt48',
  'moa-same-model': 'agent_mnrvr457qvb4rt48',
  'moa-nosynth':    'agent_mnrvr4an9n0hmlfl',
  // Frontier Mix
  'cross-chain':    'agent_mnrvr4an9n0hmlfl',
  'magicore-mixed': 'agent_mnrvr4an9n0hmlfl',
  'debate-mixed':   'agent_mnrvr4an9n0hmlfl',
  'hpe-mixed':      'agent_mnrvr4an9n0hmlfl',
  // OSS Mix
  'magicore-mixed-oss': 'agent_mnrvr4f5jbtql279',
  'debate-mixed-oss':   'agent_mnrvr4f5jbtql279',
  'hpe-mixed-oss':      'agent_mnrvr4f5jbtql279',
  // DeepSeek V3 Solo
  'magicore-deepseek':    'agent_mnrvr4jh94vf09rh',
  'debate-deepseek':      'agent_mnrvr4jh94vf09rh',
  'hpe-deepseek':         'agent_mnrvr4jh94vf09rh',
  'self-refine-deepseek': 'agent_mnrvr4jh94vf09rh',
  'best-of-n-deepseek':   'agent_mnrvr4jh94vf09rh',
  'vgs-deepseek':         'agent_mnrvr4jh94vf09rh',
  // GPT-4o Solo
  'single-shot-gpt4o':  'agent_mnrvsunlm96jmexa',
  'self-refine-gpt4o':  'agent_mnrvsunlm96jmexa',
  'best-of-n-gpt4o':    'agent_mnrvsunlm96jmexa',
  'vgs-gpt4o':          'agent_mnrvsunlm96jmexa',
  'homo-chain-gpt4o':   'agent_mnrvsunlm96jmexa',
  'magicore-gpt4o':     'agent_mnrvsunlm96jmexa',
  'debate-gpt4o':       'agent_mnrvsunlm96jmexa',
  'hpe-gpt4o':          'agent_mnrvsunlm96jmexa',
  // Gemini Solo
  'single-shot-gemini': 'agent_mnrvsuvgu5ubpcz9',
  'self-refine-gemini': 'agent_mnrvsuvgu5ubpcz9',
  'best-of-n-gemini':   'agent_mnrvsuvgu5ubpcz9',
  'vgs-gemini':         'agent_mnrvsuvgu5ubpcz9',
  'homo-chain-gemini':  'agent_mnrvsuvgu5ubpcz9',
  'magicore-gemini':    'agent_mnrvsuvgu5ubpcz9',
  'debate-gemini':      'agent_mnrvsuvgu5ubpcz9',
  'hpe-gemini':         'agent_mnrvsuvgu5ubpcz9',
};

const DEFAULT_AGENT_ID = 'agent_mnrvr457qvb4rt48';

function resolveAgentId(protocolId: string): string {
  if (PROTOCOL_AGENT_MAP[protocolId]) return PROTOCOL_AGENT_MAP[protocolId];
  // Longest prefix match
  const match = Object.keys(PROTOCOL_AGENT_MAP)
    .filter(k => protocolId.startsWith(k))
    .sort((a, b) => b.length - a.length)[0];
  return match ? PROTOCOL_AGENT_MAP[match] : DEFAULT_AGENT_ID;
}

// ── evalFinal call ───────────────────────────────────────────────────────────

interface EvalFinalResult {
  success: boolean;
  devScore: number;
  submissionId: string;
  error?: string;
}

async function evalFinal(
  challengeId: string,
  solutionData: Record<string, unknown>,
  meta: Record<string, unknown>,
): Promise<EvalFinalResult> {
  const res = await fetch(`${PLATFORM_URL}/api/v1/benchmark/evaluate`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${ADMIN_SECRET}`,
    },
    body: JSON.stringify({
      challengeId,
      solutionData,
      final: true,
      ...meta,
    }),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`evalFinal HTTP ${res.status}: ${text}`);
  }

  return res.json() as Promise<EvalFinalResult>;
}

// ── Fetch hidden score for a submission ──────────────────────────────────────

async function getHiddenScore(submissionId: string): Promise<number | null> {
  const res = await fetch(`${PLATFORM_URL}/api/v1/benchmark/submission/${encodeURIComponent(submissionId)}`, {
    headers: { 'Authorization': `Bearer ${ADMIN_SECRET}` },
  });
  if (!res.ok) return null;
  const data = await res.json() as { success: boolean; hiddenScore?: number | null };
  return data.success ? (data.hiddenScore ?? null) : null;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  console.log(`Results dir : ${resultsDir}`);
  console.log(`Platform URL: ${PLATFORM_URL}`);
  console.log(`Dry run     : ${dryRun}\n`);

  // Load all per-cell files needing resubmission
  const files = readdirSync(resultsDir).filter(
    f => f.endsWith('.json') && f !== 'results.json' && f !== 'salvage-index.json' && !f.startsWith('meg')
  );

  const toResubmit: Array<{ file: string; run: RunResult }> = [];
  let alreadyHas = 0;
  let noSolution = 0;

  for (const f of files) {
    try {
      const raw = readFileSync(join(resultsDir, f), 'utf-8');
      const run = JSON.parse(raw) as RunResult;
      if (!run.taskId || !run.protocolId) continue;

      if (typeof run.hiddenScore === 'number') {
        alreadyHas++;
        continue;
      }

      const solutionData = run.bestArtifact?.solutionData;
      if (!solutionData || typeof solutionData !== 'object') {
        noSolution++;
        continue;
      }

      toResubmit.push({ file: join(resultsDir, f), run });
    } catch {
      // skip
    }
  }

  console.log(`Already have hidden score: ${alreadyHas}`);
  console.log(`No solutionData (skip):    ${noSolution}`);
  console.log(`Need resubmission:         ${toResubmit.length}\n`);

  if (toResubmit.length === 0) {
    console.log('Nothing to resubmit.');
    return;
  }

  let submitted = 0;
  let failed = 0;
  let skipped = 0;

  for (let i = 0; i < toResubmit.length; i++) {
    const { file, run } = toResubmit[i];
    const challengeId = BENCHMARK_CHALLENGE_IDS[run.taskId];

    if (!challengeId) {
      console.warn(`  SKIP ${run.taskId}/${run.protocolId}/s${run.seed}: no challengeId mapping`);
      skipped++;
      continue;
    }

    const agentId = resolveAgentId(run.protocolId);
    const label = `${run.taskId} / ${run.protocolId} / s${run.seed}`;

    if (dryRun) {
      console.log(`  [dry] ${label} → challengeId=${challengeId} agent=${agentId}`);
      submitted++;
      continue;
    }

    try {
      // Re-submit the saved solution
      const result = await evalFinal(challengeId, run.bestArtifact!.solutionData, {
        protocolId: run.protocolId,
        runIndex: run.runIndex,
        seed: run.seed,
        title: `${run.protocolId} seed=${run.seed} (resubmit)`,
        approach: `Protocol: ${run.protocolId}, resubmitted for hidden eval`,
        agentId,
      });

      if (!result.success) {
        console.error(`  FAIL ${label}: ${result.error}`);
        failed++;
        continue;
      }

      // Fetch the hidden score
      const hiddenScore = await getHiddenScore(result.submissionId);

      // Update local file
      run.submissionId = result.submissionId;
      run.hiddenScore = hiddenScore;
      writeFileSync(file, JSON.stringify(run, null, 2));

      submitted++;
      if ((submitted % 10) === 0 || submitted === 1) {
        console.log(`  [${submitted}/${toResubmit.length}] ${label} → dev=${result.devScore} hidden=${hiddenScore} sub=${result.submissionId}`);
      }
    } catch (err) {
      console.error(`  ERROR ${label}: ${err}`);
      failed++;
    }
  }

  console.log(`\nDone: submitted=${submitted} failed=${failed} skipped=${skipped}`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
