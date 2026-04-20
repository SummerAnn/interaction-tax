#!/usr/bin/env tsx
/**
 * diagnose-chain.ts — Find out why Self-Refine / Homo-Chain / MAgICoRe
 * produce far fewer artifacts than their configured length.
 *
 * Runs each chain protocol ONCE against MaxCut with patched versions of
 * the inner loops that log every decision point:
 *   - LLM call null?
 *   - parseSolutionJSON null?
 *   - evalDev throw?
 *   - canCallLLM / canCallEval flipped mid-run?
 *
 * Does NOT submit to the platform — uses PlatformClient only for dev eval.
 * Reuses the real budget-enforced LLM stack so the diagnosis reflects real
 * Run 1 conditions.
 *
 * Usage:
 *   npx tsx src/runner/diagnose-chain.ts [--protocol self-refine|homo-chain|magicore]
 *
 * Expected output per chain step:
 *   step 1 (initial): llmOk=true parseOk=true evalOk=true devScore=2400
 *   step 2 (refine):  llmOk=true parseOk=true evalOk=true devScore=2450
 *   ...
 */

import 'dotenv/config';

import { BudgetEnforcer } from '../budget/budget-vector.js';
import { PlatformClient } from '../lib/platform-client.js';
import { primaryBackbone } from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import { SINGLE_SHOT, SELF_REFINE, HOMO_CHAIN, MAGICORE, fill } from '../config/prompts.js';
import type { BenchmarkTask } from '../types.js';

// ── Env check ─────────────────────────────────────────────────────────────────
if (!process.env.OPENROUTER_API_KEY || !process.env.ADMIN_SECRET) {
  console.error('Need OPENROUTER_API_KEY and ADMIN_SECRET in .env');
  process.exit(1);
}

// ── Target task ───────────────────────────────────────────────────────────────
const ch = ALL_BENCHMARK_CHALLENGES.find(c => c.id === 'bench-maxcut-g200')!;

function normalizeMaxCut(data: Record<string, unknown>): Record<string, unknown> {
  const raw = data['partition'];
  if (!Array.isArray(raw)) return data;
  let arr = (raw as unknown[]).map(v => (Number(v) >= 1 ? 1 : 0));
  if (arr.length > 200) arr = arr.slice(0, 200);
  while (arr.length < 200) arr.push(0);
  return { ...data, partition: arr };
}

const task: BenchmarkTask = {
  id: ch.id,
  title: ch.title,
  track: 'scientific',
  prompt: ch.description,
  devVerifier: ch.verifier,
  solutionSchema: { format: ch.solutionSchema },
  scoringDirection: ch.scoringDirection,
  challengeId: BENCHMARK_CHALLENGE_IDS[ch.id],
  parameters: {},
  normalizeOutput: normalizeMaxCut,
};

const backbone = primaryBackbone();
const platform = new PlatformClient(undefined, process.env.ADMIN_SECRET);
const budget = new BudgetEnforcer(
  { tokenCap: 200_000, wallClockSeconds: 600, evalCpuSeconds: 30, evalCallCap: 25 },
  platform,
);

// ── Shared helpers ────────────────────────────────────────────────────────────
function parseSolutionJSON(raw: string): Record<string, unknown> | null {
  try { return JSON.parse(raw.trim()); } catch { /* */ }
  const m = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (m) { try { return JSON.parse(m[1].trim()); } catch { /* */ } }
  const b = raw.match(/\{[\s\S]*\}/);
  if (b) { try { return JSON.parse(b[0]); } catch { /* */ } }
  return null;
}

async function evalStep(data: Record<string, unknown>): Promise<{ ok: boolean; score: number; error?: string }> {
  if (!budget.canCallEval) return { ok: false, score: 0, error: 'canCallEval=false' };
  try {
    const r = await budget.evalDev(task.challengeId!, data);
    if (!r) return { ok: false, score: 0, error: 'evalDev returned null' };
    return { ok: true, score: r.devScore };
  } catch (e) {
    return { ok: false, score: 0, error: String(e).slice(0, 200) };
  }
}

function logStep(label: string, info: {
  llmOk: boolean;
  parseOk: boolean;
  evalOk: boolean;
  devScore?: number;
  rawHead?: string;
  error?: string;
}): void {
  const parts = [
    `llmOk=${info.llmOk}`,
    `parseOk=${info.parseOk}`,
    `evalOk=${info.evalOk}`,
  ];
  if (info.devScore !== undefined) parts.push(`devScore=${info.devScore}`);
  if (info.error) parts.push(`err="${info.error}"`);
  console.log(`  ${label.padEnd(22)} ${parts.join(' ')}`);
  if (!info.parseOk && info.rawHead) {
    console.log(`                         rawHead="${info.rawHead.slice(0, 160).replace(/\n/g, '\\n')}"`);
  }
}

// ── Diagnostic runners ────────────────────────────────────────────────────────

async function diagSelfRefine(): Promise<void> {
  console.log('── Self-Refine (rounds=3) ──');
  const rounds = 3;
  const base = {
    PROBLEM: task.prompt,
    SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
    PARAMS: JSON.stringify(task.parameters ?? {}, null, 2),
  };

  // Step 1: initial
  const gen = await budget.llmCall(backbone, [
    { role: 'system', content: SINGLE_SHOT.system },
    { role: 'user',   content: fill(SINGLE_SHOT.user, base) },
  ], { temperature: 0.7, maxTokens: 4096 });
  if (!gen) { logStep('step 1 (initial)', { llmOk: false, parseOk: false, evalOk: false }); return; }

  let data = parseSolutionJSON(gen.content);
  if (!data) { logStep('step 1 (initial)', { llmOk: true, parseOk: false, evalOk: false, rawHead: gen.content }); return; }
  data = normalizeMaxCut(data);

  const ev = await evalStep(data);
  logStep('step 1 (initial)', { llmOk: true, parseOk: true, evalOk: ev.ok, devScore: ev.score, error: ev.error });
  if (!ev.ok) return;

  let currentScore = ev.score;
  let currentData = data;

  // Refinement rounds
  for (let r = 1; r <= rounds; r++) {
    if (!budget.canCallLLM) { console.log(`  step ${r + 1}: canCallLLM=false — STOP`); return; }
    if (!budget.canCallEval) { console.log(`  step ${r + 1}: canCallEval=false — STOP`); return; }

    const refineVars = {
      PROBLEM: task.prompt,
      SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
      SCORE: String(currentScore),
      SOLUTION: JSON.stringify(currentData, null, 2),
      DIRECTION: task.scoringDirection,
      BETTER: task.scoringDirection === 'maximize' ? 'higher' : 'lower',
    };

    const refined = await budget.llmCall(backbone, [
      { role: 'system', content: fill(SELF_REFINE.refine_system, refineVars) },
      { role: 'user',   content: fill(SELF_REFINE.refine_user, refineVars) },
    ], { temperature: 0.5, maxTokens: 4096 });

    if (!refined) {
      logStep(`step ${r + 1} (refine)`, { llmOk: false, parseOk: false, evalOk: false });
      return;
    }

    let rdata = parseSolutionJSON(refined.content);
    if (!rdata) {
      logStep(`step ${r + 1} (refine)`, { llmOk: true, parseOk: false, evalOk: false, rawHead: refined.content });
      return;
    }
    rdata = normalizeMaxCut(rdata);

    const rev = await evalStep(rdata);
    logStep(`step ${r + 1} (refine)`, { llmOk: true, parseOk: true, evalOk: rev.ok, devScore: rev.score, error: rev.error });
    if (!rev.ok) return;

    if (task.scoringDirection === 'maximize' ? rev.score > currentScore : rev.score < currentScore) {
      currentScore = rev.score;
      currentData = rdata;
    }
  }
}

async function diagHomoChain(): Promise<void> {
  console.log('── Homo-Chain (length=4) ──');
  const chainLength = 4;
  const base = {
    PROBLEM: task.prompt,
    SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
    PARAMS: JSON.stringify(task.parameters ?? {}, null, 2),
  };

  const gen = await budget.llmCall(backbone, [
    { role: 'system', content: HOMO_CHAIN.first_system },
    { role: 'user',   content: fill(HOMO_CHAIN.first_user, base) },
  ], { temperature: 0.7, maxTokens: 4096 });
  if (!gen) { logStep('step 1 (gen)', { llmOk: false, parseOk: false, evalOk: false }); return; }

  let data = parseSolutionJSON(gen.content);
  if (!data) { logStep('step 1 (gen)', { llmOk: true, parseOk: false, evalOk: false, rawHead: gen.content }); return; }
  data = normalizeMaxCut(data);

  let ev = await evalStep(data);
  logStep('step 1 (gen)', { llmOk: true, parseOk: true, evalOk: ev.ok, devScore: ev.score, error: ev.error });
  if (!ev.ok) return;

  let currentScore = ev.score;
  let currentData = data;

  for (let i = 1; i < chainLength; i++) {
    if (!budget.canCallLLM) { console.log(`  step ${i + 1}: canCallLLM=false — STOP`); return; }
    if (!budget.canCallEval) { console.log(`  step ${i + 1}: canCallEval=false — STOP`); return; }

    const refineVars = {
      ...base,
      AGENT_NUM: String(i + 1),
      DIRECTION: task.scoringDirection,
      SCORE: String(currentScore),
      SOLUTION: JSON.stringify(currentData, null, 2),
    };

    const refined = await budget.llmCall(backbone, [
      { role: 'system', content: fill(HOMO_CHAIN.refine_system, refineVars) },
      { role: 'user',   content: fill(HOMO_CHAIN.refine_user, refineVars) },
    ], { temperature: 0.5, maxTokens: 4096 });

    if (!refined) { logStep(`step ${i + 1} (refine)`, { llmOk: false, parseOk: false, evalOk: false }); return; }

    let rdata = parseSolutionJSON(refined.content);
    if (!rdata) {
      logStep(`step ${i + 1} (refine)`, { llmOk: true, parseOk: false, evalOk: false, rawHead: refined.content });
      // Note: original code does `continue`, not `return`. Mimic that:
      continue;
    }
    rdata = normalizeMaxCut(rdata);

    const rev = await evalStep(rdata);
    logStep(`step ${i + 1} (refine)`, { llmOk: true, parseOk: true, evalOk: rev.ok, devScore: rev.score, error: rev.error });

    if (rev.ok && (task.scoringDirection === 'maximize' ? rev.score > currentScore : rev.score < currentScore)) {
      currentScore = rev.score;
      currentData = rdata;
    }
  }
}

async function diagMAgICoRe(): Promise<void> {
  console.log('── MAgICoRe (rounds=2) ──');
  const rounds = 2;
  const base = {
    PROBLEM: task.prompt,
    SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
    PARAMS: JSON.stringify(task.parameters ?? {}, null, 2),
  };

  const solver = await budget.llmCall(backbone, [
    { role: 'system', content: MAGICORE.solver_system },
    { role: 'user',   content: fill(MAGICORE.solver_user, base) },
  ], { temperature: 0.7, maxTokens: 4096 });
  if (!solver) { logStep('step 1 (solver)', { llmOk: false, parseOk: false, evalOk: false }); return; }

  let data = parseSolutionJSON(solver.content);
  if (!data) { logStep('step 1 (solver)', { llmOk: true, parseOk: false, evalOk: false, rawHead: solver.content }); return; }
  data = normalizeMaxCut(data);

  let ev = await evalStep(data);
  logStep('step 1 (solver)', { llmOk: true, parseOk: true, evalOk: ev.ok, devScore: ev.score, error: ev.error });
  if (!ev.ok) return;

  let currentScore = ev.score;
  let currentData = data;

  for (let r = 0; r < rounds; r++) {
    if (!budget.canCallLLM) { console.log(`  round ${r + 1}: canCallLLM=false — STOP`); return; }

    const reviewVars = { ...base, SCORE: String(currentScore), SOLUTION: JSON.stringify(currentData, null, 2), DIRECTION: task.scoringDirection };
    const review = await budget.llmCall(backbone, [
      { role: 'system', content: MAGICORE.reviewer_system },
      { role: 'user',   content: fill(MAGICORE.reviewer_user, reviewVars) },
    ], { temperature: 0.3, maxTokens: 4096 });
    if (!review) { console.log(`  round ${r + 1} reviewer: llmOk=false — STOP`); return; }

    if (!budget.canCallLLM) { console.log(`  round ${r + 1} refiner: canCallLLM=false — STOP`); return; }
    if (!budget.canCallEval) { console.log(`  round ${r + 1} refiner: canCallEval=false — STOP`); return; }

    const refineVars = { ...reviewVars, CRITIQUE: review.content };
    const refiner = await budget.llmCall(backbone, [
      { role: 'system', content: MAGICORE.refiner_system },
      { role: 'user',   content: fill(MAGICORE.refiner_user, refineVars) },
    ], { temperature: 0.5, maxTokens: 4096 });
    if (!refiner) { logStep(`round ${r + 1} refiner`, { llmOk: false, parseOk: false, evalOk: false }); return; }

    let rdata = parseSolutionJSON(refiner.content);
    if (!rdata) {
      logStep(`round ${r + 1} refiner`, { llmOk: true, parseOk: false, evalOk: false, rawHead: refiner.content });
      continue;
    }
    rdata = normalizeMaxCut(rdata);

    const rev = await evalStep(rdata);
    logStep(`round ${r + 1} refiner`, { llmOk: true, parseOk: true, evalOk: rev.ok, devScore: rev.score, error: rev.error });

    if (rev.ok && (task.scoringDirection === 'maximize' ? rev.score > currentScore : rev.score < currentScore)) {
      currentScore = rev.score;
      currentData = rdata;
    }
  }
}

// ── Main ──────────────────────────────────────────────────────────────────────

const which = (() => {
  const idx = process.argv.indexOf('--protocol');
  return idx >= 0 ? process.argv[idx + 1] : 'all';
})();

(async () => {
  if (which === 'all' || which === 'self-refine') await diagSelfRefine();
  if (which === 'all' || which === 'homo-chain') await diagHomoChain();
  if (which === 'all' || which === 'magicore') await diagMAgICoRe();

  const trace = budget.getTrace();
  console.log(`\nBudget: tokens=${trace.tokenUsage.total} evalCalls=${trace.evalCalls.length} wall=${Math.round(trace.wallClockMs / 1000)}s`);
})().catch(e => { console.error('FATAL:', e); process.exit(1); });
