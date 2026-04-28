#!/usr/bin/env tsx
/**
 * run-full.ts — Full benchmark entrypoint
 *
 * Runs the complete 8 tasks × 11 protocols × 5 seeds = 440-cell benchmark.
 * Each cell submits its best solution to agent4science.org as a real submission.
 *
 * Usage:
 *   ADMIN_SECRET=xxx ANTHROPIC_API_KEY=yyy npx tsx src/runner/run-full.ts
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-full.ts --dry-run   # validate config, no LLM calls
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-full.ts --task bench-maxcut-g200  # single task
 *   ADMIN_SECRET=xxx npx tsx src/runner/run-full.ts --protocol single-shot    # single protocol
 *
 * Required env vars:
 *   ADMIN_SECRET         agent4science.org admin key (for benchmark evaluation endpoint)
 *   ANTHROPIC_API_KEY    for Claude Sonnet 4 (primary backbone)
 *   OPENAI_API_KEY       for GPT-4o (cross-chain + MoA)
 *   OPENROUTER_API_KEY   for Gemini 2.5 Flash (cross-chain + MoA)
 */

import 'dotenv/config';
import { runExperiment } from './experiment-runner.js';
import { BENCHMARK_SEEDS } from '../config/seeds.js';
import { primaryBackbone, crossChainBackbones, crossChainOSSBackbones, moaBackbones, moaBackbonesSameModel, fiveDiverseBackbones, deepseekBackbone, gpt4oBackbone, geminiBackbone } from '../config/models.js';
import { ALL_BENCHMARK_CHALLENGES } from '../tasks/benchmark-challenges.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';
import type { ExperimentConfig, BenchmarkTask, ProtocolConfig } from '../types.js';

// ── CLI args ─────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const taskIdx = args.indexOf('--task');
const taskFilter = taskIdx >= 0
  ? args[taskIdx + 1].split(',').map(s => s.trim()).filter(Boolean)
  : undefined;
const protocolIdx = args.indexOf('--protocol');
// Comma-separated list, e.g. --protocol magicore-mixed,debate-mixed,hpe-mixed.
// Used by the smoke-test pipeline to validate the mixed-brain wiring on a
// single task in one shot before launching the full run.
const protocolFilter = protocolIdx >= 0
  ? args[protocolIdx + 1].split(',').map(s => s.trim()).filter(Boolean)
  : undefined;
const seedsIdx = args.indexOf('--seeds');
const seedsFilter = seedsIdx >= 0
  ? args[seedsIdx + 1].split(',').map(s => Number(s.trim())).filter(n => Number.isFinite(n))
  : undefined;

// ── Task definitions ──────────────────────────────────────────────────────────

// ── Task-specific output normalizers ─────────────────────────────────────────

/**
 * MaxCut: ensure partition is exactly 200 binary integers.
 * LLMs occasionally generate wrong-length arrays; truncate or pad rather than reject.
 */
function normalizeMaxCut(data: Record<string, unknown>): Record<string, unknown> {
  const raw = data['partition'];
  if (!Array.isArray(raw)) return data;
  let arr = (raw as unknown[]).map((v) => (Number(v) >= 1 ? 1 : 0));
  if (arr.length > 200) arr = arr.slice(0, 200);
  while (arr.length < 200) arr.push(0);
  return { ...data, partition: arr };
}

const NORMALIZERS: Partial<Record<string, (d: Record<string, unknown>) => Record<string, unknown>>> = {
  'bench-maxcut-g200': normalizeMaxCut,
};

function challengeToTask(ch: typeof ALL_BENCHMARK_CHALLENGES[0]): BenchmarkTask {
  return {
    id: ch.id,
    title: ch.title,
    track: 'scientific',
    prompt: ch.description,
    devVerifier: ch.verifier,
    hiddenVerifier: ch.hiddenVerifier,
    solutionSchema: { format: ch.solutionSchema },
    scoringDirection: ch.scoringDirection,
    challengeId: BENCHMARK_CHALLENGE_IDS[ch.id],
    parameters: {},
    normalizeOutput: NORMALIZERS[ch.id],
  };
}

const ALL_TASKS: BenchmarkTask[] = ALL_BENCHMARK_CHALLENGES.map(challengeToTask);

const tasks = taskFilter
  ? ALL_TASKS.filter(t => taskFilter.includes(t.id))
  : ALL_TASKS;

if (tasks.length === 0) {
  console.error(`No tasks matched filter: ${taskFilter}`);
  process.exit(1);
}

// ── Protocol definitions (all 11) ────────────────────────────────────────────

const ALL_PROTOCOLS: ProtocolConfig[] = [
  {
    id: 'single-shot',
    name: 'Single Shot',
    backbones: [primaryBackbone()],
    params: {},
  },
  {
    id: 'self-refine',
    name: 'Self-Refine (3 rounds)',
    backbones: [primaryBackbone()],
    params: { rounds: 3 },
  },
  {
    id: 'best-of-n',
    name: 'Best-of-N (N=8)',
    backbones: [primaryBackbone()],
    params: { n: 8 },
  },
  {
    id: 'vgs',
    name: 'VGS (pop=4, gen=5)',
    backbones: [primaryBackbone()],
    params: { populationSize: 4, generations: 5, eliteCount: 2, convergenceWindow: 2 },
  },
  {
    id: 'homo-chain',
    name: 'Homo-Chain (length=4)',
    backbones: [primaryBackbone()],
    params: { chainLength: 4 },
  },
  {
    id: 'cross-chain',
    name: 'Cross-Chain (Claude→GPT-4o→Gemini)',
    backbones: crossChainBackbones(),
    params: {},
  },
  {
    id: 'magicore',
    name: 'MAgICoRe (2 rounds)',
    backbones: [primaryBackbone()],
    params: { rounds: 2 },
  },
  {
    id: 'debate',
    name: 'Debate (2 rounds)',
    backbones: [primaryBackbone()],
    params: { debateRounds: 2 },
  },
  {
    id: 'moa',
    name: 'Mixture-of-Agents (Claude+GPT-4o+Gemini)',
    backbones: moaBackbones(),
    params: {},
  },
  {
    id: 'hpe',
    name: 'Hierarchical Planner-Executor (3 executors, 2 rounds)',
    backbones: [primaryBackbone()],
    params: { numExecutors: 3, rounds: 2 },
  },
  // ── Matched N=3 control for clean 2×2 ──────────────────────────────────
  // Best-of-3: 3 proposals from Claude, select best by dev score.
  // Matches the N=3 proposal count of all MoA arms, eliminating the
  // team-size confound that Best-of-8 introduces.
  {
    id: 'best-of-3',
    name: 'Best-of-3 (Claude×3, matched N=3)',
    backbones: [primaryBackbone()],
    params: { n: 3 },
  },
  // ── MoA coverage ablation ──────────────────────────────────────────────
  // Same-model MoA: all 3 proposers are Claude (instead of Claude+GPT-4o+Gemini).
  // Tests whether MoA benefit comes from model diversity or independent generation.
  {
    id: 'moa-same-model',
    name: 'MoA Same-Model (Claude×3)',
    backbones: moaBackbonesSameModel(),
    params: {},
  },
  // No-synthesis MoA: diverse proposers but no synthesis step — just pick best.
  // Completes the 2×2: diversity (same/diverse) × synthesis (with/without).
  {
    id: 'moa-nosynth',
    name: 'MoA No-Synth (Claude+GPT-4o+Gemini, no synthesis)',
    backbones: moaBackbones(),
    params: {},
  },
  // ── Prescriptive 5-diverse protocol ──────────────────────────────────────
  // Derived from the diversity tax analysis: maximize independence by using
  // 5 diverse model families (Claude+GPT-4o+Gemini+DeepSeek+Qwen) with zero
  // interaction. Each model generates one proposal independently; best by
  // dev score wins. Tests whether the diversity tax theory is actionable.
  {
    id: 'best-of-5-diverse',
    name: 'Best-of-5 Diverse (Claude+GPT-4o+Gemini+DeepSeek+Qwen, no interaction)',
    backbones: fiveDiverseBackbones(),
    params: {},
  },
  // ── Mixed-brain variants ────────────────────────────────────────────────
  // Same protocol structure as the homogeneous originals above, but each
  // role gets a distinct backbone (Claude / GPT-4o / Gemini). Run alongside
  // the originals so the paper can answer "do different brains in different
  // roles help, or is multi-agent benefit just more rounds of the same brain?"
  // — within one experiment, with all other variables held constant.
  {
    // solver=Claude, reviewer=GPT-4o, refiner=Gemini
    id: 'magicore-mixed',
    name: 'MAgICoRe-Mixed (Claude/GPT-4o/Gemini, 2 rounds)',
    backbones: crossChainBackbones(),
    params: { rounds: 2 },
  },
  {
    // debater A=Claude, debater B=GPT-4o, synthesizer=Gemini
    // Each debater's critiques use its own backbone.
    id: 'debate-mixed',
    name: 'Debate-Mixed (Claude vs GPT-4o → Gemini synth, 2 rounds)',
    backbones: crossChainBackbones(),
    params: { debateRounds: 2 },
  },
  {
    // planner+integrator=Claude, executors round-robin Claude/GPT-4o/Gemini
    // (with numExecutors=3, every executor runs on a different model).
    id: 'hpe-mixed',
    name: 'HPE-Mixed (Claude planner, mixed executors, 2 rounds)',
    backbones: crossChainBackbones(),
    params: { numExecutors: 3, rounds: 2 },
  },
  // ── Open-source mixed-brain variants ────────────────────────────────────
  // Identical to the *-mixed entries above, but every backbone is an
  // open-weight model: DeepSeek V3 / Llama 3.3 70B / Qwen 2.5 72B. Lets the
  // paper decouple "more brains help" from "more frontier API spend helps":
  // if the OSS arm also shows multi-agent benefit, the effect is intrinsic
  // to brain diversity, not a function of any single capable model.
  {
    // solver=DeepSeek, reviewer=Llama, refiner=Qwen
    id: 'magicore-mixed-oss',
    name: 'MAgICoRe-Mixed-OSS (DeepSeek/Llama/Qwen, 2 rounds)',
    backbones: crossChainOSSBackbones(),
    params: { rounds: 2 },
  },
  {
    // debater A=DeepSeek, debater B=Llama, synthesizer=Qwen
    id: 'debate-mixed-oss',
    name: 'Debate-Mixed-OSS (DeepSeek vs Llama → Qwen synth, 2 rounds)',
    backbones: crossChainOSSBackbones(),
    params: { debateRounds: 2 },
  },
  {
    // planner+integrator=DeepSeek, executors round-robin DeepSeek/Llama/Qwen
    id: 'hpe-mixed-oss',
    name: 'HPE-Mixed-OSS (DeepSeek planner, mixed executors, 2 rounds)',
    backbones: crossChainOSSBackbones(),
    params: { numExecutors: 3, rounds: 2 },
  },
  // ── Homogeneous-DeepSeek variants (brain-diversity 2×2 closer) ──────────
  // The brain-diversity story currently has three of the four corners of a
  // (frontier vs OSS) × (homogeneous vs mixed) design: homogeneous-frontier
  // (the magicore/debate/hpe entries above), frontier-mixed (*-mixed), and
  // OSS-mixed (*-mixed-oss). The missing corner is homogeneous-OSS. These
  // three cells fill it by running the same three role-structured protocols
  // with every role pointing at DeepSeek V3 (the strongest open-weight model
  // in the ablation set). With this fourth corner in hand, the paper can
  // make a full 2×2 claim about whether brain diversity helps independently
  // of frontier-model capability, rather than the one-sided check currently
  // acknowledged as limitation 3 in Section sec:status.
  //
  // Cost note: OSS tokens via OpenRouter are ~10× cheaper than frontier
  // tokens, so the full 8-task × 3-protocol × 5-seed = 120-cell sweep costs
  // roughly half of what the 240-cell brain-diversity sweep costs.
  //
  // Launch via:
  //   npx tsx src/runner/run-full.ts --protocol magicore-deepseek,debate-deepseek,hpe-deepseek
  {
    id: 'magicore-deepseek',
    name: 'MAgICoRe-DeepSeek (homogeneous OSS, 2 rounds)',
    backbones: [deepseekBackbone()],
    params: { rounds: 2 },
  },
  {
    id: 'debate-deepseek',
    name: 'Debate-DeepSeek (homogeneous OSS, 2 rounds)',
    backbones: [deepseekBackbone()],
    params: { debateRounds: 2 },
  },
  {
    id: 'hpe-deepseek',
    name: 'HPE-DeepSeek (homogeneous OSS, 2 rounds)',
    backbones: [deepseekBackbone()],
    params: { numExecutors: 3, rounds: 2 },
  },
  // ── DeepSeek single-agent controls (MEG denominator fix) ────────────────
  // The OSS MEG for magicore-deepseek/debate-deepseek/hpe-deepseek is
  // currently computed against Claude SR/BoN/VGS controls, which conflates
  // model capability with protocol value. These three entries provide the
  // correct DeepSeek-specific denominator: MEG(p_deepseek) =
  //   E[Q(y_p)] − max(E[Q_SR-deepseek], E[Q_BoN-deepseek], E[Q_VGS-deepseek])
  //
  // Run only on the 5 tasks that completed for magicore-deepseek:
  //   npx tsx src/runner/run-full.ts \
  //     --protocol self-refine-deepseek,best-of-n-deepseek,vgs-deepseek \
  //     --task bench-maxcut-g200,bench-circle-packing-n20,bench-difference-bases,bench-erdos-overlap,bench-molecule-qed
  {
    id: 'self-refine-deepseek',
    name: 'Self-Refine-DeepSeek (OSS control, 3 rounds)',
    backbones: [deepseekBackbone()],
    params: { rounds: 3 },
  },
  {
    id: 'best-of-n-deepseek',
    name: 'Best-of-N-DeepSeek (OSS control, N=8)',
    backbones: [deepseekBackbone()],
    params: { n: 8 },
  },
  {
    id: 'vgs-deepseek',
    name: 'VGS-DeepSeek (OSS control, pop=4, gen=5)',
    backbones: [deepseekBackbone()],
    params: { populationSize: 4, generations: 5, eliteCount: 2, convergenceWindow: 2 },
  },
  // ── GPT-4o Solo variants ─────────────────────────────────────────────────
  // Identical to the homogeneous-Claude entries above but with GPT-4o as the
  // sole backbone. Lets the paper ask "does the benefit pattern depend on the
  // model, or is it universal across frontier models?"
  {
    id: 'single-shot-gpt4o',
    name: 'Single Shot (GPT-4o)',
    backbones: [gpt4oBackbone()],
    params: {},
  },
  {
    id: 'self-refine-gpt4o',
    name: 'Self-Refine (GPT-4o, 3 rounds)',
    backbones: [gpt4oBackbone()],
    params: { rounds: 3 },
  },
  {
    id: 'best-of-n-gpt4o',
    name: 'Best-of-N (GPT-4o, N=8)',
    backbones: [gpt4oBackbone()],
    params: { n: 8 },
  },
  {
    id: 'vgs-gpt4o',
    name: 'VGS (GPT-4o, pop=4, gen=5)',
    backbones: [gpt4oBackbone()],
    params: { populationSize: 4, generations: 5, eliteCount: 2, convergenceWindow: 2 },
  },
  {
    id: 'homo-chain-gpt4o',
    name: 'Homo-Chain (GPT-4o, length=4)',
    backbones: [gpt4oBackbone()],
    params: { chainLength: 4 },
  },
  {
    id: 'magicore-gpt4o',
    name: 'MAgICoRe (GPT-4o, 2 rounds)',
    backbones: [gpt4oBackbone()],
    params: { rounds: 2 },
  },
  {
    id: 'debate-gpt4o',
    name: 'Debate (GPT-4o, 2 rounds)',
    backbones: [gpt4oBackbone()],
    params: { debateRounds: 2 },
  },
  {
    id: 'hpe-gpt4o',
    name: 'HPE (GPT-4o, 3 executors, 2 rounds)',
    backbones: [gpt4oBackbone()],
    params: { numExecutors: 3, rounds: 2 },
  },
  // ── Gemini Solo variants ─────────────────────────────────────────────────
  // Identical to the homogeneous-Claude entries above but with Gemini 2.5 Flash
  // as the sole backbone.
  {
    id: 'single-shot-gemini',
    name: 'Single Shot (Gemini)',
    backbones: [geminiBackbone()],
    params: {},
  },
  {
    id: 'self-refine-gemini',
    name: 'Self-Refine (Gemini, 3 rounds)',
    backbones: [geminiBackbone()],
    params: { rounds: 3 },
  },
  {
    id: 'best-of-n-gemini',
    name: 'Best-of-N (Gemini, N=8)',
    backbones: [geminiBackbone()],
    params: { n: 8 },
  },
  {
    id: 'vgs-gemini',
    name: 'VGS (Gemini, pop=4, gen=5)',
    backbones: [geminiBackbone()],
    params: { populationSize: 4, generations: 5, eliteCount: 2, convergenceWindow: 2 },
  },
  {
    id: 'homo-chain-gemini',
    name: 'Homo-Chain (Gemini, length=4)',
    backbones: [geminiBackbone()],
    params: { chainLength: 4 },
  },
  {
    id: 'magicore-gemini',
    name: 'MAgICoRe (Gemini, 2 rounds)',
    backbones: [geminiBackbone()],
    params: { rounds: 2 },
  },
  {
    id: 'debate-gemini',
    name: 'Debate (Gemini, 2 rounds)',
    backbones: [geminiBackbone()],
    params: { debateRounds: 2 },
  },
  {
    id: 'hpe-gemini',
    name: 'HPE (Gemini, 3 executors, 2 rounds)',
    backbones: [geminiBackbone()],
    params: { numExecutors: 3, rounds: 2 },
  },
];

const protocols = protocolFilter
  ? ALL_PROTOCOLS.filter(p => protocolFilter.includes(p.id))
  : ALL_PROTOCOLS;

if (protocols.length === 0) {
  console.error(`No protocols matched filter: ${protocolFilter}`);
  process.exit(1);
}

// ── Budget vector (Scientific track) ─────────────────────────────────────────

const BUDGET = {
  tokenCap: 200_000,       // T: 200K tokens per cell (prompt + completion)
  wallClockSeconds: 600,   // W: 10 minutes per cell
  evalCpuSeconds: 30,      // C: 30s evaluator CPU per cell
  evalCallCap: 25,         // K: 25 dev evaluator calls per cell
};

// ── Experiment config ─────────────────────────────────────────────────────────

const seeds: number[] = seedsFilter ?? [...BENCHMARK_SEEDS];

const config: ExperimentConfig = {
  id: 'agent4science-bench-full-v2',
  name: 'Agent4Science-Bench Full Run (8×11×5)',
  tasks,
  protocols,
  budget: BUDGET,
  seeds,
  outputDir: 'results/full-v2',
  submitForHiddenEval: true,
  apiUrl: process.env.A4S_API_URL || 'https://agent4science.org',
  apiKey: process.env.ADMIN_SECRET,
  agentId: '', // Claude Sonnet 4 — fallback for all single-model frontier protocols
  protocolAgentMap: {
    // Frontier single-backbone protocols → Claude Sonnet 4 Solo agent
    'single-shot':    '',
    'best-of-n':      '',
    'self-refine':    '',
    'vgs':            '',
    'homo-chain':     '',
    'magicore':       '',
    'debate':         '',
    'moa':            '',
    'hpe':            '',
    'best-of-3':      '',
    'moa-same-model': '',
    'moa-nosynth':        '',  // diverse models → Frontier Mix agent
    'best-of-5-diverse':  '',  // 5 diverse models → Frontier Mix agent
    // Frontier multi-model protocols → Frontier Mix agent (Claude+GPT-4o+Gemini)
    'cross-chain':    '',
    'magicore-mixed': '',
    'debate-mixed':   '',
    'hpe-mixed':      '',
    // OSS mixed-brain protocols → OSS Mix agent
    'magicore-mixed-oss': '',
    'debate-mixed-oss':   '',
    'hpe-mixed-oss':      '',
    // Homogeneous DeepSeek protocols → DeepSeek V3 Solo agent
    'magicore-deepseek':    '',
    'debate-deepseek':      '',
    'hpe-deepseek':         '',
    // DeepSeek single-agent controls → DeepSeek V3 Solo agent
    'self-refine-deepseek': '',
    'best-of-n-deepseek':   '',
    'vgs-deepseek':         '',
    // GPT-4o Solo protocols → GPT-4o Solo agent
    'single-shot-gpt4o':  'agent_mnrvsunlm96jmexa',
    'self-refine-gpt4o':  'agent_mnrvsunlm96jmexa',
    'best-of-n-gpt4o':    'agent_mnrvsunlm96jmexa',
    'vgs-gpt4o':          'agent_mnrvsunlm96jmexa',
    'homo-chain-gpt4o':   'agent_mnrvsunlm96jmexa',
    'magicore-gpt4o':     'agent_mnrvsunlm96jmexa',
    'debate-gpt4o':       'agent_mnrvsunlm96jmexa',
    'hpe-gpt4o':          'agent_mnrvsunlm96jmexa',
    // Gemini Solo protocols → Gemini Solo agent
    'single-shot-gemini': 'agent_mnrvsuvgu5ubpcz9',
    'self-refine-gemini': 'agent_mnrvsuvgu5ubpcz9',
    'best-of-n-gemini':   'agent_mnrvsuvgu5ubpcz9',
    'vgs-gemini':         'agent_mnrvsuvgu5ubpcz9',
    'homo-chain-gemini':  'agent_mnrvsuvgu5ubpcz9',
    'magicore-gemini':    'agent_mnrvsuvgu5ubpcz9',
    'debate-gemini':      'agent_mnrvsuvgu5ubpcz9',
    'hpe-gemini':         'agent_mnrvsuvgu5ubpcz9',
  },
};

// ── Run ───────────────────────────────────────────────────────────────────────

const totalCells = tasks.length * protocols.length * seeds.length;
console.log('Agent4Science-Bench Full Run');
console.log(`Tasks:     ${tasks.map(t => t.id).join(', ')}`);
console.log(`Protocols: ${protocols.map(p => p.id).join(', ')}`);
console.log(`Seeds:     ${seeds.join(', ')}`);
console.log(`Cells:     ${totalCells}`);
console.log(`Output:    ${config.outputDir}/`);
console.log(`Submit:    ${config.submitForHiddenEval ? 'YES — agent4science.org' : 'no'}`);

if (dryRun) {
  console.log('\nDry run — config valid. Exiting without running.');
  process.exit(0);
}

if (!process.env.ADMIN_SECRET) {
  console.error('\nERROR: ADMIN_SECRET env var required');
  process.exit(1);
}

runExperiment(config).then(() => {
  console.log('\nDone.');
  process.exit(0);
}).catch((err) => {
  console.error('\nFATAL:', err);
  process.exit(1);
});
