/**
 * Multi-Agent Protocols: Sequential Chain, Cross-Model Chain, MAgICoRe, Debate, MoA
 *
 * These are the interaction protocols being tested against the MEG denominator.
 */

import type { BenchmarkTask, BackboneConfig, SolutionArtifact } from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import type { LLMMessage } from '../lib/llm-client.js';
import { HOMO_CHAIN, CROSS_CHAIN, MAGICORE, DEBATE, MOA, fill } from '../config/prompts.js';

interface InitialResult {
  data: Record<string, unknown>;
  score: number;
  raw: string;
}

/**
 * Generate + parse + dev-verifier-eval an artifact, retrying up to
 * maxAttempts times on parse failure or verifier rejection. Returns null
 * only if every attempt failed (or the LLM/eval budget was exhausted).
 *
 * On retries, the verifier error message is appended to the user prompt so
 * the LLM can self-correct (e.g., "Expected 200 labels, got 199" gives
 * Claude enough signal to fix the next attempt — without this, the LLM
 * silently makes the same mistake repeatedly).
 *
 * Without this retry, an LLM that produces a malformed maxcut solution on
 * its first try makes the entire chain cell collapse to 0 submissions.
 */
async function generateInitial(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  messages: LLMMessage[],
  temperature: number,
  maxAttempts = 5,
): Promise<InitialResult | null> {
  let lastError: string | null = null;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (!budget.canCallLLM) return null;

    // On retries, append a self-correction hint to the user message
    const attemptMessages: LLMMessage[] = lastError !== null
      ? messages.map((m, i) => i === messages.length - 1
          ? { ...m, content: `${m.content}\n\n# Previous Attempt Failed\nThe verifier rejected your previous solution: ${lastError}\nProduce a NEW solution that fixes this issue.` }
          : m)
      : messages;

    const response = await budget.llmCall(backbone, attemptMessages, { temperature });
    if (!response) return null;

    const data = parseSolutionJSON(response.content);
    if (!data) {
      lastError = 'output was not valid JSON matching the schema';
      continue;
    }
    const normalized = normalize(task, data);

    if (!budget.canCallEval || !task.challengeId) return null;
    try {
      const evalResult = await budget.evalDev(task.challengeId, normalized);
      if (evalResult === null) return null;
      return { data: normalized, score: evalResult.devScore, raw: response.content };
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
      // Strip noisy prefixes for cleaner LLM feedback
      lastError = lastError.replace(/^Dev eval failed:\s*/, '').replace(/^Verifier error:\s*/, '');
      continue;
    }
  }
  return null;
}

function parseSolutionJSON(raw: string): Record<string, unknown> | null {
  try { return JSON.parse(raw.trim()); } catch { /* fall through */ }
  const match = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (match) { try { return JSON.parse(match[1].trim()); } catch { /* fall through */ } }
  const braceMatch = raw.match(/\{[\s\S]*\}/);
  if (braceMatch) { try { return JSON.parse(braceMatch[0]); } catch { /* fall through */ } }
  return null;
}

function normalize(task: BenchmarkTask, data: Record<string, unknown>): Record<string, unknown> {
  return task.normalizeOutput ? task.normalizeOutput(data) : data;
}

/**
 * Run the dev verifier on a candidate solution.
 *
 * Returns:
 *   - number: legit verifier output (may be 0, may be negative — both valid)
 *   - null:   eval was unavailable, OR the verifier raised an error (e.g.,
 *             "Expected 200 labels, got 199"). Callers MUST treat null as
 *             "this artifact is malformed and should not be stored or
 *             submitted" — pushing it with a sentinel score causes the
 *             downstream evalFinal to fail and abort the entire chain
 *             submission. (This was the Run-1 latent bug exposed when we
 *             removed the `devScore > 0` filter from submitArtifacts.)
 */
async function evalSolution(
  task: BenchmarkTask,
  solutionData: Record<string, unknown>,
  budget: BudgetEnforcer,
): Promise<number | null> {
  if (!budget.canCallEval || !task.challengeId) return null;
  try {
    const result = await budget.evalDev(task.challengeId, solutionData);
    return result?.devScore ?? null;
  } catch {
    return null;
  }
}

// ── Homogeneous Sequential Chain ──
// Same model, agents take turns refining each other's output

export async function runHomoChain(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  chainLength = 4,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];

  const base = { PROBLEM: task.prompt, SCHEMA: JSON.stringify(task.solutionSchema, null, 2), PARAMS: JSON.stringify(task.parameters ?? {}, null, 2) };

  // Agent 1: generate (with retry on parse/verifier failure)
  const initial = await generateInitial(task, backbone, budget, [
    { role: 'system', content: HOMO_CHAIN.first_system },
    { role: 'user',   content: fill(HOMO_CHAIN.first_user, base) },
  ], 0.7);
  if (!initial) return artifacts;

  let currentData = initial.data;
  let currentScore: number = initial.score;
  artifacts.push({ solutionData: currentData, devScore: currentScore, rawOutput: initial.raw });

  // Agents 2-N: refine (with retry on parse/verifier failure, maxAttempts=2 to conserve budget)
  for (let i = 1; i < chainLength; i++) {
    if (!budget.canCallLLM || !budget.canCallEval) break;

    const refineVars = { ...base, AGENT_NUM: String(i + 1), DIRECTION: task.scoringDirection, SCORE: String(currentScore), SOLUTION: JSON.stringify(currentData, null, 2) };
    const refined = await generateInitial(task, backbone, budget, [
      { role: 'system', content: fill(HOMO_CHAIN.refine_system, refineVars) },
      { role: 'user',   content: fill(HOMO_CHAIN.refine_user,   refineVars) },
    ], 0.5, 2);
    if (!refined) continue;
    artifacts.push({ solutionData: refined.data, devScore: refined.score, rawOutput: refined.raw });

    const improved = task.scoringDirection === 'maximize' ? refined.score > currentScore : refined.score < currentScore;
    if (improved) {
      currentData = refined.data;
      currentScore = refined.score;
    }
  }

  return artifacts;
}

// ── Cross-Model Sequential Chain ──
// Different models (Claude → GPT → Gemini), same structure

export async function runCrossChain(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];

  if (backbones.length === 0 || !budget.canCallLLM) return artifacts;

  const base = { PROBLEM: task.prompt, SCHEMA: JSON.stringify(task.solutionSchema, null, 2), PARAMS: JSON.stringify(task.parameters ?? {}, null, 2) };

  // First model: generate (with retry on parse/verifier failure)
  const initial = await generateInitial(task, backbones[0], budget, [
    { role: 'system', content: CROSS_CHAIN.first_system },
    { role: 'user',   content: fill(CROSS_CHAIN.first_user, base) },
  ], 0.7);
  if (!initial) {
    console.log(`  [cross-chain] first model failed all retries`);
    return artifacts;
  }

  let currentData = initial.data;
  let currentScore: number = initial.score;
  artifacts.push({ solutionData: currentData, devScore: currentScore, rawOutput: initial.raw });

  // Subsequent models: refine (with retry on parse/verifier failure)
  for (let i = 1; i < backbones.length; i++) {
    if (!budget.canCallLLM || !budget.canCallEval) break;

    const refineVars = { ...base, SCORE: String(currentScore), SOLUTION: JSON.stringify(currentData, null, 2) };
    const refined = await generateInitial(task, backbones[i], budget, [
      { role: 'system', content: CROSS_CHAIN.refine_system },
      { role: 'user',   content: fill(CROSS_CHAIN.refine_user, refineVars) },
    ], 0.5, 2);
    if (!refined) continue;
    artifacts.push({ solutionData: refined.data, devScore: refined.score, rawOutput: refined.raw });

    const improved = task.scoringDirection === 'maximize' ? refined.score > currentScore : refined.score < currentScore;
    if (improved) {
      currentData = refined.data;
      currentScore = refined.score;
    }
  }

  return artifacts;
}

// ── MAgICoRe: Solver → Reviewer → Refiner ──
//
// Role-based backbone assignment:
//   solver   = backbones[0]
//   reviewer = backbones[1] ?? backbones[0]
//   refiner  = backbones[2] ?? backbones[0]
//
// With backbones=[claude, gpt4o, gemini], MAgICoRe becomes a true mixed-brain
// pipeline: Claude generates, GPT-4o critiques, Gemini refines. With a single
// backbone, behavior is identical to the original homogeneous version (every
// role uses the same model). This is what makes the protocol an interesting
// "more minds = different brains" test rather than just "more rounds of the
// same brain talking to itself".

export async function runMAgICoRe(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
  rounds = 2,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];

  if (backbones.length === 0) return artifacts;
  const solver   = backbones[0];
  const reviewer = backbones[1] ?? backbones[0];
  const refiner  = backbones[2] ?? backbones[0];

  const base = { PROBLEM: task.prompt, SCHEMA: JSON.stringify(task.solutionSchema, null, 2), PARAMS: JSON.stringify(task.parameters ?? {}, null, 2) };

  // Solver: generate (with retry on parse/verifier failure)
  const initial = await generateInitial(task, solver, budget, [
    { role: 'system', content: MAGICORE.solver_system },
    { role: 'user',   content: fill(MAGICORE.solver_user, base) },
  ], 0.7);
  if (!initial) return artifacts;

  let currentData = initial.data;
  let currentScore: number = initial.score;
  artifacts.push({ solutionData: currentData, devScore: currentScore, rawOutput: initial.raw });

  for (let r = 0; r < rounds; r++) {
    if (!budget.canCallLLM) break;

    // Reviewer: critique
    const reviewVars = { ...base, SCORE: String(currentScore), SOLUTION: JSON.stringify(currentData, null, 2), DIRECTION: task.scoringDirection };
    const reviewResponse = await budget.llmCall(reviewer, [
      { role: 'system', content: MAGICORE.reviewer_system },
      { role: 'user',   content: fill(MAGICORE.reviewer_user, reviewVars) },
    ], { temperature: 0.3 });
    if (!reviewResponse) break;

    if (!budget.canCallLLM || !budget.canCallEval) break;

    // Refiner: improve based on critique (with retry on parse/verifier failure)
    const refineVars = { ...reviewVars, CRITIQUE: reviewResponse.content };
    const refined = await generateInitial(task, refiner, budget, [
      { role: 'system', content: MAGICORE.refiner_system },
      { role: 'user',   content: fill(MAGICORE.refiner_user, refineVars) },
    ], 0.5, 2);
    if (!refined) continue;
    artifacts.push({ solutionData: refined.data, devScore: refined.score, rawOutput: refined.raw });

    const improved = task.scoringDirection === 'maximize' ? refined.score > currentScore : refined.score < currentScore;
    if (improved) {
      currentData = refined.data;
      currentScore = refined.score;
    }
  }

  return artifacts;
}

// ── Debate ──
// Two agents argue for different strategies, third agent synthesizes
//
// Role-based backbone assignment:
//   debater A (incl. its critiques) = backbones[0]
//   debater B (incl. its critiques) = backbones[1] ?? backbones[0]
//   synthesizer                     = backbones[2] ?? backbones[0]
//
// With three distinct backbones, debate becomes a real "different brains
// argue" experiment instead of one model role-playing both sides. With one
// backbone, behavior is identical to the original homogeneous version.

export async function runDebate(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
  debateRounds = 2,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];

  if (backbones.length === 0) return artifacts;
  const debaterA    = backbones[0];
  const debaterB    = backbones[1] ?? backbones[0];
  const synthesizer = backbones[2] ?? backbones[0];

  const base = { PROBLEM: task.prompt, SCHEMA: JSON.stringify(task.solutionSchema, null, 2), PARAMS: JSON.stringify(task.parameters ?? {}, null, 2) };

  // Agent A: generate with one strategy emphasis (with retry)
  const aResult = await generateInitial(task, debaterA, budget, [
    { role: 'system', content: DEBATE.debater_a_system },
    { role: 'user',   content: fill(DEBATE.debater_a_user, base) },
  ], 0.9);

  // Agent B: generate with a different strategy emphasis (with retry)
  const bResult = await generateInitial(task, debaterB, budget, [
    { role: 'system', content: DEBATE.debater_b_system },
    { role: 'user',   content: fill(DEBATE.debater_b_user, base) },
  ], 0.5);

  // If neither agent produced anything valid, abort the debate
  if (!aResult && !bResult) return artifacts;

  const scoreA: number | null = aResult?.score ?? null;
  const scoreB: number | null = bResult?.score ?? null;

  if (aResult) artifacts.push({ solutionData: aResult.data, devScore: aResult.score, rawOutput: aResult.raw });
  if (bResult) artifacts.push({ solutionData: bResult.data, devScore: bResult.score, rawOutput: bResult.raw });

  // Debate rounds: each agent critiques the other and revises
  let posA = aResult?.raw ?? '(no valid solution)';
  let posB = bResult?.raw ?? '(no valid solution)';
  for (let r = 0; r < debateRounds; r++) {
    if (!budget.canCallLLM) break;

    // A critiques B (uses debater A's backbone)
    const aVars = { SCORE_A: scoreA === null ? 'N/A' : String(scoreA), POSITION_A: posA, SCORE_B: scoreB === null ? 'N/A' : String(scoreB), POSITION_B: posB };
    const aCritique = await budget.llmCall(debaterA, [
      { role: 'system', content: DEBATE.a_critique_system },
      { role: 'user',   content: fill(DEBATE.a_critique_user, aVars) },
    ], { temperature: 0.5 });

    if (!budget.canCallLLM) break;

    // B critiques A (uses debater B's backbone)
    const bCritique = await budget.llmCall(debaterB, [
      { role: 'system', content: DEBATE.b_critique_system },
      { role: 'user',   content: fill(DEBATE.b_critique_user, aVars) },
    ], { temperature: 0.5 });

    if (aCritique) posA = aCritique.content;
    if (bCritique) posB = bCritique.content;
  }

  // Synthesizer: combine the best ideas
  if (!budget.canCallLLM || !budget.canCallEval) return artifacts;

  const synthVars = { ...base, SCORE_A: scoreA === null ? 'N/A' : String(scoreA), POSITION_A: posA, SCORE_B: scoreB === null ? 'N/A' : String(scoreB), POSITION_B: posB };
  const synthesis = await budget.llmCall(synthesizer, [
    { role: 'system', content: DEBATE.synthesizer_system },
    { role: 'user',   content: fill(DEBATE.synthesizer_user, synthVars) },
  ], { temperature: 0.5 });

  if (synthesis) {
    let synthData = parseSolutionJSON(synthesis.content);
    if (synthData) {
      synthData = normalize(task, synthData);
      const synthScore = await evalSolution(task, synthData, budget);
      if (synthScore !== null) {
        artifacts.push({ solutionData: synthData, devScore: synthScore, rawOutput: synthesis.content });
      }
    }
  }

  return artifacts;
}

// ── Mixture-of-Agents (No Synthesis) ──
// Multiple agents generate independently, best proposal wins by dev score

export async function runMoANoSynth(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];

  const base = { PROBLEM: task.prompt, SCHEMA: JSON.stringify(task.solutionSchema, null, 2), PARAMS: JSON.stringify(task.parameters ?? {}, null, 2) };

  for (const bb of backbones) {
    if (!budget.canCallLLM || !budget.canCallEval) break;

    const candidate = await generateInitial(task, bb, budget, [
      { role: 'system', content: MOA.agent_system },
      { role: 'user',   content: fill(MOA.agent_user, base) },
    ], 0.7);

    if (!candidate) {
      console.log(`  [moa-nosynth] ${bb.model}: failed all retries`);
      continue;
    }
    artifacts.push({ solutionData: candidate.data, devScore: candidate.score, rawOutput: candidate.raw });
  }

  // No aggregation/synthesis step — the best proposal is selected downstream
  // by getBestArtifact() based on dev scores already recorded above.
  return artifacts;
}

// ── Mixture-of-Agents ──
// Multiple agents generate independently, aggregator picks/combines

export async function runMoA(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
  aggregatorBackbone?: BackboneConfig,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];

  // Phase 1: Independent generation from each backbone
  const candidates: Array<{ data: Record<string, unknown>; score: number; raw: string }> = [];

  const base = { PROBLEM: task.prompt, SCHEMA: JSON.stringify(task.solutionSchema, null, 2), PARAMS: JSON.stringify(task.parameters ?? {}, null, 2) };

  for (const bb of backbones) {
    if (!budget.canCallLLM || !budget.canCallEval) break;

    const candidate = await generateInitial(task, bb, budget, [
      { role: 'system', content: MOA.agent_system },
      { role: 'user',   content: fill(MOA.agent_user, base) },
    ], 0.7);

    if (!candidate) {
      console.log(`  [moa] ${bb.model}: failed all retries`);
      continue;
    }
    candidates.push(candidate);
    artifacts.push({ solutionData: candidate.data, devScore: candidate.score, rawOutput: candidate.raw });
  }

  if (candidates.length < 2 || !budget.canCallLLM || !budget.canCallEval) return artifacts;

  // Phase 2: Aggregation
  const candidateSummary = candidates
    .map((c, i) => `## Agent ${i + 1} (score: ${c.score})\n${JSON.stringify(c.data, null, 2)}`)
    .join('\n\n');

  const aggregator = await budget.llmCall(aggregatorBackbone ?? backbones[0], [
    { role: 'system', content: MOA.aggregator_system },
    { role: 'user',   content: fill(MOA.aggregator_user, { ...base, CANDIDATES: candidateSummary }) },
  ], { temperature: 0.5 });

  if (aggregator) {
    let aggData = parseSolutionJSON(aggregator.content);
    if (aggData) {
      aggData = normalize(task, aggData);
      const aggScore = await evalSolution(task, aggData, budget);
      if (aggScore !== null) {
        artifacts.push({ solutionData: aggData, devScore: aggScore, rawOutput: aggregator.content });
      }
    }
  }

  return artifacts;
}
