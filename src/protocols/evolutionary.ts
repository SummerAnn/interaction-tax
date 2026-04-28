/**
 * Evolutionary Crossover Protocol
 *
 * A multi-agent protocol designed around explicit solution recombination
 * rather than iterative critique or consensus. Addresses W7 from the NeurIPS
 * review: "no comparison to learned/adaptive composition."
 *
 * Design:
 *   Round 1: N agents generate independent proposals (identical to MoA Phase 1).
 *   Round 2: A "crossover agent" receives ALL proposals and their scores, and
 *            explicitly recombines them — not by summarizing or critiquing, but
 *            by constructing a new solution that takes the best components from
 *            each. The prompt explicitly forbids averaging or picking one winner;
 *            it must construct a genuinely hybrid solution.
 *   Round 3: (Optional, if budget remains) The crossover output is sent back to
 *            each original agent with the instruction to make local improvements
 *            starting from the hybrid, one round of self-refine per agent.
 *
 * The key distinction from MoA/debate:
 *   - MoA aggregator: summarize and combine at a high level
 *   - Debate: critique and argue toward consensus
 *   - Crossover: explicit component-level recombination (like genetic crossover)
 *
 * The crossover step IS a form of interaction (agents see each other's solutions),
 * but the interaction is structured as recombination rather than convergence pressure.
 * This tests whether the diversity tax is about *any* interaction, or specifically
 * about consensus-building interaction.
 *
 * For composable tasks (DiffBases, Knapsack, Set Cover), crossover should perform
 * well because the task structure supports component-level merging.
 * For non-composable tasks (TSP), crossover should fail for the same reason MoA fails.
 */

import type { BenchmarkTask, BackboneConfig, SolutionArtifact } from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import type { LLMMessage } from '../lib/llm-client.js';

// ── Prompt templates ─────────────────────────────────────────────────────────

const CROSSOVER_SYSTEM = `You are an expert optimizer that recombines solutions.
You will receive multiple independently-generated solutions to an optimization problem, along with their scores.
Your task is to construct a NEW solution by explicitly combining the BEST COMPONENTS from each solution.

Rules:
- Do NOT simply copy the highest-scoring solution.
- Do NOT average or mix values randomly.
- Identify which components of each solution are independently strong, and combine them.
- The resulting solution must be valid (satisfy all constraints).
- Return ONLY the JSON solution object.`;

const CROSSOVER_USER = `## Problem
{{PROBLEM}}

## Solution Format
{{SCHEMA}}

## Agent Solutions (sorted by score, best first)

{{CANDIDATES}}

## Your Task
Construct a new solution that combines the strongest independent components from the solutions above.
Identify which parts of each solution contribute the most value, and recombine them into a single solution.
The new solution should be better than any individual solution by exploiting complementary strengths.

Return ONLY the JSON solution:`;

const REFINE_AFTER_CROSSOVER_SYSTEM = `You are an expert optimizer.
You will receive a candidate solution and its score, plus the original problem.
Make targeted local improvements to increase the score while maintaining feasibility.
Return ONLY the improved JSON solution.`;

const REFINE_AFTER_CROSSOVER_USER = `## Problem
{{PROBLEM}}

## Solution Format
{{SCHEMA}}

## Current Solution (score: {{SCORE}})
{{CURRENT}}

Improve this solution with targeted local changes. Keep all feasibility constraints.
Return ONLY the improved JSON solution:`;

// ── Helpers ──────────────────────────────────────────────────────────────────

function parseSolutionJSON(raw: string): Record<string, unknown> | null {
  try { return JSON.parse(raw.trim()); } catch { /* fall through */ }
  const match = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (match) { try { return JSON.parse(match[1].trim()); } catch { /* fall through */ } }
  const braceMatch = raw.match(/\{[\s\S]*\}/);
  if (braceMatch) { try { return JSON.parse(braceMatch[0]); } catch { /* fall through */ } }
  return null;
}

function fill(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, k) => vars[k] ?? '');
}

async function generateInitial(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  maxAttempts = 5,
): Promise<{ data: Record<string, unknown>; score: number; raw: string } | null> {
  const base = {
    PROBLEM: task.prompt,
    SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
    PARAMS: JSON.stringify(task.parameters ?? {}, null, 2),
  };
  let lastError: string | null = null;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (!budget.canCallLLM) return null;

    const userContent = `## Problem\n${task.prompt}\n\n## Solution Format\n${JSON.stringify(task.solutionSchema, null, 2)}\n\nSolve the problem and return ONLY the JSON solution.`;
    const messages: LLMMessage[] = lastError !== null
      ? [
          { role: 'system', content: 'You are an expert problem solver. Return only a JSON solution object.' },
          { role: 'user', content: `${userContent}\n\n# Previous Attempt Failed\n${lastError}\nFix the issue in your next solution.` },
        ]
      : [
          { role: 'system', content: 'You are an expert problem solver. Return only a JSON solution object.' },
          { role: 'user', content: userContent },
        ];

    const response = await budget.llmCall(backbone, messages, { temperature: 0.7 });
    if (!response) return null;

    const data = parseSolutionJSON(response.content);
    if (!data) {
      lastError = 'output was not valid JSON';
      continue;
    }

    if (!budget.canCallEval || !task.challengeId) return null;
    try {
      const evalResult = await budget.evalDev(task.challengeId, data);
      if (evalResult === null) return null;
      return { data, score: evalResult.devScore, raw: response.content };
    } catch (err) {
      lastError = err instanceof Error ? err.message : String(err);
      lastError = lastError.replace(/^Dev eval failed:\s*/, '').replace(/^Verifier error:\s*/, '');
    }
  }
  return null;
}

// ── Main protocol ─────────────────────────────────────────────────────────────

/**
 * Evolutionary Crossover Protocol
 *
 * Phases:
 *   1. Each backbone generates independently (no interaction)
 *   2. Crossover agent recombines best components
 *   3. Optional: local refinement from the crossover solution
 *
 * Returns all artifacts (proposals + crossover + refinements) so MEG/MIG
 * analysis can pick the best or analyze separately.
 */
export async function runEvolutionaryCrossover(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
  opts: {
    doLocalRefine?: boolean;  // whether to run Phase 3 refinement
    crossoverBackbone?: BackboneConfig;  // model for crossover step (default: backbones[0])
  } = {},
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];
  const crossoverBackbone = opts.crossoverBackbone ?? backbones[0];
  const doRefine = opts.doLocalRefine ?? false;

  // ── Phase 1: Independent generation ────────────────────────────────────────
  const proposals: Array<{ data: Record<string, unknown>; score: number; raw: string; model: string }> = [];

  for (const bb of backbones) {
    if (!budget.canCallLLM || !budget.canCallEval) break;

    const result = await generateInitial(task, bb, budget);
    if (!result) {
      console.log(`  [crossover] ${bb.model}: failed all retries`);
      continue;
    }

    proposals.push({ ...result, model: bb.model });
    artifacts.push({ solutionData: result.data, devScore: result.score, rawOutput: result.raw });
    console.log(`  [crossover] ${bb.model}: score=${result.score}`);
  }

  if (proposals.length < 2 || !budget.canCallLLM || !budget.canCallEval) {
    return artifacts;
  }

  // ── Phase 2: Crossover recombination ───────────────────────────────────────
  // Sort proposals best-first so the crossover agent sees the strongest ones first
  const sorted = [...proposals].sort((a, b) =>
    task.scoringDirection === 'maximize' ? b.score - a.score : a.score - b.score
  );

  const candidateSummary = sorted
    .map((p, i) => `### Solution ${i + 1} (model: ${p.model}, score: ${p.score})\n\`\`\`json\n${JSON.stringify(p.data, null, 2)}\n\`\`\``)
    .join('\n\n');

  const crossoverMessages: LLMMessage[] = [
    { role: 'system', content: CROSSOVER_SYSTEM },
    { role: 'user', content: fill(CROSSOVER_USER, {
      PROBLEM: task.prompt,
      SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
      CANDIDATES: candidateSummary,
    })},
  ];

  let crossoverAttempts = 0;
  let crossoverSuccess = false;
  let lastCrossoverError: string | null = null;

  while (crossoverAttempts < 3 && budget.canCallLLM && budget.canCallEval) {
    const messages: LLMMessage[] = lastCrossoverError
      ? crossoverMessages.map((m, i) => i === crossoverMessages.length - 1
          ? { ...m, content: `${m.content}\n\n# Previous Attempt Failed\n${lastCrossoverError}\nFix the issue.` }
          : m)
      : crossoverMessages;

    const response = await budget.llmCall(crossoverBackbone, messages, { temperature: 0.5 });
    crossoverAttempts++;

    if (!response) break;

    const data = parseSolutionJSON(response.content);
    if (!data) {
      lastCrossoverError = 'output was not valid JSON';
      continue;
    }

    try {
      const evalResult = await budget.evalDev(task.challengeId!, data);
      if (evalResult === null) break;

      const crossoverScore = evalResult.devScore;
      artifacts.push({ solutionData: data, devScore: crossoverScore, rawOutput: response.content });
      console.log(`  [crossover] recombination: score=${crossoverScore}`);
      crossoverSuccess = true;

      // ── Phase 3: Optional local refinement from crossover ─────────────────
      if (doRefine && budget.canCallLLM && budget.canCallEval) {
        for (const bb of backbones) {
          if (!budget.canCallLLM || !budget.canCallEval) break;

          const refineMessages: LLMMessage[] = [
            { role: 'system', content: REFINE_AFTER_CROSSOVER_SYSTEM },
            { role: 'user', content: fill(REFINE_AFTER_CROSSOVER_USER, {
              PROBLEM: task.prompt,
              SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
              SCORE: String(crossoverScore),
              CURRENT: JSON.stringify(data, null, 2),
            })},
          ];

          const refineResp = await budget.llmCall(bb, refineMessages, { temperature: 0.5 });
          if (!refineResp) continue;

          const refineData = parseSolutionJSON(refineResp.content);
          if (!refineData) continue;

          try {
            const refineEval = await budget.evalDev(task.challengeId!, refineData);
            if (refineEval === null) continue;
            artifacts.push({ solutionData: refineData, devScore: refineEval.devScore, rawOutput: refineResp.content });
            console.log(`  [crossover-refine] ${bb.model}: score=${refineEval.devScore}`);
          } catch { /* skip on verifier error */ }
        }
      }

      break;
    } catch (err) {
      lastCrossoverError = err instanceof Error ? err.message : String(err);
      lastCrossoverError = lastCrossoverError.replace(/^Dev eval failed:\s*/, '');
    }
  }

  if (!crossoverSuccess) {
    console.log(`  [crossover] recombination failed after ${crossoverAttempts} attempts`);
  }

  return artifacts;
}

/**
 * Evolutionary Crossover (no refinement) — clean 2×2-compatible variant.
 * Phases 1+2 only. Comparable budget to MoA (3 proposals + 1 crossover call).
 */
export async function runCrossoverNoRefine(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
  crossoverBackbone?: BackboneConfig,
): Promise<SolutionArtifact[]> {
  return runEvolutionaryCrossover(task, backbones, budget, {
    doLocalRefine: false,
    crossoverBackbone,
  });
}

/**
 * Evolutionary Crossover with refinement — extended variant.
 * Phases 1+2+3. Higher budget, tests whether crossover+refine beats crossover alone.
 */
export async function runCrossoverWithRefine(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
  crossoverBackbone?: BackboneConfig,
): Promise<SolutionArtifact[]> {
  return runEvolutionaryCrossover(task, backbones, budget, {
    doLocalRefine: true,
    crossoverBackbone,
  });
}
