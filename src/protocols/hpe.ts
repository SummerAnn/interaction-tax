/**
 * Hierarchical Planner-Executor (HPE) Protocol
 *
 * Multi-agent system whose distinct dimension is TASK DECOMPOSITION:
 * unlike every other multi-agent protocol in the suite (where each agent
 * sees the whole problem), HPE executors see only the slice of the problem
 * they are assigned by a top-level planner. A separate integrator stitches
 * the partial contributions back into one schema-valid solution.
 *
 *   planner    : 1 LLM call → JSON {strategy, subtasks: [{id, focus}, ...]}
 *   executor_i : 1 LLM call → free-form contribution for its assigned focus
 *   integrator : 1 LLM call → 1 schema-valid solution; eval'd by dev verifier
 *
 * Multiple rounds let the planner revise the decomposition after seeing the
 * integrated solution's score. Each round produces ONE artifact (the
 * integrated solution), so HPE's artifact list has length = rounds.
 *
 * Lit refs: Plan-and-Solve (Wang et al. 2023), MetaGPT (Hong et al. 2023),
 * AutoGen (Wu et al. 2023). HPE is the optimization-task adaptation that
 * keeps the comparison apples-to-apples with the other 9 protocols
 * (same backbone, same dev verifier, same budget vector).
 */

import type { BenchmarkTask, BackboneConfig, SolutionArtifact } from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import type { LLMMessage } from '../lib/llm-client.js';
import { HPE, fill } from '../config/prompts.js';

interface PlannerOutput {
  strategy: string;
  subtasks: Array<{ id: number; focus: string }>;
}

function parseJSON(raw: string): Record<string, unknown> | null {
  try { return JSON.parse(raw.trim()); } catch { /* */ }
  const m = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (m) { try { return JSON.parse(m[1].trim()); } catch { /* */ } }
  const b = raw.match(/\{[\s\S]*\}/);
  if (b) { try { return JSON.parse(b[0]); } catch { /* */ } }
  return null;
}

function parsePlannerOutput(raw: string, expectedN: number): PlannerOutput | null {
  const j = parseJSON(raw);
  if (!j) return null;
  const strategy = typeof j['strategy'] === 'string' ? (j['strategy'] as string) : '';
  const subtasksRaw = j['subtasks'];
  if (!Array.isArray(subtasksRaw) || subtasksRaw.length === 0) return null;
  const subtasks: Array<{ id: number; focus: string }> = [];
  for (let i = 0; i < subtasksRaw.length; i++) {
    const s = subtasksRaw[i] as Record<string, unknown> | undefined;
    if (!s) continue;
    const focus = typeof s['focus'] === 'string' ? (s['focus'] as string) : '';
    if (!focus) continue;
    subtasks.push({ id: i + 1, focus });
  }
  if (subtasks.length === 0) return null;
  // If the planner produced more sub-tasks than requested, truncate; if fewer,
  // accept (the integrator can still combine whatever contributions it gets).
  if (subtasks.length > expectedN) subtasks.length = expectedN;
  return { strategy, subtasks };
}

function normalize(task: BenchmarkTask, data: Record<string, unknown>): Record<string, unknown> {
  return task.normalizeOutput ? task.normalizeOutput(data) : data;
}

async function evalSolution(
  task: BenchmarkTask,
  solutionData: Record<string, unknown>,
  budget: BudgetEnforcer,
): Promise<number | null> {
  if (!budget.canCallEval || !task.challengeId) return null;
  try {
    const r = await budget.evalDev(task.challengeId, solutionData, { protocolId: 'hpe' });
    return r?.devScore ?? null;
  } catch {
    return null;
  }
}

/**
 * Run one HPE round: planner → executors → integrator → eval.
 * Returns the integrated artifact + the planner output (so the next round
 * can revise the decomposition with score feedback).
 *
 * Role-based backbone assignment (when multiple backbones are provided):
 *   planner    = backbones[0]                          (Claude — strategic decomposition)
 *   executor i = backbones[i % backbones.length]       (round-robin across all models)
 *   integrator = backbones[0]                          (Claude — schema-valid synthesis)
 *
 * With a single backbone the behavior collapses to the original homogeneous
 * version. With backbones=[claude, gpt4o, gemini] and numExecutors=3, every
 * executor runs on a different model, exposing genuine "different brains
 * solve different sub-problems" behavior to the integrator.
 */
async function runHPERound(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
  numExecutors: number,
  prevPlan: PlannerOutput | null,
  prevSolution: Record<string, unknown> | null,
  prevScore: number | null,
): Promise<{ artifact: SolutionArtifact; plan: PlannerOutput } | null> {
  const planner    = backbones[0];
  const integrator = backbones[0];

  const base = {
    PROBLEM: task.prompt,
    SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
    NUM_EXECUTORS: String(numExecutors),
  };

  // 1. Planner
  if (!budget.canCallLLM) return null;
  const isRevision = prevPlan !== null && prevSolution !== null && prevScore !== null;
  const plannerVars = isRevision
    ? {
        ...base,
        SCORE: String(prevScore),
        DIRECTION: task.scoringDirection,
        PREV_DECOMPOSITION: JSON.stringify(prevPlan, null, 2),
        PREV_SOLUTION: JSON.stringify(prevSolution, null, 2),
      }
    : base;
  const plannerMessages: LLMMessage[] = isRevision
    ? [
        { role: 'system', content: fill(HPE.planner_revise_system, plannerVars) },
        { role: 'user',   content: fill(HPE.planner_revise_user,   plannerVars) },
      ]
    : [
        { role: 'system', content: fill(HPE.planner_system, plannerVars) },
        { role: 'user',   content: fill(HPE.planner_user,   plannerVars) },
      ];

  const plannerResp = await budget.llmCall(planner, plannerMessages, {
    temperature: 0.5,
    maxTokens: 1024,
  });
  if (!plannerResp) return null;

  const plan = parsePlannerOutput(plannerResp.content, numExecutors);
  if (!plan) return null;

  // 2. Executors (parallel in spec; sequential here to keep budget enforcement
  //    deterministic and to interleave token-cap checks). Each executor is
  //    round-robin assigned to a backbone, so with [claude, gpt4o, gemini]
  //    and numExecutors=3 every executor runs on a different model.
  const contributions: string[] = [];
  for (const sub of plan.subtasks) {
    if (!budget.canCallLLM) break;
    const execBackbone = backbones[(sub.id - 1) % backbones.length];
    const execVars = {
      ...base,
      EXECUTOR_ID: String(sub.id),
      STRATEGY: plan.strategy,
      FOCUS: sub.focus,
    };
    const execResp = await budget.llmCall(execBackbone, [
      { role: 'system', content: fill(HPE.executor_system, execVars) },
      { role: 'user',   content: fill(HPE.executor_user,   execVars) },
    ], { temperature: 0.7, maxTokens: 2048 });
    if (!execResp) continue;
    contributions.push(`## Executor ${sub.id} (focus: ${sub.focus})\n${execResp.content}`);
  }
  if (contributions.length === 0) return null;

  // 3. Integrator
  if (!budget.canCallLLM) return null;
  const intVars = {
    ...base,
    STRATEGY: plan.strategy,
    CONTRIBUTIONS: contributions.join('\n\n'),
  };
  const intResp = await budget.llmCall(integrator, [
    { role: 'system', content: fill(HPE.integrator_system, intVars) },
    { role: 'user',   content: fill(HPE.integrator_user,   intVars) },
  ], { temperature: 0.5, maxTokens: 4096 });
  if (!intResp) return null;

  let solutionData = parseJSON(intResp.content);
  if (!solutionData) return null;
  solutionData = normalize(task, solutionData);

  // 4. Dev eval
  const devScore = await evalSolution(task, solutionData, budget);
  if (devScore === null) return null;

  return {
    artifact: { solutionData, devScore, rawOutput: intResp.content },
    plan,
  };
}

/**
 * Run the full Hierarchical Planner-Executor protocol.
 *
 * @param numExecutors  Number of parallel executor agents per round.
 * @param rounds        Number of plan→execute→integrate→eval cycles.
 *                      Round 2+ feeds the previous integrated score back to
 *                      the planner so it can revise the decomposition.
 */
export async function runHPE(
  task: BenchmarkTask,
  backbones: BackboneConfig[],
  budget: BudgetEnforcer,
  numExecutors = 3,
  rounds = 2,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];
  if (backbones.length === 0) return artifacts;

  let prevPlan: PlannerOutput | null = null;
  let prevSolution: Record<string, unknown> | null = null;
  let prevScore: number | null = null;

  for (let r = 0; r < rounds; r++) {
    if (!budget.canCallLLM || !budget.canCallEval) break;

    const result = await runHPERound(
      task, backbones, budget, numExecutors,
      prevPlan, prevSolution, prevScore,
    );
    if (!result) break;

    artifacts.push(result.artifact);
    prevPlan = result.plan;
    prevSolution = result.artifact.solutionData;
    prevScore = result.artifact.devScore;
  }

  return artifacts;
}
