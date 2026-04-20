/**
 * Baseline Protocols: Single-Shot, Self-Refine, Best-of-N
 *
 * These are the MEG denominator baselines. They must be genuinely strong
 * for the benchmark claim to be credible.
 */

import type { BenchmarkTask, BackboneConfig, SolutionArtifact } from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import type { LLMMessage } from '../lib/llm-client.js';
import { SINGLE_SHOT, SELF_REFINE, fill } from '../config/prompts.js';

// ── Shared helpers ──

function parseSolutionJSON(raw: string): Record<string, unknown> | null {
  try {
    return JSON.parse(raw.trim());
  } catch {
    const match = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
    if (match) {
      try { return JSON.parse(match[1].trim()); } catch { /* fall through */ }
    }
    const braceMatch = raw.match(/\{[\s\S]*\}/);
    if (braceMatch) {
      try { return JSON.parse(braceMatch[0]); } catch { /* fall through */ }
    }
    return null;
  }
}

function solvePrompt(task: BenchmarkTask): LLMMessage[] {
  const vars = {
    PROBLEM: task.prompt,
    SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
    PARAMS: JSON.stringify(task.parameters ?? {}, null, 2),
  };
  return [
    { role: 'system', content: SINGLE_SHOT.system },
    { role: 'user',   content: fill(SINGLE_SHOT.user, vars) },
  ];
}

async function generateAndEval(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  messages: LLMMessage[],
  temperature = 0.7,
  maxAttempts = 3,
): Promise<SolutionArtifact | null> {
  // Retry on parse failure or verifier rejection. LLMs occasionally produce
  // malformed solutions (e.g., maxcut with 199 labels instead of 200) and
  // without a retry the entire cell silently skips. On retries we append
  // the verifier error message to the user prompt so the LLM can
  // self-correct (this dramatically improves recovery rate for tasks with
  // strict shape constraints).
  let lastError: string | null = null;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (!budget.canCallLLM) return null;

    const attemptMessages: LLMMessage[] = lastError !== null
      ? messages.map((m, i) => i === messages.length - 1
          ? { ...m, content: `${m.content}\n\n# Previous Attempt Failed\nThe verifier rejected your previous solution: ${lastError}\nProduce a NEW solution that fixes this issue.` }
          : m)
      : messages;

    const response = await budget.llmCall(backbone, attemptMessages, { temperature, maxTokens: 4096 });
    if (!response) return null;

    let solutionData = parseSolutionJSON(response.content);
    if (!solutionData) {
      lastError = 'output was not valid JSON matching the schema';
      continue;
    }
    if (task.normalizeOutput) solutionData = task.normalizeOutput(solutionData);

    if (!budget.canCallEval || !task.challengeId) {
      return { solutionData, devScore: 0, rawOutput: response.content };
    }

    try {
      const evalResult = await budget.evalDev(task.challengeId, solutionData);
      if (!evalResult) return { solutionData, devScore: 0, rawOutput: response.content };
      return { solutionData, devScore: evalResult.devScore, rawOutput: response.content };
    } catch (err) {
      // Verifier rejected — feed the error back to the next attempt
      lastError = err instanceof Error ? err.message : String(err);
      lastError = lastError.replace(/^Dev eval failed:\s*/, '').replace(/^Verifier error:\s*/, '');
      continue;
    }
  }
  return null;
}

// ── Single-Shot ──

export async function runSingleShot(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
): Promise<SolutionArtifact[]> {
  const artifact = await generateAndEval(task, backbone, budget, solvePrompt(task));
  return artifact ? [artifact] : [];
}

// ── Best-of-N ──

export async function runBestOfN(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  n = 8,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];

  for (let i = 0; i < n; i++) {
    if (!budget.canCallLLM || !budget.canCallEval) break;
    // Vary temperature for diversity
    const temp = 0.6 + (i * 0.05);
    const artifact = await generateAndEval(task, backbone, budget, solvePrompt(task), temp);
    if (artifact) artifacts.push(artifact);
  }

  return artifacts;
}

// ── Self-Refine ──

export async function runSelfRefine(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  rounds = 3,
): Promise<SolutionArtifact[]> {
  const artifacts: SolutionArtifact[] = [];

  // Step 1: Initial generation + eval
  const initial = await generateAndEval(task, backbone, budget, solvePrompt(task));
  if (!initial) return artifacts;
  artifacts.push(initial);

  let current = initial;

  // Steps 2-N: Refine based on verifier feedback
  for (let r = 0; r < rounds; r++) {
    if (!budget.canCallLLM || !budget.canCallEval) break;

    const better = task.scoringDirection === 'maximize' ? 'higher' : 'lower';
    const refineVars = {
      PROBLEM:   task.prompt,
      SCHEMA:    JSON.stringify(task.solutionSchema, null, 2),
      SCORE:     String(current.devScore),
      SOLUTION:  JSON.stringify(current.solutionData, null, 2),
      DIRECTION: task.scoringDirection,
      BETTER:    better,
    };
    const refineMessages: LLMMessage[] = [
      { role: 'system', content: fill(SELF_REFINE.refine_system, refineVars) },
      { role: 'user',   content: fill(SELF_REFINE.refine_user,   refineVars) },
    ];

    const refined = await generateAndEval(task, backbone, budget, refineMessages, 0.5);
    if (!refined) break;

    artifacts.push(refined);

    // Keep the better solution
    const improved = task.scoringDirection === 'maximize'
      ? refined.devScore > current.devScore
      : refined.devScore < current.devScore;
    if (improved) {
      current = refined;
    }
  }

  return artifacts;
}
