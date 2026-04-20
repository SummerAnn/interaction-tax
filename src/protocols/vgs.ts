/**
 * Verifier-Guided Search (VGS) Protocol
 *
 * FunSearch-style evolutionary loop over solution ARTIFACTS (not programs).
 * This is the strongest single-agent control in the MEG denominator.
 *
 * Algorithm:
 *   1. Initialize population of `populationSize` candidates via independent LLM calls
 *   2. Evaluate each candidate with the dev verifier
 *   3. For each generation:
 *      a. Tournament selection: keep top `eliteCount` candidates
 *      b. Mutation: re-prompt with best-so-far + verifier feedback to generate new candidates
 *      c. Evaluate new candidates
 *      d. Merge elites + new candidates, select top `populationSize`
 *   4. Terminate on: budget exhaustion, or convergence (no improvement for `convergenceWindow` generations)
 *
 * Per the paper spec: pop=4, gen=5, eliteCount=2, convergenceWindow=2
 * Total eval calls: populationSize * generations = 20 (within K=25 cap)
 */

import type {
  BenchmarkTask,
  BackboneConfig,
  VGSConfig,
  SolutionArtifact,
} from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import type { LLMMessage } from '../lib/llm-client.js';
import { SINGLE_SHOT, VGS as VGS_PROMPTS, fill } from '../config/prompts.js';

function normalize(task: BenchmarkTask, data: Record<string, unknown>): Record<string, unknown> {
  return task.normalizeOutput ? task.normalizeOutput(data) : data;
}

interface Candidate {
  solutionData: Record<string, unknown>;
  rawOutput: string;
  devScore: number;
  generation: number;
}

/**
 * Generate the initial population via independent LLM calls.
 */
async function initializePopulation(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  popSize: number
): Promise<Candidate[]> {
  const candidates: Candidate[] = [];

  for (let i = 0; i < popSize; i++) {
    if (!budget.canCallLLM) break;

    const vars = {
      PROBLEM: task.prompt,
      SCHEMA:  JSON.stringify(task.solutionSchema, null, 2),
      PARAMS:  JSON.stringify(task.parameters ?? {}, null, 2),
    };
    const messages: LLMMessage[] = [
      { role: 'system', content: SINGLE_SHOT.system },
      { role: 'user',   content: fill(SINGLE_SHOT.user, vars) },
    ];

    const response = await budget.llmCall(backbone, messages, {
      temperature: 0.8 + (i * 0.1), // slight diversity across population
      maxTokens: 4096,
    });

    if (!response) break;

    let solutionData = parseSolutionJSON(response.content);
    if (!solutionData) continue;
    solutionData = normalize(task, solutionData);

    // Evaluate via platform dev verifier
    if (!budget.canCallEval || !task.challengeId) break;
    try {
      const evalResult = await budget.evalDev(task.challengeId, solutionData, { protocolId: 'vgs' });
      if (!evalResult) break;
      candidates.push({
        solutionData,
        rawOutput: response.content,
        devScore: evalResult.devScore,
        generation: 0,
      });
    } catch {
      // Invalid solution, skip
    }
  }

  return candidates;
}

/**
 * Mutate: re-prompt with best-so-far solution + verifier feedback.
 */
async function mutateCandidate(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  bestSoFar: Candidate,
  population: Candidate[],
  generation: number
): Promise<Candidate | null> {
  if (!budget.canCallLLM) return null;

  // Build context from top candidates
  const topK = population
    .sort((a, b) => compareCandidates(a, b, task.scoringDirection))
    .slice(0, 3);

  const candidateSummary = topK
    .map((c, i) => `## Candidate ${i + 1} (score: ${c.devScore})\n${JSON.stringify(c.solutionData, null, 2)}`)
    .join('\n\n');

  const mutateVars = {
    PROBLEM:    task.prompt,
    SCHEMA:     JSON.stringify(task.solutionSchema, null, 2),
    DIRECTION:  task.scoringDirection,
    CANDIDATES: candidateSummary,
  };
  const messages: LLMMessage[] = [
    { role: 'system', content: fill(VGS_PROMPTS.mutate_system, mutateVars) },
    { role: 'user',   content: fill(VGS_PROMPTS.mutate_user,   mutateVars) },
  ];

  const response = await budget.llmCall(backbone, messages, {
    temperature: 0.7,
    maxTokens: 4096,
  });

  if (!response) return null;

  let solutionData = parseSolutionJSON(response.content);
  if (!solutionData) return null;
  solutionData = normalize(task, solutionData);

  if (!budget.canCallEval || !task.challengeId) return null;
  try {
    const evalResult = await budget.evalDev(task.challengeId, solutionData, { protocolId: 'vgs' });
    if (!evalResult) return null;
    return {
      solutionData,
      rawOutput: response.content,
      devScore: evalResult.devScore,
      generation,
    };
  } catch {
    return null;
  }
}

/**
 * Run the full VGS evolutionary search.
 */
export async function runVGS(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  config: VGSConfig = {
    populationSize: 4,
    generations: 5,
    eliteCount: 2,
    convergenceWindow: 2,
  }
): Promise<SolutionArtifact[]> {
  const allArtifacts: SolutionArtifact[] = [];

  // Step 1: Initialize population
  let population = await initializePopulation(task, backbone, budget, config.populationSize);

  // Record initial population as artifacts
  for (const c of population) {
    allArtifacts.push({
      solutionData: c.solutionData,
      devScore: c.devScore,
      rawOutput: c.rawOutput,
    });
  }

  if (population.length === 0) {
    return allArtifacts;
  }

  // Track convergence
  let bestScore = getBestScore(population, task.scoringDirection);
  let noImprovementCount = 0;

  // Step 2: Evolutionary loop
  for (let gen = 1; gen <= config.generations; gen++) {
    if (!budget.hasBudget) break;
    if (noImprovementCount >= config.convergenceWindow) break;

    // Tournament selection: keep top eliteCount
    population.sort((a, b) => compareCandidates(a, b, task.scoringDirection));
    const elites = population.slice(0, config.eliteCount);

    // Mutation: generate new candidates from best-so-far
    const bestCandidate = elites[0];
    const newCandidates: Candidate[] = [];
    const slotsToFill = config.populationSize - config.eliteCount;

    for (let i = 0; i < slotsToFill; i++) {
      if (!budget.hasBudget) break;

      const mutant = await mutateCandidate(
        task,
        backbone,
        budget,
        bestCandidate,
        population,
        gen
      );

      if (mutant) {
        newCandidates.push(mutant);
        allArtifacts.push({
          solutionData: mutant.solutionData,
          devScore: mutant.devScore,
          rawOutput: mutant.rawOutput,
        });
      }
    }

    // Merge elites + new candidates
    population = [...elites, ...newCandidates];

    // Check convergence
    const currentBest = getBestScore(population, task.scoringDirection);
    if (isBetter(currentBest, bestScore, task.scoringDirection)) {
      bestScore = currentBest;
      noImprovementCount = 0;
    } else {
      noImprovementCount++;
    }
  }

  return allArtifacts;
}

// ── Helpers ──

function parseSolutionJSON(raw: string): Record<string, unknown> | null {
  try {
    // Try direct parse
    return JSON.parse(raw.trim());
  } catch {
    // Try extracting JSON from markdown code block
    const match = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
    if (match) {
      try {
        return JSON.parse(match[1].trim());
      } catch {
        return null;
      }
    }
    // Try finding first { ... } block
    const braceMatch = raw.match(/\{[\s\S]*\}/);
    if (braceMatch) {
      try {
        return JSON.parse(braceMatch[0]);
      } catch {
        return null;
      }
    }
    return null;
  }
}

function getBestScore(population: Candidate[], direction: 'maximize' | 'minimize'): number {
  if (population.length === 0) return direction === 'maximize' ? -Infinity : Infinity;
  return direction === 'maximize'
    ? Math.max(...population.map(c => c.devScore))
    : Math.min(...population.map(c => c.devScore));
}

function isBetter(a: number, b: number, direction: 'maximize' | 'minimize'): boolean {
  return direction === 'maximize' ? a > b : a < b;
}

function compareCandidates(a: Candidate, b: Candidate, direction: 'maximize' | 'minimize'): number {
  return direction === 'maximize' ? b.devScore - a.devScore : a.devScore - b.devScore;
}
