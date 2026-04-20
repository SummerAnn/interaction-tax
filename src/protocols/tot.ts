/**
 * Tree-of-Thoughts (ToT) Protocol — Yao et al. 2023
 *
 * Tree search over solution candidates with the platform dev verifier as
 * the value function. ToT adds the topologically distinct cell of "tree"
 * to the search-structure dimension of the protocol grid:
 *
 *   linear refinement   → self-refine, homo-chain
 *   parallel sampling   → best-of-n, MoA
 *   evolutionary pop    → VGS
 *   tree (this)         → ToT
 *
 * Algorithm:
 *   1. Generate `branching` root candidates from scratch.
 *   2. Evaluate each with the dev verifier.
 *   3. For depth = 1..maxDepth:
 *        a. Beam-select top `beamWidth` candidates by dev score.
 *        b. Expand each: generate `branching` children that explore a
 *           STRUCTURALLY DIFFERENT branch (not refine — branch).
 *        c. Evaluate each child.
 *   4. Return all visited candidates as artifacts; the cell's bestArtifact
 *      is the highest-scoring leaf.
 *
 * Default budget: depth=2, branching=3, beamWidth=2 → 3 + 6 + 6 = 15
 * LLM calls and 15 dev eval calls per cell, fitting under K=25.
 *
 * NOTE: Each child is generated INDEPENDENTLY conditioned on its parent —
 * different from Self-Refine (which keeps refining the same chain) and
 * different from VGS (which mutates from the population centroid). The
 * branch hint is sampled from a small set of strategies to encourage
 * diversity across siblings.
 */

import type { BenchmarkTask, BackboneConfig, SolutionArtifact } from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import { SINGLE_SHOT, TOT, fill } from '../config/prompts.js';

interface Node {
  solutionData: Record<string, unknown>;
  rawOutput: string;
  devScore: number;
  depth: number;
  parentIdx: number | null;
}

const BRANCH_HINTS: readonly string[] = [
  'try a fundamentally different decomposition or partitioning',
  'try a heuristic from a different family (e.g., random restart, greedy, gradient, simulated annealing)',
  'invert the current solution\'s primary structural choice',
  'apply a coarse-to-fine resolution change',
  'shift parameters toward an unexplored region of the search space',
];

function parseJSON(raw: string): Record<string, unknown> | null {
  try { return JSON.parse(raw.trim()); } catch { /* */ }
  const m = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (m) { try { return JSON.parse(m[1].trim()); } catch { /* */ } }
  const b = raw.match(/\{[\s\S]*\}/);
  if (b) { try { return JSON.parse(b[0]); } catch { /* */ } }
  return null;
}

function normalize(task: BenchmarkTask, data: Record<string, unknown>): Record<string, unknown> {
  return task.normalizeOutput ? task.normalizeOutput(data) : data;
}

async function evalNode(
  task: BenchmarkTask,
  data: Record<string, unknown>,
  budget: BudgetEnforcer,
): Promise<number | null> {
  if (!budget.canCallEval || !task.challengeId) return null;
  try {
    const r = await budget.evalDev(task.challengeId, data, { protocolId: 'tot' });
    return r?.devScore ?? null;
  } catch {
    return null;
  }
}

async function generateRoot(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  rootIdx: number,
): Promise<Node | null> {
  if (!budget.canCallLLM) return null;
  const vars = {
    PROBLEM: task.prompt,
    SCHEMA:  JSON.stringify(task.solutionSchema, null, 2),
    PARAMS:  JSON.stringify(task.parameters ?? {}, null, 2),
  };
  // Vary temperature across roots so the trees start from genuinely
  // different seeds rather than near-duplicates.
  const resp = await budget.llmCall(backbone, [
    { role: 'system', content: SINGLE_SHOT.system },
    { role: 'user',   content: fill(SINGLE_SHOT.user, vars) },
  ], { temperature: 0.7 + rootIdx * 0.1, maxTokens: 4096 });
  if (!resp) return null;

  let data = parseJSON(resp.content);
  if (!data) return null;
  data = normalize(task, data);

  const score = await evalNode(task, data, budget);
  if (score === null) return null;
  return { solutionData: data, rawOutput: resp.content, devScore: score, depth: 0, parentIdx: null };
}

async function expandNode(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  parent: Node,
  parentIdx: number,
  childIdx: number,
): Promise<Node | null> {
  if (!budget.canCallLLM) return null;

  const expandVars = {
    PROBLEM: task.prompt,
    SCHEMA: JSON.stringify(task.solutionSchema, null, 2),
    SCORE: String(parent.devScore),
    SOLUTION: JSON.stringify(parent.solutionData, null, 2),
    DIRECTION: task.scoringDirection,
    BRANCH_HINT: BRANCH_HINTS[childIdx % BRANCH_HINTS.length],
  };
  const resp = await budget.llmCall(backbone, [
    { role: 'system', content: fill(TOT.expand_system, expandVars) },
    { role: 'user',   content: fill(TOT.expand_user,   expandVars) },
  ], { temperature: 0.8, maxTokens: 4096 });
  if (!resp) return null;

  let data = parseJSON(resp.content);
  if (!data) return null;
  data = normalize(task, data);

  const score = await evalNode(task, data, budget);
  if (score === null) return null;
  return {
    solutionData: data,
    rawOutput: resp.content,
    devScore: score,
    depth: parent.depth + 1,
    parentIdx,
  };
}

function isBetter(a: number, b: number, dir: 'maximize' | 'minimize'): boolean {
  return dir === 'maximize' ? a > b : a < b;
}

/**
 * Run the full Tree-of-Thoughts protocol.
 *
 * @param maxDepth   Number of expansion levels after the root layer.
 * @param branching  Children generated per expanded parent (also = number of roots).
 * @param beamWidth  How many top nodes from each level survive to the next.
 */
export async function runToT(
  task: BenchmarkTask,
  backbone: BackboneConfig,
  budget: BudgetEnforcer,
  maxDepth = 2,
  branching = 3,
  beamWidth = 2,
): Promise<SolutionArtifact[]> {
  const all: Node[] = [];

  // Level 0: roots
  for (let i = 0; i < branching; i++) {
    if (!budget.canCallLLM || !budget.canCallEval) break;
    const root = await generateRoot(task, backbone, budget, i);
    if (root) all.push(root);
  }
  if (all.length === 0) return [];

  // Levels 1..maxDepth: beam-expand
  let frontier: Array<{ node: Node; idx: number }> = all.map((n, i) => ({ node: n, idx: i }));

  for (let depth = 1; depth <= maxDepth; depth++) {
    if (!budget.canCallLLM || !budget.canCallEval) break;

    // Beam select top `beamWidth` from current frontier
    const sorted = [...frontier].sort((a, b) =>
      task.scoringDirection === 'maximize'
        ? b.node.devScore - a.node.devScore
        : a.node.devScore - b.node.devScore,
    );
    const beam = sorted.slice(0, beamWidth);

    const newFrontier: Array<{ node: Node; idx: number }> = [];
    for (const parent of beam) {
      for (let c = 0; c < branching; c++) {
        if (!budget.canCallLLM || !budget.canCallEval) break;
        const child = await expandNode(
          task, backbone, budget, parent.node, parent.idx, c,
        );
        if (!child) continue;
        const newIdx = all.length;
        all.push(child);
        newFrontier.push({ node: child, idx: newIdx });
      }
    }

    if (newFrontier.length === 0) break;
    frontier = newFrontier;
  }

  // Return all visited nodes as artifacts. The runner picks the best leaf
  // via getBestArtifact (same as VGS).
  return all.map(n => ({
    solutionData: n.solutionData,
    devScore: n.devScore,
    rawOutput: n.rawOutput,
  }));
}

// Re-export the comparator just in case downstream code wants it.
export { isBetter as totIsBetter };
