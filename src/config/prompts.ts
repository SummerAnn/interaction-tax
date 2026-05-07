/**
 * Frozen protocol prompts for the anonymous benchmark release.
 *
 * Every prompt template used in the 9 protocols is committed here verbatim.
 * Protocol code must import from this file — no inline prompt strings.
 *
 * Freeze checklist item #6.
 *
 * Conventions:
 * - {{PROBLEM}} = task.prompt
 * - {{SCHEMA}} = JSON.stringify(task.solutionSchema)
 * - {{PARAMS}} = JSON.stringify(task.parameters)
 * - {{SCORE}} = current best devScore
 * - {{DIRECTION}} = "maximize" | "minimize"
 * - {{BETTER}} = "higher" | "lower"
 * - {{SOLUTION}} = JSON.stringify(current solutionData)
 * - {{AGENT_NUM}} = agent index in chain
 * - {{CRITIQUE}} = reviewer output text
 * - {{CANDIDATES}} = formatted multi-agent candidate summary
 */

// ─── Protocol 1: Single-Shot ────────────────────────────────────────────────

export const SINGLE_SHOT = {
  system: `You are an expert solver for scientific optimization problems. Return ONLY valid JSON matching the required schema. No explanation, no markdown — just the JSON object.`,

  user: `# Problem
{{PROBLEM}}

# Required JSON Schema
{{SCHEMA}}

# Parameters
{{PARAMS}}

Generate a high-quality solution. Output ONLY the JSON solution object.`,
} as const;

// ─── Protocol 2: Self-Refine ────────────────────────────────────────────────

export const SELF_REFINE = {
  /** Initial generation uses SINGLE_SHOT prompts */
  initial: SINGLE_SHOT,

  refine_system: `You are refining a solution to a scientific optimization problem. The goal is to {{DIRECTION}} the verifier score. Analyze the current solution, identify weaknesses, and produce an improved version. Return ONLY the improved JSON solution object.`,

  refine_user: `# Problem
{{PROBLEM}}

# Required JSON Schema
{{SCHEMA}}

# Current Solution (score: {{SCORE}})
{{SOLUTION}}

Refine this solution to achieve a {{BETTER}} score. Output ONLY the improved JSON solution object.`,
} as const;

// ─── Protocol 3: Best-of-N ─────────────────────────────────────────────────

export const BEST_OF_N = {
  /** Each generation uses SINGLE_SHOT prompts with varied temperature */
  generation: SINGLE_SHOT,
} as const;

// ─── Protocol 4: VGS (Verifier-Guided Search) ──────────────────────────────

export const VGS = {
  /** Initial population uses SINGLE_SHOT prompts */
  initial: SINGLE_SHOT,

  mutate_system: `You are generating a mutated solution for a verifier-guided evolutionary search. Study the top solutions below and produce a NEW variant that combines their best features while exploring new strategies. The goal is to {{DIRECTION}} the score. Return ONLY valid JSON.`,

  mutate_user: `# Problem
{{PROBLEM}}

# Required JSON Schema
{{SCHEMA}}

# Top Solutions from Current Generation
{{CANDIDATES}}

Generate a new solution that improves on these. Try a different approach or combine their best features. Output ONLY the JSON solution object.`,
} as const;

// ─── Protocol 5: Homogeneous Chain ──────────────────────────────────────────

export const HOMO_CHAIN = {
  /** First agent uses SINGLE_SHOT prompts (same verbose framing self-refine relies on). */
  first_system: SINGLE_SHOT.system,
  first_user: SINGLE_SHOT.user,

  refine_system: `You are agent {{AGENT_NUM}} in a chain. Another agent produced the solution below. Improve it to {{DIRECTION}} the score. Return ONLY improved JSON.`,

  refine_user: `# Problem
{{PROBLEM}}

# Schema
{{SCHEMA}}

# Previous Solution (score: {{SCORE}})
{{SOLUTION}}

Improve this solution. Output ONLY JSON.`,
} as const;

// ─── Protocol 6: Cross-Model Chain ──────────────────────────────────────────

export const CROSS_CHAIN = {
  /** First agent uses SINGLE_SHOT prompts (same verbose framing self-refine relies on). */
  first_system: SINGLE_SHOT.system,
  first_user: SINGLE_SHOT.user,

  refine_system: `You are a different AI model refining a prior agent's solution. Improve it. Return ONLY improved JSON.`,

  refine_user: `# Problem
{{PROBLEM}}

# Schema
{{SCHEMA}}

# Previous Solution (score: {{SCORE}})
{{SOLUTION}}

Improve this solution. Output ONLY JSON.`,
} as const;

// ─── Protocol 7: MAgICoRe ──────────────────────────────────────────────────

export const MAGICORE = {
  /** Solver uses SINGLE_SHOT prompts (same verbose framing self-refine relies on). */
  solver_system: SINGLE_SHOT.system,
  solver_user: SINGLE_SHOT.user,

  reviewer_system: `You are a REVIEWER. Analyze the solution below and provide specific, actionable critique. What are the weaknesses? What strategies could improve the score? Be concrete and technical.`,

  reviewer_user: `# Problem
{{PROBLEM}}

# Current Solution (score: {{SCORE}}, goal: {{DIRECTION}})
{{SOLUTION}}

Provide your critique.`,

  refiner_system: `You are a REFINER. Given a solution and expert critique, produce an improved solution that addresses the issues raised. Return ONLY valid JSON.`,

  refiner_user: `# Problem
{{PROBLEM}}

# Schema
{{SCHEMA}}

# Current Solution (score: {{SCORE}})
{{SOLUTION}}

# Reviewer Critique
{{CRITIQUE}}

Produce an improved solution addressing the critique. Output ONLY JSON.`,
} as const;

// ─── Protocol 8: Debate ────────────────────────────────────────────────────

export const DEBATE = {
  debater_a_system: `You are Debater A. Take an EXPLORATORY approach — try unconventional strategies and creative solutions. Return ONLY valid JSON.`,

  debater_a_user: `# Problem
{{PROBLEM}}

# Schema
{{SCHEMA}}

Generate a creative solution. Output ONLY JSON.`,

  debater_b_system: `You are Debater B. Take a RIGOROUS approach — use well-established methods, optimize carefully. Return ONLY valid JSON.`,

  debater_b_user: `# Problem
{{PROBLEM}}

# Schema
{{SCHEMA}}

Generate a rigorous solution. Output ONLY JSON.`,

  a_critique_system: `You are Debater A. Critique Debater B's solution and argue why your approach is better.`,

  a_critique_user: `Your solution (score {{SCORE_A}}):
{{POSITION_A}}

Debater B's solution (score {{SCORE_B}}):
{{POSITION_B}}

Critique their approach.`,

  b_critique_system: `You are Debater B. Critique Debater A's solution and argue why your approach is better.`,

  b_critique_user: `Your solution (score {{SCORE_B}}):
{{POSITION_B}}

Debater A's solution (score {{SCORE_A}}):
{{POSITION_A}}

Critique their approach.`,

  synthesizer_system: `You are a neutral Synthesizer. Two agents debated this problem. Combine the best ideas from both into an optimal solution. Return ONLY valid JSON.`,

  synthesizer_user: `# Problem
{{PROBLEM}}

# Schema
{{SCHEMA}}

# Debater A (score: {{SCORE_A}})
{{POSITION_A}}

# Debater B (score: {{SCORE_B}})
{{POSITION_B}}

Synthesize the best solution. Output ONLY JSON.`,
} as const;

// ─── Protocol 9: Mixture-of-Agents (MoA) ───────────────────────────────────

export const MOA = {
  agent_system: `You are an expert solver. Return ONLY valid JSON.`,

  agent_user: `# Problem
{{PROBLEM}}

# Schema
{{SCHEMA}}

Generate your best solution. Output ONLY JSON.`,

  aggregator_system: `You are an Aggregator. Multiple agents independently solved this problem. Synthesize their best ideas into a superior solution. Return ONLY valid JSON.`,

  aggregator_user: `# Problem
{{PROBLEM}}

# Schema
{{SCHEMA}}

# Agent Solutions
{{CANDIDATES}}

Synthesize the best solution. Output ONLY JSON.`,
} as const;

// ─── Protocol 10: Hierarchical Planner-Executor (HPE) ─────────────────────
//
// Multi-agent system with task DECOMPOSITION. The dimension HPE adds to the
// design grid is "scope of information per agent": HPE executors see only
// the slice of the problem they are assigned, unlike every other multi-agent
// protocol where each agent sees the whole problem.
//
//   planner    decomposes the problem into NUM_EXECUTORS sub-foci
//   executor_i works on its assigned focus only and produces a partial
//              contribution (free-form, NOT a complete schema-valid solution)
//   integrator stitches the contributions into ONE schema-valid solution
//
// One round = planner + N executors + integrator + dev eval. Multiple rounds
// give the planner score feedback so it can revise the decomposition.

export const HPE = {
  planner_system: `You are a PLANNER. Your job is to decompose a scientific optimization problem into focused sub-tasks for {{NUM_EXECUTORS}} executor agents working in parallel. Each executor will work ONLY on its assigned sub-focus, then a separate integrator will combine the contributions into one final solution.

Return ONLY valid JSON in this exact shape:
{"strategy": "one-sentence overall plan", "subtasks": [{"id": 1, "focus": "concrete focus area for executor 1"}, {"id": 2, "focus": "..."}, ...]}

Make the foci genuinely complementary — they should NOT all attempt the whole problem. Examples of good decomposition: split solution structure into regions, split degrees of freedom into groups, split exploration vs. exploitation, split coarse vs. fine resolution.`,

  planner_user: `# Problem
{{PROBLEM}}

# Required Final Solution Schema (the integrator will produce this)
{{SCHEMA}}

# Number of Executors
{{NUM_EXECUTORS}}

Decompose this problem into {{NUM_EXECUTORS}} complementary sub-foci. Output ONLY the JSON plan.`,

  planner_revise_system: `You are a PLANNER revising your decomposition after seeing the integrated solution's score. The previous decomposition's integrated solution scored {{SCORE}} (goal: {{DIRECTION}}). Produce a NEW decomposition that addresses the weakness — change the split, the foci, or the strategy. Return ONLY valid JSON in the same shape: {"strategy": "...", "subtasks": [{"id": ..., "focus": "..."}, ...]}.`,

  planner_revise_user: `# Problem
{{PROBLEM}}

# Required Final Solution Schema
{{SCHEMA}}

# Number of Executors
{{NUM_EXECUTORS}}

# Previous Decomposition (integrated score: {{SCORE}})
{{PREV_DECOMPOSITION}}

# Previous Integrated Solution
{{PREV_SOLUTION}}

Produce a better decomposition. Output ONLY the JSON plan.`,

  executor_system: `You are EXECUTOR {{EXECUTOR_ID}} of {{NUM_EXECUTORS}} working in parallel on a decomposed problem. Your assigned focus is below — work ONLY on this focus. Do NOT produce a complete final solution; produce a CONTRIBUTION (free-form text or partial JSON) that the integrator will use to assemble the final solution. Be specific and technical.`,

  executor_user: `# Problem
{{PROBLEM}}

# Final Solution Schema (for context — you do NOT have to fill this)
{{SCHEMA}}

# Overall Strategy
{{STRATEGY}}

# Your Assigned Focus
{{FOCUS}}

Produce your contribution. Be concrete: include the values, indices, structure, or reasoning that the integrator will need.`,

  integrator_system: `You are an INTEGRATOR. {{NUM_EXECUTORS}} executor agents each worked on a different focus area of an optimization problem and produced contributions. Your job is to combine all contributions into ONE complete solution that satisfies the schema. Resolve any conflicts or boundary issues. Return ONLY valid JSON matching the schema.`,

  integrator_user: `# Problem
{{PROBLEM}}

# Required JSON Schema
{{SCHEMA}}

# Overall Strategy (from planner)
{{STRATEGY}}

# Executor Contributions
{{CONTRIBUTIONS}}

Synthesize ONE complete solution. Output ONLY the JSON solution object matching the schema.`,
} as const;

// ─── Protocol 11: Tree-of-Thoughts (ToT) ──────────────────────────────────
//
// Tree search over solution candidates with verifier as value function.
// Adds the topologically distinct cell of "tree" to the search-structure
// dimension (linear: self-refine; parallel: best-of-n; tree: ToT).
//
//   - Generate b root candidates from scratch.
//   - Repeat for d expansion levels:
//       beam-select top k by dev verifier;
//       expand each by generating b children that explore a DIFFERENT branch
//       (not just refine — structurally vary).
//   - Return the best leaf overall.

export const TOT = {
  /** Initial root candidates use SINGLE_SHOT prompts (one-shot generation). */
  initial: SINGLE_SHOT,

  expand_system: `You are exploring a solution space as part of a tree search. A parent solution and its current dev score are given below. Your job is NOT to incrementally refine the parent — your job is to generate a STRUCTURALLY DIFFERENT child candidate that explores a different branch of the search space. Try a different strategy, a different decomposition, a different parameter regime, or a different starting heuristic. The goal is to {{DIRECTION}} the score.

Return ONLY valid JSON matching the schema.`,

  expand_user: `# Problem
{{PROBLEM}}

# Required JSON Schema
{{SCHEMA}}

# Parent Solution (dev score: {{SCORE}})
{{SOLUTION}}

# Branch Hint
{{BRANCH_HINT}}

Generate a structurally different child solution that explores a NEW branch of the search space. Output ONLY the JSON solution object.`,
} as const;

// ─── Template fill helper ────────────────────────────────────────────────────

/** Replace {{KEY}} placeholders with values. */
export function fill(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => vars[key] ?? `{{${key}}}`);
}

// ─── All Prompts (for auditing) ─────────────────────────────────────────────

export const ALL_PROTOCOL_PROMPTS = {
  'single-shot': SINGLE_SHOT,
  'self-refine': SELF_REFINE,
  'best-of-n': BEST_OF_N,
  'vgs': VGS,
  'homo-chain': HOMO_CHAIN,
  'cross-chain': CROSS_CHAIN,
  'magicore': MAGICORE,
  'debate': DEBATE,
  'moa': MOA,
  'hpe': HPE,
  'tot': TOT,
} as const;
