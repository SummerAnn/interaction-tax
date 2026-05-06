/**
 * Experiment Runner
 *
 * Orchestrates the full benchmark: tasks × protocols × seeds.
 * Produces RunResult[] with full budget traces, then computes MEG.
 *
 * Usage:
 *   npx tsx src/runner/experiment-runner.ts --config config/pilot.json
 *   npx tsx src/runner/experiment-runner.ts --pilot
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from 'fs';
import { join } from 'path';
import type {
  ExperimentConfig,
  ExperimentResults,
  RunResult,
  MEGResult,
  ProtocolId,
  BenchmarkTask,
  ProtocolConfig,
  SolutionArtifact,
} from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import { PlatformClient } from '../lib/platform-client.js';
import { runPythonVerifier } from '../lib/local-evaluator.js';
import { PROTOCOL_RUNNERS, getBestArtifact } from '../protocols/index.js';
import { normalizeQ, NORMALIZATION_ANCHORS } from '../config/normalization-anchors.js';

// Protocols where every artifact in the chain is submitted with improvesUpon links.
// HPE qualifies because each round emits ONE integrated artifact and round N+1's
// planner is fed the previous round's solution + score — a linear refinement at
// the artifact level. ToT does NOT qualify: its artifacts form a tree (each child
// has parentIdx) and the linear improvesUpon submission shape can't represent that
// without dropping siblings, so ToT only submits its best leaf (default branch).
const CHAIN_PROTOCOLS = new Set<ProtocolId>([
  'self-refine', 'homo-chain', 'cross-chain', 'magicore', 'hpe',
  'magicore-mixed', 'hpe-mixed',
  'magicore-mixed-oss', 'hpe-mixed-oss',
  'magicore-deepseek', 'hpe-deepseek',
  // GPT-4o solo chain protocols
  'self-refine-gpt4o', 'homo-chain-gpt4o', 'magicore-gpt4o', 'hpe-gpt4o',
  // Gemini solo chain protocols
  'self-refine-gemini', 'homo-chain-gemini', 'magicore-gemini', 'hpe-gemini',
]);

function shortModel(model: string): string {
  // 'anthropic/claude-sonnet-4' → 'claude-sonnet-4'
  return model.split('/').pop() ?? model;
}

/**
 * Build a human-readable label for the set of models a protocol actually
 * uses. For single-model protocols this is just the short model name; for
 * mixed-model protocols (cross-chain, MoA, mixed-brain Debate / MAgICoRe /
 * HPE) it joins the unique models with '+'. Without this, every submission
 * title says "(claude-sonnet-4)" even when the run actually rotated through
 * three different backbones — masking the experimental design in the UI.
 */
function modelsLabel(backbones: import('../types.js').BackboneConfig[]): string {
  const seen = new Set<string>();
  const ordered: string[] = [];
  for (const b of backbones) {
    const m = shortModel(b.model);
    if (!seen.has(m)) { seen.add(m); ordered.push(m); }
  }
  return ordered.join('+');
}

// ── Submission Logic ──

/**
 * Submit artifacts to the platform.
 * - Chain protocols (self-refine, homo-chain, cross-chain, magicore): submit every
 *   artifact in order with improvesUpon links so the UI shows a revision thread.
 * - All other protocols: submit only the best artifact.
 * Returns the submission ID of the best-scoring artifact.
 */
async function submitArtifacts(
  task: BenchmarkTask,
  protocol: ProtocolConfig,
  seed: number,
  runIndex: number,
  allArtifacts: SolutionArtifact[],
  config: ExperimentConfig,
  platform: PlatformClient,
  budgetTrace: import('../types.js').BudgetTrace,
): Promise<string | undefined> {
  if (!task.challengeId) return undefined;

  const allModels = modelsLabel(protocol.backbones) || shortModel(protocol.backbones[0]?.model ?? 'unknown');
  const budgetLabel = `T=${config.budget.tokenCap} W=${config.budget.wallClockSeconds}s`;

  // Build scoreTrace from eval call history (one entry per dev verifier call)
  const scoreTrace = budgetTrace.evalCalls.map((e, i) => ({ step: i, score: e.score }));
  const tokenUsage = budgetTrace.tokenUsage;

  const isChain = CHAIN_PROTOCOLS.has(protocol.id);
  // Every artifact that reaches submitArtifacts has already been successfully
  // parsed by its protocol runner (parse failures return null and are dropped
  // upstream) and scored by the platform dev verifier. devScore=0 is a
  // LEGITIMATE score (e.g., MaxCut all-same-side partition) and negative
  // scores are legitimate on minimize tasks (LJ-n41 energy, HP-folding
  // -contacts). The earlier `devScore > 0` filter silently truncated chains on
  // MaxCut and would have collapsed every LJ/HP chain to a single submission.
  const validArtifacts = allArtifacts;

  // Resolve per-protocol agent ID: check protocolAgentMap by exact match or longest prefix match.
  const resolvedAgentId = (() => {
    const map = config.protocolAgentMap;
    if (!map) return config.agentId;
    if (map[protocol.id]) return map[protocol.id];
    // Longest matching prefix (e.g. 'magicore-mixed-oss' matches 'magicore-mixed-oss' key)
    const match = Object.keys(map)
      .filter(k => protocol.id.startsWith(k))
      .sort((a, b) => b.length - a.length)[0];
    return match ? map[match] : config.agentId;
  })();

  if (!isChain || validArtifacts.length <= 1) {
    // Submit only the best artifact
    const best = getBestArtifact(allArtifacts, task.scoringDirection);
    if (!best) return undefined;
    const result = await platform.evalFinal(task.challengeId, best.solutionData, {
      protocolId: protocol.id,
      runIndex,
      seed,
      title: `${protocol.name} (${allModels}) seed=${seed}`,
      approach: `Protocol: ${protocol.id}, Models: ${allModels}, Budget: ${budgetLabel}`,
      agentId: resolvedAgentId,
      scoreTrace,
      tokenUsage,
    });
    return result.submissionId;
  }

  // Chain: submit each artifact in order, each improvesUpon the previous
  let prevId: string | undefined;
  let bestId: string | undefined;
  let bestScore = task.scoringDirection === 'maximize' ? -Infinity : Infinity;

  for (let i = 0; i < validArtifacts.length; i++) {
    const artifact = validArtifacts[i];
    const round = `round ${i + 1}/${validArtifacts.length}`;
    const isBestRound = i === validArtifacts.length - 1; // attach full trace only to last
    const result = await platform.evalFinal(task.challengeId, artifact.solutionData, {
      protocolId: protocol.id,
      runIndex,
      seed,
      title: `${protocol.name} ${round} (${allModels}) seed=${seed}`,
      approach: `Protocol: ${protocol.id}, ${round}, Models: ${allModels}, Budget: ${budgetLabel}`,
      agentId: resolvedAgentId,
      improvesUpon: prevId,
      scoreTrace: isBestRound ? scoreTrace : undefined,
      tokenUsage: isBestRound ? tokenUsage : undefined,
    });
    prevId = result.submissionId;
    const isBetter = task.scoringDirection === 'maximize'
      ? artifact.devScore > bestScore
      : artifact.devScore < bestScore;
    if (isBetter) { bestScore = artifact.devScore; bestId = result.submissionId; }
  }

  return bestId;
}

// ── Single Cell ──

async function runCell(
  task: BenchmarkTask,
  protocol: ProtocolConfig,
  seed: number,
  runIndex: number,
  config: ExperimentConfig,
  platform: PlatformClient,
): Promise<RunResult | null> {
  const budget = new BudgetEnforcer(config.budget, platform);

  console.log(`  [${task.id}] × [${protocol.id}] seed=${seed} run=${runIndex}...`);

  const runner = PROTOCOL_RUNNERS[protocol.id];
  if (!runner) {
    throw new Error(`Unknown protocol: ${protocol.id}`);
  }

  let allArtifacts: SolutionArtifact[] = [];
  let runnerError: unknown = null;
  try {
    allArtifacts = await runner(task, protocol, budget);
  } catch (err) {
    runnerError = err;
    console.error(`  ERROR in ${task.id}×${protocol.id}: ${err}`);
  }

  const best = getBestArtifact(allArtifacts, task.scoringDirection);
  const trace = budget.getTrace();

  // Drop the cell entirely if the runner threw OR produced no valid artifact.
  // Writing an empty placeholder JSON would silently corrupt later aggregation
  // (this is exactly what produced results/full-v2's all-zeros meg_table).
  if (runnerError !== null || best === null) {
    console.log(`  → SKIP (no valid artifact); tokens=${trace.tokenUsage.total} wall=${Math.round(trace.wallClockMs / 1000)}s`);
    return null;
  }

  // Submit artifact(s) to platform for hidden evaluation. The hidden score
  // is intentionally not retrieved here — runs stay hidden-blind. Use the
  // offline analysis script with platform.getSubmission() for offline analysis.
  let submissionId: string | undefined;
  if (task.challengeId && config.submitForHiddenEval) {
    try {
      submissionId = await submitArtifacts(task, protocol, seed, runIndex, allArtifacts, config, platform, trace);
      if (submissionId) console.log(`  → submitted ${submissionId} for hidden eval`);
    } catch (err) {
      console.error(`  → final submission failed: ${err}`);
    }
  }

  // Offline hidden eval: if no platform submission and the task has a local
  // hidden verifier, run it now on the best artifact. This keeps runs
  // hidden-blind during search (agents only see devScore) while making
  // hiddenScore immediately available without a backfill step.
  let hiddenScore: number | null = null;
  if (!config.submitForHiddenEval && task.hiddenVerifier) {
    try {
      hiddenScore = await runPythonVerifier(task.hiddenVerifier, best.solutionData);
      console.log(`  → hiddenScore=${hiddenScore} (local)`);
    } catch (err) {
      console.error(`  → local hidden eval failed: ${err}`);
    }
  }

  const result: RunResult = {
    taskId: task.id,
    protocolId: protocol.id,
    runIndex,
    seed,
    bestArtifact: best,
    allArtifacts,
    budgetTrace: trace,
    submissionId,
    hiddenScore,
    timestamp: new Date().toISOString(),
  };

  // Log budget usage
  console.log(`  → devScore=${best.devScore} tokens=${trace.tokenUsage.total} wall=${Math.round(trace.wallClockMs / 1000)}s evalCalls=${trace.evalCalls.length} exhausted=${JSON.stringify(trace.exhausted)}`);

  return result;
}

// ── Full Experiment ──

export async function runExperiment(
  config: ExperimentConfig,
  platformOverride?: PlatformClient,
): Promise<ExperimentResults> {
  mkdirSync(config.outputDir, { recursive: true });

  // Initialize platform client for server-side evaluation.
  // Tests may pass an instrumented instance (e.g., one that captures every
  // submissionId written via evalFinal) via platformOverride.
  const platform = platformOverride ?? new PlatformClient(config.apiUrl, config.apiKey);

  const totalCells = config.tasks.length * config.protocols.length * config.seeds.length;
  console.log(`\n=== Experiment: ${config.name} ===`);
  console.log(`Platform: ${config.apiUrl || 'https://agent4science.org'}`);
  console.log(`Tasks: ${config.tasks.length}, Protocols: ${config.protocols.length}, Seeds: ${config.seeds.length}`);
  console.log(`Total cells: ${totalCells}`);
  console.log(`Budget: T=${config.budget.tokenCap} W=${config.budget.wallClockSeconds}s C=${config.budget.evalCpuSeconds}s K=${config.budget.evalCallCap}\n`);

  const runs: RunResult[] = [];
  let completed = 0;
  let skippedExisting = 0;

  for (const task of config.tasks) {
    for (const protocol of config.protocols) {
      for (let i = 0; i < config.seeds.length; i++) {
        const seed = config.seeds[i];
        const filename = `${task.id}_${protocol.id}_s${seed}.json`;
        const filepath = join(config.outputDir, filename);

        // Resume support: skip cells that already have a result file on disk.
        // This makes re-runs after credit exhaustion or crashes safe — only
        // missing cells are executed, completed cells are loaded from disk.
        if (existsSync(filepath)) {
          try {
            const existing = JSON.parse(readFileSync(filepath, 'utf-8')) as RunResult;
            runs.push(existing);
            skippedExisting++;
          } catch {
            // Corrupt file — re-run this cell
          }
          completed++;
          continue;
        }

        try {
          const result = await runCell(task, protocol, seed, i, config, platform);
          if (result !== null) {
            runs.push(result);
            // Save incrementally — only successful cells get a JSON
            writeFileSync(filepath, JSON.stringify(result, null, 2));
          }
        } catch (err) {
          console.error(`  FATAL: ${task.id}×${protocol.id}×s${seed}: ${err}`);
        }

        completed++;
        console.log(`  Progress: ${completed}/${totalCells}\n`);
      }
    }
  }

  // Compute MEG
  const meg = computeMEG(runs, config);

  // Compute aggregate MEG
  const aggregateMEG = computeAggregateMEG(meg);

  const results: ExperimentResults = {
    config,
    runs,
    meg,
    aggregateMEG,
    completedAt: new Date().toISOString(),
  };

  // Save full results
  writeFileSync(
    join(config.outputDir, 'results.json'),
    JSON.stringify(results, null, 2)
  );

  // Save MEG table as TSV for easy viewing
  writeMEGTable(meg, aggregateMEG, config.outputDir);

  console.log(`\n=== Experiment complete ===`);
  console.log(`Results saved to ${config.outputDir}/`);
  if (skippedExisting > 0) console.log(`Resumed: ${skippedExisting} existing cells loaded from disk, ${completed - skippedExisting} new cells run`);
  printMEGSummary(meg, aggregateMEG);

  return results;
}

// ── MEG Computation ──

const BASELINE_PROTOCOLS: ProtocolId[] = ['self-refine', 'best-of-n', 'vgs'];

/**
 * Pull the per-cell hidden score for MEG computation. Runs that have not yet
 * been salvaged (hiddenScore null/undefined) are dropped from the
 * computation — MEG is defined on V_c^final and silently substituting dev
 * scores would reproduce the original proxy-overfit-confounded bug.
 */
function hiddenScoresFor(runs: RunResult[]): number[] {
  return runs
    .map(r => r.hiddenScore)
    .filter((s): s is number => typeof s === 'number');
}

export function computeMEG(runs: RunResult[], config: ExperimentConfig): MEGResult[] {
  const results: MEGResult[] = [];

  // Group runs by (task, protocol)
  const grouped = new Map<string, RunResult[]>();
  for (const run of runs) {
    const key = `${run.taskId}::${run.protocolId}`;
    if (!grouped.has(key)) grouped.set(key, []);
    grouped.get(key)!.push(run);
  }

  // For each task × protocol, compute MEG using Q-normalized HIDDEN scores
  for (const task of config.tasks) {
    const anchor = NORMALIZATION_ANCHORS[task.id];

    // Get baseline Q scores from hidden scores only
    const baselineQScores: Record<string, number[]> = {};
    const baselineRawScores: Record<string, number[]> = {};
    for (const baseId of BASELINE_PROTOCOLS) {
      const key = `${task.id}::${baseId}`;
      const baseRuns = grouped.get(key) ?? [];
      const rawScores = hiddenScoresFor(baseRuns);
      baselineRawScores[baseId] = rawScores;
      baselineQScores[baseId] = anchor
        ? rawScores.map(s => normalizeQ(s, anchor))
        : rawScores;
    }

    for (const protocol of config.protocols) {
      const key = `${task.id}::${protocol.id}`;
      const protocolRuns = grouped.get(key) ?? [];
      const rawScores = hiddenScoresFor(protocolRuns);
      if (rawScores.length === 0) continue;

      const qScores = anchor
        ? rawScores.map(s => normalizeQ(s, anchor))
        : rawScores;
      const protocolQMean = mean(qScores);
      const protocolRawMean = mean(rawScores);

      // Find best baseline by Q score (MEG denominator = max of SR/BoN/VGS)
      let bestBaselineQMean = -Infinity;
      let bestBaselineRawMean = 0;
      let bestBaselineId: ProtocolId = 'self-refine';

      for (const baseId of BASELINE_PROTOCOLS) {
        const bq = baselineQScores[baseId] ?? [];
        if (bq.length === 0) continue;
        const m = mean(bq);
        if (m > bestBaselineQMean) {
          bestBaselineQMean = m;
          bestBaselineRawMean = mean(baselineRawScores[baseId] ?? []);
          bestBaselineId = baseId as ProtocolId;
        }
      }

      // If no baselines have hidden data, skip — MEG is undefined
      if (bestBaselineQMean === -Infinity) continue;

      // MEG = E[Q(y_p)] - max(E[Q(y_SR)], E[Q(y_BoN)], E[Q(y_VGS)])
      const meg = protocolQMean - bestBaselineQMean;

      // Bootstrap 95% CI on Q-score differences
      const bestBaselineQArr = baselineQScores[bestBaselineId] ?? [];
      const { lower, upper } = bootstrapCI(qScores, bestBaselineQArr, 1000);

      // Permutation test p-value on Q scores
      const pValue = permutationTest(qScores, bestBaselineQArr, 1000);

      results.push({
        taskId: task.id,
        protocolId: protocol.id,
        meg,
        ci95Lower: lower,
        ci95Upper: upper,
        pValue,
        meanHiddenScore: protocolRawMean,
        bestBaselineHiddenScore: bestBaselineRawMean,
        bestBaselineId,
        meanQScore: protocolQMean,
        bestBaselineQScore: bestBaselineQMean,
        nCells: rawScores.length,
      });
    }
  }

  return results;
}

export function computeAggregateMEG(taskMEG: MEGResult[]): ExperimentResults['aggregateMEG'] {
  const byProtocol = new Map<ProtocolId, MEGResult[]>();
  for (const r of taskMEG) {
    if (!byProtocol.has(r.protocolId)) byProtocol.set(r.protocolId, []);
    byProtocol.get(r.protocolId)!.push(r);
  }

  const aggregate: ExperimentResults['aggregateMEG'] = [];
  for (const [protocolId, results] of byProtocol) {
    const megs = results.map(r => r.meg);
    const m = mean(megs);
    // Simple bootstrap on aggregated MEGs
    const boot = bootstrapMeanCI(megs, 1000);
    aggregate.push({
      protocolId,
      meg: m,
      ci95Lower: boot.lower,
      ci95Upper: boot.upper,
    });
  }

  return aggregate;
}

// ── Statistics ──

function mean(arr: number[]): number {
  if (arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function bootstrapCI(
  treatment: number[],
  control: number[],
  nBootstrap: number
): { lower: number; upper: number } {
  if (treatment.length === 0 || control.length === 0) {
    return { lower: 0, upper: 0 };
  }

  const diffs: number[] = [];
  for (let i = 0; i < nBootstrap; i++) {
    const tSample = sampleWithReplacement(treatment);
    const cSample = sampleWithReplacement(control);
    diffs.push(mean(tSample) - mean(cSample));
  }

  diffs.sort((a, b) => a - b);
  return {
    lower: diffs[Math.floor(nBootstrap * 0.025)],
    upper: diffs[Math.floor(nBootstrap * 0.975)],
  };
}

function bootstrapMeanCI(
  values: number[],
  nBootstrap: number
): { lower: number; upper: number } {
  if (values.length === 0) return { lower: 0, upper: 0 };

  const means: number[] = [];
  for (let i = 0; i < nBootstrap; i++) {
    means.push(mean(sampleWithReplacement(values)));
  }

  means.sort((a, b) => a - b);
  return {
    lower: means[Math.floor(nBootstrap * 0.025)],
    upper: means[Math.floor(nBootstrap * 0.975)],
  };
}

function permutationTest(
  treatment: number[],
  control: number[],
  nPermutations: number
): number {
  if (treatment.length === 0 || control.length === 0) return 1;

  const observedDiff = mean(treatment) - mean(control);
  const combined = [...treatment, ...control];
  let count = 0;

  for (let i = 0; i < nPermutations; i++) {
    // Shuffle combined
    const shuffled = [...combined];
    for (let j = shuffled.length - 1; j > 0; j--) {
      const k = Math.floor(Math.random() * (j + 1));
      [shuffled[j], shuffled[k]] = [shuffled[k], shuffled[j]];
    }

    const permT = shuffled.slice(0, treatment.length);
    const permC = shuffled.slice(treatment.length);
    const permDiff = mean(permT) - mean(permC);

    if (permDiff >= observedDiff) count++;
  }

  return count / nPermutations;
}

function sampleWithReplacement(arr: number[]): number[] {
  const result: number[] = [];
  for (let i = 0; i < arr.length; i++) {
    result.push(arr[Math.floor(Math.random() * arr.length)]);
  }
  return result;
}

// ── Output ──

export function writeMEGTable(meg: MEGResult[], aggregate: ExperimentResults['aggregateMEG'], outputDir: string): void {
  const lines = ['task\tprotocol\tMEG\tCI_lower\tCI_upper\tp_value\tmean_Q\tbaseline_Q\tbest_baseline\tmean_hidden_raw\tbaseline_hidden_raw\tn_cells'];
  for (const r of meg) {
    lines.push([
      r.taskId,
      r.protocolId,
      r.meg.toFixed(4),
      r.ci95Lower.toFixed(4),
      r.ci95Upper.toFixed(4),
      r.pValue.toFixed(4),
      (r.meanQScore ?? r.meanHiddenScore).toFixed(4),
      (r.bestBaselineQScore ?? r.bestBaselineHiddenScore).toFixed(4),
      r.bestBaselineId,
      r.meanHiddenScore.toFixed(4),
      r.bestBaselineHiddenScore.toFixed(4),
      String(r.nCells ?? 0),
    ].join('\t'));
  }
  lines.push('');
  lines.push('# Aggregate MEG');
  lines.push('protocol\tMEG\tCI_lower\tCI_upper');
  for (const a of aggregate) {
    lines.push(`${a.protocolId}\t${a.meg.toFixed(4)}\t${a.ci95Lower.toFixed(4)}\t${a.ci95Upper.toFixed(4)}`);
  }

  writeFileSync(join(outputDir, 'meg_table.tsv'), lines.join('\n'));
}

export function printMEGSummary(meg: MEGResult[], aggregate: ExperimentResults['aggregateMEG']): void {
  console.log('\n── MEG Results (per task × protocol) ──');
  console.log('Task\t\t\tProtocol\tMEG\t\tCI\t\t\tp-value');
  for (const r of meg) {
    if (BASELINE_PROTOCOLS.includes(r.protocolId)) continue; // skip baselines in summary
    const taskShort = r.taskId.slice(0, 16).padEnd(16);
    const proto = r.protocolId.padEnd(12);
    console.log(`${taskShort}\t${proto}\t${r.meg.toFixed(3)}\t\t[${r.ci95Lower.toFixed(3)}, ${r.ci95Upper.toFixed(3)}]\t${r.pValue.toFixed(3)}`);
  }

  console.log('\n── Aggregate MEG (per protocol) ──');
  for (const a of aggregate) {
    if (BASELINE_PROTOCOLS.includes(a.protocolId)) continue;
    console.log(`${a.protocolId.padEnd(12)}\tMEG=${a.meg.toFixed(3)}\t[${a.ci95Lower.toFixed(3)}, ${a.ci95Upper.toFixed(3)}]`);
  }
}

// ── CLI ──

async function main() {
  const args = process.argv.slice(2);

  let configPath: string | undefined;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--config' && args[i + 1]) {
      configPath = args[i + 1];
    }
  }

  if (!configPath) {
    console.error('Usage: npx tsx src/runner/experiment-runner.ts --config <path>');
    console.error('  e.g. npx tsx src/runner/experiment-runner.ts --config src/config/pilot.json');
    process.exit(1);
  }

  if (!existsSync(configPath)) {
    console.error(`Config file not found: ${configPath}`);
    process.exit(1);
  }

  const config = JSON.parse(readFileSync(configPath, 'utf-8')) as ExperimentConfig;
  await runExperiment(config);
}

// Only run CLI when this file is the direct entry point (not when imported as a module)
const isMain = process.argv[1]?.endsWith('experiment-runner.ts') ||
               process.argv[1]?.endsWith('experiment-runner.js');
if (isMain) {
  main().catch(err => {
    console.error('Fatal error:', err);
    process.exit(1);
  });
}
