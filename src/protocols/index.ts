/**
 * Protocol Registry
 *
 * Maps protocol IDs to their runner functions.
 * All protocols take (task, backbone(s), budget) and return SolutionArtifact[].
 */

import type { BackboneConfig, BenchmarkTask, ProtocolId, ProtocolConfig, SolutionArtifact } from '../types.js';
import { BudgetEnforcer } from '../budget/budget-vector.js';
import { runSingleShot, runSelfRefine, runBestOfN } from './baselines.js';
import { runVGS } from './vgs.js';
import { runHomoChain, runCrossChain, runMAgICoRe, runDebate, runMoA, runMoANoSynth } from './multi-agent.js';
import { runHPE } from './hpe.js';

export type ProtocolRunner = (
  task: BenchmarkTask,
  config: ProtocolConfig,
  budget: BudgetEnforcer,
) => Promise<SolutionArtifact[]>;

export const PROTOCOL_RUNNERS: Record<ProtocolId, ProtocolRunner> = {
  'single-shot': async (task, config, budget) =>
    runSingleShot(task, config.backbones[0], budget),

  'self-refine': async (task, config, budget) =>
    runSelfRefine(task, config.backbones[0], budget, (config.params.rounds as number) ?? 3),

  'best-of-n': async (task, config, budget) =>
    runBestOfN(task, config.backbones[0], budget, (config.params.n as number) ?? 8),

  'vgs': async (task, config, budget) =>
    runVGS(task, config.backbones[0], budget, {
      populationSize: (config.params.populationSize as number) ?? 4,
      generations: (config.params.generations as number) ?? 5,
      eliteCount: (config.params.eliteCount as number) ?? 2,
      convergenceWindow: (config.params.convergenceWindow as number) ?? 2,
    }),

  'homo-chain': async (task, config, budget) =>
    runHomoChain(task, config.backbones[0], budget, (config.params.chainLength as number) ?? 4),

  'cross-chain': async (task, config, budget) =>
    runCrossChain(task, config.backbones, budget),

  'magicore': async (task, config, budget) =>
    runMAgICoRe(task, config.backbones, budget, (config.params.rounds as number) ?? 2),

  'debate': async (task, config, budget) =>
    runDebate(task, config.backbones, budget, (config.params.debateRounds as number) ?? 2),

  'moa': async (task, config, budget) =>
    runMoA(task, config.backbones, budget),

  // MoA no-synthesis ablation — diverse models generate in parallel, best
  // proposal wins by dev score (no aggregation step). Tests whether MoA's
  // synthesis adds value over simply picking the best individual output.
  'moa-nosynth': async (task, config, budget) =>
    runMoANoSynth(task, config.backbones, budget),

  // MoA same-model ablation — reuses runMoA; the only difference is that
  // run-full.ts passes 3× Claude instead of Claude+GPT-4o+Gemini.
  'moa-same-model': async (task, config, budget) =>
    runMoA(task, config.backbones, budget),

  'hpe': async (task, config, budget) =>
    runHPE(
      task,
      config.backbones,
      budget,
      (config.params.numExecutors as number) ?? 3,
      (config.params.rounds as number) ?? 2,
    ),

  // Mixed-brain variants reuse the same runner functions — the role-based
  // backbone assignment lives inside each runner and activates automatically
  // when the protocol passes more than one backbone.
  'magicore-mixed': async (task, config, budget) =>
    runMAgICoRe(task, config.backbones, budget, (config.params.rounds as number) ?? 2),

  'debate-mixed': async (task, config, budget) =>
    runDebate(task, config.backbones, budget, (config.params.debateRounds as number) ?? 2),

  'hpe-mixed': async (task, config, budget) =>
    runHPE(
      task,
      config.backbones,
      budget,
      (config.params.numExecutors as number) ?? 3,
      (config.params.rounds as number) ?? 2,
    ),

  // Open-source mixed-brain variants — identical wiring to the *-mixed
  // entries above. The OSS-vs-frontier distinction is purely a matter of
  // which backbones run-full.ts hands in (crossChainOSSBackbones() vs
  // crossChainBackbones()), so we just reuse the same runners.
  'magicore-mixed-oss': async (task, config, budget) =>
    runMAgICoRe(task, config.backbones, budget, (config.params.rounds as number) ?? 2),

  'debate-mixed-oss': async (task, config, budget) =>
    runDebate(task, config.backbones, budget, (config.params.debateRounds as number) ?? 2),

  'hpe-mixed-oss': async (task, config, budget) =>
    runHPE(
      task,
      config.backbones,
      budget,
      (config.params.numExecutors as number) ?? 3,
      (config.params.rounds as number) ?? 2,
    ),

  // Homogeneous-DeepSeek variants — same runner code as magicore/debate/hpe,
  // but run-full.ts hands in a single-element [deepseekBackbone()] array.
  // The role-based fan-out inside each runner degrades gracefully to "every
  // role gets backbones[0]" when only one backbone is supplied, so the same
  // runner function produces both the homogeneous and the mixed behaviour.
  // These three cells close the brain-diversity 2×2 (frontier vs OSS) × (homo
  // vs mixed) by supplying the missing homogeneous-OSS corner.
  'magicore-deepseek': async (task, config, budget) =>
    runMAgICoRe(task, config.backbones, budget, (config.params.rounds as number) ?? 2),

  'debate-deepseek': async (task, config, budget) =>
    runDebate(task, config.backbones, budget, (config.params.debateRounds as number) ?? 2),

  'hpe-deepseek': async (task, config, budget) =>
    runHPE(
      task,
      config.backbones,
      budget,
      (config.params.numExecutors as number) ?? 3,
      (config.params.rounds as number) ?? 2,
    ),

  // GPT-4o Solo variants — identical runners to the Claude originals; the sole
  // difference is that run-full.ts passes [gpt4oBackbone()] as backbones.
  'single-shot-gpt4o': async (task, config, budget) =>
    runSingleShot(task, config.backbones[0], budget),

  'self-refine-gpt4o': async (task, config, budget) =>
    runSelfRefine(task, config.backbones[0], budget, (config.params.rounds as number) ?? 3),

  'best-of-n-gpt4o': async (task, config, budget) =>
    runBestOfN(task, config.backbones[0], budget, (config.params.n as number) ?? 8),

  'vgs-gpt4o': async (task, config, budget) =>
    runVGS(task, config.backbones[0], budget, {
      populationSize: (config.params.populationSize as number) ?? 4,
      generations: (config.params.generations as number) ?? 5,
      eliteCount: (config.params.eliteCount as number) ?? 2,
      convergenceWindow: (config.params.convergenceWindow as number) ?? 2,
    }),

  'homo-chain-gpt4o': async (task, config, budget) =>
    runHomoChain(task, config.backbones[0], budget, (config.params.chainLength as number) ?? 4),

  'magicore-gpt4o': async (task, config, budget) =>
    runMAgICoRe(task, config.backbones, budget, (config.params.rounds as number) ?? 2),

  'debate-gpt4o': async (task, config, budget) =>
    runDebate(task, config.backbones, budget, (config.params.debateRounds as number) ?? 2),

  'hpe-gpt4o': async (task, config, budget) =>
    runHPE(
      task,
      config.backbones,
      budget,
      (config.params.numExecutors as number) ?? 3,
      (config.params.rounds as number) ?? 2,
    ),

  // Gemini Solo variants — identical runners with [geminiBackbone()] as backbones.
  'single-shot-gemini': async (task, config, budget) =>
    runSingleShot(task, config.backbones[0], budget),

  'self-refine-gemini': async (task, config, budget) =>
    runSelfRefine(task, config.backbones[0], budget, (config.params.rounds as number) ?? 3),

  'best-of-n-gemini': async (task, config, budget) =>
    runBestOfN(task, config.backbones[0], budget, (config.params.n as number) ?? 8),

  'vgs-gemini': async (task, config, budget) =>
    runVGS(task, config.backbones[0], budget, {
      populationSize: (config.params.populationSize as number) ?? 4,
      generations: (config.params.generations as number) ?? 5,
      eliteCount: (config.params.eliteCount as number) ?? 2,
      convergenceWindow: (config.params.convergenceWindow as number) ?? 2,
    }),

  'homo-chain-gemini': async (task, config, budget) =>
    runHomoChain(task, config.backbones[0], budget, (config.params.chainLength as number) ?? 4),

  'magicore-gemini': async (task, config, budget) =>
    runMAgICoRe(task, config.backbones, budget, (config.params.rounds as number) ?? 2),

  'debate-gemini': async (task, config, budget) =>
    runDebate(task, config.backbones, budget, (config.params.debateRounds as number) ?? 2),

  'hpe-gemini': async (task, config, budget) =>
    runHPE(
      task,
      config.backbones,
      budget,
      (config.params.numExecutors as number) ?? 3,
      (config.params.rounds as number) ?? 2,
    ),

  // Best-of-5-diverse: 5 diverse models generate independently, pick best by
  // dev score. Prescriptive protocol: maximize independence (5 model families,
  // zero interaction). Tests whether the diversity tax theory is actionable.
  'best-of-5-diverse': async (task, config, budget) =>
    runMoANoSynth(task, config.backbones, budget),

  // Best-of-3: matched N=3 control for the clean 2×2 ablation.
  // Generates 3 proposals from a single model (Claude) and selects the best
  // by dev score. Matches MoA/MoA-nosynth/MoA-same-model arm size (N=3)
  // to eliminate the team-size confound with Best-of-8.
  'best-of-3': async (task, config, budget) =>
    runBestOfN(task, config.backbones[0], budget, (config.params.n as number) ?? 3),

  // ── Budget-scaling variants (2× budget) ──────────────────────────────────
  // Identical runner code, different protocol IDs so results land in
  // separate files. Budget enforcement comes from the BudgetEnforcer
  // passed in by the experiment runner, not from the protocol itself.
  'single-shot-2x': async (task, config, budget) =>
    runSingleShot(task, config.backbones[0], budget),

  'self-refine-2x': async (task, config, budget) =>
    runSelfRefine(task, config.backbones[0], budget, (config.params.rounds as number) ?? 6),

  'best-of-n-2x': async (task, config, budget) =>
    runBestOfN(task, config.backbones[0], budget, (config.params.n as number) ?? 16),

  'vgs-2x': async (task, config, budget) =>
    runVGS(task, config.backbones[0], budget, {
      populationSize: (config.params.populationSize as number) ?? 4,
      generations: (config.params.generations as number) ?? 10,
      eliteCount: (config.params.eliteCount as number) ?? 2,
      convergenceWindow: (config.params.convergenceWindow as number) ?? 2,
    }),

  'moa-2x': async (task, config, budget) =>
    runMoA(task, config.backbones, budget),

  // DeepSeek single-agent controls — MEG denominator for the OSS arm.
  // Identical runner functions as the Claude originals; backbone distinction
  // is in run-full.ts via deepseekBackbone().
  'self-refine-deepseek': async (task, config, budget) =>
    runSelfRefine(task, config.backbones[0], budget, (config.params.rounds as number) ?? 3),

  'best-of-n-deepseek': async (task, config, budget) =>
    runBestOfN(task, config.backbones[0], budget, (config.params.n as number) ?? 8),

  'vgs-deepseek': async (task, config, budget) =>
    runVGS(task, config.backbones[0], budget, {
      populationSize: (config.params.populationSize as number) ?? 4,
      generations: (config.params.generations as number) ?? 5,
      eliteCount: (config.params.eliteCount as number) ?? 2,
      convergenceWindow: (config.params.convergenceWindow as number) ?? 2,
    }),

  // ── Multi-synthesizer 2×2 expansion ──────────────────────────────────────
  // Same MoA structure (N=3 proposers → synthesizer) but with a configurable
  // synthesizer model. The aggregatorBackbone is passed via config.params
  // so the proposer backbones array stays clean.
  'moa-synth-gpt4o': async (task, config, budget) =>
    runMoA(task, config.backbones, budget, config.params.aggregatorBackbone as BackboneConfig),

  'moa-same-synth-gpt4o': async (task, config, budget) =>
    runMoA(task, config.backbones, budget, config.params.aggregatorBackbone as BackboneConfig),

  'moa-synth-gemini': async (task, config, budget) =>
    runMoA(task, config.backbones, budget, config.params.aggregatorBackbone as BackboneConfig),

  'moa-same-synth-gemini': async (task, config, budget) =>
    runMoA(task, config.backbones, budget, config.params.aggregatorBackbone as BackboneConfig),

};

/**
 * Get the best artifact from a run based on scoring direction.
 */
export function getBestArtifact(
  artifacts: SolutionArtifact[],
  scoringDirection: 'maximize' | 'minimize',
): SolutionArtifact | null {
  if (artifacts.length === 0) return null;
  return artifacts.reduce((best, curr) => {
    const better = scoringDirection === 'maximize'
      ? curr.devScore > best.devScore
      : curr.devScore < best.devScore;
    return better ? curr : best;
  });
}
