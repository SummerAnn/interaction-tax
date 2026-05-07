/**
 * Benchmark experiment types for the anonymous release artifact.
 *
 * Extended types for optimization/coding benchmark tasks with deterministic
 * verifiers. Compatible with flamebird's benchmark module but designed for
 * verifier-backed scientific tasks rather than open-ended T2 peer scoring.
 */

// ── Budget Vector ──

/** The 4-component budget vector B = (T, W, C, K) */
export interface BudgetVector {
  /** T: billed token cap (prompt + completion) */
  tokenCap: number;
  /** W: wall-clock limit in seconds */
  wallClockSeconds: number;
  /** C: evaluator CPU-seconds cap */
  evalCpuSeconds: number;
  /** K: hard evaluator call cap */
  evalCallCap: number;
}

// ── Tasks ──

export type BenchmarkTrack = 'scientific' | 'coding';

export interface BenchmarkTask {
  id: string;
  title: string;
  track: BenchmarkTrack;
  /** Problem statement shown to agents */
  prompt: string;
  /** Python verifier source: def evaluate(data: dict) -> float */
  devVerifier: string;
  /** Hidden Python verifier source (full objective, not shown to agents during search) */
  hiddenVerifier?: string;
  /** JSON schema for solutionData */
  solutionSchema: Record<string, unknown>;
  /** Score direction for this task */
  scoringDirection: 'maximize' | 'minimize';
  /** Known optimal score (if available, for saturation analysis) */
  knownOptimal?: number;
  /** Task-specific parameters (e.g., N for Thomson N=47) */
  parameters?: Record<string, unknown>;
  /** Optional remote-evaluation challenge ID */
  challengeId?: string;
  /** Optional post-processor applied to parsed LLM output before evaluation */
  normalizeOutput?: (data: Record<string, unknown>) => Record<string, unknown>;
}

// ── Protocols ──

export type ProtocolId =
  | 'single-shot'
  | 'self-refine'
  | 'best-of-n'
  | 'vgs'
  | 'homo-chain'
  | 'cross-chain'
  | 'magicore'
  | 'debate'
  | 'moa'
  | 'hpe'
  // MoA no-synthesis ablation — runs diverse models in parallel like MoA but
  // skips the aggregation step; returns the best individual proposal by dev
  // score. Tests whether MoA's synthesis actually adds value over argmax.
  | 'moa-nosynth'
  // MoA coverage ablation — same-model variant to test whether MoA benefit
  // comes from model diversity or just independent parallel generation.
  | 'moa-same-model'
  // Mixed-brain variants — same protocol structure but each role uses a
  // different backbone (Claude / GPT-4o / Gemini). Run alongside the
  // homogeneous originals so we can ask "do different brains help, or just
  // more rounds of the same brain?" within one paper.
  | 'magicore-mixed'
  | 'debate-mixed'
  | 'hpe-mixed'
  // Open-source mixed-brain variants — same protocol structure but the three
  // backbones are weight-available models (DeepSeek V3 / Llama 3.3 70B /
  // Qwen 2.5 72B). Lets us ask "does brain diversity help when none of the
  // brains are frontier?" — decoupling diversity from raw model capability
  // and from API spend.
  | 'magicore-mixed-oss'
  | 'debate-mixed-oss'
  | 'hpe-mixed-oss'
  // Homogeneous-DeepSeek variants — the missing corner of the brain-diversity
  // 2×2. Same role-structured protocols as magicore/debate/hpe, but every role
  // runs on the single strongest open-weight model (DeepSeek V3). Pairs with
  // the *-mixed-oss cells to close the (frontier vs OSS) × (homo vs mixed)
  // design, turning the OSS arm from a one-sided check into a full 2×2.
  | 'magicore-deepseek'
  | 'debate-deepseek'
  | 'hpe-deepseek'
  // GPT-4o Solo variants — same protocol structure as the Claude homogeneous
  // originals but with GPT-4o as the sole backbone. Lets the paper ask whether
  // the multi-agent benefit pattern is model-universal.
  | 'single-shot-gpt4o'
  | 'self-refine-gpt4o'
  | 'best-of-n-gpt4o'
  | 'vgs-gpt4o'
  | 'homo-chain-gpt4o'
  | 'magicore-gpt4o'
  | 'debate-gpt4o'
  | 'hpe-gpt4o'
  // Gemini Solo variants — same protocol structure with Gemini 2.5 Flash.
  | 'single-shot-gemini'
  | 'self-refine-gemini'
  | 'best-of-n-gemini'
  | 'vgs-gemini'
  | 'homo-chain-gemini'
  | 'magicore-gemini'
  | 'debate-gemini'
  | 'hpe-gemini'
  // DeepSeek single-agent controls — MEG denominator for the OSS arm.
  // Mirrors the Claude SR/BoN/VGS controls but uses DeepSeek V3 as backbone,
  // so magicore-deepseek / debate-deepseek / hpe-deepseek MEG is computed
  // relative to the best single-agent DeepSeek baseline rather than Claude.
  | 'self-refine-deepseek'
  | 'best-of-n-deepseek'
  | 'vgs-deepseek'
  // Multi-synthesizer 2×2 expansion — same 2×2 (diversity × aggregation) design
  // but with GPT-4o or Gemini as the synthesizer instead of Claude. Tests whether
  // the diversity advantage holds regardless of which model does the synthesis.
  // Best-of-5-diverse: 5 diverse models generate independently, pick best.
  // Prescriptive protocol derived from the diversity tax analysis — maximize
  // independence by adding more model families with zero interaction.
  | 'best-of-5-diverse'
  | 'moa-synth-gpt4o'
  | 'moa-same-synth-gpt4o'
  | 'moa-synth-gemini'
  | 'moa-same-synth-gemini'
  // MoA-5-Diverse: 5 diverse models propose independently, Claude synthesizes
  // all 5. Second composition protocol to test whether the composition-positive
  // MIG finding generalizes beyond 3-model MoA.
  | 'moa-5-diverse'
  // Best-of-3 matched control for 2×2 ablation.
  | 'best-of-3'
  // Evolutionary crossover protocols — explicit component-level recombination
  // rather than consensus or critique. Addresses W7 (no adaptive composition tested).
  // crossover: Phase 1 (independent) + Phase 2 (crossover recombination), no refinement.
  // crossover-refine: Phase 1 + Phase 2 + Phase 3 (local refinement from crossover output).
  | 'crossover'
  | 'crossover-refine'
  // Budget-scaling variants (2× budget cap)
  | 'single-shot-2x'
  | 'self-refine-2x'
  | 'best-of-n-2x'
  | 'vgs-2x'
  | 'moa-2x';

export type LLMProvider = 'openrouter' | 'anthropic' | 'openai';

export interface BackboneConfig {
  provider: LLMProvider;
  model: string;
  apiKey: string;
}

export interface ProtocolConfig {
  id: ProtocolId;
  name: string;
  /** Backbone(s) used by this protocol */
  backbones: BackboneConfig[];
  /** Protocol-specific parameters */
  params: Record<string, unknown>;
}

// ── VGS-specific ──

export interface VGSConfig {
  populationSize: number;
  generations: number;
  /** Number of top candidates carried over to next generation */
  eliteCount: number;
  /** Stop after this many generations with no improvement */
  convergenceWindow: number;
}

export const DEFAULT_VGS_CONFIG: VGSConfig = {
  populationSize: 4,
  generations: 5,
  eliteCount: 2,
  convergenceWindow: 2,
};

// ── Artifacts & Results ──

export interface TokenUsage {
  prompt: number;
  completion: number;
  total: number;
}

export interface EvalCall {
  timestamp: string;
  score: number;
  cpuMs: number;
}

export interface BudgetTrace {
  tokenUsage: TokenUsage;
  wallClockMs: number;
  evalCpuMs: number;
  evalCalls: EvalCall[];
  /** Which caps were hit */
  exhausted: {
    tokens: boolean;
    wallClock: boolean;
    evalCpu: boolean;
    evalCalls: boolean;
  };
}

export interface SolutionArtifact {
  /** Solution data matching the task's solutionSchema */
  solutionData: Record<string, unknown>;
  /** Dev verifier score (public feedback during search) */
  devScore: number;
  /** Raw LLM output that produced this solution */
  rawOutput: string;
}

export interface RunResult {
  taskId: string;
  protocolId: ProtocolId;
  runIndex: number;
  seed: number;
  /** Best artifact found during the run */
  bestArtifact: SolutionArtifact;
  /** All intermediate artifacts (for revision chain analysis) */
  allArtifacts: SolutionArtifact[];
  /** Full budget trace */
  budgetTrace: BudgetTrace;
  /** Remote-evaluation submission ID (if used) */
  submissionId?: string;
  /**
   * Hidden verifier score from V_c^final (the actual benchmark score).
   * Populated only by offline analysis (e.g., salvage script) — runs are
   * intentionally hidden-blind during execution. Null if not yet retrieved
   * or if the submission did not run a hidden verifier.
   */
  hiddenScore?: number | null;
  timestamp: string;
}

// ── Experiment Configuration ──

export interface ExperimentConfig {
  id: string;
  name: string;
  /** Tasks to run */
  tasks: BenchmarkTask[];
  /** Protocols to run */
  protocols: ProtocolConfig[];
  /** Budget vector (same for all protocols in a track) */
  budget: BudgetVector;
  /** Seeds per (task, protocol) cell */
  seeds: number[];
  /** Output directory for results */
  outputDir: string;
  /** Submit to a remote evaluator for hidden evaluation? */
  submitForHiddenEval?: boolean;
  /** Remote-evaluation API key */
  apiKey?: string;
  /** Remote-evaluation API base URL */
  apiUrl?: string;
  /** Agent ID to attribute all submissions to (fallback when protocolAgentMap has no match) */
  agentId?: string;
  /** Per-protocol-family agent IDs. Keys are protocol ID prefixes (e.g. "magicore-mixed-oss"). */
  protocolAgentMap?: Record<string, string>;
}

// ── MEG Results ──

export interface MEGResult {
  taskId: string;
  protocolId: ProtocolId;
  /** MEG = E[Q(y_p)] - max(E[Q(y_SR)], E[Q(y_BoN)], E[Q(y_VGS)]) — computed on Q-normalized hidden scores */
  meg: number;
  /** 95% bootstrap CI on MEG */
  ci95Lower: number;
  ci95Upper: number;
  /** p-value from permutation test on Q scores */
  pValue: number;
  /** Mean raw hidden score for this protocol */
  meanHiddenScore: number;
  /** Mean raw hidden score for best baseline */
  bestBaselineHiddenScore: number;
  bestBaselineId: ProtocolId;
  /** Mean Q-normalized hidden score for this protocol (Q_bench in [0,1]) */
  meanQScore?: number;
  /** Mean Q-normalized hidden score for best baseline */
  bestBaselineQScore?: number;
  /** Number of cells aggregated (some may be missing due to API errors) */
  nCells?: number;
}

export interface ExperimentResults {
  config: ExperimentConfig;
  runs: RunResult[];
  meg: MEGResult[];
  /** Aggregate MEG per protocol across tasks */
  aggregateMEG: Array<{
    protocolId: ProtocolId;
    meg: number;
    ci95Lower: number;
    ci95Upper: number;
  }>;
  completedAt: string;
}
