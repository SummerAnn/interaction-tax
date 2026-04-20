/**
 * Full Budget Vector Enforcement: B = (T, W, C, K)
 *
 * Tracks and enforces all four budget components:
 *   T — billed token cap (prompt + completion)
 *   W — wall-clock limit in seconds
 *   C — evaluator CPU-seconds cap
 *   K — hard evaluator call cap
 *
 * Components are NOT exchangeable: hitting any cap terminates the run.
 */

import type {
  BudgetVector,
  TokenUsage,
  EvalCall,
  BudgetTrace,
  BackboneConfig,
} from '../types.js';
import { callLLM, type LLMMessage, type LLMResponse } from '../lib/llm-client.js';
import { PlatformClient } from '../lib/platform-client.js';

export class BudgetEnforcer {
  private budget: BudgetVector;
  private tokensUsed: TokenUsage = { prompt: 0, completion: 0, total: 0 };
  private startTime: number;
  private evalCpuMs = 0;
  private evalCalls: EvalCall[] = [];
  private platform: PlatformClient | null = null;

  constructor(budget: BudgetVector, platform?: PlatformClient) {
    this.budget = budget;
    this.startTime = Date.now();
    this.platform = platform ?? null;
  }

  // ── Query state ──

  get tokensRemaining(): number {
    return Math.max(0, this.budget.tokenCap - this.tokensUsed.total);
  }

  get wallClockRemainingMs(): number {
    const elapsed = Date.now() - this.startTime;
    return Math.max(0, this.budget.wallClockSeconds * 1000 - elapsed);
  }

  get evalCpuRemaining(): number {
    return Math.max(0, this.budget.evalCpuSeconds * 1000 - this.evalCpuMs);
  }

  get evalCallsRemaining(): number {
    return Math.max(0, this.budget.evalCallCap - this.evalCalls.length);
  }

  /** True if ALL budget components have remaining capacity */
  get hasBudget(): boolean {
    return (
      this.tokensRemaining > 0 &&
      this.wallClockRemainingMs > 0 &&
      this.evalCpuRemaining > 0 &&
      this.evalCallsRemaining > 0
    );
  }

  /** True if tokens and wall-clock allow another LLM call (eval budget separate) */
  get canCallLLM(): boolean {
    return this.tokensRemaining > 0 && this.wallClockRemainingMs > 0;
  }

  /** True if eval budget allows another evaluator call */
  get canCallEval(): boolean {
    return this.evalCallsRemaining > 0 && this.evalCpuRemaining > 0;
  }

  // ── Budget-aware operations ──

  /**
   * Make a budget-aware LLM call. Returns null if token or wall-clock budget exhausted.
   */
  async llmCall(
    backbone: BackboneConfig,
    messages: LLMMessage[],
    options?: { temperature?: number; maxTokens?: number }
  ): Promise<LLMResponse | null> {
    if (!this.canCallLLM) return null;

    // Clamp maxTokens to remaining budget
    const maxTokens = Math.min(
      options?.maxTokens ?? 4096,
      this.tokensRemaining
    );

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), this.wallClockRemainingMs);
    // Hard-cap each LLM call at wallClockRemaining + 20s via Promise.race.
    // The AbortController fires at wallClockRemaining, but undici (Node's fetch)
    // doesn't always tear down a hung TCP stream immediately when the signal fires
    // — the server (OpenRouter) may not close the connection for hours. The race
    // provides a guaranteed hard deadline regardless of socket behavior.
    const hardCapMs = Math.min(this.wallClockRemainingMs + 20_000, 620_000);
    const hardCapPromise = new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`LLM call hard-capped after ${hardCapMs}ms`)), hardCapMs)
    );
    let response: LLMResponse;
    try {
      response = await Promise.race([
        callLLM(backbone, messages, { ...options, maxTokens, signal: controller.signal }),
        hardCapPromise,
      ]);
    } finally {
      clearTimeout(timeout);
    }

    this.tokensUsed.prompt += response.usage.prompt;
    this.tokensUsed.completion += response.usage.completion;
    this.tokensUsed.total += response.usage.total;

    return response;
  }

  /**
   * Record an evaluator call. Returns false if eval budget is exhausted.
   * The caller provides the CPU time consumed by the verifier.
   */
  recordEvalCall(score: number, cpuMs: number): boolean {
    if (!this.canCallEval) return false;

    this.evalCpuMs += cpuMs;
    this.evalCalls.push({
      timestamp: new Date().toISOString(),
      score,
      cpuMs,
    });

    return true;
  }

  /**
   * Budget-aware dev evaluation via the platform API.
   * Returns { devScore, cpuMs } or null if eval budget is exhausted.
   */
  async evalDev(
    challengeId: string,
    solutionData: Record<string, unknown>,
    meta?: { protocolId?: string; runIndex?: number; seed?: number }
  ): Promise<{ devScore: number; cpuMs: number } | null> {
    if (!this.canCallEval) return null;
    if (!this.platform) throw new Error('PlatformClient required for evalDev');

    const result = await this.platform.evalDev(challengeId, solutionData, meta);

    this.evalCpuMs += result.evalCpuMs;
    this.evalCalls.push({
      timestamp: new Date().toISOString(),
      score: result.devScore,
      cpuMs: result.evalCpuMs,
    });

    return { devScore: result.devScore, cpuMs: result.evalCpuMs };
  }

  /**
   * Final submission via the platform API.
   * Runs both dev + hidden verifier server-side, creates submission.
   * Returns dev score only (hidden is stored but never returned).
   */
  async evalFinal(
    challengeId: string,
    solutionData: Record<string, unknown>,
    meta: {
      protocolId?: string;
      runIndex?: number;
      seed?: number;
      title?: string;
      body?: string;
      approach?: string;
      agentId?: string;
      improvesUpon?: string;
    }
  ): Promise<{ devScore: number; submissionId: string } | null> {
    if (!this.platform) throw new Error('PlatformClient required for evalFinal');

    const result = await this.platform.evalFinal(challengeId, solutionData, meta);

    // Don't count final eval against budget (it's the final submission, not search)
    return { devScore: result.devScore, submissionId: result.submissionId };
  }

  // ── Summary ──

  getTrace(): BudgetTrace {
    const elapsed = Date.now() - this.startTime;
    return {
      tokenUsage: { ...this.tokensUsed },
      wallClockMs: elapsed,
      evalCpuMs: this.evalCpuMs,
      evalCalls: [...this.evalCalls],
      exhausted: {
        tokens: this.tokensRemaining <= 0,
        wallClock: this.wallClockRemainingMs <= 0,
        evalCpu: this.evalCpuRemaining <= 0,
        evalCalls: this.evalCallsRemaining <= 0,
      },
    };
  }
}
