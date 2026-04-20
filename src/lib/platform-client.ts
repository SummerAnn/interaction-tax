/**
 * Agent4Science Platform Client
 *
 * Calls the benchmark evaluation endpoint on agent4science.org.
 * All verifier execution happens server-side — the experiment runner
 * never sees the hidden verifier, only gets dev scores back.
 */

const DEFAULT_API_URL = 'https://agent4science.org';

export interface DevEvalResult {
  devScore: number;
  evalCpuMs: number;
}

export interface FinalEvalResult {
  devScore: number;
  evalCpuMs: number;
  submissionId: string;
}

/**
 * Stored submission record returned by the offline retrieval endpoint.
 * `hiddenScore` is the V_c^final score and is intentionally only available
 * via this admin path, not via `evalFinal`, so the live runner stays
 * hidden-blind.
 */
export interface BenchmarkSubmissionRecord {
  submissionId: string;
  challengeId: string | null;
  devScore: number | null;
  hiddenScore: number | null;
  benchmarkMeta: {
    protocolId?: string | null;
    runIndex?: number | null;
    seed?: number | null;
    evalCpuMs?: number | null;
  } | null;
  improvesUpon: string | null;
  createdAt: string | null;
}

export class PlatformClient {
  private apiUrl: string;
  private adminSecret: string;

  constructor(apiUrl?: string, adminSecret?: string) {
    this.apiUrl = (apiUrl || process.env.A4S_API_URL || DEFAULT_API_URL).replace(/\/$/, '');
    this.adminSecret = adminSecret || process.env.ADMIN_SECRET || '';
    if (!this.adminSecret) {
      throw new Error('ADMIN_SECRET is required for benchmark evaluation');
    }
  }

  /**
   * Run dev verifier only (fast path for iterative search).
   * Returns the dev score for budget-aware protocols like VGS and Self-Refine.
   */
  async evalDev(
    challengeId: string,
    solutionData: Record<string, unknown>,
    meta?: { protocolId?: string; runIndex?: number; seed?: number }
  ): Promise<DevEvalResult> {
    const res = await fetch(`${this.apiUrl}/api/v1/benchmark/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.adminSecret}`,
      },
      body: JSON.stringify({
        challengeId,
        solutionData,
        final: false,
        ...meta,
      }),
    });

    const data = await res.json() as { success: boolean; devScore?: number; evalCpuMs?: number; error?: string };

    // Any non-success response is a verifier error (e.g., "Expected 200
    // labels, got 199"), even if the server returns a placeholder devScore.
    // The previous `!data.success && data.devScore === undefined` guard let
    // failed evals through with devScore=0, which then collided with the
    // stricter evalFinal error handling: chains would pile up score=0
    // malformed artifacts that passed evalDev but later aborted submission
    // when evalFinal re-rejected them.
    if (!data.success) {
      throw new Error(`Dev eval failed: ${data.error || res.statusText}`);
    }

    return {
      devScore: data.devScore ?? 0,
      evalCpuMs: data.evalCpuMs ?? 0,
    };
  }

  /**
   * Final submission: runs dev + hidden verifier, creates submission on platform.
   * Hidden score is stored server-side but never returned.
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
      scoreTrace?: Array<{ step: number; score: number }>;
      tokenUsage?: { prompt: number; completion: number; total: number };
    }
  ): Promise<FinalEvalResult> {
    const res = await fetch(`${this.apiUrl}/api/v1/benchmark/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.adminSecret}`,
      },
      body: JSON.stringify({
        challengeId,
        solutionData,
        final: true,
        ...meta,
      }),
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Final eval failed (${res.status}): ${text}`);
    }

    const data = await res.json() as {
      success: boolean;
      devScore: number;
      evalCpuMs: number;
      submissionId: string;
      error?: string;
    };

    if (!data.success) {
      throw new Error(`Final eval failed: ${data.error}`);
    }

    return {
      devScore: data.devScore,
      evalCpuMs: data.evalCpuMs,
      submissionId: data.submissionId,
    };
  }

  /**
   * Offline retrieval: fetch a previously created benchmark submission's
   * stored dev + hidden verifier scores. Used by salvage/analysis scripts;
   * NEVER call this from inside a protocol runner — doing so would break
   * the hidden-blind invariant.
   */
  async getSubmission(submissionId: string): Promise<BenchmarkSubmissionRecord> {
    const res = await fetch(`${this.apiUrl}/api/v1/benchmark/submission/${encodeURIComponent(submissionId)}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${this.adminSecret}`,
      },
    });

    if (!res.ok) {
      const text = await res.text();
      throw new Error(`getSubmission failed (${res.status}): ${text}`);
    }

    const data = await res.json() as {
      success: boolean;
      submissionId: string;
      challengeId: string | null;
      devScore: number | null;
      hiddenScore: number | null;
      benchmarkMeta: BenchmarkSubmissionRecord['benchmarkMeta'];
      improvesUpon: string | null;
      createdAt: string | null;
      error?: string;
    };

    if (!data.success) {
      throw new Error(`getSubmission failed: ${data.error}`);
    }

    return {
      submissionId: data.submissionId,
      challengeId: data.challengeId,
      devScore: data.devScore,
      hiddenScore: data.hiddenScore,
      benchmarkMeta: data.benchmarkMeta,
      improvesUpon: data.improvesUpon,
      createdAt: data.createdAt,
    };
  }
}
