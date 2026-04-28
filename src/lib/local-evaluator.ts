/**
 * Local Python Evaluator
 *
 * Runs Python verifiers locally via subprocess, matching the PlatformClient
 * interface so all existing protocol runners work without modification.
 *
 * Used for offline benchmark runs that don't touch agent4science.org.
 */

import { execSync } from 'child_process';
import { PlatformClient } from './platform-client.js';

/**
 * Run a Python verifier snippet locally.
 * The snippet must define `def evaluate(data: dict) -> float`.
 * Returns the float score on success, throws on verifier error.
 */
export async function runPythonVerifier(
  verifierCode: string,
  data: Record<string, unknown>,
): Promise<number> {
  const script = `
import json, sys
${verifierCode}

data = json.loads(sys.stdin.read())
try:
    score = evaluate(data)
    print(json.dumps({"score": float(score)}))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
`;

  const stdout = execSync(`python3 -c ${shellEscape(script)}`, {
    input: JSON.stringify(data),
    encoding: 'utf-8',
    timeout: 30_000,
    maxBuffer: 10 * 1024 * 1024,
  }).trim();

  const result = JSON.parse(stdout) as { score: number };
  return result.score;
}

export class LocalEvaluator extends PlatformClient {
  private devVerifiers: Map<string, string>;
  private hiddenVerifiers: Map<string, string>;

  constructor(
    devVerifiers: Map<string, string>,
    hiddenVerifiers?: Map<string, string>,
  ) {
    // Pass a dummy secret — we never call the real API
    super('http://localhost:0', 'local-eval-dummy');
    this.devVerifiers = devVerifiers;
    this.hiddenVerifiers = hiddenVerifiers ?? new Map();
  }

  /**
   * Override evalDev to run the dev verifier locally via subprocess.
   */
  override async evalDev(
    challengeId: string,
    solutionData: Record<string, unknown>,
    _meta?: { protocolId?: string; runIndex?: number; seed?: number },
  ): Promise<{ devScore: number; evalCpuMs: number }> {
    const verifierCode = this.devVerifiers.get(challengeId);
    if (!verifierCode) {
      throw new Error(`No local dev verifier registered for challenge: ${challengeId}`);
    }

    const t0 = Date.now();
    try {
      const score = await runPythonVerifier(verifierCode, solutionData);
      return { devScore: score, evalCpuMs: Date.now() - t0 };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Dev verifier error [${challengeId}]: ${msg}`);
    }
  }

  /**
   * Override evalFinal — runs dev verifier locally; no platform submission.
   * If a hidden verifier is registered, it runs that too and returns hiddenScore.
   */
  override async evalFinal(
    challengeId: string,
    solutionData: Record<string, unknown>,
    _meta: Record<string, unknown>,
  ): Promise<{ devScore: number; evalCpuMs: number; submissionId: string }> {
    const result = await this.evalDev(challengeId, solutionData);
    return {
      ...result,
      submissionId: `local-${challengeId}-${Date.now()}`,
    };
  }
}

/** Shell-escape a string for use in -c argument. */
function shellEscape(s: string): string {
  return "'" + s.replace(/'/g, "'\\''") + "'";
}
