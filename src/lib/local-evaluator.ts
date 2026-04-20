/**
 * Local Python Evaluator
 *
 * Runs Python verifiers locally via subprocess, matching the PlatformClient
 * interface so all existing protocol runners work without modification.
 *
 * Used for external validation tasks that don't live on agent4science.org.
 */

import { execSync } from 'child_process';
import { PlatformClient } from './platform-client.js';

export class LocalEvaluator extends PlatformClient {
  private verifiers: Map<string, string>;

  constructor(verifiers: Map<string, string>) {
    // Pass a dummy secret — we never call the real API
    super('http://localhost:0', 'local-eval-dummy');
    this.verifiers = verifiers;
  }

  /**
   * Override evalDev to run the Python verifier locally via subprocess.
   */
  override async evalDev(
    challengeId: string,
    solutionData: Record<string, unknown>,
    _meta?: { protocolId?: string; runIndex?: number; seed?: number },
  ): Promise<{ devScore: number; evalCpuMs: number }> {
    const verifierCode = this.verifiers.get(challengeId);
    if (!verifierCode) {
      throw new Error(`No local verifier registered for challenge: ${challengeId}`);
    }

    const script = `
import json, sys, time
${verifierCode}

data = json.loads(sys.stdin.read())
t0 = time.monotonic()
try:
    score = evaluate(data)
    cpu_ms = (time.monotonic() - t0) * 1000
    print(json.dumps({"score": score, "cpu_ms": cpu_ms}))
except Exception as e:
    print(json.dumps({"error": str(e)}), file=sys.stderr)
    sys.exit(1)
`;

    const inputJson = JSON.stringify(solutionData);
    const startMs = Date.now();

    try {
      const stdout = execSync(`python3 -c ${shellEscape(script)}`, {
        input: inputJson,
        encoding: 'utf-8',
        timeout: 30_000,
        maxBuffer: 10 * 1024 * 1024,
      }).trim();

      const result = JSON.parse(stdout) as { score: number; cpu_ms: number };
      return {
        devScore: result.score,
        evalCpuMs: result.cpu_ms,
      };
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      throw new Error(`Verifier error: ${msg}`);
    }
  }

  /**
   * Override evalFinal — for local tasks, just re-run the dev verifier.
   * No platform submission.
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
