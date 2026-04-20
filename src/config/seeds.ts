/**
 * Frozen Seed List — FROZEN 2026-04-06
 *
 * 5 seeds per (task, protocol) cell → 8 tasks × 9 protocols × 5 seeds = 360 runs total.
 *
 * Seeds are used as run indices that:
 * 1. Label each run uniquely in output filenames and logs
 * 2. Offset temperature schedules in Best-of-N and VGS (seed*0.01 jitter on temperature)
 * 3. Provide a stable reproduction key for re-running any individual cell
 *
 * The LLM API does not guarantee deterministic outputs for a given seed,
 * so seeds here are primarily labels and temperature offsets, not PRNG seeds.
 * Cross-run variance comes from LLM stochasticity.
 *
 * FREEZE RULE: Do not add, remove, or reorder seeds after the full run begins.
 * If a run fails, re-run with the exact same seed and log the failure reason.
 */

/** Five seeds used for every (task, protocol) cell in the full benchmark run. */
export const BENCHMARK_SEEDS: readonly number[] = [1, 2, 3, 4, 5] as const;

/**
 * Total cells: tasks × protocols × seeds.
 * 8 tasks × 9 protocols × 5 seeds = 360 runs.
 */
export const TOTAL_CELLS = 8 * 9 * 5; // 360

/**
 * Returns the temperature for a given seed within a Best-of-N or VGS run.
 * Base temperature + small seed-derived offset ensures diversity across seeds
 * while keeping temperatures in a reasonable range.
 *
 * Used by Best-of-N (indexed samples) and VGS population initialization.
 */
export function seedTemperature(seed: number, baseTemp = 0.7): number {
  // Seeds 1–5 produce temperatures: 0.71, 0.72, 0.73, 0.74, 0.75
  return baseTemp + seed * 0.01;
}
