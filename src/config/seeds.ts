/**
 * Frozen Seed List — FROZEN 2026-04-06
 *
 * The core benchmark uses five seeds per (task, configuration) cell.
 * Some appendix-only follow-up analyses use larger seed counts and are stored
 * directly in results/full-v2 rather than being implied by this constant.
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

/** Five seeds used for each core benchmark cell. */
export const BENCHMARK_SEEDS: readonly number[] = [1, 2, 3, 4, 5] as const;

/**
 * Core benchmark cell count for the main paper sweep.
 * 11 tasks × 10 core configurations × 5 seeds = 550 cells.
 */
export const TOTAL_CELLS = 11 * 10 * 5; // 550

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
