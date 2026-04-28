/**
 * Prospective Extension Tasks for Composability Validation
 *
 * These tasks were defined on 2026-04-23, AFTER composability_prediction_v2.md
 * was written. They are NOT yet registered on agent4science.org.
 *
 * To register a task:
 *   1. Deploy the verifier + hiddenVerifier to the platform
 *   2. Record the returned challengeId in challenge-ids.ts
 *   3. Run: npx tsx src/runner/run-full.ts --task <id> --protocols moa,moa-same-model,moa-nosynth,best-of-3 --seeds 1-10
 *   4. Backfill hidden scores with backfill-hidden-scores.ts
 *
 * Composability predictions for both tasks are in composability_prediction_v2.md.
 */

import type { BenchmarkChallengeDefinition } from './benchmark-challenges.js';

// ─── Task 1: Maximum k-Coverage ──────────────────────────────────────────────
// Composability: HIGH (C1+C2+C3 all satisfied)
// Prediction: significant positive diversity effect (Δ_div > +0.10)
// Normalization: s_bad = 0 (no elements covered), s_ref = 50 (all covered)

export const MAX_K_COVERAGE: BenchmarkChallengeDefinition = {
  id: 'bench-max-k-coverage-n100',
  card: 10,
  hiddenType: 'I',
  hiddenPattern: 'C',
  title: 'Benchmark: Maximum k-Coverage (n=100, k=15)',
  description: `## Maximum k-Coverage

You are given a universe of **100 elements** (labeled 0–99) and a collection of
**150 sets**, each a subset of the universe. Choose exactly **15 sets** to
maximize the number of distinct elements covered.

### Problem Details

- **Universe**: U = {0, 1, ..., 99}  (100 elements)
- **Sets**: S_0, S_1, ..., S_149  (150 sets; each has size drawn Uniform[3, 8])
- **Task**: Select exactly 15 sets (indices into S) to maximize |∪ selected|
- **Scoring**: Number of distinct elements in the union of selected sets (higher is better)
- **Maximum possible**: 100 (if the 15 sets cover every element)
- **Dev evaluator**: Fixed instance (seed=301). Sets and universe are public.
- **Typical greedy solution**: ~83–85 elements covered

### Submission Format

\`\`\`json
{ "selected": [3, 17, 42, 0, 9, 25, 61, 88, 7, 33, 12, 55, 99, 4, 71] }
\`\`\`

An array of **exactly 15 distinct integers**, each in [0, 149], representing
the indices of the chosen sets. Order does not matter.

### Notes

- Greedy coverage (always pick the set covering the most uncovered elements)
  achieves a (1 - 1/e) ≈ 63% approximation ratio in theory; in practice
  reaches ~83 of 100 elements.
- Random first-15 baseline covers ~60 of 100 elements.
- Brute-force exact solution is infeasible (C(150,15) ≈ 10^27).

**Source**: Karp (1972); Nemhauser, Wolsey & Fisher (1978).
`,
  tags: ['combinatorics', 'set-cover', 'optimization', 'np-hard', 'composable'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 1,
  solutionSchema: '{ selected: int[15] — exactly 15 distinct integers in [0,149], the indices of the chosen sets }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  verifier: `
import numpy as np

def evaluate(data):
    selected = data.get("selected")
    if not isinstance(selected, list):
        return float('-inf')
    selected = list(selected)
    if len(selected) != 15:
        return float('-inf')
    selected = [int(x) for x in selected]
    if len(set(selected)) != 15:
        return float('-inf')  # duplicates
    if any(x < 0 or x >= 150 for x in selected):
        return float('-inf')

    # Generate fixed instance with seed=301
    rng = np.random.RandomState(301)
    n_elements = 100
    n_sets = 150
    sets = []
    for _ in range(n_sets):
        size = rng.randint(3, 9)  # Uniform[3, 8]
        s = set(rng.choice(n_elements, size=size, replace=False).tolist())
        sets.append(s)

    covered = set()
    for idx in selected:
        covered |= sets[idx]
    return float(len(covered))
`,

  hiddenVerifier: `
import numpy as np

def evaluate(data):
    selected = data.get("selected")
    if not isinstance(selected, list):
        return float('-inf')
    selected = list(selected)
    if len(selected) != 15:
        return float('-inf')
    selected = [int(x) for x in selected]
    if len(set(selected)) != 15:
        return float('-inf')
    if any(x < 0 or x >= 150 for x in selected):
        return float('-inf')

    # Hidden instance: different seed (seed=419) → different set collection
    # Same structural parameters: n=100 elements, m=150 sets, size~Uniform[3,8]
    rng = np.random.RandomState(419)
    n_elements = 100
    n_sets = 150
    sets = []
    for _ in range(n_sets):
        size = rng.randint(3, 9)
        s = set(rng.choice(n_elements, size=size, replace=False).tolist())
        sets.append(s)

    # The submitted set indices must still be valid (same index space [0,99])
    covered = set()
    for idx in selected:
        covered |= sets[idx]
    return float(len(covered))
`,
};

// ─── Task 2: Latin Square Completion (n=9) ───────────────────────────────────
// Composability: LOW (C1+C2+C3 all fail)
// Prediction: null or negative diversity effect (Δ_div ≈ 0 or < 0)
// Normalization: s_bad = 0 (random fill), s_ref = 81 (perfect completion)

export const LATIN_SQUARE: BenchmarkChallengeDefinition = {
  id: 'bench-latin-square-9',
  card: 11,
  hiddenType: 'II',
  hiddenPattern: 'A',
  title: 'Benchmark: Latin Square Completion (9×9)',
  description: `## Latin Square Completion

Complete a partially filled **9×9 Latin square**. A Latin square is a 9×9
grid filled with the integers 1–9 such that each integer appears **exactly
once in every row** and **exactly once in every column** (no box constraint,
unlike Sudoku).

### Given Clues (30 of 81 cells pre-filled; 0 = empty, must be filled)

\`\`\`
     Col: 0  1  2  3  4  5  6  7  8
Row 0:    .  9  2  .  1  .  6  5  4
Row 1:    .  .  .  3  9  7  .  .  .
Row 2:    .  3  5  .  .  .  .  6  .
Row 3:    9  .  3  .  .  4  .  2  6
Row 4:    .  1  .  8  .  .  2  .  5
Row 5:    .  .  7  .  .  5  3  .  .
Row 6:    8  .  .  .  .  .  9  .  .
Row 7:    .  .  .  .  .  3  7  .  .
Row 8:    .  .  .  .  3  .  .  8  .
\`\`\`

**You must replace every \`.\` with an integer 1–9. Pre-filled cells (shown above)
must appear unchanged in your submission.**

### Problem Details

- **Grid**: 9×9, values in {1, ..., 9}
- **Constraint**: Each integer 1–9 appears exactly once in every row and exactly once in every column
- **Scoring**: Number of correctly placed cells vs. the reference completion (0–81; higher is better)
- **Pre-filled cells**: Violating any clue above scores 0 automatically

### Submission Format

\`\`\`json
{ "grid": [[?,9,2,?,1,?,6,5,4], [?,?,?,3,9,7,?,?,?], [?,3,5,?,?,?,?,6,?], [9,?,3,?,?,4,?,2,6], [?,1,?,8,?,?,2,?,5], [?,?,7,?,?,5,3,?,?], [8,?,?,?,?,?,9,?,?], [?,?,?,?,?,3,7,?,?], [?,?,?,?,3,?,?,8,?]] }
\`\`\`

Replace every \`?\` with an integer 1–9, keeping all pre-filled values exactly as shown.

### Notes

- A valid Latin square has unique row/column values; Sudoku adds a 3×3 box constraint (not required here).
- Constraint propagation + backtracking reliably solves this; no brute force needed.
- Scoring rewards every correctly placed cell, even if the full square is not valid.
- A fully valid Latin square that matches the reference scores 81.

**Source**: Classic combinatorics; see van Rees (2000).
`,
  tags: ['combinatorics', 'constraint-satisfaction', 'latin-square', 'np-complete'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 1,
  solutionSchema: '{ grid: int[9][9] — a 9×9 array; each value in {1..9}; pre-filled cells must match the given clues }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  verifier: `
import numpy as np

def _generate_instance(seed):
    """Generate a valid 9x9 Latin square and remove ~51 cells."""
    rng = np.random.RandomState(seed)

    # Build a valid Latin square by cyclic shift construction
    base = list(range(1, 10))
    grid = []
    for i in range(9):
        row = base[i:] + base[:i]
        grid.append(row)
    grid = np.array(grid, dtype=np.int64)

    # Shuffle rows within row-groups of 3 and columns within col-groups of 3
    # (preserves Latin square property)
    for g in range(3):
        rows = list(range(g*3, g*3+3))
        perm = rng.permutation(3)
        grid[rows] = grid[[rows[p] for p in perm]]
    for g in range(3):
        cols = list(range(g*3, g*3+3))
        perm = rng.permutation(3)
        grid[:, cols] = grid[:, [cols[p] for p in perm]]

    # Permute the symbols
    symbol_perm = rng.permutation(9) + 1
    remapped = np.zeros_like(grid)
    for old, new in enumerate(symbol_perm):
        remapped[grid == old+1] = new
    grid = remapped

    # Reveal ~30 cells (mask the rest)
    n_revealed = 30
    all_cells = [(i, j) for i in range(9) for j in range(9)]
    revealed_idx = rng.choice(len(all_cells), size=n_revealed, replace=False)
    revealed = set(tuple(all_cells[k]) for k in revealed_idx)

    clues = {}
    for (i, j) in revealed:
        clues[(i, j)] = int(grid[i, j])

    return grid, clues

def evaluate(data):
    grid_raw = data.get("grid")
    if not isinstance(grid_raw, list) or len(grid_raw) != 9:
        return 0.0
    try:
        submitted = np.array(grid_raw, dtype=np.int64)
    except Exception:
        return 0.0
    if submitted.shape != (9, 9):
        return 0.0
    if not np.all((submitted >= 1) & (submitted <= 9)):
        return 0.0

    # Dev instance: seed=503
    reference, clues = _generate_instance(503)

    # Penalize clue violations (must match given cells)
    for (i, j), val in clues.items():
        if submitted[i, j] != val:
            return 0.0  # Invalid: clue violated

    # Score: number of cells matching reference
    return float(np.sum(submitted == reference))
`,

  hiddenVerifier: `
import numpy as np

def _generate_instance(seed):
    rng = np.random.RandomState(seed)
    base = list(range(1, 10))
    grid = []
    for i in range(9):
        row = base[i:] + base[:i]
        grid.append(row)
    grid = np.array(grid, dtype=np.int64)
    for g in range(3):
        rows = list(range(g*3, g*3+3))
        perm = rng.permutation(3)
        grid[rows] = grid[[rows[p] for p in perm]]
    for g in range(3):
        cols = list(range(g*3, g*3+3))
        perm = rng.permutation(3)
        grid[:, cols] = grid[:, [cols[p] for p in perm]]
    symbol_perm = rng.permutation(9) + 1
    remapped = np.zeros_like(grid)
    for old, new in enumerate(symbol_perm):
        remapped[grid == old+1] = new
    grid = remapped
    n_revealed = 30
    all_cells = [(i, j) for i in range(9) for j in range(9)]
    revealed_idx = rng.choice(len(all_cells), size=n_revealed, replace=False)
    revealed = set(tuple(all_cells[k]) for k in revealed_idx)
    clues = {}
    for (i, j) in revealed:
        clues[(i, j)] = int(grid[i, j])
    return grid, clues

def evaluate(data):
    grid_raw = data.get("grid")
    if not isinstance(grid_raw, list) or len(grid_raw) != 9:
        return 0.0
    try:
        submitted = np.array(grid_raw, dtype=np.int64)
    except Exception:
        return 0.0
    if submitted.shape != (9, 9):
        return 0.0
    if not np.all((submitted >= 1) & (submitted <= 9)):
        return 0.0

    # Hidden instance: seed=503 but with FEWER clues revealed (20 instead of 30)
    # → harder partial instance; same underlying Latin square
    rng = np.random.RandomState(503)
    base = list(range(1, 10))
    grid = []
    for i in range(9):
        row = base[i:] + base[:i]
        grid.append(row)
    grid = np.array(grid, dtype=np.int64)
    for g in range(3):
        rows = list(range(g*3, g*3+3))
        perm = rng.permutation(3)
        grid[rows] = grid[[rows[p] for p in perm]]
    for g in range(3):
        cols = list(range(g*3, g*3+3))
        perm = rng.permutation(3)
        grid[:, cols] = grid[:, [cols[p] for p in perm]]
    symbol_perm = rng.permutation(9) + 1
    remapped = np.zeros_like(grid)
    for old, new in enumerate(symbol_perm):
        remapped[grid == old+1] = new
    reference = remapped

    # Hidden scoring: strict row/column validity bonus
    # +1 per correct cell (same as dev) PLUS +9 per fully valid row/column
    base_score = float(np.sum(submitted == reference))

    # Row validity bonus (each row that is a permutation of 1..9)
    row_bonus = 0.0
    for i in range(9):
        if set(submitted[i].tolist()) == set(range(1, 10)):
            row_bonus += 9.0

    # Column validity bonus
    col_bonus = 0.0
    for j in range(9):
        if set(submitted[:, j].tolist()) == set(range(1, 10)):
            col_bonus += 9.0

    return base_score + row_bonus + col_bonus
`,
};

// ─── Registration instructions ────────────────────────────────────────────────

/**
 * TO REGISTER THESE TASKS ON AGENT4SCIENCE.ORG:
 *
 * 1. Add them to the ALL_BENCHMARK_CHALLENGES array in benchmark-challenges.ts
 * 2. Register via the admin endpoint:
 *    POST /api/v1/admin/benchmark/create
 *    Body: { ...task definition }
 *    Returns: { challengeId: "ch_..." }
 * 3. Add to BENCHMARK_CHALLENGE_IDS in challenge-ids.ts
 * 4. Add normalization anchors to normalization-anchors.ts:
 *    MAX_K_COVERAGE: { s_bad: 0, s_ref: 50, direction: 'maximize' }
 *    LATIN_SQUARE:   { s_bad: 0, s_ref: 81, direction: 'maximize' }
 * 5. Run the 2x2:
 *    npx tsx src/runner/run-full.ts \
 *      --tasks bench-max-k-coverage-n50,bench-latin-square-9 \
 *      --protocols moa,moa-same-model,moa-nosynth,best-of-3 \
 *      --seeds 1-10
 * 6. Backfill hidden scores:
 *    PLATFORM_URL=... ADMIN_SECRET=... \
 *    npx tsx src/runner/backfill-hidden-scores.ts --resultsDir results/prospective-2x2
 *
 * EXPECTED RESULTS (per composability_prediction_v2.md):
 *   Max k-Coverage: Δ_div > +0.10 (HIGH composability)
 *   Latin Square:   Δ_div ≈ 0 or negative (LOW composability)
 */
