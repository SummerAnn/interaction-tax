/**
 * Agent4Science-Bench — Benchmark Challenge Definitions
 *
 * All 9 benchmark tasks (8 core + Erdős near-core) with dev and hidden verifiers.
 * Parameterizations match benchmark_plan.md exactly.
 *
 * Hidden verifiers are stored server-side and never returned to the experiment runner.
 */

export interface BenchmarkChallengeDefinition {
  id: string;
  title: string;
  description: string;
  tags: string[];
  evaluationType: 'deterministic';
  scoringDirection: 'maximize' | 'minimize';
  minImprovement: number;
  verifier: string;
  hiddenVerifier: string;
  solutionSchema: string;
  experimental: true;
  benchmarkTrack: 'scientific';
  closesAt: string;
  /** Card number from benchmark_plan.md */
  card: number | 'near-core';
  /** Hidden evaluator type (I = same-objective, II = robustness-variant) */
  hiddenType: 'I' | 'II';
  /** Hidden pattern (A = held-out subset, B = precision split, C = instance split) */
  hiddenPattern: 'A' | 'B' | 'C';
}

// ─── Card 1: MaxCut ─────────────────────────────────────────────────────────

export const MAXCUT: BenchmarkChallengeDefinition = {
  id: 'bench-maxcut-g200',
  card: 1,
  hiddenType: 'I',
  hiddenPattern: 'A',
  title: 'Benchmark: MaxCut on G(200, 0.3) (v2)',
  description: `## MaxCut on Erdős–Rényi Random Graph

Find a partition of the 200 vertices of a fixed Erdős–Rényi random graph G(200, 0.3) that **maximizes the number of edges crossing the partition**.

### Problem Details

- **Graph**: G(200, 0.3) generated with seed=17. Fixed and public.
- **Submission**: A binary partition of all 200 vertices (0 or 1 labels).
- **Scoring**: Number of edges from the evaluation set that cross the partition (higher is better).
- **Dev evaluator**: Scores on a public 80% edge subset (~4,776 of ~5,970 edges, selected with seed=71).

### Submission Format

\`\`\`json
{ "partition": [0, 1, 1, 0, ...] }
\`\`\`

Array of exactly 200 integers, each 0 or 1.

### Notes

- Random partition baseline ≈ 2,985 cut edges on the full graph.
- Good heuristic solutions reach 3,500+.
- Optimal MaxCut on G(200,0.3) is NP-hard.

**Source**: Karp (1972)
`,
  tags: ['graph-theory', 'combinatorics', 'optimization', 'np-hard'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 1,
  solutionSchema: '{ partition: int[200] — EXACTLY 200 integers, each 0 or 1. The array must have exactly 200 elements, one per vertex (vertices 0–199). Do not truncate or extend. }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: scores on 80% public edge subset (seed=71)
  verifier: `
import numpy as np

def evaluate(data):
    partition = np.array(data["partition"], dtype=np.int64)
    n = 200
    assert len(partition) == n, f"Expected {n} labels, got {len(partition)}"
    assert np.all((partition == 0) | (partition == 1)), "Labels must be 0 or 1"

    # Generate fixed graph G(200, 0.3) with seed=17
    rng = np.random.RandomState(17)
    adj = (rng.random((n, n)) < 0.3).astype(np.int64)
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj, 1)  # upper triangle only

    # Select 80% public edge subset with seed=71
    edge_rows, edge_cols = np.where(adj == 1)
    rng2 = np.random.RandomState(71)
    mask = rng2.random(len(edge_rows)) < 0.8
    pub_rows = edge_rows[mask]
    pub_cols = edge_cols[mask]

    # Count cut edges on public subset
    cut = int(np.sum(partition[pub_rows] != partition[pub_cols]))
    return float(cut)
`,

  // Hidden verifier: scores on ALL edges (full graph)
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    partition = np.array(data["partition"], dtype=np.int64)
    n = 200
    assert len(partition) == n, f"Expected {n} labels, got {len(partition)}"
    assert np.all((partition == 0) | (partition == 1)), "Labels must be 0 or 1"

    # Generate fixed graph G(200, 0.3) with seed=17
    rng = np.random.RandomState(17)
    adj = (rng.random((n, n)) < 0.3).astype(np.int64)
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj, 1)

    # Count cut edges on FULL graph
    edge_rows, edge_cols = np.where(adj == 1)
    cut = int(np.sum(partition[edge_rows] != partition[edge_cols]))
    return float(cut)
`,
};

// ─── Card 2: Circle Packing ─────────────────────────────────────────────────

export const CIRCLE_PACKING: BenchmarkChallengeDefinition = {
  id: 'bench-circle-packing-n20',
  card: 2,
  hiddenType: 'I',
  hiddenPattern: 'B',
  title: 'Benchmark: Circle Packing in a Square (N=20) (v2)',
  description: `## Circle Packing in a Square

Pack **20 non-overlapping circles** inside the unit square [0,1]² to **maximize the sum of all radii**.

### Problem Details

- **N**: 20 circles
- **Domain**: Unit square [0,1]²
- **Constraints**: No overlaps between circles, all circles fully contained in the square.
- **Scoring**: Sum of all radii (higher is better). Violations return -inf.
- **Dev evaluator**: Overlap and containment tolerance = 1e-6 (relaxed).

### Submission Format

\`\`\`json
{ "circles": [[x1, y1, r1], [x2, y2, r2], ...] }
\`\`\`

Array of exactly 20 triples [x, y, radius].

### Notes

- Solutions that are "barely valid" under the dev tolerance may become invalid under stricter evaluation.
- Best known packings for N=20 are available in the Packomania database but exact coordinates are non-trivial.

**Source**: Specht (2024)
`,
  tags: ['geometry', 'packing', 'continuous-optimization', 'constraints'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 0.0001,
  solutionSchema: '{ circles: float[20][3] — [x, y, r] triples }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: tolerance 1e-6
  verifier: `
import numpy as np

def evaluate(data):
    try:
        circles = np.array(data["circles"], dtype=np.float64)
    except (TypeError, ValueError, KeyError):
        return float('-inf')
    if circles.shape != (20, 3):
        return float('-inf')

    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    tol = 1e-6

    # All radii must be positive
    if np.any(rs <= 0):
        return float('-inf')

    # Containment: circle must be inside [0,1]²
    if np.any(xs - rs < -tol) or np.any(xs + rs > 1 + tol):
        return float('-inf')
    if np.any(ys - rs < -tol) or np.any(ys + rs > 1 + tol):
        return float('-inf')

    # Non-overlap: pairwise distance >= sum of radii
    for i in range(20):
        for j in range(i + 1, 20):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if dist < rs[i] + rs[j] - tol:
                return float('-inf')

    return float(np.sum(rs))
`,

  // Hidden verifier: tolerance 1e-12 (strict)
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    try:
        circles = np.array(data["circles"], dtype=np.float64)
    except (TypeError, ValueError, KeyError):
        return float('-inf')
    if circles.shape != (20, 3):
        return float('-inf')

    xs, ys, rs = circles[:, 0], circles[:, 1], circles[:, 2]
    tol = 1e-12

    if np.any(rs <= 0):
        return float('-inf')

    if np.any(xs - rs < -tol) or np.any(xs + rs > 1 + tol):
        return float('-inf')
    if np.any(ys - rs < -tol) or np.any(ys + rs > 1 + tol):
        return float('-inf')

    for i in range(20):
        for j in range(i + 1, 20):
            dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
            if dist < rs[i] + rs[j] - tol:
                return float('-inf')

    return float(np.sum(rs))
`,
};

// ─── Card 3: Difference Bases ────────────────────────────────────────────────

export const DIFFERENCE_BASES: BenchmarkChallengeDefinition = {
  id: 'bench-difference-bases',
  card: 3,
  hiddenType: 'II',
  hiddenPattern: 'A',
  title: 'Benchmark: Difference Bases (v2)',
  description: `## Difference Bases

Find a set B ⊂ ℤ≥0 that **minimizes |B|²/v**, where v is the largest integer such that every integer in {1, 2, ..., v} appears as a pairwise difference of elements in B.

### Problem Details

- **Submission**: A set of non-negative integers (at most 2000 elements).
- **Scoring**: |B|²/v where v = max contiguous coverage (lower is better).
- **Theoretical lower bound**: 1.

### Submission Format

\`\`\`json
{ "set": [0, 1, 3, 7, 12, ...] }
\`\`\`

Array of non-negative integers, max 2000 elements.

### Notes

- 0 is automatically included if not present.
- The dev evaluator requires perfect contiguous coverage: all of {1..v} must appear as pairwise differences.
- This is a niche number theory problem with low contamination risk.

**Source**: Erdős & Turán (1941)
`,
  tags: ['number-theory', 'combinatorics', 'optimization', 'set-theory'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 0.001,
  solutionSchema: '{ set: int[] — non-negative integers, max 2000 elements }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: strict contiguous coverage
  verifier: `
import numpy as np

def evaluate(data):
    elements = sorted(set(int(x) for x in data["set"]))
    if not elements:
        return float('inf')
    if elements[0] != 0:
        elements = [0] + elements
    if len(elements) > 2000:
        return float('inf')

    B = len(elements)
    arr = np.array(elements, dtype=np.int64)

    # Compute all pairwise positive differences
    diffs = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            diffs.add(int(arr[j] - arr[i]))

    if not diffs:
        return float('inf')

    # Find max contiguous v: all of 1..v present
    v = 0
    for k in range(1, max(diffs) + 1):
        if k in diffs:
            v = k
        else:
            break

    if v == 0:
        return float('inf')

    return float(B * B) / float(v)
`,

  // Hidden verifier: 95% robust coverage
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    elements = sorted(set(int(x) for x in data["set"]))
    if not elements:
        return float('inf')
    if elements[0] != 0:
        elements = [0] + elements
    if len(elements) > 2000:
        return float('inf')

    B = len(elements)
    arr = np.array(elements, dtype=np.int64)

    # Compute all pairwise positive differences
    diffs = set()
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            diffs.add(int(arr[j] - arr[i]))

    if not diffs:
        return float('inf')

    # Find max v such that >= 95% of 1..v appear as differences
    max_diff = max(diffs)
    best_v = 0
    for candidate_v in range(1, max_diff + 1):
        covered = sum(1 for k in range(1, candidate_v + 1) if k in diffs)
        coverage_ratio = covered / candidate_v
        if coverage_ratio >= 0.95:
            best_v = candidate_v

    if best_v == 0:
        return float('inf')

    return float(B * B) / float(best_v)
`,
};

// ─── Card 4: Flat Polynomials ────────────────────────────────────────────────

export const FLAT_POLYNOMIALS: BenchmarkChallengeDefinition = {
  id: 'bench-flat-poly-deg50',
  card: 4,
  hiddenType: 'I',
  hiddenPattern: 'B',
  title: 'Benchmark: Flat Polynomials (Degree 50) (v2)',
  description: `## Flat Polynomials — Littlewood Problem

Find a ±1 coefficient polynomial of degree 50 that **minimizes the peak-to-RMS ratio** on the unit circle.

### Problem Details

- **Degree**: 50 (51 coefficients)
- **Coefficients**: Each must be exactly +1 or -1 (Littlewood polynomial)
- **Scoring**: max|P(z)| / √51 evaluated on the unit circle (lower is better)
- **Dev evaluator**: Evaluates on 100,000 evenly-spaced unit-circle points

### Submission Format

\`\`\`json
{ "coefficients": [1, -1, 1, 1, -1, ...] }
\`\`\`

Array of exactly 51 integers, each +1 or -1.

### Notes

- The Littlewood conjecture states the ratio cannot reach 1. Best known ratios for degree 50 are well above 1.
- The theoretical lower bound is 1, useful for normalization.

**Source**: Littlewood (1966)
`,
  tags: ['algebra', 'polynomials', 'signal-processing', 'optimization'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 0.001,
  solutionSchema: '{ coefficients: int[51] — each +1 or -1 }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: 100K evenly-spaced points
  verifier: `
import numpy as np

def evaluate(data):
    coeffs = np.array(data["coefficients"], dtype=np.float64)
    n = 51
    assert len(coeffs) == n, f"Expected {n} coefficients, got {len(coeffs)}"
    assert np.all(np.isin(coeffs, [-1, 1])), "All coefficients must be +1 or -1"

    # Evaluate on 100K evenly-spaced unit-circle points
    N_pts = 100000
    theta = np.linspace(0, 2 * np.pi, N_pts, endpoint=False)
    z = np.exp(1j * theta)

    # Horner's method for polynomial evaluation
    P = np.zeros(N_pts, dtype=np.complex128)
    for k in range(n - 1, -1, -1):
        P = P * z + coeffs[k]

    peak = np.max(np.abs(P))
    rms = np.sqrt(n)
    return float(peak / rms)
`,

  // Hidden verifier: 2M points including adversarially chosen points near peaks
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    coeffs = np.array(data["coefficients"], dtype=np.float64)
    n = 51
    assert len(coeffs) == n, f"Expected {n} coefficients, got {len(coeffs)}"
    assert np.all(np.isin(coeffs, [-1, 1])), "All coefficients must be +1 or -1"

    # Phase 1: 500K evenly-spaced points
    N_coarse = 500000
    theta_coarse = np.linspace(0, 2 * np.pi, N_coarse, endpoint=False)
    z_coarse = np.exp(1j * theta_coarse)

    P_coarse = np.zeros(N_coarse, dtype=np.complex128)
    for k in range(n - 1, -1, -1):
        P_coarse = P_coarse * z_coarse + coeffs[k]

    abs_coarse = np.abs(P_coarse)

    # Phase 2: find top-100 peaks, refine with ±0.001 rad neighborhood
    top_indices = np.argsort(abs_coarse)[-100:]
    top_thetas = theta_coarse[top_indices]

    adversarial_thetas = []
    for t in top_thetas:
        adversarial_thetas.extend(np.linspace(t - 0.001, t + 0.001, 150).tolist())

    theta_adv = np.array(adversarial_thetas)
    z_adv = np.exp(1j * theta_adv)

    P_adv = np.zeros(len(theta_adv), dtype=np.complex128)
    for k in range(n - 1, -1, -1):
        P_adv = P_adv * z_adv + coeffs[k]

    # Phase 3: combine all evaluations
    all_abs = np.concatenate([abs_coarse, np.abs(P_adv)])
    peak = np.max(all_abs)
    rms = np.sqrt(n)
    return float(peak / rms)
`,
};

// ─── Card 5: Thomson Problem (N=83) ─────────────────────────────────────────

export const THOMSON: BenchmarkChallengeDefinition = {
  id: 'bench-thomson-n83',
  card: 5,
  hiddenType: 'II',
  hiddenPattern: 'B',
  title: 'Benchmark: Thomson Problem (N=83)',
  description: `## Thomson Problem — Point Placement on a Sphere

Place **83 points** on the unit sphere to **minimize the Coulomb energy** (sum of 1/r_ij for all pairs).

### Problem Details

- **N**: 83 (non-canonical, reduces contamination vs N=100 or N=50)
- **Domain**: Unit sphere in ℝ³
- **Scoring**: Coulomb energy = Σ_{i<j} 1/||p_i - p_j|| (lower is better)
- **Dev evaluator**: Standard Coulomb energy

### Submission Format

\`\`\`json
{ "vectors": [[x1, y1, z1], [x2, y2, z2], ...] }
\`\`\`

Array of exactly 83 three-element vectors. Points are projected to the unit sphere.

### Notes

- N=83 is chosen to be non-canonical (not N=100, 50, etc.) to reduce contamination.
- This is a CONDITIONAL task — requires pilot validation that single-shot does not saturate.

**Source**: Thomson (1904)
`,
  tags: ['physics', 'geometry', 'continuous-optimization', 'electrostatics'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 0.01,
  solutionSchema: '{ vectors: float[83][3] — points projected to unit sphere }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: Coulomb energy
  verifier: `
import numpy as np

def evaluate(data):
    pts = np.array(data["vectors"], dtype=np.float64)
    N = 83
    assert pts.shape == (N, 3), f"Expected ({N}, 3), got {pts.shape}"

    # Project to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms

    # Coulomb energy: sum of 1/r_ij for all pairs
    energy = 0.0
    for i in range(N):
        diffs = pts[i + 1:] - pts[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists = np.maximum(dists, 1e-15)
        energy += np.sum(1.0 / dists)

    return float(energy)
`,

  // Hidden verifier: Riesz s=1.05 energy
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    pts = np.array(data["vectors"], dtype=np.float64)
    N = 83
    assert pts.shape == (N, 3), f"Expected ({N}, 3), got {pts.shape}"

    # Project to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms

    # Riesz s-energy with s=1.05: sum of 1/r_ij^1.05
    s = 1.05
    energy = 0.0
    for i in range(N):
        diffs = pts[i + 1:] - pts[i]
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        dists = np.maximum(dists, 1e-15)
        energy += np.sum(1.0 / (dists ** s))

    return float(energy)
`,
};

// ─── Card 6: TSP-100 ────────────────────────────────────────────────────────

export const TSP_100: BenchmarkChallengeDefinition = {
  id: 'bench-tsp-100',
  card: 6,
  hiddenType: 'II',
  hiddenPattern: 'A',
  title: 'Benchmark: Traveling Salesman (N=100) (v2)',
  description: `## Traveling Salesman Problem (N=100)

Find the **shortest tour** through 100 cities in the [0, 1000]² domain.

### Problem Details

- **N**: 100 cities, positions generated deterministically from seed=42
- **Domain**: [0, 1000]²
- **Scoring**: Total Euclidean tour length including return to start (lower is better)
- **Dev evaluator**: Tour scored on public city coordinates

### Submission Format

\`\`\`json
{ "tour": [0, 42, 17, 3, ...] }
\`\`\`

A permutation of integers 0..99 representing the visit order.

### Notes

- Random tour baseline ≈ 50,000. Good heuristics (LKH) reach ≈ 7,000–8,500.
- Replaces Golomb Rulers (dropped for very high contamination — order 20 optimal is widely published).

**Source**: Dantzig, Fulkerson & Johnson (1954)
`,
  tags: ['combinatorics', 'optimization', 'np-hard', 'routing'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 1,
  solutionSchema: '{ tour: int[100] — permutation of 0..99 }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: exact city coordinates from seed=42
  verifier: `
import numpy as np

def evaluate(data):
    tour = np.array(data["tour"], dtype=np.int64)
    N = 100
    assert len(tour) == N, f"Expected {N} cities, got {len(tour)}"
    assert set(tour.tolist()) == set(range(N)), "Tour must be a permutation of 0..99"

    # Generate city coordinates from seed=42
    rng = np.random.RandomState(42)
    cities = rng.uniform(0, 1000, size=(N, 2))

    # Compute round-trip tour length
    total = 0.0
    for i in range(N):
        a = cities[tour[i]]
        b = cities[tour[(i + 1) % N]]
        total += np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    return float(total)
`,

  // Hidden verifier: perturbed city coordinates (seed=91, ±5 units)
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    tour = np.array(data["tour"], dtype=np.int64)
    N = 100
    assert len(tour) == N, f"Expected {N} cities, got {len(tour)}"
    assert set(tour.tolist()) == set(range(N)), "Tour must be a permutation of 0..99"

    # Generate original city coordinates from seed=42
    rng = np.random.RandomState(42)
    cities = rng.uniform(0, 1000, size=(N, 2))

    # Apply hidden perturbation: ±5 units per coordinate (seed=91)
    rng2 = np.random.RandomState(91)
    perturbation = rng2.uniform(-5, 5, size=(N, 2))
    cities = cities + perturbation

    # Compute round-trip tour length on perturbed coordinates
    total = 0.0
    for i in range(N):
        a = cities[tour[i]]
        b = cities[tour[(i + 1) % N]]
        total += np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    return float(total)
`,
};

// ─── Card 7: HP Lattice Folding ──────────────────────────────────────────────

// 50-mer sequence (label "48mer" retained in ID for Firestore consistency)
const HP_DEV_SEQUENCE = 'HPHPPHHPHPPHPHHPPHPHPHHPPHHPHPPHPHHPPHPHPHHPPHHPHH';

export const HP_FOLDING: BenchmarkChallengeDefinition = {
  id: 'bench-hp-folding-48mer',
  card: 7,
  hiddenType: 'II',
  hiddenPattern: 'C',
  title: 'Benchmark: HP Protein Folding (2D, 50-mer) (v2)',
  description: `## HP Lattice Protein Folding (2D)

Fold a 50-residue HP protein sequence on a 2D square lattice to **maximize the number of H-H contacts** (non-consecutive hydrophobic residues on adjacent lattice sites).

### Problem Details

- **Sequence**: \`${HP_DEV_SEQUENCE}\` (50 residues, 26 H)
- **Lattice**: 2D square lattice
- **Scoring**: Negative contact count (more contacts = lower score = better)
- **Constraints**: Self-avoiding walk, consecutive residues must be lattice neighbors

### Submission Format

\`\`\`json
{ "steps": "RRUULLDD..." }
\`\`\`

A string of exactly **49 characters**, each one of R/L/U/D, encoding the move from residue i to residue i+1.

### Notes

- Best known for this sequence via greedy random-restart (100K trials): -19 contacts (19 H-H contacts).
- Starts at position (0,0). R=right, L=left, U=up, D=down.

**Source**: Lau (1986)
`,
  tags: ['bioinformatics', 'protein-folding', 'lattice', 'optimization'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 1,
  // FIX 2026-04-06: Changed submission from coords to steps.
  // Steps are far easier for LLMs to generate reliably than 50 absolute coordinates.
  solutionSchema: '{ steps: string — 49 chars, each R/L/U/D }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: accepts steps string, expands to coords, then scores
  verifier: `
def evaluate(data):
    seq = "${HP_DEV_SEQUENCE}"
    n = len(seq)  # 48

    # Accept steps string (47 chars R/L/U/D) and expand to coordinates
    if "steps" in data:
        steps = str(data["steps"]).upper().strip()
        if len(steps) != n - 1:
            return float('inf')
        dir_map = {'R': (1,0), 'L': (-1,0), 'U': (0,1), 'D': (0,-1)}
        coords = [(0, 0)]
        x, y = 0, 0
        for ch in steps:
            if ch not in dir_map:
                return float('inf')
            dx, dy = dir_map[ch]
            x, y = x + dx, y + dy
            coords.append((x, y))
    elif "coords" in data:
        coords = [tuple(c) for c in data["coords"]]
    else:
        return float('inf')

    assert len(coords) == n, f"Expected {n} coordinates, got {len(coords)}"

    # Check self-avoidance
    if len(set(coords)) != n:
        return float('inf')

    # Check connectivity
    for i in range(n - 1):
        dx = abs(coords[i + 1][0] - coords[i][0])
        dy = abs(coords[i + 1][1] - coords[i][1])
        if dx + dy != 1:
            return float('inf')

    # Count H-H contacts
    pos_to_idx = {c: i for i, c in enumerate(coords)}
    contacts = 0
    for i in range(n):
        if seq[i] != 'H':
            continue
        x, y = coords[i]
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (x + ddx, y + ddy)
            j = pos_to_idx.get(neighbor, -1)
            if j > i + 1 and seq[j] == 'H':
                contacts += 1

    return float(-contacts)
`,

  // Hidden verifier: different 48-mer HP sequence, same steps-based interface
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    rng = np.random.RandomState(2026)
    hidden_seq_arr = rng.choice(['H', 'P'], size=48, p=[0.48, 0.52])
    seq = ''.join(hidden_seq_arr)
    n = len(seq)

    if "steps" in data:
        steps = str(data["steps"]).upper().strip()
        if len(steps) != n - 1:
            return float('inf')
        dir_map = {'R': (1,0), 'L': (-1,0), 'U': (0,1), 'D': (0,-1)}
        coords = [(0, 0)]
        x, y = 0, 0
        for ch in steps:
            if ch not in dir_map:
                return float('inf')
            dx, dy = dir_map[ch]
            x, y = x + dx, y + dy
            coords.append((x, y))
    elif "coords" in data:
        coords = [tuple(c) for c in data["coords"]]
    else:
        return float('inf')

    if len(set(coords)) != n:
        return float('inf')

    for i in range(n - 1):
        dx = abs(coords[i + 1][0] - coords[i][0])
        dy = abs(coords[i + 1][1] - coords[i][1])
        if dx + dy != 1:
            return float('inf')

    pos_to_idx = {c: i for i, c in enumerate(coords)}
    contacts = 0
    for i in range(n):
        if seq[i] != 'H':
            continue
        x, y = coords[i]
        for ddx, ddy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (x + ddx, y + ddy)
            j = pos_to_idx.get(neighbor, -1)
            if j > i + 1 and seq[j] == 'H':
                contacts += 1

    return float(-contacts)
`,
};

// ─── Card 8: Lennard-Jones Cluster (N=41) ────────────────────────────────────

export const LENNARD_JONES: BenchmarkChallengeDefinition = {
  id: 'bench-lj-n41',
  card: 8,
  hiddenType: 'II',
  hiddenPattern: 'B',
  title: 'Benchmark: Lennard-Jones Cluster (N=41) (v2)',
  description: `## Lennard-Jones Cluster Optimization

Place **41 atoms** in 3D space to **minimize the Lennard-Jones potential energy**.

### Problem Details

- **N**: 41 atoms (non-canonical — less studied than N=38 or N=40)
- **Potential**: Standard LJ: 4 · Σ_{i<j} (1/r_ij¹² - 1/r_ij⁶)
- **Scoring**: Total LJ energy (lower is better)
- **Dev evaluator**: Standard pairwise LJ potential

### Submission Format

\`\`\`json
{ "positions": [[x1, y1, z1], [x2, y2, z2], ...] }
\`\`\`

Array of exactly 41 three-element vectors (atom coordinates).

### Notes

- N=41 is chosen to reduce contamination vs the famous N=38 truncated octahedron.
- This is a CONDITIONAL task — requires pilot validation.
- Cambridge Cluster Database has reference energies for LJ clusters.

**Source**: Wales & Doye (1997)
`,
  tags: ['physics', 'chemistry', 'molecular-optimization', 'continuous-optimization'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 0.01,
  solutionSchema: '{ positions: float[41][3] — atom coordinates }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: standard LJ potential
  verifier: `
import numpy as np

# Sentinel for physically invalid submissions (atoms too close, causing
# runaway repulsion with energy >> 0). Returned instead of raising an
# exception so the platform can record and filter the result cleanly.
_INVALID_SENTINEL = float(1e30)

def evaluate(data):
    try:
        pos = np.array(data["positions"], dtype=np.float64)
    except (TypeError, ValueError, KeyError):
        return _INVALID_SENTINEL
    N = 41
    if pos.shape != (N, 3):
        return _INVALID_SENTINEL

    energy = 0.0
    for i in range(N):
        diffs = pos[i + 1:] - pos[i]
        r2 = np.sum(diffs ** 2, axis=1)
        r2 = np.maximum(r2, 1e-30)
        r6_inv = 1.0 / (r2 ** 3)
        r12_inv = r6_inv ** 2
        energy += np.sum(4.0 * (r12_inv - r6_inv))

    # Physical validity gate: a well-formed LJ cluster always has
    # negative total energy (net attractive). Positive energy means
    # atoms are in unphysical configurations (colliding or degenerate).
    if energy > 0:
        return _INVALID_SENTINEL

    return float(energy)
`,

  // Hidden verifier: LJ + Axilrod-Teller three-body correction
  hiddenVerifier: `
import numpy as np

_INVALID_SENTINEL = float(1e30)

def evaluate(data):
    try:
        pos = np.array(data["positions"], dtype=np.float64)
    except (TypeError, ValueError, KeyError):
        return _INVALID_SENTINEL
    N = 41
    if pos.shape != (N, 3):
        return _INVALID_SENTINEL

    # Standard pairwise LJ energy
    energy = 0.0
    for i in range(N):
        diffs = pos[i + 1:] - pos[i]
        r2 = np.sum(diffs ** 2, axis=1)
        r2 = np.maximum(r2, 1e-30)
        r6_inv = 1.0 / (r2 ** 3)
        r12_inv = r6_inv ** 2
        energy += np.sum(4.0 * (r12_inv - r6_inv))

    # Axilrod-Teller three-body correction (small coefficient)
    # V_AT = nu * (3*cos(a)*cos(b)*cos(c) + 1) / (r_ij * r_ik * r_jk)^3
    # Using nu = 0.01 (small enough to preserve ranking, large enough for signal)
    nu = 0.01
    at_energy = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = pos[j] - pos[i]
            d_ij = np.sqrt(np.sum(r_ij ** 2))
            if d_ij < 1e-10:
                continue
            for k in range(j + 1, N):
                r_ik = pos[k] - pos[i]
                r_jk = pos[k] - pos[j]
                d_ik = np.sqrt(np.sum(r_ik ** 2))
                d_jk = np.sqrt(np.sum(r_jk ** 2))
                if d_ik < 1e-10 or d_jk < 1e-10:
                    continue

                # Cosines of angles at each vertex
                cos_i = np.dot(r_ij, r_ik) / (d_ij * d_ik)
                cos_j = np.dot(-r_ij, r_jk) / (d_ij * d_jk)
                cos_k = np.dot(-r_ik, -r_jk) / (d_ik * d_jk)

                denom = (d_ij * d_ik * d_jk) ** 3
                if denom < 1e-30:
                    continue

                at_energy += nu * (3.0 * cos_i * cos_j * cos_k + 1.0) / denom

    total = energy + at_energy
    # Same physical validity gate as dev verifier
    if total > 0:
        return _INVALID_SENTINEL

    return float(total)
`,
};

// ─── Near-Core: Erdős Minimum Overlap ────────────────────────────────────────

export const ERDOS_OVERLAP: BenchmarkChallengeDefinition = {
  id: 'bench-erdos-overlap',
  card: 'near-core',
  hiddenType: 'I',
  hiddenPattern: 'B',
  title: 'Benchmark: Erdős Minimum Overlap (Upper Bound) (v2)',
  description: `## Erdős Minimum Overlap Problem

Find a measurable function f: [0,1] → [0,1] with ∫f = 1/2 that **minimizes the maximum overlap** between f and its translates.

### Problem Details

- **Goal**: Minimize the supremum of the cross-correlation functional.
- **Constraint**: f must integrate to 1/2 over [0,1], with values in [0,1].
- **Scoring**: Maximum of the overlap functional (lower is better).
- **Dev evaluator**: Discretization with N=1000 sample points.

### Submission Format

\`\`\`json
{ "values": [0.5, 0.3, 0.8, ...] }
\`\`\`

Array of at least 100 floats in [0, 1] representing f sampled uniformly on [0, 1].

### Notes

- Erdős (1955) asked this question. The answer is known to be strictly less than 1/4.
- AlphaEvolve (2025) improved the upper bound — direct comparison possible.
- LOW contamination: niche analysis/measure theory problem.

**Source**: Erdős (1955); AlphaEvolve (2025)
`,
  tags: ['analysis', 'measure-theory', 'optimization', 'number-theory'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 0.0001,
  solutionSchema: '{ values: float[n] — at least 100 samples of f on [0,1], each in [0,1] }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: N=1000 discretization
  verifier: `
import numpy as np

def evaluate(data):
    values = np.array(data["values"], dtype=np.float64)
    n = len(values)
    assert n >= 100, f"Need at least 100 samples, got {n}"
    assert np.all(values >= 0) and np.all(values <= 1), "Values must be in [0, 1]"

    # Normalize to integral = 1/2. Do NOT clip afterward.
    # Clipping destroys the normalization for sparse/spike submissions and enables
    # gaming: sparse values → scale ×K → clip to [0,1] → sparse indicator → near-zero
    # autocorrelation (artificially good). Without clipping: sparse values → scale ×K →
    # huge values → autocorrelation = K² × density >> 0.25 → terrible score, no gaming.
    integral = np.mean(values)
    if integral < 1e-10:
        return float('inf')
    values = values * (0.5 / integral)
    # Note: values may exceed 1.0 after scaling if submitted mean < 0.5.
    # This is intentional — it penalizes infeasible submissions geometrically.

    # Compute overlap functional on N=1000 discretization
    N_eval = 1000
    # Resample to N_eval points
    x_eval = np.linspace(0, 1, N_eval, endpoint=False)
    x_orig = np.linspace(0, 1, n, endpoint=False)
    f_eval = np.interp(x_eval, x_orig, values)

    max_overlap = 0.0
    for shift in range(1, N_eval):
        shifted = np.roll(f_eval, shift)
        overlap = np.mean(f_eval * shifted)
        max_overlap = max(max_overlap, overlap)

    return float(max_overlap)
`,

  // Hidden verifier: N=10000 discretization + adversarial test points
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    values = np.array(data["values"], dtype=np.float64)
    n = len(values)
    assert n >= 100, f"Need at least 100 samples, got {n}"
    assert np.all(values >= 0) and np.all(values <= 1), "Values must be in [0, 1]"

    # Normalize to integral = 1/2. Do NOT clip afterward.
    # Clipping destroys the normalization for sparse/spike submissions and enables
    # gaming: sparse values → scale ×K → clip to [0,1] → sparse indicator → near-zero
    # autocorrelation (artificially good). Without clipping: sparse values → scale ×K →
    # huge values → autocorrelation = K² × density >> 0.25 → terrible score, no gaming.
    integral = np.mean(values)
    if integral < 1e-10:
        return float('inf')
    values = values * (0.5 / integral)
    # Note: values may exceed 1.0 after scaling if submitted mean < 0.5.
    # This is intentional — it penalizes infeasible submissions geometrically.

    # Phase 1: N=10000 uniform discretization
    N_eval = 10000
    x_eval = np.linspace(0, 1, N_eval, endpoint=False)
    x_orig = np.linspace(0, 1, n, endpoint=False)
    f_eval = np.interp(x_eval, x_orig, values)

    max_overlap = 0.0
    best_shifts = []
    for shift in range(1, N_eval):
        shifted = np.roll(f_eval, shift)
        overlap = np.mean(f_eval * shifted)
        if overlap > max_overlap:
            max_overlap = overlap
            best_shifts = [shift]
        elif overlap > max_overlap - 0.001:
            best_shifts.append(shift)

    # Phase 2: adversarial refinement around top shifts
    N_fine = 50000
    x_fine = np.linspace(0, 1, N_fine, endpoint=False)
    f_fine = np.interp(x_fine, x_orig, values)

    for coarse_shift in best_shifts[:20]:
        # Convert coarse shift to fine resolution and search neighborhood
        center = int(coarse_shift * N_fine / N_eval)
        for delta in range(-50, 51):
            s = (center + delta) % N_fine
            shifted = np.roll(f_fine, s)
            overlap = np.mean(f_fine * shifted)
            max_overlap = max(max_overlap, overlap)

    return float(max_overlap)
`,
};

// ─── Card 9: MaxCut G(200, 0.3, seed=99) — Contamination Probe ───────────────

export const MAXCUT_SEED99: BenchmarkChallengeDefinition = {
  id: 'bench-maxcut-g200-s99',
  card: 9,
  hiddenType: 'I',
  hiddenPattern: 'A',
  title: 'Benchmark: MaxCut on G(200, 0.3, seed=99) — Contamination Probe',
  description: `## MaxCut on Erdős–Rényi Random Graph (Fresh Instance)

Find a partition of the 200 vertices of a fixed Erdős–Rényi random graph G(200, 0.3, seed=99) that **maximizes the number of edges crossing the partition**.

This is a fresh instance of the same G(200, 0.3) family used in Card 1 (seed=17). It is included as a **contamination probe**: if protocols score substantially higher on the canonical seed=17 instance than on this unseen seed=99 instance, that is evidence of training-data contamination for MaxCut on G(200, 0.3).

### Problem Details

- **Graph**: G(200, 0.3) generated with seed=99. Fixed and public.
- **Submission**: A binary partition of all 200 vertices (0 or 1 labels).
- **Scoring**: Number of edges from the evaluation set that cross the partition (higher is better).
- **Dev evaluator**: Scores on a public 80% edge subset (~4,776 of ~5,970 edges, selected with seed=199).

### Submission Format

\`\`\`json
{ "partition": [0, 1, 1, 0, ...] }
\`\`\`

Array of exactly 200 integers, each 0 or 1.

### Notes

- Random partition baseline ≈ 2,985 cut edges on the full graph.
- Good heuristic solutions reach 3,500+.
- Compare your score here against Card 1 (seed=17) to assess contamination.

**Source**: Karp (1972)
`,
  tags: ['graph-theory', 'combinatorics', 'optimization', 'np-hard', 'contamination-probe'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 1,
  solutionSchema: '{ partition: int[200] — EXACTLY 200 integers, each 0 or 1. }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: 80% public edge subset (seed=199)
  verifier: `
import numpy as np

def evaluate(data):
    partition = np.array(data["partition"], dtype=np.int64)
    n = 200
    assert len(partition) == n, f"Expected {n} labels, got {len(partition)}"
    assert np.all((partition == 0) | (partition == 1)), "Labels must be 0 or 1"

    # Generate fixed graph G(200, 0.3) with seed=99
    rng = np.random.RandomState(99)
    adj = (rng.random((n, n)) < 0.3).astype(np.int64)
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj, 1)

    # Select 80% public edge subset with seed=199
    edge_rows, edge_cols = np.where(adj == 1)
    rng2 = np.random.RandomState(199)
    mask = rng2.random(len(edge_rows)) < 0.8
    pub_rows = edge_rows[mask]
    pub_cols = edge_cols[mask]

    cut = int(np.sum(partition[pub_rows] != partition[pub_cols]))
    return float(cut)
`,

  // Hidden verifier: full graph (all edges)
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    partition = np.array(data["partition"], dtype=np.int64)
    n = 200
    assert len(partition) == n, f"Expected {n} labels, got {len(partition)}"
    assert np.all((partition == 0) | (partition == 1)), "Labels must be 0 or 1"

    # Generate fixed graph G(200, 0.3) with seed=99
    rng = np.random.RandomState(99)
    adj = (rng.random((n, n)) < 0.3).astype(np.int64)
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj, 1)

    edge_rows, edge_cols = np.where(adj == 1)
    cut = int(np.sum(partition[edge_rows] != partition[edge_cols]))
    return float(cut)
`,
};

// ─── Card 10: Graph Coloring G(80, 0.4) — Interaction-Friendly ───────────────

export const GRAPH_COLORING: BenchmarkChallengeDefinition = {
  id: 'bench-graph-coloring-g80',
  card: 10,
  hiddenType: 'I',
  hiddenPattern: 'A',
  title: 'Benchmark: Graph Coloring on G(80, 0.4) (v1)',
  description: `## Graph Coloring — Minimize Colors

Assign colors to the 80 vertices of a fixed random graph G(80, 0.4) such that **no two adjacent vertices share a color**, using **as few distinct colors as possible**.

### Problem Details

- **Graph**: G(80, 0.4) generated with seed=7. Fixed and public (80 vertices, ~1,264 edges).
- **Submission**: A color assignment — one non-negative integer per vertex.
- **Scoring**: Number of distinct colors used, IF the coloring is valid. Invalid colorings (any edge has same-color endpoints in the checked set) return ∞.
- **Dev evaluator**: Checks a random 70% subset of edges (seed=107). A coloring that satisfies only the visible 70% may violate the hidden 30%.
- **Goal**: Minimize the number of distinct colors in a fully valid coloring.

### Submission Format

\`\`\`json
{ "coloring": [0, 2, 1, 0, 3, ...] }
\`\`\`

Array of exactly 80 non-negative integers — one color index per vertex.

### Notes

- The chromatic number of this instance is approximately 10–12 (needs calibration).
- A greedy coloring from highest-to-lowest degree typically uses ~16–20 colors.
- **Multi-agent opportunity**: propose → identify-conflict-edges → repair cycles align naturally with MAgICoRe and Debate.
- **Proxy-overfit trap**: the dev evaluator only checks 70% of edges. Overfitting to the visible edges (e.g., deliberately using fewer colors while placing same-color pairs on invisible edges) will score well on dev but fail on hidden evaluation.

**Source**: Karp (1972)
`,
  tags: ['graph-theory', 'combinatorics', 'optimization', 'np-hard', 'constraint-satisfaction'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 1,
  solutionSchema: '{ coloring: int[80] — non-negative color index per vertex, exactly 80 integers }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: 70% of edges (seed=107)
  verifier: `
import numpy as np

def evaluate(data):
    coloring = data.get("coloring")
    if coloring is None or len(coloring) != 80:
        return float('inf')

    col_arr = np.array([int(c) for c in coloring], dtype=np.int64)
    if np.any(col_arr < 0):
        return float('inf')

    n = 80
    # Generate G(80, 0.4) with seed=7
    rng_graph = np.random.RandomState(7)
    adj = (rng_graph.random((n, n)) < 0.4).astype(np.int64)
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj, 1)

    edge_i, edge_j = np.where(adj == 1)
    num_edges = len(edge_i)

    # Select 70% of edges for dev check (seed=107)
    rng_dev = np.random.RandomState(107)
    dev_mask = rng_dev.random(num_edges) < 0.70
    dev_i = edge_i[dev_mask]
    dev_j = edge_j[dev_mask]

    # Check validity on dev edges
    if np.any(col_arr[dev_i] == col_arr[dev_j]):
        return float('inf')

    return float(len(set(col_arr.tolist())))
`,

  // Hidden verifier: all edges — exposes any coloring that overfits the proxy dev 70%
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    coloring = data.get("coloring")
    if coloring is None or len(coloring) != 80:
        return float('inf')

    col_arr = np.array([int(c) for c in coloring], dtype=np.int64)
    if np.any(col_arr < 0):
        return float('inf')

    n = 80
    # Same graph as dev — G(80, 0.4) with seed=7
    rng_graph = np.random.RandomState(7)
    adj = (rng_graph.random((n, n)) < 0.4).astype(np.int64)
    np.fill_diagonal(adj, 0)
    adj = np.triu(adj, 1)

    edge_i, edge_j = np.where(adj == 1)

    # Check ALL edges — no partial subset
    if np.any(col_arr[edge_i] == col_arr[edge_j]):
        return float('inf')

    return float(len(set(col_arr.tolist())))
`,
};

// ─── Card 11: TSP-50 — Mid-Difficulty Replacement ────────────────────────────

export const TSP_50: BenchmarkChallengeDefinition = {
  id: 'bench-tsp-50',
  card: 11,
  hiddenType: 'II',
  hiddenPattern: 'A',
  title: 'Benchmark: Traveling Salesman (N=50) (v1)',
  description: `## Traveling Salesman Problem (N=50)

Find the **shortest tour** through 50 cities in the [0, 1000]² domain.

### Problem Details

- **N**: 50 cities, positions generated deterministically from seed=7
- **Domain**: [0, 1000]²
- **Scoring**: Total Euclidean tour length including return to start (lower is better)
- **Dev evaluator**: Tour scored on public city coordinates

### Submission Format

\`\`\`json
{ "tour": [0, 17, 42, 3, ...] }
\`\`\`

A permutation of integers 0..49 representing the visit order.

### Notes

- Random tour baseline ≈ 35,000. Nearest-neighbor + 2-opt reaches ≈ 5,000–6,000 on this instance.
- Unlike TSP-100 (where all protocols floored at random-tour quality), this smaller instance is within the search range of LLM-based protocols.
- A different city set from TSP-100 (seed=7 vs seed=42) to test generalization.

**Source**: Dantzig, Fulkerson & Johnson (1954)
`,
  tags: ['combinatorics', 'optimization', 'np-hard', 'routing'],
  evaluationType: 'deterministic',
  scoringDirection: 'minimize',
  minImprovement: 1,
  solutionSchema: '{ tour: int[50] — permutation of 0..49 }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: exact city coordinates from seed=7
  verifier: `
import numpy as np

def evaluate(data):
    tour = np.array(data["tour"], dtype=np.int64)
    N = 50
    assert len(tour) == N, f"Expected {N} cities, got {len(tour)}"
    assert set(tour.tolist()) == set(range(N)), "Tour must be a permutation of 0..49"

    # Generate city coordinates from seed=7
    rng = np.random.RandomState(7)
    cities = rng.uniform(0, 1000, size=(N, 2))

    total = 0.0
    for i in range(N):
        a = cities[tour[i]]
        b = cities[tour[(i + 1) % N]]
        total += np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    return float(total)
`,

  // Hidden verifier: perturbed city coordinates (seed=13, ±3 units)
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    tour = np.array(data["tour"], dtype=np.int64)
    N = 50
    assert len(tour) == N, f"Expected {N} cities, got {len(tour)}"
    assert set(tour.tolist()) == set(range(N)), "Tour must be a permutation of 0..49"

    # Generate original city coordinates from seed=7
    rng = np.random.RandomState(7)
    cities = rng.uniform(0, 1000, size=(N, 2))

    # Apply hidden perturbation: ±3 units per coordinate (seed=13)
    rng2 = np.random.RandomState(13)
    perturbation = rng2.uniform(-3, 3, size=(N, 2))
    cities = cities + perturbation

    total = 0.0
    for i in range(N):
        a = cities[tour[i]]
        b = cities[tour[(i + 1) % N]]
        total += np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    return float(total)
`,
};

// ─── Card 12: Molecule QED ───────────────────────────────────────────────────

export const MOLECULE_QED: BenchmarkChallengeDefinition = {
  id: 'bench-molecule-qed',
  card: 12,
  hiddenType: 'II',
  hiddenPattern: 'B',
  title: 'Benchmark: Molecule Drug-Likeness (QED × SA) (v1)',
  description: `## Molecular Drug-Likeness Optimization

Design a drug-like molecule by providing a valid SMILES string that **maximizes the QED (Quantitative Estimate of Drug-likeness)** score.

### Problem Details

- **Objective**: Maximize QED, a composite drug-likeness score ∈ [0, 1] (Bickerton et al. 2012).
- **Scoring**: Dev evaluator returns plain QED computed via RDKit. Higher is better.
- **Constraint**: Submission must be a valid SMILES string parseable by RDKit.
- **Proxy-overfit trap**: The hidden evaluator applies QED × synthesizability favorability. Molecules that maximize QED by incorporating complex ring systems may score lower on the hidden evaluator if they are synthetically inaccessible.

### Submission Format

\`\`\`json
{ "smiles": "CC(=O)Oc1ccccc1C(=O)O" }
\`\`\`

A single valid SMILES string representing a drug-like molecule.

### Notes

- Aspirin (CC(=O)Oc1ccccc1C(=O)O): QED ≈ 0.61
- Ibuprofen (CC(C)Cc1ccc(C(C)C(=O)O)cc1): QED ≈ 0.62
- Caffeine (Cn1cnc2c1c(=O)n(c(=O)n2C)C): QED ≈ 0.55
- Best known QED = 1.0, but typical drug-like candidates score 0.6–0.9.
- Multi-agent critic–refine loops can iteratively improve QED, mimicking real CADD workflows.

**Source**: Bickerton et al. (2012)
`,
  tags: ['chemistry', 'drug-discovery', 'molecular-design', 'cadd'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 0.001,
  solutionSchema: '{ smiles: string — a valid RDKit-parseable SMILES string }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: plain QED score (RDKit)
  verifier: `
from rdkit import Chem
from rdkit.Chem import QED as _QED

def evaluate(data):
    smiles = data.get("smiles", "")
    if not isinstance(smiles, str) or not smiles.strip():
        return float('-inf')
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return float('-inf')
    try:
        return float(_QED.qed(mol))
    except Exception:
        return float('-inf')
`,

  // Hidden verifier: QED × synthesizability favorability (SA_Score)
  // Type II: different-but-related objective — proxy-overfit trap.
  // SA_Score (Ertl et al.): 1 = easy to synthesize, 10 = hard.
  // SA_favorable = (10 - SA_Score) / 9, clipped to [0, 1].
  // hidden = QED × SA_favorable.
  // A molecule with QED=0.9 but SA_Score=7 gets hidden = 0.9 × 0.33 = 0.30;
  // aspirin (QED=0.61, SA_Score=1.17) gets hidden = 0.61 × 0.98 = 0.60.
  hiddenVerifier: `
from rdkit import Chem
from rdkit.Chem import QED as _QED

def _sa_favorable(mol):
    """Return SA synthesizability favorability in [0, 1].

    Tries rdkit.Contrib.SA_Score; falls back to a Lipinski Ro5 proxy
    if the Contrib module is unavailable.
    """
    try:
        from rdkit.Contrib.SA_Score import sascorer
        sa_raw = sascorer.calculateScore(mol)
    except (ImportError, ModuleNotFoundError):
        # Lipinski Ro5 proxy: each violation adds ~1.5 to effective SA_Score
        from rdkit.Chem import Descriptors, rdMolDescriptors
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
        sa_raw = 2.0 + violations * 1.5  # 0 viol → 2.0, 4 viol → 8.0
    # SA_Score in [1, 10]; map to favorability in [0, 1]
    return max(0.0, min(1.0, (10.0 - sa_raw) / 9.0))

def evaluate(data):
    smiles = data.get("smiles", "")
    if not isinstance(smiles, str) or not smiles.strip():
        return float('-inf')
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return float('-inf')
    try:
        qed_score = float(_QED.qed(mol))
        sa_fav = _sa_favorable(mol)
        return qed_score * sa_fav
    except Exception:
        return float('-inf')
`,
};

// ─── Card 13: TFBind-10 ──────────────────────────────────────────────────────

export const TFBIND_10: BenchmarkChallengeDefinition = {
  id: 'bench-tfbind-10',
  card: 13,
  hiddenType: 'I',
  hiddenPattern: 'B',
  title: 'Benchmark: TF Binding Affinity — SPI1/PU.1 10-mer (v1)',
  description: `## Transcription Factor Binding Affinity Optimization

Design a 10-nucleotide DNA sequence (over {A, C, G, T}) that maximizes predicted binding affinity to the **SPI1 (PU.1)** transcription factor — a master regulator of haematopoiesis.

### Problem Details

- **Objective**: Maximize SPI1 binding affinity scored by a Position Probability Matrix (PPM) derived from JASPAR MA0080.
- **Submission**: Exactly 10 characters over {A, C, G, T}. Case-insensitive.
- **Scoring**: Log-odds score relative to uniform background (higher = better match), normalized to [0, 1].
- **Dev evaluator**: Scores the best 8-mer sliding window within the 10-mer against the core SPI1 PPM (canonical core motif positions only).
- **Proxy-overfit trap**: The hidden evaluator scores the full 10-mer against an extended PPM that also weights the flanking positions 0 and 9. A sequence optimized purely for the 8-mer core may have suboptimal flanking bases.

### Submission Format

\`\`\`json
{ "sequence": "AGGAAGTGAC" }
\`\`\`

Exactly 10 characters from {A, C, G, T}.

### Notes

- SPI1 canonical core motif: GGAA (ETS family).
- Extended site: ~AGGAAGTGA in genomic contexts.
- Sequence space is 4^10 = 1,048,576 — too large to saturate in 25 queries.
- Iterative refinement (VGS, chain protocols) is expected to outperform single-shot.

**Source**: JASPAR MA0080 (SPI1/PU.1, Homo sapiens); Fornes et al. (2020) NAR.
`,
  tags: ['genomics', 'transcription-factor', 'sequence-design', 'biology'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 0.001,
  solutionSchema: '{ sequence: string — exactly 10 characters from {A, C, G, T}, case-insensitive }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: best 8-mer sliding window scored against core SPI1 PPM.
  // Represents a "partial oracle" covering only the canonical core motif positions.
  // Proxy-overfit trap: agents that hill-climb the 8-mer core may ignore flanking positions
  // that the hidden evaluator weights differently.
  verifier: `
import numpy as np
import re

# SPI1/PU.1 core PPM (8 bp, simplified from JASPAR MA0080)
# Columns: A  C  G  T  (0-indexed)
# Core motif: -GGAA GT G-  (ETS family canonical)
_PPM_CORE = np.array([
    [0.18, 0.12, 0.57, 0.13],  # pos 0: G-bias (5' side of GG)
    [0.10, 0.08, 0.74, 0.08],  # pos 1: G-strong (first G of GG core)
    [0.08, 0.07, 0.76, 0.09],  # pos 2: G-strong (second G of GG core)
    [0.82, 0.05, 0.07, 0.06],  # pos 3: A-strong (first A of AA core)
    [0.78, 0.06, 0.08, 0.08],  # pos 4: A-strong (second A of AA core)
    [0.09, 0.08, 0.74, 0.09],  # pos 5: G (GGAAG extension)
    [0.09, 0.07, 0.09, 0.75],  # pos 6: T (GGAAGT)
    [0.28, 0.16, 0.40, 0.16],  # pos 7: G-bias (3' side)
], dtype=np.float64)
_PPM_CORE = np.clip(_PPM_CORE, 1e-4, 1.0)

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def _score_8mer(seq8):
    """Log-odds score for an 8-mer, normalized to [0, 1]."""
    total = sum(np.log(_PPM_CORE[i, BASES[b]] / 0.25) for i, b in enumerate(seq8))
    min_s = sum(np.log(np.min(_PPM_CORE[i]) / 0.25) for i in range(8))
    max_s = sum(np.log(np.max(_PPM_CORE[i]) / 0.25) for i in range(8))
    return float((total - min_s) / (max_s - min_s))

def evaluate(data):
    seq = data.get("sequence", "")
    if not isinstance(seq, str):
        return float('-inf')
    seq = seq.upper().strip()
    if not re.fullmatch(r'[ACGT]{10}', seq):
        return float('-inf')
    # Best of the 3 possible 8-mer windows within the 10-mer
    return max(_score_8mer(seq[i:i+8]) for i in range(3))
`,

  // Hidden verifier: full 10-mer scored against extended PPM including flanking positions.
  // Positions 0 and 9 have strong preferences (A at pos 0, A at pos 9) absent from the
  // dev core PPM — the proxy-overfit gap. A sequence that maximizes the 8-mer window but has
  // non-A flanking bases will score lower on the hidden evaluator.
  hiddenVerifier: `
import numpy as np
import re

# SPI1/PU.1 extended PPM (10 bp, full-context version)
# Adds calibrated flanking preferences for positions 0 and 9.
# Key proxy-overfit asymmetry: pos 0 strongly prefers A (5' purine context);
# pos 9 strongly prefers A (3' anchor); neither is visible in the dev core PPM.
_PPM_FULL = np.array([
    [0.72, 0.08, 0.12, 0.08],  # pos 0: A-strong (5' flanking — hidden only)
    [0.18, 0.12, 0.57, 0.13],  # pos 1: G-bias
    [0.10, 0.08, 0.74, 0.08],  # pos 2: G-strong
    [0.08, 0.07, 0.76, 0.09],  # pos 3: G-strong
    [0.82, 0.05, 0.07, 0.06],  # pos 4: A-strong
    [0.78, 0.06, 0.08, 0.08],  # pos 5: A-strong
    [0.09, 0.08, 0.74, 0.09],  # pos 6: G
    [0.09, 0.07, 0.09, 0.75],  # pos 7: T
    [0.28, 0.16, 0.40, 0.16],  # pos 8: G-bias
    [0.70, 0.10, 0.10, 0.10],  # pos 9: A-strong (3' flanking — hidden only)
], dtype=np.float64)
_PPM_FULL = np.clip(_PPM_FULL, 1e-4, 1.0)

BASES = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def _score_10mer(seq):
    """Log-odds score for the full 10-mer, normalized to [0, 1]."""
    total = sum(np.log(_PPM_FULL[i, BASES[b]] / 0.25) for i, b in enumerate(seq))
    min_s = sum(np.log(np.min(_PPM_FULL[i]) / 0.25) for i in range(10))
    max_s = sum(np.log(np.max(_PPM_FULL[i]) / 0.25) for i in range(10))
    return float((total - min_s) / (max_s - min_s))

def evaluate(data):
    seq = data.get("sequence", "")
    if not isinstance(seq, str):
        return float('-inf')
    seq = seq.upper().strip()
    if not re.fullmatch(r'[ACGT]{10}', seq):
        return float('-inf')
    return _score_10mer(seq)
`,
};

// ─── Card 14: Heilbronn Triangle n=12 ────────────────────────────────────────

export const HEILBRONN_N12: BenchmarkChallengeDefinition = {
  id: 'bench-heilbronn-n12',
  card: 14,
  hiddenType: 'I',
  hiddenPattern: 'A',
  title: 'Benchmark: Heilbronn Triangle Problem (n=12) (v1)',
  description: `## Heilbronn Triangle Problem

Place **12 points** in the unit square [0,1]² to **maximize the area of the smallest triangle** formed by any 3 of the 12 points.

### Problem Details

- **Objective**: Maximize the minimum triangle area over all C(12,3) = 220 triples.
- **Scoring**: Minimum area among a fixed random sample of 50 triples (seed=7).
- **Proxy-overfit trap**: The hidden evaluator checks all 220 triples. Hill-climbing only the 50 sampled triangles may unknowingly create near-zero-area configurations among the unsampled triples.

### Submission Format

\`\`\`json
{ "points": [[0.12, 0.34], [0.56, 0.78], ...] }
\`\`\`

Exactly 12 points, each a [x, y] pair. Points outside [0,1]² are silently clipped.

### Notes

- Best known lower bound for n=12: ~0.00631 (Comellas & Ozón 2004; confirmed by AlphaEvolve 2025).
- Random uniform placement baseline ≈ 0.0015.
- This is a classical open problem in combinatorial geometry — the exact maximum is unknown for most n.
- AlphaEvolve (arXiv:2511.02864) found improved lower bounds for n ≤ 22 using LLM-guided search.

**Sources**: Heilbronn (1950); Comellas & Ozón (2004); Georgiev et al. (2025) AlphaEvolve.
`,
  tags: ['combinatorics', 'discrete-geometry', 'optimization', 'alphaevolve'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 1e-6,
  solutionSchema: '{ points: [[x, y], ...] — exactly 12 points, each x,y ∈ [0,1] (clipped if out of range) }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: checks a fixed random 50-of-220 subset of triangles (seed=7).
  verifier: `
import numpy as np
import random
from itertools import combinations

def evaluate(data):
    pts_raw = data.get("points")
    if pts_raw is None:
        return float('-inf')
    try:
        pts = np.array(pts_raw, dtype=np.float64)
    except (ValueError, TypeError):
        return float('-inf')
    if pts.shape != (12, 2):
        return float('-inf')
    pts = np.clip(pts, 0.0, 1.0)

    random.seed(7)
    all_triples = list(combinations(range(12), 3))
    sample = random.sample(all_triples, 50)

    def tri_area(i, j, k):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        return abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0

    return float(min(tri_area(i, j, k) for i, j, k in sample))
`,

  // Hidden verifier: checks all 220 triples (the true objective).
  hiddenVerifier: `
import numpy as np
from itertools import combinations

def evaluate(data):
    pts_raw = data.get("points")
    if pts_raw is None:
        return float('-inf')
    try:
        pts = np.array(pts_raw, dtype=np.float64)
    except (ValueError, TypeError):
        return float('-inf')
    if pts.shape != (12, 2):
        return float('-inf')
    pts = np.clip(pts, 0.0, 1.0)

    def tri_area(i, j, k):
        v1 = pts[j] - pts[i]
        v2 = pts[k] - pts[i]
        return abs(v1[0] * v2[1] - v1[1] * v2[0]) / 2.0

    return float(min(tri_area(i, j, k)
                     for i, j, k in combinations(range(12), 3)))
`,
};

// ─── Card 15: Kissing Number in Dimension 11 ─────────────────────────────────

export const KISSING_11D: BenchmarkChallengeDefinition = {
  id: 'bench-kissing-11d',
  card: 15,
  hiddenType: 'I',
  hiddenPattern: 'B',
  title: 'Benchmark: Kissing Number in Dimension 11 (v1)',
  description: `## Kissing Number Problem (11 Dimensions)

Place as many unit spheres as possible in ℝ¹¹ such that each touches a central unit sphere at the origin. Maximize the kissing number K.

### Problem Details

- **Objective**: Maximize the count K of non-conflicting sphere centers in ℝ¹¹.
- **Constraints**: Each center must satisfy ||v|| = 2.0 ± 1e-6. All pairwise center-to-center distances must be ≥ 2.0.
- **Scoring**: Dev evaluator uses a looser distance tolerance (1.98) and checks 70% of pairs (seed=13) to encourage exploration.
- **Proxy-overfit trap**: Hidden evaluator applies exact tolerance (2.0 − 1e-6) to all pairs. Spheres packed slightly too close will be eliminated.

### Submission Format

\`\`\`json
{ "centers": [[x1, x2, ..., x11], ...] }
\`\`\`

List of vectors in ℝ¹¹. Each must satisfy ||v|| = 2.0 ± 1e-6 (off-shell centers are silently discarded).

### Notes

- The exact kissing number in 11D is unknown.
- Andreev (1999) gave a constructive 582-sphere packing.
- AlphaEvolve / EinsteinArena established **593** as the current best (2025).
- Greedy random baseline ≈ 200 spheres.

**Sources**: Andreev (1999); Georgiev et al. (2025) AlphaEvolve arXiv:2511.02864.
`,
  tags: ['sphere-packing', 'discrete-geometry', 'optimization', 'alphaevolve'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 1,
  solutionSchema: '{ centers: [[x1,...,x11], ...] — list of 11-dim vectors; each must have ||v|| = 2.0 ± 1e-6 }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier: loose tolerance (1.98) + 70% pair sampling (seed=13).
  // Encourages packing more spheres by being forgiving of small distance violations.
  verifier: `
import numpy as np

def evaluate(data):
    raw = data.get("centers")
    if raw is None:
        return 0
    try:
        centers = np.array(raw, dtype=np.float64)
    except (ValueError, TypeError):
        return 0
    if centers.ndim != 2 or centers.shape[1] != 11:
        return 0

    # Keep only on-shell centers (||v|| = 2.0 ± 1e-6)
    norms = np.linalg.norm(centers, axis=1)
    centers = centers[np.abs(norms - 2.0) < 1e-6]
    n = len(centers)
    if n == 0:
        return 0

    # Sample 70% of all pairs (seed=13)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    k = max(1, int(0.70 * len(all_pairs)))
    rng = np.random.default_rng(13)
    sampled_idx = rng.choice(len(all_pairs), size=min(k, len(all_pairs)), replace=False)

    conflicts: set = set()
    for idx in sampled_idx:
        i, j = all_pairs[idx]
        if np.linalg.norm(centers[i] - centers[j]) < 1.98:
            conflicts.add(i)
            conflicts.add(j)

    return n - len(conflicts)
`,

  // Hidden verifier: exact tolerance (2.0 - 1e-6) applied to all pairs.
  // Greedily removes the later sphere (j) in each conflicting pair.
  hiddenVerifier: `
import numpy as np

def evaluate(data):
    raw = data.get("centers")
    if raw is None:
        return 0
    try:
        centers = np.array(raw, dtype=np.float64)
    except (ValueError, TypeError):
        return 0
    if centers.ndim != 2 or centers.shape[1] != 11:
        return 0

    # Keep only on-shell centers (||v|| = 2.0 ± 1e-6)
    norms = np.linalg.norm(centers, axis=1)
    centers = centers[np.abs(norms - 2.0) < 1e-6]
    n = len(centers)
    if n == 0:
        return 0

    # Greedy: remove j (the later sphere) for each conflicting pair
    removed: set = set()
    for i in range(n):
        for j in range(i + 1, n):
            if j not in removed and np.linalg.norm(centers[i] - centers[j]) < 2.0 - 1e-6:
                removed.add(j)

    return n - len(removed)
`,
};

// ─── Card 16: 3-AP-Free Subset of {1..100} ───────────────────────────────────

export const AP_FREE_100: BenchmarkChallengeDefinition = {
  id: 'bench-3ap-free-100',
  card: 16,
  hiddenType: 'I',
  hiddenPattern: 'A',
  title: 'Benchmark: Maximum 3-AP-Free Subset of {1,...,100} (v1)',
  description: `## Maximum 3-Arithmetic-Progression-Free Subset

Find the largest subset S ⊆ {1, 2, ..., 100} containing **no 3-term arithmetic progression** (no distinct a < b < c ∈ S with a + c = 2b).

### Problem Details

- **Objective**: Maximize |S| subject to S being AP-free.
- **Scoring**: Returns |S| if S is AP-free, else 0 (binary validity).
- **No proxy-overfit split**: Dev and hidden evaluators are identical — this is a pure discriminative task with no proxy to exploit.

### Submission Format

\`\`\`json
{ "subset": [1, 3, 5, 11, 22, ...] }
\`\`\`

A list of distinct integers from {1,...,100}. Duplicates and out-of-range values are silently ignored.

### Notes

- The AP-free set problem is central to additive combinatorics (Szemerédi, Roth, Behrend).
- Greedy construction baseline ≈ 7–14 elements.
- Best known construction for N=100: ~28 elements (Salem-Spencer style; verify against OEIS A003002).
- Kelley & Meka (FOCS 2023) established near-optimal asymptotic bounds (arXiv:2302.05461) but optimal sets for small N are not published — making this a genuine search task.

**Sources**: Roth (1953); Behrend (1946); Kelley & Meka (2023); OEIS A003002.
`,
  tags: ['additive-combinatorics', 'number-theory', 'combinatorics', 'optimization'],
  evaluationType: 'deterministic',
  scoringDirection: 'maximize',
  minImprovement: 1,
  solutionSchema: '{ subset: int[] — list of distinct integers from {1,...,100}; out-of-range values silently ignored }',
  experimental: true,
  benchmarkTrack: 'scientific',
  closesAt: '2026-12-31T23:59:59Z',

  // Dev verifier (= hidden verifier): binary AP-free check, return |S| or 0.
  // No dev/hidden split — single clean objective.
  verifier: `
from itertools import combinations

def evaluate(data):
    raw = data.get("subset", [])
    if not isinstance(raw, list):
        return 0
    subset = sorted(set(x for x in raw if isinstance(x, int) and 1 <= x <= 100))
    if len(subset) < 3:
        return len(subset)  # trivially AP-free
    for a, b, c in combinations(subset, 3):
        if a + c == 2 * b:
            return 0  # not AP-free
    return len(subset)
`,

  hiddenVerifier: `
from itertools import combinations

def evaluate(data):
    raw = data.get("subset", [])
    if not isinstance(raw, list):
        return 0
    subset = sorted(set(x for x in raw if isinstance(x, int) and 1 <= x <= 100))
    if len(subset) < 3:
        return len(subset)
    for a, b, c in combinations(subset, 3):
        if a + c == 2 * b:
            return 0
    return len(subset)
`,
};

// ─── All Benchmark Challenges ────────────────────────────────────────────────

/**
 * Active benchmark challenges for the v1.0 sweep.
 *
 * MEG analysis tasks (discriminative middle):
 *   Cards 1–4, 7 (near-core), 9–16
 *
 * Capability-floor exhibits (reported in §3, excluded from MEG):
 *   Card 6 (TSP-100), Card 8 (LJ-n41)
 *
 * Dropped (data exists but excluded entirely):
 *   Card 5 (Thomson) — representation mismatch
 *   Card 7-orig (HP Folding) — <5% validity rate
 *
 * New cards (2026-04-09): Cards 13–16 — TFBind-10, Heilbronn, Kissing 11D, 3-AP-Free
 */
export const ALL_BENCHMARK_CHALLENGES: BenchmarkChallengeDefinition[] = [
  MAXCUT,           // Card 1  — discriminative, proxy-overfit anchor
  CIRCLE_PACKING,   // Card 2  — discriminative, possible dev ceiling
  DIFFERENCE_BASES, // Card 3  — discriminative, brute-force / sample-friendly
  FLAT_POLYNOMIALS, // Card 4  — discriminative, VGS-friendly
  // THOMSON dropped — Card 5 — representation mismatch
  TSP_100,          // Card 6  — capability-floor exhibit (all Q=0 on dev)
  // HP_FOLDING dropped — Card 7-orig — <5% validity rate across all protocols
  LENNARD_JONES,    // Card 8  — capability-floor exhibit (all Q=0 on dev, OSS broken)
  ERDOS_OVERLAP,    // Near-core — discriminative, possible dev saturation
  MAXCUT_SEED99,    // Card 9  — contamination probe (fresh G(200,0.3) instance)
  GRAPH_COLORING,   // Card 10 — interaction-friendly, repair cycle, proxy-overfit trap
  TSP_50,           // Card 11 — mid-difficulty TSP (agents above floor at n=50)
  MOLECULE_QED,     // Card 12 — real CADD task, Type II hidden (QED × SA_Score proxy-overfit)
  TFBIND_10,        // Card 13 — genomics task, 8-mer vs 10-mer proxy-overfit (seeding pending calibration)
  HEILBRONN_N12,    // Card 14 — AlphaEvolve geometry, 50-of-220 vs all-220 triples proxy-overfit
  KISSING_11D,      // Card 15 — AlphaEvolve sphere packing, loose vs exact tolerance proxy-overfit
  AP_FREE_100,      // Card 16 — additive combinatorics, pure discriminative (no proxy-overfit split)
];
