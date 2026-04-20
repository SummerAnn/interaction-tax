/**
 * External Validation Tasks — CO-Bench-inspired combinatorial optimization
 *
 * Three tasks from distinct optimization domains (routing, selection, balancing)
 * designed to test whether the coordination-diversity tradeoff replicates
 * on problems outside the Agent4Science-Bench suite.
 *
 * All verifiers are self-contained Python with numpy, deterministically seeded.
 * Scoring direction is 'maximize' throughout (negative costs where needed).
 */

import type { BenchmarkTask } from '../types.js';

// ── Task 1: Travelling Salesman (25 cities) ─────────────────────────────────

const TSP_VERIFIER = `
import numpy as np

def evaluate(data):
    rng = np.random.RandomState(42)
    cities = rng.uniform(0, 100, (25, 2))

    tour = np.array(data["tour"], dtype=int)
    n = 25
    assert len(tour) == n, f"Expected {n} cities in tour, got {len(tour)}"
    assert set(tour.tolist()) == set(range(n)), "Tour must be a permutation of 0..24"

    total = 0.0
    for i in range(n):
        d = cities[tour[i]] - cities[tour[(i + 1) % n]]
        total += float(np.sqrt(np.sum(d ** 2)))

    # Return negative distance so maximize = shorter tour
    return -total
`;

const TSP_TASK: BenchmarkTask = {
  id: 'ext-tsp-25',
  title: 'External: TSP on 25 Random Cities',
  track: 'scientific',
  prompt: `## Travelling Salesman Problem (25 Cities)

Find the shortest tour visiting all 25 cities exactly once and returning to the start.

### Problem Details

- **Cities**: 25 points in [0,100]² generated with numpy RandomState seed=42.
  To reproduce: \`rng = np.random.RandomState(42); cities = rng.uniform(0, 100, (25, 2))\`
- **Submission**: A permutation of integers 0–24 representing the visit order.
- **Scoring**: Negative total Euclidean tour length (higher/less negative = shorter tour).

### Submission Format

\`\`\`json
{ "tour": [0, 5, 12, 3, ...] }
\`\`\`

Array of exactly 25 integers — a permutation of 0 through 24.

### Notes

- Random tour baseline ≈ −900 to −1100.
- Good heuristic (nearest neighbor + 2-opt) can reach ≈ −450 to −550.
- Optimal tour for 25 cities is tractable but NP-hard in general.`,
  devVerifier: TSP_VERIFIER,
  solutionSchema: { format: '{ tour: int[25] — a permutation of integers 0 through 24 }' },
  scoringDirection: 'maximize',
  challengeId: 'ext-tsp-25',
};

// ── Task 2: Maximum Weight Independent Set (40 vertices) ────────────────────
// Highly composable: models find heavy non-adjacent vertices in different graph
// regions; the synthesizer can merge disjoint independent sets trivially.

const MWIS_VERIFIER = `
import numpy as np

def evaluate(data):
    weights = [52,44,91,70,9,33,59,46,55,55,53,25,11,42,40,7,39,51,66,72,44,82,26,34,6,20,8,38,58,72,64,20,25,55,26,88,5,77,26,19]
    edges = [(0,11),(0,12),(0,19),(0,23),(0,30),(1,5),(1,7),(1,11),(1,12),(1,19),(1,22),(1,24),(1,27),(1,37),(1,38),(2,6),(2,9),(2,18),(2,20),(2,28),(2,37),(3,6),(3,8),(3,10),(3,12),(3,13),(3,19),(3,29),(4,6),(4,7),(4,8),(4,20),(4,31),(5,8),(5,19),(5,21),(5,23),(5,32),(6,9),(6,16),(6,17),(6,18),(6,20),(6,23),(6,32),(6,34),(6,36),(7,10),(7,11),(7,15),(7,21),(7,29),(7,33),(8,12),(8,15),(8,20),(8,21),(8,26),(8,32),(8,38),(9,11),(9,14),(9,16),(9,17),(9,37),(9,39),(10,17),(10,35),(10,38),(11,14),(12,14),(12,15),(12,16),(12,34),(12,39),(13,16),(13,22),(13,28),(13,33),(14,20),(14,26),(14,29),(14,37),(15,16),(15,25),(15,34),(15,38),(16,29),(16,31),(16,38),(17,25),(17,29),(18,23),(18,25),(18,34),(18,36),(19,25),(19,26),(20,23),(20,27),(21,33),(22,26),(22,33),(23,24),(23,28),(23,29),(24,30),(24,36),(24,38),(24,39),(25,28),(25,31),(26,30),(26,32),(26,33),(26,34),(28,30),(29,30),(29,32),(29,39),(30,34),(31,32),(31,35),(32,34),(32,36),(33,34),(35,36),(37,39)]
    n = 40

    selected = np.array(data["selected"], dtype=int)
    assert len(selected) == n, f"Expected {n} entries, got {len(selected)}"
    assert np.all((selected == 0) | (selected == 1)), "Entries must be 0 or 1"

    # Check independence: no edge between selected vertices
    for i, j in edges:
        if selected[i] == 1 and selected[j] == 1:
            return -1000.0  # not independent

    return float(sum(weights[i] for i in range(n) if selected[i] == 1))
`;

const MWIS_TASK: BenchmarkTask = {
  id: 'ext-mwis-40',
  title: 'External: Max-Weight Independent Set (40 vertices)',
  track: 'scientific',
  prompt: `## Maximum Weight Independent Set (40 Vertices)

Find a subset of vertices with no edges between them (an independent set) that maximizes the total vertex weight.

### Graph Data

**Vertex weights** (vertex: weight):
0:52, 1:44, 2:91, 3:70, 4:9, 5:33, 6:59, 7:46, 8:55, 9:55, 10:53, 11:25, 12:11, 13:42, 14:40, 15:7, 16:39, 17:51, 18:66, 19:72, 20:44, 21:82, 22:26, 23:34, 24:6, 25:20, 26:8, 27:38, 28:58, 29:72, 30:64, 31:20, 32:25, 33:55, 34:26, 35:88, 36:5, 37:77, 38:26, 39:19

**Edges** (130 total):
0-11, 0-12, 0-19, 0-23, 0-30, 1-5, 1-7, 1-11, 1-12, 1-19, 1-22, 1-24, 1-27, 1-37, 1-38,
2-6, 2-9, 2-18, 2-20, 2-28, 2-37, 3-6, 3-8, 3-10, 3-12, 3-13, 3-19, 3-29,
4-6, 4-7, 4-8, 4-20, 4-31, 5-8, 5-19, 5-21, 5-23, 5-32,
6-9, 6-16, 6-17, 6-18, 6-20, 6-23, 6-32, 6-34, 6-36,
7-10, 7-11, 7-15, 7-21, 7-29, 7-33, 8-12, 8-15, 8-20, 8-21, 8-26, 8-32, 8-38,
9-11, 9-14, 9-16, 9-17, 9-37, 9-39, 10-17, 10-35, 10-38,
11-14, 12-14, 12-15, 12-16, 12-34, 12-39, 13-16, 13-22, 13-28, 13-33,
14-20, 14-26, 14-29, 14-37, 15-16, 15-25, 15-34, 15-38,
16-29, 16-31, 16-38, 17-25, 17-29, 18-23, 18-25, 18-34, 18-36,
19-25, 19-26, 20-23, 20-27, 21-33, 22-26, 22-33, 23-24, 23-28, 23-29,
24-30, 24-36, 24-38, 24-39, 25-28, 25-31, 26-30, 26-32, 26-33, 26-34,
28-30, 29-30, 29-32, 29-39, 30-34, 31-32, 31-35, 32-34, 32-36, 33-34, 35-36, 37-39

### Constraints

- Select a subset of the 40 vertices (binary vector).
- **No two selected vertices may share an edge** (independence constraint).
- Violations score −1000.

### Submission Format

\`\`\`json
{ "selected": [0, 1, 0, 1, ...] }
\`\`\`

Array of exactly 40 integers, each 0 or 1. selected[i]=1 means vertex i is in the independent set.

### Notes

- Greedy (heaviest-first, skip neighbors) reaches ≈ 500–550.
- Total vertex weight is 1713; the challenge is finding a heavy independent set.
- This is NP-hard in general.

### Strategy Hints

The graph is sparse (avg degree 6.5). Different regions of the graph may have heavy vertices that don't conflict. Finding complementary vertex groups in different neighborhoods is key.`,
  devVerifier: MWIS_VERIFIER,
  solutionSchema: { format: '{ selected: int[40] — EXACTLY 40 integers, each 0 or 1. selected[i]=1 means vertex i is in the independent set. }' },
  scoringDirection: 'maximize',
  challengeId: 'ext-mwis-40',
};

// ── Task 3: Multi-Depot TSP (5 depots × 8 cities) ────────────────────────────
// Composable counterpart to ext-tsp-25: same routing domain but the 5 depots
// are INDEPENDENT sub-problems. Diversity should help because the synthesizer
// can pick the best route per depot from different agents' solutions.

const MDTSP_VERIFIER = `
import numpy as np

def evaluate(data):
    rng = np.random.RandomState(99)
    K = 5   # depots
    N = 8   # cities per depot

    all_cities = [rng.uniform(0, 100, (N, 2)) for _ in range(K)]

    routes = data["routes"]
    assert len(routes) == K, f"Expected {K} routes, got {len(routes)}"

    total = 0.0
    for d in range(K):
        route = np.array(routes[d], dtype=int)
        assert len(route) == N, f"Depot {d}: expected {N} cities, got {len(route)}"
        assert set(route.tolist()) == set(range(N)), f"Depot {d}: route must be permutation of 0..{N-1}"

        cities = all_cities[d]
        for i in range(N):
            diff = cities[route[i]] - cities[route[(i + 1) % N]]
            total += float(np.sqrt(np.sum(diff ** 2)))

    return -total
`;

const MDTSP_TASK: BenchmarkTask = {
  id: 'ext-mdtsp-40',
  title: 'External: Multi-Depot TSP (5×8 cities)',
  track: 'scientific',
  prompt: `## Multi-Depot Travelling Salesman Problem (5 Depots × 8 Cities)

Find the shortest tour for EACH of 5 independent depots, each with 8 cities.

### Key Property

**The 5 depots are completely independent.** Each depot has its own set of 8 cities. The total score is the sum of 5 separate tour lengths. You can optimize each depot's route independently.

### Problem Details

- **Depots**: 5 independent depots, each with 8 cities in [0,100]².
- **Generation**: \`rng = np.random.RandomState(99); all_cities = [rng.uniform(0, 100, (8, 2)) for _ in range(5)]\`
- **Depot 0 cities**: (67.23, 48.81), (82.55, 3.14), (80.80, 56.56), (29.76, 4.67), (99.06, 0.68), (76.98, 74.68), (37.74, 49.41), (92.89, 39.55)
- **Depot 1 cities**: (35.76, 5.89), (31.00, 72.56), (75.18, 32.74), (37.22, 79.17), (78.98, 20.92), (13.10, 60.02), (80.62, 81.37), (93.02, 68.24)
- **Depot 2 cities**: (95.33, 43.93), (53.59, 0.85), (9.74, 44.61), (11.64, 76.39), (33.69, 85.46), (46.14, 15.51), (56.20, 20.85), (14.19, 21.73)
- **Depot 3 cities**: (78.27, 65.47), (64.17, 43.28), (68.12, 73.22), (57.40, 41.92), (1.51, 95.13), (77.68, 98.86), (75.77, 54.70), (78.98, 75.48)
- **Depot 4 cities**: (99.39, 64.87), (45.83, 36.20), (49.17, 56.61), (10.91, 19.54), (7.45, 90.16), (14.97, 18.72), (52.04, 18.87), (67.00, 56.97)

### Scoring

Negative total Euclidean tour length across ALL 5 depots (higher/less negative = shorter total).

### Submission Format

\`\`\`json
{ "routes": [[0, 3, 5, ...], [2, 0, 4, ...], ...] }
\`\`\`

Array of exactly 5 routes. Each route is a permutation of integers 0 through 7 (the visit order for that depot's 8 cities).

### Notes

- Random baseline ≈ −1650 to −2000 (total across all depots).
- Good heuristic (nearest neighbor + 2-opt per depot) ≈ −1200 to −1400.
- Since depots are independent, you can optimize each route separately.`,
  devVerifier: MDTSP_VERIFIER,
  solutionSchema: { format: '{ routes: int[5][8] — 5 routes, each a permutation of 0 through 7 }' },
  scoringDirection: 'maximize',
  challengeId: 'ext-mdtsp-40',
};

// ── Task 4: Weighted MAX-SAT (60 variables, 200 clauses) ─────────────────────
// Composable: variable-level recombination — take x_i from whichever agent's
// assignment satisfies the most weight for variable i's neighborhood.
// Every 0/1 assignment is valid (no catastrophic constraint failures).
// Sparse clause structure creates natural variable clusters → different agents
// reason about different variable neighborhoods with different strategies.

const MAXSAT_VERIFIER = `
import numpy as np

def evaluate(data):
    clauses = [[6,[28,51,-58]],[1,[-22,-26,56]],[7,[-14,23,-32]],[1,[-15,-46,-56]],[2,[-5,-6,-28]],[3,[-16,-29,-57]],[5,[-1,-8,32]],[7,[2,14,50]],[5,[15,18,32]],[1,[31,34,42]],[2,[-6,20,21]],[8,[2,-6,-15]],[6,[-18,43,60]],[7,[-15,-23]],[10,[-19,45,-53]],[9,[-15,-53,-56]],[1,[-2,-17,-53]],[9,[46,47,-59]],[1,[13,25,-32]],[4,[24,26]],[2,[-6,-40]],[8,[-20,-21,26]],[7,[35,-50,-53]],[8,[-3,8]],[5,[3,-45,-59]],[10,[15,45]],[6,[-31,44,-46]],[1,[-4,8,18]],[2,[19,47,-58]],[2,[22,-35]],[3,[-22,29]],[3,[52,57]],[10,[38,39,58]],[3,[-8,48]],[2,[-33,-50,57]],[5,[5,17,32]],[2,[11,-51,57]],[5,[6,18,-52]],[8,[-34,-36,-55]],[7,[-46,56]],[1,[20,-29]],[6,[-40,-41,54]],[9,[-11,41]],[8,[-6,15,53]],[4,[32,-42]],[10,[40,-49]],[6,[15,-27,35]],[9,[27,52]],[2,[1,-10,23]],[5,[-24,-35,56]],[5,[-46,52]],[10,[9,-39,-59]],[10,[1,12,-53]],[4,[-6,-8,35]],[2,[-17,52]],[1,[6,-31,-34]],[2,[-14,28,-41]],[3,[-9,12,-59]],[6,[-8,-51]],[7,[2,4,15]],[9,[-12,-24,-43]],[7,[-3,11]],[6,[-8,49]],[9,[-24,-34]],[1,[-29,32,50]],[5,[-20,31,51]],[1,[-1,5,37]],[10,[-34,-52]],[10,[19,-30,50]],[9,[12,-35,-57]],[4,[-20,52]],[8,[-6,19,-22]],[9,[-15,-49]],[2,[4,21]],[6,[7,-37,-56]],[5,[-3,4,-6]],[6,[-10,-37,60]],[3,[-55,-60]],[4,[4,-7,-37]],[3,[-7,-53]],[1,[-21,38,53]],[7,[-20,-32]],[8,[-16,31,42]],[10,[-3,37]],[10,[11,16,44]],[8,[10,19,58]],[10,[10,-23,48]],[2,[4,-39,54]],[4,[-27,-44]],[5,[7,-30,40]],[4,[7,26]],[4,[-15,-52]],[4,[16,-31,-42]],[7,[-3,-4,-25]],[4,[-7,49]],[4,[10,15]],[7,[8,26,44]],[10,[-39,47,56]],[3,[4,51,59]],[2,[-4,-11,55]],[5,[-4,20,-26]],[9,[-9,16,-46]],[8,[-13,17,41]],[2,[-2,15,47]],[5,[-9,38,-52]],[8,[-8,-22]],[6,[-1,-5,-52]],[9,[15,-41,47]],[3,[35,55]],[9,[11,-26,-50]],[5,[7,8,31]],[5,[8,-29,31]],[7,[-32,-60]],[7,[31,42,57]],[5,[4,-50,-54]],[2,[-2,-15,50]],[10,[-5,59,60]],[8,[16,-43]],[5,[1,-27,-59]],[5,[8,18]],[10,[-3,-20,-49]],[2,[-1,-8,-45]],[9,[-30,-38,-58]],[10,[-4,8,9]],[7,[19,60]],[8,[-11,-28,-47]],[1,[1,15,-39]],[10,[14,-46]],[6,[8,13,-58]],[1,[-9,-34]],[1,[-12,56,57]],[2,[-25,-35,41]],[8,[6,7,-39]],[7,[6,29,-43]],[8,[-21,-54]],[6,[2,37,-48]],[8,[13,-29,-46]],[6,[-18,42,43]],[2,[14,-22,-36]],[7,[-33,48,-53]],[2,[-11,17,-26]],[1,[32,44,-54]],[7,[15,-32,53]],[1,[12,-28,-34]],[10,[-47,51,-53]],[3,[42,59]],[4,[21,24,45]],[9,[-15,-26]],[1,[-16,34,37]],[1,[28,-45]],[7,[-22,-34,-49]],[5,[1,42]],[9,[-39,45,51]],[10,[-3,-8,27]],[8,[3,-22,46]],[1,[-13,-28]],[7,[2,-6,36]],[3,[-5,-18]],[6,[18,-28,43]],[8,[-5,17,32]],[1,[5,17,22]],[7,[-29,47,53]],[8,[6,51,-58]],[8,[20,51,-52]],[1,[-27,52,-55]],[7,[8,-31,49]],[7,[31,-51,-58]],[5,[9,-57]],[2,[21,51]],[9,[3,-19,48]],[7,[18,33,49]],[7,[-22,-47,-48]],[10,[-3,37]],[4,[22,-33,-53]],[2,[-26,-30,-46]],[10,[18,27,28]],[10,[6,15,18]],[6,[15,-23,40]],[10,[6,29,50]],[7,[-5,34,-37]],[6,[-8,51]],[4,[37,-48,-54]],[7,[25,32,-55]],[3,[32,-46,49]],[2,[-9,-51,-55]],[3,[-6,-29,49]],[3,[10,-56]],[5,[15,16,22]],[5,[13,21,29]],[2,[5,-33,-34]],[3,[1,40,50]],[4,[31,-46]],[2,[-7,-13,-38]],[8,[11,47]],[6,[-28,-37,42]],[3,[18,36,-48]],[2,[-1,2,-36]],[3,[-2,17]],[7,[1,6,-15]],[10,[22,-33,-34]]]
    n = 60
    assignment = np.array(data["assignment"], dtype=int)
    assert len(assignment) == n, f"Expected {n} variables, got {len(assignment)}"
    assert np.all((assignment == 0) | (assignment == 1)), "Values must be 0 or 1"

    score = 0.0
    for weight, lits in clauses:
        satisfied = False
        for l in lits:
            var_idx = abs(l) - 1  # 1-indexed to 0-indexed
            if l > 0 and assignment[var_idx] == 1:
                satisfied = True
                break
            if l < 0 and assignment[var_idx] == 0:
                satisfied = True
                break
        if satisfied:
            score += weight
    return score
`;

const MAXSAT_TASK: BenchmarkTask = {
  id: 'ext-maxsat-60',
  title: 'External: Weighted MAX-SAT (60 vars, 200 clauses)',
  track: 'scientific',
  prompt: `## Weighted Maximum Satisfiability (60 Variables, 200 Clauses)

Find a truth assignment for 60 Boolean variables that maximizes the total weight of satisfied clauses.

### Key Property

**Every assignment is valid and receives a meaningful score.** There are no hard constraints — your score is simply the sum of weights of all clauses your assignment satisfies. The goal is to satisfy as many high-weight clauses as possible.

### Composability Note

The clause-variable graph is sparse (each clause touches 2–3 of 60 variables). Variables in different regions of the graph can often be set independently. Different strategies (greedy by variable frequency, greedy by clause weight, local search from different starting points) may each find good settings for different variable subsets.

### Problem Data

60 Boolean variables: x1 through x60. Assignment: 1 = TRUE, 0 = FALSE.
200 weighted clauses (total weight = 1096). Format: (weight) literal OR literal [OR literal].
Positive literal (e.g., x5) is satisfied when x5=1. Negated literal (e.g., ~x5) is satisfied when x5=0.

1. (w=6) x28 OR x51 OR ~x58
2. (w=1) ~x22 OR ~x26 OR x56
3. (w=7) ~x14 OR x23 OR ~x32
4. (w=1) ~x15 OR ~x46 OR ~x56
5. (w=2) ~x5 OR ~x6 OR ~x28
6. (w=3) ~x16 OR ~x29 OR ~x57
7. (w=5) ~x1 OR ~x8 OR x32
8. (w=7) x2 OR x14 OR x50
9. (w=5) x15 OR x18 OR x32
10. (w=1) x31 OR x34 OR x42
11. (w=2) ~x6 OR x20 OR x21
12. (w=8) x2 OR ~x6 OR ~x15
13. (w=6) ~x18 OR x43 OR x60
14. (w=7) ~x15 OR ~x23
15. (w=10) ~x19 OR x45 OR ~x53
16. (w=9) ~x15 OR ~x53 OR ~x56
17. (w=1) ~x2 OR ~x17 OR ~x53
18. (w=9) x46 OR x47 OR ~x59
19. (w=1) x13 OR x25 OR ~x32
20. (w=4) x24 OR x26
21. (w=2) ~x6 OR ~x40
22. (w=8) ~x20 OR ~x21 OR x26
23. (w=7) x35 OR ~x50 OR ~x53
24. (w=8) ~x3 OR x8
25. (w=5) x3 OR ~x45 OR ~x59
26. (w=10) x15 OR x45
27. (w=6) ~x31 OR x44 OR ~x46
28. (w=1) ~x4 OR x8 OR x18
29. (w=2) x19 OR x47 OR ~x58
30. (w=2) x22 OR ~x35
31. (w=3) ~x22 OR x29
32. (w=3) x52 OR x57
33. (w=10) x38 OR x39 OR x58
34. (w=3) ~x8 OR x48
35. (w=2) ~x33 OR ~x50 OR x57
36. (w=5) x5 OR x17 OR x32
37. (w=2) x11 OR ~x51 OR x57
38. (w=5) x6 OR x18 OR ~x52
39. (w=8) ~x34 OR ~x36 OR ~x55
40. (w=7) ~x46 OR x56
41. (w=1) x20 OR ~x29
42. (w=6) ~x40 OR ~x41 OR x54
43. (w=9) ~x11 OR x41
44. (w=8) ~x6 OR x15 OR x53
45. (w=4) x32 OR ~x42
46. (w=10) x40 OR ~x49
47. (w=6) x15 OR ~x27 OR x35
48. (w=9) x27 OR x52
49. (w=2) x1 OR ~x10 OR x23
50. (w=5) ~x24 OR ~x35 OR x56
51. (w=5) ~x46 OR x52
52. (w=10) x9 OR ~x39 OR ~x59
53. (w=10) x1 OR x12 OR ~x53
54. (w=4) ~x6 OR ~x8 OR x35
55. (w=2) ~x17 OR x52
56. (w=1) x6 OR ~x31 OR ~x34
57. (w=2) ~x14 OR x28 OR ~x41
58. (w=3) ~x9 OR x12 OR ~x59
59. (w=6) ~x8 OR ~x51
60. (w=7) x2 OR x4 OR x15
61. (w=9) ~x12 OR ~x24 OR ~x43
62. (w=7) ~x3 OR x11
63. (w=6) ~x8 OR x49
64. (w=9) ~x24 OR ~x34
65. (w=1) ~x29 OR x32 OR x50
66. (w=5) ~x20 OR x31 OR x51
67. (w=1) ~x1 OR x5 OR x37
68. (w=10) ~x34 OR ~x52
69. (w=10) x19 OR ~x30 OR x50
70. (w=9) x12 OR ~x35 OR ~x57
71. (w=4) ~x20 OR x52
72. (w=8) ~x6 OR x19 OR ~x22
73. (w=9) ~x15 OR ~x49
74. (w=2) x4 OR x21
75. (w=6) x7 OR ~x37 OR ~x56
76. (w=5) ~x3 OR x4 OR ~x6
77. (w=6) ~x10 OR ~x37 OR x60
78. (w=3) ~x55 OR ~x60
79. (w=4) x4 OR ~x7 OR ~x37
80. (w=3) ~x7 OR ~x53
81. (w=1) ~x21 OR x38 OR x53
82. (w=7) ~x20 OR ~x32
83. (w=8) ~x16 OR x31 OR x42
84. (w=10) ~x3 OR x37
85. (w=10) x11 OR x16 OR x44
86. (w=8) x10 OR x19 OR x58
87. (w=10) x10 OR ~x23 OR x48
88. (w=2) x4 OR ~x39 OR x54
89. (w=4) ~x27 OR ~x44
90. (w=5) x7 OR ~x30 OR x40
91. (w=4) x7 OR x26
92. (w=4) ~x15 OR ~x52
93. (w=4) x16 OR ~x31 OR ~x42
94. (w=7) ~x3 OR ~x4 OR ~x25
95. (w=4) ~x7 OR x49
96. (w=4) x10 OR x15
97. (w=7) x8 OR x26 OR x44
98. (w=10) ~x39 OR x47 OR x56
99. (w=3) x4 OR x51 OR x59
100. (w=2) ~x4 OR ~x11 OR x55
101. (w=5) ~x4 OR x20 OR ~x26
102. (w=9) ~x9 OR x16 OR ~x46
103. (w=8) ~x13 OR x17 OR x41
104. (w=2) ~x2 OR x15 OR x47
105. (w=5) ~x9 OR x38 OR ~x52
106. (w=8) ~x8 OR ~x22
107. (w=6) ~x1 OR ~x5 OR ~x52
108. (w=9) x15 OR ~x41 OR x47
109. (w=3) x35 OR x55
110. (w=9) x11 OR ~x26 OR ~x50
111. (w=5) x7 OR x8 OR x31
112. (w=5) x8 OR ~x29 OR x31
113. (w=7) ~x32 OR ~x60
114. (w=7) x31 OR x42 OR x57
115. (w=5) x4 OR ~x50 OR ~x54
116. (w=2) ~x2 OR ~x15 OR x50
117. (w=10) ~x5 OR x59 OR x60
118. (w=8) x16 OR ~x43
119. (w=5) x1 OR ~x27 OR ~x59
120. (w=5) x8 OR x18
121. (w=10) ~x3 OR ~x20 OR ~x49
122. (w=2) ~x1 OR ~x8 OR ~x45
123. (w=9) ~x30 OR ~x38 OR ~x58
124. (w=10) ~x4 OR x8 OR x9
125. (w=7) x19 OR x60
126. (w=8) ~x11 OR ~x28 OR ~x47
127. (w=1) x1 OR x15 OR ~x39
128. (w=10) x14 OR ~x46
129. (w=6) x8 OR x13 OR ~x58
130. (w=1) ~x9 OR ~x34
131. (w=1) ~x12 OR x56 OR x57
132. (w=2) ~x25 OR ~x35 OR x41
133. (w=8) x6 OR x7 OR ~x39
134. (w=7) x6 OR x29 OR ~x43
135. (w=8) ~x21 OR ~x54
136. (w=6) x2 OR x37 OR ~x48
137. (w=8) x13 OR ~x29 OR ~x46
138. (w=6) ~x18 OR x42 OR x43
139. (w=2) x14 OR ~x22 OR ~x36
140. (w=7) ~x33 OR x48 OR ~x53
141. (w=2) ~x11 OR x17 OR ~x26
142. (w=1) x32 OR x44 OR ~x54
143. (w=7) x15 OR ~x32 OR x53
144. (w=1) x12 OR ~x28 OR ~x34
145. (w=10) ~x47 OR x51 OR ~x53
146. (w=3) x42 OR x59
147. (w=4) x21 OR x24 OR x45
148. (w=9) ~x15 OR ~x26
149. (w=1) ~x16 OR x34 OR x37
150. (w=1) x28 OR ~x45
151. (w=7) ~x22 OR ~x34 OR ~x49
152. (w=5) x1 OR x42
153. (w=9) ~x39 OR x45 OR x51
154. (w=10) ~x3 OR ~x8 OR x27
155. (w=8) x3 OR ~x22 OR x46
156. (w=1) ~x13 OR ~x28
157. (w=7) x2 OR ~x6 OR x36
158. (w=3) ~x5 OR ~x18
159. (w=6) x18 OR ~x28 OR x43
160. (w=8) ~x5 OR x17 OR x32
161. (w=1) x5 OR x17 OR x22
162. (w=7) ~x29 OR x47 OR x53
163. (w=8) x6 OR x51 OR ~x58
164. (w=8) x20 OR x51 OR ~x52
165. (w=1) ~x27 OR x52 OR ~x55
166. (w=7) x8 OR ~x31 OR x49
167. (w=7) x31 OR ~x51 OR ~x58
168. (w=5) x9 OR ~x57
169. (w=2) x21 OR x51
170. (w=9) x3 OR ~x19 OR x48
171. (w=7) x18 OR x33 OR x49
172. (w=7) ~x22 OR ~x47 OR ~x48
173. (w=10) ~x3 OR x37
174. (w=4) x22 OR ~x33 OR ~x53
175. (w=2) ~x26 OR ~x30 OR ~x46
176. (w=10) x18 OR x27 OR x28
177. (w=10) x6 OR x15 OR x18
178. (w=6) x15 OR ~x23 OR x40
179. (w=10) x6 OR x29 OR x50
180. (w=7) ~x5 OR x34 OR ~x37
181. (w=6) ~x8 OR x51
182. (w=4) x37 OR ~x48 OR ~x54
183. (w=7) x25 OR x32 OR ~x55
184. (w=3) x32 OR ~x46 OR x49
185. (w=2) ~x9 OR ~x51 OR ~x55
186. (w=3) ~x6 OR ~x29 OR x49
187. (w=3) x10 OR ~x56
188. (w=5) x15 OR x16 OR x22
189. (w=5) x13 OR x21 OR x29
190. (w=2) x5 OR ~x33 OR ~x34
191. (w=3) x1 OR x40 OR x50
192. (w=4) x31 OR ~x46
193. (w=2) ~x7 OR ~x13 OR ~x38
194. (w=8) x11 OR x47
195. (w=6) ~x28 OR ~x37 OR x42
196. (w=3) x18 OR x36 OR ~x48
197. (w=2) ~x1 OR x2 OR ~x36
198. (w=3) ~x2 OR x17
199. (w=7) x1 OR x6 OR ~x15
200. (w=10) x22 OR ~x33 OR ~x34

### Scoring

Your score = total weight of all satisfied clauses (out of 1096 maximum).

### Submission Format

\\\`\\\`\\\`json
{ "assignment": [1, 0, 0, 1, ...] }
\\\`\\\`\\\`

Array of exactly 60 integers, each 0 or 1. assignment[0] is x1, assignment[1] is x2, ..., assignment[59] is x60.

### Notes

- Random assignment baseline ≈ 920 (84% of total weight).
- Greedy heuristic (set each variable to satisfy the most remaining clause weight) ≈ 1030 (94%).
- The theoretical maximum may be less than 1096 if some clauses conflict.`,
  devVerifier: MAXSAT_VERIFIER,
  solutionSchema: { format: '{ assignment: int[60] — EXACTLY 60 integers, each 0 or 1. assignment[i-1] is the truth value for variable xi. }' },
  scoringDirection: 'maximize',
  challengeId: 'ext-maxsat-60',
};

// ── Task 5: Number Partitioning (40 numbers) ────────────────────────────────

const PARTITION_VERIFIER = `
import numpy as np

def evaluate(data):
    rng = np.random.RandomState(77)
    n = 40
    numbers = rng.randint(1, 1000, n)

    assignment = np.array(data["assignment"], dtype=int)
    assert len(assignment) == n, f"Expected {n} assignments, got {len(assignment)}"
    assert np.all((assignment == 0) | (assignment == 1)), "Assignments must be 0 or 1"

    sum0 = int(np.sum(numbers[assignment == 0]))
    sum1 = int(np.sum(numbers[assignment == 1]))

    # Negative absolute difference: maximize → minimize imbalance
    return -abs(sum0 - sum1)
`;

const PARTITION_TASK: BenchmarkTask = {
  id: 'ext-partition-40',
  title: 'External: Number Partitioning (40 numbers)',
  track: 'scientific',
  prompt: `## Number Partitioning Problem (40 Numbers)

Partition 40 positive integers into two subsets such that the difference between the subset sums is minimized.

### Problem Details

- **Numbers**: 40 integers in [1,999], generated with numpy RandomState seed=77.
  To reproduce: \`rng = np.random.RandomState(77); numbers = rng.randint(1, 1000, 40)\`
- **Submission**: A binary assignment of each number to subset 0 or subset 1.
- **Scoring**: Negative absolute difference of subset sums (higher = more balanced).
  A perfect partition scores 0 (or −1 if the total is odd).

### Submission Format

\`\`\`json
{ "assignment": [0, 1, 1, 0, ...] }
\`\`\`

Array of exactly 40 integers, each 0 or 1.

### Notes

- Random partition baseline ≈ −2000 to −4000.
- Greedy differencing (Karmarkar-Karp) achieves near-optimal.
- This is a classic NP-hard problem related to subset-sum.`,
  devVerifier: PARTITION_VERIFIER,
  solutionSchema: { format: '{ assignment: int[40] — EXACTLY 40 integers, each 0 or 1 }' },
  scoringDirection: 'maximize',
  challengeId: 'ext-partition-40',
};

// ── Exports ──────────────────────────────────────────────────────────────────

export const EXTERNAL_TASKS: BenchmarkTask[] = [TSP_TASK, MAXSAT_TASK, PARTITION_TASK];

/** Map from challengeId → Python verifier source (for LocalEvaluator). */
export const EXTERNAL_VERIFIERS = new Map<string, string>([
  ['ext-tsp-25', TSP_VERIFIER],
  ['ext-maxsat-60', MAXSAT_VERIFIER],
  ['ext-partition-40', PARTITION_VERIFIER],
]);
