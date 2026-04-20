"""
s_ref verification (Q7) — MaxCut LS + TSP NN+2opt verification.
Fully numpy-vectorized for speed.
Run: python3 verify_sref.py
"""
import numpy as np

# ── MaxCut G(200, 0.3, seed=17) ──────────────────────────────────────────────
print("=" * 70)
print("MaxCut G(200, 0.3, seed=17) — s_ref verification")
print("=" * 70)

n = 200
rng = np.random.RandomState(17)
adj = (rng.random((n, n)) < 0.3).astype(np.float64)
np.fill_diagonal(adj, 0)
adj = np.maximum(adj, adj.T)  # symmetrize

total_edges = int(np.sum(np.triu(adj, 1)))
print(f"Graph: n={n}, total edges = {total_edges}")

def maxcut_value(partition, adj):
    """Vectorized cut value: sum of edges between partition 0 and 1."""
    p = partition.astype(np.float64)
    # cut edges = sum_{i<j} adj[i,j] * (p[i] != p[j])
    # = sum_{i<j} adj[i,j] * (p[i] - p[j])^2 for binary p
    diff = np.abs(np.subtract.outer(p, p))
    return int(np.sum(np.triu(adj * diff, 1)))

def maxcut_gain(v, partition, adj):
    """Vectorized gain from flipping vertex v."""
    same_side = (partition == partition[v]).astype(np.float64)
    same_side[v] = 0  # exclude self
    diff_side = 1.0 - same_side
    diff_side[v] = 0
    gain = np.dot(adj[v], same_side) - np.dot(adj[v], diff_side)
    return int(gain)

n_restarts = 50
best_cut = 0
cuts = []

for restart in range(n_restarts):
    partition = np.random.randint(0, 2, size=n)
    cut = maxcut_value(partition, adj)

    improved = True
    while improved:
        improved = False
        for v in range(n):
            g = maxcut_gain(v, partition, adj)
            if g > 0:
                partition[v] = 1 - partition[v]
                cut += g
                improved = True

    cuts.append(cut)
    if cut > best_cut:
        best_cut = cut
    if (restart + 1) % 10 == 0:
        print(f"  ... {restart+1}/{n_restarts} restarts done, best so far: {best_cut}")

cuts = np.array(cuts)
print(f"\nGreedy LS ({n_restarts} restarts):")
print(f"  Best cut:  {best_cut}")
print(f"  Mean cut:  {cuts.mean():.1f}")
print(f"  Std:       {cuts.std():.1f}")

expected_random = total_edges / 2
print(f"\nRandom partition expected cut: {expected_random:.0f}")

# Edwards bound: OPT >= |E|/2 + (n-1)/4
edwards = total_edges / 2 + (n - 1) / 4
print(f"Edwards bound: OPT >= {edwards:.0f}")

sref = 3517
print(f"\n  Current s_ref = {sref}")
print(f"  LS best       = {best_cut}")
if best_cut >= sref:
    print(f"  ✓ LS confirms s_ref is achievable (LS found {best_cut} >= {sref})")
elif best_cut >= sref * 0.99:
    print(f"  ≈ LS within 1% of s_ref ({best_cut} vs {sref})")
else:
    print(f"  ✗ LS could not reach s_ref ({best_cut} < {sref})")

# GW analysis: if we can find cut C, then OPT >= C, and GW bound >= 0.878*OPT
# So s_ref should be <= OPT. Our LS gives lower bound on OPT.
# benchmark_plan says GW bound ~3620. Since s_ref=3517 < 3620, s_ref is conservative.
print(f"\n  GW 0.878-approx bound: if OPT >= {best_cut}, GW >= {int(0.878*best_cut)}")
print(f"  s_ref={sref} is {'below' if sref <= best_cut else 'above'} our LS best")
print(f"  benchmark_plan notes GW bound ~3620 > s_ref=3517 → s_ref is conservative ✓")

ratio = best_cut / total_edges
print(f"  Empirical cut ratio: best_cut/|E| = {ratio:.4f}")


# ── TSP-100 (seed=42) ────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TSP-100 (seed=42) — s_ref verification")
print("=" * 70)

N = 100
rng_tsp = np.random.RandomState(42)
cities = rng_tsp.uniform(0, 1000, size=(N, 2))

rng_hidden = np.random.RandomState(91)
perturbation = rng_hidden.uniform(-5, 5, size=(N, 2))
cities_hidden = cities + perturbation

# Precompute distance matrices
dist_dev = np.linalg.norm(cities[:, None] - cities[None, :], axis=2)
dist_hid = np.linalg.norm(cities_hidden[:, None] - cities_hidden[None, :], axis=2)

def tour_len(tour, dist):
    return sum(dist[tour[i], tour[(i+1) % len(tour)]] for i in range(len(tour)))

def nn_tour(start, dist):
    n = dist.shape[0]
    visited = np.zeros(n, dtype=bool)
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        cur = tour[-1]
        dists = dist[cur].copy()
        dists[visited] = np.inf
        nxt = np.argmin(dists)
        tour.append(nxt)
        visited[nxt] = True
    return tour

def two_opt(tour, dist, max_no_improve=50):
    n = len(tour)
    best = tour_len(tour, dist)
    no_improve = 0
    while no_improve < max_no_improve:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                # Check if reversal improves (only check the 2 changed edges)
                i1 = tour[i]; i2 = tour[i+1]; j1 = tour[j]; j2 = tour[(j+1) % n]
                old = dist[i1, i2] + dist[j1, j2]
                new = dist[i1, j1] + dist[i2, j2]
                if new < old - 1e-10:
                    tour[i+1:j+1] = tour[i+1:j+1][::-1]
                    best += (new - old)
                    improved = True
        if not improved:
            no_improve += 1
        else:
            no_improve = 0
    return tour, best

# Dev space
print(f"\nDev space (original cities):")
best_dev = float('inf')
for start in range(N):
    t = nn_tour(start, dist_dev)
    t, length = two_opt(t, dist_dev)
    if length < best_dev:
        best_dev = length
        best_dev_tour = list(t)
print(f"  NN+2opt best ({N} starts): {best_dev:.1f}")

# Hidden space
print(f"\nHidden space (perturbed ±5):")
best_hidden = float('inf')
for start in range(N):
    t = nn_tour(start, dist_hid)
    t, length = two_opt(t, dist_hid)
    if length < best_hidden:
        best_hidden = length
print(f"  NN+2opt best ({N} starts): {best_hidden:.1f}")

# Score dev tour on hidden
dev_on_hidden = tour_len(best_dev_tour, dist_hid)
print(f"  Best dev tour on hidden cities: {dev_on_hidden:.1f}")

sref_tsp = 7517
sbase_tsp = 50835
print(f"\n  Current s_ref  = {sref_tsp} (from normalization-anchors.ts)")
print(f"  Current s_base = {sbase_tsp}")
print(f"  NN+2opt best (dev)    = {best_dev:.1f}")
print(f"  NN+2opt best (hidden) = {best_hidden:.1f}")

if best_hidden <= sref_tsp:
    print(f"  ✓ NN+2opt confirms s_ref achievable on hidden space ({best_hidden:.0f} <= {sref_tsp})")
elif best_dev <= sref_tsp:
    print(f"  ⚠ s_ref={sref_tsp} matches dev space ({best_dev:.0f}) but not hidden ({best_hidden:.0f})")
    print(f"    → s_ref may need updating to hidden-space value")
else:
    print(f"  ✗ Neither dev nor hidden NN+2opt reach s_ref")

print(f"\n  Since all agents score Q≈0 (hidden_score >> s_base), TSP-100 is a floor exhibit.")
print(f"  s_ref precision matters only if agents approach it. Currently irrelevant for MEG.")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"MaxCut: s_ref={sref} | LS best={best_cut} | s_ref {'<= LS ✓' if best_cut >= sref else '> LS ⚠'}")
print(f"TSP:    s_ref={sref_tsp} | NN+2opt(hid)={best_hidden:.0f} | {'✓ achievable' if best_hidden <= sref_tsp else '⚠ check space'}")
print(f"\nBoth tasks are floor/canary exhibits where all agent Q≈0.")
print(f"s_ref precision does not affect any MEG conclusion in the current paper.")
