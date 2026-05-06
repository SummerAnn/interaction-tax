#!/usr/bin/env python3
"""
compute-anchors.py — Resolve 3 provisional normalization anchors.

Computes s_ref for:
  1. MaxCut G(200,0.3,seed=17)  — greedy local search on full graph
  2. TSP-100 seed=42             — nearest-neighbor + 2-opt
  3. HP Folding 50-mer           — random-restart greedy + backtrack search

Run from neurips-experiments/:
  python3 src/tasks/compute-anchors.py

Outputs final s_ref values to update normalization-anchors.ts.
"""

import numpy as np
import random
import time
import sys

VERBOSE = True
def log(msg):
    if VERBOSE:
        print(msg, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1. MaxCut G(200, 0.3, seed=17)
# ─────────────────────────────────────────────────────────────────────────────

def maxcut_dev_score(partition, adj_80pct, rows, cols):
    """Score partition on 80% public edge subset (matches dev verifier exactly)."""
    p = np.array(partition, dtype=np.int64)
    return int(np.sum(p[rows] != p[cols]))

def maxcut_full_score(partition, adj):
    """Score partition on full graph."""
    p = np.array(partition, dtype=np.int64)
    edge_r, edge_c = np.where(adj == 1)
    return int(np.sum(p[edge_r] != p[edge_c]))

def compute_maxcut_anchor():
    log("\n=== MaxCut G(200, 0.3, seed=17) ===")
    n = 200

    # Exactly match dev verifier
    rng = np.random.RandomState(17)
    adj_full = (rng.random((n, n)) < 0.3).astype(np.int64)
    np.fill_diagonal(adj_full, 0)
    adj_full = np.triu(adj_full, 1)

    edge_rows, edge_cols = np.where(adj_full == 1)
    total_edges = len(edge_rows)
    log(f"  Full graph edges: {total_edges}")

    # 80% public subset (dev verifier, seed=71)
    rng2 = np.random.RandomState(71)
    mask = rng2.random(total_edges) < 0.8
    pub_rows = edge_rows[mask]
    pub_cols = edge_cols[mask]
    log(f"  Public 80% subset edges: {len(pub_rows)}")

    # Build adjacency lists for local search
    adj_list = [[] for _ in range(n)]
    for r, c in zip(edge_rows, edge_cols):
        adj_list[r].append(c)
        adj_list[c].append(r)

    def local_search_maxcut(partition):
        """Greedy local search: flip vertex if it improves full-graph cut."""
        p = list(partition)
        improved = True
        while improved:
            improved = False
            for v in range(n):
                # Count neighbors in same vs different partition
                same = sum(1 for u in adj_list[v] if p[u] == p[v])
                diff = len(adj_list[v]) - same
                if same > diff:  # flipping v improves cut
                    p[v] = 1 - p[v]
                    improved = True
        return p

    best_dev = 0
    best_full = 0
    n_restarts = 500

    start = time.time()
    for trial in range(n_restarts):
        # Random initialization
        init = list(np.random.randint(0, 2, size=n))
        p = local_search_maxcut(init)

        dev_score = maxcut_dev_score(p, None, pub_rows, pub_cols)
        full_score = maxcut_full_score(p, adj_full)

        if dev_score > best_dev:
            best_dev = dev_score
            best_full = full_score
            log(f"  [{trial+1}/{n_restarts}] New best: dev={best_dev} full={best_full} ({time.time()-start:.1f}s)")

    log(f"\n  RESULT MaxCut:")
    log(f"    s_ref (dev 80% subset) = {best_dev}")
    log(f"    s_ref (full graph)     = {best_full}")
    log(f"    s_base (pilot random)  = 2445")
    log(f"    Scale: dev/full = {best_dev/best_full:.3f} (expected ~0.80)")
    return best_dev, best_full


# ─────────────────────────────────────────────────────────────────────────────
# 2. TSP-100, seed=42, domain [0,1000]²
# ─────────────────────────────────────────────────────────────────────────────

def tour_length(tour, cities):
    n = len(tour)
    total = 0.0
    for i in range(n):
        a = cities[tour[i]]
        b = cities[tour[(i + 1) % n]]
        total += np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return total

def nearest_neighbor_tour(start, cities):
    n = len(cities)
    visited = [False] * n
    tour = [start]
    visited[start] = True
    for _ in range(n - 1):
        last = tour[-1]
        best_d = float('inf')
        best_j = -1
        for j in range(n):
            if not visited[j]:
                d = np.sqrt((cities[last][0] - cities[j][0])**2 +
                            (cities[last][1] - cities[j][1])**2)
                if d < best_d:
                    best_d = d
                    best_j = j
        tour.append(best_j)
        visited[best_j] = True
    return tour

def two_opt(tour, cities, max_iter=500):
    n = len(tour)
    t = list(tour)
    improved = True
    iterations = 0
    while improved and iterations < max_iter:
        improved = False
        iterations += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if j == n - 1 and i == 0:
                    continue
                # Compute gain of reversing t[i+1..j]
                a, b = t[i], t[i + 1]
                c, d = t[j], t[(j + 1) % n]
                d_before = (np.sqrt((cities[a][0]-cities[b][0])**2+(cities[a][1]-cities[b][1])**2) +
                            np.sqrt((cities[c][0]-cities[d][0])**2+(cities[c][1]-cities[d][1])**2))
                d_after  = (np.sqrt((cities[a][0]-cities[c][0])**2+(cities[a][1]-cities[c][1])**2) +
                            np.sqrt((cities[b][0]-cities[d][0])**2+(cities[b][1]-cities[d][1])**2))
                if d_after < d_before - 1e-10:
                    t[i+1:j+1] = t[i+1:j+1][::-1]
                    improved = True
    return t

def compute_tsp_anchor():
    log("\n=== TSP-100, seed=42 ===")
    n = 100

    # Exactly match dev verifier
    rng = np.random.RandomState(42)
    cities = rng.uniform(0, 1000, size=(n, 2))

    best_length = float('inf')
    best_tour = None
    n_starts = 100  # try all 100 starting cities

    start = time.time()
    for s in range(n_starts):
        tour = nearest_neighbor_tour(s, cities)
        tour = two_opt(tour, cities)
        length = tour_length(tour, cities)
        if length < best_length:
            best_length = length
            best_tour = list(tour)
            log(f"  [start={s}] New best tour: {best_length:.2f} ({time.time()-start:.1f}s)")

    log(f"\n  RESULT TSP-100:")
    log(f"    s_ref (dev, NN+2opt best of {n_starts} starts) = {best_length:.2f}")
    log(f"    s_base (pilot honest random) = 50835")
    log(f"    Note: LKH3 typically finds 5-15% shorter tours than NN+2opt.")
    log(f"    Estimated LKH3 s_ref ≈ {best_length * 0.92:.0f}–{best_length * 0.97:.0f}")
    return best_length, best_tour


# ─────────────────────────────────────────────────────────────────────────────
# 3. HP Folding — 50-mer
# ─────────────────────────────────────────────────────────────────────────────

HP_SEQ = "HPHPPHHPHPPHPHHPPHPHPHHPPHHPHPPHPHHPPHPHPHHPPHHPHH"
DIRS = [(1,0),(-1,0),(0,1),(0,-1)]
DIR_NAMES = {(1,0):'R',(-1,0):'L',(0,1):'U',(0,-1):'D'}

def count_contacts(positions, seq):
    n = len(positions)
    pos_to_idx = {p: i for i, p in enumerate(positions)}
    contacts = 0
    for i, (x, y) in enumerate(positions):
        if seq[i] == 'H':
            for dx, dy in DIRS:
                nb = (x + dx, y + dy)
                if nb in pos_to_idx:
                    j = pos_to_idx[nb]
                    if j > i + 1 and seq[j] == 'H':
                        contacts += 1
    return contacts

def greedy_fold(seq, rng_local):
    """One greedy fold attempt with randomized tie-breaking."""
    n = len(seq)
    positions = [(0, 0)]
    occupied = {(0, 0): 0}
    steps = []

    for i in range(1, n):
        valid = []
        for dx, dy in DIRS:
            nx, ny = positions[-1][0] + dx, positions[-1][1] + dy
            if (nx, ny) not in occupied:
                valid.append((dx, dy, nx, ny))
        if not valid:
            return None, None  # stuck

        # Greedy: if next residue is H, prefer moves that maximize H-H adjacency
        if seq[i] == 'H':
            scored = []
            for dx, dy, nx, ny in valid:
                score = 0
                for ddx, ddy in DIRS:
                    nb = (nx + ddx, ny + ddy)
                    if nb in occupied and seq[occupied[nb]] == 'H' and occupied[nb] < i - 1:
                        score += 1
                scored.append((score, dx, dy, nx, ny))
            max_score = max(s for s, *_ in scored)
            best = [(dx, dy, nx, ny) for s, dx, dy, nx, ny in scored if s == max_score]
        else:
            best = [(dx, dy, nx, ny) for dx, dy, nx, ny in valid]

        # Randomized choice among tied best
        dx, dy, nx, ny = best[rng_local.randint(0, len(best) - 1)]
        positions.append((nx, ny))
        occupied[(nx, ny)] = i
        steps.append((dx, dy))

    return positions, steps

def backtrack_fold(seq, max_depth=50, time_limit=30.0):
    """Backtracking search for maximum contacts with time limit."""
    n = len(seq)
    best_contacts = [0]
    best_positions = [None]
    start_t = time.time()
    calls = [0]

    def dfs(pos, occupied, depth):
        if time.time() - start_t > time_limit:
            return
        calls[0] += 1

        if depth == n:
            c = count_contacts(pos, seq)
            if c > best_contacts[0]:
                best_contacts[0] = c
                best_positions[0] = list(pos)
                log(f"    Backtrack found {c} contacts at depth={depth} ({calls[0]} calls, {time.time()-start_t:.1f}s)")
            return

        x, y = pos[-1]
        for dx, dy in DIRS:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in occupied:
                pos.append((nx, ny))
                occupied.add((nx, ny))
                dfs(pos, occupied, depth + 1)
                if time.time() - start_t > time_limit:
                    return
                pos.pop()
                occupied.discard((nx, ny))

    dfs([(0,0)], {(0,0)}, 1)
    return best_contacts[0], best_positions[0]

def compute_hp_anchor():
    log(f"\n=== HP Folding 50-mer ===")
    log(f"  Sequence: {HP_SEQ}")
    log(f"  Length: {len(HP_SEQ)}")
    h_count = HP_SEQ.count('H')
    log(f"  H residues: {h_count} of {len(HP_SEQ)}")

    # Phase 1: random-restart greedy (fast, many trials)
    rng_local = random.Random(42)
    np_rng = np.random.RandomState(42)
    best_contacts = 0
    best_pos = None
    n_greedy = 100_000

    log(f"\n  Phase 1: {n_greedy} greedy random restarts...")
    start = time.time()
    for trial in range(n_greedy):
        pos, steps = greedy_fold(HP_SEQ, rng_local)
        if pos is None or len(pos) < len(HP_SEQ):
            continue
        c = count_contacts(pos, HP_SEQ)
        if c > best_contacts:
            best_contacts = c
            best_pos = pos
            log(f"    [{trial+1}] New best: {c} contacts ({time.time()-start:.1f}s)")
        if time.time() - start > 60.0:
            log(f"    Greedy time limit hit at trial {trial+1}")
            break

    log(f"  Phase 1 result: {best_contacts} contacts (greedy)")

    # Phase 2: backtracking for small n (might not finish for n=50, use as verification)
    # Only run backtracking if greedy found ≤ 5 contacts (likely stuck)
    if best_contacts <= 3:
        log(f"\n  Phase 2: backtracking search (30s time limit)...")
        bt_contacts, bt_pos = backtrack_fold(HP_SEQ, time_limit=30.0)
        if bt_contacts > best_contacts:
            best_contacts = bt_contacts
            best_pos = bt_pos
            log(f"  Phase 2 improved to: {best_contacts} contacts")
    else:
        log(f"  Phase 2: skipped (greedy found {best_contacts} which is promising)")

    # Verify the best solution
    if best_pos:
        verified = count_contacts(best_pos, HP_SEQ)
        log(f"\n  RESULT HP Folding:")
        log(f"    Best H-H contacts found: {verified}")
        log(f"    Score (= -contacts): {-verified}")
        log(f"    s_ref = {-verified}")
        log(f"    s_base = 0 (random SAW = 0 contacts)")
        log(f"    Note: This is a lower bound on optimal. True optimal may be higher.")
        log(f"    Steps: {''.join(DIR_NAMES[d] for d in [(p[0]-best_pos[i][0], p[1]-best_pos[i][1]) for i, p in enumerate(best_pos[1:])])}")
    else:
        log(f"  WARNING: No valid fold found!")
        verified = 0

    return -verified  # return as score (negative contacts)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    print("=" * 70)
    print("the benchmark — Normalization Anchor Calibration")
    print("=" * 70)

    # 1. MaxCut
    mc_dev, mc_full = compute_maxcut_anchor()

    # 2. TSP
    tsp_len, tsp_tour = compute_tsp_anchor()

    # 3. HP Folding
    hp_score = compute_hp_anchor()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANCHOR SUMMARY — update normalization-anchors.ts with these values")
    print("=" * 70)
    print(f"""
Task           | Anchor | Current   | Computed  | Action
---------------|--------|-----------|-----------|-----------------------------------
MaxCut (dev)   | s_ref  | 3500      | {mc_dev:<9} | Update if {mc_dev} > 3500 or adjust estimate
MaxCut (full)  | note   | —         | {mc_full:<9} | For paper: report both dev and full
TSP-100 NN+2opt| s_ref  | 7544      | {tsp_len:<9.0f} | Use this or LKH3 if available
HP Folding     | s_ref  | -9        | {hp_score:<9} | Update if computed < -9 (more contacts)
""")
    print("Note: NN+2opt for TSP is typically 3-8% above LKH3 optimal.")
    print(f"  NN+2opt: {tsp_len:.0f}")
    print(f"  LKH3 estimate: {tsp_len*0.93:.0f}–{tsp_len*0.97:.0f}")
    print("\nUpdate normalization-anchors.ts and mark all 3 anchors as resolved.")
