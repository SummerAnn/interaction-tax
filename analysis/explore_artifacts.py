"""
Artifact exploration: score trajectory, solution diversity, reasoning content.
Run: python analysis/explore_artifacts.py
"""

import json
import re
import math
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results" / "full-v2"

# ── Load all run files ──────────────────────────────────────────────────────

runs = []
for f in sorted(RESULTS_DIR.glob("*.json")):
    if f.name == "results.json":
        continue
    with open(f) as fp:
        try:
            data = json.load(fp)
        except json.JSONDecodeError:
            print(f"  SKIP (bad json): {f.name}")
            continue
    if "tot" in data.get("protocolId", ""):
        continue
    runs.append(data)

print(f"Loaded {len(runs)} run files\n")

# ── Helpers ─────────────────────────────────────────────────────────────────

def protocol_family(pid):
    """Collapse variant suffixes to a base family for grouping."""
    for base in ["single-shot", "best-of-n", "self-refine", "homo-chain",
                 "cross-chain", "magicore", "debate", "moa-nosynth",
                 "moa-same-model", "moa", "vgs", "hpe", "tot"]:
        if pid.startswith(base):
            return base
    return pid

def extract_cot(raw: str) -> str | None:
    """Return chain-of-thought text before the JSON/code block, or None."""
    # Strip leading whitespace
    raw = raw.strip()
    # If it starts directly with { or [ it's pure JSON — no CoT
    if raw.startswith("{") or raw.startswith("["):
        return None
    # Find first JSON block marker
    json_start = raw.find("```")
    if json_start > 0:
        cot = raw[:json_start].strip()
        return cot if len(cot) > 20 else None
    # No code block — find first { and treat everything before as CoT
    brace_start = raw.find("{")
    if brace_start > 30:
        cot = raw[:brace_start].strip()
        return cot if len(cot) > 20 else None
    return None

def solution_values(sol_data: dict) -> list[float] | None:
    """Extract a flat numeric vector from a solution dict."""
    for key in ["values", "selection", "selected", "labels", "assignment",
                "positions", "tour", "sequence", "partition"]:
        if key in sol_data:
            v = sol_data[key]
            if isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                return [float(x) for x in v]
    # Try first list-valued key
    for v in sol_data.values():
        if isinstance(v, list) and len(v) > 1 and all(isinstance(x, (int, float)) for x in v):
            return [float(x) for x in v]
    return None

def cosine_dist(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - dot / (na * nb)

def hamming_frac(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 1.0
    return sum(1 for x, y in zip(a, b) if round(x) != round(y)) / len(a)

def is_binary(v: list[float]) -> bool:
    return all(x in (0.0, 1.0) for x in v)

# ── 1. SCORE TRAJECTORY ANALYSIS ────────────────────────────────────────────

print("=" * 60)
print("1. SCORE TRAJECTORY ANALYSIS")
print("   (for each protocol, does each step improve on the last?)")
print("=" * 60)

# Gather per-step score sequences, keyed by (family, task)
traj: dict[str, list[list[float]]] = defaultdict(list)
traj_dir: dict[str, str] = {}  # protocol family → scoring direction

for run in runs:
    pid = run.get("protocolId", "")
    fam = protocol_family(pid)
    artifacts = run.get("allArtifacts", [])
    if len(artifacts) < 2:
        continue
    scores = [a["devScore"] for a in artifacts]
    traj[fam].append(scores)
    # Record direction from first run we see
    if fam not in traj_dir:
        # Infer from task: if hiddenScore, check sign convention — just use raw scores
        traj_dir[fam] = "?"

# Report
for fam in sorted(traj.keys()):
    seqs = traj[fam]
    max_steps = max(len(s) for s in seqs)
    if max_steps < 2:
        continue

    print(f"\n  {fam}  ({len(seqs)} runs, up to {max_steps} steps)")

    for step in range(1, max_steps):
        n_improved, n_same, n_worse, n_total = 0, 0, 0, 0
        deltas = []
        for seq in seqs:
            if len(seq) <= step:
                continue
            n_total += 1
            delta = seq[step] - seq[step - 1]
            deltas.append(delta)
            if delta > 1e-9:
                n_improved += 1
            elif delta < -1e-9:
                n_worse += 1
            else:
                n_same += 1
        if n_total == 0:
            continue
        avg_delta = sum(deltas) / len(deltas)
        print(f"    step {step}→{step+1}: "
              f"improved={n_improved}/{n_total} ({100*n_improved//n_total}%)  "
              f"same={n_same}  worse={n_worse}  "
              f"avg_delta={avg_delta:+.4f}")

# ── 2. SOLUTION DIVERSITY ANALYSIS ──────────────────────────────────────────

print("\n\n" + "=" * 60)
print("2. SOLUTION DIVERSITY ANALYSIS")
print("   (within MoA runs: how different are parallel agent solutions?)")
print("=" * 60)

# Focus on protocols that have parallel independent generation
diversity_families = {"moa", "moa-nosynth", "moa-same-model", "best-of-n", "debate"}

diversity_by_task: dict[str, list[float]] = defaultdict(list)

for run in runs:
    pid = run.get("protocolId", "")
    fam = protocol_family(pid)
    if fam not in diversity_families:
        continue
    task = run.get("taskId", "?")
    artifacts = run.get("allArtifacts", [])
    if len(artifacts) < 2:
        continue

    vectors = []
    for a in artifacts:
        v = solution_values(a.get("solutionData", {}))
        if v:
            vectors.append(v)

    if len(vectors) < 2:
        continue

    dists = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            a, b = vectors[i], vectors[j]
            if len(a) != len(b):
                continue
            if is_binary(a) and is_binary(b):
                dists.append(hamming_frac(a, b))
            else:
                dists.append(cosine_dist(a, b))
    if dists:
        diversity_by_task[f"{fam}|{task}"].extend(dists)

print(f"\n  {'Protocol+Task':<45} {'avg_dist':>9} {'n_pairs':>8}")
print("  " + "-" * 65)
for key in sorted(diversity_by_task.keys()):
    dists = diversity_by_task[key]
    avg = sum(dists) / len(dists)
    proto, task = key.split("|", 1)
    label = f"{proto} / {task}"
    print(f"  {label:<45} {avg:>9.4f} {len(dists):>8}")

# Summary by family
print(f"\n  {'Family':<20} {'avg_dist':>9} {'n_runs':>8}")
print("  " + "-" * 40)
family_dists: dict[str, list[float]] = defaultdict(list)
for key, dists in diversity_by_task.items():
    fam = key.split("|")[0]
    family_dists[fam].extend(dists)
for fam in sorted(family_dists.keys()):
    d = family_dists[fam]
    print(f"  {fam:<20} {sum(d)/len(d):>9.4f} {len(d):>8}")

# ── 3. REASONING CONTENT (CoT EXTRACTION) ───────────────────────────────────

print("\n\n" + "=" * 60)
print("3. REASONING CONTENT ANALYSIS")
print("   (what fraction of artifacts have pre-JSON chain-of-thought?)")
print("=" * 60)

cot_by_family: dict[str, dict] = defaultdict(lambda: {"total": 0, "has_cot": 0, "cot_lengths": []})
cot_examples: dict[str, str] = {}

for run in runs:
    pid = run.get("protocolId", "")
    fam = protocol_family(pid)
    for a in run.get("allArtifacts", []):
        raw = a.get("rawOutput", "")
        if not raw:
            continue
        cot_by_family[fam]["total"] += 1
        cot = extract_cot(raw)
        if cot:
            cot_by_family[fam]["has_cot"] += 1
            cot_by_family[fam]["cot_lengths"].append(len(cot))
            if fam not in cot_examples:
                cot_examples[fam] = cot[:300]

print(f"\n  {'Family':<20} {'artifacts':>10} {'has_cot':>8} {'%':>6} {'avg_len':>8}")
print("  " + "-" * 56)
for fam in sorted(cot_by_family.keys()):
    d = cot_by_family[fam]
    pct = 100 * d["has_cot"] / d["total"] if d["total"] else 0
    avg_len = (sum(d["cot_lengths"]) / len(d["cot_lengths"])) if d["cot_lengths"] else 0
    print(f"  {fam:<20} {d['total']:>10} {d['has_cot']:>8} {pct:>5.0f}% {avg_len:>8.0f}")

print("\n\n  --- CoT EXAMPLES (first 300 chars per family) ---")
for fam, example in sorted(cot_examples.items()):
    print(f"\n  [{fam}]")
    # Indent the example
    for line in example.splitlines()[:6]:
        print(f"    {line}")
    print("    ...")
