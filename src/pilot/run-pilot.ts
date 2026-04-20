#!/usr/bin/env npx tsx
/**
 * Agent4Science-Bench Pilot Runner
 *
 * Phase 1: Random baseline calibration (10 random submissions per task) → s_base
 * Phase 2: Single-shot calibration (10 LLM single-shot per task) → saturation check
 * Phase 3: VGS on conditional/near-core tasks (Thomson, LJ, Erdős) → go/no-go
 *
 * Uses OpenRouter for all LLM calls (routes to Claude, GPT-4o, Gemini).
 *
 * Usage:
 *   npx tsx src/pilot/run-pilot.ts [--phase=1|2|3] [--task=bench-maxcut-g200]
 */

import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

config({ path: resolve(__dirname, '../../../scibook/.env.local') });

const ADMIN_SECRET = process.env.ADMIN_SECRET || '';
const OPENROUTER_API_KEY = process.env.OPENROUTER_API_KEY || '';
const API_URL = 'https://agent4science.org';
const RESULTS_DIR = resolve(__dirname, '../../results/pilot-v1');

// OpenRouter model IDs (pinned versions)
const CLAUDE_MODEL = 'anthropic/claude-sonnet-4';

import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';

// ─── Task definitions (minimal for pilot) ─���─────────────────────────────────

interface PilotTask {
  id: string;
  challengeId: string;
  title: string;
  scoringDirection: 'maximize' | 'minimize';
  prompt: string;
  schemaHint: string;
  randomGenerator: () => Record<string, unknown>;
  conditional: boolean;
}

const TASKS: PilotTask[] = [
  {
    id: 'bench-maxcut-g200',
    challengeId: BENCHMARK_CHALLENGE_IDS['bench-maxcut-g200'],
    title: 'MaxCut G(200,0.3)',
    scoringDirection: 'maximize',
    prompt: 'Find a maximum cut of G(n=200, p=0.3, seed=17). Output ONLY this JSON object, nothing else:\n{"partition": [X,X,X,...,X]}\nwhere the array contains EXACTLY 200 integers, each 0 or 1, assigning each of the 200 nodes to side 0 or side 1. Do not include any explanation.',
    schemaHint: '{"partition": [0,1,0,1,...]} — EXACTLY 200 integers, each 0 or 1',
    randomGenerator: () => ({ partition: Array.from({ length: 200 }, () => Math.random() < 0.5 ? 1 : 0) }),
    conditional: false,
  },
  {
    id: 'bench-circle-packing-n20',
    challengeId: BENCHMARK_CHALLENGE_IDS['bench-circle-packing-n20'],
    title: 'Circle Packing N=20',
    scoringDirection: 'maximize',
    prompt: 'Pack 20 non-overlapping circles in [0,1]² to maximize sum of radii. Return JSON: {"circles": [[x,y,r], ...]} with 20 triples.',
    schemaHint: '{"circles": "float[20][3], each [x, y, radius]"}',
    randomGenerator: () => {
      const circles = [];
      for (let i = 0; i < 20; i++) {
        const r = 0.01 + Math.random() * 0.03;
        circles.push([r + Math.random() * (1 - 2 * r), r + Math.random() * (1 - 2 * r), r]);
      }
      return { circles };
    },
    conditional: false,
  },
  {
    id: 'bench-difference-bases',
    challengeId: BENCHMARK_CHALLENGE_IDS['bench-difference-bases'],
    title: 'Difference Bases',
    scoringDirection: 'minimize',
    prompt: 'Find a set B of non-negative integers that minimizes |B|²/v, where v is the largest integer such that every integer from 1 to v appears as a pairwise difference of elements in B. Output ONLY this JSON object with no other text:\n{"set": [0, 1, 3, ...]}\nThe set must contain non-negative integers with maximum value ≤ 2000. Always include 0. A good starting point: B={0,1,2,3} gives v=3, score=16/3≈5.3. Try to beat that.',
    schemaHint: '{"set": [0, 1, 3, ...]} — non-negative integers ≤ 2000, include 0. OUTPUT ONLY JSON.',
    randomGenerator: () => {
      const size = 10 + Math.floor(Math.random() * 20);
      const set = [0];
      for (let i = 1; i < size; i++) set.push(set[set.length - 1] + 1 + Math.floor(Math.random() * 10));
      return { set };
    },
    conditional: false,
  },
  {
    id: 'bench-flat-poly-deg50',
    challengeId: BENCHMARK_CHALLENGE_IDS['bench-flat-poly-deg50'],
    title: 'Flat Polynomials deg 50',
    scoringDirection: 'minimize',
    prompt: 'Find ±1 coefficients for a degree-50 polynomial minimizing peak-to-RMS ratio on unit circle. Return JSON: {"coefficients": [1,-1,1,...]} with 51 values.',
    schemaHint: '{"coefficients": "int[51], each +1 or -1"}',
    randomGenerator: () => ({ coefficients: Array.from({ length: 51 }, () => Math.random() < 0.5 ? 1 : -1) }),
    conditional: false,
  },
  // Thomson N=83 DROPPED (2026-04-06): LLM cannot reliably generate 83×3 unit vectors.
  // Representation mismatch — not a prompt bug. Replaced by Erdős in core.
  {
    id: 'bench-tsp-100',
    challengeId: BENCHMARK_CHALLENGE_IDS['bench-tsp-100'],
    title: 'TSP-100',
    scoringDirection: 'minimize',
    prompt: 'Find shortest tour through 100 cities in [0,1000]² (seed=42). Return JSON: {"tour": [0,42,17,...]} permutation of 0..99.',
    schemaHint: '{"tour": "int[100], permutation of 0..99"}',
    randomGenerator: () => {
      const arr = Array.from({ length: 100 }, (_, i) => i);
      for (let i = arr.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [arr[i], arr[j]] = [arr[j], arr[i]]; }
      return { tour: arr };
    },
    conditional: false,
  },
  {
    id: 'bench-hp-folding-48mer',
    challengeId: BENCHMARK_CHALLENGE_IDS['bench-hp-folding-48mer'],
    title: 'HP Folding 48-mer',
    scoringDirection: 'minimize',
    // FIX 2026-04-06: Changed submission from coords to steps. 47 direction letters
    // (R/L/U/D) are far easier for the LLM to generate reliably than 48 absolute coords.
    // The platform verifier now expands steps to coords before scoring.
    // NOTE: The sequence HPHPPHHPHPPHPHHPPHPHPHHPPHHPHPPHPHHPPHPHPHHPPHHPHH is 50 chars (50-mer).
    // Needs 49 steps, not 47. The card was mislabeled "48-mer" — fix propagated here.
    prompt: 'Fold HP sequence HPHPPHHPHPPHPHHPPHPHPHHPPHHPHPPHPHHPPHPHPHHPPHHPHH (50 residues) on a 2D square lattice to maximize H-H contacts. Specify the fold as 49 step directions (R=right, L=left, U=up, D=down). The walk must be self-avoiding (no position revisited). Output ONLY this JSON:\n{"steps": "RRUULLDD..."}\n49 characters total, each R/L/U/D. Maximize H-H contacts (non-consecutive H residues that are lattice-adjacent).',
    schemaHint: '{"steps": "string of EXACTLY 49 chars, each R/L/U/D"}',
    randomGenerator: () => {
      // Generate a random self-avoiding walk as steps (49 steps for 50-mer)
      const dirs: Record<string, [number, number]> = { R:[1,0], L:[-1,0], U:[0,1], D:[0,-1] };
      const dirNames = ['R','L','U','D'];
      const steps: string[] = [];
      let x = 0, y = 0;
      const occupied = new Set(['0,0']);
      for (let i = 0; i < 49; i++) {
        const shuffled = dirNames.slice().sort(() => Math.random() - 0.5);
        let moved = false;
        for (const d of shuffled) {
          const [dx, dy] = dirs[d];
          const nx = x + dx, ny = y + dy;
          if (!occupied.has(`${nx},${ny}`)) {
            steps.push(d); x = nx; y = ny;
            occupied.add(`${nx},${ny}`);
            moved = true; break;
          }
        }
        if (!moved) { steps.push('R'); x += 1; }
      }
      return { steps: steps.join('') };
    },
    conditional: false,
  },
  {
    id: 'bench-lj-n41',
    challengeId: BENCHMARK_CHALLENGE_IDS['bench-lj-n41'],
    title: 'LJ N=41',
    scoringDirection: 'minimize',
    prompt: 'Place 41 atoms in 3D minimizing Lennard-Jones energy 4·Σ(1/r¹² - 1/r⁶). Return JSON: {"positions": [[x,y,z], ...]} with 41 triples.',
    schemaHint: '{"positions": "float[41][3]"}',
    randomGenerator: () => {
      // FIXED 2026-04-06: Use uniform random positions in a box, NOT equilibrium grid.
      // The old grid generator was a strong heuristic (s_base=-136) that made LLM look
      // worse than it was. Honest random baseline uses positions in a 5×5×5 box.
      const positions = [];
      const box = 5.0; // 5 Å box — reasonable for 41 atoms at LJ equilibrium ~1.12
      for (let i = 0; i < 41; i++) {
        positions.push([
          (Math.random() - 0.5) * box,
          (Math.random() - 0.5) * box,
          (Math.random() - 0.5) * box,
        ]);
      }
      return { positions };
    },
    conditional: true,
  },
  {
    id: 'bench-erdos-overlap',
    challengeId: BENCHMARK_CHALLENGE_IDS['bench-erdos-overlap'],
    title: 'Erdős Overlap',
    scoringDirection: 'minimize',
    prompt: 'Find f:[0,1]→[0,1] with ∫f=1/2 minimizing max overlap. Return JSON: {"values": [0.5, 0.3, ...]} with ��100 floats in [0,1].',
    schemaHint: '{"values": "float[n], n≥100, each in [0,1]"}',
    randomGenerator: () => {
      const n = 200;
      const values = Array.from({ length: n }, () => Math.random());
      // Normalize to ~integral 1/2
      const mean = values.reduce((a, b) => a + b, 0) / n;
      return { values: values.map(v => Math.min(1, Math.max(0, v * 0.5 / mean))) };
    },
    conditional: true,
  },
];

// ─── Platform eval ��─────────────────────────────────────────────────────────

async function evalDev(challengeId: string, solutionData: Record<string, unknown>): Promise<{ devScore: number; cpuMs: number } | null> {
  try {
    const res = await fetch(`${API_URL}/api/v1/benchmark/evaluate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${ADMIN_SECRET}`,
      },
      body: JSON.stringify({ challengeId, solutionData, final: false }),
    });
    const data = await res.json() as { success: boolean; devScore?: number; evalCpuMs?: number; error?: string };
    if (!data.success && data.devScore === undefined) return null;
    return { devScore: data.devScore ?? 0, cpuMs: data.evalCpuMs ?? 0 };
  } catch (e) {
    console.error(`  eval error: ${e}`);
    return null;
  }
}

// ─── LLM call via OpenRouter ────────────────────────────────────────────────

async function llmCall(systemPrompt: string, userPrompt: string, temp = 0.7): Promise<string | null> {
  try {
    const res = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${OPENROUTER_API_KEY}`,
        'HTTP-Referer': 'https://agent4science.org',
        'X-Title': 'Agent4Science-Bench Pilot',
      },
      body: JSON.stringify({
        model: CLAUDE_MODEL,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        temperature: temp,
        max_tokens: 16384,
      }),
    });
    if (!res.ok) {
      const text = await res.text();
      console.error(`  LLM error ${res.status}: ${text.slice(0, 200)}`);
      return null;
    }
    const data = await res.json() as { choices: Array<{ message: { content: string } }> };
    return data.choices[0]?.message?.content || null;
  } catch (e) {
    console.error(`  LLM call failed: ${e}`);
    return null;
  }
}

function parseJSON(raw: string): Record<string, unknown> | null {
  try { return JSON.parse(raw.trim()); } catch { /* fall through */ }
  const match = raw.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
  if (match) { try { return JSON.parse(match[1].trim()); } catch { /* fall through */ } }
  const braceMatch = raw.match(/\{[\s\S]*\}/);
  if (braceMatch) { try { return JSON.parse(braceMatch[0]); } catch { /* fall through */ } }
  return null;
}

// ─── Results storage ─��──────────────────────────────────────────────────────

interface PilotResult {
  taskId: string;
  phase: string;
  index: number;
  score: number | null;
  cpuMs: number;
  timestamp: string;
}

function saveResults(results: PilotResult[], filename: string) {
  mkdirSync(RESULTS_DIR, { recursive: true });
  const path = resolve(RESULTS_DIR, filename);
  writeFileSync(path, JSON.stringify(results, null, 2));
  console.log(`Saved ${results.length} results to ${path}`);
}

function loadResults(filename: string): PilotResult[] {
  const path = resolve(RESULTS_DIR, filename);
  if (existsSync(path)) return JSON.parse(readFileSync(path, 'utf-8'));
  return [];
}

// ─── Phase 1: Random baseline calibration ───────────────────────────────────

async function phase1(tasks: PilotTask[]) {
  console.log('\n========== PHASE 1: Random Baseline Calibration ==========');
  const results: PilotResult[] = loadResults('phase1-random.json');
  const done = new Set(results.map(r => `${r.taskId}:${r.index}`));

  for (const task of tasks) {
    console.log(`\n--- ${task.title} (${task.id}) ---`);
    for (let i = 0; i < 10; i++) {
      const key = `${task.id}:${i}`;
      if (done.has(key)) { console.log(`  random[${i}] skip (cached)`); continue; }

      const solution = task.randomGenerator();
      const result = await evalDev(task.challengeId, solution);
      const entry: PilotResult = {
        taskId: task.id,
        phase: 'random',
        index: i,
        score: result?.devScore ?? null,
        cpuMs: result?.cpuMs ?? 0,
        timestamp: new Date().toISOString(),
      };
      results.push(entry);
      console.log(`  random[${i}] score=${entry.score}`);

      // Save incrementally
      saveResults(results, 'phase1-random.json');
    }
  }

  return results;
}

// ─── Phase 2: Single-shot LLM calibration ───────────────────────────────────

async function phase2(tasks: PilotTask[]) {
  console.log('\n========== PHASE 2: Single-Shot LLM Calibration ==========');
  const results: PilotResult[] = loadResults('phase2-singleshot.json');
  const done = new Set(results.map(r => `${r.taskId}:${r.index}`));

  const systemPrompt = 'You are an expert solver for scientific optimization problems. You MUST output ONLY a valid JSON object — no markdown, no code blocks, no explanation, no comments. The ENTIRE response must be parseable by JSON.parse(). Start your response with { and end with }.';

  for (const task of tasks) {
    console.log(`\n--- ${task.title} (${task.id}) ---`);
    for (let i = 0; i < 10; i++) {
      const key = `${task.id}:${i}`;
      if (done.has(key)) { console.log(`  shot[${i}] skip (cached)`); continue; }

      const temp = 0.6 + i * 0.04; // vary temperature for diversity
      const userPrompt = `# Problem\n${task.prompt}\n\n# Schema\n${task.schemaHint}\n\nGenerate a high-quality solution. Output ONLY the JSON solution object.`;

      const raw = await llmCall(systemPrompt, userPrompt, temp);
      if (!raw) { console.log(`  shot[${i}] LLM failed`); continue; }

      const solution = parseJSON(raw);
      if (!solution) { console.log(`  shot[${i}] parse failed`); continue; }

      const result = await evalDev(task.challengeId, solution);
      const entry: PilotResult = {
        taskId: task.id,
        phase: 'single-shot',
        index: i,
        score: result?.devScore ?? null,
        cpuMs: result?.cpuMs ?? 0,
        timestamp: new Date().toISOString(),
      };
      results.push(entry);
      console.log(`  shot[${i}] score=${entry.score} (temp=${temp.toFixed(2)})`);

      saveResults(results, 'phase2-singleshot.json');

      // Rate limit courtesy
      await new Promise(r => setTimeout(r, 500));
    }
  }

  return results;
}

// ─── Phase 3: VGS on conditional/near-core tasks ────────────────────────────

async function phase3(tasks: PilotTask[]) {
  console.log('\n========== PHASE 3: VGS on Conditional Tasks ==========');
  const conditionalTasks = tasks.filter(t => t.conditional);
  const results: PilotResult[] = loadResults('phase3-vgs.json');
  const done = new Set(results.map(r => `${r.taskId}:${r.index}`));

  const systemPrompt = 'You are an expert solver for scientific optimization problems. You MUST output ONLY a valid JSON object — no markdown, no code blocks, no explanation, no comments. The ENTIRE response must be parseable by JSON.parse(). Start your response with { and end with }.';
  const mutateSystem = 'You are generating a mutated solution for a verifier-guided evolutionary search. Study the top solutions and produce a NEW variant combining their best features. Output ONLY a valid JSON object. Start with { and end with }. No markdown, no explanation.';

  for (const task of conditionalTasks) {
    console.log(`\n--- VGS: ${task.title} (${task.id}) ---`);

    // VGS params: pop=4, gen=5, elite=2
    const POP = 4, GEN = 5, ELITE = 2;
    let population: Array<{ data: Record<string, unknown>; score: number }> = [];

    // Initialize population
    console.log('  Initializing population...');
    for (let p = 0; p < POP; p++) {
      const key = `${task.id}:init-${p}`;
      if (done.has(key)) { console.log(`    pop[${p}] skip (cached)`); continue; }

      const temp = 0.6 + p * 0.1;
      const raw = await llmCall(systemPrompt, `# Problem\n${task.prompt}\n\n# Schema\n${task.schemaHint}\n\nGenerate a high-quality solution. Output ONLY the JSON solution object.`, temp);
      if (!raw) continue;
      const solution = parseJSON(raw);
      if (!solution) continue;

      const result = await evalDev(task.challengeId, solution);
      const score = result?.devScore ?? (task.scoringDirection === 'minimize' ? Infinity : -Infinity);
      population.push({ data: solution, score });

      results.push({
        taskId: task.id, phase: 'vgs-init', index: p,
        score: result?.devScore ?? null, cpuMs: result?.cpuMs ?? 0,
        timestamp: new Date().toISOString(),
      });
      console.log(`    pop[${p}] score=${score} (temp=${temp.toFixed(1)})`);
      saveResults(results, 'phase3-vgs.json');
      await new Promise(r => setTimeout(r, 500));
    }

    if (population.length === 0) { console.log('  No valid initial population, skipping'); continue; }

    // Evolution
    for (let gen = 0; gen < GEN; gen++) {
      console.log(`  Generation ${gen + 1}/${GEN} (pop=${population.length})`);

      // Sort by score
      population.sort((a, b) => task.scoringDirection === 'minimize' ? a.score - b.score : b.score - a.score);
      const elites = population.slice(0, ELITE);
      const topSummary = elites.map((e, i) => `## Solution ${i + 1} (score: ${e.score})\n${JSON.stringify(e.data).slice(0, 500)}`).join('\n\n');

      // Mutate to fill population
      const newPop = [...elites];
      for (let m = 0; m < POP - ELITE; m++) {
        const raw = await llmCall(
          mutateSystem,
          `# Problem\n${task.prompt}\n\n# Schema\n${task.schemaHint}\n\n# Top Solutions\n${topSummary}\n\nGenerate an improved variant. Output ONLY JSON.`,
          0.7 + m * 0.1,
        );
        if (!raw) continue;
        const solution = parseJSON(raw);
        if (!solution) continue;

        const result = await evalDev(task.challengeId, solution);
        const score = result?.devScore ?? (task.scoringDirection === 'minimize' ? Infinity : -Infinity);
        newPop.push({ data: solution, score });

        results.push({
          taskId: task.id, phase: `vgs-gen${gen + 1}`, index: m,
          score: result?.devScore ?? null, cpuMs: result?.cpuMs ?? 0,
          timestamp: new Date().toISOString(),
        });
        console.log(`    mutant[${m}] score=${score}`);
        saveResults(results, 'phase3-vgs.json');
        await new Promise(r => setTimeout(r, 500));
      }

      population = newPop;

      // Convergence check
      const bestScore = task.scoringDirection === 'minimize'
        ? Math.min(...population.map(p => p.score))
        : Math.max(...population.map(p => p.score));
      console.log(`    best=${bestScore}`);
    }
  }

  return results;
}

// ─── Analysis ───────────────────────────────────────────────────────────────

function analyze(
  randomResults: PilotResult[],
  singleShotResults: PilotResult[],
  vgsResults: PilotResult[],
) {
  console.log('\n========== PILOT ANALYSIS ==========\n');
  console.log('Task                        | s_base (random) | s_shot (mean) | s_shot (best) | Sat Ratio | VGS Best  | VGS Improves? | Go/No-Go');
  console.log('----------------------------|-----------------|---------------|---------------|-----------|-----------|---------------|----------');

  for (const task of TASKS) {
    const rScores = randomResults.filter(r => r.taskId === task.id && r.score !== null).map(r => r.score!);
    const ssScores = singleShotResults.filter(r => r.taskId === task.id && r.score !== null).map(r => r.score!);
    const vScores = vgsResults.filter(r => r.taskId === task.id && r.score !== null).map(r => r.score!);

    const sBase = rScores.length > 0
      ? (task.scoringDirection === 'maximize' ? Math.max(...rScores) : Math.min(...rScores))
      : NaN;
    const sMean = ssScores.length > 0
      ? ssScores.reduce((a, b) => a + b, 0) / ssScores.length
      : NaN;
    const sBest = ssScores.length > 0
      ? (task.scoringDirection === 'maximize' ? Math.max(...ssScores) : Math.min(...ssScores))
      : NaN;
    const vBest = vScores.length > 0
      ? (task.scoringDirection === 'maximize' ? Math.max(...vScores) : Math.min(...vScores))
      : NaN;

    // Saturation ratio (higher = more saturated, regardless of direction)
    // For maximize: ratio = sMean / known_good_reference
    // For minimize: we can't easily compute without a reference, so we compare to random
    let satRatio = NaN;
    if (!isNaN(sBase) && !isNaN(sMean) && sBase !== 0) {
      if (task.scoringDirection === 'maximize') {
        satRatio = sMean / sBase; // >1 means LLM beats random
      } else {
        satRatio = sBase / sMean; // >1 means LLM beats random (lower is better)
      }
    }

    const vgsImproves = !isNaN(vBest) && !isNaN(sBest) && (
      task.scoringDirection === 'maximize' ? vBest > sBest : vBest < sBest
    );

    const goNoGo = task.conditional
      ? (vgsImproves ? 'GO' : 'NEEDS REVIEW')
      : 'GO (unconditional)';

    const pad = (s: string | number, w: number) => String(s).slice(0, w).padEnd(w);
    const num = (n: number, d = 1) => isNaN(n) ? 'N/A' : n.toFixed(d);

    console.log(
      `${pad(task.title, 28)}| ${pad(num(sBase), 16)}| ${pad(num(sMean), 14)}| ${pad(num(sBest), 14)}| ${pad(num(satRatio, 2), 10)}| ${pad(num(vBest), 10)}| ${pad(vgsImproves ? 'YES' : 'no', 14)}| ${goNoGo}`
    );
  }
}

// ─── Main ───────────────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const phaseArg = args.find(a => a.startsWith('--phase='));
const taskArg = args.find(a => a.startsWith('--task='));
const requestedPhase = phaseArg ? parseInt(phaseArg.split('=')[1]) : 0; // 0 = all
const requestedTask = taskArg ? taskArg.split('=')[1] : null;

async function main() {
  console.log('Agent4Science-Bench Pilot Runner');
  console.log(`API: ${API_URL}`);
  console.log(`Model: ${CLAUDE_MODEL}`);
  console.log(`Results: ${RESULTS_DIR}`);
  if (requestedPhase) console.log(`Phase: ${requestedPhase} only`);
  if (requestedTask) console.log(`Task: ${requestedTask} only`);
  console.log('---');

  if (!ADMIN_SECRET) { console.error('ADMIN_SECRET not set'); process.exit(1); }
  if (!OPENROUTER_API_KEY) { console.error('OPENROUTER_API_KEY not set'); process.exit(1); }

  const tasks = requestedTask ? TASKS.filter(t => t.id === requestedTask) : TASKS;

  let randomResults: PilotResult[] = [];
  let singleShotResults: PilotResult[] = [];
  let vgsResults: PilotResult[] = [];

  if (requestedPhase === 0 || requestedPhase === 1) {
    randomResults = await phase1(tasks);
  } else {
    randomResults = loadResults('phase1-random.json');
  }

  if (requestedPhase === 0 || requestedPhase === 2) {
    singleShotResults = await phase2(tasks);
  } else {
    singleShotResults = loadResults('phase2-singleshot.json');
  }

  if (requestedPhase === 0 || requestedPhase === 3) {
    vgsResults = await phase3(tasks);
  } else {
    vgsResults = loadResults('phase3-vgs.json');
  }

  analyze(randomResults, singleShotResults, vgsResults);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
