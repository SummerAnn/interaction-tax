/**
 * Pinned backbone model versions for Agent4Science-Bench.
 *
 * FROZEN before pilot. Every run must use these exact model IDs.
 * Unversioned model names (e.g., "gpt-4o") make results non-reproducible.
 *
 * Freeze checklist item #5.
 */

import type { BackboneConfig, LLMProvider } from '../types.js';

// ── Pinned Model IDs ────────────────────────────────────────────────────────

// ── All routed through OpenRouter (one API key for all models) ──────────────

export const MODEL_IDS = {
  // ── Frontier arm ─────────────────────────────────────────────────────────
  /** Primary backbone for all single-model protocols */
  claude: 'anthropic/claude-sonnet-4',
  /** Cross-model chain: step 2 */
  gpt4o: 'openai/gpt-4o',
  /** Cross-model chain: step 3, also MoA diversity agent */
  // NOTE: 'google/gemini-2.5-flash-preview-04-17' (the original frozen ID) was
  // decommissioned by OpenRouter between freeze and the full run — every
  // cross-chain and moa cell in Run 1 failed with a 400 "not a valid model ID".
  // Replaced with the GA 'google/gemini-2.5-flash' on 2026-04-08 after
  // verifying via the /api/v1/models endpoint.
  gemini: 'google/gemini-2.5-flash',

  // ── Open-source arm ──────────────────────────────────────────────────────
  // Three weight-available models served via OpenRouter, used for the *-mixed-oss
  // variants. Picked to span training lineages: DeepSeek (China, MoE),
  // Llama (Meta, dense), Qwen (Alibaba, dense). Lets the paper say "more brains
  // helps even when none of them are frontier", decoupling brain-diversity
  // from API spend.
  deepseek: 'deepseek/deepseek-chat',
  llama:    'meta-llama/llama-3.3-70b-instruct',
  qwen:     'qwen/qwen-2.5-72b-instruct',
} as const;

// ── Provider Routing — everything goes via OpenRouter ───────────────────────

export const MODEL_PROVIDER: Record<string, LLMProvider> = {
  [MODEL_IDS.claude]:   'openrouter',
  [MODEL_IDS.gpt4o]:    'openrouter',
  [MODEL_IDS.gemini]:   'openrouter',
  [MODEL_IDS.deepseek]: 'openrouter',
  [MODEL_IDS.llama]:    'openrouter',
  [MODEL_IDS.qwen]:     'openrouter',
};

// ── Backbone Constructors ───────────────────────────────────────────────────

function bb(model: string): BackboneConfig {
  const apiKey = process.env.OPENROUTER_API_KEY || '';
  return { provider: 'openrouter', model, apiKey };
}

/** Primary backbone (Claude Sonnet 4 via OpenRouter) */
export function primaryBackbone(): BackboneConfig {
  return bb(MODEL_IDS.claude);
}

/** Cross-model chain backbones: Claude → GPT-4o → Gemini (all via OpenRouter) */
export function crossChainBackbones(): BackboneConfig[] {
  return [bb(MODEL_IDS.claude), bb(MODEL_IDS.gpt4o), bb(MODEL_IDS.gemini)];
}

/** MoA diversity backbones: all three models generate independently */
export function moaBackbones(): BackboneConfig[] {
  return crossChainBackbones();
}

/** MoA same-model backbones: all three proposers use Claude (diversity ablation) */
export function moaBackbonesSameModel(): BackboneConfig[] {
  return [bb(MODEL_IDS.claude), bb(MODEL_IDS.claude), bb(MODEL_IDS.claude)];
}

/**
 * Open-source cross-model backbones: DeepSeek V3 → Llama 3.3 70B → Qwen 2.5 72B.
 * Used for the *-mixed-oss protocol arm so we can ask "does brain diversity
 * help when none of the brains are frontier?" — decoupling diversity from
 * raw model capability and from API spend.
 */
export function crossChainOSSBackbones(): BackboneConfig[] {
  return [bb(MODEL_IDS.deepseek), bb(MODEL_IDS.llama), bb(MODEL_IDS.qwen)];
}

/**
 * Homogeneous DeepSeek backbone for the brain-diversity 2×2 closer.
 * The frontier arm has matched (homogeneous Claude) and (frontier mixed) cells,
 * but the OSS arm is missing a homogeneous-OSS control — making the OSS-mixed
 * arm a one-sided check instead of a full 2×2. This constructor wires up
 * the missing corner: three role-structured protocols (MAgICoRe, Debate, HPE)
 * are run with every role pointing at DeepSeek V3 (the strongest of the three
 * OSS models), so the 2×2 (frontier vs OSS) × (homogeneous vs mixed) design
 * becomes complete. DeepSeek is picked over Llama/Qwen because it's the
 * strongest single open-weight model in the ablation set.
 */
export function deepseekBackbone(): BackboneConfig {
  return bb(MODEL_IDS.deepseek);
}

/** 5 diverse model backbones: Claude + GPT-4o + Gemini + DeepSeek + Qwen */
export function fiveDiverseBackbones(): BackboneConfig[] {
  return [bb(MODEL_IDS.claude), bb(MODEL_IDS.gpt4o), bb(MODEL_IDS.gemini), bb(MODEL_IDS.deepseek), bb(MODEL_IDS.qwen)];
}

/** GPT-4o backbone (solo) — for single-model protocol variants */
export function gpt4oBackbone(): BackboneConfig {
  return bb(MODEL_IDS.gpt4o);
}

/** Gemini 2.5 Flash backbone (solo) — for single-model protocol variants */
export function geminiBackbone(): BackboneConfig {
  return bb(MODEL_IDS.gemini);
}
