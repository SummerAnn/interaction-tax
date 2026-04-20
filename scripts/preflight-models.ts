#!/usr/bin/env tsx
/**
 * preflight-models.ts — Verify every backbone the experiment will call is
 * actually alive on OpenRouter before launching a real run.
 *
 * Why this exists: Run 1 wasted ~80 cells because the frozen Gemini ID
 * 'google/gemini-2.5-flash-preview-04-17' was decommissioned by OpenRouter
 * between freeze and the full run, and every cross-chain / MoA cell failed
 * with a 400 "not a valid model ID". A 5-second pre-run catalog check would
 * have caught it.
 *
 * Usage:
 *   npx tsx scripts/preflight-models.ts
 *
 * Exits 0 if every model in MODEL_IDS is present in the OpenRouter catalog,
 * non-zero otherwise (so it composes with `&&` in shell pipelines).
 */

import 'dotenv/config';
import { MODEL_IDS } from '../src/config/models.js';

interface OpenRouterModel {
  id: string;
  name?: string;
  context_length?: number;
}

interface OpenRouterModelsResponse {
  data: OpenRouterModel[];
}

async function main() {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    console.error('FAIL: OPENROUTER_API_KEY not set in environment');
    process.exit(2);
  }

  console.log('Fetching OpenRouter model catalog…');
  const resp = await fetch('https://openrouter.ai/api/v1/models', {
    headers: { Authorization: `Bearer ${apiKey}` },
  });

  if (!resp.ok) {
    console.error(`FAIL: OpenRouter /models returned ${resp.status} ${resp.statusText}`);
    const body = await resp.text().catch(() => '<no body>');
    console.error(body.slice(0, 500));
    process.exit(3);
  }

  const json = (await resp.json()) as OpenRouterModelsResponse;
  const catalog = new Set(json.data.map((m) => m.id));
  console.log(`Catalog size: ${catalog.size} models\n`);

  const required = Object.entries(MODEL_IDS);
  let allOk = true;

  for (const [label, id] of required) {
    if (catalog.has(id)) {
      console.log(`  OK   ${label.padEnd(8)} ${id}`);
    } else {
      console.log(`  MISS ${label.padEnd(8)} ${id}  ← NOT IN CATALOG`);
      allOk = false;
    }
  }

  if (!allOk) {
    console.error('\nFAIL: at least one model ID is no longer served by OpenRouter.');
    console.error('Update src/config/models.ts and re-run preflight before launching the experiment.');
    process.exit(1);
  }

  console.log('\nOK: every backbone is alive on OpenRouter. Safe to launch.');
  process.exit(0);
}

main().catch((err) => {
  console.error('FAIL: unexpected error:', err);
  process.exit(4);
});
