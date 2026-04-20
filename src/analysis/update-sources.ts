#!/usr/bin/env npx tsx
/**
 * Update challenge descriptions in Firestore to use real scientific sources
 * instead of "Agent4Science-Bench (NeurIPS 2026)".
 */
import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: resolve(__dirname, '../../../scibook/.env.local') });
import { initializeApp, cert, getApps } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

const SOURCES: Record<string, string> = {
  'ch_bench_onxf48mfic1gxv2e': 'Karp (1972)',                           // MaxCut v2
  'ch_bench_ud9ae8gh27e5dmgf': 'Specht (2024)',                          // Circle Packing v2
  'ch_bench_jqyo8gwozh9pkzyn': 'Erdős & Turán (1941)',                   // Difference Bases v2
  'ch_bench_c1fpbzjqjs43xpji': 'Littlewood (1966)',                      // Flat Polynomials v2
  'ch_bench_6yll5au03pmzu3xb': 'Dantzig, Fulkerson & Johnson (1954)',    // TSP-100 v2
  'ch_bench_vff8q5o7gipoqm0o': 'Lau (1986)',                             // HP Folding v2
  'ch_bench_ud4cxzu0c7rvuqqf': 'Wales & Doye (1997)',                    // LJ-n41 v2
  'ch_bench_njhlgq06ouu7ohlw': 'Erdős (1955); AlphaEvolve (2025)',       // Erdős Overlap v2
  'ch_bench_r51isvqeglm6feqb': 'Karp (1972)',                            // MaxCut seed=99
  'ch_bench_l637t546sufugy6e': 'Karp (1972)',                            // Graph Coloring
  'ch_bench_amy5xqfggjky2pmi': 'Dantzig, Fulkerson & Johnson (1954)',    // TSP-50
  'ch_bench_cxbn0yvml5v91zow': 'Bickerton et al. (2012)',                // Molecule QED
};

async function main() {
  const b64 = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64!;
  const sa = JSON.parse(Buffer.from(b64, 'base64').toString('utf-8'));
  if (!getApps().length) initializeApp({ credential: cert(sa), projectId: sa.projectId });
  const db = getFirestore(getApps()[0], 'agent4science');

  for (const [id, newSource] of Object.entries(SOURCES)) {
    const ref = db.collection('challenges').doc(id);
    const snap = await ref.get();
    if (!snap.exists) {
      console.log(`SKIP  ${id} (not found)`);
      continue;
    }
    const data = snap.data()!;
    const desc: string = data.description || '';

    // Replace the **Source**: line (handles both old "Agent4Science-Bench..." and any prior value)
    const updated = desc.replace(
      /^\*{0,2}Source\*{0,2}:\s*.+$/m,
      `**Source**: ${newSource}`
    );

    if (updated === desc) {
      console.log(`SKIP  ${id} — no Source line found or already correct`);
      continue;
    }

    await ref.update({ description: updated });
    console.log(`OK    ${id} → "${newSource}"`);
  }

  console.log('\nDone.');
}

main().catch(console.error);
