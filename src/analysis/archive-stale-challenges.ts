#!/usr/bin/env npx tsx
/**
 * Archive stale experimental challenges that are not in the active 12-task benchmark set.
 * Sets experimental=false so they disappear from /challenge_experiment.
 */
import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: resolve(__dirname, '../../../scibook/.env.local') });
import { initializeApp, cert, getApps } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

// The 12 active benchmark challenge IDs
const ACTIVE_IDS = new Set([
  'ch_bench_onxf48mfic1gxv2e', // MaxCut v2
  'ch_bench_ud9ae8gh27e5dmgf', // Circle Packing v2
  'ch_bench_jqyo8gwozh9pkzyn', // Difference Bases v2
  'ch_bench_c1fpbzjqjs43xpji', // Flat Polynomials v2
  'ch_bench_6yll5au03pmzu3xb', // TSP-100 v2
  'ch_bench_vff8q5o7gipoqm0o', // HP Folding v2
  'ch_bench_ud4cxzu0c7rvuqqf', // LJ-n41 v2
  'ch_bench_njhlgq06ouu7ohlw', // Erdős Overlap v2
  'ch_bench_r51isvqeglm6feqb', // MaxCut seed=99
  'ch_bench_l637t546sufugy6e', // Graph Coloring
  'ch_bench_amy5xqfggjky2pmi', // TSP-50
  'ch_bench_cxbn0yvml5v91zow', // Molecule QED
]);

async function main() {
  const b64 = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64!;
  const sa = JSON.parse(Buffer.from(b64, 'base64').toString('utf-8'));
  if (!getApps().length) initializeApp({ credential: cert(sa), projectId: sa.projectId });
  const db = getFirestore(getApps()[0], 'agent4science');

  const snap = await db.collection('challenges').where('experimental', '==', true).get();
  console.log(`Found ${snap.size} experimental challenges total\n`);

  const toArchive: Array<{ id: string; title: string }> = [];
  const active: Array<{ id: string; title: string }> = [];

  for (const doc of snap.docs) {
    const data = doc.data();
    if (!data.title?.startsWith('Benchmark:')) continue;
    if (ACTIVE_IDS.has(doc.id)) {
      active.push({ id: doc.id, title: data.title });
    } else {
      toArchive.push({ id: doc.id, title: data.title });
    }
  }

  console.log(`Active (keep):`);
  active.forEach(c => console.log(`  KEEP    [${c.id}]  ${c.title}`));

  console.log(`\nStale (archive):`);
  if (toArchive.length === 0) {
    console.log('  (none — all Benchmark: challenges are in the active set)');
  } else {
    for (const c of toArchive) {
      console.log(`  ARCHIVE [${c.id}]  ${c.title}`);
    }

    const args = process.argv.slice(2);
    if (!args.includes('--execute')) {
      console.log('\nDry run — pass --execute to actually archive.');
      return;
    }

    for (const c of toArchive) {
      await db.collection('challenges').doc(c.id).update({ experimental: false });
      console.log(`  OK      archived ${c.id}`);
    }
    console.log('\nDone.');
  }
}

main().catch(console.error);
