#!/usr/bin/env npx tsx
/**
 * Close all benchmark challenges that are no longer part of the active experiment.
 *
 * "Active" = the Run-2 core tasks + Cards 9–16 (seeded 2026-04-08 / 2026-04-09).
 * Everything else (Run-1 archive, Thomson, HP Folding v2) gets set to status='closed'.
 *
 * Usage:
 *   npx tsx src/tasks/close-inactive-benchmarks.ts --dry-run
 *   npx tsx src/tasks/close-inactive-benchmarks.ts
 */
import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: resolve(__dirname, '../../../scibook/.env.local') });

import { initializeApp, cert, getApps } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

function getDb() {
  if (getApps().length === 0) {
    const b64 = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64;
    if (!b64) throw new Error('FIREBASE_SERVICE_ACCOUNT_BASE64 not set');
    const sa = JSON.parse(Buffer.from(b64, 'base64').toString('utf-8'));
    initializeApp({ credential: cert(sa), projectId: sa.projectId });
  }
  return getFirestore(getApps()[0], 'agent4science');
}

// Active Run-2 + Cards 9–16 Firestore IDs. Everything else gets closed.
const ACTIVE_IDS = new Set([
  // ── v2 core tasks (Run-2) ────────────────────────────
  'ch_bench_onxf48mfic1gxv2e',  // MaxCut G(200,0.3)
  'ch_bench_ud9ae8gh27e5dmgf',  // Circle Packing N=20
  'ch_bench_jqyo8gwozh9pkzyn',  // Difference Bases
  'ch_bench_c1fpbzjqjs43xpji',  // Flat Polynomials deg50
  'ch_bench_6yll5au03pmzu3xb',  // TSP-100
  // ch_bench_vff8q5o7gipoqm0o — HP Folding 48mer DROPPED — will be closed
  'ch_bench_ud4cxzu0c7rvuqqf',  // LJ-n41
  'ch_bench_njhlgq06ouu7ohlw',  // Erdős Overlap
  // ── Cards 9–12 (seeded 2026-04-08) ──────────────────
  'ch_bench_r51isvqeglm6feqb',  // MaxCut seed=99
  'ch_bench_l637t546sufugy6e',  // Graph Coloring G(80,0.4)
  'ch_bench_amy5xqfggjky2pmi',  // TSP-50
  'ch_bench_cxbn0yvml5v91zow',  // Molecule QED
  // ── Cards 13–16 (seeded 2026-04-09) ─────────────────
  'ch_bench_kfhfy5zpd5lgqol7',  // TFBind-10
  'ch_bench_20p5qefrvs2cm2qj',  // Heilbronn n=12
  'ch_bench_jv67nq0ffxyvujoc',  // Kissing 11D
  'ch_bench_8dwuookpbxchhysz',  // 3-AP-Free {1..100}
]);

const DRY_RUN = process.argv.includes('--dry-run');

async function main() {
  const db = getDb();

  // Only touch ch_bench_ prefixed IDs — never close regular platform challenges
  const snap = await db.collection('challenges')
    .where('benchmarkTrack', '==', 'scientific')
    .get();

  console.log(`Found ${snap.docs.length} benchmark challenges total.`);

  const toClose: { id: string; title: string; status: string }[] = [];
  const active: string[] = [];

  for (const doc of snap.docs) {
    const data = doc.data();
    // Safety guard: only touch ch_bench_ prefixed IDs
    if (!doc.id.startsWith('ch_bench_')) {
      console.log(`SKIP (not ch_bench_)  ${doc.id}  "${data.title}"`);
      continue;
    }
    if (ACTIVE_IDS.has(doc.id)) {
      active.push(`ACTIVE   ${doc.id}  (${data.status})  "${data.title}"`);
    } else {
      toClose.push({ id: doc.id, title: data.title, status: data.status });
    }
  }

  console.log('\n--- Active (will stay open) ---');
  active.forEach(l => console.log(l));

  console.log('\n--- Will be closed ---');
  toClose.forEach(({ id, title, status }) =>
    console.log(`CLOSE  ${id}  (currently: ${status})  "${title}"`)
  );

  console.log(`\nActive: ${active.length}  |  To close: ${toClose.length}`);

  if (toClose.length === 0) {
    console.log('\nNothing to close.');
    return;
  }

  if (DRY_RUN) {
    console.log('\n[DRY RUN] No changes made. Run without --dry-run to apply.');
    return;
  }

  console.log('\nClosing...');
  for (const { id, title } of toClose) {
    await db.collection('challenges').doc(id).update({ status: 'closed' });
    console.log(`CLOSED  "${title}"`);
  }
  console.log('\nDone.');
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
