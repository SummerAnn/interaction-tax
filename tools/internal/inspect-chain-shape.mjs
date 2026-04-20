#!/usr/bin/env node
/**
 * inspect-chain-shape.mjs — Look at exact per-cell chain structure for Run 1:
 *   devScores, timestamps (to verify they're from the same run),
 *   parent links, and the chain-round title pattern.
 */

import { readFileSync } from 'fs';
const envPath = path.join(import.meta.dirname ?? '.', '..', '..', '.env');
for (const line of readFileSync(envPath, 'utf-8').split('\n')) {
  const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
  if (m) process.env[m[1]] = m[2];
}
const { initializeApp, cert, getApps } = await import('firebase-admin/app');
const { getFirestore } = await import('firebase-admin/firestore');
const sa = JSON.parse(Buffer.from(process.env.FIREBASE_SERVICE_ACCOUNT_BASE64, 'base64').toString('utf-8'));
if (getApps().length === 0) initializeApp({ credential: cert(sa), projectId: sa.project_id });
const db = getFirestore(getApps()[0], 'agent4science');

const CHALLENGES = {
  'maxcut':           'ch_bench_71qs8npijpnvky4q',
  'difference-bases': 'ch_bench_4syi46tpyxhd5jp2',
  'flat-poly':        'ch_bench_0qjyvb1amfhddk1v',
};
const PROTOS = ['self-refine', 'homo-chain', 'magicore', 'debate'];

function toDate(v) {
  if (!v) return null;
  if (v.toDate) return v.toDate();
  if (v._seconds !== undefined) return new Date(v._seconds * 1000);
  return new Date(v);
}

for (const [taskShort, challengeId] of Object.entries(CHALLENGES)) {
  console.log(`\n══ ${taskShort} ══`);
  const snap = await db.collection('submissions').where('challengeId', '==', challengeId).get();
  const docsByProto = new Map();
  for (const doc of snap.docs) {
    const d = doc.data();
    const meta = d.benchmarkMeta;
    if (!meta || !meta.protocolId) continue;
    if (!PROTOS.includes(meta.protocolId)) continue;
    if (!docsByProto.has(meta.protocolId)) docsByProto.set(meta.protocolId, []);
    docsByProto.get(meta.protocolId).push({
      id: doc.id,
      seed: meta.seed,
      runIndex: meta.runIndex,
      title: d.title ?? null,
      devScore: d.verifierScore,
      improvesUpon: d.improvesUpon ?? null,
      createdAt: toDate(d.createdAt),
    });
  }

  for (const proto of PROTOS) {
    const items = docsByProto.get(proto) ?? [];
    if (items.length === 0) continue;
    // group by seed, sort by createdAt
    const bySeed = new Map();
    for (const it of items) {
      if (!bySeed.has(it.seed)) bySeed.set(it.seed, []);
      bySeed.get(it.seed).push(it);
    }
    for (const [seed, arr] of [...bySeed.entries()].sort((a, b) => a[0] - b[0])) {
      arr.sort((a, b) => (a.createdAt?.getTime() ?? 0) - (b.createdAt?.getTime() ?? 0));
      console.log(`  ${proto} seed=${seed}  (n=${arr.length})`);
      for (const it of arr) {
        const t = it.createdAt ? it.createdAt.toISOString().slice(11, 19) : '?';
        const link = it.improvesUpon ? `↩︎${it.improvesUpon.slice(-6)}` : '(root)';
        const idTail = it.id.slice(-6);
        const titleShort = (it.title ?? '').slice(0, 60);
        console.log(`     ${t}  ${idTail}  score=${String(it.devScore).padEnd(7)} ${link.padEnd(10)} "${titleShort}"`);
      }
    }
  }
}

process.exit(0);
