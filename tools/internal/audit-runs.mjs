#!/usr/bin/env node
/**
 * audit-runs.mjs — Group existing submissions by (protocol, runIndex, createdAt-bucket)
 * to reconstruct how many distinct runs got mixed into each challenge.
 */

import { readFileSync } from 'fs';
import path from 'path';
const envPath = path.join(import.meta.dirname ?? '.', '..', '..', '..', '.env');
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
  'circle-packing':   'ch_bench_d1v5b25w8y3jk58a',
  'difference-bases': 'ch_bench_4syi46tpyxhd5jp2',
  'flat-poly':        'ch_bench_0qjyvb1amfhddk1v',
  'tsp-100':          'ch_bench_l9x80j6i5n2q27ic',
  'hp-folding':       'ch_bench_u6r7yzs8apfkkz89',
  'lj-n41':           'ch_bench_42dj6zp8mz7hjrjt',
  'erdos-overlap':    'ch_bench_btmj7mk7aetnn65a',
};

function toDate(v) {
  if (!v) return null;
  if (v.toDate) return v.toDate();
  if (v._seconds !== undefined) return new Date(v._seconds * 1000);
  return new Date(v);
}

const DAY_MS = 24 * 60 * 60 * 1000;
const HOUR_MS = 60 * 60 * 1000;

console.log('\n══ Run clustering by (challenge, protocol) ══\n');

for (const [taskShort, challengeId] of Object.entries(CHALLENGES)) {
  const snap = await db.collection('submissions').where('challengeId', '==', challengeId).get();

  // group by protocol
  const byProto = new Map();
  for (const doc of snap.docs) {
    const d = doc.data();
    const meta = d.benchmarkMeta;
    if (!meta || !meta.protocolId) continue;
    const p = meta.protocolId;
    if (!byProto.has(p)) byProto.set(p, []);
    byProto.get(p).push({
      id: doc.id,
      seed: meta.seed ?? null,
      runIndex: meta.runIndex ?? null,
      createdAt: toDate(d.createdAt),
      devScore: d.verifierScore,
      improvesUpon: d.improvesUpon ?? null,
    });
  }

  console.log(`── ${taskShort} (${challengeId}) total=${snap.size}`);
  const protoNames = [...byProto.keys()].sort();
  for (const p of protoNames) {
    const subs = byProto.get(p);
    subs.sort((a, b) => (a.createdAt?.getTime() ?? 0) - (b.createdAt?.getTime() ?? 0));

    // cluster by createdAt gaps > 1h
    const clusters = [];
    for (const s of subs) {
      const last = clusters[clusters.length - 1];
      if (!last || !s.createdAt || !last.end || (s.createdAt - last.end) > HOUR_MS) {
        clusters.push({ start: s.createdAt, end: s.createdAt, items: [s] });
      } else {
        last.end = s.createdAt;
        last.items.push(s);
      }
    }

    const seeds = new Set(subs.map(s => s.seed));
    const runIdxs = new Set(subs.map(s => s.runIndex));
    const linkedCount = subs.filter(s => s.improvesUpon).length;

    console.log(
      `   ${p.padEnd(13)}  n=${String(subs.length).padEnd(3)} ` +
      `seeds={${[...seeds].sort((a,b)=>a-b).join(',')}} ` +
      `runIdx={${[...runIdxs].sort((a,b)=>a-b).join(',')}} ` +
      `linked=${linkedCount}/${subs.length} ` +
      `clusters=${clusters.length}`
    );
    for (const c of clusters) {
      const fmt = d => d ? d.toISOString().slice(0, 16).replace('T', ' ') : '?';
      console.log(`        ${fmt(c.start)} → ${fmt(c.end)}  (${c.items.length} subs)`);
    }
  }
  console.log();
}

process.exit(0);
