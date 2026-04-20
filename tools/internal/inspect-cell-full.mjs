#!/usr/bin/env node
/**
 * Look at all chain protocol cells across all 8 challenges (Run 1 time window).
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
if (getApps().length === 0) {
  initializeApp({ credential: cert(sa), projectId: sa.project_id });
}
const db = getFirestore(getApps()[0], 'agent4science');

const CHALLENGES = {
  'bench-maxcut-g200': 'ch_bench_71qs8npijpnvky4q',
  'bench-circle-packing-n20': 'ch_bench_d1v5b25w8y3jk58a',
  'bench-difference-bases': 'ch_bench_4syi46tpyxhd5jp2',
  'bench-flat-poly-deg50': 'ch_bench_0qjyvb1amfhddk1v',
  'bench-tsp-100': 'ch_bench_l9x80j6i5n2q27ic',
  'bench-hp-folding-48mer': 'ch_bench_u6r7yzs8apfkkz89',
  'bench-lj-n41': 'ch_bench_42dj6zp8mz7hjrjt',
  'bench-erdos-overlap': 'ch_bench_btmj7mk7aetnn65a',
};

const CHAIN_EXPECTED = {
  'self-refine': 4,   // 1 initial + 3 refinements
  'homo-chain': 4,    // chainLength
  'magicore': 3,      // 1 solver + 2 refiners
  'cross-chain': 3,   // 3 backbones (though Gemini failed)
};

// Query each challenge separately
const table = [];
for (const [taskId, challengeId] of Object.entries(CHALLENGES)) {
  const snap = await db.collection('submissions').where('challengeId', '==', challengeId).get();

  // Group by (protocol, seed)
  const groups = new Map();
  for (const doc of snap.docs) {
    const d = doc.data();
    const meta = d.benchmarkMeta;
    if (!meta || !meta.protocolId) continue;
    const key = `${meta.protocolId}::${meta.seed ?? '?'}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push({
      id: doc.id,
      devScore: d.verifierScore,
      improvesUpon: d.improvesUpon,
      createdAt: d.createdAt,
    });
  }

  for (const [proto, expected] of Object.entries(CHAIN_EXPECTED)) {
    for (let seed = 1; seed <= 5; seed++) {
      const subs = groups.get(`${proto}::${seed}`) ?? [];
      if (subs.length === 0) continue;
      const linked = subs.filter(s => s.improvesUpon).length;
      table.push({ taskId, proto, seed, total: subs.length, linked, expected });
    }
  }
}

console.log('task                         proto         seed  total linked  expected');
for (const r of table) {
  console.log(
    r.taskId.padEnd(28) + ' ' +
    r.proto.padEnd(12) + ' ' +
    String(r.seed).padEnd(5) + ' ' +
    String(r.total).padEnd(6) + ' ' +
    String(r.linked).padEnd(7) + ' ' +
    String(r.expected)
  );
}

// Summary by protocol × task
const summary = new Map();
for (const r of table) {
  const key = `${r.taskId}::${r.proto}`;
  if (!summary.has(key)) summary.set(key, { total: [], linked: [], expected: r.expected });
  summary.get(key).total.push(r.total);
  summary.get(key).linked.push(r.linked);
}

console.log('\n── Per (task, protocol) means ──');
console.log('task                          proto         mean_total  mean_linked  expected');
for (const [key, s] of summary) {
  const [taskId, proto] = key.split('::');
  const mt = (s.total.reduce((a, b) => a + b, 0) / s.total.length).toFixed(1);
  const ml = (s.linked.reduce((a, b) => a + b, 0) / s.linked.length).toFixed(1);
  console.log(taskId.padEnd(28) + ' ' + proto.padEnd(12) + ' ' + mt.padStart(10) + ' ' + ml.padStart(12) + ' ' + String(s.expected).padStart(8));
}

process.exit(0);
