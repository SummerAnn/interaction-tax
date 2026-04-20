#!/usr/bin/env node
/**
 * Count real platform submissions per (task, protocol, seed) for Run 1,
 * specifically for chain protocols, to see whether chainLength/rounds
 * produced the expected number of linked submissions.
 *
 * Reads Firestore 'submissions' collection directly.
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, resolve } from 'path';

// Load FIREBASE_SERVICE_ACCOUNT_BASE64 from scibook .env.local
const envPath = path.join(import.meta.dirname ?? '.', '..', '..', '.env');
const envText = readFileSync(envPath, 'utf-8');
for (const line of envText.split('\n')) {
  const m = line.match(/^([A-Z_][A-Z0-9_]*)=(.*)$/);
  if (m) process.env[m[1]] = m[2];
}

const b64 = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64;
if (!b64) {
  console.error('FIREBASE_SERVICE_ACCOUNT_BASE64 not found');
  process.exit(1);
}

const { initializeApp, cert, getApps } = await import('firebase-admin/app');
const { getFirestore } = await import('firebase-admin/firestore');

const sa = JSON.parse(Buffer.from(b64, 'base64').toString('utf-8'));
if (getApps().length === 0) {
  initializeApp({ credential: cert(sa), projectId: sa.project_id });
}
const db = getFirestore(getApps()[0], 'agent4science');

// MaxCut challenge ID
const CHALLENGE_ID = 'ch_bench_71qs8npijpnvky4q';

console.log(`Querying Firestore for all submissions to ${CHALLENGE_ID}...`);

const snap = await db
  .collection('submissions')
  .where('challengeId', '==', CHALLENGE_ID)
  .get();

console.log(`Total submissions on MaxCut challenge: ${snap.size}`);

// Group by (protocolId, seed)
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
    hiddenScore: d.hiddenVerifierScore,
    improvesUpon: d.improvesUpon,
    createdAt: d.createdAt,
  });
}

// Sort keys, print count per group
const keys = [...groups.keys()].sort();
console.log('\nChain length per cell (protocol::seed → # submissions):');
for (const key of keys) {
  const subs = groups.get(key);
  console.log(`  ${key.padEnd(25)} ${subs.length}`);
}

// Aggregate: mean chain length per protocol
const protocolStats = new Map();
for (const [key, subs] of groups) {
  const protocolId = key.split('::')[0];
  if (!protocolStats.has(protocolId)) protocolStats.set(protocolId, []);
  protocolStats.get(protocolId).push(subs.length);
}

console.log('\nMean chain length per protocol (on MaxCut):');
for (const [proto, counts] of protocolStats) {
  const mean = counts.reduce((a, b) => a + b, 0) / counts.length;
  const min = Math.min(...counts);
  const max = Math.max(...counts);
  console.log(`  ${proto.padEnd(15)} mean=${mean.toFixed(2)} min=${min} max=${max} cells=${counts.length}`);
}

process.exit(0);
