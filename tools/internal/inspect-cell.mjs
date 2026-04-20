#!/usr/bin/env node
/**
 * Inspect a single (challenge, protocol, seed) cell in full detail —
 * list every submission with devScore, hiddenScore, improvesUpon, createdAt.
 * Used to diagnose why the chain length on the platform doesn't match config.
 */

import { readFileSync } from 'fs';

const envPath = path.join(import.meta.dirname ?? '.', '..', '..', '.env');
const envText = readFileSync(envPath, 'utf-8');
for (const line of envText.split('\n')) {
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

const CHALLENGE_ID = 'ch_bench_71qs8npijpnvky4q'; // MaxCut

const snap = await db
  .collection('submissions')
  .where('challengeId', '==', CHALLENGE_ID)
  .get();

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
    runIndex: meta.runIndex,
  });
}

// For each chain protocol, show the chain for seed=1
const chainProtocols = ['self-refine', 'homo-chain', 'magicore'];
for (const proto of chainProtocols) {
  const key = `${proto}::1`;
  const subs = groups.get(key) ?? [];
  console.log(`\n── ${proto} seed=1 (${subs.length} submissions) ──`);
  // Sort by createdAt
  subs.sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)));
  for (const s of subs) {
    console.log(`  ${s.id}`);
    console.log(`    devScore=${s.devScore} hidden=${s.hiddenScore} runIndex=${s.runIndex}`);
    console.log(`    improvesUpon=${s.improvesUpon ?? '(none)'}`);
    console.log(`    createdAt=${s.createdAt}`);
  }
}

// Also single-shot seed=1 since it had 6 entries
console.log(`\n── single-shot seed=1 (${(groups.get('single-shot::1') ?? []).length} submissions) ──`);
const ssSubs = (groups.get('single-shot::1') ?? []).slice().sort((a, b) => String(a.createdAt).localeCompare(String(b.createdAt)));
for (const s of ssSubs) {
  console.log(`  ${s.id}  dev=${s.devScore} hidden=${s.hiddenScore} runIdx=${s.runIndex} at=${s.createdAt}`);
}

process.exit(0);
