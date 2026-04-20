#!/usr/bin/env npx tsx
/**
 * One-time script: update Erdős Overlap challenge in Firestore to use
 * normalize-without-clip verifier (fix 2026-04-09).
 *
 * Background: the old verifier normalized then clipped values to [0,1].
 * Sparse submissions could game this: sparse values → scale ×K → clip → sparse
 * indicator → near-zero autocorrelation (artificially good score).
 * Fix: remove the clip. Sparse → scale ×K → huge values → autocorr = K²×density >> 0.25 → bad score.
 *
 * Run: npx tsx src/tasks/update-erdos-verifier.ts
 */
import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { initializeApp, cert, getApps } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { ERDOS_OVERLAP } from './benchmark-challenges.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: resolve(__dirname, '../../../scibook/.env.local') });

const raw = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64!;
const svc = JSON.parse(Buffer.from(raw, 'base64').toString('utf8'));
if (!getApps().length) initializeApp({ credential: cert(svc) });
const db = getFirestore('agent4science');

const ERDOS_CHALLENGE_ID = 'ch_bench_njhlgq06ouu7ohlw';  // v2 run-2 ID

async function run() {
  await db.collection('challenges').doc(ERDOS_CHALLENGE_ID).update({
    verifier:       ERDOS_OVERLAP.verifier,
    hiddenVerifier: ERDOS_OVERLAP.hiddenVerifier,
    solutionSchema: ERDOS_OVERLAP.solutionSchema,
    updatedAt: new Date(),
  });
  console.log('✓ Updated Erdős Overlap verifier (normalize-without-clip):', ERDOS_CHALLENGE_ID);
  console.log('  Next: re-run all Erdős cells (16 protocols × 5 seeds = 80 cells)');
  console.log('  Command: npx tsx src/runner/run-full.ts --task bench-erdos-overlap');
}

run().catch(console.error);
