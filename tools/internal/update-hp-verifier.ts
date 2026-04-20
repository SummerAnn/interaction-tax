#!/usr/bin/env npx tsx
/**
 * One-time script: update HP Folding challenge in Firestore to use steps-based verifier.
 * Run from neurips-experiments/: npx tsx src/tasks/update-hp-verifier.ts
 */
import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
import { initializeApp, cert, getApps } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { HP_FOLDING } from './benchmark-challenges.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: resolve(__dirname, '../../../scibook/.env.local') });

const raw = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64!;
const svc = JSON.parse(Buffer.from(raw, 'base64').toString('utf8'));
if (!getApps().length) initializeApp({ credential: cert(svc) });
const db = getFirestore('agent4science');

const HP_FOLDING_ID = 'ch_bench_u6r7yzs8apfkkz89';

async function run() {
  await db.collection('challenges').doc(HP_FOLDING_ID).update({
    verifier: HP_FOLDING.verifier,
    hiddenVerifier: HP_FOLDING.hiddenVerifier,
    solutionSchema: HP_FOLDING.solutionSchema,
    updatedAt: new Date(),
  });
  console.log('✓ Updated HP Folding verifier to steps-based schema:', HP_FOLDING_ID);
}

run().catch(console.error);
