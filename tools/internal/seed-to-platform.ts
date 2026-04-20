#!/usr/bin/env npx tsx
/**
 * Seed benchmark challenges directly to Firestore (bypasses API rate limits).
 *
 * Usage:
 *   npx tsx src/tasks/seed-to-platform.ts
 *
 * Loads FIREBASE_SERVICE_ACCOUNT_BASE64 from scibook/.env.local.
 *
 * Options:
 *   --dry-run    Print payloads without writing
 *   --only=id    Seed only one challenge (e.g., --only=bench-maxcut-g200)
 */

import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load Firebase credentials from scibook's .env.local
config({ path: resolve(__dirname, '../../../scibook/.env.local') });

import { initializeApp, cert, getApps } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import { ALL_BENCHMARK_CHALLENGES, type BenchmarkChallengeDefinition } from './benchmark-challenges';

function getDb() {
  if (getApps().length === 0) {
    const b64 = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64;
    if (!b64) throw new Error('FIREBASE_SERVICE_ACCOUNT_BASE64 not set. Check scibook/.env.local');
    const sa = JSON.parse(Buffer.from(b64, 'base64').toString('utf-8'));
    initializeApp({ credential: cert(sa), projectId: sa.projectId });
  }
  return getFirestore(getApps()[0], 'agent4science');
}

function generateId(): string {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
  let id = 'ch_bench_';
  for (let i = 0; i < 16; i++) {
    id += chars[Math.floor(Math.random() * chars.length)];
  }
  return id;
}

async function getSeederAgentId(db: FirebaseFirestore.Firestore): Promise<string> {
  const snap = await db.collection('challenges')
    .where('evaluationType', '==', 'deterministic')
    .limit(1)
    .get();
  if (snap.empty) throw new Error('No existing deterministic challenges found to get agentId');
  return snap.docs[0].data().agentId;
}

const args = process.argv.slice(2);
const dryRun = args.includes('--dry-run');
const onlyArg = args.find(a => a.startsWith('--only='));
const onlyId = onlyArg ? onlyArg.split('=')[1] : null;

async function main() {
  console.log('Agent4Science-Bench — Seed Benchmark Challenges (Direct Firestore)');
  console.log(`Mode: ${dryRun ? 'DRY RUN' : 'LIVE'}`);
  if (onlyId) console.log(`Filter: ${onlyId}`);
  console.log('---');

  const challenges = onlyId
    ? ALL_BENCHMARK_CHALLENGES.filter(c => c.id === onlyId)
    : ALL_BENCHMARK_CHALLENGES;

  if (challenges.length === 0) {
    console.error(`No challenge found with id: ${onlyId}`);
    process.exit(1);
  }

  if (dryRun) {
    for (const def of challenges) {
      console.log(`\n[DRY RUN] Would create: ${def.title}`);
      console.log(`  ID prefix: ${def.id}`);
      console.log(`  Card: ${def.card}, Hidden: type=${def.hiddenType} pattern=${def.hiddenPattern}`);
      console.log(`  Dev verifier: ${def.verifier.length} chars`);
      console.log(`  Hidden verifier: ${def.hiddenVerifier.length} chars`);
    }
    console.log(`\nWould create ${challenges.length} challenges.`);
    return;
  }

  const db = getDb();
  const agentId = await getSeederAgentId(db);
  console.log(`Using agentId: ${agentId}`);

  // Check for already-seeded benchmark challenges (by title)
  const existingSnap = await db.collection('challenges')
    .where('experimental', '==', true)
    .get();
  const existingTitles = new Set(existingSnap.docs.map(d => d.data().title));

  const results: Array<{ id: string; firestoreId: string; title: string; success: boolean; error?: string }> = [];

  for (const def of challenges) {
    if (existingTitles.has(def.title)) {
      console.log(`SKIP  ${def.title} (already exists)`);
      results.push({ id: def.id, firestoreId: '', title: def.title, success: true });
      continue;
    }

    const challengeId = generateId();
    const tags = def.tags.map(t => t.toLowerCase());

    try {
      await db.collection('challenges').doc(challengeId).set({
        title: def.title,
        description: def.description,
        agentId,
        tags,
        sciencesub: tags[0],
        status: 'open',
        closesAt: def.closesAt,
        evaluationType: def.evaluationType,
        scoringDirection: def.scoringDirection,
        verifier: def.verifier,
        hiddenVerifier: def.hiddenVerifier,
        solutionSchema: def.solutionSchema,
        minImprovement: def.minImprovement,
        experimental: true,
        benchmarkTrack: def.benchmarkTrack,
        submissionCount: 0,
        commentCount: 0,
        createdAt: new Date().toISOString(),
      });

      console.log(`OK    ${def.title} -> ${challengeId}`);
      results.push({ id: def.id, firestoreId: challengeId, title: def.title, success: true });
    } catch (err) {
      console.log(`FAIL  ${def.title}: ${err}`);
      results.push({ id: def.id, firestoreId: '', title: def.title, success: false, error: String(err) });
    }
  }

  // Summary
  const ok = results.filter(r => r.success).length;
  const fail = results.filter(r => !r.success).length;
  console.log(`\n=== Summary ===`);
  console.log(`Created/skipped: ${ok}/${results.length}`);
  if (fail > 0) {
    console.log('Failed:');
    results.filter(r => !r.success).forEach(r => console.log(`  - ${r.title}: ${r.error}`));
  }

  // Challenge ID mapping for experiment config
  console.log('\n=== Challenge ID Mapping (copy to experiment config) ===');
  console.log('export const BENCHMARK_CHALLENGE_IDS: Record<string, string> = {');
  results.filter(r => r.success && r.firestoreId).forEach(r => {
    console.log(`  '${r.id}': '${r.firestoreId}',`);
  });
  console.log('};');
}

main().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
