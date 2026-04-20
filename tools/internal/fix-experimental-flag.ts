#!/usr/bin/env npx tsx
/**
 * One-shot fix: add experimental:true to all benchmark challenges
 * that have benchmarkTrack='scientific' but are missing experimental:true.
 *
 * Usage:
 *   npx tsx src/tasks/fix-experimental-flag.ts --dry-run   # preview
 *   npx tsx src/tasks/fix-experimental-flag.ts              # live patch
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

const DRY_RUN = process.argv.includes('--dry-run');

async function main() {
  const db = getDb();

  // Find all challenges with benchmarkTrack='scientific'
  const snap = await db.collection('challenges')
    .where('benchmarkTrack', '==', 'scientific')
    .get();

  console.log(`Found ${snap.docs.length} docs with benchmarkTrack='scientific'`);

  const toFix: { id: string; title: string }[] = [];
  const alreadySet: string[] = [];

  for (const doc of snap.docs) {
    const data = doc.data();
    if (data.experimental === true) {
      alreadySet.push(doc.id);
    } else {
      toFix.push({ id: doc.id, title: data.title as string });
      console.log(`NEEDS_FIX  ${doc.id}  experimental=${JSON.stringify(data.experimental)}  "${data.title}"`);
    }
  }

  console.log(`\nAlready correct (experimental=true): ${alreadySet.length}`);
  console.log(`Need fix (missing experimental:true): ${toFix.length}`);

  if (toFix.length === 0) {
    console.log('\nAll benchmark challenges already have experimental:true. Nothing to do.');
    return;
  }

  if (DRY_RUN) {
    console.log('\n[DRY RUN] Would set experimental:true on the above docs. Run without --dry-run to apply.');
    return;
  }

  console.log('\nPatching...');
  for (const { id, title } of toFix) {
    await db.collection('challenges').doc(id).update({ experimental: true });
    console.log(`FIXED  ${id}  "${title}"`);
  }
  console.log('\nAll done — benchmark challenges are now hidden from the main listing.');
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
