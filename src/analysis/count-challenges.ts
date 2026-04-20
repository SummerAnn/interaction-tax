#!/usr/bin/env npx tsx
import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: resolve(__dirname, '../../../scibook/.env.local') });
import { initializeApp, cert, getApps } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

async function main() {
  const b64 = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64!;
  const sa = JSON.parse(Buffer.from(b64, 'base64').toString('utf-8'));
  if (!getApps().length) initializeApp({ credential: cert(sa), projectId: sa.projectId });
  const db = getFirestore(getApps()[0], 'agent4science');
  const snap = await db.collection('challenges').where('experimental', '==', true).get();
  const docs = snap.docs.map(d => ({ id: d.id, title: d.data().title as string }))
    .sort((a, b) => a.title.localeCompare(b.title));
  console.log(`Total experimental challenges: ${docs.length}\n`);
  docs.forEach(d => console.log(`  [${d.id}]  ${d.title}`));
}
main().catch(console.error);
