#!/usr/bin/env npx tsx
import { config } from 'dotenv';
import { resolve, dirname } from 'path';
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
config({ path: resolve(__dirname, '../../../scibook/.env.local') });
import { initializeApp, cert, getApps } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
function getDb() {
  if (!getApps().length) {
    const b64 = process.env.FIREBASE_SERVICE_ACCOUNT_BASE64!;
    const sa = JSON.parse(Buffer.from(b64, 'base64').toString('utf-8'));
    initializeApp({ credential: cert(sa), projectId: sa.projectId });
  }
  return getFirestore(getApps()[0], 'agent4science');
}
async function main() {
  const db = getDb();
  const updates: [string, string[], string][] = [
    ['ch_bench_20p5qefrvs2cm2qj', ['alphaevolve', 'discrete-geometry', 'combinatorics', 'optimization'], 'Heilbronn n=12'],
    ['ch_bench_jv67nq0ffxyvujoc', ['alphaevolve', 'sphere-packing', 'discrete-geometry', 'optimization'], 'Kissing 11D'],
  ];
  for (const [id, tags, name] of updates) {
    await db.collection('challenges').doc(id).update({ tags });
    console.log(`Updated ${name}: tags[0]=${tags[0]}`);
  }
  console.log('Done.');
}
main().catch(err => { console.error(err); process.exit(1); });
