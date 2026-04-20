#!/usr/bin/env tsx
/**
 * Direct Firestore patch: update all benchmark submissions from agentId='benchmark-runner'
 * to agentId='agent_mn9unhblb3o5swf2' (Dr. Falsify, Claude Sonnet 4).
 *
 * Runs against the scibook Firebase project using GOOGLE_APPLICATION_CREDENTIALS or
 * the service account key in scibook/.env.local (FIREBASE_SERVICE_ACCOUNT_KEY).
 *
 * Usage (from neurips-experiments dir):
 *   GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json npx tsx src/tasks/patch-agent-ids-direct.ts
 *
 * Or just paste the submission IDs and patch via the evaluate endpoint with agentId.
 *
 * Simpler: use the curl approach below for each submission ID from the run log.
 */

const SUBMISSION_IDS_FROM_LOG = `
sub_dsm60ngk3z56m7g0y4hn
sub_2niohpv6xrxrld2vs27w
sub_nvbuil3womgopjyr2wq1
sub_tlm3evens8kqllsar10i
sub_r193ppgzgpu05ca46byz
`.trim().split('\n').filter(Boolean);

console.log(`To patch ${SUBMISSION_IDS_FROM_LOG.length} submissions, run this in scibook:`);
console.log(`\nnpx tsx scripts/patch-benchmark-agent-ids.ts`);
console.log('\nOr use the Firebase Console:');
console.log('  Collection: submissions');
console.log('  Filter: agentId == "benchmark-runner"');
console.log('  Update field: agentId = "agent_mn9unhblb3o5swf2"');
