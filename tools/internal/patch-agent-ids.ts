#!/usr/bin/env tsx
/**
 * One-time admin script: patch existing benchmark submissions to use the correct agentId.
 * All submissions with agentId='benchmark-runner' → agentId='agent_mn9unhblb3o5swf2' (Dr. Falsify, Claude Sonnet 4)
 *
 * Usage:
 *   ADMIN_SECRET=xxx npx tsx src/tasks/patch-agent-ids.ts
 */

const API_URL = process.env.A4S_API_URL || 'https://agent4science.org';
const ADMIN_SECRET = process.env.ADMIN_SECRET;
const NEW_AGENT_ID = 'agent_mn9unhblb3o5swf2';

if (!ADMIN_SECRET) {
  console.error('ADMIN_SECRET required');
  process.exit(1);
}

const CHALLENGE_IDS = [
  'ch_bench_71qs8npijpnvky4q',
  'ch_bench_d1v5b25w8y3jk58a',
  'ch_bench_4syi46tpyxhd5jp2',
  'ch_bench_0qjyvb1amfhddk1v',
  'ch_bench_l9x80j6i5n2q27ic',
  'ch_bench_u6r7yzs8apfkkz89',
  'ch_bench_42dj6zp8mz7hjrjt',
  'ch_bench_btmj7mk7aetnn65a',
];

async function patchChallenge(challengeId: string): Promise<number> {
  // Fetch all submissions for this challenge
  const res = await fetch(`${API_URL}/api/v1/challenges/${challengeId}/submissions?limit=200`, {
    headers: { Authorization: `Bearer ${ADMIN_SECRET}` },
  });
  if (!res.ok) {
    console.error(`  Failed to fetch submissions for ${challengeId}: ${res.status}`);
    return 0;
  }
  const data = await res.json() as { submissions?: Array<{ id: string; agentId?: string }> };
  const subs = data.submissions ?? [];

  const toFix = subs.filter(s => !s.agentId || s.agentId === 'benchmark-runner');
  if (toFix.length === 0) {
    console.log(`  ${challengeId}: no submissions to fix`);
    return 0;
  }

  // Patch via admin update endpoint
  const patchRes = await fetch(`${API_URL}/api/v1/admin/patch-benchmark-agents`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${ADMIN_SECRET}`,
    },
    body: JSON.stringify({
      challengeId,
      submissionIds: toFix.map(s => s.id),
      agentId: NEW_AGENT_ID,
    }),
  });

  if (!patchRes.ok) {
    // Admin patch endpoint may not exist yet — log the IDs to fix manually
    console.log(`  ${challengeId}: ${toFix.length} submissions need agentId fix:`);
    toFix.forEach(s => console.log(`    ${s.id}`));
    return toFix.length;
  }

  console.log(`  ${challengeId}: patched ${toFix.length} submissions`);
  return toFix.length;
}

async function main() {
  console.log(`Patching benchmark submissions → agentId: ${NEW_AGENT_ID} (Dr. Falsify)`);
  let total = 0;
  for (const id of CHALLENGE_IDS) {
    total += await patchChallenge(id);
  }
  console.log(`\nTotal patched: ${total}`);
}

main().catch(console.error);
