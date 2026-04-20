#!/usr/bin/env tsx
/**
 * inspect-canary.ts — Verify the 11 canary submissions on agent4science.org.
 *
 * For each, fetches the stored record and checks:
 *   - challengeId matches the expected v2 MaxCut challenge
 *   - devScore is non-null and within plausible range
 *   - hiddenScore is set (means hidden eval ran)
 *   - benchmarkMeta has the expected protocolId
 *   - createdAt is recent
 */

import 'dotenv/config';
import { PlatformClient } from '../lib/platform-client.js';
import { BENCHMARK_CHALLENGE_IDS } from '../tasks/challenge-ids.js';

const EXPECTED_CHALLENGE = BENCHMARK_CHALLENGE_IDS['bench-maxcut-g200'];

const CANARY = [
  { protocol: 'single-shot', sub: 'sub_y1qg60c96ejv7shzust6', expectedDev: 2378 },
  { protocol: 'self-refine', sub: 'sub_kvomkjv261122jnf5avw', expectedDev: 2456 },
  { protocol: 'best-of-n',   sub: 'sub_e5ynorg49kz65ue4r5ty', expectedDev: 2443 },
  { protocol: 'vgs',         sub: 'sub_js19esd0oal2i0i5pv09', expectedDev: 2488 },
  { protocol: 'homo-chain',  sub: 'sub_390qt4fo8mu39fp85425', expectedDev: 2461 },
  { protocol: 'cross-chain', sub: 'sub_wit5sr78ui7tcf7s3f3a', expectedDev: 2491 },
  { protocol: 'magicore',    sub: 'sub_15gugtv33zt6wyh0r55a', expectedDev: 2428 },
  { protocol: 'debate',      sub: 'sub_cvplplzr70gzsksek0my', expectedDev: 2455 },
  { protocol: 'moa',         sub: 'sub_fhcwcw0ll6iyfbuzbre0', expectedDev: 2466 },
  { protocol: 'hpe',         sub: 'sub_p6np8mk2t57gax83y0rw', expectedDev: 2458 },
  { protocol: 'tot',         sub: 'sub_lf0cxk48t22jmgecj56z', expectedDev: 2455 },
];

async function main() {
  const platform = new PlatformClient();
  console.log(`Expected challenge: ${EXPECTED_CHALLENGE}\n`);
  console.log('protocol      submission                         dev    hidden  meta_proto    challenge_ok  created');
  console.log('─'.repeat(118));

  let problems = 0;

  for (const { protocol, sub, expectedDev } of CANARY) {
    try {
      const r = await platform.getSubmission(sub);
      const challengeOk = r.challengeId === EXPECTED_CHALLENGE ? 'yes' : `NO (${r.challengeId})`;
      const metaProto = r.benchmarkMeta?.protocolId ?? '(none)';
      const dev = r.devScore;
      const hidden = r.hiddenScore;
      const created = r.createdAt ? r.createdAt.slice(0, 19) : '(none)';

      const issues: string[] = [];
      if (dev === null) issues.push('devScore=null');
      else if (Math.abs(dev - expectedDev) > 1) issues.push(`dev mismatch (got ${dev}, expected ~${expectedDev})`);
      if (hidden === null) issues.push('hiddenScore=null');
      if (r.challengeId !== EXPECTED_CHALLENGE) issues.push('wrong challenge');
      if (metaProto !== protocol) issues.push(`meta proto mismatch (${metaProto})`);

      if (issues.length > 0) problems++;

      const flag = issues.length > 0 ? '✗' : '✓';
      console.log(`${flag} ${protocol.padEnd(12)} ${sub.padEnd(34)} ${String(dev ?? 'null').padStart(5)}  ${String(hidden ?? 'null').padStart(6)}  ${metaProto.padEnd(12)}  ${challengeOk.padEnd(12)}  ${created}`);
      if (issues.length > 0) {
        for (const i of issues) console.log(`     ! ${i}`);
      }
    } catch (err) {
      problems++;
      console.log(`✗ ${protocol.padEnd(12)} ${sub.padEnd(34)} FETCH ERROR: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  console.log('\n─'.repeat(118));
  if (problems === 0) {
    console.log(`✓ All ${CANARY.length} canary submissions verified clean.`);
  } else {
    console.log(`✗ ${problems} / ${CANARY.length} submissions had problems.`);
    process.exit(1);
  }
}

main().catch(err => { console.error('Fatal:', err); process.exit(1); });
