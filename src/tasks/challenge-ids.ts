/**
 * Synthetic task IDs used for local evaluator bookkeeping in the release
 * artifact. The anonymous release does not include the original remote
 * platform identifiers.
 */
export const BENCHMARK_CHALLENGE_IDS: Record<string, string> = {
  'bench-maxcut-g200':        'local-bench-maxcut-g200',
  'bench-circle-packing-n20': 'local-bench-circle-packing-n20',
  'bench-difference-bases':   'local-bench-difference-bases',
  'bench-flat-poly-deg50':    'local-bench-flat-poly-deg50',
  'bench-tsp-100':            'local-bench-tsp-100',
  'bench-lj-n41':             'local-bench-lj-n41',
  'bench-erdos-overlap':      'local-bench-erdos-overlap',
  'bench-tsp-50':             'local-bench-tsp-50',
  'bench-molecule-qed':       'local-bench-molecule-qed',
  'bench-3ap-free-100':       'local-bench-3ap-free-100',
  'bench-knapsack-50':        'local-bench-knapsack-50',
};
