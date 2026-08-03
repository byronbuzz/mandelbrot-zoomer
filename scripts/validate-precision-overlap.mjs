import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const read = path => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const canonicalKernelSource = source => source.replace(/\r\n/g, '\n').replace(/\n+$/, '');
const policy = read('src/numerical/precisionPolicy.ts');
const renderer = read('src/presentation/progressiveTileFieldRenderer.ts');
const atlas = read('src/references/tileReferenceAtlasV13.ts');
const worker = read('src/v4/referenceWorker.ts');
const perturbationShader = read('src/numerical/tilePerturbationShader.ts');

const requirements = [
  ['bounded viewport seeding', policy, 'PREFERRED_REFERENCE_SEED_BUDGET = 16'],
  ['bounded reference demand', policy, 'MAX_PENDING_REFERENCE_DEMAND = 16'],
  ['overlap before coordinate collapse', policy, 'PERTURBATION_OVERLAP_SAMPLE_EXPONENT = -23'],
  ['separate required and preferred decisions', policy, 'required: boolean'],
  ['transport demand includes guard bits', policy, '-Math.floor(sampleExponent) + REFERENCE_TRANSPORT_GUARD_BITS'],
  ['request epochs enter the atlas', renderer, 'setDemandEpoch(request.requestId)'],
  ['stale queued reference cancellation', atlas, "queued.reject(new Error('Reference request superseded'))"],
  ['active work survives navigation epochs', atlas, 'Keep the bounded active worker set alive'],
  ['local moving viewport reference reuse', atlas, 'MAX_REUSE_COVERAGE_DISTANCE = 1'],
  ['reuse requires requested iteration horizon', atlas, 'reference.requestedIterations < iterations'],
  ['candidate probes cover requested iteration horizon', atlas, 'probeIterations: request.iterations'],
  ['reference activation is coalesced', renderer, 'REFERENCE_ACTIVATION_COALESCE_MS = 24'],
  ['reference activation preserves the spatial plan', renderer, 'applyReferenceRefresh(request)'],
  ['transport contract rejection', atlas, 'response.transportBits < active.requiredTransportBits'],
  ['worker reports working precision', worker, 'workingBits: bits'],
  ['versioned wide reference contract', policy, 'REFERENCE_CONTRACT_VERSION = 2'],
  ['legacy reference transport remains selectable', renderer, "get('referenceTransport') === '96'"],
  ['wide reference transport ceiling', worker, 'WIDE_REFERENCE_TRANSPORT_BITS = 192'],
  ['exact fixed deep reference recurrence', worker, 'buildExactFixedReference'],
  ['dynamic orbit stride', renderer, 'reference.floatsPerPoint'],
  ['scaled perturbation headroom', renderer, 'tile.descriptor.sampleExponent + 64'],
  ['all planned deep levels prefer perturbation', policy, '&& input.sampleExponent <= PERTURBATION_OVERLAP_SAMPLE_EXPONENT'],
  ['pending references keep renderer busy', renderer, 'this.referenceAtlas.pendingCount > 0'],
  ['escaped reference sample is transported', worker, 'zx * zx + zy * zy > escapeSquared'],
  ['scaled orbit-end rebasing', perturbationShader, 'referenceExhausted || perturbationDominates']
];

const failures = requirements
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing precision-overlap invariant: ${label}`);

const requiredBits = sampleExponent => Math.max(48, -Math.floor(sampleExponent) + 32);
if (requiredBits(-26) > 96) failures.push('The 10^5 overlap band must fit the declared current transport.');
if (requiredBits(-126) <= 96) failures.push('The 10^35 band must reject legacy transport.');
if (requiredBits(-126) > 192) failures.push('The 10^35 band must fit the wide transport contract.');

const frozen = new Map([
  ['src/numerical/tileDirectShader.ts', 'e5ae92fcc35c04ecfbd1ffb4bf56bdbbdb2b6664e9669b89f9ff91f48b0add23'],
  ['src/numerical/tilePerturbationShader.ts', '475639182ce5471bee7cc35c58a5b59d89f90a396e05904c1f2c54b95254dd6f']
]);
for (const [path, expected] of frozen) {
  const actual = createHash('sha256').update(canonicalKernelSource(read(path))).digest('hex');
  if (actual !== expected) failures.push(`${path} changed outside the numerical-kernel phase.`);
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log('Validated bounded precision overlap, demand epochs, transport ceiling, and frozen kernels.');

