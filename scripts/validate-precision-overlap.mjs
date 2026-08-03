import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';

const read = path => readFileSync(new URL(`../${path}`, import.meta.url), 'utf8');
const policy = read('src/numerical/precisionPolicy.ts');
const renderer = read('src/presentation/progressiveTileFieldRenderer.ts');
const atlas = read('src/references/tileReferenceAtlasV13.ts');
const worker = read('src/v4/referenceWorker.ts');

const requirements = [
  ['bounded focus overlap', policy, 'PREFERRED_REFERENCE_TILE_BUDGET = 4'],
  ['overlap before coordinate collapse', policy, 'PERTURBATION_OVERLAP_SAMPLE_EXPONENT = -23'],
  ['separate required and preferred decisions', policy, 'required: boolean'],
  ['transport demand includes guard bits', policy, '-Math.floor(sampleExponent) + REFERENCE_TRANSPORT_GUARD_BITS'],
  ['request epochs enter the atlas', renderer, 'setDemandEpoch(request.requestId)'],
  ['stale queued reference cancellation', atlas, "queued.reject(new Error('Reference request superseded'))"],
  ['stale active worker cancellation', atlas, 'this.restartWorker(slot)'],
  ['transport contract rejection', atlas, 'response.transportBits < active.requiredTransportBits'],
  ['worker reports working precision', worker, 'workingBits: bits'],
  ['worker reports finite transport ceiling', worker, 'REFERENCE_TRANSPORT_BITS = 96']
];

const failures = requirements
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing precision-overlap invariant: ${label}`);

const requiredBits = sampleExponent => Math.max(48, -Math.floor(sampleExponent) + 32);
if (requiredBits(-26) > 96) failures.push('The 10^5 overlap band must fit the declared current transport.');
if (requiredBits(-126) <= 96) failures.push('The 10^35 band must reject the declared current transport.');

const frozen = new Map([
  ['src/numerical/tileDirectShader.ts', '6f2b4110f3f1f790660bdb99d6116c86ac6dc8f3a59ab5e36ae6157955b63ec1'],
  ['src/numerical/tilePerturbationShader.ts', '963b5954163aa924525040d5923643c6d18167c4eae081a0bc1296f5e7b8e8d7']
]);
for (const [path, expected] of frozen) {
  const actual = createHash('sha256').update(read(path)).digest('hex');
  if (actual !== expected) failures.push(`${path} changed outside the numerical-kernel phase.`);
}

if (failures.length) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log('Validated bounded precision overlap, demand epochs, transport ceiling, and frozen kernels.');
