import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/progressiveTileFieldRenderer.ts', import.meta.url), 'utf8');
const directShader = readFileSync(new URL('../src/numerical/tileDirectShader.ts', import.meta.url), 'utf8');
const perturbShader = readFileSync(new URL('../src/numerical/tilePerturbationShader.ts', import.meta.url), 'utf8');
const atlas = readFileSync(new URL('../src/references/tileReferenceAtlasV13.ts', import.meta.url), 'utf8');
const planner = readFileSync(new URL('../src/tiles/worldTilePlanner.ts', import.meta.url), 'utf8');

const checks = [
  ['persistent recurrence state buffers', renderer, 'stateBuffer: GPUBuffer'],
  ['persistent recurrence metadata buffers', renderer, 'metaBuffer: GPUBuffer'],
  ['world-space tile cache', renderer, 'Map<PersistentTileKey, FieldTile>'],
  ['numerical freshness', renderer, 'lastNumericalUpdateAt'],
  ['bounded iteration chunks', directShader, 'p.chunkIterations'],
  ['perturbation continuation', perturbShader, 'referenceIndex = pixelMeta.z'],
  ['GPU active counter', directShader, 'atomicAdd(&counters.activePixels, 1u)'],
  ['separate provisional coverage', renderer, 'coveragePixels'],
  ['separate resolved coverage', renderer, 'resolvedPixels'],
  ['spatially progressive lattice plan', renderer, '[2, 1, 0]'],
  ['bounded scheduler quantum', renderer, 'quantumBudget'],
  ['local reference assignment', atlas, 'repairGeometry'],
  ['reference reuse cache', atlas, 'private readonly cache'],
  ['authoritative dyadic pixel lattice', planner, 'Math.floor(pixelExponent)']
];

const failures = checks
  .filter(([, source, token]) => !source.includes(token))
  .map(([label]) => `Missing architecture invariant: ${label}`);

if (/depth\s*[<>]=?|scale\.exponent[\s\S]{0,160}perturb/i.test(renderer)) {
  failures.push('Tile precision scheduling must not use a global depth crossover');
}
if (/MOVING_DIRECT_ITERATIONS|SETTLING_DIRECT_ITERATIONS/.test(renderer)) {
  failures.push('Persistent continuation must not be replaced by global interaction iteration caps');
}
if (/seedShader|sourceImage|destinationImage/.test(renderer)) {
  failures.push('Numerical persistence must not be simulated by copying colour history');
}
if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`Validated ${checks.length} active persistent-field architecture invariants.`);
