import { readFileSync } from 'node:fs';

const tile = readFileSync(new URL('../src/tiles/persistentTile.ts', import.meta.url), 'utf8');
const scheduler = readFileSync(new URL('../src/scheduler/tilePriority.ts', import.meta.url), 'utf8');
const atlas = readFileSync(new URL('../src/references/referenceAtlas.ts', import.meta.url), 'utf8');
const shader = readFileSync(new URL('../src/numerical/persistentDirectShader.ts', import.meta.url), 'utf8');

const checks = [
  ['persistent recurrence state', tile, 'stateBuffer: GPUBuffer | null'],
  ['active index state', tile, 'activeIndexBuffer: GPUBuffer | null'],
  ['numerical freshness', tile, 'newestNumericalSampleAt'],
  ['tile health consumed by scheduler', scheduler, 'sentinelMismatchRate'],
  ['bounded navigation chunk', scheduler, 'NAVIGATION_MAX_CHUNK'],
  ['local reference assignment', atlas, 'tileAssignments'],
  ['reference cache bound', atlas, 'maxReferences = 32'],
  ['chunked recurrence kernel', shader, 'chunkIterations'],
  ['GPU active counter', shader, 'atomicAdd(&activeCount, 1u)']
];

const failures = checks
  .filter(([, source, token]) => !source.includes(token))
  .map(([label]) => `Missing architecture invariant: ${label}`);

if (scheduler.includes("depth >=") || scheduler.includes('scale.exponent')) {
  failures.push('Tile precision scheduling must not use a global depth crossover');
}
if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`Validated ${checks.length} persistent-field architecture invariants.`);
