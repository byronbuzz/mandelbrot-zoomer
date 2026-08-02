import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/persistentTileRenderer.ts', import.meta.url), 'utf8');
const shader = readFileSync(new URL('../src/numerical/persistentTileShaders.ts', import.meta.url), 'utf8');
const planner = readFileSync(new URL('../src/tiles/worldTilePlanner.ts', import.meta.url), 'utf8');

const required = [
  ['world-space tile keys', planner, 'sampleExponent}:${tileX.toString()}:${tileY.toString()}'],
  ['exact dyadic tile centres', planner, 'function dyadicFixed'],
  ['persistent recurrence state', shader, 'recurrenceState: array<vec4f>'],
  ['persistent recurrence metadata', shader, 'recurrenceMeta: array<vec2u>'],
  ['bounded iteration chunks', shader, 'iteration + p.chunkIterations'],
  ['active tile telemetry', shader, 'activePixels: atomic<u32>'],
  ['separate quality resource', shader, 'qualityTexture: texture_storage_2d<rgba8unorm, write>'],
  ['provisional coverage remains transparent', shader, 'textureStore(colourTexture, pixel, vec4f(0.0))'],
  ['tile cache survives camera requests', renderer, 'private readonly tileMap'],
  ['batch-boundary request coalescing', renderer, 'hasNewerRequest(request.requestId)'],
  ['numerical freshness telemetry', renderer, 'numericalFreshnessMs'],
  ['bounded tile cache', renderer, 'MAX_CACHED_TILES'],
  ['alpha is accepted coverage only', renderer, "srcFactor: 'src-alpha'"]
];

const failures = required
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing 1.1 invariant: ${label}`);

if (/seedShader|sourceImage|destinationImage/.test(renderer)) {
  failures.push('Persistent numerical tiles must not be seeded from colour history.');
}
if (/blockSize:\s*8[\s\S]*blockSize:\s*4[\s\S]*blockSize:\s*2[\s\S]*blockSize:\s*1/.test(renderer)) {
  failures.push('Persistent tile scheduling must not recreate the static full-viewport 8/4/2/1 ladder.');
}
if (/PERTURBATION_SCALE_EXPONENT|DOUBLE_FLOAT_THRESHOLD|depth\s*[<>]=?/.test(renderer)) {
  failures.push('Precision must not be selected by a global depth threshold.');
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log('Validated WebGPU Fractal Zoomer 1.1 persistent-tile invariants.');
