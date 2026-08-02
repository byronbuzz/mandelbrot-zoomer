import { readFileSync } from 'node:fs';

const shader = readFileSync(new URL('../src/v6/shaders.ts', import.meta.url), 'utf8');
const renderer = readFileSync(new URL('../src/v6/progressiveRenderer.ts', import.meta.url), 'utf8');
const referenceService = readFileSync(new URL('../src/v6/referenceService.ts', import.meta.url), 'utf8');

const checks = [
  ['persistent reference index', shader, 'recurrenceMeta: array<vec2u>'],
  ['scaled perturbation state', shader, 'fn scaleByViewport'],
  ['scaled quadratic term', shader, 'fn quadraticByViewport'],
  ['viewport-normalized rebase', shader, 'fn divideByViewport'],
  ['orbit-end rebase trigger', shader, 'let referenceExhausted'],
  ['reference index reset', shader, 'referenceIndex = 0u'],
  ['progressive quality confidence', shader, 'fn qualityAlpha'],
  ['opaque final target', shader, 'p.blockSize == 1u && p.iterations >= p.targetIterations'],
  ['stable-current confidence blend', shader, 'let weight = max(confidence, movingFloor)'],
  ['expanded persistent metadata buffer', renderer, 'const META_BYTES_PER_PIXEL = 8'],
  ['orbit telemetry binding', renderer, '{ binding: 5, resource:'],
  ['telemetry readback', renderer, 'private async captureTelemetry'],
  ['target iteration uniform', renderer, 'unsigned[21] = surface.snapshot.iterations'],
  ['complete-reference admission gate', referenceService, 'response.length < requiredLength'],
  ['escaped-reference rejection', referenceService, 'response.escaped || response.length < requiredLength'],
  ['direct fallback explanation', referenceService, 'using double-float fallback until tile-local reference repair is available']
];

const failures = checks
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing V6 invariant: ${label}`);

if (/iteration\s*>=\s*p\.orbitLength[\s\S]{0,160}STATUS_FALLBACK/.test(shader)) {
  failures.push('V6 must rebase before orbit exhaustion rather than immediately falling back');
}

if (/textureStore\(colourTexture, pixel, vec4f\(palette\(cycle\), 1\.0\)\)/.test(shader)) {
  failures.push('Escaped pixels must retain progressive confidence instead of publishing coarse blocks as final');
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`Validated ${checks.length} V6 deep-rendering invariants.`);
