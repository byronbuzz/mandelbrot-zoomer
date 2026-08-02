import { readFileSync } from 'node:fs';

const shader = readFileSync(new URL('../src/v6/shaders.ts', import.meta.url), 'utf8');
const renderer = readFileSync(new URL('../src/v6/progressiveRenderer.ts', import.meta.url), 'utf8');

const checks = [
  ['persistent reference index', shader, 'recurrenceMeta: array<vec2u>'],
  ['scaled perturbation state', shader, 'fn scaleByViewport'],
  ['scaled quadratic term', shader, 'fn quadraticByViewport'],
  ['viewport-normalized rebase', shader, 'fn divideByViewport'],
  ['orbit-end rebase trigger', shader, 'let referenceExhausted'],
  ['reference index reset', shader, 'referenceIndex = 0u'],
  ['non-transparent final cap pixels', shader, 'let alpha = select(0.35, 1.0'],
  ['final target carried to shader', shader, 'p.iterations >= p.targetIterations'],
  ['expanded persistent metadata buffer', renderer, 'const META_BYTES_PER_PIXEL = 8'],
  ['orbit telemetry binding', renderer, '{ binding: 5, resource:'],
  ['telemetry readback', renderer, 'private async captureTelemetry'],
  ['target iteration uniform', renderer, 'unsigned[21] = surface.snapshot.iterations']
];

const failures = checks
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing V6 invariant: ${label}`);

if (/iteration\s*>=\s*p\.orbitLength[\s\S]{0,160}STATUS_FALLBACK/.test(shader)) {
  failures.push('V6 must rebase before orbit exhaustion rather than immediately falling back');
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`Validated ${checks.length} V6 deep-rendering invariants.`);
