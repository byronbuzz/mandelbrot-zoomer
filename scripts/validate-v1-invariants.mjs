import { readFileSync } from 'node:fs';

const entry = readFileSync(new URL('../src/entry.ts', import.meta.url), 'utf8');
const app = readFileSync(new URL('../src/app/main.ts', import.meta.url), 'utf8');
const renderer = readFileSync(new URL('../src/presentation/tileFieldRenderer.ts', import.meta.url), 'utf8');
const shaders = readFileSync(new URL('../src/numerical/tileFieldShaders.ts', import.meta.url), 'utf8');
const architecture = readFileSync(new URL('../docs/webgpu-fractal-zoomer-architecture.md', import.meta.url), 'utf8');
const build = readFileSync(new URL('../src/app/build.ts', import.meta.url), 'utf8');

const checks = [
  ['greenfield entry point', entry, "import('./app/main')"],
  ['1.x semantic version', build, "APP_VERSION = '1."],
  ['explicit moving state', app, "'moving'"],
  ['explicit settling state', app, "'settling'"],
  ['explicit settled state', app, "'settled'"],
  ['request coalescing', renderer, 'private latestRequest'],
  ['bounded batched jobs', renderer, 'MAX_BATCH_TILES'],
  ['persistent tile cache', renderer, 'private readonly tileMap'],
  ['separate quality texture', renderer, 'qualityTexture'],
  ['opaque calculated colour', shaders, 'vec4f(palette(cycle), 1.0)'],
  ['architecture governing model', architecture, 'persistent numerical field']
];

const failures = checks
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing 1.x foundation invariant: ${label}`);

if (entry.includes("./v6/app") || entry.includes("./app')") || entry.includes('engine ===')) {
  failures.push('The production entry point must not route through a legacy engine selector');
}
if (shaders.includes('qualityAlpha') || shaders.includes('movingFloor')) {
  failures.push('Numerical confidence must not be encoded as colour brightness or blend weight');
}
if (renderer.includes('motionPressure')) {
  failures.push('Interaction state must not be inferred from a motion-pressure threshold');
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}

console.log(`Validated ${checks.length} WebGPU Fractal Zoomer 1.x foundation invariants.`);
