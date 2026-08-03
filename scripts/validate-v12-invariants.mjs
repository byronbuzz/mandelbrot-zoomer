import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/tileFieldRenderer.ts', import.meta.url), 'utf8');
const shader = [
  '../src/numerical/tileDirectShader.ts',
  '../src/numerical/tilePerturbationShader.ts',
  '../src/numerical/tileDisplayShaders.ts'
].map(path => readFileSync(new URL(path, import.meta.url), 'utf8')).join('\n');
const atlas = readFileSync(new URL('../src/references/tileReferenceAtlas.ts', import.meta.url), 'utf8');
const planner = readFileSync(new URL('../src/tiles/worldTilePlanner.ts', import.meta.url), 'utf8');
const main = readFileSync(new URL('../src/app/main.ts', import.meta.url), 'utf8');
const build = readFileSync(new URL('../src/app/build.ts', import.meta.url), 'utf8');

const required = [
  ['1.2 semantic version', build, "APP_VERSION = '1.2."],
  ['production tile field engine', main, 'TileFieldRenderer'],
  ['persistent reference worker pool', atlas, 'private readonly workers: WorkerSlot[]'],
  ['local reference groups', atlas, 'REFERENCE_GROUP_TILE_SPAN'],
  ['group-centred candidate geometry', atlas, 'groupGeometry'],
  ['cross-level reference coverage', atlas, 'coverageExponent'],
  ['short escaped orbit admission', atlas, 'A short escaped orbit is still a valid local reference'],
  ['specialised perturbation pipeline', shader, 'export const tilePerturbationShader'],
  ['four-limb reference use', shader, 'referenceTimesDs'],
  ['tile glitch detection', shader, 'cancellationGlitch'],
  ['orbit exhaustion telemetry', shader, 'orbitExhaustedPixels'],
  ['portable unresolved-only reset', shader, 'pixelMeta.y == STATUS_ESCAPED || pixelMeta.y == STATUS_INTERIOR'],
  ['local repair requests', renderer, 'maybeRequestRepair'],
  ['repair preserves accepted samples', renderer, "pendingReset === 'unresolved'"],
  ['interaction-bounded reference targets', renderer, 'referenceTargetForRequest'],
  ['direct safety ceiling while reference pending', renderer, 'DIRECT_SAFETY_ITERATIONS'],
  ['no direct cap acceptance when perturbation required', renderer, 'return !this.tileNeedsPerturbation'],
  ['true convergence accounting', renderer, 'tile.acceptedPixels >= tilePixelCount'],
  ['one-sample-per-pixel base LOD', planner, 'Math.ceil(pixelExponent)'],
  ['separate numerical freshness telemetry', main, 'Numerical freshness']
];

const failures = required
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing 1.2 invariant: ${label}`);

if (/response\.escaped\s*\|\||response\.length\s*<\s*active\.iterations/.test(atlas)) {
  failures.push('Short or escaped local references must reach the GPU repair path instead of being rejected globally.');
}
if (/catch\([\s\S]{0,500}queueReference\(tile, request, repairPass \+ 1/.test(renderer)) {
  failures.push('CPU reference failures must not create an unbounded retry cascade.');
}
if (/depth\s*[<>]=?|log10Magnification\(\)[\s\S]{0,120}perturb/i.test(renderer)) {
  failures.push('Tile perturbation policy must not use a global zoom-depth crossover.');
}
if (/single-reference|viewport reference|global fallback/i.test(renderer)) {
  failures.push('The deep path must remain local rather than viewport-global.');
}
if (/acceptIterationCap[^\n]*true[\s\S]{0,200}double-float-direct/.test(renderer)) {
  failures.push('Unsafe direct iteration-cap results must not be accepted while perturbation is required.');
}
if (shader.includes('rgba8unorm, read_write')) {
  failures.push('The deployed tile path must not depend on optional read/write storage-texture support.');
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log('Validated WebGPU Fractal Zoomer 1.2 local-reference and repair invariants.');
