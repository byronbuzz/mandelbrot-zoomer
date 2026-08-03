import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/progressiveTileFieldRenderer.ts', import.meta.url), 'utf8');
const shader = [
  '../src/numerical/tileDirectShader.ts',
  '../src/numerical/tilePerturbationShader.ts',
  '../src/numerical/tileFieldShadersV13.ts'
].map(path => readFileSync(new URL(path, import.meta.url), 'utf8')).join('\n');
const atlas = readFileSync(new URL('../src/references/tileReferenceAtlasV13.ts', import.meta.url), 'utf8');
const planner = readFileSync(new URL('../src/tiles/worldTilePlanner.ts', import.meta.url), 'utf8');
const main = readFileSync(new URL('../src/app/main.ts', import.meta.url), 'utf8');
const build = readFileSync(new URL('../src/app/build.ts', import.meta.url), 'utf8');

const required = [
  ['1.x staged semantic version', build, "APP_VERSION = '1."],
  ['production tile field engine', main, 'TileFieldRenderer'],
  ['persistent reference worker pool', atlas, 'private readonly workers: WorkerSlot[]'],
  ['small initial reference groups', atlas, 'INITIAL_REFERENCE_GROUP_TILE_SPAN'],
  ['group-centred initial geometry', atlas, 'initialGroupGeometry'],
  ['tile-local repair geometry', atlas, 'repairGeometry'],
  ['cross-level reference coverage', atlas, 'coverageExponent'],
  ['finite short reference admission', atlas, 'response.length < 2'],
  ['specialised perturbation pipeline', shader, 'export const tilePerturbationShader'],
  ['four-limb reference use', shader, 'referenceTimesDs'],
  ['tile glitch detection', shader, 'cancellationGlitch'],
  ['orbit exhaustion telemetry', shader, 'orbitExhaustedPixels'],
  ['portable unresolved-only reset', shader, 'pixelMeta.y == STATUS_ESCAPED || pixelMeta.y == STATUS_INTERIOR'],
  ['local repair requests', renderer, 'maybeRequestRepair'],
  ['repair preserves accepted samples', renderer, "pendingReset === 'unresolved'"],
  ['direct safety while reference pending', renderer, 'DIRECT_SAFETY_ITERATIONS'],
  ['direct cap remains provisional while perturbation is required', renderer, 'finalTarget >= request.targetIterations'],
  ['true convergence accounting', renderer, 'tile.resolvedPixels >= TILE_PIXEL_COUNT'],
  ['authoritative no-larger-than-pixel lattice', planner, 'Math.floor(pixelExponent)'],
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
if (/MOVING_DIRECT_ITERATIONS|SETTLING_DIRECT_ITERATIONS/.test(renderer)) {
  failures.push('Interaction state must not impose a global numerical iteration ceiling.');
}
if (shader.includes('rgba8unorm, read_write')) {
  failures.push('The deployed tile path must not depend on optional read/write storage-texture support.');
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log('Validated WebGPU Fractal Zoomer local-reference and repair invariants.');
