import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/progressiveTileFieldRenderer.ts', import.meta.url), 'utf8');
const atlas = readFileSync(new URL('../src/references/tileReferenceAtlasV13.ts', import.meta.url), 'utf8');
const shaders = readFileSync(new URL('../src/numerical/tileFieldShadersV13.ts', import.meta.url), 'utf8');
const planner = readFileSync(new URL('../src/tiles/worldTilePlanner.ts', import.meta.url), 'utf8');
const architecture = readFileSync(new URL('../docs/webgpu-fractal-zoomer-architecture.md', import.meta.url), 'utf8');
const build = readFileSync(new URL('../src/app/build.ts', import.meta.url), 'utf8');
const packageJson = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));

const required = [
  ['1.3.0 application version', build, "APP_VERSION = '1.3.0'"],
  ['1.3.0 package version', JSON.stringify(packageJson), '"version":"1.3.0"'],
  ['moving lattice pyramid', renderer, "? [2, 1, 0] : [1, 0]"],
  ['stable numerical render height', renderer, 'MAX_NUMERICAL_PIXELS'],
  ['bounded scheduler quanta', renderer, 'quantumBudget'],
  ['event-loop yield between quanta', renderer, 'await schedulerYield()'],
  ['adaptive GPU batch sizing', renderer, 'adaptBatchSize'],
  ['provisional coverage accounting', renderer, 'coveragePixels'],
  ['authoritative convergence accounting', renderer, 'resolvedPixels'],
  ['provisional cap mode', renderer, 'CapPresentationMode'],
  ['full user iteration target retained', renderer, 'request.targetIterations'],
  ['fine lattice margin removed', renderer, 'levelOffset > 0 ? 1 : 0'],
  ['finest-lattice nearest presentation', renderer, 'presentNearestGroup'],
  ['resource validation scope', renderer, "pushErrorScope('validation')"],
  ['validation failure becomes scheduler error', renderer, 'Tile resource layout validation failed'],
  ['16-byte clear host allocation', renderer, 'CLEAR_PARAMETER_BYTES = 16'],
  ['16-byte scalar clear uniform', shaders, 'tileSize: u32,\n  _pad0: u32,\n  _pad1: u32,\n  _pad2: u32'],
  ['pixel lattice uses floor exponent', planner, 'Math.floor(pixelExponent)'],
  ['configurable tile margin', planner, 'tileMargin = 1'],
  ['shared initial reference geometry', atlas, 'initialGroupGeometry'],
  ['tile-local repair geometry', atlas, 'repairPass === 0 ? initialGroupGeometry(tile) : repairGeometry(tile)'],
  ['demand-scaled reference precision', atlas, 'MIN_REFERENCE_WORKING_BITS = 224'],
  ['reference coordinates rescaled before worker', atlas, 'fixedRescale(request.geometry.centerX, bits)'],
  ['development plan realigned', architecture, '## 1.3.0 — progressive spatial and iteration field']
];

const failures = required
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing 1.3 invariant: ${label}`);

if (/MOVING_DIRECT_ITERATIONS|SETTLING_DIRECT_ITERATIONS/.test(renderer)) {
  failures.push('1.3 must not restore global moving or settling iteration ceilings.');
}
if (/interaction[\s\S]{0,160}Math\.min\(request\.targetIterations/.test(renderer)) {
  failures.push('Interaction state must influence scheduling latency, not the tile numerical target.');
}
if (/Math\.ceil\(pixelExponent\)/.test(planner)) {
  failures.push('The authoritative lattice must not undersample the display pixel pitch.');
}
if (/groupGeometry\(tile\)[\s\S]{0,300}repairPass/.test(atlas)) {
  failures.push('Repair references must not remain locked to the initial multi-tile group geometry.');
}
if (/depth\s*[<>]=?|log10Magnification\(\)[\s\S]{0,160}perturb/i.test(renderer)) {
  failures.push('The precision path must remain free of a global zoom-depth crossover.');
}
const clearStruct = shaders.slice(shaders.indexOf('struct ClearParams'), shaders.indexOf('struct ClearParams') + 220);
if (clearStruct.includes('vec3u')) {
  failures.push('The host-mirrored clear uniform must avoid implicit vec3 uniform padding.');
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`Validated ${required.length} WebGPU Fractal Zoomer 1.3 progressive-field invariants.`);
