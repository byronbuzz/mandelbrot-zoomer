import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/progressiveTileFieldRenderer.ts', import.meta.url), 'utf8');
const atlas = readFileSync(new URL('../src/references/tileReferenceAtlasV13.ts', import.meta.url), 'utf8');
const displayShaders = readFileSync(new URL('../src/numerical/tileFieldShadersV13.ts', import.meta.url), 'utf8');
const directShader = readFileSync(new URL('../src/numerical/tileDirectShader.ts', import.meta.url), 'utf8');
const perturbShader = readFileSync(new URL('../src/numerical/tilePerturbationShader.ts', import.meta.url), 'utf8');
const planner = readFileSync(new URL('../src/tiles/worldTilePlanner.ts', import.meta.url), 'utf8');
const architecture = readFileSync(new URL('../docs/webgpu-fractal-zoomer-architecture.md', import.meta.url), 'utf8');
const build = readFileSync(new URL('../src/app/build.ts', import.meta.url), 'utf8');
const packageJson = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));

const required = [
  ['1.3.1 application version', build, "APP_VERSION = '1.3.1'"],
  ['1.3.1 package version', JSON.stringify(packageJson), '"version":"1.3.1"'],
  ['lazy lattice level queue', renderer, 'pendingLevelOffsets = [2, 1, 0]'],
  ['one-level-at-a-time admission', renderer, 'admitNextSpatialLevel'],
  ['stable numerical render height', renderer, 'MAX_NUMERICAL_PIXELS'],
  ['bounded scheduler quanta', renderer, 'quantumBudget'],
  ['event-loop yield between quanta', renderer, 'await schedulerYield()'],
  ['bounded resource creation batches', renderer, 'RESOURCE_CREATION_BATCH_TILES'],
  ['adaptive GPU batch sizing', renderer, 'adaptBatchSize'],
  ['provisional coverage accounting', renderer, 'coveragePixels'],
  ['authoritative convergence accounting', renderer, 'resolvedPixels'],
  ['monotonic presentation coverage', renderer, 'Math.max(tile.coveragePixels, acceptedCoverage)'],
  ['double-float coordinate distinguishability', renderer, 'requiresPerturbation = !splitChanged'],
  ['health-driven perturbation escalation', renderer, 'tile.health.nonFinitePixels > 0'],
  ['full user iteration target retained', renderer, 'request.targetIterations'],
  ['fine lattice margin removed', renderer, 'levelOffset > 0 ? 1 : 0'],
  ['finest-lattice nearest presentation', renderer, 'presentNearestGroup'],
  ['fully covered parent culling', renderer, 'hasCompleteChildren'],
  ['resource validation scope', renderer, "pushErrorScope('validation')"],
  ['validation failure becomes scheduler error', renderer, 'Tile resource layout validation failed'],
  ['16-byte clear host allocation', renderer, 'CLEAR_PARAMETER_BYTES = 16'],
  ['16-byte scalar clear uniform', displayShaders, 'tileSize: u32,\n  _pad0: u32,\n  _pad1: u32,\n  _pad2: u32'],
  ['recurrence-only reset bind group', renderer, '{ binding: 2, resource: { buffer: metaBuffer } }'],
  ['pixel lattice uses floor exponent', planner, 'Math.floor(pixelExponent)'],
  ['configurable tile margin', planner, 'tileMargin = 1'],
  ['shared initial reference geometry', atlas, 'initialGroupGeometry'],
  ['tile-local repair geometry', atlas, 'repairPass === 0 ? initialGroupGeometry(tile) : repairGeometry(tile)'],
  ['demand-scaled reference precision', atlas, 'MIN_REFERENCE_WORKING_BITS = 224'],
  ['reference coordinates rescaled before worker', atlas, 'fixedRescale(request.geometry.centerX, bits)'],
  ['development plan retained', architecture, '## 1.3.0 — progressive spatial and iteration field']
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
if (/return tile\.directMode !== 0/.test(renderer)) {
  failures.push('Double-float mode must not itself force perturbation.');
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
const clearStruct = displayShaders.slice(
  displayShaders.indexOf('struct ClearParams'),
  displayShaders.indexOf('struct ClearParams') + 220
);
if (clearStruct.includes('vec3u')) {
  failures.push('The host-mirrored clear uniform must avoid implicit vec3 uniform padding.');
}
const resetShader = displayShaders.slice(
  displayShaders.indexOf('export const tileResetNumericalShader'),
  displayShaders.indexOf('export const tilePresentShader')
);
if (resetShader.includes('resultTexture') || resetShader.includes('qualityTexture')) {
  failures.push('Numerical reset must not erase accepted presentation resources.');
}
for (const [label, shader] of [['direct', directShader], ['perturbation', perturbShader]]) {
  const activeCounter = shader.lastIndexOf('atomicAdd(&counters.activePixels, 1u);');
  const capReturn = shader.lastIndexOf('return;', activeCounter);
  const unresolvedTail = shader.slice(capReturn + 'return;'.length, activeCounter);
  if (activeCounter < 0 || capReturn < 0 || unresolvedTail.includes('textureStore(')) {
    failures.push(`${label} active continuation must not erase prior accepted presentation.`);
  }
}

if (failures.length > 0) {
  console.error(failures.join('\n'));
  process.exit(1);
}
console.log(`Validated ${required.length} WebGPU Fractal Zoomer 1.3 progressive-field invariants.`);
