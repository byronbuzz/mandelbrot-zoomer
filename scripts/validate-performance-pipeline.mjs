import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/progressiveTileFieldRenderer.ts', import.meta.url), 'utf8');
const atlas = readFileSync(new URL('../src/presentation/acceptedTileAtlas.ts', import.meta.url), 'utf8');
const shaders = readFileSync(new URL('../src/numerical/tileFieldShadersV13.ts', import.meta.url), 'utf8');
const stats = readFileSync(new URL('../src/tiles/persistentTileTypes.ts', import.meta.url), 'utf8');
const performanceGate = readFileSync(new URL('./run-performance-browser-gate.mjs', import.meta.url), 'utf8');

const submitStart = renderer.indexOf('private submitBatch(');
const submitEnd = renderer.indexOf('private createDirectParameterData', submitStart);
const hotPath = renderer.slice(submitStart, submitEnd);
const required = [
  ['three-deep bounded submission window', renderer, 'const MAX_IN_FLIGHT_BATCHES = 3'],
  ['submission-scoped readback ring', renderer, 'tile-counter-readback-ring-'],
  ['one aggregate map per batch', hotPath, 'slot.buffer.mapAsync(GPUMapMode.READ)'],
  ['FIFO retirement', renderer, 'private async retireOldestBatch()'],
  ['request preemption drains old work', renderer, 'await this.drainPendingBatches()'],
  ['same-tile overlap rejection', renderer, 'A tile already has an in-flight GPU mutation'],
  ['counter total assertion', renderer, 'Invalid tile counter total'],
  ['in-flight authoritative gate', renderer, '&& this.pendingBatches.length === 0'],
  ['direct atlas storage publication', shaders, 'tileAtlasPublishShader'],
  ['first-lease slot clearing', shaders, 'if (firstPublication)'],
  ['incremental escaped publication', shaders, 'newlyEscaped'],
  ['palette-only escaped recolour', shaders, 'let recolour = p.paletteChanged != 0u && status == 1u'],
  ['cap acceptance publication', shaders, 'forceCapPublication'],
  ['storage-capable atlas', atlas, 'GPUTextureUsage.STORAGE_BINDING'],
  ['submission telemetry', stats, 'inFlightBatches'],
  ['avoided-copy telemetry', stats, 'avoidedAtlasCopies'],
  ['same-GPU baseline comparison', performanceGate, 'completedChunkRatioAt1000Ms']
];

const failures = required
  .filter(([, source, needle]) => !source.includes(needle))
  .map(([label]) => `Missing performance-pipeline invariant: ${label}`);
if (hotPath.includes('queue.onSubmittedWorkDone')) {
  failures.push('The numerical hot path must not wait for the whole GPU queue.');
}
if (hotPath.includes('tile.counterReadback')) {
  failures.push('The numerical hot path must not map once per tile.');
}
if (renderer.includes('this.acceptedAtlas.encodeCopy')) {
  failures.push('Production publication must not copy full tile textures into the atlas.');
}
if ((hotPath.match(/mapAsync\(GPUMapMode\.READ\)/g) ?? []).length !== 1) {
  failures.push('The numerical hot path must contain exactly one aggregate mapAsync call.');
}

if (failures.length) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log(`Validated ${required.length} asynchronous scheduling and direct-publication invariants.`);
