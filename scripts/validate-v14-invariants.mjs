import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/progressiveTileFieldRenderer.ts', import.meta.url), 'utf8');
const atlas = readFileSync(new URL('../src/presentation/acceptedTileAtlas.ts', import.meta.url), 'utf8');
const presenter = readFileSync(new URL('../src/presentation/atlasHistoryPresenter.ts', import.meta.url), 'utf8');
const shaders = readFileSync(new URL('../src/presentation/atlasPresentationShaders.ts', import.meta.url), 'utf8');
const math = readFileSync(new URL('../src/presentation/presentationMath.ts', import.meta.url), 'utf8');
const main = readFileSync(new URL('../src/app/main.ts', import.meta.url), 'utf8');
const browserGate = readFileSync(new URL('./run-continuity-browser-gate.mjs', import.meta.url), 'utf8');

const required = [
  ['persistent accepted colour atlas', atlas, 'accepted-tile-colour-atlas'],
  ['discrete accepted quality atlas', atlas, 'accepted-tile-quality-atlas'],
  ['per-pixel accepted evidence atlas', atlas, 'accepted-tile-evidence-atlas'],
  ['monotonic slot leases', atlas, 'this.leases[index] + 1'],
  ['GPU lease directory', atlas, 'leaseDirectory'],
  ['stale instance rejection', shaders, 'leases[input.slotLease.x] != input.slotLease.y'],
  ['nearest validity lookup', shaders, 'textureLoad(qualityAtlas'],
  ['single instanced current-tile candidate', presenter, 'pass.draw(6, instanceCount)'],
  ['persistent history reprojection', presenter, 'historyTransform'],
  ['rolling display ping-pong', presenter, 'otherSurface(resources, head)'],
  ['rolling colour and provenance', presenter, "format: 'r32uint'"],
  ['explicit provenance merge', shaders, 'compatiblePreference'],
  ['conflict-preserving composition', shaders, 'SEMANTIC_CONFLICT'],
  ['no scalar depth dominance', shaders, 'atlasMergeShader'],
  ['depth limited to current spatial candidate', shaders, 'currentCandidateDepth'],
  ['semantic cap distinction', shaders, 'SEMANTIC_PROVISIONAL_CAP'],
  ['reprojected final cap demotion', shaders, 'previousSemantic == SEMANTIC_FINAL_CAP'],
  ['exact target participates in resolved view key', presenter, 'left.targetIterations === right.targetIterations'],
  ['bounded multiscale history', presenter, 'RETAINED_MEMORY_BUDGET_BYTES'],
  ['actual retained allocation accounting', presenter, 'surfaceBytes(item.view)'],
  ['separate stable checkpoint surface', presenter, 'stableCheckpoint'],
  ['resolved-only retained snapshots', presenter, 'promoteCheckpoint'],
  ['GPU continuity reduction', shaders, 'atlasContinuityReductionShader'],
  ['continuity readback ring', presenter, 'CONTINUITY_READBACK_SLOTS'],
  ['GPU-eligibility checkpoint gate', presenter, 'frame.checkpointEligible'],
  ['exact integer slot', shaders, 'slot: u32'],
  ['exact integer lease', shaders, 'lease: u32'],
  ['resize retires source after queue completion', presenter, 'retireSetAfterSubmittedWork'],
  ['deep relative tile transform', renderer, 'fixedDifferenceOverDyadic'],
  ['legacy comparison fallback', renderer, "get('presenter') === 'legacy'"],
  ['independent legacy allocation path', renderer, 'const acceptedAtlas = useLegacyPresenter ? null'],
  ['tile publication copy', renderer, 'this.acceptedAtlas.encodeCopy'],
  ['device-loss recreation hook', main, 'onDeviceLost'],
  ['executable device-loss test hook', main, '__ZOOMER_FORCE_DEVICE_LOSS__'],
  ['deterministic iteration test hook', main, "get('testIterations')"],
  ['test-only continuity hook gate', main, "get('continuityTest') === '1'"],
  ['exact camera test hook', main, 'setExactCamera'],
  ['scheduler batch gate', renderer, 'releaseTestBatches'],
  ['frame synchronized continuity hook', main, 'nextPresentedFrame'],
  ['executable real-browser gate', browserGate, 'chromium.launch'],
  ['cold coverage control trace', browserGate, 'cold deep-to-coarse transformed'],
  ['warmed bidirectional trace', browserGate, "['out', outward], ['in', inward]"],
  ['request-correlated browser assertions', browserGate, 'frame.requestId !== requestId'],
  ['batch-correlated browser assertions', browserGate, 'frame.completedBatchRevision < minimumBatchRevision'],
  ['hardware GPU assertion', browserGate, 'Continuity gate requires a hardware GPU'],
  ['visual capture evidence', browserGate, 'page.screenshot'],
  ['performance trace evidence', browserGate, 'context.tracing.start'],
  ['zero-size numerical suspension hook', main, 'renderer.setSuspended(suspended)'],
  ['scheduler submission suspension', renderer, '!this.dead && !this.suspended'],
  ['rAF survival guard', main, 'finally {\n    requestAnimationFrame(tick)'],
  ['no absolute scale materialization', math, 'scaleRatio'],
  ['zero-size presentation suspension', renderer, 'cssWidth <= 0 || cssHeight <= 0'],
  ['adapter texture-limit clamp', renderer, 'maxTextureDimension2D'],
  ['aspect-preserving display clamp', renderer, 'limit / requestedWidth, limit / requestedHeight'],
  ['resolved-only checkpoint eligibility', renderer, 'tile.resolvedPixels >= TILE_PIXEL_COUNT']
];
const failures = required.filter(([, source, needle]) => !source.includes(needle)).map(([label]) => `Missing 1.4 invariant: ${label}`);
if (renderer.includes('if (this.hasCompleteChildren(tile)) continue;')) {
  failures.push('Premature parent culling can create uncovered presentation rectangles.');
}
if (presenter.includes('promotionPending')) {
  failures.push('Rolling presentation must not stall on checkpoint promotion.');
}
if (!shaders.includes('isDisplayDefinite(semantic(base)) && isCap(semantic(candidate))')) {
  failures.push('Continuity instrumentation does not explicitly detect definite-to-cap regression.');
}

function scaled(mantissa, exponent) {
  return mantissa * Math.pow(2, exponent);
}
for (const exponent of [-1022, -1074, -2048, -4096]) {
  const source = { mantissa: 0.75, exponent };
  const target = { mantissa: 0.5625, exponent: exponent + 1 };
  const ratio = scaled(target.mantissa / source.mantissa, target.exponent - source.exponent);
  if (ratio !== 1.5) failures.push(`Deep relative scale oracle failed at exponent ${exponent}.`);
}

const transform = { scaleX: 1.125, scaleY: 0.875, offsetX: 0.03125, offsetY: -0.046875 };
const packed = Object.fromEntries(Object.entries(transform).map(([key, value]) => [key, Math.fround(value)]));
const sourceWidth = 3840;
const sourceHeight = 2160;
let worst = 0;
for (const [u, v] of [[0.5, 0.5], [0.5 / 1920, 0.5 / 1080], [1919.5 / 1920, 1079.5 / 1080]]) {
  const exactX = 0.5 + transform.offsetX + (u - 0.5) * transform.scaleX;
  const exactY = 0.5 + transform.offsetY + (v - 0.5) * transform.scaleY;
  const gpuX = Math.fround(Math.fround(0.5 + packed.offsetX) + Math.fround(Math.fround(u - 0.5) * packed.scaleX));
  const gpuY = Math.fround(Math.fround(0.5 + packed.offsetY) + Math.fround(Math.fround(v - 0.5) * packed.scaleY));
  worst = Math.max(worst, Math.abs(exactX - gpuX) * sourceWidth, Math.abs(exactY - gpuY) * sourceHeight);
}
if (worst > 0.01) failures.push(`F32 shader arithmetic oracle exceeded 0.01 texel: ${worst}.`);

if (failures.length) {
  for (const failure of failures) console.error(failure);
  process.exit(1);
}
console.log(`Validated ${required.length} 1.4 presentation invariants; deep-scale and f32 oracle worst ${worst.toExponential(3)} texel.`);
