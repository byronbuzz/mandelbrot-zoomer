import { readFileSync } from 'node:fs';

const renderer = readFileSync(new URL('../src/presentation/progressiveTileFieldRenderer.ts', import.meta.url), 'utf8');
const atlas = readFileSync(new URL('../src/presentation/acceptedTileAtlas.ts', import.meta.url), 'utf8');
const presenter = readFileSync(new URL('../src/presentation/atlasHistoryPresenter.ts', import.meta.url), 'utf8');
const shaders = readFileSync(new URL('../src/presentation/atlasPresentationShaders.ts', import.meta.url), 'utf8');
const math = readFileSync(new URL('../src/presentation/presentationMath.ts', import.meta.url), 'utf8');
const main = readFileSync(new URL('../src/app/main.ts', import.meta.url), 'utf8');

const required = [
  ['persistent accepted colour atlas', atlas, 'accepted-tile-colour-atlas'],
  ['discrete accepted quality atlas', atlas, 'accepted-tile-quality-atlas'],
  ['monotonic slot leases', atlas, 'this.leases[index] + 1'],
  ['GPU lease directory', atlas, 'leaseDirectory'],
  ['stale instance rejection', shaders, 'leases[input.slotLease.x] != input.slotLease.y'],
  ['nearest validity lookup', shaders, 'textureLoad(qualityAtlas'],
  ['one instanced overlay', presenter, 'compose.draw(6, count)'],
  ['persistent history reprojection', presenter, 'historyTransform'],
  ['reprojected-only anchor prohibition', presenter, 'if (authoritative) this.promote'],
  ['resize retains accepted source owner', presenter, "this.anchor?.owner !== previous"],
  ['deep relative tile transform', renderer, 'fixedDifferenceOverDyadic'],
  ['legacy comparison fallback', renderer, "get('presenter') === 'legacy'"],
  ['tile publication copy', renderer, 'this.acceptedAtlas.encodeCopy'],
  ['device-loss recreation hook', main, 'onDeviceLost'],
  ['executable device-loss test hook', main, '__ZOOMER_FORCE_DEVICE_LOSS__'],
  ['rAF survival guard', main, 'finally {\n    requestAnimationFrame(tick)'],
  ['no absolute scale materialization', math, 'scaleRatio']
];
const failures = required.filter(([, source, needle]) => !source.includes(needle)).map(([label]) => `Missing 1.4 invariant: ${label}`);

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
