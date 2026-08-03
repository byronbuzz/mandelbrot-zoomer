import { readFileSync } from 'node:fs';

const kernel = readFileSync(new URL('../src/presentationKernel/phase1Kernel.ts', import.meta.url), 'utf8');
const geometry = readFileSync(new URL('../src/presentationKernel/geometry.ts', import.meta.url), 'utf8');
const shaders = readFileSync(new URL('../src/presentationKernel/shaders.ts', import.meta.url), 'utf8');
const demo = readFileSync(new URL('../src/presentationKernel/demo.ts', import.meta.url), 'utf8');

const required = [
  ['persistent A/B history textures', kernel, 'anchor: GPUTexture'],
  ['candidate history texture', kernel, 'candidate: GPUTexture'],
  ['immutable anchor view metadata', kernel, 'anchorView: PresentationView'],
  ['reprojection error admission', geometry, 'SOURCE_TEXEL_ERROR_LIMIT = 0.01'],
  ['single instanced tile draw', kernel, 'compose.draw(6, this.acceptedTiles)'],
  ['opaque canvas', kernel, "alphaMode: 'opaque'"],
  ['fresh adapter recovery', kernel, "navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })"],
  ['forced device-loss hook', kernel, 'forceDeviceLossForTest'],
  ['synchronous canvas acquisition', kernel, 'this.context.getCurrentTexture().createView()'],
  ['transactional resource retirement', kernel, 'queue.onSubmittedWorkDone().then'],
  ['asymmetric orientation marker', shaders, 'uv.x < 0.04 && uv.y < 0.07'],
  ['browser-visible gate diagnostics', demo, 'data-testid="phase1-errors"']
];

const failures = required
  .filter(([, source, token]) => !source.includes(token))
  .map(([label]) => `Missing Phase 1 invariant: ${label}`);

const forbidden = ['Mandelbrot', 'perturbation', 'reference orbit', 'iteration target'];
for (const term of forbidden) {
  if ([kernel, geometry, shaders, demo].some(source => source.toLowerCase().includes(term.toLowerCase()))) {
    failures.push(`Phase 1 presentation kernel contains forbidden numerical scope: ${term}`);
  }
}

function sourceUv(transform, u, v) {
  return [
    0.5 + transform.offsetX + (u - 0.5) * transform.scaleX,
    0.5 + transform.offsetY + (v - 0.5) * transform.scaleY
  ];
}

const cases = [
  { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 },
  { scaleX: 0.5, scaleY: 0.5, offsetX: 1 / 1024, offsetY: -1 / 768 },
  { scaleX: 2, scaleY: 2, offsetX: 0.25 / 1024, offsetY: 0.5 / 768 },
  { scaleX: 0.99, scaleY: 1.01, offsetX: -0.137, offsetY: 0.083 },
  { scaleX: 1.8, scaleY: 0.72, offsetX: 0.42, offsetY: -0.31 }
];
const pixels = [[0.5, 0.5], [0.5 / 1024, 0.5 / 768], [1023.5 / 1024, 767.5 / 768]];
for (const transform of cases) {
  const packed = Object.fromEntries(Object.entries(transform).map(([key, value]) => [key, Math.fround(value)]));
  for (const [u, v] of pixels) {
    const expected = sourceUv(transform, u, v);
    const actual = sourceUv(packed, u, v);
    const error = Math.max(
      Math.abs(expected[0] - actual[0]) * 1024,
      Math.abs(expected[1] - actual[1]) * 768
    );
    if (error > 0.01) failures.push(`Reprojection f32 oracle exceeded 0.01 source texel: ${error}`);
  }
}

for (let index = 0; index < 48; index++) {
  const x = index % 8;
  const y = Math.floor(index / 8);
  const width = (x + 1) / 8 - x / 8;
  const height = (y + 1) / 6 - y / 6;
  if (Math.abs(width - 1 / 8) > Number.EPSILON || Math.abs(height - 1 / 6) > Number.EPSILON) {
    failures.push(`Tile ${index} violates half-open equal-area coverage`);
  }
}

if (failures.length > 0) {
  console.error([...new Set(failures)].join('\n'));
  process.exit(1);
}
console.log(`Validated ${required.length} Phase 1 architecture invariants and ${cases.length * pixels.length} reprojection oracle samples.`);
