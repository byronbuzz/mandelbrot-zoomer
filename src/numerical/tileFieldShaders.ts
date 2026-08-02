export const tileDirectIterationShader = /* wgsl */ `
struct DirectParams {
  centerX: vec2f,
  centerY: vec2f,
  sampleExponent: i32,
  tileSize: u32,
  iterationTarget: u32,
  chunkIterations: u32,
  mode: u32,
  acceptIterationCap: u32,
  _pad0: vec4u,
}

struct TileCounters {
  activePixels: atomic<u32>,
  escapedPixels: atomic<u32>,
  analyticInteriorPixels: atomic<u32>,
  cappedPixels: atomic<u32>,
  nonFinitePixels: atomic<u32>,
  glitchPixels: atomic<u32>,
  orbitExhaustedPixels: atomic<u32>,
}

const STATUS_ACTIVE = 0u;
const STATUS_ESCAPED = 1u;
const STATUS_INTERIOR = 2u;
const STATUS_NON_FINITE = 3u;

@group(0) @binding(0) var<uniform> p: DirectParams;
@group(0) @binding(1) var<storage, read_write> recurrenceState: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> recurrenceMeta: array<vec4u>;
@group(0) @binding(3) var resultTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var qualityTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read_write> counters: TileCounters;

fn twoSum(a: f32, b: f32) -> vec2f {
  let sumValue = a + b;
  let virtualB = sumValue - a;
  return vec2f(sumValue, (a - (sumValue - virtualB)) + (b - virtualB));
}
fn dsAdd(a: vec2f, b: vec2f) -> vec2f {
  let primary = twoSum(a.x, b.x);
  return twoSum(primary.x, primary.y + a.y + b.y);
}
fn dsSub(a: vec2f, b: vec2f) -> vec2f { return dsAdd(a, vec2f(-b.x, -b.y)); }
fn dsMul(a: vec2f, b: vec2f) -> vec2f {
  let product = a.x * b.x;
  let error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x + a.y * b.y;
  return twoSum(product, error);
}
fn dsScale(a: vec2f, scalar: f32) -> vec2f {
  let product = a.x * scalar;
  return twoSum(product, fma(a.x, scalar, -product) + a.y * scalar);
}
fn dsValue(a: vec2f) -> f32 { return a.x + a.y; }
fn dsLessEqual(a: vec2f, b: vec2f) -> bool {
  let difference = dsSub(a, b);
  return difference.x < 0.0 || (difference.x == 0.0 && difference.y <= 0.0);
}
fn finiteF32(value: f32) -> bool { return value == value && abs(value) <= 3.402823e38; }
fn finiteDs(value: vec2f) -> bool { return finiteF32(value.x) && finiteF32(value.y); }

fn analyticInteriorF32(c: vec2f) -> bool {
  let shiftedX = c.x - 0.25;
  let ySquared = c.y * c.y;
  let q = shiftedX * shiftedX + ySquared;
  let bulbX = c.x + 1.0;
  return q * (q + shiftedX) <= 0.25 * ySquared
    || bulbX * bulbX + ySquared <= 0.0625;
}
fn analyticInteriorDs(cx: vec2f, cy: vec2f) -> bool {
  let shiftedX = dsSub(cx, vec2f(0.25, 0.0));
  let ySquared = dsMul(cy, cy);
  let q = dsAdd(dsMul(shiftedX, shiftedX), ySquared);
  let bulbX = dsAdd(cx, vec2f(1.0, 0.0));
  return dsLessEqual(dsMul(q, dsAdd(q, shiftedX)), dsScale(ySquared, 0.25))
    || dsLessEqual(dsAdd(dsMul(bulbX, bulbX), ySquared), vec2f(0.0625, 0.0));
}
fn tileCoordinate(pixelX: u32, pixelY: u32) -> array<vec2f, 2> {
  let halfSize = f32(p.tileSize) * 0.5;
  return array<vec2f, 2>(
    dsAdd(p.centerX, vec2f(ldexp(f32(pixelX) + 0.5 - halfSize, p.sampleExponent), 0.0)),
    dsAdd(p.centerY, vec2f(ldexp(f32(pixelY) + 0.5 - halfSize, p.sampleExponent), 0.0))
  );
}
fn complexSquare(x: vec2f, y: vec2f) -> array<vec2f, 2> {
  return array<vec2f, 2>(dsSub(dsMul(x, x), dsMul(y, y)), dsScale(dsMul(x, y), 2.0));
}
fn smoothEscape(iteration: u32, radiusSquared: f32) -> f32 {
  return f32(iteration) + 1.0 - log2(log2(sqrt(max(radiusSquared, 4.000001))));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.tileSize || gid.y >= p.tileSize) { return; }
  let index = gid.y * p.tileSize + gid.x;
  let pixel = vec2i(gid.xy);
  var meta = recurrenceMeta[index];
  if (meta.y == STATUS_ESCAPED) { atomicAdd(&counters.escapedPixels, 1u); return; }
  if (meta.y == STATUS_INTERIOR) { atomicAdd(&counters.analyticInteriorPixels, 1u); return; }
  if (meta.y == STATUS_NON_FINITE) { atomicAdd(&counters.nonFinitePixels, 1u); return; }

  let coordinate = tileCoordinate(gid.x, gid.y);
  if (meta.x == 0u) {
    let interior = select(
      analyticInteriorF32(vec2f(dsValue(coordinate[0]), dsValue(coordinate[1]))),
      analyticInteriorDs(coordinate[0], coordinate[1]),
      p.mode != 0u
    );
    if (interior) {
      recurrenceState[index] = vec4f(0.0);
      recurrenceMeta[index] = vec4u(0u, STATUS_INTERIOR, 0u, 0u);
      textureStore(resultTexture, pixel, vec4f(0.0, 2.0, 0.0, 0.0));
      textureStore(qualityTexture, pixel, vec4f(1.0, 1.0, 0.5, select(0.1, 0.3, p.mode != 0u)));
      atomicAdd(&counters.analyticInteriorPixels, 1u);
      return;
    }
  }

  var zx = recurrenceState[index].xy;
  var zy = recurrenceState[index].zw;
  var iteration = meta.x;
  let iterationEnd = min(p.iterationTarget, iteration + p.chunkIterations);
  var radiusSquared = 0.0;
  loop {
    let x = dsValue(zx);
    let y = dsValue(zy);
    radiusSquared = x * x + y * y;
    if (iteration >= iterationEnd || radiusSquared > 256.0) { break; }
    if (p.mode == 0u) {
      zx = vec2f(fma(x, x, -(y * y)) + dsValue(coordinate[0]), 0.0);
      zy = vec2f(fma(2.0 * x, y, dsValue(coordinate[1])), 0.0);
    } else {
      let squared = complexSquare(zx, zy);
      zx = dsAdd(squared[0], coordinate[0]);
      zy = dsAdd(squared[1], coordinate[1]);
    }
    iteration++;
  }

  if (!finiteDs(zx) || !finiteDs(zy) || !finiteF32(radiusSquared)) {
    recurrenceMeta[index] = vec4u(iteration, STATUS_NON_FINITE, 0u, 0u);
    textureStore(resultTexture, pixel, vec4f(0.0, 3.0, f32(iteration), 0.0));
    textureStore(qualityTexture, pixel, vec4f(0.0, 0.0, 1.0, 0.0));
    atomicAdd(&counters.nonFinitePixels, 1u);
    return;
  }
  if (radiusSquared > 256.0) {
    let smoothValue = smoothEscape(iteration, radiusSquared);
    recurrenceState[index] = vec4f(smoothValue, 0.0, 0.0, 0.0);
    recurrenceMeta[index] = vec4u(iteration, STATUS_ESCAPED, 0u, 0u);
    textureStore(resultTexture, pixel, vec4f(smoothValue, 1.0, f32(iteration), 0.0));
    textureStore(qualityTexture, pixel, vec4f(1.0, 1.0, 0.25, select(0.1, 0.3, p.mode != 0u)));
    atomicAdd(&counters.escapedPixels, 1u);
    return;
  }

  recurrenceState[index] = vec4f(zx, zy);
  recurrenceMeta[index] = vec4u(iteration, STATUS_ACTIVE, 0u, 0u);
  if (iteration >= p.iterationTarget) {
    let accepted = select(0.0, 1.0, p.acceptIterationCap != 0u);
    textureStore(resultTexture, pixel, vec4f(f32(iteration), 4.0, f32(iteration), 0.0));
    textureStore(qualityTexture, pixel, vec4f(accepted, 1.0, 0.75, select(0.1, 0.3, p.mode != 0u)));
    atomicAdd(&counters.cappedPixels, 1u);
    return;
  }
  let progress = clamp(f32(iteration) / max(1.0, f32(p.iterationTarget)), 0.0, 1.0);
  textureStore(resultTexture, pixel, vec4f(f32(iteration), 0.0, f32(iteration), 0.0));
  textureStore(qualityTexture, pixel, vec4f(0.0, progress, 0.0, select(0.1, 0.3, p.mode != 0u)));
  atomicAdd(&counters.activePixels, 1u);
}`;

export const tilePerturbationShader = /* wgsl */ `
struct PerturbParams {
  centerX: vec2f,
  centerY: vec2f,
  referenceDeltaX: vec2f,
  referenceDeltaY: vec2f,
  sampleExponent: i32,
  tileSize: u32,
  iterationTarget: u32,
  chunkIterations: u32,
  orbitLength: u32,
  acceptIterationCap: u32,
  referenceBits: u32,
  repairPass: u32,
  glitchRatio: f32,
  _pad0: vec3u,
}
struct OrbitPoint { x: vec4f, y: vec4f }
struct TileCounters {
  activePixels: atomic<u32>,
  escapedPixels: atomic<u32>,
  analyticInteriorPixels: atomic<u32>,
  cappedPixels: atomic<u32>,
  nonFinitePixels: atomic<u32>,
  glitchPixels: atomic<u32>,
  orbitExhaustedPixels: atomic<u32>,
}
const STATUS_ACTIVE = 0u;
const STATUS_ESCAPED = 1u;
const STATUS_INTERIOR = 2u;
const STATUS_NON_FINITE = 3u;
const STATUS_GLITCH = 5u;
const STATUS_ORBIT_EXHAUSTED = 6u;

@group(0) @binding(0) var<uniform> p: PerturbParams;
@group(0) @binding(1) var<storage, read_write> recurrenceState: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> recurrenceMeta: array<vec4u>;
@group(0) @binding(3) var resultTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var qualityTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var<storage, read_write> counters: TileCounters;
@group(0) @binding(6) var<storage, read> referenceOrbit: array<OrbitPoint>;

fn twoSum(a: f32, b: f32) -> vec2f {
  let sumValue = a + b;
  let virtualB = sumValue - a;
  return vec2f(sumValue, (a - (sumValue - virtualB)) + (b - virtualB));
}
fn dsAdd(a: vec2f, b: vec2f) -> vec2f {
  let primary = twoSum(a.x, b.x);
  return twoSum(primary.x, primary.y + a.y + b.y);
}
fn dsSub(a: vec2f, b: vec2f) -> vec2f { return dsAdd(a, vec2f(-b.x, -b.y)); }
fn dsMul(a: vec2f, b: vec2f) -> vec2f {
  let product = a.x * b.x;
  let error = fma(a.x, b.x, -product) + a.x * b.y + a.y * b.x + a.y * b.y;
  return twoSum(product, error);
}
fn dsScale(a: vec2f, scalar: f32) -> vec2f {
  let product = a.x * scalar;
  return twoSum(product, fma(a.x, scalar, -product) + a.y * scalar);
}
fn dsValue(a: vec2f) -> f32 { return a.x + a.y; }
fn dsLessEqual(a: vec2f, b: vec2f) -> bool {
  let difference = dsSub(a, b);
  return difference.x < 0.0 || (difference.x == 0.0 && difference.y <= 0.0);
}
fn finiteF32(value: f32) -> bool { return value == value && abs(value) <= 3.402823e38; }
fn finiteDs(value: vec2f) -> bool { return finiteF32(value.x) && finiteF32(value.y); }
fn referenceDs(value: vec4f) -> vec2f {
  var result = vec2f(0.0);
  result = dsAdd(result, vec2f(value.w, 0.0));
  result = dsAdd(result, vec2f(value.z, 0.0));
  result = dsAdd(result, vec2f(value.y, 0.0));
  return dsAdd(result, vec2f(value.x, 0.0));
}
fn referenceTimesDs(reference: vec4f, value: vec2f) -> vec2f {
  var result = vec2f(0.0);
  result = dsAdd(result, dsMul(vec2f(reference.w, 0.0), value));
  result = dsAdd(result, dsMul(vec2f(reference.z, 0.0), value));
  result = dsAdd(result, dsMul(vec2f(reference.y, 0.0), value));
  return dsAdd(result, dsMul(vec2f(reference.x, 0.0), value));
}
fn analyticInteriorDs(cx: vec2f, cy: vec2f) -> bool {
  let shiftedX = dsSub(cx, vec2f(0.25, 0.0));
  let ySquared = dsMul(cy, cy);
  let q = dsAdd(dsMul(shiftedX, shiftedX), ySquared);
  let bulbX = dsAdd(cx, vec2f(1.0, 0.0));
  return dsLessEqual(dsMul(q, dsAdd(q, shiftedX)), dsScale(ySquared, 0.25))
    || dsLessEqual(dsAdd(dsMul(bulbX, bulbX), ySquared), vec2f(0.0625, 0.0));
}
fn pixelOffset(pixelX: u32, pixelY: u32) -> array<vec2f, 2> {
  let halfSize = f32(p.tileSize) * 0.5;
  return array<vec2f, 2>(
    vec2f(ldexp(f32(pixelX) + 0.5 - halfSize, p.sampleExponent), 0.0),
    vec2f(ldexp(f32(pixelY) + 0.5 - halfSize, p.sampleExponent), 0.0)
  );
}
fn complexSquare(x: vec2f, y: vec2f) -> array<vec2f, 2> {
  return array<vec2f, 2>(dsSub(dsMul(x, x), dsMul(y, y)), dsScale(dsMul(x, y), 2.0));
}
fn smoothEscape(iteration: u32, radiusSquared: f32) -> f32 {
  return f32(iteration) + 1.0 - log2(log2(sqrt(max(radiusSquared, 4.000001))));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.tileSize || gid.y >= p.tileSize) { return; }
  let index = gid.y * p.tileSize + gid.x;
  let pixel = vec2i(gid.xy);
  var meta = recurrenceMeta[index];
  if (meta.y == STATUS_ESCAPED) { atomicAdd(&counters.escapedPixels, 1u); return; }
  if (meta.y == STATUS_INTERIOR) { atomicAdd(&counters.analyticInteriorPixels, 1u); return; }
  if (meta.y == STATUS_NON_FINITE) { atomicAdd(&counters.nonFinitePixels, 1u); return; }
  if (meta.y == STATUS_GLITCH) { atomicAdd(&counters.glitchPixels, 1u); return; }
  if (meta.y == STATUS_ORBIT_EXHAUSTED) { atomicAdd(&counters.orbitExhaustedPixels, 1u); return; }

  let offset = pixelOffset(gid.x, gid.y);
  let coordinateX = dsAdd(p.centerX, offset[0]);
  let coordinateY = dsAdd(p.centerY, offset[1]);
  if (meta.x == 0u && analyticInteriorDs(coordinateX, coordinateY)) {
    recurrenceState[index] = vec4f(0.0);
    recurrenceMeta[index] = vec4u(0u, STATUS_INTERIOR, 0u, p.repairPass);
    textureStore(resultTexture, pixel, vec4f(0.0, 2.0, 0.0, 0.0));
    textureStore(qualityTexture, pixel, vec4f(1.0, 1.0, 0.5, 0.8));
    atomicAdd(&counters.analyticInteriorPixels, 1u);
    return;
  }

  let dcx = dsAdd(p.referenceDeltaX, offset[0]);
  let dcy = dsAdd(p.referenceDeltaY, offset[1]);
  var dx = recurrenceState[index].xy;
  var dy = recurrenceState[index].zw;
  var iteration = meta.x;
  var referenceIndex = meta.z;
  let iterationEnd = min(p.iterationTarget, iteration + p.chunkIterations);
  var radiusSquared = 0.0;

  loop {
    if (referenceIndex >= p.orbitLength) {
      recurrenceMeta[index] = vec4u(iteration, STATUS_ORBIT_EXHAUSTED, referenceIndex, p.repairPass);
      textureStore(resultTexture, pixel, vec4f(0.0, 6.0, f32(iteration), 0.0));
      textureStore(qualityTexture, pixel, vec4f(0.0, 0.0, 1.0, 0.8));
      atomicAdd(&counters.orbitExhaustedPixels, 1u);
      return;
    }
    let point = referenceOrbit[referenceIndex];
    let referenceX = referenceDs(point.x);
    let referenceY = referenceDs(point.y);
    let currentX = dsAdd(referenceX, dx);
    let currentY = dsAdd(referenceY, dy);
    let x = dsValue(currentX);
    let y = dsValue(currentY);
    radiusSquared = x * x + y * y;
    if (iteration >= iterationEnd || radiusSquared > 256.0) { break; }

    let referenceRadius = dsValue(referenceX) * dsValue(referenceX)
      + dsValue(referenceY) * dsValue(referenceY);
    let cancellationGlitch = iteration > 8u
      && referenceRadius > 1e-24
      && radiusSquared < p.glitchRatio * referenceRadius;
    if (cancellationGlitch) {
      recurrenceMeta[index] = vec4u(iteration, STATUS_GLITCH, referenceIndex, p.repairPass);
      textureStore(resultTexture, pixel, vec4f(0.0, 5.0, f32(iteration), 0.0));
      textureStore(qualityTexture, pixel, vec4f(0.0, 0.0, 1.0, 0.8));
      atomicAdd(&counters.glitchPixels, 1u);
      return;
    }

    let quadratic = complexSquare(dx, dy);
    let crossX = dsScale(
      dsSub(referenceTimesDs(point.x, dx), referenceTimesDs(point.y, dy)),
      2.0
    );
    let crossY = dsScale(
      dsAdd(referenceTimesDs(point.x, dy), referenceTimesDs(point.y, dx)),
      2.0
    );
    dx = dsAdd(dsAdd(crossX, quadratic[0]), dcx);
    dy = dsAdd(dsAdd(crossY, quadratic[1]), dcy);
    iteration++;
    referenceIndex++;
    if (!finiteDs(dx) || !finiteDs(dy)) {
      recurrenceMeta[index] = vec4u(iteration, STATUS_NON_FINITE, referenceIndex, p.repairPass);
      textureStore(resultTexture, pixel, vec4f(0.0, 3.0, f32(iteration), 0.0));
      textureStore(qualityTexture, pixel, vec4f(0.0, 0.0, 1.0, 0.8));
      atomicAdd(&counters.nonFinitePixels, 1u);
      return;
    }
  }

  if (radiusSquared > 256.0) {
    let smoothValue = smoothEscape(iteration, radiusSquared);
    recurrenceState[index] = vec4f(smoothValue, 0.0, 0.0, 0.0);
    recurrenceMeta[index] = vec4u(iteration, STATUS_ESCAPED, referenceIndex, p.repairPass);
    textureStore(resultTexture, pixel, vec4f(smoothValue, 1.0, f32(iteration), 0.0));
    textureStore(qualityTexture, pixel, vec4f(1.0, 1.0, 0.25, 0.8));
    atomicAdd(&counters.escapedPixels, 1u);
    return;
  }

  recurrenceState[index] = vec4f(dx, dy);
  recurrenceMeta[index] = vec4u(iteration, STATUS_ACTIVE, referenceIndex, p.repairPass);
  if (iteration >= p.iterationTarget) {
    let accepted = select(0.0, 1.0, p.acceptIterationCap != 0u);
    textureStore(resultTexture, pixel, vec4f(f32(iteration), 4.0, f32(iteration), 0.0));
    textureStore(qualityTexture, pixel, vec4f(accepted, 1.0, 0.75, 0.8));
    atomicAdd(&counters.cappedPixels, 1u);
    return;
  }
  let progress = clamp(f32(iteration) / max(1.0, f32(p.iterationTarget)), 0.0, 1.0);
  textureStore(resultTexture, pixel, vec4f(f32(iteration), 0.0, f32(iteration), 0.0));
  textureStore(qualityTexture, pixel, vec4f(0.0, progress, 0.0, 0.8));
  atomicAdd(&counters.activePixels, 1u);
}`;

export const tileColourShader = /* wgsl */ `
struct ColourParams {
  tileSize: u32,
  _pad0: u32,
  palettePhase: f32,
  paletteLength: f32,
}
@group(0) @binding(0) var<uniform> p: ColourParams;
@group(0) @binding(1) var resultTexture: texture_2d<f32>;
@group(0) @binding(2) var qualityTexture: texture_2d<f32>;
@group(0) @binding(3) var colourTexture: texture_storage_2d<rgba8unorm, write>;
fn palette(t: f32) -> vec3f {
  return 0.5 + 0.5 * cos(6.2831853 * (vec3f(t) + vec3f(0.0, 0.12, 0.24) + p.palettePhase));
}
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.tileSize || gid.y >= p.tileSize) { return; }
  let pixel = vec2i(gid.xy);
  let result = textureLoad(resultTexture, pixel, 0);
  let quality = textureLoad(qualityTexture, pixel, 0);
  if (quality.x < 0.5) { return; }
  let status = u32(round(result.y));
  if (status == 1u) {
    let cycle = fract(result.x / max(1.0, p.paletteLength));
    textureStore(colourTexture, pixel, vec4f(palette(cycle), 1.0));
  } else if (status == 2u || status == 4u) {
    textureStore(colourTexture, pixel, vec4f(0.0, 0.0, 0.0, 1.0));
  }
}`;

export const tileClearShader = /* wgsl */ `
struct ClearParams { tileSize: u32, _pad0: vec3u }
@group(0) @binding(0) var<uniform> p: ClearParams;
@group(0) @binding(1) var resultTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var qualityTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var colourTexture: texture_storage_2d<rgba8unorm, write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.tileSize || gid.y >= p.tileSize) { return; }
  let pixel = vec2i(gid.xy);
  textureStore(resultTexture, pixel, vec4f(0.0));
  textureStore(qualityTexture, pixel, vec4f(0.0));
  textureStore(colourTexture, pixel, vec4f(0.0));
}`;

export const tileResetNumericalShader = /* wgsl */ `
struct ResetParams { tileSize: u32, preserveAccepted: u32, _pad0: vec2u }
@group(0) @binding(0) var<uniform> p: ResetParams;
@group(0) @binding(1) var<storage, read_write> recurrenceState: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> recurrenceMeta: array<vec4u>;
@group(0) @binding(3) var resultTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(4) var qualityTexture: texture_storage_2d<rgba8unorm, read_write>;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.tileSize || gid.y >= p.tileSize) { return; }
  let pixel = vec2i(gid.xy);
  let quality = textureLoad(qualityTexture, pixel);
  if (p.preserveAccepted != 0u && quality.x >= 0.5) { return; }
  let index = gid.y * p.tileSize + gid.x;
  recurrenceState[index] = vec4f(0.0);
  recurrenceMeta[index] = vec4u(0u);
  textureStore(resultTexture, pixel, vec4f(0.0));
  textureStore(qualityTexture, pixel, vec4f(0.0));
}`;

export const tilePresentShader = /* wgsl */ `
struct PresentParams { tileScale: vec2f, tileOffset: vec2f }
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}
@group(0) @binding(0) var<uniform> p: PresentParams;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var tileImage: texture_2d<f32>;
@group(0) @binding(3) var tileQuality: texture_2d<f32>;
@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0)
  );
  var uvs = array<vec2f, 3>(
    vec2f(0.0, 1.0), vec2f(2.0, 1.0), vec2f(0.0, -1.0)
  );
  var output: VertexOutput;
  output.position = vec4f(positions[vertexIndex], 0.0, 1.0);
  output.uv = uvs[vertexIndex];
  return output;
}
@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let tileUv = vec2f(0.5) + p.tileOffset + (input.uv - vec2f(0.5)) * p.tileScale;
  if (any(tileUv < vec2f(0.0)) || any(tileUv > vec2f(1.0))) { discard; }
  let colour = textureSampleLevel(tileImage, imageSampler, tileUv, 0.0);
  let quality = textureSampleLevel(tileQuality, imageSampler, tileUv, 0.0);
  return vec4f(colour.rgb, clamp(quality.x, 0.0, 1.0));
}`;
