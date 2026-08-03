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
