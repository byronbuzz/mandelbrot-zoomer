export const persistentTileIterationShader = /* wgsl */ `
struct TileParams {
  centerX: vec2f,
  centerY: vec2f,
  sampleExponent: i32,
  tileSize: u32,
  iterationTarget: u32,
  chunkIterations: u32,
  mode: u32,
  acceptIterationCap: u32,
  palettePhase: f32,
  paletteLength: f32,
  _pad0: vec4u,
}

struct TileCounters {
  activePixels: atomic<u32>,
  escapedPixels: atomic<u32>,
  analyticInteriorPixels: atomic<u32>,
  cappedPixels: atomic<u32>,
  nonFinitePixels: atomic<u32>,
}

const STATUS_ACTIVE = 0u;
const STATUS_ESCAPED = 1u;
const STATUS_INTERIOR = 2u;
const STATUS_NON_FINITE = 3u;
const RESULT_CAP = 4.0;

@group(0) @binding(0) var<uniform> p: TileParams;
@group(0) @binding(1) var<storage, read_write> recurrenceState: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> recurrenceMeta: array<vec2u>;
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

fn dsSub(a: vec2f, b: vec2f) -> vec2f {
  return dsAdd(a, vec2f(-b.x, -b.y));
}

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

fn finiteF32(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823e38;
}

fn finiteDs(value: vec2f) -> bool {
  return finiteF32(value.x) && finiteF32(value.y);
}

fn analyticInteriorF32(c: vec2f) -> bool {
  let shiftedX = c.x - 0.25;
  let ySquared = c.y * c.y;
  let q = shiftedX * shiftedX + ySquared;
  let inCardioid = q * (q + shiftedX) <= 0.25 * ySquared;
  let bulbX = c.x + 1.0;
  let inPeriodTwoBulb = bulbX * bulbX + ySquared <= 0.0625;
  return inCardioid || inPeriodTwoBulb;
}

fn analyticInteriorDs(cx: vec2f, cy: vec2f) -> bool {
  let shiftedX = dsSub(cx, vec2f(0.25, 0.0));
  let ySquared = dsMul(cy, cy);
  let q = dsAdd(dsMul(shiftedX, shiftedX), ySquared);
  let cardioidLeft = dsMul(q, dsAdd(q, shiftedX));
  let cardioidRight = dsScale(ySquared, 0.25);
  let bulbX = dsAdd(cx, vec2f(1.0, 0.0));
  let bulbRadius = dsAdd(dsMul(bulbX, bulbX), ySquared);
  return dsLessEqual(cardioidLeft, cardioidRight)
    || dsLessEqual(bulbRadius, vec2f(0.0625, 0.0));
}

fn tileCoordinate(pixelX: u32, pixelY: u32) -> array<vec2f, 2> {
  let halfSize = f32(p.tileSize) * 0.5;
  let offsetX = f32(pixelX) + 0.5 - halfSize;
  let offsetY = f32(pixelY) + 0.5 - halfSize;
  return array<vec2f, 2>(
    dsAdd(p.centerX, vec2f(ldexp(offsetX, p.sampleExponent), 0.0)),
    dsAdd(p.centerY, vec2f(ldexp(offsetY, p.sampleExponent), 0.0))
  );
}

fn complexSquare(x: vec2f, y: vec2f) -> array<vec2f, 2> {
  return array<vec2f, 2>(
    dsSub(dsMul(x, x), dsMul(y, y)),
    dsScale(dsMul(x, y), 2.0)
  );
}

fn smoothEscape(iteration: u32, radiusSquared: f32) -> f32 {
  let magnitude = sqrt(max(radiusSquared, 4.000001));
  return f32(iteration) + 1.0 - log2(log2(magnitude));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.tileSize || gid.y >= p.tileSize) { return; }
  let index = gid.y * p.tileSize + gid.x;
  let pixel = vec2i(gid.xy);
  var meta = recurrenceMeta[index];
  let storedStatus = meta.y;

  if (storedStatus == STATUS_ESCAPED) {
    atomicAdd(&counters.escapedPixels, 1u);
    return;
  }
  if (storedStatus == STATUS_INTERIOR) {
    atomicAdd(&counters.analyticInteriorPixels, 1u);
    return;
  }
  if (storedStatus == STATUS_NON_FINITE) {
    atomicAdd(&counters.nonFinitePixels, 1u);
    return;
  }

  let coordinate = tileCoordinate(gid.x, gid.y);
  if (meta.x == 0u) {
    let analyticallyInterior = select(
      analyticInteriorF32(vec2f(dsValue(coordinate[0]), dsValue(coordinate[1]))),
      analyticInteriorDs(coordinate[0], coordinate[1]),
      p.mode != 0u
    );
    if (analyticallyInterior) {
      recurrenceState[index] = vec4f(0.0);
      recurrenceMeta[index] = vec2u(0u, STATUS_INTERIOR);
      textureStore(resultTexture, pixel, vec4f(0.0, 2.0, 0.0, 0.0));
      textureStore(qualityTexture, pixel, vec4f(1.0, 1.0, 0.5, 1.0));
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
      let cx = dsValue(coordinate[0]);
      let cy = dsValue(coordinate[1]);
      let nextX = fma(x, x, -(y * y)) + cx;
      let nextY = fma(2.0 * x, y, cy);
      zx = vec2f(nextX, 0.0);
      zy = vec2f(nextY, 0.0);
    } else {
      let squared = complexSquare(zx, zy);
      zx = dsAdd(squared[0], coordinate[0]);
      zy = dsAdd(squared[1], coordinate[1]);
    }
    iteration++;
  }

  if (!finiteDs(zx) || !finiteDs(zy) || !finiteF32(radiusSquared)) {
    recurrenceState[index] = vec4f(0.0);
    recurrenceMeta[index] = vec2u(iteration, STATUS_NON_FINITE);
    textureStore(resultTexture, pixel, vec4f(0.0, 3.0, f32(iteration), 0.0));
    textureStore(qualityTexture, pixel, vec4f(1.0, 0.0, 1.0, 1.0));
    atomicAdd(&counters.nonFinitePixels, 1u);
    return;
  }

  if (radiusSquared > 256.0) {
    let smoothValue = smoothEscape(iteration, radiusSquared);
    recurrenceState[index] = vec4f(smoothValue, 0.0, 0.0, 0.0);
    recurrenceMeta[index] = vec2u(iteration, STATUS_ESCAPED);
    textureStore(resultTexture, pixel, vec4f(smoothValue, 1.0, f32(iteration), 0.0));
    textureStore(qualityTexture, pixel, vec4f(1.0, 1.0, 0.25, 1.0));
    atomicAdd(&counters.escapedPixels, 1u);
    return;
  }

  recurrenceState[index] = vec4f(zx, zy);
  recurrenceMeta[index] = vec2u(iteration, STATUS_ACTIVE);
  let progress = clamp(f32(iteration) / max(1.0, f32(p.iterationTarget)), 0.0, 1.0);
  if (iteration >= p.iterationTarget) {
    textureStore(resultTexture, pixel, vec4f(f32(iteration), RESULT_CAP, f32(iteration), 0.0));
    let accepted = select(0.0, 1.0, p.acceptIterationCap != 0u);
    textureStore(qualityTexture, pixel, vec4f(accepted, 1.0, 0.75, 1.0));
    atomicAdd(&counters.cappedPixels, 1u);
    return;
  }

  textureStore(resultTexture, pixel, vec4f(f32(iteration), 0.0, f32(iteration), 0.0));
  textureStore(qualityTexture, pixel, vec4f(0.0, progress, 0.0, 1.0));
  atomicAdd(&counters.activePixels, 1u);
}`;

export const persistentTileColourShader = /* wgsl */ `
struct TileParams {
  centerX: vec2f,
  centerY: vec2f,
  sampleExponent: i32,
  tileSize: u32,
  iterationTarget: u32,
  chunkIterations: u32,
  mode: u32,
  acceptIterationCap: u32,
  palettePhase: f32,
  paletteLength: f32,
  _pad0: vec4u,
}

@group(0) @binding(0) var<uniform> p: TileParams;
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
  let status = u32(round(result.y));
  let accepted = quality.x >= 0.5;

  if (status == 1u && accepted) {
    let cycle = fract(result.x / max(1.0, p.paletteLength));
    textureStore(colourTexture, pixel, vec4f(palette(cycle), 1.0));
    return;
  }
  if (status == 2u && accepted) {
    textureStore(colourTexture, pixel, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  if (status == 3u) {
    textureStore(colourTexture, pixel, vec4f(0.16, 0.0, 0.05, 1.0));
    return;
  }
  if (status == 4u && accepted) {
    textureStore(colourTexture, pixel, vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  textureStore(colourTexture, pixel, vec4f(0.0));
}`;

export const persistentTileClearShader = /* wgsl */ `
struct ClearParams {
  tileSize: u32,
  _pad0: vec3u,
}

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

export const persistentTilePresentShader = /* wgsl */ `
struct PresentParams {
  tileScale: vec2f,
  tileOffset: vec2f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> p: PresentParams;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var tileImage: texture_2d<f32>;

@vertex
fn vertexMain(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var positions = array<vec2f, 3>(
    vec2f(-1.0, -1.0),
    vec2f(3.0, -1.0),
    vec2f(-1.0, 3.0)
  );
  var uvs = array<vec2f, 3>(
    vec2f(0.0, 1.0),
    vec2f(2.0, 1.0),
    vec2f(0.0, -1.0)
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
  return textureSampleLevel(tileImage, imageSampler, tileUv, 0.0);
}`;
