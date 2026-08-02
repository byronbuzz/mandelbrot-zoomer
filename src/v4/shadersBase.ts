export const computeShader = /* wgsl */ `
const TILE_COLUMNS = 16u;
const TILE_ROWS = 16u;
const REASON_EXHAUSTED = 1u;
const REASON_MAGNITUDE = 2u;
const REASON_NON_FINITE = 3u;
const REASON_REBASE = 4u;

struct Params {
  centerX: vec2f,
  centerY: vec2f,
  referenceOffsetX: vec2f,
  referenceOffsetY: vec2f,
  scaleMantissa: f32,
  aspect: f32,
  iterations: u32,
  phase: f32,
  width: u32,
  height: u32,
  mode: u32,
  orbitLength: u32,
  scaleExponent: i32,
  _pad0: u32,
}
struct Telemetry {
  unresolved: atomic<u32>,
  exhausted: atomic<u32>,
  magnitudeGuard: atomic<u32>,
  nonFinite: atomic<u32>,
  rebaseFailures: atomic<u32>,
  tiles: array<atomic<u32>, 256>,
  maxExponentBits: atomic<u32>,
}
struct OrbitPoint {
  x: vec4f,
  y: vec4f,
}
struct Expansion8 {
  values: array<f32, 8>,
  length: u32,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> referenceOrbit: array<OrbitPoint>;
@group(0) @binding(3) var<storage, read_write> telemetry: Telemetry;

fn twoSum(a: f32, b: f32) -> vec2f {
  let s = a + b;
  let bb = s - a;
  return vec2f(s, (a - (s - bb)) + (b - bb));
}
fn dsAdd(a: vec2f, b: vec2f) -> vec2f {
  let s = twoSum(a.x, b.x);
  return twoSum(s.x, s.y + a.y + b.y);
}
fn dsSub(a: vec2f, b: vec2f) -> vec2f { return dsAdd(a, vec2f(-b.x, -b.y)); }
fn dsMul(a: vec2f, b: vec2f) -> vec2f {
  let q = a.x * b.x;
  let e = fma(a.x, b.x, -q) + a.x * b.y + a.y * b.x + a.y * b.y;
  return twoSum(q, e);
}
fn dsScale(a: vec2f, b: f32) -> vec2f {
  let q = a.x * b;
  return twoSum(q, fma(a.x, b, -q) + a.y * b);
}
fn dsValue(a: vec2f) -> f32 { return a.x + a.y; }
fn complexSquare(x: vec2f, y: vec2f) -> array<vec2f, 2> {
  return array<vec2f, 2>(dsSub(dsMul(x, x), dsMul(y, y)), dsScale(dsMul(x, y), 2.0));
}
fn growExpansion(expansion: Expansion8, value: f32) -> Expansion8 {
  var result: Expansion8;
  var q = value;
  var outputLength = 0u;
  var index = 0u;
  loop {
    if (index >= expansion.length) { break; }
    let sum = twoSum(q, expansion.values[index]);
    if (sum.y != 0.0 && outputLength < 8u) {
      result.values[outputLength] = sum.y;
      outputLength++;
    }
    q = sum.x;
    index++;
  }
  if ((q != 0.0 || outputLength == 0u) && outputLength < 8u) {
    result.values[outputLength] = q;
    outputLength++;
  }
  result.length = outputLength;
  return result;
}
fn expansionToDs(expansion: Expansion8) -> vec2f {
  var result = vec2f(0.0);
  var index = 0u;
  loop {
    if (index >= expansion.length) { break; }
    result = dsAdd(result, vec2f(expansion.values[index], 0.0));
    index++;
  }
  return result;
}
fn referencePlusDelta(reference: vec4f, delta: vec2f) -> vec2f {
  var expansion: Expansion8;
  expansion.length = 0u;
  expansion = growExpansion(expansion, reference.w);
  expansion = growExpansion(expansion, reference.z);
  expansion = growExpansion(expansion, reference.y);
  expansion = growExpansion(expansion, reference.x);
  expansion = growExpansion(expansion, delta.y);
  expansion = growExpansion(expansion, delta.x);
  return expansionToDs(expansion);
}
fn referenceTimesDs(reference: vec4f, value: vec2f) -> vec2f {
  var result = vec2f(0.0);
  result = dsAdd(result, dsMul(vec2f(reference.w, 0.0), value));
  result = dsAdd(result, dsMul(vec2f(reference.z, 0.0), value));
  result = dsAdd(result, dsMul(vec2f(reference.y, 0.0), value));
  result = dsAdd(result, dsMul(vec2f(reference.x, 0.0), value));
  return result;
}
fn paletteColour(t: f32) -> vec3f {
  return .5 + .5 * cos(6.28318 * (vec3f(t) + vec3f(0.0, .12, .24) + p.phase));
}
fn writeResult(id: vec2u, escaped: bool, iteration: u32, radius: f32) {
  if (!escaped) {
    textureStore(outTex, vec2i(id), vec4f(0, 0, 0, 1));
    return;
  }
  let smoothValue = f32(iteration) + 1.0 - log2(log2(sqrt(max(radius, 1.0001))));
  textureStore(outTex, vec2i(id), vec4f(paletteColour(fract(.018 * smoothValue)), 1));
}
fn finiteF32(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823e38;
}
fn finiteDs(value: vec2f) -> bool {
  return finiteF32(value.x) && finiteF32(value.y) && finiteF32(dsValue(value));
}
fn exponentBits(value: f32) -> u32 {
  if (value == 0.0 || !finiteF32(value)) { return 0u; }
  return (bitcast<u32>(abs(value)) >> 23u) & 255u;
}
fn recordPerturbationMagnitude(x: vec2f, y: vec2f) {
  let exponent = max(exponentBits(dsValue(x)), exponentBits(dsValue(y)));
  atomicMax(&telemetry.maxExponentBits, exponent);
}
fn recordUnresolved(id: vec2u, reason: u32) {
  atomicAdd(&telemetry.unresolved, 1u);
  if (reason == REASON_EXHAUSTED) { atomicAdd(&telemetry.exhausted, 1u); }
  if (reason == REASON_MAGNITUDE) { atomicAdd(&telemetry.magnitudeGuard, 1u); }
  if (reason == REASON_NON_FINITE) { atomicAdd(&telemetry.nonFinite, 1u); }
  if (reason == REASON_REBASE) { atomicAdd(&telemetry.rebaseFailures, 1u); }
  let tileX = min((id.x * TILE_COLUMNS) / max(p.width, 1u), TILE_COLUMNS - 1u);
  let tileY = min((id.y * TILE_ROWS) / max(p.height, 1u), TILE_ROWS - 1u);
  atomicAdd(&telemetry.tiles[tileY * TILE_COLUMNS + tileX], 1u);
}
fn unresolvedAlpha(reason: u32) -> f32 {
  if (reason == REASON_EXHAUSTED) { return 0.0; }
  if (reason == REASON_MAGNITUDE) { return .125; }
  if (reason == REASON_NON_FINITE) { return .25; }
  return .375;
}
fn writeUnresolved(id: vec2u, reason: u32) {
  recordUnresolved(id, reason);
  textureStore(outTex, vec2i(id), vec4f(0, 0, 0, unresolvedAlpha(reason)));
}
fn pixelDelta(value: f32) -> vec2f {
  return vec2f(ldexp(value * p.scaleMantissa, p.scaleExponent), 0.0);
}
fn scaleByViewport(value: vec2f) -> vec2f {
  let scaled = dsScale(value, p.scaleMantissa);
  return vec2f(ldexp(scaled.x, p.scaleExponent), ldexp(scaled.y, p.scaleExponent));
}
fn scaleBySqrtViewport(value: vec2f) -> vec2f {
  var halfExponent = p.scaleExponent / 2;
  var remainder = p.scaleExponent - halfExponent * 2;
  if (remainder < 0) {
    halfExponent -= 1;
    remainder += 2;
  }
  let exponentFactor = select(1.0, 2.0, remainder == 1);
  let scaled = dsScale(value, sqrt(p.scaleMantissa * exponentFactor));
  return vec2f(ldexp(scaled.x, halfExponent), ldexp(scaled.y, halfExponent));
}
fn quadraticByViewport(x: vec2f, y: vec2f) -> array<vec2f, 2> {
  return complexSquare(scaleBySqrtViewport(x), scaleBySqrtViewport(y));
}
fn divideByViewport(value: vec2f) -> vec2f {
  let scaled = dsScale(value, 1.0 / p.scaleMantissa);
  return vec2f(ldexp(scaled.x, -p.scaleExponent), ldexp(scaled.y, -p.scaleExponent));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.xy;
  if (id.x >= p.width || id.y >= p.height) { return; }
  let uv = (vec2f(id) + .5) / vec2f(f32(p.width), f32(p.height));
  let normalizedX = (uv.x - .5) * p.aspect;
  let normalizedY = uv.y - .5;
  let pixelX = pixelDelta(normalizedX);
  let pixelY = pixelDelta(normalizedY);

  if (p.mode == 0u) {
    let c = vec2f(p.centerX.x, p.centerY.x) + vec2f(pixelX.x, pixelY.x);
    var z = vec2f(0.0);
    var iteration = 0u;
    var radius = 0.0;
    loop {
      radius = dot(z, z);
      if (iteration >= p.iterations || radius > 256.0) { break; }
      z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
      iteration++;
    }
    writeResult(id, iteration < p.iterations, iteration, radius);
    return;
  }

  let cx = dsAdd(p.centerX, pixelX);
  let cy = dsAdd(p.centerY, pixelY);
  if (p.mode == 1u) {
    var zx = vec2f(0.0);
    var zy = vec2f(0.0);
    var iteration = 0u;
    var radius = 0.0;
    loop {
      radius = dsValue(zx) * dsValue(zx) + dsValue(zy) * dsValue(zy);
      if (iteration >= p.iterations || radius > 256.0) { break; }
      let squared = complexSquare(zx, zy);
      zx = dsAdd(squared[0], cx);
      zy = dsAdd(squared[1], cy);
      iteration++;
    }
    writeResult(id, iteration < p.iterations, iteration, radius);
    return;
  }

  let dcx = dsAdd(p.referenceOffsetX, vec2f(normalizedX, 0.0));
  let dcy = dsAdd(p.referenceOffsetY, vec2f(normalizedY, 0.0));
  var ux = vec2f(0.0);
  var uy = vec2f(0.0);
  var iteration = 0u;
  var refIndex = 0u;
  var radius = 0.0;
  var unresolvedReason = 0u;

  loop {
    let point = referenceOrbit[refIndex];
    let deltaX = scaleByViewport(ux);
    let deltaY = scaleByViewport(uy);
    let currentX = referencePlusDelta(point.x, deltaX);
    let currentY = referencePlusDelta(point.y, deltaY);
    if (!finiteDs(currentX) || !finiteDs(currentY)) {
      unresolvedReason = REASON_NON_FINITE;
      break;
    }
    let currentXf = dsValue(currentX);
    let currentYf = dsValue(currentY);
    radius = currentXf * currentXf + currentYf * currentYf;
    if (!finiteF32(radius)) {
      unresolvedReason = REASON_NON_FINITE;
      break;
    }
    if (iteration >= p.iterations || radius > 256.0) { break; }
    if (refIndex + 1u >= p.orbitLength) {
      unresolvedReason = REASON_EXHAUSTED;
      break;
    }

    let deltaXf = dsValue(deltaX);
    let deltaYf = dsValue(deltaY);
    let deltaRadius = deltaXf * deltaXf + deltaYf * deltaYf;
    if (!finiteF32(deltaRadius)) {
      unresolvedReason = REASON_NON_FINITE;
      break;
    }
    if (refIndex > 0u && radius < deltaRadius) {
      ux = divideByViewport(currentX);
      uy = divideByViewport(currentY);
      recordPerturbationMagnitude(ux, uy);
      if (!finiteDs(ux) || !finiteDs(uy)
          || abs(dsValue(ux)) > 1e37 || abs(dsValue(uy)) > 1e37) {
        unresolvedReason = REASON_REBASE;
        break;
      }
      refIndex = 0u;
      continue;
    }

    let quadratic = quadraticByViewport(ux, uy);
    let crossX = dsScale(dsSub(referenceTimesDs(point.x, ux), referenceTimesDs(point.y, uy)), 2.0);
    let crossY = dsScale(dsAdd(referenceTimesDs(point.x, uy), referenceTimesDs(point.y, ux)), 2.0);
    ux = dsAdd(dsAdd(crossX, quadratic[0]), dcx);
    uy = dsAdd(dsAdd(crossY, quadratic[1]), dcy);
    recordPerturbationMagnitude(ux, uy);
    if (!finiteDs(ux) || !finiteDs(uy)) {
      unresolvedReason = REASON_NON_FINITE;
      break;
    }
    if (abs(dsValue(ux)) > 1e37 || abs(dsValue(uy)) > 1e37) {
      unresolvedReason = REASON_MAGNITUDE;
      break;
    }
    iteration++;
    refIndex++;
  }

  if (unresolvedReason != 0u) {
    writeUnresolved(id, unresolvedReason);
    return;
  }
  writeResult(id, iteration < p.iterations, iteration, radius);
}`;

export const composeShader = /* wgsl */ `
const TILE_COLUMNS = 16u;
const TILE_ROWS = 16u;
const REASON_EXHAUSTED = 1u;
const REASON_MAGNITUDE = 2u;
const REASON_NON_FINITE = 3u;
const REASON_REBASE = 4u;

struct ComposeParams {
  width: u32,
  height: u32,
}
struct Telemetry {
  unresolved: atomic<u32>,
  exhausted: atomic<u32>,
  magnitudeGuard: atomic<u32>,
  nonFinite: atomic<u32>,
  rebaseFailures: atomic<u32>,
  tiles: array<atomic<u32>, 256>,
  maxExponentBits: atomic<u32>,
}
@group(0) @binding(0) var<uniform> p: ComposeParams;
@group(0) @binding(1) var baseTex: texture_2d<f32>;
@group(0) @binding(2) var repairTex: texture_2d<f32>;
@group(0) @binding(3) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<storage, read_write> telemetry: Telemetry;

fn reasonFromAlpha(alpha: f32) -> u32 {
  if (alpha < .0625) { return REASON_EXHAUSTED; }
  if (alpha < .1875) { return REASON_MAGNITUDE; }
  if (alpha < .3125) { return REASON_NON_FINITE; }
  return REASON_REBASE;
}
fn recordUnresolved(id: vec2u, reason: u32) {
  atomicAdd(&telemetry.unresolved, 1u);
  if (reason == REASON_EXHAUSTED) { atomicAdd(&telemetry.exhausted, 1u); }
  if (reason == REASON_MAGNITUDE) { atomicAdd(&telemetry.magnitudeGuard, 1u); }
  if (reason == REASON_NON_FINITE) { atomicAdd(&telemetry.nonFinite, 1u); }
  if (reason == REASON_REBASE) { atomicAdd(&telemetry.rebaseFailures, 1u); }
  let tileX = min((id.x * TILE_COLUMNS) / max(p.width, 1u), TILE_COLUMNS - 1u);
  let tileY = min((id.y * TILE_ROWS) / max(p.height, 1u), TILE_ROWS - 1u);
  atomicAdd(&telemetry.tiles[tileY * TILE_COLUMNS + tileX], 1u);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.xy;
  if (id.x >= p.width || id.y >= p.height) { return; }
  let base = textureLoad(baseTex, vec2i(id), 0);
  let repair = textureLoad(repairTex, vec2i(id), 0);
  let result = select(base, repair, repair.a >= .5);
  textureStore(outTex, vec2i(id), result);
  if (result.a < .5) { recordUnresolved(id, reasonFromAlpha(result.a)); }
}`;

export const presentShader = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

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

@group(0) @binding(0) var imageSampler: sampler;
@group(0) @binding(1) var imageTexture: texture_2d<f32>;

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let colour = textureSample(imageTexture, imageSampler, input.uv);
  if (colour.a >= .5) { return vec4f(colour.rgb, 1); }
  let checker = f32((u32(floor(input.position.x / 8.0)) + u32(floor(input.position.y / 8.0))) & 1u);
  let level = .025 + checker * .015;
  return vec4f(level, level, level, 1);
}`;
