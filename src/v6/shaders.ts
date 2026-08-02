export const progressiveOrbitShader = /* wgsl */ `
struct OrbitParams {
  centerX: vec2f,
  centerY: vec2f,
  scaleMantissa: f32,
  aspect: f32,
  width: u32,
  height: u32,
  iterations: u32,
  tileX: u32,
  tileY: u32,
  tileWidth: u32,
  tileHeight: u32,
  blockSize: u32,
  mode: u32,
  scaleExponent: i32,
  referenceDeltaX: vec2f,
  referenceDeltaY: vec2f,
  orbitLength: u32,
  targetIterations: u32,
  _pad0: u32,
  _pad1: u32,
}

struct OrbitPoint {
  x: vec4f,
  y: vec4f,
}

struct Expansion8 {
  values: array<f32, 8>,
  length: u32,
}

struct OrbitTelemetry {
  rebaseEvents: atomic<u32>,
  fallbackPixels: atomic<u32>,
  nonFiniteEvents: atomic<u32>,
  orbitExhaustions: atomic<u32>,
}

@group(0) @binding(0) var<uniform> p: OrbitParams;
@group(0) @binding(1) var resultTexture: texture_storage_2d<rgba32float, write>;
@group(0) @binding(2) var<storage, read_write> recurrenceState: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> recurrenceMeta: array<vec2u>;
@group(0) @binding(4) var<storage, read> referenceOrbit: array<OrbitPoint>;
@group(0) @binding(5) var<storage, read_write> telemetry: OrbitTelemetry;

const META_VALUE_MASK = 0x00ffffffu;
const STATUS_NEW = 0u;
const STATUS_ESCAPED = 1u;
const STATUS_INTERIOR = 2u;
const STATUS_PROVISIONAL = 3u;
const STATUS_FALLBACK = 4u;
const REASON_ORBIT = 1u;
const REASON_NON_FINITE = 2u;
const REASON_REBASE_LIMIT = 3u;
const REASON_MAGNITUDE = 4u;
const MAX_LOCAL_REBASES = 64u;

fn twoSum(a: f32, b: f32) -> vec2f {
  let sum = a + b;
  let virtualB = sum - a;
  return vec2f(sum, (a - (sum - virtualB)) + (b - virtualB));
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
  return dsMul(a, vec2f(scalar, 0.0));
}

fn dsValue(a: vec2f) -> f32 {
  return a.x + a.y;
}

fn dsLessEqual(a: vec2f, b: vec2f) -> bool {
  let difference = dsSub(a, b);
  return difference.x < 0.0 || (difference.x == 0.0 && difference.y <= 0.0);
}

fn finiteF32(value: f32) -> bool {
  return value == value && abs(value) <= 3.402823e38;
}

fn finiteDs(value: vec2f) -> bool {
  return finiteF32(value.x) && finiteF32(value.y) && finiteF32(dsValue(value));
}

fn analyticInteriorF32(c: vec2f) -> bool {
  let shiftedX = c.x - 0.25;
  let q = shiftedX * shiftedX + c.y * c.y;
  let inCardioid = q * (q + shiftedX) <= 0.25 * c.y * c.y;
  let bulbX = c.x + 1.0;
  let inPeriodTwoBulb = bulbX * bulbX + c.y * c.y <= 0.0625;
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

fn complexSquare(x: vec2f, y: vec2f) -> array<vec2f, 2> {
  return array<vec2f, 2>(
    dsSub(dsMul(x, x), dsMul(y, y)),
    dsScale(dsMul(x, y), 2.0)
  );
}

fn growExpansion(expansion: Expansion8, value: f32) -> Expansion8 {
  var result: Expansion8;
  var accumulator = value;
  var outputLength = 0u;
  var index = 0u;
  loop {
    if (index >= expansion.length) { break; }
    let sum = twoSum(accumulator, expansion.values[index]);
    if (sum.y != 0.0 && outputLength < 8u) {
      result.values[outputLength] = sum.y;
      outputLength++;
    }
    accumulator = sum.x;
    index++;
  }
  if ((accumulator != 0.0 || outputLength == 0u) && outputLength < 8u) {
    result.values[outputLength] = accumulator;
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
  return dsAdd(result, dsMul(vec2f(reference.x, 0.0), value));
}

fn scaleByViewport(value: vec2f) -> vec2f {
  let scaled = dsScale(value, p.scaleMantissa);
  return vec2f(
    ldexp(scaled.x, p.scaleExponent),
    ldexp(scaled.y, p.scaleExponent)
  );
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
  return vec2f(
    ldexp(scaled.x, halfExponent),
    ldexp(scaled.y, halfExponent)
  );
}

fn quadraticByViewport(x: vec2f, y: vec2f) -> array<vec2f, 2> {
  return complexSquare(scaleBySqrtViewport(x), scaleBySqrtViewport(y));
}

fn divideByViewport(value: vec2f) -> vec2f {
  let scaled = dsScale(value, 1.0 / p.scaleMantissa);
  return vec2f(
    ldexp(scaled.x, -p.scaleExponent),
    ldexp(scaled.y, -p.scaleExponent)
  );
}

fn smoothEscape(iteration: u32, radiusSquared: f32) -> f32 {
  let magnitude = sqrt(max(radiusSquared, 4.000001));
  return f32(iteration) + 1.0 - log2(log2(magnitude));
}

fn qualityAlpha() -> f32 {
  if (p.blockSize == 1u && p.iterations >= p.targetIterations) { return 1.0; }
  if (p.blockSize == 1u) { return 0.72; }
  if (p.blockSize == 2u) { return 0.48; }
  if (p.blockSize == 4u) { return 0.32; }
  return 0.20;
}

fn mapOffset(pixelX: u32, pixelY: u32) -> vec2f {
  let uv = (vec2f(f32(pixelX), f32(pixelY)) + 0.5)
    / vec2f(f32(p.width), f32(p.height));
  return vec2f((uv.x - 0.5) * p.aspect, uv.y - 0.5);
}

fn mapCoordinate(pixelX: u32, pixelY: u32) -> array<vec2f, 2> {
  let offset = mapOffset(pixelX, pixelY);
  let pixelScaleX = ldexp(offset.x * p.scaleMantissa, p.scaleExponent);
  let pixelScaleY = ldexp(offset.y * p.scaleMantissa, p.scaleExponent);
  return array<vec2f, 2>(
    dsAdd(p.centerX, vec2f(pixelScaleX, 0.0)),
    dsAdd(p.centerY, vec2f(pixelScaleY, 0.0))
  );
}

fn packMeta(iteration: u32, status: u32, referenceIndex: u32, rebaseCount: u32) -> vec2u {
  return vec2u(
    (iteration & META_VALUE_MASK) | (status << 24u),
    (referenceIndex & META_VALUE_MASK) | (min(rebaseCount, 255u) << 24u)
  );
}

fn recordFallback(reason: u32) {
  atomicAdd(&telemetry.fallbackPixels, 1u);
  if (reason == REASON_NON_FINITE) {
    atomicAdd(&telemetry.nonFiniteEvents, 1u);
  }
}

fn calculateDirect(pixelX: u32, pixelY: u32) -> vec4f {
  let coordinate = mapCoordinate(pixelX, pixelY);
  let c = vec2f(dsValue(coordinate[0]), dsValue(coordinate[1]));
  let analyticallyInterior = (p.mode == 0u && analyticInteriorF32(c))
    || (p.mode != 0u && analyticInteriorDs(coordinate[0], coordinate[1]));
  if (analyticallyInterior) {
    return vec4f(0.0, 2.0, 0.0, qualityAlpha());
  }

  var iteration = 0u;
  var radiusSquared = 0.0;

  if (p.mode == 0u) {
    var z = vec2f(0.0);
    loop {
      radiusSquared = dot(z, z);
      if (iteration >= p.iterations || radiusSquared > 4.0) { break; }
      z = vec2f(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
      iteration++;
    }
  } else {
    var zx = vec2f(0.0);
    var zy = vec2f(0.0);
    loop {
      let x = dsValue(zx);
      let y = dsValue(zy);
      radiusSquared = x * x + y * y;
      if (iteration >= p.iterations || radiusSquared > 4.0) { break; }
      let squared = complexSquare(zx, zy);
      zx = dsAdd(squared[0], coordinate[0]);
      zy = dsAdd(squared[1], coordinate[1]);
      iteration++;
    }
  }

  if (radiusSquared > 4.0) {
    return vec4f(smoothEscape(iteration, radiusSquared), 1.0, f32(iteration), qualityAlpha());
  }
  return vec4f(f32(iteration), 3.0, f32(iteration), qualityAlpha());
}

fn calculatePerturbation(pixelX: u32, pixelY: u32) -> vec4f {
  let index = pixelY * p.width + pixelX;
  let packed = recurrenceMeta[index];
  var iteration = packed.x & META_VALUE_MASK;
  let storedStatus = packed.x >> 24u;
  var referenceIndex = packed.y & META_VALUE_MASK;
  var rebaseCount = packed.y >> 24u;
  let storedState = recurrenceState[index];

  if (storedStatus == STATUS_ESCAPED) {
    return vec4f(storedState.x, 1.0, f32(iteration), qualityAlpha());
  }
  if (storedStatus == STATUS_INTERIOR) {
    return vec4f(0.0, 2.0, f32(iteration), qualityAlpha());
  }
  if (storedStatus == STATUS_FALLBACK) {
    return calculateDirect(pixelX, pixelY);
  }

  let coordinate = mapCoordinate(pixelX, pixelY);
  if (iteration == 0u && analyticInteriorDs(coordinate[0], coordinate[1])) {
    recurrenceState[index] = vec4f(0.0);
    recurrenceMeta[index] = packMeta(0u, STATUS_INTERIOR, 0u, 0u);
    return vec4f(0.0, 2.0, 0.0, qualityAlpha());
  }

  let offset = mapOffset(pixelX, pixelY);
  let referenceOffsetX = divideByViewport(p.referenceDeltaX);
  let referenceOffsetY = divideByViewport(p.referenceDeltaY);
  let dcx = dsAdd(referenceOffsetX, vec2f(offset.x, 0.0));
  let dcy = dsAdd(referenceOffsetY, vec2f(offset.y, 0.0));
  var ux = vec2f(storedState.x, storedState.y);
  var uy = vec2f(storedState.z, storedState.w);
  var radiusSquared = 0.0;
  var fallbackReason = 0u;
  var localRebases = 0u;

  loop {
    if (p.orbitLength < 2u || referenceIndex >= p.orbitLength) {
      fallbackReason = REASON_ORBIT;
      break;
    }

    let point = referenceOrbit[referenceIndex];
    let deltaX = scaleByViewport(ux);
    let deltaY = scaleByViewport(uy);
    let currentX = referencePlusDelta(point.x, deltaX);
    let currentY = referencePlusDelta(point.y, deltaY);
    let x = dsValue(currentX);
    let y = dsValue(currentY);
    radiusSquared = x * x + y * y;

    if (!finiteDs(currentX) || !finiteDs(currentY)
        || !finiteDs(ux) || !finiteDs(uy) || !finiteF32(radiusSquared)) {
      fallbackReason = REASON_NON_FINITE;
      break;
    }
    if (radiusSquared > 256.0 || iteration >= p.iterations) { break; }

    let deltaXf = dsValue(deltaX);
    let deltaYf = dsValue(deltaY);
    let deltaRadius = deltaXf * deltaXf + deltaYf * deltaYf;
    if (!finiteF32(deltaRadius)) {
      fallbackReason = REASON_NON_FINITE;
      break;
    }

    let referenceExhausted = referenceIndex + 1u >= p.orbitLength;
    let perturbationDominates = referenceIndex > 0u && radiusSquared < deltaRadius;
    if (referenceExhausted || perturbationDominates) {
      ux = divideByViewport(currentX);
      uy = divideByViewport(currentY);
      referenceIndex = 0u;
      rebaseCount = min(255u, rebaseCount + 1u);
      localRebases++;
      atomicAdd(&telemetry.rebaseEvents, 1u);
      if (referenceExhausted) {
        atomicAdd(&telemetry.orbitExhaustions, 1u);
      }
      if (!finiteDs(ux) || !finiteDs(uy)
          || abs(dsValue(ux)) > 1e37 || abs(dsValue(uy)) > 1e37) {
        fallbackReason = REASON_MAGNITUDE;
        break;
      }
      if (localRebases > MAX_LOCAL_REBASES) {
        fallbackReason = REASON_REBASE_LIMIT;
        break;
      }
      continue;
    }

    let quadratic = quadraticByViewport(ux, uy);
    let crossX = dsScale(
      dsSub(referenceTimesDs(point.x, ux), referenceTimesDs(point.y, uy)),
      2.0
    );
    let crossY = dsScale(
      dsAdd(referenceTimesDs(point.x, uy), referenceTimesDs(point.y, ux)),
      2.0
    );
    ux = dsAdd(dsAdd(crossX, quadratic[0]), dcx);
    uy = dsAdd(dsAdd(crossY, quadratic[1]), dcy);
    iteration++;
    referenceIndex++;

    if (!finiteDs(ux) || !finiteDs(uy)) {
      fallbackReason = REASON_NON_FINITE;
      break;
    }
    if (abs(dsValue(ux)) > 1e37 || abs(dsValue(uy)) > 1e37) {
      fallbackReason = REASON_MAGNITUDE;
      break;
    }
  }

  if (fallbackReason != 0u) {
    recordFallback(fallbackReason);
    recurrenceState[index] = vec4f(0.0);
    recurrenceMeta[index] = packMeta(iteration, STATUS_FALLBACK, 0u, rebaseCount);
    return calculateDirect(pixelX, pixelY);
  }
  if (radiusSquared > 256.0) {
    let smoothValue = smoothEscape(iteration, radiusSquared);
    recurrenceState[index] = vec4f(smoothValue, 0.0, 0.0, 0.0);
    recurrenceMeta[index] = packMeta(iteration, STATUS_ESCAPED, referenceIndex, rebaseCount);
    return vec4f(smoothValue, 1.0, f32(iteration), qualityAlpha());
  }

  recurrenceState[index] = vec4f(ux, uy);
  recurrenceMeta[index] = packMeta(iteration, STATUS_PROVISIONAL, referenceIndex, rebaseCount);
  return vec4f(f32(iteration), 3.0, f32(iteration), qualityAlpha());
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let blockOriginX = p.tileX + gid.x * p.blockSize;
  let blockOriginY = p.tileY + gid.y * p.blockSize;
  let tileEndX = min(p.width, p.tileX + p.tileWidth);
  let tileEndY = min(p.height, p.tileY + p.tileHeight);
  if (blockOriginX >= tileEndX || blockOriginY >= tileEndY) { return; }

  let sampleX = min(p.width - 1u, blockOriginX + p.blockSize / 2u);
  let sampleY = min(p.height - 1u, blockOriginY + p.blockSize / 2u);
  var result: vec4f;
  if (p.mode == 2u) {
    result = calculatePerturbation(sampleX, sampleY);
  } else {
    result = calculateDirect(sampleX, sampleY);
  }

  var localY = 0u;
  loop {
    if (localY >= p.blockSize || blockOriginY + localY >= tileEndY) { break; }
    var localX = 0u;
    loop {
      if (localX >= p.blockSize || blockOriginX + localX >= tileEndX) { break; }
      textureStore(
        resultTexture,
        vec2i(i32(blockOriginX + localX), i32(blockOriginY + localY)),
        result
      );
      localX++;
    }
    localY++;
  }
}`;

export const progressiveColourShader = /* wgsl */ `
struct ColourParams {
  width: u32,
  height: u32,
  tileX: u32,
  tileY: u32,
  tileWidth: u32,
  tileHeight: u32,
  phase: f32,
  paletteLength: f32,
}

@group(0) @binding(0) var<uniform> p: ColourParams;
@group(0) @binding(1) var resultTexture: texture_2d<f32>;
@group(0) @binding(2) var colourTexture: texture_storage_2d<rgba8unorm, write>;

fn palette(t: f32) -> vec3f {
  return 0.5 + 0.5 * cos(6.2831853 * (vec3f(t) + vec3f(0.0, 0.12, 0.24) + p.phase));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let x = p.tileX + gid.x;
  let y = p.tileY + gid.y;
  if (x >= p.width || y >= p.height || x >= p.tileX + p.tileWidth || y >= p.tileY + p.tileHeight) {
    return;
  }

  let pixel = vec2i(i32(x), i32(y));
  let result = textureLoad(resultTexture, pixel, 0);
  let status = u32(round(result.y));
  let confidence = clamp(result.w, 0.0, 1.0);
  if (status == 0u) {
    textureStore(colourTexture, pixel, vec4f(0.0));
    return;
  }
  if (status == 4u) {
    textureStore(colourTexture, pixel, vec4f(0.12, 0.0, 0.04, max(0.4, confidence)));
    return;
  }
  if (status == 2u) {
    textureStore(colourTexture, pixel, vec4f(0.0, 0.0, 0.0, confidence));
    return;
  }
  if (status == 3u) {
    let provisional = 0.07 + 0.06 * cos(vec3f(0.0, 1.7, 3.4) + result.x * 0.025);
    let alpha = select(confidence * 0.65, 1.0, confidence >= 0.999);
    textureStore(colourTexture, pixel, vec4f(provisional, alpha));
    return;
  }

  let cycle = fract(result.x / max(1.0, p.paletteLength));
  textureStore(colourTexture, pixel, vec4f(palette(cycle), confidence));
}`;

export const progressivePresentShader = /* wgsl */ `
struct PresentParams {
  currentScale: vec2f,
  currentOffset: vec2f,
  stableScale: vec2f,
  stableOffset: vec2f,
  hasStable: u32,
  preferCurrent: u32,
  _pad0: vec2u,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> p: PresentParams;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var currentImage: texture_2d<f32>;
@group(0) @binding(3) var stableImage: texture_2d<f32>;

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

fn inside(uv: vec2f) -> bool {
  return all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let currentUv = vec2f(0.5) + p.currentOffset + (input.uv - vec2f(0.5)) * p.currentScale;
  let stableUv = vec2f(0.5) + p.stableOffset + (input.uv - vec2f(0.5)) * p.stableScale;
  let currentInside = inside(currentUv);
  let stableInside = p.hasStable != 0u && inside(stableUv);

  let current = textureSampleLevel(
    currentImage,
    imageSampler,
    clamp(currentUv, vec2f(0.0), vec2f(1.0)),
    0.0
  );
  let stable = textureSampleLevel(
    stableImage,
    imageSampler,
    clamp(stableUv, vec2f(0.0), vec2f(1.0)),
    0.0
  );

  let background = vec3f(0.008, 0.01, 0.014);
  var stableBase = background;
  if (stableInside && stable.a > 0.0) {
    stableBase = mix(background, stable.rgb, clamp(stable.a, 0.0, 1.0));
  }

  if (currentInside && current.a > 0.0) {
    if (stableInside && stable.a > 0.0) {
      let confidence = clamp(current.a, 0.0, 1.0);
      let movingFloor = select(0.0, 0.35, p.preferCurrent != 0u);
      let weight = max(confidence, movingFloor);
      return vec4f(mix(stableBase, current.rgb, weight), 1.0);
    }
    return vec4f(current.rgb, 1.0);
  }
  if (stableInside && stable.a > 0.0) {
    return vec4f(stableBase, 1.0);
  }

  let source = select(current, stable, p.hasStable != 0u);
  let sourceUv = select(currentUv, stableUv, p.hasStable != 0u);
  let overflow = max(abs(sourceUv - vec2f(0.5)) - vec2f(0.5), vec2f(0.0));
  let edgeWeight = 0.78 * exp2(-12.0 * length(overflow));
  return vec4f(mix(background, source.rgb, edgeWeight * clamp(source.a, 0.0, 1.0)), 1.0);
}`;
