export const directOrbitShader = /* wgsl */ `
struct RenderParams {
  centerX: vec2f,
  centerY: vec2f,
  scaleMantissa: f32,
  aspect: f32,
  width: u32,
  height: u32,
  iterations: u32,
  blockSize: u32,
  tileX: u32,
  tileY: u32,
  tileWidth: u32,
  tileHeight: u32,
  mode: u32,
  scaleExponent: i32,
  palettePhase: f32,
  paletteLength: f32,
  acceptIterationCap: u32,
  _pad0: vec3u,
}

@group(0) @binding(0) var<uniform> p: RenderParams;
@group(0) @binding(1) var resultTexture: texture_storage_2d<rgba32float, write>;

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

fn mapOffset(pixelX: u32, pixelY: u32) -> vec2f {
  let uv = (vec2f(f32(pixelX), f32(pixelY)) + 0.5)
    / vec2f(f32(p.width), f32(p.height));
  return vec2f((uv.x - 0.5) * p.aspect, uv.y - 0.5);
}

fn mappedCoordinate(pixelX: u32, pixelY: u32) -> array<vec2f, 2> {
  let offset = mapOffset(pixelX, pixelY);
  let pixelXValue = ldexp(offset.x * p.scaleMantissa, p.scaleExponent);
  let pixelYValue = ldexp(offset.y * p.scaleMantissa, p.scaleExponent);
  return array<vec2f, 2>(
    dsAdd(p.centerX, vec2f(pixelXValue, 0.0)),
    dsAdd(p.centerY, vec2f(pixelYValue, 0.0))
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

fn calculate(pixelX: u32, pixelY: u32) -> vec4f {
  let coordinate = mappedCoordinate(pixelX, pixelY);
  if (p.mode == 0u) {
    let c = vec2f(dsValue(coordinate[0]), dsValue(coordinate[1]));
    if (analyticInteriorF32(c)) { return vec4f(0.0, 2.0, 0.0, 0.0); }
    var z = vec2f(0.0);
    var iteration = 0u;
    var radiusSquared = 0.0;
    loop {
      radiusSquared = dot(z, z);
      if (iteration >= p.iterations || radiusSquared > 256.0) { break; }
      z = vec2f(
        fma(z.x, z.x, -(z.y * z.y)) + c.x,
        fma(2.0 * z.x, z.y, c.y)
      );
      iteration++;
    }
    if (radiusSquared > 256.0) {
      return vec4f(smoothEscape(iteration, radiusSquared), 1.0, f32(iteration), 0.0);
    }
    return vec4f(f32(iteration), 3.0, f32(iteration), 0.0);
  }

  if (analyticInteriorDs(coordinate[0], coordinate[1])) {
    return vec4f(0.0, 2.0, 0.0, 0.0);
  }
  var zx = vec2f(0.0);
  var zy = vec2f(0.0);
  var iteration = 0u;
  var radiusSquared = 0.0;
  loop {
    let x = dsValue(zx);
    let y = dsValue(zy);
    radiusSquared = x * x + y * y;
    if (iteration >= p.iterations || radiusSquared > 256.0) { break; }
    let squared = complexSquare(zx, zy);
    zx = dsAdd(squared[0], coordinate[0]);
    zy = dsAdd(squared[1], coordinate[1]);
    iteration++;
  }
  if (radiusSquared > 256.0) {
    return vec4f(smoothEscape(iteration, radiusSquared), 1.0, f32(iteration), 0.0);
  }
  return vec4f(f32(iteration), 3.0, f32(iteration), 0.0);
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
  let result = calculate(sampleX, sampleY);

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

export const colourShader = /* wgsl */ `
struct RenderParams {
  centerX: vec2f,
  centerY: vec2f,
  scaleMantissa: f32,
  aspect: f32,
  width: u32,
  height: u32,
  iterations: u32,
  blockSize: u32,
  tileX: u32,
  tileY: u32,
  tileWidth: u32,
  tileHeight: u32,
  mode: u32,
  scaleExponent: i32,
  palettePhase: f32,
  paletteLength: f32,
  acceptIterationCap: u32,
  _pad0: vec3u,
}

@group(0) @binding(0) var<uniform> p: RenderParams;
@group(0) @binding(1) var resultTexture: texture_2d<f32>;
@group(0) @binding(2) var colourTexture: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var qualityTexture: texture_storage_2d<rgba8unorm, write>;

fn palette(t: f32) -> vec3f {
  return 0.5 + 0.5 * cos(6.2831853 * (vec3f(t) + vec3f(0.0, 0.12, 0.24) + p.palettePhase));
}

fn spatialQuality() -> f32 {
  if (p.blockSize == 1u) { return 1.0; }
  if (p.blockSize == 2u) { return 0.86; }
  if (p.blockSize == 4u) { return 0.72; }
  return 0.56;
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
  if (status == 0u) { return; }

  let quality = spatialQuality();
  if (status == 1u) {
    let cycle = fract(result.x / max(1.0, p.paletteLength));
    textureStore(colourTexture, pixel, vec4f(palette(cycle), 1.0));
    textureStore(qualityTexture, pixel, vec4f(1.0, quality, 0.25, 1.0));
    return;
  }
  if (status == 2u) {
    textureStore(colourTexture, pixel, vec4f(0.0, 0.0, 0.0, 1.0));
    textureStore(qualityTexture, pixel, vec4f(1.0, quality, 0.5, 1.0));
    return;
  }
  if (status == 3u && p.acceptIterationCap != 0u) {
    textureStore(colourTexture, pixel, vec4f(0.0, 0.0, 0.0, 1.0));
    textureStore(qualityTexture, pixel, vec4f(1.0, quality, 0.75, 1.0));
  }
}`;

export const seedShader = /* wgsl */ `
struct SeedParams {
  width: u32,
  height: u32,
  _pad0: vec2u,
  sourceScale: vec2f,
  sourceOffset: vec2f,
}

@group(0) @binding(0) var<uniform> p: SeedParams;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var sourceImage: texture_2d<f32>;
@group(0) @binding(3) var destinationImage: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var destinationQuality: texture_storage_2d<rgba8unorm, write>;

fn inside(uv: vec2f) -> bool {
  return all(uv >= vec2f(0.0)) && all(uv <= vec2f(1.0));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  let uv = (vec2f(gid.xy) + 0.5) / vec2f(f32(p.width), f32(p.height));
  let sourceUv = vec2f(0.5) + p.sourceOffset + (uv - vec2f(0.5)) * p.sourceScale;
  let clampedUv = clamp(sourceUv, vec2f(0.0), vec2f(1.0));
  let source = textureSampleLevel(sourceImage, imageSampler, clampedUv, 0.0);
  let background = vec3f(0.008, 0.01, 0.014);
  var colour = source.rgb;
  if (!inside(sourceUv)) {
    let overflow = max(abs(sourceUv - vec2f(0.5)) - vec2f(0.5), vec2f(0.0));
    let edgeWeight = 0.78 * exp2(-12.0 * length(overflow));
    colour = mix(background, source.rgb, edgeWeight);
  }
  textureStore(destinationImage, vec2i(gid.xy), vec4f(colour, 1.0));
  textureStore(destinationQuality, vec2i(gid.xy), vec4f(0.0));
}`;

export const clearFieldShader = /* wgsl */ `
struct ClearParams {
  width: u32,
  height: u32,
}

@group(0) @binding(0) var<uniform> p: ClearParams;
@group(0) @binding(1) var destinationImage: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var destinationQuality: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.width || gid.y >= p.height) { return; }
  textureStore(destinationImage, vec2i(gid.xy), vec4f(0.008, 0.01, 0.014, 1.0));
  textureStore(destinationQuality, vec2i(gid.xy), vec4f(0.0));
}`;

export const presentShader = /* wgsl */ `
struct PresentParams {
  sourceScale: vec2f,
  sourceOffset: vec2f,
}

struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}

@group(0) @binding(0) var<uniform> p: PresentParams;
@group(0) @binding(1) var imageSampler: sampler;
@group(0) @binding(2) var sourceImage: texture_2d<f32>;

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
  let sourceUv = vec2f(0.5) + p.sourceOffset + (input.uv - vec2f(0.5)) * p.sourceScale;
  let clampedUv = clamp(sourceUv, vec2f(0.0), vec2f(1.0));
  let source = textureSampleLevel(sourceImage, imageSampler, clampedUv, 0.0);
  if (inside(sourceUv)) { return vec4f(source.rgb, 1.0); }
  let overflow = max(abs(sourceUv - vec2f(0.5)) - vec2f(0.5), vec2f(0.0));
  let edgeWeight = 0.78 * exp2(-12.0 * length(overflow));
  let background = vec3f(0.008, 0.01, 0.014);
  return vec4f(mix(background, source.rgb, edgeWeight), 1.0);
}`;
