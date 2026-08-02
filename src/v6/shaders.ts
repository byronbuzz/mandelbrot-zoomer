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
}

@group(0) @binding(0) var<uniform> p: OrbitParams;
@group(0) @binding(1) var resultTexture: texture_storage_2d<rgba32float, write>;

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

fn analyticInterior(c: vec2f) -> bool {
  let shiftedX = c.x - 0.25;
  let q = shiftedX * shiftedX + c.y * c.y;
  let inCardioid = q * (q + shiftedX) <= 0.25 * c.y * c.y;
  let bulbX = c.x + 1.0;
  let inPeriodTwoBulb = bulbX * bulbX + c.y * c.y <= 0.0625;
  return inCardioid || inPeriodTwoBulb;
}

fn smoothEscape(iteration: u32, radiusSquared: f32) -> f32 {
  let magnitude = sqrt(max(radiusSquared, 4.000001));
  return f32(iteration) + 1.0 - log2(log2(magnitude));
}

fn mapCoordinate(pixelX: u32, pixelY: u32) -> array<vec2f, 2> {
  let uv = (vec2f(f32(pixelX), f32(pixelY)) + 0.5)
    / vec2f(f32(p.width), f32(p.height));
  let offsetX = (uv.x - 0.5) * p.aspect;
  let offsetY = uv.y - 0.5;
  let pixelScaleX = ldexp(offsetX * p.scaleMantissa, p.scaleExponent);
  let pixelScaleY = ldexp(offsetY * p.scaleMantissa, p.scaleExponent);
  return array<vec2f, 2>(
    dsAdd(p.centerX, vec2f(pixelScaleX, 0.0)),
    dsAdd(p.centerY, vec2f(pixelScaleY, 0.0))
  );
}

fn calculate(pixelX: u32, pixelY: u32) -> vec4f {
  let coordinate = mapCoordinate(pixelX, pixelY);
  let c = vec2f(dsValue(coordinate[0]), dsValue(coordinate[1]));
  if (analyticInterior(c)) {
    return vec4f(0.0, 2.0, 0.0, f32(p.blockSize));
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
      let xSquared = dsMul(zx, zx);
      let ySquared = dsMul(zy, zy);
      let xy = dsMul(zx, zy);
      zx = dsAdd(dsSub(xSquared, ySquared), coordinate[0]);
      zy = dsAdd(dsScale(xy, 2.0), coordinate[1]);
      iteration++;
    }
  }

  if (radiusSquared > 4.0) {
    return vec4f(smoothEscape(iteration, radiusSquared), 1.0, f32(iteration), f32(p.blockSize));
  }
  return vec4f(f32(iteration), 3.0, f32(iteration), f32(p.blockSize));
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

  let result = textureLoad(resultTexture, vec2i(x, y), 0);
  let status = u32(round(result.y));
  if (status == 0u) {
    textureStore(colourTexture, vec2i(x, y), vec4f(0.0));
    return;
  }
  if (status == 2u) {
    textureStore(colourTexture, vec2i(x, y), vec4f(0.0, 0.0, 0.0, 1.0));
    return;
  }
  if (status == 3u) {
    let provisional = 0.07 + 0.06 * cos(vec3f(0.0, 1.7, 3.4) + result.x * 0.025);
    textureStore(colourTexture, vec2i(x, y), vec4f(provisional, 0.35));
    return;
  }

  let cycle = fract(result.x / max(1.0, p.paletteLength));
  textureStore(colourTexture, vec2i(x, y), vec4f(palette(cycle), 1.0));
}`;

export const progressivePresentShader = /* wgsl */ `
struct PresentParams {
  currentScale: vec2f,
  currentOffset: vec2f,
  stableScale: vec2f,
  stableOffset: vec2f,
  hasStable: u32,
  _pad0: vec3u,
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

  if (currentInside && current.a >= 0.5) {
    return vec4f(current.rgb, 1.0);
  }
  if (stableInside && stable.a > 0.0) {
    return vec4f(stable.rgb, 1.0);
  }
  if (currentInside && current.a > 0.0) {
    return vec4f(current.rgb, 1.0);
  }

  let source = select(current, stable, p.hasStable != 0u);
  let sourceUv = select(currentUv, stableUv, p.hasStable != 0u);
  let overflow = max(abs(sourceUv - vec2f(0.5)) - vec2f(0.5), vec2f(0.0));
  let edgeWeight = 0.78 * exp2(-12.0 * length(overflow));
  let background = vec3f(0.008, 0.01, 0.014);
  return vec4f(mix(background, source.rgb, edgeWeight), 1.0);
}`;
