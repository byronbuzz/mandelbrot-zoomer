export const computeShader = /* wgsl */ `
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
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var<storage, read> referenceOrbit: array<vec4f>;

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
fn pixelDelta(value: f32) -> vec2f {
  return vec2f(ldexp(value * p.scaleMantissa, p.scaleExponent), 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let id = gid.xy;
  if (id.x >= p.width || id.y >= p.height) { return; }
  let uv = (vec2f(id) + .5) / vec2f(f32(p.width), f32(p.height));
  let pixelX = pixelDelta((uv.x - .5) * p.aspect);
  let pixelY = pixelDelta(uv.y - .5);

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

  let dcx = dsAdd(p.referenceOffsetX, pixelX);
  let dcy = dsAdd(p.referenceOffsetY, pixelY);
  var dzx = vec2f(0.0);
  var dzy = vec2f(0.0);
  var currentX = vec2f(0.0);
  var currentY = vec2f(0.0);
  var iteration = 0u;
  var refIndex = 0u;
  var radius = 0.0;
  var direct = false;
  loop {
    let parts = referenceOrbit[refIndex];
    let rx = vec2f(parts.x, parts.y);
    let ry = vec2f(parts.z, parts.w);
    currentX = dsAdd(rx, dzx);
    currentY = dsAdd(ry, dzy);
    let currentXf = dsValue(currentX);
    let currentYf = dsValue(currentY);
    radius = currentXf * currentXf + currentYf * currentYf;
    if (iteration >= p.iterations || radius > 256.0) { break; }
    if (refIndex + 1u >= p.orbitLength) { direct = true; break; }

    let deltaRadius = dsValue(dzx) * dsValue(dzx) + dsValue(dzy) * dsValue(dzy);
    if (refIndex > 0u && radius < deltaRadius) {
      dzx = currentX;
      dzy = currentY;
      refIndex = 0u;
      continue;
    }

    let dzSquared = complexSquare(dzx, dzy);
    let crossX = dsScale(dsSub(dsMul(rx, dzx), dsMul(ry, dzy)), 2.0);
    let crossY = dsScale(dsAdd(dsMul(rx, dzy), dsMul(ry, dzx)), 2.0);
    dzx = dsAdd(dsAdd(crossX, dzSquared[0]), dcx);
    dzy = dsAdd(dsAdd(crossY, dzSquared[1]), dcy);
    if (abs(dsValue(dzx)) > 1e12 || abs(dsValue(dzy)) > 1e12) {
      direct = true;
      break;
    }
    iteration++;
    refIndex++;
  }

  if (direct && iteration < p.iterations && radius <= 256.0) {
    var zx = currentX;
    var zy = currentY;
    loop {
      radius = dsValue(zx) * dsValue(zx) + dsValue(zy) * dsValue(zy);
      if (iteration >= p.iterations || radius > 256.0) { break; }
      let squared = complexSquare(zx, zy);
      zx = dsAdd(squared[0], cx);
      zy = dsAdd(squared[1], cy);
      iteration++;
    }
  }
  writeResult(id, iteration < p.iterations, iteration, radius);
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
  return textureSample(imageTexture, imageSampler, input.uv);
}`;
