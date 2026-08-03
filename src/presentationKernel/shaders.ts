export const reprojectShader = /* wgsl */ `
struct ReprojectParams {
  transform: vec4f,
  historyValid: u32,
  generation: u32,
  _pad0: vec2u,
}
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}
@group(0) @binding(0) var<uniform> p: ReprojectParams;
@group(0) @binding(1) var historySampler: sampler;
@group(0) @binding(2) var historyTexture: texture_2d<f32>;

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

fn fallback(uv: vec2f) -> vec3f {
  let checker = f32((u32(floor(uv.x * 24.0)) + u32(floor(uv.y * 16.0))) & 1u);
  let base = mix(vec3f(0.018, 0.024, 0.035), vec3f(0.028, 0.038, 0.055), checker);
  let marker = select(0.0, 0.35, uv.x < 0.04 && uv.y < 0.07);
  return base + vec3f(marker, marker * 0.35, 0.0);
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  if (p.historyValid == 0u) { return vec4f(fallback(input.uv), 1.0); }
  let sourceUv = vec2f(0.5) + p.transform.zw
    + (input.uv - vec2f(0.5)) * p.transform.xy;
  if (any(sourceUv < vec2f(0.0)) || any(sourceUv > vec2f(1.0))) {
    return vec4f(fallback(input.uv), 1.0);
  }
  return vec4f(textureSampleLevel(historyTexture, historySampler, sourceUv, 0.0).rgb, 1.0);
}`;

export const overlayShader = /* wgsl */ `
struct ViewParams {
  view: vec4f,
  generation: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}
@group(0) @binding(0) var<uniform> p: ViewParams;
@group(0) @binding(1) var<storage, read> tileRects: array<vec4f>;

@vertex
fn vertexMain(
  @builtin(vertex_index) vertexIndex: u32,
  @builtin(instance_index) instanceIndex: u32
) -> VertexOutput {
  var corners = array<vec2f, 6>(
    vec2f(0.0, 0.0), vec2f(1.0, 0.0), vec2f(0.0, 1.0),
    vec2f(0.0, 1.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0)
  );
  let rect = tileRects[instanceIndex];
  let uv = mix(rect.xy, rect.zw, corners[vertexIndex]);
  var output: VertexOutput;
  output.position = vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0);
  output.uv = uv;
  return output;
}

@fragment
fn fragmentMain(input: VertexOutput) -> @location(0) vec4f {
  let world = p.view.xy + (input.uv - vec2f(0.5)) * vec2f(p.view.z * p.view.w, p.view.z);
  let waves = 0.5 + 0.5 * sin(vec3f(
    world.x * 13.0 + world.y * 3.0,
    world.y * 17.0 - world.x * 2.0 + 1.7,
    (world.x + world.y) * 9.0 + 3.1
  ));
  let gridX = 1.0 - smoothstep(0.465, 0.5, abs(fract(world.x * 4.0) - 0.5));
  let gridY = 1.0 - smoothstep(0.465, 0.5, abs(fract(world.y * 4.0) - 0.5));
  let marker = select(0.0, 0.8, input.uv.x < 0.055 && input.uv.y < 0.085);
  let colour = 0.08 + 0.72 * waves + 0.12 * max(gridX, gridY)
    + vec3f(marker, marker * 0.22, 0.0);
  return vec4f(colour, 1.0);
}`;

export const presentShader = /* wgsl */ `
struct VertexOutput {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
}
@group(0) @binding(0) var imageSampler: sampler;
@group(0) @binding(1) var imageTexture: texture_2d<f32>;
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
  return vec4f(textureSampleLevel(imageTexture, imageSampler, input.uv, 0.0).rgb, 1.0);
}`;
