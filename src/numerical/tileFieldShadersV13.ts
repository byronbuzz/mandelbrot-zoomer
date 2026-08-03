export { tileDirectIterationShader } from './tileDirectShader';
export { tilePerturbationShader } from './tilePerturbationShader';

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
@group(0) @binding(4) var evidenceTexture: texture_storage_2d<r32uint, write>;
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
  textureStore(evidenceTexture, pixel, vec4u(u32(max(0.0, round(result.z))), 0u, 0u, 0u));
  if (status == 1u) {
    let cycle = fract(result.x / max(1.0, p.paletteLength));
    textureStore(colourTexture, pixel, vec4f(palette(cycle), 1.0));
  } else if (status == 2u || status == 4u) {
    textureStore(colourTexture, pixel, vec4f(0.0, 0.0, 0.0, 1.0));
  }
}`;

// Production atlas publication is deliberately separate from the frozen
// numerical kernels. It publishes only newly accepted pixels (or escaped
// pixels whose palette changed) directly into the accepted atlas. The first
// publication clears the whole leased slot so data from an older lease can
// never become visible.
export const tileAtlasPublishShader = /* wgsl */ `
struct PublishParams {
  tileSize: u32,
  paletteChanged: u32,
  previousFrontier: u32,
  forceCapPublication: u32,
  atlasOrigin: vec2u,
  palettePhase: f32,
  paletteLength: f32,
}
@group(0) @binding(0) var<uniform> p: PublishParams;
@group(0) @binding(1) var resultTexture: texture_2d<f32>;
@group(0) @binding(2) var qualityTexture: texture_2d<f32>;
@group(0) @binding(3) var colourAtlas: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var qualityAtlas: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var evidenceAtlas: texture_storage_2d<r32uint, write>;

fn palette(t: f32) -> vec3f {
  return 0.5 + 0.5 * cos(6.2831853 * (vec3f(t) + vec3f(0.0, 0.12, 0.24) + p.palettePhase));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.tileSize || gid.y >= p.tileSize) { return; }
  let local = vec2i(gid.xy);
  let destination = vec2i(p.atlasOrigin) + local;
  let quality = textureLoad(qualityTexture, local, 0);
  let firstPublication = p.previousFrontier == 0u;

  // A fresh lease must overwrite invalid pixels as well as accepted ones.
  if (firstPublication) {
    textureStore(qualityAtlas, destination, quality);
    textureStore(evidenceAtlas, destination, vec4u(0u));
    textureStore(colourAtlas, destination, vec4f(0.0, 0.0, 0.0, 1.0));
  }
  if (quality.x <= 0.0) { return; }

  let result = textureLoad(resultTexture, local, 0);
  let status = u32(round(result.y));
  let pixelEvidence = u32(max(0.0, round(result.z)));
  let newlyEscaped = status == 1u && pixelEvidence > p.previousFrontier;
  let newlyAnalytic = status == 2u && firstPublication;
  let capAdvanced = status == 4u && (
    pixelEvidence > p.previousFrontier || firstPublication || p.forceCapPublication != 0u
  );
  let recolour = p.paletteChanged != 0u && status == 1u;
  if (!(newlyEscaped || newlyAnalytic || capAdvanced || recolour)) { return; }

  textureStore(qualityAtlas, destination, quality);
  textureStore(evidenceAtlas, destination, vec4u(pixelEvidence, 0u, 0u, 0u));
  if (status == 1u) {
    let cycle = fract(result.x / max(1.0, p.paletteLength));
    textureStore(colourAtlas, destination, vec4f(palette(cycle), 1.0));
  } else {
    textureStore(colourAtlas, destination, vec4f(0.0, 0.0, 0.0, 1.0));
  }
}`;

export const tileClearShader = /* wgsl */ `
struct ClearParams {
  tileSize: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
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

export const tileResetNumericalShader = /* wgsl */ `
struct ResetParams { tileSize: u32, preserveAccepted: u32, _pad0: vec2u }
@group(0) @binding(0) var<uniform> p: ResetParams;
@group(0) @binding(1) var<storage, read_write> recurrenceState: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> recurrenceMeta: array<vec4u>;
const STATUS_ESCAPED = 1u;
const STATUS_INTERIOR = 2u;
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x >= p.tileSize || gid.y >= p.tileSize) { return; }
  let index = gid.y * p.tileSize + gid.x;
  let pixelMeta = recurrenceMeta[index];
  if (p.preserveAccepted != 0u && (pixelMeta.y == STATUS_ESCAPED || pixelMeta.y == STATUS_INTERIOR)) {
    return;
  }
  recurrenceState[index] = vec4f(0.0);
  recurrenceMeta[index] = vec4u(0u);
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
